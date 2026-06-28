#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    from requests.adapters import HTTPAdapter  # type: ignore
except Exception:
    HTTPAdapter = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


POS_SYSTEM_PROMPT = (
    "You are helping create supervision data for correctly retrieved video-query pairs.\n\n"
    "Context:\n"
    "- The retrieved top-1 video is the correct match for the query (it matches ground truth).\n"
    "- You may be given reference temporal information (ground-truth start/end in seconds).\n\n"
    "Tasks:\n"
    "1. Explain briefly why the retrieved video matches the query.\n\n"
    "Rules:\n"
    "- Ground your explanation in query-required visual evidence.\n"
    "- Include temporal cues in the reasoning (what starts/finishes around the grounded span).\n"
    "- If reference temporal span is provided, keep your final span consistent with it.\n"
    "- If reference temporal span is provided, explicitly justify both boundaries:\n"
    "  why the start should not be earlier (no required event before start, first required event begins at start),\n"
    "  and why the end should not be later (required event is no longer present after end).\n"
    "- Keep reasoning concise and factual (one to three sentences). Reasoning should be concise, but also needs to contain all necessary information to decide answer\n\n"
    "Output format:\n"
    "Reasoning: <concise sentences with temporal rationale>\n"
)

NEG_SYSTEM_PROMPT = (
    "You are helping create supervision data for wrongly retrieved video-query pairs.\n\n"
    "Task:\n"
    "Explain why the retrieved video does NOT satisfy the query.\n\n"
    "Rules:\n"
    "- Base your reasoning on query-required elements and what is visibly retrieved in video.\n"
    "- Explain why this is still not the correct match.\n"
    "- Explain what is missing in videos but query has.\n"
    "- You may briefly mention confusing overlap, but end with the key mismatch.\n"
    "- Include concise grounding evidence in the reasoning.\n"
    "- Keep the reasoning concise (one or two sentences) and avoid hallucination. Reasoning should be concise, but also needs to contain all necessary information to decide answer\n\n"
    "Output format:\n"
    "Reasoning: <concise sentences with temporal rationale>\n"
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not (x == x):
        return None
    return float(x)


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _safe_int_list(values: Any) -> Optional[List[int]]:
    if not isinstance(values, (list, tuple)):
        return None
    out: List[int] = []
    for v in values:
        try:
            iv = int(round(float(v)))
        except Exception:
            continue
        if iv < 0:
            iv = 0
        out.append(iv)
    return out if out else None


def _row_gold_label_norm(row: Dict[str, Any]) -> str:
    g = str(row.get("gold_label", "")).strip().lower()
    if g in {"match", "pos", "positive"}:
        return "match"
    if g in {"not_match", "neg", "negative", "no_match"}:
        return "not_match"
    top1 = str(row.get("top1_doc_id", "")).strip()
    pos = str(row.get("query_pos_doc_id", "")).strip()
    if top1 and pos:
        return "match" if top1 == pos else "not_match"
    return ""


def _norm_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_video_meta_index(meta_jsonl_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    path = str(meta_jsonl_path or "").strip()
    if not path:
        return {"by_path": {}, "by_basename": {}, "by_stem": {}, "by_video_id": {}}
    if not os.path.exists(path):
        print(f"[video_meta] not found: {path}")
        return {"by_path": {}, "by_basename": {}, "by_stem": {}, "by_video_id": {}}

    by_path: Dict[str, Dict[str, Any]] = {}
    by_basename: Dict[str, Dict[str, Any]] = {}
    by_stem: Dict[str, Dict[str, Any]] = {}
    by_video_id: Dict[str, Dict[str, Any]] = {}
    used = 0
    for row in _iter_jsonl(path):
        raw_fps = _safe_float(row.get("raw_fps", row.get("fps", None)))
        frames_indices = _safe_int_list(
            row.get(
                "sampled_indices_src",
                row.get("frames_indices", row.get("requested_indices_src", None)),
            )
        )
        total_num_frames = _safe_int(
            row.get("source_total_frames", row.get("total_num_frames", None))
        )
        if total_num_frames is None and frames_indices:
            total_num_frames = int(max(frames_indices) + 1)

        payload: Dict[str, Any] = {}
        if raw_fps is not None and raw_fps > 0:
            payload["raw_fps"] = float(raw_fps)
        if frames_indices is not None:
            payload["frames_indices"] = frames_indices
        if total_num_frames is not None and total_num_frames > 0:
            payload["total_num_frames"] = int(total_num_frames)
        if not payload:
            continue

        video_id = str(row.get("video_id", "")).strip()
        if video_id:
            by_video_id[video_id] = dict(payload)

        cands: List[str] = []
        for key in ("output_npy_path", "npy_path", "output_path", "path", "video_path", "video"):
            val = row.get(key, None)
            if isinstance(val, str) and val.strip():
                cands.append(val.strip())
        for cand in cands:
            try:
                ap = _norm_path(cand)
            except Exception:
                continue
            by_path[ap] = dict(payload)
            base = os.path.basename(ap)
            if base:
                by_basename[base] = dict(payload)
            st = _stem(ap)
            if st:
                by_stem[st] = dict(payload)
            used += 1
    print(
        f"[video_meta] loaded source={path} entries={used} "
        f"by_path={len(by_path)} by_basename={len(by_basename)} "
        f"by_stem={len(by_stem)} by_video_id={len(by_video_id)}"
    )
    return {
        "by_path": by_path,
        "by_basename": by_basename,
        "by_stem": by_stem,
        "by_video_id": by_video_id,
    }


def _resolve_video_meta(
    row: Dict[str, Any],
    video_path: str,
    video_meta_index: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    if not video_meta_index:
        return None
    by_path = video_meta_index.get("by_path", {})
    by_basename = video_meta_index.get("by_basename", {})
    by_stem = video_meta_index.get("by_stem", {})
    by_video_id = video_meta_index.get("by_video_id", {})

    video_id = str(row.get("top1_video_id", "")).strip() or str(row.get("top1_doc_id", "")).strip()
    if video_id and video_id in by_video_id:
        return dict(by_video_id[video_id])
    try:
        ap = _norm_path(video_path)
    except Exception:
        ap = str(video_path)
    hit = by_path.get(ap)
    if hit is not None:
        return dict(hit)
    base = os.path.basename(ap)
    hit = by_basename.get(base)
    if hit is not None:
        return dict(hit)
    st = _stem(ap)
    hit = by_stem.get(st)
    if hit is not None:
        return dict(hit)
    return None


def _video_url(video_path: str, prefix: str) -> str:
    if video_path.startswith(("http://", "https://", "file://")):
        return video_path
    return f"{prefix}{video_path}"


def _resolve_video_path(
    row: Dict[str, Any],
    prefer_npy: bool,
    video_npy_root: str,
    video_npy_ext: str,
) -> str:
    video_id = str(row.get("top1_video_id", "")).strip() or str(row.get("top1_doc_id", "")).strip()
    top1_path = str(row.get("top1_video_path", "")).strip()
    fallback_path = str(row.get("video_path", "")).strip()

    if prefer_npy and video_npy_root and video_id:
        ext = video_npy_ext if str(video_npy_ext).startswith(".") else f".{video_npy_ext}"
        npy_path = os.path.join(video_npy_root, f"{video_id}{ext}")
        if os.path.exists(npy_path):
            return npy_path

    if top1_path:
        return top1_path
    return fallback_path


def _build_user_text(row: Dict[str, Any]) -> str:
    query = str(row.get("query", "")).strip()
    lines = [f'Query: "{query}"']
    gt_start = row.get("gt_start", None)
    gt_end = row.get("gt_end", None)
    if _row_gold_label_norm(row) == "match" and (gt_start is not None or gt_end is not None):
        lines.append(f"Ground-truth segment for query: start={gt_start}, end={gt_end}")
    return "\n".join(lines)


def _select_system_prompt(row: Dict[str, Any]) -> str:
    return POS_SYSTEM_PROMPT if _row_gold_label_norm(row) == "match" else NEG_SYSTEM_PROMPT


def _build_messages(
    row: Dict[str, Any],
    use_video: bool,
    video_input_type: str,
    video_url_prefix: str,
    prefer_npy: bool,
    video_npy_root: str,
    video_npy_ext: str,
    video_max_frames: int,
    video_fps: float,
    video_min_pixels: int,
    video_max_pixels: int,
    video_total_pixels: int,
    video_meta_index: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    force_video_field: bool = False,
) -> List[Dict[str, Any]]:
    system_prompt = _select_system_prompt(row)
    user_text = _build_user_text(row)
    if not use_video:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    video_path = _resolve_video_path(
        row=row,
        prefer_npy=prefer_npy,
        video_npy_root=video_npy_root,
        video_npy_ext=video_npy_ext,
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if video_path:
        if force_video_field or video_input_type == "video":
            video_obj: Dict[str, Any] = {"type": "video", "video": video_path}
            # Preserve temporal metadata for npy inputs (raw fps + source frame indices).
            if video_path.endswith((".npy", ".npz")):
                video_meta = _resolve_video_meta(row, video_path, video_meta_index)
                if video_meta is not None:
                    raw_fps = _safe_float(video_meta.get("raw_fps", None))
                    frames_indices = _safe_int_list(video_meta.get("frames_indices", None))
                    total_num_frames = _safe_int(video_meta.get("total_num_frames", None))
                    if raw_fps is not None and raw_fps > 0:
                        video_obj["raw_fps"] = float(raw_fps)
                    if frames_indices is not None:
                        video_obj["frames_indices"] = [int(x) for x in frames_indices]
                    if total_num_frames is not None and total_num_frames > 0:
                        video_obj["total_num_frames"] = int(total_num_frames)
            if video_max_frames > 0:
                video_obj["max_frames"] = int(video_max_frames)
            if video_fps > 0:
                video_obj["fps"] = float(video_fps)
            if video_min_pixels > 0:
                video_obj["min_pixels"] = int(video_min_pixels)
            if video_max_pixels > 0:
                video_obj["max_pixels"] = int(video_max_pixels)
            if video_total_pixels > 0:
                video_obj["total_pixels"] = int(video_total_pixels)
            content.append(video_obj)
        else:
            content.append(
                {"type": "video_url", "video_url": {"url": _video_url(video_path, video_url_prefix)}}
            )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _default_vision_process_path() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    eccv_root = os.path.dirname(os.path.dirname(this_dir))
    return os.path.join(
        eccv_root,
        "VideoSearch-R1",
        "videosearch_r1",
        "model",
        "qwen_vl_utils",
        "vision_process.py",
    )


def _load_cached_process_vision_info(path: str):
    path = str(path).strip()
    if not path:
        raise RuntimeError("vision_process path is empty")
    if not os.path.exists(path):
        raise RuntimeError(f"vision_process.py not found: {path}")
    spec = importlib.util.spec_from_file_location("va_r1_vision_process", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "cached_process_vision_info", None)
    if fn is None:
        raise RuntimeError(f"cached_process_vision_info not found in {path}")
    return fn


def _prepare_local_vllm_inputs(
    messages: List[List[Dict[str, Any]]],
    prompts_text: List[str],
    image_inputs: Optional[List[Any]],
    video_inputs: Optional[List[Any]],
    video_kwargs: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    vllm_inputs: List[Dict[str, Any]] = []
    image_idx = 0
    video_idx = 0

    for message, prompt in zip(messages, prompts_text):
        tmp_image_inputs: List[Any] = []
        tmp_video_inputs: List[Any] = []
        for msg in message:
            content = msg.get("content")
            if isinstance(content, list):
                for ele in content:
                    if isinstance(ele, dict):
                        if "image" in ele or "image_url" in ele:
                            if image_inputs is not None and image_idx < len(image_inputs):
                                tmp_image_inputs.append(image_inputs[image_idx])
                            image_idx += 1
                        elif "video" in ele:
                            if video_inputs is not None and video_idx < len(video_inputs):
                                tmp_video_inputs.append(video_inputs[video_idx])
                            video_idx += 1

        item: Dict[str, Any] = {"prompt": prompt}
        mm_data: Dict[str, Any] = {}
        if tmp_image_inputs:
            mm_data["image"] = tmp_image_inputs
        if tmp_video_inputs:
            mm_data["video"] = tmp_video_inputs
        if mm_data:
            item["multi_modal_data"] = mm_data
            if tmp_video_inputs and isinstance(video_kwargs, dict):
                item["mm_processor_kwargs"] = dict(video_kwargs)
        vllm_inputs.append(item)
    return vllm_inputs


def _call_openai_api(
    session: "requests.Session",
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Tuple[str, Optional[str], Optional[str]]:
    if requests is None:
        raise RuntimeError("requests is required")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}: {resp.text[:200]}"
                time.sleep(retry_sleep)
                continue
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            finish_reason = choice.get("finish_reason")
            return str(content), None, finish_reason
        except Exception as exc:
            last_err = str(exc)
            time.sleep(retry_sleep)
    return "", last_err, None


def _extract_tagged_line(text: str, tag: str) -> str:
    if not text:
        return ""
    tag_lower = tag.strip().lower()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(f"{tag_lower}:"):
            return line.split(":", 1)[1].strip()
    return ""


def _split_think(text: str) -> Tuple[str, str]:
    if "</think>" not in text:
        return "", text.strip()
    before, after = text.split("</think>", 1)
    before = before.replace("<think>", "").strip()
    return before, after.strip()


def _normalize_label(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""
    norm = re.sub(r"[_\-\s]+", " ", t)
    # Use label-like patterns only; avoid accidental hits from words like "notebook".
    if re.search(
        r"\b(?:not match|no match|non match|notmatching|unmatch|unmatched|mismatch|"
        r"does not match|doesn't match|do not match|fails to match)\b",
        norm,
    ):
        return "not_match"
    if re.search(r"\bmatch(?:es|ed|ing)?\b", norm):
        return "match"
    return ""


def _parse_temporal_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,~-]\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if not match:
        return None, None
    try:
        start = float(match.group(1))
        end = float(match.group(2))
    except Exception:
        return None, None
    return start, end


def _parse_response(raw_text: str) -> Tuple[str, str, str, Optional[float], Optional[float], bool]:
    think_text, body = _split_think(raw_text)
    content = body if body else raw_text

    label_text = _extract_tagged_line(content, "Label")
    reasoning = _extract_tagged_line(content, "Reasoning")
    temporal = _extract_tagged_line(content, "Temporal Grounding")

    label = _normalize_label(label_text)
    if not label:
        # Fallback to free-text label inference only for very short outputs.
        compact = content.strip()
        if compact and len(compact) <= 32:
            label = _normalize_label(compact)

    if not reasoning:
        reasoning = content.strip()
    if not temporal:
        temporal = "N/A"

    start, end = _parse_temporal_range(temporal)
    parsed_ok = bool(reasoning)
    return label, reasoning, temporal, start, end, parsed_ok


def _row_pair_id(row: Dict[str, Any], fallback_idx: int) -> str:
    pair_id = str(row.get("pair_id", "")).strip()
    if pair_id:
        return pair_id
    qid = str(row.get("qid", "")).strip() or f"idx_{fallback_idx}"
    top1 = str(row.get("top1_doc_id", "")).strip()
    if top1:
        return f"{qid}::{top1}"
    return qid


def _gold_to_norm(gold_label: str) -> str:
    g = str(gold_label or "").strip().lower()
    if g in {"match", "pos", "positive"}:
        return "match"
    if g in {"not_match", "neg", "negative", "no_match"}:
        return "not_match"
    return ""


def _build_output_row(
    row: Dict[str, Any],
    text: str,
    err: Optional[str],
    finish_reason: Optional[str],
    include_raw_response: bool,
) -> Dict[str, Any]:
    think_text, body = _split_think(text)
    parse_source = body if body else text
    label, reasoning, temporal, t_start, t_end, parsed_ok = _parse_response(parse_source)

    out = dict(row)
    gold_norm = _row_gold_label_norm(out)
    if not label and gold_norm:
        label = gold_norm
    out["model_label"] = label
    out["model_reasoning"] = reasoning
    out["model_temporal_grounding"] = temporal
    out["model_temporal_start"] = t_start
    out["model_temporal_end"] = t_end
    out["model_think"] = think_text
    out["model_finish_reason"] = finish_reason or ""
    out["model_error"] = err or ""
    out["parsed_ok"] = bool(parsed_ok)

    out["model_gold_consistent"] = bool(gold_norm and label and gold_norm == label)
    if include_raw_response:
        out["model_raw_response"] = text
    return out


def _resolve_jobs(
    pool_jsonl: str,
    output_jsonl: str,
    jobs_jsonl: str,
) -> List[Tuple[str, str]]:
    jobs_path = str(jobs_jsonl or "").strip()
    if jobs_path:
        if not os.path.exists(jobs_path):
            raise SystemExit(f"Missing jobs_jsonl: {jobs_path}")
        jobs: List[Tuple[str, str]] = []
        for line_idx, row in enumerate(_iter_jsonl(jobs_path), start=1):
            pool = str(row.get("pool_jsonl", "")).strip()
            out = str(row.get("output_jsonl", "")).strip()
            if not pool or not out:
                raise SystemExit(
                    f"jobs_jsonl line {line_idx} requires pool_jsonl and output_jsonl: {jobs_path}"
                )
            jobs.append((pool, out))
        if not jobs:
            raise SystemExit(f"jobs_jsonl is empty: {jobs_path}")
        return jobs

    pool = str(pool_jsonl or "").strip()
    out = str(output_jsonl or "").strip()
    if not pool or not out:
        raise SystemExit("Either --jobs_jsonl or both --pool_jsonl/--output_jsonl are required")
    return [(pool, out)]


def _prepare_remaining_rows(
    pool_jsonl: str,
    output_jsonl: str,
    resume: bool,
    shuffle: bool,
    seed: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not os.path.exists(pool_jsonl):
        raise SystemExit(f"Missing pool_jsonl: {pool_jsonl}")

    rows = list(_iter_jsonl(pool_jsonl))
    if not rows:
        print(f"[generate] skip empty pool_jsonl: {pool_jsonl}")
        return [], False

    seen: set[str] = set()
    if resume and os.path.exists(output_jsonl):
        for row in _iter_jsonl(output_jsonl):
            pid = _row_pair_id(row, 0)
            if pid:
                seen.add(pid)
        print(f"[resume] {output_jsonl} existing={len(seen)}")

    for i, row in enumerate(rows):
        if not row.get("pair_id"):
            row["pair_id"] = _row_pair_id(row, i)

    remaining = [row for row in rows if _row_pair_id(row, 0) not in seen]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(remaining)
    if limit > 0:
        remaining = remaining[:limit]

    append = os.path.exists(output_jsonl) and resume
    return remaining, append


def _run_openai_job(
    *,
    args: argparse.Namespace,
    remaining: List[Dict[str, Any]],
    output_jsonl: str,
    append: bool,
) -> None:
    thread_local = threading.local()
    pool_size = max(8, int(args.workers))

    def _get_session() -> "requests.Session":
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"Connection": "keep-alive"})
            sess.trust_env = False
            if HTTPAdapter is not None:
                adapter = HTTPAdapter(
                    pool_connections=pool_size,
                    pool_maxsize=pool_size,
                    max_retries=0,
                    pool_block=True,
                )
                sess.mount("http://", adapter)
                sess.mount("https://", adapter)
            thread_local.session = sess
        return sess

    def _process(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = _build_messages(
            row=row,
            use_video=args.use_video == 1,
            video_input_type=args.video_input_type,
            video_url_prefix=args.video_url_prefix,
            prefer_npy=args.prefer_npy == 1,
            video_npy_root=str(args.video_npy_root).strip(),
            video_npy_ext=str(args.video_npy_ext).strip(),
            video_max_frames=args.video_max_frames,
            video_fps=args.video_fps,
            video_min_pixels=args.video_min_pixels,
            video_max_pixels=args.video_max_pixels,
            video_total_pixels=args.video_total_pixels,
            video_meta_index=getattr(args, "_video_meta_index", None),
            force_video_field=False,
        )
        text, err, finish_reason = _call_openai_api(
            _get_session(),
            args.vllm_url,
            args.api_key,
            args.model,
            messages,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.timeout,
            args.retries,
            args.retry_sleep,
        )
        return _build_output_row(
            row=row,
            text=text,
            err=err,
            finish_reason=finish_reason,
            include_raw_response=args.include_raw_response == 1,
        )

    if args.workers <= 1:
        iterator: Iterable[Any] = remaining
        if tqdm is not None:
            iterator = tqdm(remaining, total=len(remaining), desc="generate", unit="pair")
        with open(output_jsonl, "a" if append else "w", encoding="utf-8") as f:
            for idx, row in enumerate(iterator, start=1):
                out = _process(row)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                if idx % 50 == 0:
                    print(f"[generate] {idx}/{len(remaining)}")
    else:
        with open(output_jsonl, "a" if append else "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_process, row): row for row in remaining}
                iterator = as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(remaining), desc="generate", unit="pair")
                done = 0
                for fut in iterator:
                    out = fut.result()
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"[generate] {done}/{len(remaining)}")


def _init_local_backend(args: argparse.Namespace):
    try:
        from transformers import AutoProcessor  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"backend=local_vllm requires transformers and vllm packages: {exc}"
        ) from exc

    cached_process_vision_info = _load_cached_process_vision_info(args.vision_process_path)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=bool(args.local_trust_remote_code == 1),
    )
    llm = LLM(
        model=args.model,
        dtype=args.local_dtype,
        tensor_parallel_size=max(1, int(args.local_tensor_parallel_size)),
        gpu_memory_utilization=float(args.local_gpu_memory_utilization),
        max_model_len=int(args.local_max_model_len),
        trust_remote_code=bool(args.local_trust_remote_code == 1),
        seed=int(args.seed),
    )

    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    return processor, llm, sampling_params, cached_process_vision_info


def _run_local_job(
    *,
    args: argparse.Namespace,
    remaining: List[Dict[str, Any]],
    output_jsonl: str,
    append: bool,
    processor: Any,
    llm: Any,
    sampling_params: Any,
    cached_process_vision_info: Any,
) -> None:
    bs = max(1, int(args.local_batch_size))
    num_batches = (len(remaining) + bs - 1) // bs
    batch_iter: Iterable[int] = range(0, len(remaining), bs)
    if tqdm is not None:
        batch_iter = tqdm(batch_iter, total=num_batches, desc="generate(local_vllm)", unit="batch")

    with open(output_jsonl, "a" if append else "w", encoding="utf-8") as f:
        done = 0
        for start in batch_iter:
            chunk = remaining[start : start + bs]
            messages_batch = [
                _build_messages(
                    row=row,
                    use_video=args.use_video == 1,
                    video_input_type=args.video_input_type,
                    video_url_prefix=args.video_url_prefix,
                    prefer_npy=args.prefer_npy == 1,
                    video_npy_root=str(args.video_npy_root).strip(),
                    video_npy_ext=str(args.video_npy_ext).strip(),
                    video_max_frames=args.video_max_frames,
                    video_fps=args.video_fps,
                    video_min_pixels=args.video_min_pixels,
                    video_max_pixels=args.video_max_pixels,
                    video_total_pixels=args.video_total_pixels,
                    video_meta_index=getattr(args, "_video_meta_index", None),
                    force_video_field=True,
                )
                for row in chunk
            ]
            prompts_text = [
                processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for msg in messages_batch
            ]
            image_inputs, packed_video_inputs, video_kwargs = cached_process_vision_info(
                messages_batch,
                image_patch_size=int(args.local_image_patch_size),
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            vllm_inputs = _prepare_local_vllm_inputs(
                messages=messages_batch,
                prompts_text=prompts_text,
                image_inputs=image_inputs,
                video_inputs=packed_video_inputs,
                video_kwargs=video_kwargs,
            )
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"local_vllm output size mismatch: outputs={len(outputs)} chunk={len(chunk)}"
                )
            for row, output in zip(chunk, outputs):
                text = ""
                finish_reason = None
                err = None
                try:
                    if output.outputs:
                        text = str(output.outputs[0].text)
                        finish_reason = getattr(output.outputs[0], "finish_reason", None)
                except Exception as exc:
                    err = str(exc)
                out = _build_output_row(
                    row=row,
                    text=text,
                    err=err,
                    finish_reason=finish_reason,
                    include_raw_response=args.include_raw_response == 1,
                )
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 50 == 0:
                    print(f"[generate] {done}/{len(remaining)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_jsonl", default="")
    ap.add_argument("--output_jsonl", default="")
    ap.add_argument("--jobs_jsonl", default="")
    ap.add_argument("--vllm_url", default="http://localhost:8099/v1/chat/completions")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    ap.add_argument("--api_key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--backend", choices=["openai_api", "local_vllm"], default="openai_api")
    ap.add_argument("--use_video", type=int, default=1)
    ap.add_argument("--video_input_type", choices=["video_url", "video"], default="video_url")
    ap.add_argument("--video_url_prefix", default="file://")
    ap.add_argument("--prefer_npy", type=int, default=0)
    ap.add_argument("--video_npy_root", default="")
    ap.add_argument("--video_npy_ext", default=".npy")
    ap.add_argument("--video_meta_jsonl", default="")
    ap.add_argument("--video_max_frames", type=int, default=0)
    ap.add_argument("--video_fps", type=float, default=0.0)
    ap.add_argument("--video_min_pixels", type=int, default=0)
    ap.add_argument("--video_max_pixels", type=int, default=0)
    ap.add_argument("--video_total_pixels", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry_sleep", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--local_batch_size", type=int, default=4)
    ap.add_argument("--vision_process_path", default=_default_vision_process_path())
    ap.add_argument("--local_image_patch_size", type=int, default=16)
    ap.add_argument("--local_dtype", default="bfloat16")
    ap.add_argument("--local_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--local_gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--local_max_model_len", type=int, default=102768)
    ap.add_argument("--local_trust_remote_code", type=int, default=1)
    ap.add_argument("--include_raw_response", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    jobs = _resolve_jobs(args.pool_jsonl, args.output_jsonl, args.jobs_jsonl)
    if args.prefer_npy == 1 and args.video_npy_root and not os.path.isdir(args.video_npy_root):
        raise SystemExit(f"Missing video_npy_root: {args.video_npy_root}")
    video_meta_jsonl = str(args.video_meta_jsonl or "").strip()
    if not video_meta_jsonl and args.prefer_npy == 1 and args.video_npy_root:
        cand = os.path.join(str(args.video_npy_root).strip(), "meta.jsonl")
        if os.path.exists(cand):
            video_meta_jsonl = cand
    args._video_meta_index = _load_video_meta_index(video_meta_jsonl)
    if args.backend == "openai_api" and requests is None:
        raise SystemExit("requests is required for backend=openai_api")

    local_ctx: Optional[Tuple[Any, Any, Any, Any]] = None
    if args.backend == "local_vllm":
        print(f"[local_vllm] loading model once for {len(jobs)} job(s): {args.model}")
        local_ctx = _init_local_backend(args)

    finished = 0
    for job_idx, (pool_jsonl, output_jsonl) in enumerate(jobs, start=1):
        print(f"[job {job_idx}/{len(jobs)}] pool={pool_jsonl} -> output={output_jsonl}")
        remaining, append = _prepare_remaining_rows(
            pool_jsonl=pool_jsonl,
            output_jsonl=output_jsonl,
            resume=args.resume == 1,
            shuffle=args.shuffle == 1,
            seed=int(args.seed),
            limit=int(args.limit),
        )
        if not remaining:
            print("[generate] nothing to do")
            continue

        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        if args.backend == "openai_api":
            _run_openai_job(
                args=args,
                remaining=remaining,
                output_jsonl=output_jsonl,
                append=append,
            )
        else:
            assert local_ctx is not None
            processor, llm, sampling_params, cached_process_vision_info = local_ctx
            _run_local_job(
                args=args,
                remaining=remaining,
                output_jsonl=output_jsonl,
                append=append,
                processor=processor,
                llm=llm,
                sampling_params=sampling_params,
                cached_process_vision_info=cached_process_vision_info,
            )
        print(f"[generate] saved={output_jsonl}")
        finished += 1

    print(f"[generate] completed jobs={finished}/{len(jobs)}")


if __name__ == "__main__":
    main()
