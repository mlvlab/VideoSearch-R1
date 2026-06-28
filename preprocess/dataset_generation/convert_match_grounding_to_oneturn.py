#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are a video retrieval assistant. Your task is to analyze a retrieved video against the user query. "
    "Inside <think>...</think>, perform a step by step comparison between the query requirements and the visible evidence in the video. "
    "Identify whether a scene corresponding to the query appears in the video and determine the exact time span where it occurs. "
    "If a scene corresponding to the query appears in the video, output strictly in the following format: "
    "<answer>matched</answer> <start>START_TIME_IN_SECONDS</start> <end>END_TIME_IN_SECONDS</end> <REFINE>. "
    "Even if matched, you must still append the special token <REFINE> at the very end to allow further latent refinement. "
    "If no scene corresponding to the query appears in the video, output strictly: <answer>not_matched</answer> <REFINE>. "
    "In this case, the <REFINE> token is required to initiate a latent query update. "
    "You must always append the special token <REFINE> at the very end of the output. "
    "Do not invent details beyond what is visible. Be concise inside <think>. "
    "Do not output anything outside the specified tags."
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"invalid json at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"row at {path}:{line_no} is not a JSON object")
            yield row


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not (x == x):
        return None
    return x


def _strip_ext(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    if not base:
        return ""
    stem, _ = os.path.splitext(base)
    return stem or base


def _normalize_binary_label(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    if s in {"match", "matched", "pos", "positive"}:
        return "pos"
    if s in {"not_match", "not matched", "not_matched", "neg", "negative", "no_match"}:
        return "neg"

    norm = re.sub(r"[_\-\s]+", " ", s)
    if re.search(
        r"\b(?:not match|no match|non match|notmatching|unmatch|unmatched|mismatch|"
        r"does not match|doesn't match|do not match|fails to match)\b",
        norm,
    ):
        return "neg"
    if re.search(r"\bmatch(?:es|ed|ing)?\b", norm):
        return "pos"
    return ""


def _normalize_answer_label(v: Any) -> str:
    binary = _normalize_binary_label(v)
    if binary == "pos":
        return "matched"
    if binary == "neg":
        return "not_matched"
    return ""


def _normalize_reasoning(v: Any) -> str:
    text = str(v or "").strip()
    if not text:
        return ""
    if text.lower().startswith("reasoning:"):
        text = text.split(":", 1)[1].strip()
    return " ".join(text.split())


def _format_seconds(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}.0"
    out = f"{x:.6f}".rstrip("0").rstrip(".")
    return out or "0.0"


def _pick_top1_video_id(row: Dict[str, Any]) -> str:
    for key in ("top1_video_id", "top1_doc_id", "video_id", "retrieved_video"):
        val = str(row.get(key, "")).strip()
        if val:
            return _strip_ext(val)

    path = str(row.get("top1_video_path", "")).strip() or str(row.get("video_path", "")).strip()
    if path:
        return _strip_ext(path)
    return ""


def _pick_query_pos_video_id(row: Dict[str, Any]) -> str:
    for key in ("query_pos_video_id", "query_pos_doc_id", "gt_video_id"):
        val = str(row.get(key, "")).strip()
        if val:
            return _strip_ext(val)
    return ""


def _derive_pair_id(row: Dict[str, Any], idx: int, top1_video_id: str) -> str:
    pair_id = str(row.get("pair_id", "")).strip()
    if pair_id:
        return pair_id
    qid = str(row.get("qid", "")).strip() or f"idx_{idx}"
    if top1_video_id:
        return f"{qid}::{top1_video_id}"
    return qid


def _derive_gold_bin(row: Dict[str, Any], answer_label: str) -> str:
    gold = _normalize_binary_label(row.get("gold_label", ""))
    if gold:
        return gold

    top1 = str(row.get("top1_doc_id", "")).strip() or str(row.get("top1_video_id", "")).strip()
    pos = str(row.get("query_pos_doc_id", "")).strip() or str(row.get("query_pos_video_id", "")).strip()
    if top1 and pos:
        return "pos" if _strip_ext(top1) == _strip_ext(pos) else "neg"

    if answer_label == "matched":
        return "pos"
    return "neg"


def _derive_answer_label(row: Dict[str, Any]) -> str:
    model = _normalize_answer_label(row.get("model_label", ""))
    if model:
        return model

    gold = _normalize_answer_label(row.get("gold_label", ""))
    if gold:
        return gold

    top1 = str(row.get("top1_doc_id", "")).strip() or str(row.get("top1_video_id", "")).strip()
    pos = str(row.get("query_pos_doc_id", "")).strip() or str(row.get("query_pos_video_id", "")).strip()
    if top1 and pos:
        return "matched" if _strip_ext(top1) == _strip_ext(pos) else "not_matched"

    return "not_matched"


def _pick_matched_span(row: Dict[str, Any]) -> Tuple[float, float]:
    candidates = [
        (row.get("model_temporal_start"), row.get("model_temporal_end")),
        (row.get("gt_start"), row.get("gt_end")),
        (row.get("top1_start"), row.get("top1_end")),
    ]
    for start_v, end_v in candidates:
        start = _safe_float(start_v)
        end = _safe_float(end_v)
        if start is None or end is None:
            continue
        if start < 0:
            start = 0.0
        if end < start:
            continue
        return start, end

    duration = _safe_float(row.get("top1_duration"))
    if duration is not None and duration >= 0:
        return 0.0, float(duration)

    return 0.0, 0.0


def _ensure_video_name(top1_video_id: str, row: Dict[str, Any], video_ext: str) -> str:
    ext = str(video_ext or ".npy").strip()
    if not ext.startswith("."):
        ext = f".{ext}"

    if top1_video_id:
        return f"{_strip_ext(top1_video_id)}{ext}"

    path = str(row.get("top1_video_path", "")).strip() or str(row.get("video_path", "")).strip()
    if path:
        return f"{_strip_ext(path)}{ext}"

    return f"unknown{ext}"


def _build_assistant_content(row: Dict[str, Any], answer_label: str) -> str:
    reasoning = _normalize_reasoning(row.get("model_reasoning", ""))
    if not reasoning:
        reasoning = _normalize_reasoning(row.get("model_think", ""))

    if answer_label == "matched":
        if not reasoning:
            reasoning = "The retrieved video contains the key query-required visual events and timing cues."
        start, end = _pick_matched_span(row)
        return (
            f"<think> {reasoning} </think> "
            f"<answer>matched</answer> "
            f"<start>{_format_seconds(start)}</start> "
            f"<end>{_format_seconds(end)}</end> "
            f"<REFINE>"
        )

    if not reasoning:
        reasoning = "The retrieved video does not satisfy the query-required visual evidence."
    return f"<think> {reasoning} </think> <answer>not_matched</answer> <REFINE>"


def _build_user_content(query: str) -> str:
    return f'Query: "{query}"\nRetrieved video: <video>'


def _load_system_prompt(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return SYSTEM_PROMPT
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    return txt or SYSTEM_PROMPT


def _parse_exclude_finish_reasons(raw: str) -> set[str]:
    return {str(x).strip().lower() for x in str(raw or "").split(",") if str(x).strip()}


def _downsample_to_equal_binary(
    items: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    pos_items = [x for x in items if str(x.get("meta", {}).get("gold_label", "")) == "pos"]
    neg_items = [x for x in items if str(x.get("meta", {}).get("gold_label", "")) == "neg"]

    counts = {
        "before_pos": len(pos_items),
        "before_neg": len(neg_items),
        "dropped_pos": 0,
        "dropped_neg": 0,
    }
    if not pos_items or not neg_items:
        counts["after_pos"] = len(pos_items)
        counts["after_neg"] = len(neg_items)
        return items, counts

    keep_n = min(len(pos_items), len(neg_items))
    rng = random.Random(seed)

    rng.shuffle(pos_items)
    rng.shuffle(neg_items)

    kept_pos = pos_items[:keep_n]
    kept_neg = neg_items[:keep_n]
    counts["dropped_pos"] = len(pos_items) - len(kept_pos)
    counts["dropped_neg"] = len(neg_items) - len(kept_neg)
    counts["after_pos"] = len(kept_pos)
    counts["after_neg"] = len(kept_neg)

    balanced = kept_pos + kept_neg
    rng.shuffle(balanced)
    return balanced, counts


def convert(
    *,
    input_jsonl: str,
    output_json: str,
    source: str,
    episode_type: str,
    video_ext: str,
    system_prompt: str,
    limit: int,
    indent: int,
    exclude_finish_reasons: set[str],
    balance_binary_by_downsample: bool,
    balance_seed: int,
) -> None:
    rows = list(_iter_jsonl(input_jsonl))
    if limit > 0:
        rows = rows[:limit]

    data: List[Dict[str, Any]] = []
    pos = 0
    neg = 0
    skipped_finish_reason = 0
    for idx, row in enumerate(rows):
        finish_reason = str(row.get("model_finish_reason", "")).strip().lower()
        if finish_reason and finish_reason in exclude_finish_reasons:
            skipped_finish_reason += 1
            continue

        qid = str(row.get("qid", "")).strip() or f"idx_{idx}"
        query = str(row.get("query", "")).strip()
        top1_video_id = _pick_top1_video_id(row)
        query_pos_video_id = _pick_query_pos_video_id(row)
        pair_id = _derive_pair_id(row, idx, top1_video_id)

        answer_label = _derive_answer_label(row)
        gold_bin = _derive_gold_bin(row, answer_label)
        if gold_bin == "pos":
            pos += 1
        else:
            neg += 1

        assistant = _build_assistant_content(row, answer_label)

        item: Dict[str, Any] = {
            "qid": qid,
            "query": query,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_content(query)},
                {"role": "assistant", "content": assistant},
            ],
            "videos": [_ensure_video_name(top1_video_id, row, video_ext)],
            "meta": {
                "source": source,
                "gold_label": gold_bin,
                "video_id": top1_video_id,
                "query_pos_video_id": query_pos_video_id,
                "pair_id": pair_id,
                "episode_type": episode_type,
                "num_negs": 0 if gold_bin == "pos" else 1,
                "top1_video_id": top1_video_id,
                "top1_gold_label": gold_bin,
            },
        }
        data.append(item)

    balance_stats = {
        "before_pos": pos,
        "before_neg": neg,
        "dropped_pos": 0,
        "dropped_neg": 0,
        "after_pos": pos,
        "after_neg": neg,
    }
    if balance_binary_by_downsample:
        data, balance_stats = _downsample_to_equal_binary(data, seed=balance_seed)
        pos = int(balance_stats["after_pos"])
        neg = int(balance_stats["after_neg"])

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(
        "[convert_oneturn] "
        f"input={input_jsonl} rows={len(rows)} kept={len(data)} "
        f"skipped_finish_reason={skipped_finish_reason} output={output_json} "
        f"pos={pos} neg={neg} "
        f"balance_binary_by_downsample={int(balance_binary_by_downsample)} "
        f"dropped_pos={int(balance_stats['dropped_pos'])} "
        f"dropped_neg={int(balance_stats['dropped_neg'])}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--source", default="pairs_with_reasoning")
    ap.add_argument("--episode_type", default="T1")
    ap.add_argument("--video_ext", default=".npy")
    ap.add_argument("--system_prompt_file", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--indent", type=int, default=2)
    ap.add_argument("--exclude_model_finish_reasons", default="")
    ap.add_argument("--balance_binary_by_downsample", type=int, default=0)
    ap.add_argument("--balance_seed", type=int, default=42)
    args = ap.parse_args()

    input_jsonl = str(args.input_jsonl).strip()
    output_json = str(args.output_json).strip()
    if not input_jsonl or not os.path.exists(input_jsonl):
        raise SystemExit(f"Missing input_jsonl: {input_jsonl}")
    if not output_json:
        raise SystemExit("output_json is required")

    system_prompt = _load_system_prompt(args.system_prompt_file)
    convert(
        input_jsonl=input_jsonl,
        output_json=output_json,
        source=str(args.source).strip() or "pairs_with_reasoning",
        episode_type=str(args.episode_type).strip() or "T1",
        video_ext=str(args.video_ext).strip() or ".npy",
        system_prompt=system_prompt,
        limit=int(args.limit),
        indent=max(0, int(args.indent)),
        exclude_finish_reasons=_parse_exclude_finish_reasons(args.exclude_model_finish_reasons),
        balance_binary_by_downsample=bool(int(args.balance_binary_by_downsample)),
        balance_seed=int(args.balance_seed),
    )


if __name__ == "__main__":
    main()
