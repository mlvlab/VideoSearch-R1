#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
import os
import re
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("faiss is required (pip install faiss-cpu/faiss-gpu)") from exc


SFT_EVAL_REFINE_SYSTEM_PROMPT = (
    "You are a video retrieval assistant. Your task is to analyze a retrieved video against the user query. "
    "Inside <think>...</think>, perform a step by step comparison between the query requirements and the visible "
    "evidence in the video. Identify whether a scene corresponding to the query appears in the video and determine "
    "the exact time span where it occurs. If a scene corresponding to the query appears in the video, output strictly "
    "in the following format: <answer>matched</answer> <start>START_TIME_IN_SECONDS</start> "
    "<end>END_TIME_IN_SECONDS</end> <REFINE>. Even if matched, you must still append the special token <REFINE> at "
    "the very end to allow further latent refinement. If no scene corresponding to the query appears in the video, "
    "output strictly: <answer>not_matched</answer> <REFINE>. In this case, the <REFINE> token is required to "
    "initiate a latent query update. You must always append the special token <REFINE> at the very end of the output. "
    "Do not invent details beyond what is visible. Be concise inside <think>. Do not output anything outside the "
    "specified tags."
)


def _iter_json_rows(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for row in data:
                if isinstance(row, dict):
                    yield row
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _parse_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _normalize_time_span(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    start = _parse_float_or_none(raw[0])
    end = _parse_float_or_none(raw[1])
    if start is None or end is None:
        return None
    if end <= start:
        return None
    return [start, end]


def _normalize_duration(raw: Any) -> Optional[float]:
    dur = _parse_float_or_none(raw)
    if dur is None or dur <= 0.0:
        return None
    return dur


def _format_matched_solution_with_time(
    gt_time: Optional[List[float]],
    gt_duration: Optional[float],
) -> str:
    start: Optional[float] = None
    end: Optional[float] = None
    if isinstance(gt_time, list) and len(gt_time) >= 2:
        start = _parse_float_or_none(gt_time[0])
        end = _parse_float_or_none(gt_time[1])
    if (
        start is None
        or end is None
        or (not np.isfinite(start))
        or (not np.isfinite(end))
        or end <= start
    ):
        dur = _parse_float_or_none(gt_duration)
        if dur is not None and dur > 0.0:
            start = 0.0
            end = float(dur)
        else:
            return "<answer>matched</answer><REFINE>"
    return (
        "<answer>matched</answer>"
        f"<start>{float(start):.2f}</start>"
        f"<end>{float(end):.2f}</end>"
        "<REFINE>"
    )


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def _load_query_meta(query_meta_path: str) -> Dict[str, Tuple[int, str, str]]:
    out: Dict[str, Tuple[int, str, str]] = {}
    for row_idx, row in enumerate(_iter_json_rows(query_meta_path)):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        qid = str(row.get("qid", "")).strip()
        pos_doc_id = str(row.get("pos_doc_id", "")).strip()
        if query not in out:
            out[query] = (row_idx, qid, pos_doc_id)
    return out


def _qid_from_example(ex: Dict[str, Any], q_row: int, fallback_qid: str) -> str:
    qid = str(ex.get("qid", "")).strip()
    if qid:
        return qid
    if fallback_qid:
        return fallback_qid
    desc_id = str(ex.get("desc_id", "")).strip()
    if desc_id:
        return f"{desc_id}_fig"
    return f"q_{q_row}"


def _infer_default_key(dataset_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", str(dataset_name).strip()).strip("-")
    token = token or "dataset"
    return f"Verified-Search-Minimal-Top1-{token}"


def _to_config_anno_path(output_jsonl: str, videosearch_root: str) -> str:
    root = os.path.normpath(os.path.abspath(videosearch_root))
    out = os.path.normpath(os.path.abspath(output_jsonl))
    prefix = root + os.sep
    if out.startswith(prefix):
        return out[len(prefix) :].replace(os.sep, "/")
    return out


def _format_yaml_block(
    key: str,
    anno_path: str,
    video_root: str,
    query_key: str,
    gt_key: str,
    bootstrap_key: str,
) -> List[str]:
    return [
        f"{key}:",
        f"  anno_path: {anno_path}",
        f"  video_root: {video_root}",
        f"  query_key: {query_key}",
        f"  gt_key: {gt_key}",
        f"  bootstrap_key: {bootstrap_key}",
    ]


def _replace_or_append_top_level_block(text: str, key: str, block_lines: List[str]) -> str:
    lines = text.splitlines()
    key_line = f"{key}:"
    start = -1
    for i, line in enumerate(lines):
        if line.strip() == key_line and not line.startswith((" ", "\t")):
            start = i
            break

    if start >= 0:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            line = lines[j]
            stripped = line.strip()
            if not stripped:
                continue
            if line.startswith((" ", "\t")):
                continue
            if line.startswith("#"):
                continue
            end = j
            break
        new_lines = lines[:start] + block_lines + lines[end:]
    else:
        new_lines = list(lines)
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.extend(block_lines)

    return "\n".join(new_lines).rstrip() + "\n"


def _update_data_config(
    data_config_path: str,
    dataset_key: str,
    anno_path: str,
    video_root: str,
    query_key: str,
    gt_key: str,
    bootstrap_key: str,
) -> None:
    os.makedirs(os.path.dirname(data_config_path), exist_ok=True)
    original = ""
    if os.path.exists(data_config_path):
        with open(data_config_path, "r", encoding="utf-8") as f:
            original = f.read()

    block_lines = _format_yaml_block(
        key=dataset_key,
        anno_path=anno_path,
        video_root=video_root,
        query_key=query_key,
        gt_key=gt_key,
        bootstrap_key=bootstrap_key,
    )
    merged = _replace_or_append_top_level_block(original, dataset_key, block_lines)
    with open(data_config_path, "w", encoding="utf-8") as f:
        f.write(merged)


def build_dataset(
    *,
    raw_annotation: str,
    query_embeddings_path: str,
    query_meta_path: str,
    index_faiss_path: str,
    index_id_map_path: str,
    video_root: str,
    output_jsonl: str,
    output_stats_json: str,
    dataset_name: str,
    query_key: str,
    gt_key: str,
    time_key: str,
    duration_key: str,
    hard_negative_topk: int,
    hard_negative_depth: int,
    drop_missing_video_npy: bool,
    augment_match_to_balance: bool,
    target_match_ratio: float,
    augment_seed: int,
    max_rows: int,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(output_stats_json), exist_ok=True)

    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    query_meta = _load_query_meta(query_meta_path)
    index = faiss.read_index(index_faiss_path)
    with open(index_id_map_path, "r", encoding="utf-8") as f:
        id_map = [str(x) for x in json.load(f)]

    if int(index.ntotal) != len(id_map):
        raise RuntimeError(
            f"index.ntotal ({int(index.ntotal)}) != len(id_map) ({len(id_map)})"
        )

    stats: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "raw_annotation": raw_annotation,
        "rows_total": 0,
        "rows_written": 0,
        "rows_matched": 0,
        "rows_not_matched": 0,
        "rows_written_raw": 0,
        "rows_matched_raw": 0,
        "rows_not_matched_raw": 0,
        "drop_empty_query_or_gt": 0,
        "drop_query_missing_meta": 0,
        "drop_invalid_qrow": 0,
        "drop_empty_top1": 0,
        "drop_missing_video_npy": 0,
        "augment_match_to_balance": int(bool(augment_match_to_balance)),
        "target_match_ratio": float(target_match_ratio),
        "augment_seed": int(augment_seed),
        "rows_augmented_match_added": 0,
        "rows_augmented_match_target": 0,
        "rows_augmented_match_candidates": 0,
        "rows_augmented_drop_missing_gt_npy": 0,
        "rows_augmented_drop_missing_gt_time": 0,
        "hard_negative_topk": int(hard_negative_topk),
        "hard_negative_depth": int(hard_negative_depth),
        "hard_negative_avg_count": 0.0,
    }
    rows: List[Dict[str, Any]] = []
    for ex in _iter_json_rows(raw_annotation):
        if max_rows > 0 and stats["rows_total"] >= max_rows:
            break
        stats["rows_total"] += 1

        query = str(ex.get(query_key, ex.get("query", ""))).strip()
        gt_video = str(ex.get(gt_key, ex.get("gt_video", ex.get("video", "")))).strip()
        gt_time = _normalize_time_span(ex.get(time_key, ex.get("gt_time", ex.get("time", None))))
        gt_duration = _normalize_duration(
            ex.get(duration_key, ex.get("gt_duration", ex.get("duration", None)))
        )
        if not query or not gt_video:
            stats["drop_empty_query_or_gt"] += 1
            continue

        qmeta = query_meta.get(query)
        if qmeta is None:
            stats["drop_query_missing_meta"] += 1
            continue
        q_row, fallback_qid, _ = qmeta
        if q_row < 0 or q_row >= len(query_embeddings):
            stats["drop_invalid_qrow"] += 1
            continue

        q_vec = np.asarray(query_embeddings[q_row], dtype=np.float32)
        q_vec = _l2_normalize(q_vec).reshape(1, -1)
        depth = int(max(1, hard_negative_depth, hard_negative_topk + 1))
        scores, idxs = index.search(q_vec, depth)
        top_idx = int(idxs[0][0]) if idxs.size > 0 else -1
        if top_idx < 0:
            stats["drop_empty_top1"] += 1
            continue
        top1_video = id_map[top_idx]

        if drop_missing_video_npy:
            npy_path = os.path.join(video_root, f"{top1_video}.npy")
            if not os.path.exists(npy_path):
                stats["drop_missing_video_npy"] += 1
                continue

        hard_negative_ids: List[str] = []
        hard_negative_scores: List[float] = []
        hard_negative_ranks: List[int] = []
        seen_neg = set()
        if idxs.size > 0:
            for rank_pos, idx_val in enumerate(idxs[0].tolist(), start=1):
                if idx_val is None or int(idx_val) < 0:
                    continue
                vid = id_map[int(idx_val)]
                if vid == gt_video or vid in seen_neg:
                    continue
                seen_neg.add(vid)
                hard_negative_ids.append(vid)
                score_val = float(scores[0][rank_pos - 1]) if scores.size > 0 else 0.0
                hard_negative_scores.append(score_val)
                hard_negative_ranks.append(int(rank_pos))
                if len(hard_negative_ids) >= int(hard_negative_topk):
                    break

        is_match = top1_video == gt_video
        solution = (
            "<answer>matched</answer>"
            if is_match
            else "<answer>not_matched</answer><REFINE>"
        )
        problem = f"{SFT_EVAL_REFINE_SYSTEM_PROMPT}\n\nQuery:\n{query}"
        qid = _qid_from_example(ex, q_row, fallback_qid)

        row: Dict[str, Any] = {
            "problem_id": -1,
            "qid": qid,
            "problem": problem,
            "data_type": "video",
            "problem_type": "multiple choice",
            "options": [
                "<answer>matched</answer>",
                "<answer>not_matched</answer><REFINE>",
            ],
            "solution": solution,
            "path": f"{top1_video}.npy",
            "data_source": f"Verified-Search-Minimal-Top1-{dataset_name}",
            "query": query,
            "gt_video": gt_video,
            "gt_time": gt_time,
            "gt_duration": gt_duration,
            "retrieved_video": top1_video,
            "label": "matched" if is_match else "not_matched",
            "hard_negative_ids": hard_negative_ids,
            "hard_negative_scores": hard_negative_scores,
            "hard_negative_ranks": hard_negative_ranks,
            "is_augmented": False,
        }
        rows.append(row)

    rows_matched_raw = sum(1 for r in rows if str(r.get("label", "")).strip() == "matched")
    rows_not_matched_raw = sum(
        1 for r in rows if str(r.get("label", "")).strip() == "not_matched"
    )
    stats["rows_written_raw"] = len(rows)
    stats["rows_matched_raw"] = rows_matched_raw
    stats["rows_not_matched_raw"] = rows_not_matched_raw

    if augment_match_to_balance and target_match_ratio > 0.0 and rows_not_matched_raw > 0:
        target_match_count = int(math.ceil(rows_not_matched_raw * float(target_match_ratio)))
        stats["rows_augmented_match_target"] = target_match_count
        need = max(0, target_match_count - rows_matched_raw)
        if need > 0:
            candidates: List[Dict[str, Any]] = []
            for row in rows:
                if str(row.get("label", "")).strip() != "not_matched":
                    continue
                gt_video = str(row.get("gt_video", "")).strip()
                gt_time = row.get("gt_time", None)
                gt_duration = row.get("gt_duration", None)
                if not gt_video:
                    continue
                gt_path = os.path.join(video_root, f"{gt_video}.npy")
                if not os.path.exists(gt_path):
                    stats["rows_augmented_drop_missing_gt_npy"] += 1
                    continue
                formatted = _format_matched_solution_with_time(gt_time, gt_duration)
                if "<start>" not in formatted or "<end>" not in formatted:
                    stats["rows_augmented_drop_missing_gt_time"] += 1
                    continue
                candidates.append(row)
            stats["rows_augmented_match_candidates"] = len(candidates)

            if candidates:
                rng = random.Random(int(augment_seed))
                selected: List[Dict[str, Any]] = []
                while len(selected) < need:
                    cycle = list(candidates)
                    rng.shuffle(cycle)
                    take = min(need - len(selected), len(cycle))
                    selected.extend(cycle[:take])

                aug_rows: List[Dict[str, Any]] = []
                for src in selected:
                    gt_video = str(src.get("gt_video", "")).strip()
                    if not gt_video:
                        continue
                    gt_time = src.get("gt_time", None)
                    gt_duration = src.get("gt_duration", None)
                    aug_solution = _format_matched_solution_with_time(gt_time, gt_duration)
                    if "<start>" not in aug_solution or "<end>" not in aug_solution:
                        continue
                    aug = dict(src)
                    aug["problem_id"] = -1
                    aug["solution"] = aug_solution
                    aug["path"] = f"{gt_video}.npy"
                    aug["retrieved_video"] = gt_video
                    aug["label"] = "matched"
                    aug["is_augmented"] = True
                    aug["augmentation_type"] = "gt_video_positive"
                    aug["source_label"] = "not_matched"
                    aug["source_retrieved_video"] = str(src.get("retrieved_video", "")).strip()
                    aug_rows.append(aug)
                stats["rows_augmented_match_added"] = len(aug_rows)
                rows.extend(aug_rows)

    for idx, row in enumerate(rows):
        row["problem_id"] = idx

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows_matched_final = sum(1 for r in rows if str(r.get("label", "")).strip() == "matched")
    rows_not_matched_final = sum(
        1 for r in rows if str(r.get("label", "")).strip() == "not_matched"
    )
    stats["rows_written"] = len(rows)
    stats["rows_matched"] = rows_matched_final
    stats["rows_not_matched"] = rows_not_matched_final
    if stats["rows_written"] > 0:
        total_hn = 0
        for r in rows:
            hard_ids = r.get("hard_negative_ids", [])
            if isinstance(hard_ids, list):
                total_hn += len(hard_ids)
        stats["hard_negative_avg_count"] = total_hn / float(stats["rows_written"])

    with open(output_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Verified-Search-Minimal-Top1 jsonl from raw annotation and "
            "auto-register dataset entry in data_config.yaml."
        )
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, default="")
    parser.add_argument("--videosearch-root", type=str, default=os.environ.get("VIDEOSEARCH_REPO_ROOT", "."))
    parser.add_argument("--raw-annotation", type=str, required=True)
    parser.add_argument("--query-embeddings-path", type=str, required=True)
    parser.add_argument("--query-meta-path", type=str, required=True)
    parser.add_argument("--index-faiss-path", type=str, required=True)
    parser.add_argument("--index-id-map-path", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-stats-json", type=str, required=True)
    parser.add_argument("--data-config-path", type=str, required=True)
    parser.add_argument("--anno-path-in-config", type=str, default="")
    parser.add_argument("--query-key", type=str, default="fig_desc")
    parser.add_argument("--gt-key", type=str, default="video")
    parser.add_argument("--time-key", type=str, default="time")
    parser.add_argument("--duration-key", type=str, default="duration")
    parser.add_argument("--bootstrap-key", type=str, default="retrieved_video")
    parser.add_argument("--hard-negative-topk", type=int, default=24)
    parser.add_argument("--hard-negative-depth", type=int, default=200)
    parser.add_argument("--drop-missing-video-npy", type=int, default=1)
    parser.add_argument("--augment-match-to-balance", type=int, default=0)
    parser.add_argument("--target-match-ratio", type=float, default=1.0)
    parser.add_argument("--augment-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--register-config", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_key = args.dataset_key.strip() or _infer_default_key(args.dataset_name)
    stats = build_dataset(
        raw_annotation=args.raw_annotation,
        query_embeddings_path=args.query_embeddings_path,
        query_meta_path=args.query_meta_path,
        index_faiss_path=args.index_faiss_path,
        index_id_map_path=args.index_id_map_path,
        video_root=args.video_root,
        output_jsonl=args.output_jsonl,
        output_stats_json=args.output_stats_json,
        dataset_name=args.dataset_name,
        query_key=args.query_key,
        gt_key=args.gt_key,
        time_key=args.time_key,
        duration_key=args.duration_key,
        hard_negative_topk=args.hard_negative_topk,
        hard_negative_depth=args.hard_negative_depth,
        drop_missing_video_npy=bool(args.drop_missing_video_npy),
        augment_match_to_balance=bool(args.augment_match_to_balance),
        target_match_ratio=float(args.target_match_ratio),
        augment_seed=int(args.augment_seed),
        max_rows=max(0, int(args.max_rows)),
    )

    if int(args.register_config) == 1:
        anno_path = args.anno_path_in_config.strip() or _to_config_anno_path(
            args.output_jsonl, args.videosearch_root
        )
        _update_data_config(
            data_config_path=args.data_config_path,
            dataset_key=dataset_key,
            anno_path=anno_path,
            video_root=args.video_root,
            query_key="query",
            gt_key="gt_video",
            bootstrap_key=args.bootstrap_key,
        )
        print(
            f"[build_verified_search_minimal_top1] registered dataset key "
            f"'{dataset_key}' in {args.data_config_path}"
        )

    print(
        "[build_verified_search_minimal_top1] "
        f"dataset={args.dataset_name} wrote={stats['rows_written']} "
        f"matched={stats['rows_matched']} not_matched={stats['rows_not_matched']} "
        f"output={args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
