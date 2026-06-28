#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _warn_bad_jsonl_line(path: str, line_no: int, exc: Exception) -> None:
    print(
        f"[warn] skip malformed jsonl line: path={path} line={line_no} err={exc}",
        file=sys.stderr,
    )


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _warn_bad_jsonl_line(path, line_no, exc)
                continue
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-9
    return vecs / denom


def _load_video_meta(path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    id_order: List[str] = []
    by_doc: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _warn_bad_jsonl_line(path, line_no, exc)
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                doc_id = str(row.get("video_id", "")).strip()
            if not doc_id:
                continue
            id_order.append(doc_id)
            if doc_id not in by_doc:
                by_doc[doc_id] = row
    return id_order, by_doc


def _load_id_map_from_docid2row(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        docid2row = json.load(f)
    items = sorted(((int(v), str(k)) for k, v in docid2row.items()), key=lambda x: x[0])
    return [doc_id for _, doc_id in items]


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return float(x)


def _qid_to_desc_id(qid: str) -> Optional[str]:
    token = str(qid or "").strip()
    if not token:
        return None
    if token.endswith("_fig"):
        token = token[: -len("_fig")]
    if token.isdigit():
        return token
    prefix = token.split("_", 1)[0]
    if prefix.isdigit():
        return prefix
    return None


def _load_raw_annotation_index(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_desc_id: Dict[str, Dict[str, Any]] = {}
    by_fig_desc: Dict[str, Dict[str, Any]] = {}
    by_cog_desc: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        desc_id = str(row.get("desc_id", "")).strip()
        if desc_id and desc_id not in by_desc_id:
            by_desc_id[desc_id] = row
        fig_desc = _norm_text(row.get("fig_desc", ""))
        if fig_desc and fig_desc not in by_fig_desc:
            by_fig_desc[fig_desc] = row
        cog_desc = _norm_text(row.get("cog_desc", ""))
        if cog_desc and cog_desc not in by_cog_desc:
            by_cog_desc[cog_desc] = row
    print(
        f"[raw_annotation] loaded source={path} "
        f"by_desc_id={len(by_desc_id)} by_fig_desc={len(by_fig_desc)} by_cog_desc={len(by_cog_desc)}"
    )
    return {
        "by_desc_id": by_desc_id,
        "by_fig_desc": by_fig_desc,
        "by_cog_desc": by_cog_desc,
    }


def _resolve_raw_annotation_row(
    qid: str,
    query: str,
    index: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    if not index:
        return None
    by_desc_id = index.get("by_desc_id", {})
    by_fig_desc = index.get("by_fig_desc", {})
    by_cog_desc = index.get("by_cog_desc", {})

    desc_id = _qid_to_desc_id(qid)
    if desc_id and desc_id in by_desc_id:
        return by_desc_id[desc_id]

    q = _norm_text(query)
    if q in by_fig_desc:
        return by_fig_desc[q]
    if q in by_cog_desc:
        return by_cog_desc[q]
    return None


def _extract_gt_span_from_raw(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None

    time_value = row.get("time", None)
    if isinstance(time_value, (list, tuple)) and len(time_value) >= 2:
        start = _to_float(time_value[0])
        end = _to_float(time_value[1])

    if start is None:
        start = _to_float(row.get("start", None))
    if end is None:
        end = _to_float(row.get("end", None))
    # For grounding metadata, duration should match the grounded span length.
    if start is not None and end is not None:
        duration = max(0.0, float(end) - float(start))
    else:
        duration = _to_float(row.get("segment_duration", None))
        if duration is None:
            duration = _to_float(row.get("duration", None))

    return start, end, duration


def _qid_from_row(row: Dict[str, Any], idx: int) -> str:
    qid = str(row.get("qid", "")).strip()
    if qid:
        return qid
    desc_id = str(row.get("desc_id", "")).strip()
    if desc_id:
        return f"{desc_id}_fig"
    return f"idx_{idx}"


def _compute_top1_faiss(
    query_vecs: np.ndarray,
    index: Any,
    normalize_queries: bool,
    query_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = int(query_vecs.shape[0])
    top1_idx = np.empty(total, dtype=np.int64)
    top1_score = np.empty(total, dtype=np.float32)

    for start in range(0, total, max(1, query_batch_size)):
        end = min(total, start + max(1, query_batch_size))
        q_batch = query_vecs[start:end].astype(np.float32, copy=False)
        if normalize_queries:
            q_batch = _l2_normalize(q_batch)
        scores, idx = index.search(q_batch, 1)
        top1_idx[start:end] = idx[:, 0]
        top1_score[start:end] = scores[:, 0]
    return top1_idx, top1_score


def _compute_top1_bruteforce(
    query_vecs: np.ndarray,
    video_vecs: np.ndarray,
    query_batch_size: int,
    video_chunk_size: int,
    normalize_queries: bool,
    normalize_videos: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    total = int(query_vecs.shape[0])
    top1_idx = np.empty(total, dtype=np.int64)
    top1_score = np.empty(total, dtype=np.float32)

    for start in range(0, total, max(1, query_batch_size)):
        end = min(total, start + max(1, query_batch_size))
        q_batch = query_vecs[start:end].astype(np.float32, copy=False)
        if normalize_queries:
            q_batch = _l2_normalize(q_batch)

        best_scores = np.full((end - start,), -np.inf, dtype=np.float32)
        best_indices = np.full((end - start,), -1, dtype=np.int64)

        for v_start in range(0, video_vecs.shape[0], max(1, video_chunk_size)):
            v_end = min(video_vecs.shape[0], v_start + max(1, video_chunk_size))
            v_chunk = video_vecs[v_start:v_end].astype(np.float32, copy=False)
            if normalize_videos:
                v_chunk = _l2_normalize(v_chunk)

            scores = q_batch @ v_chunk.T
            chunk_best = np.argmax(scores, axis=1)
            chunk_scores = scores[np.arange(scores.shape[0]), chunk_best]
            update = chunk_scores > best_scores
            if np.any(update):
                best_scores[update] = chunk_scores[update]
                best_indices[update] = v_start + chunk_best[update]

        top1_idx[start:end] = best_indices
        top1_score[start:end] = best_scores

    return top1_idx, top1_score


def _resolve_video_embed_dir(structured_root: str, preferred: str, fallback: str) -> str:
    p = os.path.join(structured_root, preferred)
    if os.path.isdir(p):
        return p
    q = os.path.join(structured_root, fallback)
    return q


def _resolve_target_count(available: int, requested: int, allow_smaller: bool) -> int:
    if requested <= 0:
        return available
    if requested > available:
        if not allow_smaller:
            raise SystemExit(f"requested={requested} but available={available}")
        return available
    return requested


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured_root", default="data/activitynet/train")
    ap.add_argument("--video_embed_subdir", default="video_embedding_1fps")
    ap.add_argument("--fallback_video_embed_subdir", default="video_embedding")
    ap.add_argument("--query_embeddings", default="")
    ap.add_argument("--query_meta", default="")
    ap.add_argument("--video_embeddings", default="")
    ap.add_argument("--video_meta", default="")
    ap.add_argument("--docid2row", default="")
    ap.add_argument("--index_dir", default="")
    ap.add_argument("--index_path", default="")
    ap.add_argument("--id_map_json", default="")
    ap.add_argument("--use_faiss", type=int, default=1)
    ap.add_argument("--normalize_queries", type=int, default=1)
    ap.add_argument("--normalize_videos", type=int, default=1)
    ap.add_argument("--query_batch_size", type=int, default=64)
    ap.add_argument("--video_chunk_size", type=int, default=4096)
    ap.add_argument("--num_match", type=int, default=1000)
    ap.add_argument("--num_not_match", type=int, default=1000)
    ap.add_argument("--allow_smaller", type=int, default=1)
    ap.add_argument("--exclude_missing_pos", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_output", type=int, default=1)
    ap.add_argument("--raw_annotation_jsonl", default="")
    ap.add_argument("--output_jsonl", default="data/dataset_generation/top1_pool.train.jsonl")
    args = ap.parse_args()

    structured_root = args.structured_root
    video_embed_dir = _resolve_video_embed_dir(
        structured_root, args.video_embed_subdir, args.fallback_video_embed_subdir
    )

    query_embeddings = args.query_embeddings or os.path.join(
        structured_root, "query_embedding", "query_embeddings.train.npy"
    )
    query_meta_path = args.query_meta or os.path.join(
        structured_root, "query_embedding", "query_meta.train.jsonl"
    )
    video_embeddings = args.video_embeddings or os.path.join(video_embed_dir, "segment_embeds.npy")
    video_meta_path = args.video_meta or os.path.join(video_embed_dir, "meta.jsonl")
    docid2row_path = args.docid2row or os.path.join(video_embed_dir, "docid2row.json")
    index_dir = args.index_dir or os.path.join(structured_root, "index")
    index_path = args.index_path or os.path.join(index_dir, "index.faiss")
    id_map_json = args.id_map_json or os.path.join(index_dir, "id_map.json")
    raw_annotation_jsonl = str(args.raw_annotation_jsonl or "").strip()
    if not raw_annotation_jsonl:
        candidates = [
            os.path.join(os.path.dirname(structured_root), "raw_annotation", "train.jsonl"),
            os.path.join(structured_root, "raw_annotation", "train.jsonl"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                raw_annotation_jsonl = cand
                break

    if not os.path.exists(query_embeddings):
        raise SystemExit(f"Missing query_embeddings: {query_embeddings}")
    if not os.path.exists(query_meta_path):
        raise SystemExit(f"Missing query_meta: {query_meta_path}")
    if not os.path.exists(video_meta_path):
        raise SystemExit(f"Missing video_meta: {video_meta_path}")

    query_rows = _read_jsonl(query_meta_path)
    query_vecs = np.load(query_embeddings, mmap_mode="r")
    if int(query_vecs.shape[0]) != len(query_rows):
        raise SystemExit(f"query_embeddings/meta mismatch: {query_vecs.shape[0]} != {len(query_rows)}")

    if args.limit > 0:
        query_rows = query_rows[: args.limit]
        query_vecs = query_vecs[: args.limit]

    fallback_id_map, doc_meta_map = _load_video_meta(video_meta_path)
    if not fallback_id_map:
        raise SystemExit("video_meta is empty")

    raw_annotation_index: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    if raw_annotation_jsonl:
        if os.path.exists(raw_annotation_jsonl):
            raw_annotation_index = _load_raw_annotation_index(raw_annotation_jsonl)
        else:
            print(f"[warn] raw_annotation_jsonl not found: {raw_annotation_jsonl}", file=sys.stderr)

    use_faiss = False
    id_map: List[str] = []
    index = None
    if args.use_faiss == 1 and os.path.exists(index_path) and os.path.exists(id_map_json):
        try:
            import faiss  # type: ignore
        except Exception:
            faiss = None  # type: ignore
        if faiss is not None:
            index = faiss.read_index(index_path)
            with open(id_map_json, "r", encoding="utf-8") as f:
                id_map = json.load(f)
            use_faiss = True
        else:
            print("[warn] faiss is not available; falling back to brute-force", file=sys.stderr)

    video_vecs = None
    if not use_faiss:
        if not os.path.exists(video_embeddings):
            raise SystemExit(f"Missing video_embeddings for brute-force mode: {video_embeddings}")
        if not os.path.exists(docid2row_path):
            raise SystemExit(f"Missing docid2row for brute-force mode: {docid2row_path}")
        id_map = _load_id_map_from_docid2row(docid2row_path)
        video_vecs = np.load(video_embeddings, mmap_mode="r")
        if int(video_vecs.shape[0]) != len(id_map):
            raise SystemExit(f"video_embeddings/id_map mismatch: {video_vecs.shape[0]} != {len(id_map)}")

    id_set = set(id_map)
    valid_rows: List[Dict[str, Any]] = []
    valid_vecs: List[np.ndarray] = []
    for i, row in enumerate(query_rows):
        query = str(row.get("query", "")).strip()
        pos_id = str(row.get("pos_doc_id", "")).strip()
        if not query or not pos_id:
            continue
        if args.exclude_missing_pos == 1 and pos_id not in id_set:
            continue
        valid_rows.append(
            {
                "qid": _qid_from_row(row, i),
                "query": query,
                "query_index": i,
                "query_pos_doc_id": pos_id,
                "fig_score": row.get("fig_score", None),
            }
        )
        valid_vecs.append(query_vecs[i])

    if not valid_rows:
        raise SystemExit("No valid query rows after filtering")

    valid_vecs_arr = np.asarray(valid_vecs)
    if use_faiss:
        top1_idx, top1_score = _compute_top1_faiss(
            valid_vecs_arr,
            index,
            args.normalize_queries == 1,
            args.query_batch_size,
        )
    else:
        assert video_vecs is not None
        top1_idx, top1_score = _compute_top1_bruteforce(
            valid_vecs_arr,
            video_vecs,
            args.query_batch_size,
            args.video_chunk_size,
            args.normalize_queries == 1,
            args.normalize_videos == 1,
        )

    match_rows: List[Dict[str, Any]] = []
    not_rows: List[Dict[str, Any]] = []
    raw_gt_hit = 0
    for i, row in enumerate(valid_rows):
        vidx = int(top1_idx[i])
        if vidx < 0 or vidx >= len(id_map):
            continue
        top1_doc_id = str(id_map[vidx])
        pos_doc_id = str(row["query_pos_doc_id"])
        gold_label = "match" if top1_doc_id == pos_doc_id else "not_match"

        top1_meta = doc_meta_map.get(top1_doc_id, {})
        gt_meta = doc_meta_map.get(pos_doc_id, {})
        gt_video_id = str(gt_meta.get("video_id", "")).strip() or pos_doc_id
        gt_video_path = str(gt_meta.get("video_path", "")).strip()
        gt_start = gt_meta.get("start", None)
        gt_end = gt_meta.get("end", None)
        gt_duration = gt_meta.get("duration", None)

        raw_row = _resolve_raw_annotation_row(
            qid=str(row["qid"]),
            query=str(row["query"]),
            index=raw_annotation_index,
        )
        if raw_row is not None:
            raw_gt_hit += 1
            raw_video_id = str(raw_row.get("video", "")).strip()
            if raw_video_id:
                gt_video_id = raw_video_id
                gt_video_meta = doc_meta_map.get(raw_video_id, {})
                gt_video_path = str(gt_video_meta.get("video_path", "")).strip() or gt_video_path

            raw_start, raw_end, raw_duration = _extract_gt_span_from_raw(raw_row)
            if raw_start is not None:
                gt_start = raw_start
            if raw_end is not None:
                gt_end = raw_end
            if raw_duration is not None:
                gt_duration = raw_duration

        out = {
            "pair_id": f"{row['qid']}::{top1_doc_id}",
            "qid": row["qid"],
            "query": row["query"],
            "query_index": int(row["query_index"]),
            "query_pos_doc_id": pos_doc_id,
            "fig_score": row.get("fig_score", None),
            "top1_doc_id": top1_doc_id,
            "top1_score": float(top1_score[i]),
            "top1_rank": 1,
            "gold_label": gold_label,
            "top1_video_id": str(top1_meta.get("video_id", "")).strip() or top1_doc_id,
            "top1_video_path": str(top1_meta.get("video_path", "")).strip(),
            "top1_start": top1_meta.get("start", None),
            "top1_end": top1_meta.get("end", None),
            "top1_duration": top1_meta.get("duration", None),
            "gt_video_id": gt_video_id,
            "gt_video_path": gt_video_path,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "gt_duration": gt_duration,
        }
        if gold_label == "match":
            match_rows.append(out)
        else:
            not_rows.append(out)

    rng = random.Random(args.seed)
    rng.shuffle(match_rows)
    rng.shuffle(not_rows)

    n_match = _resolve_target_count(len(match_rows), args.num_match, args.allow_smaller == 1)
    n_not = _resolve_target_count(len(not_rows), args.num_not_match, args.allow_smaller == 1)

    selected = match_rows[:n_match] + not_rows[:n_not]
    if args.shuffle_output == 1:
        rng.shuffle(selected)

    _write_jsonl(args.output_jsonl, selected)

    print(f"[pool] queries_total={len(query_rows)} valid={len(valid_rows)}")
    print(f"[pool] match_available={len(match_rows)} not_match_available={len(not_rows)}")
    print(f"[pool] raw_gt_hit={raw_gt_hit}")
    print(f"[pool] selected_match={n_match} selected_not_match={n_not}")
    print(f"[pool] saved={args.output_jsonl} total={len(selected)}")


if __name__ == "__main__":
    main()
