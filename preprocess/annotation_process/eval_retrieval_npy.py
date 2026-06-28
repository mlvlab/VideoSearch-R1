import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _first_non_empty(row: dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k, "")
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _parse_topk_list(s: str) -> List[int]:
    ks = sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})
    ks = [k for k in ks if k > 0]
    if not ks:
        raise ValueError("topk_list must contain at least one positive integer")
    return ks


def _consume_bucket(
    key,
    buckets: Dict,
    cursors: Dict,
    used: np.ndarray,
) -> Optional[int]:
    arr = buckets.get(key)
    if not arr:
        return None
    i = int(cursors.get(key, 0))
    n = len(arr)
    while i < n and used[arr[i]]:
        i += 1
    if i >= n:
        cursors[key] = n
        return None
    idx = int(arr[i])
    used[idx] = True
    cursors[key] = i + 1
    return idx


def _resolve_eval_indices(
    query_meta_rows: List[dict],
    queries_rows: Optional[List[dict]],
    query_fields: Sequence[str],
    pos_fields: Sequence[str],
) -> Tuple[List[int], List[str], Dict[str, int]]:
    query_texts = [str(r.get("query", "")).strip() for r in query_meta_rows]
    pos_docs = [str(r.get("pos_doc_id", "")).strip() for r in query_meta_rows]

    pair_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    query_buckets: Dict[str, List[int]] = defaultdict(list)
    for i, (q, p) in enumerate(zip(query_texts, pos_docs)):
        pair_buckets[(q, p)].append(i)
        query_buckets[q].append(i)

    pair_cursors: Dict[Tuple[str, str], int] = {}
    query_cursors: Dict[str, int] = {}
    used = np.zeros(len(query_meta_rows), dtype=bool)

    indices: List[int] = []
    target_pos_docs: List[str] = []
    skip = {
        "query_empty": 0,
        "query_pair_not_found_in_meta": 0,
        "query_not_found_in_meta": 0,
    }

    if queries_rows is None:
        for i, p in enumerate(pos_docs):
            if query_texts[i] and p:
                indices.append(i)
                target_pos_docs.append(p)
        return indices, target_pos_docs, skip

    for row in queries_rows:
        q = _first_non_empty(row, query_fields)
        if not q:
            skip["query_empty"] += 1
            continue

        p = _first_non_empty(row, pos_fields)
        if p:
            idx = _consume_bucket((q, p), pair_buckets, pair_cursors, used)
            if idx is None:
                skip["query_pair_not_found_in_meta"] += 1
                continue
            indices.append(idx)
            target_pos_docs.append(p)
            continue

        idx = _consume_bucket(q, query_buckets, query_cursors, used)
        if idx is None:
            skip["query_not_found_in_meta"] += 1
            continue
        indices.append(idx)
        target_pos_docs.append(pos_docs[idx])

    return indices, target_pos_docs, skip


def _iter_batches(n: int, bs: int) -> Iterable[Tuple[int, int]]:
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_embeddings", required=True)
    ap.add_argument("--query_meta", required=True)
    ap.add_argument("--video_embeddings", required=True)
    ap.add_argument("--video_docid2row", required=True)
    ap.add_argument("--queries", default="")
    ap.add_argument("--query_fields", default="query,fig_desc")
    ap.add_argument("--pos_fields", default="pos_doc_id,video,query_pos_doc_id")
    ap.add_argument("--topk_list", default="1,5,10,100")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--normalize", type=_str2bool, default=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_path", default="")
    args = ap.parse_args()

    ks = _parse_topk_list(args.topk_list)
    max_k = max(ks)

    query_embeddings = np.asarray(np.load(args.query_embeddings), dtype=np.float32)
    video_embeddings = np.asarray(np.load(args.video_embeddings), dtype=np.float32)
    if bool(args.normalize):
        query_embeddings = _l2_norm_rows(query_embeddings)
        video_embeddings = _l2_norm_rows(video_embeddings)

    query_meta_rows = _load_jsonl(args.query_meta)
    if query_embeddings.shape[0] != len(query_meta_rows):
        raise RuntimeError(
            f"query_embeddings rows ({query_embeddings.shape[0]}) != query_meta rows ({len(query_meta_rows)})"
        )

    with open(args.video_docid2row, "r", encoding="utf-8") as f:
        docid2row = {str(k): int(v) for k, v in json.load(f).items()}

    queries_rows: Optional[List[dict]] = None
    if str(args.queries).strip():
        if not os.path.exists(args.queries):
            raise FileNotFoundError(f"--queries not found: {args.queries}")
        queries_rows = _load_jsonl(args.queries)

    query_fields = [x.strip() for x in str(args.query_fields).split(",") if x.strip()]
    pos_fields = [x.strip() for x in str(args.pos_fields).split(",") if x.strip()]

    eval_q_indices, eval_pos_docs, skip = _resolve_eval_indices(
        query_meta_rows=query_meta_rows,
        queries_rows=queries_rows,
        query_fields=query_fields,
        pos_fields=pos_fields,
    )

    eval_pos_rows: List[int] = []
    filtered_q_indices: List[int] = []
    for q_idx, pos_doc in zip(eval_q_indices, eval_pos_docs):
        r = docid2row.get(str(pos_doc))
        if r is None:
            skip["pos_doc_missing_in_docid2row"] = int(skip.get("pos_doc_missing_in_docid2row", 0)) + 1
            continue
        filtered_q_indices.append(int(q_idx))
        eval_pos_rows.append(int(r))

    if not filtered_q_indices:
        raise RuntimeError("No valid evaluation rows after mapping/filtering.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    q_emb = torch.from_numpy(query_embeddings)
    v_emb_t = torch.from_numpy(video_embeddings).to(device=device, dtype=torch.float32).t().contiguous()

    max_k_eff = min(int(max_k), int(video_embeddings.shape[0]))
    ks_eff = [min(int(k), int(video_embeddings.shape[0])) for k in ks]
    hit_counts = {k: 0 for k in ks}

    total = len(filtered_q_indices)
    bs = max(1, int(args.batch_size))
    for s, e in _iter_batches(total, bs):
        q_batch = q_emb[filtered_q_indices[s:e]].to(device=device, dtype=torch.float32, non_blocking=True)
        pos_batch = torch.tensor(eval_pos_rows[s:e], device=device, dtype=torch.long)
        scores = torch.matmul(q_batch, v_emb_t)
        top_idx = torch.topk(scores, k=max_k_eff, dim=1, largest=True, sorted=True).indices
        for k_eff, k_req in zip(ks_eff, ks):
            hits = (top_idx[:, :k_eff] == pos_batch.unsqueeze(1)).any(dim=1).sum().item()
            hit_counts[k_req] += int(hits)

    metrics = {f"R@{k}": (float(hit_counts[k]) / float(max(total, 1))) for k in ks}
    result = {
        "query_embeddings": args.query_embeddings,
        "query_meta": args.query_meta,
        "video_embeddings": args.video_embeddings,
        "video_docid2row": args.video_docid2row,
        "queries": args.queries if str(args.queries).strip() else "",
        "topk_list": ks,
        "normalize": bool(args.normalize),
        "device": device,
        "total_rows_requested": int(len(queries_rows) if queries_rows is not None else len(query_meta_rows)),
        "total_rows_evaluated": int(total),
        "skip": {k: int(v) for k, v in skip.items()},
        "metrics": metrics,
    }

    print(f"rows_evaluated={total}")
    for k in ks:
        print(f"R@{k}: {metrics[f'R@{k}']:.6f}")
    print("skip:", json.dumps(result["skip"], ensure_ascii=False))

    if args.out_path:
        out_dir = os.path.dirname(args.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
