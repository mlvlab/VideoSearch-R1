import argparse
import json
import os
from typing import Dict, List, Sequence

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


def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _build_row2doc(docid2row: Dict[str, int], n_rows: int) -> List[str]:
    max_row = -1
    if docid2row:
        max_row = max(int(v) for v in docid2row.values())
    size = max(n_rows, max_row + 1)
    out = [""] * size
    for doc_id, row in docid2row.items():
        r = int(row)
        if 0 <= r < size:
            out[r] = str(doc_id)
    return out


def _iter_batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def _non_empty_str(row: dict, key: str) -> str:
    v = row.get(key, "")
    if v is None:
        return ""
    return str(v).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_embeddings", required=True)
    ap.add_argument("--query_meta", required=True)
    ap.add_argument("--video_embeddings", required=True)
    ap.add_argument("--video_docid2row", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--topk_pool", type=int, default=50)
    ap.add_argument("--max_negatives", type=int, default=50)
    ap.add_argument("--qid_key", type=str, default="qid")
    ap.add_argument("--pos_key", type=str, default="pos_doc_id")
    ap.add_argument("--dedup", type=_str2bool, default=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--normalize", type=_str2bool, default=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

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

    row2doc = _build_row2doc(docid2row, int(video_embeddings.shape[0]))
    mapped_rows = sum(1 for x in row2doc[: video_embeddings.shape[0]] if x)
    if mapped_rows < int(video_embeddings.shape[0]):
        print(
            "[08_hard_neg][warn] unmapped embedding rows:",
            int(video_embeddings.shape[0]) - mapped_rows,
        )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    topk_pool = max(1, int(args.topk_pool))
    max_negatives = max(0, int(args.max_negatives))
    topk_eff = min(topk_pool, int(video_embeddings.shape[0]))

    q_emb = torch.from_numpy(query_embeddings)
    v_emb_t = torch.from_numpy(video_embeddings).to(device=device, dtype=torch.float32).t().contiguous()

    out: Dict[str, List[str]] = {}
    skip_missing_qid = 0
    skip_duplicate_qid = 0

    bs = max(1, int(args.batch_size))
    for s, e in _iter_batches(len(query_meta_rows), bs):
        q_batch = q_emb[s:e].to(device=device, dtype=torch.float32, non_blocking=True)
        scores = torch.matmul(q_batch, v_emb_t)
        top_idx = torch.topk(scores, k=topk_eff, dim=1, largest=True, sorted=True).indices.cpu().numpy()

        for i, idx_list in enumerate(top_idx):
            row = query_meta_rows[s + i]
            qid = _non_empty_str(row, args.qid_key)
            if not qid:
                skip_missing_qid += 1
                continue
            if qid in out:
                skip_duplicate_qid += 1
                continue

            pos_doc_id = _non_empty_str(row, args.pos_key)
            negs: List[str] = []
            seen = set()
            for rid in idx_list.tolist():
                if rid < 0 or rid >= len(row2doc):
                    continue
                doc_id = row2doc[rid]
                if not doc_id:
                    continue
                if pos_doc_id and doc_id == pos_doc_id:
                    continue
                if bool(args.dedup):
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                negs.append(doc_id)
                if max_negatives > 0 and len(negs) >= max_negatives:
                    break
            out[qid] = negs

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lens = [len(v) for v in out.values()]
    min_len = min(lens) if lens else 0
    max_len = max(lens) if lens else 0
    mean_len = (sum(lens) / len(lens)) if lens else 0.0
    print(f"[08_hard_neg] device={device}")
    print(f"[08_hard_neg] query_rows={len(query_meta_rows)} output_qids={len(out)}")
    print(
        f"[08_hard_neg] per_qid_neg_count: min={min_len} max={max_len} mean={mean_len:.4f} "
        f"(topk_pool={topk_pool}, max_negatives={max_negatives})"
    )
    print(
        f"[08_hard_neg] skipped: missing_qid={skip_missing_qid} duplicate_qid={skip_duplicate_qid}"
    )
    print(f"[08_hard_neg] out={args.out_path}")


if __name__ == "__main__":
    main()
