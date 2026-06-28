import argparse
import json
import os
from typing import List

import faiss
import numpy as np

from extract_embed.softqmr.qwen_query_embed import QwenQueryEmbedder


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--embed_device", default=None)
    ap.add_argument("--query_embeddings", default=None)
    ap.add_argument("--query_meta", default=None)
    ap.add_argument("--topk_list", default="1,5,10,100")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    index_path = os.path.join(args.index_dir, "index.faiss")
    id_map_path = os.path.join(args.index_dir, "id_map.json")

    index = faiss.read_index(index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    topk_list = [int(x) for x in args.topk_list.split(",") if x.strip()]
    max_k = max(topk_list)

    rows = load_jsonl(args.queries)
    precomputed = None
    if args.query_embeddings and args.query_meta:
        if os.path.exists(args.query_embeddings) and os.path.exists(args.query_meta):
            precomputed = np.load(args.query_embeddings)
            rows = load_jsonl(args.query_meta)
            if precomputed.shape[0] != len(rows):
                print("[warn] query embeddings/meta size mismatch, falling back to on-the-fly.")
                precomputed = None

    embedder = None if precomputed is not None else QwenQueryEmbedder(
        args.embed_model, device=args.embed_device
    )

    hit_counts = {k: 0 for k in topk_list}
    total = 0
    for i, ex in enumerate(rows):
        q = ex.get("query", "")
        pos_id = ex.get("pos_doc_id")
        if not q or not pos_id:
            continue
        if precomputed is not None:
            qv = precomputed[i : i + 1]
        else:
            qv = embedder.embed_text(q)
        scores, idx = index.search(qv.astype(np.float32), max_k)
        ids = [id_map[j] for j in idx[0].tolist()]
        total += 1
        for k in topk_list:
            if pos_id in ids[:k]:
                hit_counts[k] += 1

    metrics = {f"R@{k}": (hit_counts[k] / max(total, 1)) for k in topk_list}
    metrics["total"] = total

    print("Queries:", total)
    for k in topk_list:
        print(f"R@{k}: {metrics[f'R@{k}']:.4f}")

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
