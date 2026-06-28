import argparse
import json
import os
import time
from typing import List

import numpy as np

from extract_embed.softqmr.qwen_query_embed import QwenQueryEmbedder


def load_queries(path: str) -> List[dict]:
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
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--embed_device", default=None)
    ap.add_argument("--instruction", default="Represent the user's input")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--split_name", default="train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"query_meta.{args.split_name}.jsonl")
    emb_path = os.path.join(args.out_dir, f"query_embeddings.{args.split_name}.npy")

    if args.skip_existing and os.path.exists(meta_path) and os.path.exists(emb_path):
        print("Skip:", meta_path)
        print("Skip:", emb_path)
        return

    rows = load_queries(args.queries)
    if args.limit > 0:
        rows = rows[: args.limit]

    embedder = QwenQueryEmbedder(
        args.embed_model,
        device=args.embed_device,
        default_instruction=args.instruction,
    )
    vecs = []
    start_time = time.time()
    total = len(rows)
    for i in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[i : i + max(1, args.batch_size)]
        embs = embedder.embedder.process(
            [{"text": ex["query"], "instruction": args.instruction} for ex in batch],
            normalize=True,
        )
        vecs.append(embs.float().detach().cpu().numpy())
        done = min(i + max(1, args.batch_size), total)
        if done % max(1, args.log_every) == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / max(elapsed, 1e-6)
            print(f"[embed_query] {done}/{total} rate={rate:.2f}/s")

    if vecs:
        vecs = np.concatenate(vecs, axis=0)
    else:
        vecs = np.zeros((0, 2048), dtype=np.float32)

    with open(meta_path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    np.save(emb_path, vecs)
    print("Saved:", meta_path)
    print("Saved:", emb_path)
    print("Queries:", len(rows))


if __name__ == "__main__":
    main()
