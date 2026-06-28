import argparse
import json
import os
import random

import faiss
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    out_root = os.environ.get("OUTPUT_ROOT", "data")
    ap.add_argument("--embeds", default=os.path.join(out_root, "structured", "segment_embeds.npy"))
    ap.add_argument("--docid2row", default=os.path.join(out_root, "structured", "docid2row.json"))
    ap.add_argument("--out_dir", default=os.path.join(out_root, "structured"))
    ap.add_argument("--debug_queries", default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vecs = np.load(args.embeds).astype(np.float32)
    faiss.normalize_L2(vecs)

    with open(args.docid2row, "r", encoding="utf-8") as f:
        docid2row = json.load(f)
    id_map = [doc_id for doc_id, _ in sorted(docid2row.items(), key=lambda kv: kv[1])]

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    index_path = os.path.join(args.out_dir, "index.faiss")
    id_map_path = os.path.join(args.out_dir, "id_map.json")

    faiss.write_index(index, index_path)
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("Saved:", index_path)
    print("Saved:", id_map_path)

    if args.debug_queries:
        from extract_embed.softqmr.qwen_query_embed import QwenQueryEmbedder

        embedder = QwenQueryEmbedder()
        with open(args.debug_queries, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        random.shuffle(lines)
        for ex in lines[:5]:
            q = ex["query"]
            qv = embedder.embed_text(q)
            scores, idx = index.search(qv.astype(np.float32), args.topk)
            ids = [id_map[i] for i in idx[0].tolist()]
            print("[debug] q=", q)
            print("  topk=", ids, "scores=", scores[0].round(3).tolist())


if __name__ == "__main__":
    main()
