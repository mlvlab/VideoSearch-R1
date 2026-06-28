import json
import os
import logging
import itertools
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("faiss is required (pip install faiss-cpu/faiss-gpu)") from exc

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


def _ensure_softqmr_on_path() -> None:
    import sys
    candidate = os.environ.get("LATENT_SOFTQMR_ROOT", "").strip()
    if candidate:
        root = candidate
    else:
        root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "latent-softquery", "latent-softqmr")
        root = os.path.abspath(root)
    if os.path.exists(root) and root not in sys.path:
        sys.path.insert(0, root)


_ensure_softqmr_on_path()

from src.softqmr.qwen_query_embed import QwenQueryEmbedder  # noqa: E402


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 1
    rank_for_ids: Optional[List[str]] = None
    rank_k: Optional[int] = None


def _load_index(index_dir: str) -> Tuple[Any, List[str]]:
    index_path = os.path.join(index_dir, "index.faiss")
    id_map_path = os.path.join(index_dir, "id_map.json")
    if not os.path.exists(index_path) or not os.path.exists(id_map_path):
        raise SystemExit(f"Missing index files in {index_dir}")
    index = faiss.read_index(index_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map


def _embed_query(embedder: QwenQueryEmbedder, text: str) -> np.ndarray:
    return embedder.embed_text(text)


app = FastAPI()
index = None
id_map = None
embedder = None
logger = logging.getLogger("retriever")
logging.basicConfig(level=logging.INFO)
_REQ_COUNTER = itertools.count(1)


def _debug_enabled() -> bool:
    flag = os.environ.get("RETRIEVER_DEBUG", "").lower()
    return flag in {"1", "true", "yes", "y"}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if index is None or id_map is None or embedder is None:
        return {"error": "retriever not initialized"}

    req_id = next(_REQ_COUNTER)
    topk = request.topk or 1
    rank_for_ids = request.rank_for_ids or None
    rank_k = request.rank_k or 0
    if rank_k < 0:
        rank_k = 0
    if rank_k > 0 and index is not None:
        try:
            rank_k = min(int(rank_k), int(index.ntotal))
        except Exception:
            rank_k = 0
    ids_out = []
    scores_out = []
    ranks_out = []
    rank_for_ids_out = []
    for q_idx, q in enumerate(request.queries):
        if _debug_enabled():
            logger.info(
                "[retriever][rid=%d][i=%d] query len=%d repr=%r",
                req_id,
                q_idx,
                len(q),
                q,
            )
        qv = _embed_query(embedder, q)
        # Ensure unit-norm for IP search.
        try:
            import faiss  # type: ignore
            faiss.normalize_L2(qv)
        except Exception:
            pass
        search_k = topk
        if rank_k > 0:
            search_k = max(search_k, rank_k)
        scores, idx = index.search(qv.astype(np.float32), search_k)
        idx = idx[0].tolist()
        scores = scores[0].tolist()
        ids = [id_map[i] for i in idx]
        ids_out.append(ids[:topk])
        scores_out.append(scores[:topk])
        if rank_k > 0:
            rank = -1
            tgt = None
            if rank_for_ids is not None and q_idx < len(rank_for_ids):
                tgt = rank_for_ids[q_idx]
                try:
                    rank = ids.index(tgt) + 1
                except ValueError:
                    rank = -1
                if rank > rank_k:
                    rank = -1
            else:
                if _debug_enabled():
                    logger.warning(
                        "[retriever][rid=%d][i=%d] rank_for_id missing (len=%s)",
                        req_id,
                        q_idx,
                        len(rank_for_ids) if rank_for_ids is not None else 0,
                    )
            ranks_out.append(rank)
            rank_for_ids_out.append(tgt)
            if _debug_enabled():
                logger.info(
                    "[retriever][rid=%d][i=%d] rank_for_id=%s rank=%s rank_k=%s",
                    req_id,
                    q_idx,
                    tgt,
                    rank,
                    rank_k,
                )
        if _debug_enabled():
            logger.info(
                "[retriever][rid=%d][i=%d] topk=%s scores=%s",
                req_id,
                q_idx,
                ids[: min(5, len(ids))],
                [round(s, 4) for s in scores[: min(5, len(scores))]],
            )
    payload = {"ids": ids_out, "scores": scores_out, "rid": req_id}
    if ranks_out:
        payload["ranks"] = ranks_out
        payload["rank_for_ids_echo"] = rank_for_ids_out
    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--embed_model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embed_device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global index, id_map, embedder
    index, id_map = _load_index(args.index_dir)
    embedder = QwenQueryEmbedder(args.embed_model, device=args.embed_device)
    if _debug_enabled():
        try:
            logger.info(
                "[retriever] index_dir=%s dim=%s ntotal=%s",
                args.index_dir,
                getattr(index, "d", None),
                getattr(index, "ntotal", None),
            )
            logger.info("[retriever] id_map_head=%s", id_map[:5])
            logger.info("[retriever] embed_model=%s device=%s", args.embed_model, args.embed_device)
        except Exception as exc:
            logger.warning("[retriever] debug log failed: %s", exc)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()