import argparse
import json
import os
import sys
import time
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from extract_embed.softqmr.data import load_segments
from extract_embed.softqmr.qwen_query_embed import QwenQueryEmbedder
from extract_embed.softqmr.video_io import sample_segment_frames


def _estimate_tokens_per_frame(frame_size: int) -> float:
    # Empirical approximation for Qwen3-VL video token growth.
    side = float(frame_size if frame_size and frame_size > 0 else 336)
    return max(1.0, 0.85 * (side / 28.0) ** 2)


def _segment_seconds(seg) -> float:
    start = float(getattr(seg, "start", 0.0) or 0.0)
    end = float(getattr(seg, "end", 0.0) or 0.0)
    if end > start:
        return max(end - start, 1e-3)
    duration = float(getattr(seg, "duration", 0.0) or 0.0)
    return max(duration, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    data_root = os.environ.get("DATA_ROOT", "data/activitynet")
    ap.add_argument(
        "--corpus",
        default=os.path.join(data_root, "train", "corpus_segments.jsonl"),
    )
    out_root = os.environ.get("OUTPUT_ROOT", "data/activitynet")
    ap.add_argument("--out_dir", default=os.path.join(out_root, "train", "video_embedding_1fps"))
    ap.add_argument("--train_queries", default=os.path.join(data_root, "train", "train_queries.jsonl"))
    ap.add_argument(
        "--query_files",
        default=",".join(
            [
                os.path.join(data_root, "train", "train_queries.jsonl"),
                os.path.join(data_root, "test", "test_queries.jsonl"),
            ]
        ),
    )
    ap.add_argument("--only_train", action="store_true")
    ap.add_argument("--only_queries", action="store_true")
    ap.add_argument("--embed_model", default=os.environ.get("QWEN_VL_EMBED_ID", "Qwen/Qwen3-VL-Embedding-2B"))
    ap.add_argument("--embed_device", default=os.environ.get("QWEN_VL_EMBED_DEVICE", None))
    ap.add_argument("--instruction", default="Represent the user's input")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--sample_fps", type=float, default=0.0)
    ap.add_argument("--sample_max_frames", type=int, default=0)
    ap.add_argument("--frame_size", type=int, default=0)
    ap.add_argument("--max_input_tokens", type=int, default=16384)
    ap.add_argument("--input_token_reserve", type=int, default=512)
    ap.add_argument("--auto_sample_by_token_budget", dest="auto_sample_by_token_budget", action="store_true")
    ap.add_argument("--no_auto_sample_by_token_budget", dest="auto_sample_by_token_budget", action="store_false")
    ap.add_argument("--video_backend", default="opencv")
    ap.add_argument("--segment_batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--tqdm", dest="use_tqdm", action="store_true")
    ap.add_argument("--no_tqdm", dest="use_tqdm", action="store_false")
    ap.add_argument("--reuse_folder", default=None)
    ap.add_argument("--meta_append_path", default="")
    ap.add_argument("--fail_append_path", default="")
    ap.set_defaults(use_tqdm=True, auto_sample_by_token_budget=True)
    args = ap.parse_args()
    if args.sample_fps < 0:
        raise ValueError("--sample_fps must be >= 0")
    if args.segment_batch_size < 1:
        raise ValueError("--segment_batch_size must be >= 1")
    if args.input_token_reserve < 0:
        raise ValueError("--input_token_reserve must be >= 0")
    valid_backends = {"opencv", "decord", "torchvision", "vision", "tv", "auto"}
    if str(args.video_backend).lower() not in valid_backends:
        raise ValueError(f"--video_backend must be one of: {sorted(valid_backends)}")

    os.makedirs(args.out_dir, exist_ok=True)
    embed_path = os.path.join(args.out_dir, "segment_embeds.npy")
    map_path = os.path.join(args.out_dir, "docid2row.json")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    fail_path = os.path.join(args.out_dir, "failed_docs.jsonl")
    config_path = os.path.join(args.out_dir, "config.json")

    cfg = {
        "embed_model": args.embed_model,
        "embed_device": args.embed_device,
        "instruction": args.instruction,
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "sample_max_frames": args.sample_max_frames,
        "frame_size": args.frame_size,
        "max_input_tokens": args.max_input_tokens,
        "input_token_reserve": args.input_token_reserve,
        "auto_sample_by_token_budget": args.auto_sample_by_token_budget,
        "video_backend": args.video_backend,
        "segment_batch_size": args.segment_batch_size,
        "limit": args.limit,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "save_every": args.save_every,
        "only_train": args.only_train,
        "only_queries": args.only_queries,
        "reuse_folder": args.reuse_folder,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    segments = load_segments(args.corpus)
    if args.only_train or args.only_queries:
        keep = set()
        if args.only_train:
            if not os.path.exists(args.train_queries):
                raise FileNotFoundError(args.train_queries)
            with open(args.train_queries, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    keep.add(j["pos_doc_id"])
        if args.only_queries:
            files = [p for p in args.query_files.split(",") if p]
            for path in files:
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        j = json.loads(line)
                        keep.add(j["pos_doc_id"])
        before = len(segments)
        segments = [s for s in segments if s.doc_id in keep]
        print(f"[embed] query filter: {before} -> {len(segments)} (keep={len(keep)})")
    shard = [s for i, s in enumerate(segments) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        shard = shard[: args.limit]

    existing_vecs = None
    docid2row = {}
    if args.resume and args.reuse_folder is None and os.path.exists(map_path) and os.path.exists(embed_path):
        with open(map_path, "r", encoding="utf-8") as f:
            docid2row = json.load(f)
        existing_vecs = np.load(embed_path)

    embedder = QwenQueryEmbedder(
        args.embed_model,
        device=args.embed_device,
        default_instruction=args.instruction,
        video_num_frames=None,
        video_max_frames=None,
    )

    reuse_vecs = None
    reuse_docid2row = None
    if args.reuse_folder:
        reuse_embed_path = os.path.join(args.reuse_folder, "segment_embeds.npy")
        reuse_map_path = os.path.join(args.reuse_folder, "docid2row.json")
        if os.path.exists(reuse_embed_path) and os.path.exists(reuse_map_path):
            reuse_vecs = np.load(reuse_embed_path, mmap_mode="r")
            with open(reuse_map_path, "r", encoding="utf-8") as f:
                reuse_docid2row = json.load(f)
        else:
            print("[embed] reuse_folder missing files:", args.reuse_folder)

    new_vecs = []
    new_docs = []
    last_saved_new = 0
    adjusted_sampling_count = 0
    start_time = time.time()
    use_tqdm = bool(args.use_tqdm and tqdm is not None)
    if args.use_tqdm and tqdm is None:
        print("[embed] tqdm is not available; falling back to plain logs")
    if args.auto_sample_by_token_budget:
        print(
            f"[embed] token_budget={args.max_input_tokens} "
            f"reserve={args.input_token_reserve} frame_size={args.frame_size or 'orig'}"
        )

    meta_out_path = args.meta_append_path.strip() if args.meta_append_path else meta_path
    fail_out_path = args.fail_append_path.strip() if args.fail_append_path else fail_path
    meta_mode = "a" if args.meta_append_path else ("w" if args.reuse_folder else "a")
    fail_mode = "a" if args.fail_append_path else ("w" if args.reuse_folder else "a")

    # Determine output embedding dim for empty-checkpoint case.
    embed_dim = 0
    if existing_vecs is not None and int(existing_vecs.ndim) == 2:
        embed_dim = int(existing_vecs.shape[1])
    else:
        model_cfg = getattr(embedder.model, "config", None)
        embed_dim = int(getattr(model_cfg, "hidden_size", 0) or 0)

    def build_outputs():
        if existing_vecs is None:
            if new_vecs:
                vecs = np.stack(new_vecs, axis=0).astype(np.float32)
                current_map = {doc_id: i for i, doc_id in enumerate(new_docs)}
            else:
                vecs = np.zeros((0, embed_dim), dtype=np.float32)
                current_map = {}
            return vecs, current_map

        current_map = dict(docid2row)
        if new_vecs:
            vecs = np.vstack([existing_vecs, np.stack(new_vecs, axis=0).astype(np.float32)])
            offset = len(docid2row)
            for i, doc_id in enumerate(new_docs):
                current_map[doc_id] = offset + i
        else:
            vecs = np.asarray(existing_vecs, dtype=np.float32)
        return vecs, current_map

    def write_outputs(vecs: np.ndarray, current_map: dict):
        tmp_embed_path = embed_path + ".tmp.npy"
        tmp_map_path = map_path + ".tmp.json"
        np.save(tmp_embed_path, vecs)
        with open(tmp_map_path, "w", encoding="utf-8") as f:
            json.dump(current_map, f, ensure_ascii=False, indent=2)
        os.replace(tmp_embed_path, embed_path)
        os.replace(tmp_map_path, map_path)

    def maybe_checkpoint():
        nonlocal last_saved_new
        if args.save_every <= 0:
            return
        if (len(new_vecs) - last_saved_new) < args.save_every:
            return
        vecs_ckpt, map_ckpt = build_outputs()
        write_outputs(vecs_ckpt, map_ckpt)
        last_saved_new = len(new_vecs)
        print(f"[embed] checkpoint save rows={int(vecs_ckpt.shape[0])}")

    with open(fail_out_path, fail_mode, encoding="utf-8", buffering=1) as fail_f, open(
        meta_out_path, meta_mode, encoding="utf-8", buffering=1
    ) as meta_f:
        pending = []
        progress = (
            tqdm(
                total=len(shard),
                desc=f"embed shard {args.shard_id}/{args.num_shards}",
                unit="seg",
                file=sys.stdout,
                dynamic_ncols=True,
            )
            if use_tqdm
            else None
        )

        def write_meta(seg, info):
            meta_f.write(
                json.dumps(
                    {
                        "doc_id": seg.doc_id,
                        "video_id": seg.video_id,
                        "video_path": seg.video_path,
                        "start": seg.start,
                        "end": seg.end,
                        "duration": seg.duration,
                        "info": info,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            meta_f.flush()

        def flush_pending():
            nonlocal pending
            if not pending:
                return
            batch_frames = [item["frames"] for item in pending]
            try:
                embs = embedder.embed_video_frames_batch(batch_frames, instruction=args.instruction)
                if int(embs.shape[0]) != len(pending):
                    raise RuntimeError(
                        f"batch embedding size mismatch: got={int(embs.shape[0])} expected={len(pending)}"
                    )
                for i, item in enumerate(pending):
                    new_docs.append(item["seg"].doc_id)
                    new_vecs.append(embs[i].astype(np.float32))
                    write_meta(item["seg"], item["info"])
            except Exception as batch_err:
                fail_f.write(
                    json.dumps(
                        {
                            "doc_id": "__batch__",
                            "error": f"batch_failed: {str(batch_err)}",
                            "count": len(pending),
                        }
                    )
                    + "\n"
                )
                fail_f.flush()
                for item in pending:
                    try:
                        emb = embedder.embed_video_frames(item["frames"], instruction=args.instruction)
                        new_docs.append(item["seg"].doc_id)
                        new_vecs.append(emb[0].astype(np.float32))
                        write_meta(item["seg"], item["info"])
                    except Exception as e:
                        fail_f.write(json.dumps({"doc_id": item["seg"].doc_id, "error": str(e)}) + "\n")
                        fail_f.flush()
            pending = []

        try:
            for idx, seg in enumerate(shard):
                try:
                    if seg.doc_id in docid2row:
                        continue
                    reused = False
                    info = {}
                    if reuse_vecs is not None and reuse_docid2row is not None:
                        reuse_idx = reuse_docid2row.get(seg.doc_id)
                        if reuse_idx is not None:
                            emb = reuse_vecs[reuse_idx]
                            new_docs.append(seg.doc_id)
                            new_vecs.append(emb.astype(np.float32))
                            maybe_checkpoint()
                            reused = True
                            info = {"reused_from": args.reuse_folder}
                            write_meta(seg, info)
                    if not reused:
                        num_frames = int(args.num_frames)
                        sample_fps = float(args.sample_fps)
                        sample_max_frames = args.sample_max_frames if args.sample_max_frames > 0 else None
                        sampling_ctl = {}
                        if args.auto_sample_by_token_budget and args.max_input_tokens > 0:
                            token_budget = max(1, int(args.max_input_tokens) - int(args.input_token_reserve))
                            tpf = _estimate_tokens_per_frame(args.frame_size)
                            max_frames_budget = max(1, int(token_budget // max(tpf, 1e-6)))
                            if sample_max_frames is None:
                                sample_max_frames = max_frames_budget
                            else:
                                sample_max_frames = min(sample_max_frames, max_frames_budget)

                            sec = _segment_seconds(seg)
                            if sample_fps > 0:
                                # _build_indices uses arange(start, end+eps, 1/fps), so +1 is a close estimate.
                                estimated_frames = int(np.floor(sec * sample_fps)) + 1
                                if estimated_frames > sample_max_frames:
                                    sample_fps = max((float(sample_max_frames) - 1.0) / max(sec, 1e-6), 1e-3)
                                    adjusted_sampling_count += 1
                            else:
                                if num_frames > sample_max_frames:
                                    num_frames = int(sample_max_frames)
                                    adjusted_sampling_count += 1
                            sampling_ctl = {
                                "token_budget": int(token_budget),
                                "tokens_per_frame_est": float(round(tpf, 4)),
                                "max_frames_budget": int(max_frames_budget),
                                "effective_sample_fps": float(sample_fps),
                                "effective_max_frames": int(sample_max_frames),
                                "effective_num_frames": int(num_frames),
                            }

                        frames, info = sample_segment_frames(
                            seg.video_path,
                            seg.start,
                            seg.end,
                            num_frames=num_frames,
                            frame_size=args.frame_size or None,
                            sample_fps=sample_fps,
                            max_frames=sample_max_frames,
                            backend=args.video_backend,
                        )
                        if sampling_ctl:
                            info = dict(info)
                            info["sampling_control"] = sampling_ctl
                        pending.append({"seg": seg, "frames": frames, "info": info})
                        if len(pending) >= args.segment_batch_size:
                            flush_pending()
                            maybe_checkpoint()

                    if (idx + 1) % args.log_every == 0:
                        elapsed = time.time() - start_time
                        rate = (idx + 1) / max(elapsed, 1e-6)
                        sample_mode = (
                            f"{args.sample_fps:g}fps"
                            if args.sample_fps > 0
                            else f"{args.num_frames}frames"
                        )
                        msg = (
                            f"[embed] {idx+1}/{len(shard)} rate={rate:.2f}/s "
                            f"sample={sample_mode} size={args.frame_size or 'orig'} "
                            f"backend={args.video_backend} "
                            f"batch={args.segment_batch_size} "
                            f"auto_adjusted={adjusted_sampling_count}"
                        )
                        if progress is not None:
                            progress.write(msg)
                        else:
                            print(msg)
                except Exception as e:
                    fail_f.write(json.dumps({"doc_id": seg.doc_id, "error": str(e)}) + "\n")
                    fail_f.flush()
                finally:
                    if progress is not None:
                        progress.update(1)

            flush_pending()
            maybe_checkpoint()
        finally:
            if progress is not None:
                progress.close()
    if not new_vecs and existing_vecs is not None:
        print("No new segments. Keeping existing embeddings.")
        return

    vecs, docid2row_out = build_outputs()
    write_outputs(vecs, docid2row_out)

    print("Saved:", embed_path)
    print("Saved:", map_path)
    print("Total embeddings:", vecs.shape[0])


if __name__ == "__main__":
    main()
