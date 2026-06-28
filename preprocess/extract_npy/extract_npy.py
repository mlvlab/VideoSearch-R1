#!/usr/bin/env python3

import json
import math
import os
import time
import multiprocessing as mp
from datetime import datetime, timezone

import numpy as np

try:
    import av  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyAV is required (pip install av)") from exc
try:
    from PIL import Image  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required (pip install pillow)") from exc

dataset_jsonl = os.environ.get("DATASET_JSONL")
output_dir = os.environ.get("OUTPUT_DIR")
video_fps = float(os.environ.get("VIDEO_FPS", "2.0"))
video_maxlen = int(os.environ.get("VIDEO_MAXLEN", "20"))
video_max_pixels = int(os.environ.get("VIDEO_MAX_PIXELS", "200704"))
video_min_pixels = int(os.environ.get("VIDEO_MIN_PIXELS", "256"))
video_source_root = os.environ.get("VIDEO_SOURCE_ROOT", "").strip()
video_id_key = os.environ.get("VIDEO_ID_KEY", "video").strip() or "video"
video_ext_priority = [
    ext.strip().lower().lstrip(".")
    for ext in os.environ.get("VIDEO_EXT_PRIORITY", "mp4,mkv,webm,avi,mov").split(",")
    if ext.strip()
]
save_meta = str(os.environ.get("SAVE_META", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
meta_path = os.environ.get("META_PATH", "").strip()
failed_path = os.environ.get("FAILED_PATH", "").strip()
overwrite = str(os.environ.get("OVERWRITE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
write_meta_for_skipped = str(os.environ.get("WRITE_META_FOR_SKIPPED", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
max_unresolved_log = int(os.environ.get("MAX_UNRESOLVED_LOG", "20"))
limit = int(os.environ.get("LIMIT", "0"))
log_every = int(os.environ.get("LOG_EVERY", "200"))
num_workers = int(os.environ.get("NUM_WORKERS", "1"))

if video_max_pixels <= 0:
    video_max_pixels = None
if video_min_pixels <= 0:
    video_min_pixels = None

if not dataset_jsonl or not os.path.isfile(dataset_jsonl):
    raise SystemExit(f"Missing DATASET_JSONL: {dataset_jsonl}")
if not output_dir:
    raise SystemExit("Missing OUTPUT_DIR")

if not meta_path:
    meta_path = os.path.join(output_dir, "meta.jsonl")
if not failed_path:
    failed_path = os.path.join(output_dir, "failed_videos.jsonl")


def _build_video_id_map(root: str, ext_priority: list[str]) -> dict[str, str]:
    if not root:
        return {}
    if not os.path.isdir(root):
        raise SystemExit(f"VIDEO_SOURCE_ROOT does not exist: {root}")
    rank = {ext: i for i, ext in enumerate(ext_priority)}
    out: dict[str, tuple[int, str]] = {}
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue
        base, ext = os.path.splitext(name)
        ext = ext.lower().lstrip(".")
        pri = rank.get(ext, len(rank))
        prev = out.get(base)
        if prev is None or pri < prev[0]:
            out[base] = (pri, full)
    return {k: v[1] for k, v in out.items()}


video_id_to_path = _build_video_id_map(video_source_root, video_ext_priority)


def _resolve_video_candidate(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if not raw:
        return None

    if os.path.isabs(raw) and os.path.exists(raw):
        return os.path.normpath(raw)
    if os.path.exists(raw):
        return os.path.normpath(raw)

    if not video_source_root:
        return None

    joined = os.path.join(video_source_root, raw)
    if os.path.exists(joined):
        return os.path.normpath(joined)

    base, ext = os.path.splitext(raw)
    if ext:
        return None

    resolved = video_id_to_path.get(raw)
    if resolved:
        return os.path.normpath(resolved)

    for e in video_ext_priority:
        cand = os.path.join(video_source_root, f"{raw}.{e}")
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return None


def _get_sample_indices(video_stream, fps: float, maxlen: int) -> np.ndarray:
    total_frames = video_stream.frames
    if total_frames == 0:
        return np.linspace(0, maxlen - 1, maxlen).astype(np.int32)
    sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * fps))
    sample_frames = min(total_frames, maxlen, sample_frames)
    return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)


def _resize_frame(arr: np.ndarray, max_pixels: int | None, min_pixels: int | None) -> np.ndarray:
    height, width = arr.shape[:2]
    new_width, new_height = width, height
    if max_pixels is not None and (height * width) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (height * width))
        new_width = max(1, int(width * resize_factor))
        new_height = max(1, int(height * resize_factor))
        width, height = new_width, new_height
    if min_pixels is not None and (height * width) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (height * width))
        new_width = max(1, int(width * resize_factor))
        new_height = max(1, int(height * resize_factor))
    if new_width == arr.shape[1] and new_height == arr.shape[0]:
        return arr
    img = Image.fromarray(arr)
    img = img.resize((new_width, new_height), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _extract_frames(video_path: str) -> tuple[np.ndarray, dict]:
    container = av.open(video_path, "r")
    try:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        raw_fps = float(video_stream.average_rate) if video_stream.average_rate else float(video_fps)
        if not np.isfinite(raw_fps) or raw_fps <= 0:
            raw_fps = float(video_fps)
        if not np.isfinite(raw_fps) or raw_fps <= 0:
            raw_fps = 30.0

        total_frames_src = int(video_stream.frames) if int(video_stream.frames) > 0 else 0
        requested_indices = _get_sample_indices(video_stream, video_fps, video_maxlen).astype(np.int32)
        requested_indices_list = requested_indices.tolist()
        needed = set(int(x) for x in requested_indices_list)

        container.seek(0)
        frames_by_idx: dict[int, np.ndarray] = {}
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in needed:
                arr = frame.to_ndarray(format="rgb24")
                arr = _resize_frame(arr, video_max_pixels, video_min_pixels)
                frames_by_idx[int(frame_idx)] = arr
                if len(frames_by_idx) >= len(needed):
                    break

        if not frames_by_idx:
            raise RuntimeError(f"No frames extracted: {video_path}")

        available = sorted(frames_by_idx.keys())
        sampled_indices_src: list[int] = []
        frames: list[np.ndarray] = []
        for idx in requested_indices_list:
            pick = int(idx)
            if pick not in frames_by_idx:
                pick = min(available, key=lambda x: abs(x - int(idx)))
            sampled_indices_src.append(int(pick))
            frames.append(frames_by_idx[pick])

        sampled_before_pad = len(frames)
        even_pad_applied = False
        if len(frames) % 2 != 0:
            frames.append(frames[-1].copy())
            sampled_indices_src.append(int(sampled_indices_src[-1]))
            even_pad_applied = True

        stacked = np.stack(frames, axis=0).astype(np.uint8)
        if total_frames_src <= 0:
            total_frames_src = int(max(sampled_indices_src) + 1)

        if video_stream.duration is not None and video_stream.time_base is not None:
            duration_sec = float(video_stream.duration * video_stream.time_base)
        else:
            duration_sec = float(total_frames_src) / float(max(raw_fps, 1e-6))
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            duration_sec = float(total_frames_src) / float(max(raw_fps, 1e-6))

        h = int(stacked.shape[1]) if stacked.ndim == 4 else 0
        w = int(stacked.shape[2]) if stacked.ndim == 4 else 0
        metadata = {
            "source_video_path": os.path.normpath(video_path),
            "source_total_frames": int(total_frames_src),
            "source_duration_sec": float(duration_sec),
            "raw_fps": float(raw_fps),
            "target_sample_fps": float(video_fps),
            "target_max_frames": int(video_maxlen),
            "requested_indices_src": [int(x) for x in requested_indices_list],
            "sampled_indices_src": [int(x) for x in sampled_indices_src],
            "sampled_timestamps_sec": [float(x) / float(raw_fps) for x in sampled_indices_src],
            "sampled_frames_before_even_pad": int(sampled_before_pad),
            "even_pad_applied": bool(even_pad_applied),
            "output_num_frames": int(stacked.shape[0]),
            "output_height": int(h),
            "output_width": int(w),
            "output_dtype": str(stacked.dtype),
            "extract_utc": datetime.now(timezone.utc).isoformat(),
        }
        return stacked, metadata
    finally:
        container.close()


def _iter_videos(path: str):
    unresolved = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            videos = row.get("videos")
            if isinstance(videos, list) and videos:
                resolved = _resolve_video_candidate(str(videos[0]))
                if resolved:
                    yield resolved
                else:
                    unresolved += 1
                    if unresolved <= max_unresolved_log:
                        print(f"[extract][warn] unresolved videos[0]: {videos[0]}")
                continue
            if isinstance(videos, str) and videos:
                resolved = _resolve_video_candidate(videos)
                if resolved:
                    yield resolved
                else:
                    unresolved += 1
                    if unresolved <= max_unresolved_log:
                        print(f"[extract][warn] unresolved videos: {videos}")
                continue
            video_path = row.get("video_path") or row.get("pos_video_path") or row.get("neg_video_path")
            if video_path:
                resolved = _resolve_video_candidate(str(video_path))
                if resolved:
                    yield resolved
                else:
                    unresolved += 1
                    if unresolved <= max_unresolved_log:
                        print(f"[extract][warn] unresolved video_path: {video_path}")
                continue

            video_id = row.get(video_id_key, "")
            if video_id:
                resolved = _resolve_video_candidate(str(video_id))
                if resolved:
                    yield resolved
                else:
                    unresolved += 1
                    if unresolved <= max_unresolved_log:
                        print(f"[extract][warn] unresolved {video_id_key}: {video_id}")

    if unresolved > 0:
        print(f"[extract] unresolved rows={unresolved} (see warnings above)")


seen = set()
video_list = []
for vp in _iter_videos(dataset_jsonl):
    if vp in seen:
        continue
    seen.add(vp)
    video_list.append(vp)

if limit > 0:
    video_list = video_list[:limit]

start = time.time()
total = len(video_list)
print(
    f"[extract] videos={total} fps={video_fps} maxlen={video_maxlen} "
    f"max_pixels={video_max_pixels} min_pixels={video_min_pixels} out={output_dir} "
    f"video_source_root={video_source_root or 'none'} video_id_key={video_id_key}"
)

os.makedirs(output_dir, exist_ok=True)
if save_meta:
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(failed_path) or ".", exist_ok=True)


def _process_one(video_path: str):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{base}.npy")
    if os.path.exists(out_path) and not overwrite:
        return ("skip", video_path, "", None)
    try:
        frames, meta = _extract_frames(video_path)
        np.save(out_path, frames)
        meta["video_id"] = base
        meta["output_npy_path"] = os.path.normpath(out_path)
        return ("ok", video_path, "", meta)
    except Exception as exc:
        return ("fail", video_path, str(exc), None)


ok_count = 0
skip_count = 0
fail_count = 0

if num_workers <= 1:
    meta_f = open(meta_path, "a", encoding="utf-8", buffering=1) if save_meta else None
    fail_f = open(failed_path, "a", encoding="utf-8", buffering=1) if save_meta else None
    try:
        for idx, video_path in enumerate(video_list, 1):
            status, _, err, meta = _process_one(video_path)
            if status == "ok":
                ok_count += 1
                if meta_f is not None and meta is not None:
                    meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            elif status == "skip":
                skip_count += 1
                if meta_f is not None and write_meta_for_skipped:
                    meta_f.write(
                        json.dumps(
                            {
                                "video_id": os.path.splitext(os.path.basename(video_path))[0],
                                "source_video_path": os.path.normpath(video_path),
                                "output_npy_path": os.path.normpath(
                                    os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy")
                                ),
                                "status": "skip_exists",
                                "extract_utc": datetime.now(timezone.utc).isoformat(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                fail_count += 1
                print(f"[extract] failed {video_path}: {err}")
                if fail_f is not None:
                    fail_f.write(
                        json.dumps(
                            {
                                "source_video_path": os.path.normpath(video_path),
                                "error": str(err),
                                "extract_utc": datetime.now(timezone.utc).isoformat(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if log_every > 0 and idx % log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[extract] {idx}/{total} done (elapsed {elapsed:.1f}s) "
                    f"ok={ok_count} skip={skip_count} fail={fail_count}"
                )
    finally:
        if meta_f is not None:
            meta_f.close()
        if fail_f is not None:
            fail_f.close()
else:
    meta_f = open(meta_path, "a", encoding="utf-8", buffering=1) if save_meta else None
    fail_f = open(failed_path, "a", encoding="utf-8", buffering=1) if save_meta else None
    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(processes=num_workers) as pool:
            for idx, (status, video_path, err, meta) in enumerate(
                pool.imap_unordered(_process_one, video_list, chunksize=4), 1
            ):
                if status == "ok":
                    ok_count += 1
                    if meta_f is not None and meta is not None:
                        meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                elif status == "skip":
                    skip_count += 1
                    if meta_f is not None and write_meta_for_skipped:
                        vid = os.path.splitext(os.path.basename(video_path))[0]
                        meta_f.write(
                            json.dumps(
                                {
                                    "video_id": vid,
                                    "source_video_path": os.path.normpath(video_path),
                                    "output_npy_path": os.path.normpath(os.path.join(output_dir, f"{vid}.npy")),
                                    "status": "skip_exists",
                                    "extract_utc": datetime.now(timezone.utc).isoformat(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    fail_count += 1
                    print(f"[extract] failed {video_path}: {err}")
                    if fail_f is not None:
                        fail_f.write(
                            json.dumps(
                                {
                                    "source_video_path": os.path.normpath(video_path),
                                    "error": str(err),
                                    "extract_utc": datetime.now(timezone.utc).isoformat(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                if log_every > 0 and idx % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"[extract] {idx}/{total} done (elapsed {elapsed:.1f}s) "
                        f"ok={ok_count} skip={skip_count} fail={fail_count}"
                    )
    finally:
        if meta_f is not None:
            meta_f.close()
        if fail_f is not None:
            fail_f.close()

elapsed = time.time() - start
print(f"[extract] completed {total} videos in {elapsed:.1f}s (ok={ok_count}, skip={skip_count}, fail={fail_count})")
if save_meta:
    print(f"[extract] meta={meta_path}")
    print(f"[extract] failed={failed_path}")
