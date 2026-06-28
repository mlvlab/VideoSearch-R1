import os
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from torchvision.io import read_video  # type: ignore
except Exception:
    read_video = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from decord import VideoReader, cpu  # type: ignore
except Exception:
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore


def _resolve_bounds(total: int, fps: float, start: float, end: float) -> Tuple[int, int]:
    if total <= 0:
        raise RuntimeError("video has zero frames")

    if end is None or end <= 0 or end <= start:
        start_idx = 0
        end_idx = total - 1
    else:
        start_idx = int(float(start) * fps)
        end_idx = int(float(end) * fps)
        start_idx = max(0, min(start_idx, total - 1))
        end_idx = max(0, min(end_idx, total - 1))

    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, total - 1)
    return start_idx, end_idx


def _build_indices(
    total: int,
    fps: float,
    start: float,
    end: float,
    num_frames: int,
    sample_fps: float,
    max_frames: int | None,
) -> Tuple[int, int, np.ndarray]:
    start_idx, end_idx = _resolve_bounds(total, fps, start, end)

    if sample_fps and sample_fps > 0:
        start_sec = max(0.0, float(start or 0.0))
        if end is None or end <= 0 or end <= start_sec:
            end_sec = float(end_idx) / max(fps, 1e-6)
        else:
            end_sec = max(start_sec, float(end))
        if end_sec <= start_sec:
            end_sec = start_sec + (1.0 / max(fps, 1e-6))

        step = 1.0 / float(sample_fps)
        ts = np.arange(start_sec, end_sec + 1e-6, step, dtype=np.float64)
        if ts.size == 0:
            ts = np.array([start_sec], dtype=np.float64)
        idxs = np.clip(np.round(ts * fps).astype(int), 0, total - 1)
        idxs = np.unique(idxs)
        if max_frames is not None and max_frames > 0 and idxs.size > max_frames:
            idxs = np.linspace(int(idxs[0]), int(idxs[-1]), int(max_frames), dtype=int)
    else:
        req_frames = max(1, int(num_frames))
        idxs = np.linspace(start_idx, end_idx, req_frames, dtype=int)

    return start_idx, end_idx, idxs


def _sample_with_opencv(
    video_path: str,
    start: float,
    end: float,
    num_frames: int,
    frame_size: int | None,
    sample_fps: float,
    max_frames: int | None,
) -> Tuple[List[Image.Image], Dict[str, float]]:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"opencv cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise RuntimeError(f"opencv failed to read frame count: {video_path}")

        start_idx, end_idx, idxs = _build_indices(
            total=total,
            fps=fps,
            start=start,
            end=end,
            num_frames=num_frames,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )

        images: List[Image.Image] = []
        for idx in idxs.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            if frame_size and frame_size > 0:
                img = img.resize((frame_size, frame_size), Image.BICUBIC)
            images.append(img)

        if not images:
            raise RuntimeError(f"opencv returned zero decoded frames: {video_path}")
        while len(images) < len(idxs):
            images.append(images[-1].copy())

        info = {
            "backend": "opencv",
            "fps": fps,
            "num_total": total,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "idxs": idxs.tolist(),
            "frame_size": frame_size or 0,
            "sample_fps": float(sample_fps or 0.0),
            "max_frames": int(max_frames) if (max_frames is not None and max_frames > 0) else 0,
            "sampled_frames": int(len(idxs)),
        }
        return images, info
    finally:
        cap.release()


def _sample_with_torchvision(
    video_path: str,
    start: float,
    end: float,
    num_frames: int,
    frame_size: int | None,
    sample_fps: float,
    max_frames: int | None,
) -> Tuple[List[Image.Image], Dict[str, float]]:
    if read_video is None:
        raise RuntimeError("torchvision video IO is not available")

    start_sec = max(0.0, float(start or 0.0))
    end_sec = float(end) if end is not None else 0.0
    if end_sec > start_sec:
        frames_t, _, info = read_video(
            video_path, start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
        )
        segment_start = start_sec
        segment_end = end_sec
    else:
        frames_t, _, info = read_video(video_path, pts_unit="sec")
        segment_start = 0.0
        segment_end = 0.0

    if frames_t is None or int(frames_t.numel()) == 0:
        raise RuntimeError(f"torchvision returned zero decoded frames: {video_path}")

    fps = float(info.get("video_fps", 0.0) or 0.0)
    if fps <= 0:
        fps = 30.0
    total = int(frames_t.shape[0])
    if total <= 0:
        raise RuntimeError(f"torchvision failed to decode frames: {video_path}")

    if sample_fps and sample_fps > 0:
        if segment_end > segment_start:
            duration = segment_end - segment_start
        else:
            duration = float(total - 1) / max(fps, 1e-6)
        step = 1.0 / float(sample_fps)
        ts = np.arange(0.0, duration + 1e-6, step, dtype=np.float64)
        if ts.size == 0:
            ts = np.array([0.0], dtype=np.float64)
        idxs = np.clip(np.round(ts * fps).astype(int), 0, total - 1)
        idxs = np.unique(idxs)
        if max_frames is not None and max_frames > 0 and idxs.size > max_frames:
            idxs = np.linspace(int(idxs[0]), int(idxs[-1]), int(max_frames), dtype=int)
    else:
        req_frames = max(1, int(num_frames))
        idxs = np.linspace(0, total - 1, req_frames, dtype=int)

    sel = frames_t[idxs.tolist()]
    if hasattr(sel, "detach"):
        sel = sel.detach()
    if hasattr(sel, "cpu"):
        sel = sel.cpu()
    frames_np = sel.numpy()
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames_np]
    if frame_size and frame_size > 0:
        images = [img.resize((frame_size, frame_size), Image.BICUBIC) for img in images]

    info_out = {
        "backend": "torchvision",
        "fps": fps,
        "num_total": total,
        "start_idx": int(idxs[0]) if len(idxs) else 0,
        "end_idx": int(idxs[-1]) if len(idxs) else max(0, total - 1),
        "idxs": idxs.tolist(),
        "frame_size": frame_size or 0,
        "sample_fps": float(sample_fps or 0.0),
        "max_frames": int(max_frames) if (max_frames is not None and max_frames > 0) else 0,
        "sampled_frames": int(len(idxs)),
        "segment_start_sec": float(segment_start),
        "segment_end_sec": float(segment_end),
    }
    return images, info_out


def _sample_with_decord(
    video_path: str,
    start: float,
    end: float,
    num_frames: int,
    frame_size: int | None,
    sample_fps: float,
    max_frames: int | None,
) -> Tuple[List[Image.Image], Dict[str, float]]:
    if VideoReader is None or cpu is None:
        raise RuntimeError("decord is not installed")

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    fps = float(vr.get_avg_fps())
    if fps <= 0:
        fps = 30.0

    start_idx, end_idx, idxs = _build_indices(
        total=total,
        fps=fps,
        start=start,
        end=end,
        num_frames=num_frames,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    frames = vr.get_batch(idxs).asnumpy()
    images = [Image.fromarray(frame) for frame in frames]
    if frame_size and frame_size > 0:
        images = [img.resize((frame_size, frame_size), Image.BICUBIC) for img in images]

    info = {
        "backend": "decord",
        "fps": fps,
        "num_total": total,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "idxs": idxs.tolist(),
        "frame_size": frame_size or 0,
        "sample_fps": float(sample_fps or 0.0),
        "max_frames": int(max_frames) if (max_frames is not None and max_frames > 0) else 0,
        "sampled_frames": int(len(idxs)),
    }
    return images, info


def sample_segment_frames(
    video_path: str,
    start: float,
    end: float,
    num_frames: int = 8,
    frame_size: int | None = None,
    sample_fps: float = 0.0,
    max_frames: int | None = None,
    backend: str | None = None,
) -> Tuple[List[Image.Image], Dict[str, float]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    mode = (backend or os.environ.get("VIDEO_IO_BACKEND", "opencv")).strip().lower()
    if mode in ("cv2",):
        mode = "opencv"
    if mode in ("vision", "tv"):
        mode = "torchvision"
    if mode == "torchcodec":
        # For now map torchcodec requests to torchvision video decoder path.
        mode = "torchvision"

    if mode == "torchvision":
        return _sample_with_torchvision(
            video_path, start, end, num_frames, frame_size, sample_fps, max_frames
        )

    if mode == "opencv":
        return _sample_with_opencv(video_path, start, end, num_frames, frame_size, sample_fps, max_frames)
    if mode == "decord":
        return _sample_with_decord(video_path, start, end, num_frames, frame_size, sample_fps, max_frames)
    if mode == "auto":
        try:
            return _sample_with_torchvision(
                video_path, start, end, num_frames, frame_size, sample_fps, max_frames
            )
        except Exception:
            pass
        try:
            return _sample_with_opencv(
                video_path, start, end, num_frames, frame_size, sample_fps, max_frames
            )
        except Exception:
            return _sample_with_decord(
                video_path, start, end, num_frames, frame_size, sample_fps, max_frames
            )

    raise ValueError(f"Unknown backend: {mode}. Use torchvision | opencv | decord | auto")
