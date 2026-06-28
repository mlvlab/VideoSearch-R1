import json
import logging
import os
from typing import Any, Dict, Iterable, Optional


logger = logging.getLogger(__name__)


def _safe_int_list(values: Any) -> Optional[list[int]]:
    if not isinstance(values, (list, tuple)):
        return None
    out: list[int] = []
    for v in values:
        try:
            iv = int(round(float(v)))
        except Exception:
            continue
        if iv < 0:
            iv = 0
        out.append(iv)
    return out if out else None


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not (x == x):  # NaN check
        return None
    return float(x)


def _safe_int(v: Any) -> Optional[int]:
    try:
        x = int(v)
    except Exception:
        return None
    return int(x)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _extract_candidate_paths(row: dict) -> list[str]:
    keys = (
        "npy_path",
        "output_npy_path",
        "output_path",
        "path",
        "video_path",
        "video",
    )
    out: list[str] = []
    for key in keys:
        val = row.get(key, None)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    return out


def _extract_meta_payload(row: dict) -> Optional[dict]:
    raw_fps = _safe_float(row.get("raw_fps", row.get("fps", None)))
    target_fps = _safe_float(
        row.get("target_sample_fps", row.get("sample_fps", row.get("output_fps", None)))
    )
    frames_indices = _safe_int_list(
        row.get(
            "sampled_indices_src",
            row.get("frames_indices", row.get("requested_indices_src", None)),
        )
    )
    output_num_frames = _safe_int(
        row.get("output_num_frames", row.get("sampled_frames_before_even_pad", None))
    )
    has_presampled_npy = bool(
        str(row.get("output_npy_path", row.get("npy_path", ""))).strip()
        or output_num_frames is not None
    )

    if has_presampled_npy:
        total_num_frames = output_num_frames
        if total_num_frames is None and frames_indices:
            total_num_frames = len(frames_indices)
        fps = target_fps or _safe_float(row.get("fps", None)) or 1.0
        payload: dict = {}
        if fps is not None and fps > 0:
            payload["fps"] = float(fps)
        if total_num_frames is not None and total_num_frames > 0:
            payload["total_num_frames"] = int(total_num_frames)
        return payload if payload else None

    total_num_frames = _safe_int(
        row.get("source_total_frames", row.get("total_num_frames", None))
    )

    if total_num_frames is None and frames_indices:
        total_num_frames = max(frames_indices) + 1

    if raw_fps is None and frames_indices is None and total_num_frames is None:
        return None

    payload: dict = {}
    if raw_fps is not None and raw_fps > 0:
        payload["fps"] = float(raw_fps)
    if frames_indices is not None:
        payload["frames_indices"] = frames_indices
    if total_num_frames is not None and total_num_frames > 0:
        payload["total_num_frames"] = int(total_num_frames)
    return payload if payload else None


def _norm_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_video_meta_index(meta_jsonl_path: Optional[str]) -> dict:
    """
    Build a lookup index from meta jsonl.

    Returns dict with:
      - by_path: absolute normalized path -> payload
      - by_basename: basename (e.g. xxx.npy) -> payload
      - by_stem: stem (e.g. xxx) -> payload
      - source: original meta jsonl path
    """
    path = str(meta_jsonl_path or "").strip()
    if not path:
        return {"by_path": {}, "by_basename": {}, "by_stem": {}, "source": ""}
    if not os.path.exists(path):
        logger.warning(f"video meta jsonl not found: {path}")
        return {"by_path": {}, "by_basename": {}, "by_stem": {}, "source": path}

    by_path: dict[str, dict] = {}
    by_basename: dict[str, dict] = {}
    by_stem: dict[str, dict] = {}

    used = 0
    for row in _iter_jsonl(path):
        payload = _extract_meta_payload(row)
        if payload is None:
            continue
        cands = _extract_candidate_paths(row)
        if not cands:
            # fallback: video id like "v_xxx"
            vid = str(row.get("video_id", row.get("video", ""))).strip()
            if vid:
                by_stem[vid] = payload
                used += 1
            continue
        for p in cands:
            try:
                ap = _norm_path(p)
            except Exception:
                continue
            by_path[ap] = payload
            base = os.path.basename(ap)
            if base:
                by_basename[base] = payload
            st = _stem(ap)
            if st:
                by_stem[st] = payload
            used += 1

    logger.info(
        f"Loaded video meta index: source={path}, entries={used}, "
        f"by_path={len(by_path)}, by_basename={len(by_basename)}, by_stem={len(by_stem)}"
    )
    return {
        "by_path": by_path,
        "by_basename": by_basename,
        "by_stem": by_stem,
        "source": path,
    }


def resolve_video_meta_for_video_path(video_path: str, meta_index: Optional[dict]) -> Optional[dict]:
    if not meta_index:
        return None
    by_path = meta_index.get("by_path", {})
    by_basename = meta_index.get("by_basename", {})
    by_stem = meta_index.get("by_stem", {})
    if not isinstance(by_path, dict) or not isinstance(by_basename, dict) or not isinstance(by_stem, dict):
        return None

    try:
        ap = _norm_path(video_path)
    except Exception:
        ap = str(video_path)

    payload = by_path.get(ap)
    if payload is not None:
        return dict(payload)

    base = os.path.basename(ap)
    payload = by_basename.get(base)
    if payload is not None:
        return dict(payload)

    st = _stem(ap)
    payload = by_stem.get(st)
    if payload is not None:
        return dict(payload)
    return None
