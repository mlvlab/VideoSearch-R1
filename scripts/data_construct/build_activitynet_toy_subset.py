#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_video_index(sources):
    index = {}
    for source in sources:
        if not source:
            continue
        root = Path(source).expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            index.setdefault(path.stem, path)
    return index


def youtube_id(activitynet_id):
    if activitynet_id.startswith("v_"):
        return activitynet_id[2:]
    return activitynet_id


def maybe_download_video(video_id, out_dir):
    if shutil.which("yt-dlp") is None:
        return None
    out_tpl = str(out_dir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={youtube_id(video_id)}"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tpl,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return None
    for ext in VIDEO_EXTS:
        candidate = out_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def materialize_video(src, dst_dir, mode, overwrite):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}{src.suffix.lower()}"
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return dst
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        dst.symlink_to(src.resolve())
    return dst


def valid_activitynet_row(row):
    video_id = str(row.get("video", "")).strip()
    fig_desc = str(row.get("fig_desc", "")).strip()
    time_span = row.get("time")
    try:
        duration = float(row.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    if not video_id or not fig_desc:
        return False
    if not isinstance(time_span, (list, tuple)) or len(time_span) < 2:
        return False
    try:
        start, end = float(time_span[0]), float(time_span[1])
    except (TypeError, ValueError):
        return False
    return duration > 0 and end > start >= 0


def select_rows(path, split_name, target_videos, rows_per_video, video_index, raw_video_dir, args):
    selected = []
    seen_counts = {}
    manifest = []
    for row in load_jsonl(path):
        if not valid_activitynet_row(row):
            continue
        video_id = str(row["video"]).strip()
        if seen_counts.get(video_id, 0) >= rows_per_video:
            continue
        src = video_index.get(video_id)
        if src is None and args.download_missing:
            src = maybe_download_video(video_id, raw_video_dir)
            if src is not None:
                video_index[video_id] = src
        if src is None:
            continue
        dst = materialize_video(src, raw_video_dir, args.link_mode, args.overwrite)
        row = dict(row)
        selected.append(row)
        seen_counts[video_id] = seen_counts.get(video_id, 0) + 1
        manifest.append(
            {
                "split": split_name,
                "video": video_id,
                "source": str(src),
                "local": str(dst),
                "desc_id": row.get("desc_id"),
            }
        )
        unique_videos = len(seen_counts)
        if unique_videos >= target_videos and all(v >= rows_per_video for v in seen_counts.values()):
            break
    if len(seen_counts) < target_videos and not args.allow_smaller:
        raise SystemExit(
            f"[toy_activitynet][error] only found {len(seen_counts)} videos for {split_name}; "
            f"requested {target_videos}. Add --allow-smaller, provide more --video-source paths, "
            "or set --download-missing."
        )
    return selected, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-root", required=True)
    parser.add_argument("--output-data-root", required=True)
    parser.add_argument("--video-source", action="append", default=[])
    parser.add_argument("--train-videos", type=int, default=10)
    parser.add_argument("--val-videos", type=int, default=10)
    parser.add_argument("--test-videos", type=int, default=10)
    parser.add_argument("--rows-per-video", type=int, default=1)
    parser.add_argument("--link-mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-smaller", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    verified_root = Path(args.verified_root).expanduser()
    output_root = Path(args.output_data_root).expanduser()
    anno_out = output_root / "fine-grained-anno" / "activitynet-fig"
    raw_video_dir = output_root / "raw_videos" / "activitynet" / "videos"

    split_specs = [
        ("train", verified_root / "activitynet_fig_train.jsonl", anno_out / "activitynet_fig_train.jsonl", args.train_videos),
        ("val", verified_root / "activitynet_fig_val_1.jsonl", anno_out / "activitynet_fig_val_1.jsonl", args.val_videos),
        ("test", verified_root / "activitynet_fig_val_2.jsonl", anno_out / "activitynet_fig_val_2.jsonl", args.test_videos),
    ]
    for _, src, _, _ in split_specs:
        if not src.exists():
            raise SystemExit(f"[toy_activitynet][error] missing annotation file: {src}")

    video_index = build_video_index(args.video_source)
    if not video_index and not args.download_missing:
        raise SystemExit(
            "[toy_activitynet][error] no local videos found. Pass --video-source /path/to/ActivityNet "
            "or use --download-missing with yt-dlp installed."
        )

    all_manifest = []
    for split_name, src_jsonl, dst_jsonl, target_videos in split_specs:
        rows, manifest = select_rows(
            src_jsonl,
            split_name,
            target_videos,
            max(args.rows_per_video, 1),
            video_index,
            raw_video_dir,
            args,
        )
        write_jsonl(dst_jsonl, rows)
        all_manifest.extend(manifest)
        unique = len({row["video"] for row in rows})
        print(f"[toy_activitynet] {split_name}: rows={len(rows)} videos={unique} -> {dst_jsonl}")

    manifest_path = output_root / "activitynet_toy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_sources": args.video_source,
                "link_mode": args.link_mode,
                "rows_per_video": args.rows_per_video,
                "items": all_manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[toy_activitynet] raw videos -> {raw_video_dir}")
    print(f"[toy_activitynet] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
