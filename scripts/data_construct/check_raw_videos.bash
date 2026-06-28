#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_construct/check_raw_videos.bash activitynet
  bash scripts/data_construct/check_raw_videos.bash didemo
  bash scripts/data_construct/check_raw_videos.bash charades

Checks that each video id used by the VERIFIED annotations has a matching
raw video file under VIDEOSEARCH_DATA_ROOT/raw_videos.
EOF
}

dataset="${1:-}"
case "${dataset}" in
  -h|--help|"")
    usage
    exit 0
    ;;
  activitynet)
    anno_dir="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/activitynet-fig"
    video_dir="${VIDEOSEARCH_DATA_ROOT}/raw_videos/activitynet/videos"
    files=("${anno_dir}/activitynet_fig_train.jsonl" "${anno_dir}/activitynet_fig_val_1.jsonl" "${anno_dir}/activitynet_fig_val_2.jsonl")
    ;;
  didemo)
    anno_dir="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/didemo-fig"
    video_dir="${VIDEOSEARCH_DATA_ROOT}/raw_videos/didemo/videos"
    files=("${anno_dir}/didemo_fig_train.jsonl" "${anno_dir}/didemo_fig_val.jsonl" "${anno_dir}/didemo_fig_test.jsonl")
    ;;
  charades|charades-sta|charades_sta)
    anno_dir="${VIDEOSEARCH_DATA_ROOT}/fine-grained-anno/charades-fig"
    video_dir="${VIDEOSEARCH_DATA_ROOT}/raw_videos/charades_sta/videos"
    files=("${anno_dir}/charades_fig_train.jsonl" "${anno_dir}/charades_fig_test.jsonl")
    ;;
  *)
    echo "[check_raw_videos][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac

for file in "${files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "[check_raw_videos][error] missing annotation: ${file}" >&2
    exit 1
  fi
done
if [[ ! -d "${video_dir}" ]]; then
  echo "[check_raw_videos][error] missing video directory: ${video_dir}" >&2
  exit 1
fi

python3 - "${video_dir}" "${files[@]}" <<'PY'
import json
import os
import sys

video_dir = sys.argv[1]
files = sys.argv[2:]
exts = [".mp4", ".mkv", ".webm", ".avi"]
seen = []
seen_set = set()

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_id = str(row.get("video", "")).strip()
            if video_id and video_id not in seen_set:
                seen_set.add(video_id)
                seen.append(video_id)

missing = []
for video_id in seen:
    root, ext = os.path.splitext(video_id)
    candidates = []
    if ext:
        candidates.append(os.path.join(video_dir, video_id))
    candidates.extend(os.path.join(video_dir, video_id + suffix) for suffix in exts)
    if not any(os.path.exists(path) and os.path.getsize(path) > 0 for path in candidates):
        missing.append(video_id)

print(f"[check_raw_videos] video_dir={video_dir}")
print(f"[check_raw_videos] required={len(seen)} found={len(seen) - len(missing)} missing={len(missing)}")
if missing:
    for video_id in missing[:20]:
        print(f"[check_raw_videos][missing] {video_id}")
    if len(missing) > 20:
        print(f"[check_raw_videos][missing] ... {len(missing) - 20} more")
    sys.exit(1)
PY
