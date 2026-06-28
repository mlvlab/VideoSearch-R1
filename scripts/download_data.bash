#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_construct/download_preextracted_data.bash activitynet
  bash scripts/data_construct/download_preextracted_data.bash didemo
  bash scripts/data_construct/download_preextracted_data.bash charades
  bash scripts/data_construct/download_preextracted_data.bash all

Env:
  VIDEOSEARCH_DATA_ROOT=/path/to/data
  VIDEOSEARCH_HF_BUCKET=hf://buckets/VideoSearchR1/data
  HF_TOKEN=...  # optional for public buckets, required for private buckets
EOF
}

dataset_key() {
  case "${1:-}" in
    activitynet) echo "activitynet" ;;
    didemo) echo "didemo" ;;
    charades|charades-sta|charades_sta) echo "charades-sta" ;;
    *)
      echo "[download_data][error] unknown dataset: ${1:-}" >&2
      return 1
      ;;
  esac
}

download_from_bucket() {
  local dataset="$1"
  local key target bucket_src cache_dir marker_dir shard expected_shards
  key="$(dataset_key "${dataset}")"
  target="$(videosearch_dataset_dir "${dataset}")"
  bucket_src="${VIDEOSEARCH_HF_BUCKET%/}/datasets/${key}"
  cache_dir="${VIDEOSEARCH_CACHE_ROOT}/bucket_shards/${key}"
  marker_dir="${target}/.videosearch_shards"
  mkdir -p "${target}" "${cache_dir}" "${marker_dir}"

  if ! command -v hf >/dev/null 2>&1; then
    echo "[download_data][error] hf CLI is required for bucket downloads." >&2
    echo "[download_data][hint] install with: curl -LsSf https://hf.co/cli/install.sh | bash" >&2
    return 1
  fi

  echo "[download_data] ${bucket_src} -> ${target}"
  hf buckets cp "${bucket_src}/manifest.json" "${cache_dir}/manifest.json"

  if [[ ! -f "${cache_dir}/manifest.json" ]]; then
    echo "[download_data][error] missing bucket manifest: ${cache_dir}/manifest.json" >&2
    return 1
  fi

  local download_workers="${VIDEOSEARCH_HF_DOWNLOAD_WORKERS:-6}"
  echo "[download_data] mirror shards with workers=${download_workers}"
  python - "${cache_dir}/manifest.json" "${bucket_src}" "${cache_dir}" "${download_workers}" <<'PY'
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1])
bucket_src = sys.argv[2].rstrip("/")
cache_dir = Path(sys.argv[3])
workers = max(1, int(sys.argv[4]))
timeout_sec = int(os.environ.get("VIDEOSEARCH_HF_DOWNLOAD_TIMEOUT_SEC", "1800"))
retries = max(1, int(os.environ.get("VIDEOSEARCH_HF_DOWNLOAD_RETRIES", "3")))

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

tasks = []
for shard in manifest.get("shards", []):
    name = shard["name"]
    parts = shard.get("parts")
    if parts:
        for rel in parts:
            local = cache_dir / rel
            if local.exists() and local.stat().st_size > 0:
                continue
            tasks.append((rel, local))
    else:
        rel = f"shards/{name}"
        local = cache_dir / rel
        if local.exists() and local.stat().st_size > 0:
            continue
        tasks.append((rel, local))

if not tasks:
    print("[download_data] shard mirror already complete", flush=True)
    raise SystemExit(0)

print(f"[download_data] downloading {len(tasks)} files", flush=True)

def copy_one(index_and_item):
    index, item = index_and_item
    rel, local = item
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{bucket_src}/{rel}"
    tmp = local.with_suffix(local.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    env = os.environ.copy()
    worker_cache = cache_dir / ".hf_transfer_cache" / f"worker_{index % workers}"
    worker_cache.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(worker_cache)
    env["HF_XET_CACHE"] = str(worker_cache / "xet")
    last_error = None
    for attempt in range(1, retries + 1):
        if tmp.exists():
            tmp.unlink()
        try:
            subprocess.run(
                ["hf", "buckets", "cp", remote, str(tmp)],
                check=True,
                env=env,
                timeout=timeout_sec,
            )
            last_error = None
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[download_data][warn] retry {attempt}/{retries} after failed download: {rel}",
                flush=True,
            )
            time.sleep(min(30, 2 * attempt))
    if last_error is not None:
        raise RuntimeError(f"failed to download {rel}") from last_error
    os.replace(tmp, local)
    return rel

done = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(copy_one, (index, task)) for index, task in enumerate(tasks)]
    for future in concurrent.futures.as_completed(futures):
        rel = future.result()
        done += 1
        print(f"[download_data] downloaded {done}/{len(tasks)} {rel}", flush=True)
PY

  local extraction_plan="${cache_dir}/extract_plan.tsv"
  expected_shards="$(
    python - "${cache_dir}/manifest.json" "${cache_dir}" "${extraction_plan}" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], "r", encoding="utf-8") as f:
    manifest = json.load(f)
cache_dir = Path(sys.argv[2])
plan_path = Path(sys.argv[3])
rows = []
missing = []
for shard in manifest.get("shards", []):
    name = shard["name"]
    parts = shard.get("parts")
    if parts:
        local_parts = []
        for part in parts:
            path = cache_dir / part
            if not path.exists():
                missing.append(str(path))
            local_parts.append(str(path))
        rows.append(("PART", name, "|".join(local_parts)))
    else:
        path = cache_dir / "shards" / name
        if not path.exists():
            missing.append(str(path))
        rows.append(("TAR", name, str(path)))
if missing:
    print("[download_data][error] incomplete bucket mirror; missing:", file=sys.stderr)
    for path in missing[:20]:
        print(f"  {path}", file=sys.stderr)
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
    raise SystemExit(1)
with open(plan_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")
print(len(rows))
PY
  )"
  if [[ ! "${expected_shards}" =~ ^[0-9]+$ ]] || (( expected_shards == 0 )); then
    echo "[download_data][error] invalid bucket manifest for ${key}: expected_shards=${expected_shards}" >&2
    return 1
  fi

  local joined_dir="${cache_dir}/.joined"
  mkdir -p "${joined_dir}"
  while IFS=$'\t' read -r kind shard_name source_value; do
    local marker="${marker_dir}/${shard_name}.done"
    if [[ -f "${marker}" ]]; then
      echo "[download_data] already extracted ${shard_name}"
      continue
    fi
    if [[ "${kind}" == "PART" ]]; then
      local joined="${joined_dir}/${shard_name}"
      echo "[download_data] join ${shard_name}"
      : > "${joined}"
      IFS='|' read -r -a part_paths <<< "${source_value}"
      for part in "${part_paths[@]}"; do
        cat "${part}" >> "${joined}"
      done
      echo "[download_data] extract ${shard_name}"
      tar -xf "${joined}" -C "${target}"
      rm -f "${joined}"
    else
      echo "[download_data] extract ${shard_name}"
      tar -xf "${source_value}" -C "${target}"
    fi
    touch "${marker}"
  done < "${extraction_plan}"
}

download_one() {
  local dataset="$1"
  download_from_bucket "${dataset}"
}

dataset="${1:-}"
case "${dataset}" in
  -h|--help|"")
    usage
    exit 0
    ;;
  all)
    download_one activitynet
    download_one didemo
    download_one charades
    ;;
  activitynet|didemo|charades|charades-sta|charades_sta)
    download_one "${dataset}"
    ;;
  *)
    echo "[download_data][error] unknown dataset: ${dataset}" >&2
    usage
    exit 1
    ;;
esac
