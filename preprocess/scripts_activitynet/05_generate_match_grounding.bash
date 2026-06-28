#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir activitynet)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PREPROCESS_ROOT}"

POOL_JSONL="${POOL_JSONL:-${DATASET_ROOT}/sft_data/top1_pool.train.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${DATASET_ROOT}/sft_data/top1_reasoning_grounding.train.jsonl}"
JOBS_JSONL="${JOBS_JSONL:-}"  # optional: jsonl lines with {"pool_jsonl": "...", "output_jsonl": "..."}

VLLM_URL="${VLLM_URL:-http://localhost:8099/v1/chat/completions}"
MODEL_NAME="${MODEL_NAME:-${VIDEOSEARCH_REASONING_MODEL}}"
API_KEY="${API_KEY:-${VLLM_API_KEY:-}}"
BACKEND="${BACKEND:-local_vllm}"                  # openai_api | local_vllm

USE_VIDEO="${USE_VIDEO:-1}"
VIDEO_INPUT_TYPE="${VIDEO_INPUT_TYPE:-video_url}"  # video_url | video
VIDEO_URL_PREFIX="${VIDEO_URL_PREFIX:-file://}"
PREFER_NPY="${PREFER_NPY:-1}"                      # 1 => prefer ${VIDEO_NPY_ROOT}/${top1_video_id}.npy
VIDEO_NPY_ROOT="${VIDEO_NPY_ROOT:-${DATASET_ROOT}/train/video_npy_with_meta}"
VIDEO_NPY_EXT="${VIDEO_NPY_EXT:-.npy}"
VIDEO_META_JSONL="${VIDEO_META_JSONL:-${VIDEO_NPY_ROOT}/meta.jsonl}"  # carries raw_fps/indices for temporal grounding
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-0}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-200704}"      # 256 * 28 * 28
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-0}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-2}"
RETRY_SLEEP="${RETRY_SLEEP:-1.0}"
WORKERS="${WORKERS:-4}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
LOCAL_IMAGE_PATCH_SIZE="${LOCAL_IMAGE_PATCH_SIZE:-16}"  # 14 * spatial_merge(2) = 28
LOCAL_DTYPE="${LOCAL_DTYPE:-bfloat16}"
LOCAL_TENSOR_PARALLEL_SIZE="${LOCAL_TENSOR_PARALLEL_SIZE:-1}"
LOCAL_GPU_MEMORY_UTILIZATION="${LOCAL_GPU_MEMORY_UTILIZATION:-0.9}"
LOCAL_MAX_MODEL_LEN="${LOCAL_MAX_MODEL_LEN:-32768}"
LOCAL_TRUST_REMOTE_CODE="${LOCAL_TRUST_REMOTE_CODE:-1}"
VISION_PROCESS_PATH="${VISION_PROCESS_PATH:-${VIDEOSEARCH_REPO_ROOT}/videosearch_r1/model/qwen_vl_utils/vision_process.py}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
RESUME="${RESUME:-1}"
INCLUDE_RAW_RESPONSE="${INCLUDE_RAW_RESPONSE:-0}"
LIMIT="${LIMIT:-0}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-42}"
SHARD_GPUS="${SHARD_GPUS:-"0,1"}"          # e.g. "0,1,2,3" => one local_vllm process per GPU
SHARD_TMP_DIR="${SHARD_TMP_DIR:-}"    # optional path for shard pool/output/log files
SHARD_KEEP_TMP="${SHARD_KEEP_TMP:-1}" # 1 keeps shard files for resume/debug
SHARD_SHOW_TQDM="${SHARD_SHOW_TQDM:-1}" # 1 streams shard logs to stdout (tqdm visible)

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${JOBS_JSONL}" && ! -f "${POOL_JSONL}" ]]; then
  echo "Missing POOL_JSONL: ${POOL_JSONL}" >&2
  exit 1
fi
if [[ -n "${JOBS_JSONL}" && ! -f "${JOBS_JSONL}" ]]; then
  echo "Missing JOBS_JSONL: ${JOBS_JSONL}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

run_generate() {
  local pool_jsonl="$1"
  local output_jsonl="$2"
  local jobs_jsonl="$3"
  local cuda_visible_devices="${4:-}"
  local local_tp="${5:-${LOCAL_TENSOR_PARALLEL_SIZE}}"

  local cmd=(
    "${PYTHON_BIN}" -m dataset_generation.generate_match_grounding
    --pool_jsonl "${pool_jsonl}"
    --output_jsonl "${output_jsonl}"
    --jobs_jsonl "${jobs_jsonl}"
    --vllm_url "${VLLM_URL}"
    --model "${MODEL_NAME}"
    --api_key "${API_KEY}"
    --backend "${BACKEND}"
    --use_video "${USE_VIDEO}"
    --video_input_type "${VIDEO_INPUT_TYPE}"
    --video_url_prefix "${VIDEO_URL_PREFIX}"
    --prefer_npy "${PREFER_NPY}"
    --video_npy_root "${VIDEO_NPY_ROOT}"
    --video_npy_ext "${VIDEO_NPY_EXT}"
    --video_meta_jsonl "${VIDEO_META_JSONL}"
    --video_max_frames "${VIDEO_MAX_FRAMES}"
    --video_fps "${VIDEO_FPS}"
    --video_min_pixels "${VIDEO_MIN_PIXELS}"
    --video_max_pixels "${VIDEO_MAX_PIXELS}"
    --video_total_pixels "${VIDEO_TOTAL_PIXELS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_tokens "${MAX_TOKENS}"
    --timeout "${TIMEOUT}"
    --retries "${RETRIES}"
    --retry_sleep "${RETRY_SLEEP}"
    --workers "${WORKERS}"
    --local_batch_size "${LOCAL_BATCH_SIZE}"
    --local_image_patch_size "${LOCAL_IMAGE_PATCH_SIZE}"
    --local_dtype "${LOCAL_DTYPE}"
    --local_tensor_parallel_size "${local_tp}"
    --local_gpu_memory_utilization "${LOCAL_GPU_MEMORY_UTILIZATION}"
    --local_max_model_len "${LOCAL_MAX_MODEL_LEN}"
    --local_trust_remote_code "${LOCAL_TRUST_REMOTE_CODE}"
    --vision_process_path "${VISION_PROCESS_PATH}"
    --include_raw_response "${INCLUDE_RAW_RESPONSE}"
    --resume "${RESUME}"
    --limit "${LIMIT}"
    --shuffle "${SHUFFLE}"
    --seed "${SEED}"
  )

  if [[ -n "${cuda_visible_devices}" ]]; then
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD}" \
    PYTHONPATH=. "${cmd[@]}"
  else
    VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD}" \
    PYTHONPATH=. "${cmd[@]}"
  fi
}

if [[ -z "${SHARD_GPUS}" ]]; then
  run_generate "${POOL_JSONL}" "${OUTPUT_JSONL}" "${JOBS_JSONL}"
  echo "[dataset_generation][activitynet] generated: ${OUTPUT_JSONL}"
  exit 0
fi

if [[ "${BACKEND}" != "local_vllm" ]]; then
  echo "[dataset_generation][activitynet][error] SHARD_GPUS mode requires BACKEND=local_vllm." >&2
  exit 1
fi
if [[ -n "${JOBS_JSONL}" ]]; then
  echo "[dataset_generation][activitynet][error] SHARD_GPUS mode does not support JOBS_JSONL yet." >&2
  exit 1
fi

SHARD_GPU_ARR=()
IFS=',' read -r -a _raw_gpus <<< "${SHARD_GPUS}"
for _gpu in "${_raw_gpus[@]}"; do
  _gpu="${_gpu//[[:space:]]/}"
  if [[ -n "${_gpu}" ]]; then
    SHARD_GPU_ARR+=("${_gpu}")
  fi
done
NUM_SHARDS="${#SHARD_GPU_ARR[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "[dataset_generation][activitynet][error] SHARD_GPUS is set but no valid GPU id was parsed." >&2
  exit 1
fi

if [[ "${LOCAL_TENSOR_PARALLEL_SIZE}" != "1" ]]; then
  echo "[dataset_generation][activitynet][warn] SHARD_GPUS mode forces LOCAL_TENSOR_PARALLEL_SIZE=1 (current=${LOCAL_TENSOR_PARALLEL_SIZE})."
fi

if [[ -z "${SHARD_TMP_DIR}" ]]; then
  out_base="$(basename "${OUTPUT_JSONL}")"
  out_stem="${out_base%.*}"
  SHARD_TMP_DIR="$(dirname "${OUTPUT_JSONL}")/.shard_runs_${out_stem}"
fi
mkdir -p "${SHARD_TMP_DIR}"
SHARD_PENDING_POOL="${SHARD_TMP_DIR}/pool.pending.jsonl"

"${PYTHON_BIN}" - "${POOL_JSONL}" "${OUTPUT_JSONL}" "${SHARD_TMP_DIR}" "${NUM_SHARDS}" "${RESUME}" "${SHUFFLE}" "${SEED}" "${LIMIT}" "${SHARD_PENDING_POOL}" <<'PY'
import json
import os
import random
import sys

src = sys.argv[1]
output_path = sys.argv[2]
out_dir = sys.argv[3]
n = int(sys.argv[4])
resume = int(sys.argv[5]) == 1
shuffle = int(sys.argv[6]) == 1
seed = int(sys.argv[7])
limit = int(sys.argv[8])
pending_path = sys.argv[9]


def row_pair_id(row, fallback_idx):
    pair_id = str(row.get("pair_id", "")).strip()
    if pair_id:
        return pair_id
    qid = str(row.get("qid", "")).strip() or f"idx_{fallback_idx}"
    top1 = str(row.get("top1_doc_id", "")).strip()
    if top1:
        return f"{qid}::{top1}"
    return qid


rows = []
with open(src, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise SystemExit(f"invalid json at {src}:{line_idx}: {exc}") from exc
        if not row.get("pair_id"):
            row["pair_id"] = row_pair_id(row, len(rows))
        rows.append(row)

seen = set()
if resume and os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for out_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out = json.loads(line)
            except Exception:
                continue
            pid = row_pair_id(out, out_idx)
            if pid:
                seen.add(pid)

remaining = [row for idx, row in enumerate(rows) if row_pair_id(row, idx) not in seen]
if shuffle:
    random.Random(seed).shuffle(remaining)
if limit > 0:
    remaining = remaining[:limit]

with open(pending_path, "w", encoding="utf-8") as wf:
    for row in remaining:
        wf.write(json.dumps(row, ensure_ascii=False) + "\n")

paths = [os.path.join(out_dir, f"pool.shard{i}.jsonl") for i in range(n)]
fps = [open(p, "w", encoding="utf-8") for p in paths]
counts = [0] * n
for row_idx, row in enumerate(remaining):
    shard_id = row_idx % n
    fps[shard_id].write(json.dumps(row, ensure_ascii=False) + "\n")
    counts[shard_id] += 1
for fp in fps:
    fp.close()

print(
    f"[shard] split source={src} source_rows={len(rows)} "
    f"resume_seen={len(seen)} remaining={len(remaining)} shards={n}"
)
print(f"[shard] pending_pool={pending_path}")
for i, p in enumerate(paths):
    print(f"[shard] shard={i} rows={counts[i]} pool={p}")
PY

if [[ ! -s "${SHARD_PENDING_POOL}" ]]; then
  echo "[shard] nothing to generate after resume/limit/shuffle filtering."
  echo "[dataset_generation][activitynet] generated: ${OUTPUT_JSONL}"
  exit 0
fi

pids=()
for shard_id in "${!SHARD_GPU_ARR[@]}"; do
  gpu="${SHARD_GPU_ARR[$shard_id]}"
  shard_pool="${SHARD_TMP_DIR}/pool.shard${shard_id}.jsonl"
  shard_out="${SHARD_TMP_DIR}/out.shard${shard_id}.jsonl"
  shard_log="${SHARD_TMP_DIR}/run.shard${shard_id}.log"

  if [[ "${RESUME}" != "1" ]]; then
    rm -f "${shard_out}"
  fi

  if [[ "${SHARD_SHOW_TQDM}" == "1" ]]; then
    (
      LIMIT=0 SHUFFLE=0 run_generate "${shard_pool}" "${shard_out}" "" "${gpu}" "1" 2>&1 | tee "${shard_log}"
    ) &
  else
    (
      LIMIT=0 SHUFFLE=0 run_generate "${shard_pool}" "${shard_out}" "" "${gpu}" "1"
    ) > "${shard_log}" 2>&1 &
  fi
  pid=$!
  pids+=("${pid}")
  echo "[shard] started shard=${shard_id} gpu=${gpu} pid=${pid} out=${shard_out}"
done

failed=0
for shard_id in "${!pids[@]}"; do
  pid="${pids[$shard_id]}"
  if ! wait "${pid}"; then
    failed=1
    echo "[shard][error] shard=${shard_id} failed. inspect log: ${SHARD_TMP_DIR}/run.shard${shard_id}.log" >&2
  fi
done

if (( failed != 0 )); then
  exit 1
fi

"${PYTHON_BIN}" - "${OUTPUT_JSONL}" "${SHARD_TMP_DIR}" "${NUM_SHARDS}" "${RESUME}" <<'PY'
import json
import os
import sys

output_path = sys.argv[1]
shard_dir = sys.argv[2]
n = int(sys.argv[3])
resume = int(sys.argv[4]) == 1
tmp_path = output_path + ".tmp"

seen_pair_ids = set()
written = 0
dropped_dup = 0
with open(tmp_path, "w", encoding="utf-8") as w:
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair_id = None
                try:
                    obj = json.loads(line)
                    pid = obj.get("pair_id", None)
                    if pid is not None:
                        pair_id = str(pid).strip() or None
                except Exception:
                    pair_id = None
                if pair_id is not None:
                    if pair_id in seen_pair_ids:
                        dropped_dup += 1
                        continue
                    seen_pair_ids.add(pair_id)
                w.write(line + "\n")
                written += 1

    for shard_id in range(n):
        path = os.path.join(shard_dir, f"out.shard{shard_id}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair_id = None
                try:
                    obj = json.loads(line)
                    pid = obj.get("pair_id", None)
                    if pid is not None:
                        pair_id = str(pid).strip() or None
                except Exception:
                    pair_id = None
                if pair_id is not None:
                    if pair_id in seen_pair_ids:
                        dropped_dup += 1
                        continue
                    seen_pair_ids.add(pair_id)
                w.write(line + "\n")
                written += 1

os.replace(tmp_path, output_path)
print(
    f"[shard] merged shards={n} rows={written} "
    f"unique_pair_id={len(seen_pair_ids)} dropped_duplicate_pair_id={dropped_dup} "
    f"output={output_path}"
)
PY

if [[ "${SHARD_KEEP_TMP}" != "1" ]]; then
  rm -rf "${SHARD_TMP_DIR}"
fi

echo "[dataset_generation][activitynet] generated: ${OUTPUT_JSONL}"
