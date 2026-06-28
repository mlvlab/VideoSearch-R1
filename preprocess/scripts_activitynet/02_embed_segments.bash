#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir activitynet)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DEBUG="${DEBUG:-1}"
export QWEN_VL_EMBED_ID="${QWEN_VL_EMBED_ID:-${VIDEOSEARCH_EMBED_MODEL}}"

# transformers v5 deprecates TRANSFORMERS_CACHE; map it once and silence warnings.
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  export HF_HOME="${HF_HOME:-${TRANSFORMERS_CACHE}}"
  unset TRANSFORMERS_CACHE
fi

# Decoder log noise control (h264 mmco warnings, etc.)
FFMPEG_LOGLEVEL="${FFMPEG_LOGLEVEL:-error}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS:-loglevel;${FFMPEG_LOGLEVEL}}"
SUPPRESS_FFMPEG_MMCO="${SUPPRESS_FFMPEG_MMCO:-1}"  # 1 => hide known noisy h264 mmco warnings
USE_TQDM="${USE_TQDM:-1}"                          # 1 => show per-shard tqdm progress

INPUT_STRUCTURED_ROOT="${INPUT_STRUCTURED_ROOT:-${DATASET_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
OUTPUT_STRUCTURED_ROOT="${OUTPUT_STRUCTURED_ROOT:-${DATASET_ROOT}}"
SPLITS="${SPLITS:-train,val,test}"

GPU_IDS="${GPU_IDS:-0,1,2,4}"   # comma-separated GPU ids
NUM_SHARDS="${NUM_SHARDS:-4}"    # 0 => number of GPUs
KEEP_SHARD_DIRS="${KEEP_SHARD_DIRS:-0}"
LIVE_META="${LIVE_META:-1}"      # 1 => append meta/fail directly to final out dir in real time
PRESERVE_LIVE_LOGS_ON_RESUME="${PRESERVE_LIVE_LOGS_ON_RESUME:-1}"  # 1 => keep existing meta/failed when RESUME=1

ONLY_TRAIN="${ONLY_TRAIN:-0}"
ONLY_QUERIES="${ONLY_QUERIES:-0}"
REUSE_FOLDER="${REUSE_FOLDER:-}"
RESUME="${RESUME:-0}"

SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
SAMPLE_MAX_FRAMES="${SAMPLE_MAX_FRAMES:-64}"  # 0 => no cap
NUM_FRAMES="${NUM_FRAMES:-64}"               # used only when SAMPLE_FPS<=0
FRAME_SIZE="${FRAME_SIZE:-0}"
LIMIT="${LIMIT:-0}"
SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-8}"                # default batch size per gpu process
SEGMENT_BATCH_SIZE_PER_GPU="${SEGMENT_BATCH_SIZE_PER_GPU:-}" # ex) "2,2,1,1"
VIDEO_BACKEND="${VIDEO_BACKEND:-opencv}"                     # safest default with qwen_vl_utils pipeline
SAVE_EVERY="${SAVE_EVERY:-1}"                                # default real-time checkpoint saving
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"                # 8192 * 2 token budget
INPUT_TOKEN_RESERVE="${INPUT_TOKEN_RESERVE:-512}"            # reserve for text/system tokens
AUTO_SAMPLE_BY_TOKEN_BUDGET="${AUTO_SAMPLE_BY_TOKEN_BUDGET:-1}"  # 1 => lower fps/frames when token budget is exceeded

INSTRUCTION="${INSTRUCTION:-}"
if [[ -z "${INSTRUCTION}" ]]; then
  INSTRUCTION="Represent the user's input"
fi
OUT_SUFFIX="${OUT_SUFFIX:-1fps}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

GPU_IDS="${GPU_IDS// /}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if [[ ${#GPU_ARR[@]} -eq 0 || -z "${GPU_ARR[0]}" ]]; then
  GPU_ARR=("0")
fi

if [[ "${NUM_SHARDS}" -le 0 ]]; then
  NUM_SHARDS="${#GPU_ARR[@]}"
fi
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  NUM_SHARDS=1
fi

resolve_query_jsonl() {
  local split_dir="$1"
  local split_name="$2"
  local candidates=()
  if [[ "${split_name}" == "train" ]]; then
    candidates=("train_queries.jsonl" "test_queries.jsonl" "val_queries.jsonl")
  elif [[ "${split_name}" == "val" ]]; then
    candidates=("val_queries.jsonl" "test_queries.jsonl" "train_queries.jsonl")
  else
    candidates=("test_queries.jsonl" "val_queries.jsonl" "train_queries.jsonl")
  fi
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${split_dir}/${c}" ]]; then
      echo "${split_dir}/${c}"
      return 0
    fi
  done
  return 1
}

echo "[embed_segments_1fps] in_root=${INPUT_STRUCTURED_ROOT} out_root=${OUTPUT_STRUCTURED_ROOT} splits=${SPLITS}"
echo "[embed_segments_1fps] gpus=${GPU_IDS} num_shards=${NUM_SHARDS} sample_fps=${SAMPLE_FPS}"
echo "[embed_segments_1fps] video_backend=${VIDEO_BACKEND}"
echo "[embed_segments_1fps] ffmpeg_loglevel=${FFMPEG_LOGLEVEL}"
echo "[embed_segments_1fps] suppress_mmco=${SUPPRESS_FFMPEG_MMCO}"
echo "[embed_segments_1fps] use_tqdm=${USE_TQDM}"
echo "[embed_segments_1fps] save_every=${SAVE_EVERY}"
echo "[embed_segments_1fps] max_input_tokens=${MAX_INPUT_TOKENS} reserve=${INPUT_TOKEN_RESERVE} auto_sample=${AUTO_SAMPLE_BY_TOKEN_BUDGET}"

BATCH_ARR=()
if [[ -n "${SEGMENT_BATCH_SIZE_PER_GPU// /}" ]]; then
  batch_csv="${SEGMENT_BATCH_SIZE_PER_GPU// /}"
  IFS=',' read -r -a BATCH_ARR <<< "${batch_csv}"
fi

for SPLIT in $(echo "${SPLITS}" | tr "," " "); do
  IN_SPLIT_DIR="${INPUT_STRUCTURED_ROOT}/${SPLIT}"
  OUT_SPLIT_DIR="${OUTPUT_STRUCTURED_ROOT}/${SPLIT}"

  if ! QUERY_JSON="$(resolve_query_jsonl "${IN_SPLIT_DIR}" "${SPLIT}")"; then
    echo "[embed_segments][error] no query jsonl found in ${IN_SPLIT_DIR} for split=${SPLIT}" >&2
    exit 1
  fi

  CORPUS="${IN_SPLIT_DIR}/corpus_segments.jsonl"
  OUT_DIR="${OUT_SPLIT_DIR}/video_embedding_${OUT_SUFFIX}"
  FINAL_META="${OUT_DIR}/meta.jsonl"
  FINAL_FAIL="${OUT_DIR}/failed_docs.jsonl"

  EXTRA=()
  if [[ "${RESUME}" == "1" ]]; then
    EXTRA+=("--resume")
  fi
  if [[ "${ONLY_TRAIN}" == "1" ]]; then
    EXTRA+=("--only_train" "--train_queries" "${QUERY_JSON}")
  fi
  if [[ "${ONLY_QUERIES}" == "1" ]]; then
    EXTRA+=("--only_queries" "--query_files" "${QUERY_JSON}")
  fi
  if [[ -n "${REUSE_FOLDER}" ]]; then
    EXTRA+=("--reuse_folder" "${REUSE_FOLDER}")
  fi

  echo "[split:${SPLIT}] corpus=${CORPUS}"
  echo "[split:${SPLIT}] out=${OUT_DIR}"
  LIVE_META_ARGS=()
  if [[ "${LIVE_META}" -eq 1 ]]; then
    mkdir -p "${OUT_DIR}"
    if [[ "${RESUME}" == "1" && "${PRESERVE_LIVE_LOGS_ON_RESUME}" == "1" ]]; then
      touch "${FINAL_META}" "${FINAL_FAIL}"
      echo "[split:${SPLIT}] resume=1 keep existing live logs: ${FINAL_META} ${FINAL_FAIL}"
    else
      : > "${FINAL_META}"
      : > "${FINAL_FAIL}"
    fi
    LIVE_META_ARGS+=(--meta_append_path "${FINAL_META}" --fail_append_path "${FINAL_FAIL}")
  fi

  PIDS=()
  SHARD_DIRS=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu_idx=$((shard % ${#GPU_ARR[@]}))
    gpu="${GPU_ARR[$gpu_idx]}"
    batch_size="${SEGMENT_BATCH_SIZE}"
    if [[ ${#BATCH_ARR[@]} -gt 0 ]]; then
      if [[ ${#BATCH_ARR[@]} -eq 1 && -n "${BATCH_ARR[0]}" ]]; then
        batch_size="${BATCH_ARR[0]}"
      elif [[ ${gpu_idx} -lt ${#BATCH_ARR[@]} && -n "${BATCH_ARR[$gpu_idx]}" ]]; then
        batch_size="${BATCH_ARR[$gpu_idx]}"
      fi
    fi
    shard_out="${OUT_DIR}.shard${shard}"
    SHARD_DIRS+=("${shard_out}")
    mkdir -p "${shard_out}"

    echo "[spawn] split=${SPLIT} shard=${shard}/${NUM_SHARDS} gpu=${gpu} batch=${batch_size} out=${shard_out}"
    (
      CMD=(
        "${PYTHON_BIN}" -m extract_embed.embed_segments
        --corpus "${CORPUS}"
        --out_dir "${shard_out}"
        --embed_model "${QWEN_VL_EMBED_ID}"
        --embed_device "0"
        --instruction "${INSTRUCTION}"
        --sample_fps "${SAMPLE_FPS}"
        --sample_max_frames "${SAMPLE_MAX_FRAMES}"
        --num_frames "${NUM_FRAMES}"
        --frame_size "${FRAME_SIZE}"
        --max_input_tokens "${MAX_INPUT_TOKENS}"
        --input_token_reserve "${INPUT_TOKEN_RESERVE}"
        --video_backend "${VIDEO_BACKEND}"
        --segment_batch_size "${batch_size}"
        --save_every "${SAVE_EVERY}"
        --limit "${LIMIT}"
        --shard_id "${shard}"
        --num_shards "${NUM_SHARDS}"
        "${LIVE_META_ARGS[@]}"
        "${EXTRA[@]}"
      )
      if [[ "${USE_TQDM}" == "1" ]]; then
        CMD+=(--tqdm)
      else
        CMD+=(--no_tqdm)
      fi
      if [[ "${AUTO_SAMPLE_BY_TOKEN_BUDGET}" == "1" ]]; then
        CMD+=(--auto_sample_by_token_budget)
      else
        CMD+=(--no_auto_sample_by_token_budget)
      fi
      if [[ "${SUPPRESS_FFMPEG_MMCO}" == "1" ]]; then
        PYTHONPATH=. CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" \
          2> >(grep -E -v "mmco: unref short failure|Missing reference picture, default is" >&2)
      else
        PYTHONPATH=. CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}"
      fi
    ) &
    PIDS+=("$!")
  done

  failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[split:${SPLIT}] one or more shard jobs failed" >&2
    exit 1
  fi

  echo "[split:${SPLIT}] merging shard outputs..."
  "${PYTHON_BIN}" - "${OUT_DIR}" "${LIVE_META}" "${SHARD_DIRS[@]}" <<'PY'
import json
import os
import sys
import numpy as np

out_dir = sys.argv[1]
live_meta = str(sys.argv[2]).strip() == "1"
shard_dirs = sys.argv[3:]

os.makedirs(out_dir, exist_ok=True)

embed_path = os.path.join(out_dir, "segment_embeds.npy")
map_path = os.path.join(out_dir, "docid2row.json")
meta_path = os.path.join(out_dir, "meta.jsonl")
fail_path = os.path.join(out_dir, "failed_docs.jsonl")
cfg_path = os.path.join(out_dir, "config.json")

arr_infos = []
for sd in shard_dirs:
    emb_p = os.path.join(sd, "segment_embeds.npy")
    map_p = os.path.join(sd, "docid2row.json")
    if not (os.path.exists(emb_p) and os.path.exists(map_p)):
        raise FileNotFoundError(f"missing shard outputs: {sd}")
    arr = np.load(emb_p, mmap_mode="r")
    if arr.ndim != 2:
        raise RuntimeError(f"invalid embedding array shape in {emb_p}: {arr.shape}")
    arr_infos.append((sd, arr))

if not arr_infos:
    raise RuntimeError("no shard outputs found")

emb_dim = int(arr_infos[0][1].shape[1])
total_rows = int(sum(arr.shape[0] for _, arr in arr_infos))
for sd, arr in arr_infos:
    if int(arr.shape[1]) != emb_dim:
        raise RuntimeError(f"embedding dim mismatch in {sd}: {arr.shape[1]} != {emb_dim}")

merged = np.lib.format.open_memmap(embed_path, mode="w+", dtype=np.float32, shape=(total_rows, emb_dim))

docid2row = {}
offset = 0
meta_count = 0
fail_count = 0
meta_f = None
fail_f = None
if not live_meta:
    meta_f = open(meta_path, "w", encoding="utf-8")
    fail_f = open(fail_path, "w", encoding="utf-8")

try:
    for sd, arr in arr_infos:
        rows = int(arr.shape[0])
        merged[offset : offset + rows] = arr

        map_p = os.path.join(sd, "docid2row.json")
        with open(map_p, "r", encoding="utf-8") as f:
            local_map = json.load(f)
        items = sorted(((int(v), str(k)) for k, v in local_map.items()), key=lambda x: x[0])
        for local_row, doc_id in items:
            if not (0 <= local_row < rows):
                raise RuntimeError(f"row out of range in {map_p}: {doc_id}->{local_row}")
            if doc_id in docid2row:
                raise RuntimeError(f"duplicate doc_id during merge: {doc_id}")
            docid2row[doc_id] = offset + local_row

        if not live_meta:
            local_meta = os.path.join(sd, "meta.jsonl")
            if os.path.exists(local_meta):
                with open(local_meta, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            meta_f.write(line)
                            meta_count += 1

            local_fail = os.path.join(sd, "failed_docs.jsonl")
            if os.path.exists(local_fail):
                with open(local_fail, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            fail_f.write(line)
                            fail_count += 1

        offset += rows
finally:
    if meta_f is not None:
        meta_f.close()
    if fail_f is not None:
        fail_f.close()

del merged

with open(map_path, "w", encoding="utf-8") as f:
    json.dump(docid2row, f, ensure_ascii=False, indent=2)

merged_cfg = {"merged_from_shards": shard_dirs, "num_rows": total_rows, "embed_dim": emb_dim}
first_cfg = os.path.join(shard_dirs[0], "config.json")
if os.path.exists(first_cfg):
    try:
        with open(first_cfg, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        if isinstance(base_cfg, dict):
            merged_cfg.update(base_cfg)
    except Exception:
        pass
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(merged_cfg, f, ensure_ascii=False, indent=2)

print(f"[merge] saved: {embed_path}")
print(f"[merge] saved: {map_path}")
if live_meta:
    print(f"[merge] rows={total_rows} meta/fail were streamed live")
else:
    print(f"[merge] rows={total_rows} meta_lines={meta_count} fail_lines={fail_count}")
PY

  if [[ "${KEEP_SHARD_DIRS}" -ne 1 ]]; then
    rm -rf "${SHARD_DIRS[@]}"
  fi

  echo "[split:${SPLIT}] done"
done

echo "[embed_segments_1fps_4gpu] done"
