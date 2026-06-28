#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/toy/toy_activitynet_full_process.bash

What it does:
  1) Builds a tiny ActivityNet-FIG raw layout from VERIFIED annotations and about 10 videos per split.
  2) Runs Start From Scratch data construction.
  3) Runs Stage 1 SFT smoke training.
  4) Runs Stage 2 GRPO smoke training.
  5) Runs temporal-grounding inference and report on the toy test split.

Key env knobs:
  TOY_GPUS=1,2,3
  TOY_ACTIVITYNET_VIDEO_SOURCE=/path/to/ActivityNet
  TOY_ACTIVITYNET_DOWNLOAD_MISSING=0  # set 1 to try yt-dlp for missing videos
  TOY_ACTIVITYNET_TRAIN_VIDEOS=10
  TOY_ACTIVITYNET_VAL_VIDEOS=10
  TOY_ACTIVITYNET_TEST_VIDEOS=10
  RUN_DATA_CONSTRUCT=1
  RUN_STAGE1=1
  RUN_STAGE2=1
  RUN_INFERENCE=1
  RUN_REPORT=1

Resume:
  RUN_FROM_STEP=5 RUN_STAGE1=0 RUN_STAGE2=0 RUN_INFERENCE=0 bash scripts/toy/toy_activitynet_full_process.bash
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TOY_CONDA_ENV="${TOY_CONDA_ENV:-vlpo}"
if [[ -n "${TOY_CONDA_ENV}" && "${TOY_SKIP_CONDA_ACTIVATE:-0}" != "1" && "${TOY_CONDA_REEXEC:-0}" != "1" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${TOY_CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      # Re-exec keeps downstream torchrun/python on the requested env PATH.
      export TOY_CONDA_REEXEC=1
      # shellcheck disable=SC1090
      source "${conda_base}/etc/profile.d/conda.sh"
      conda activate "${TOY_CONDA_ENV}"
      exec bash "$0" "$@"
    fi
  fi
fi

username="${USER:-$(id -un 2>/dev/null || echo user)}"

toy_default_workspace() {
  local root
  local min_free_gb="${TOY_MIN_FREE_GB:-150}"
  for root in /hub_data1 /hub_data3 /hub_data2; do
    [[ -d "${root}" ]] || continue
    local candidate="${root}/${username}/videosearchr1_toy/activitynet_full_process"
    if mkdir -p "${candidate}" 2>/dev/null; then
      local avail_kb
      avail_kb="$(df -Pk "${candidate}" 2>/dev/null | awk 'NR==2 {print $4}')"
      if [[ -z "${avail_kb}" || "${avail_kb}" -ge $((min_free_gb * 1024 * 1024)) ]]; then
        echo "${candidate}"
        return 0
      fi
    fi
  done
  echo "${repo_root}/.cache/toy_activitynet_full_process"
}

export VIDEOSEARCH_WORKSPACE="${VIDEOSEARCH_WORKSPACE:-$(toy_default_workspace)}"
source "${repo_root}/scripts/common/env.bash"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    export PYTHON_BIN="python3"
  fi
fi

TOY_GPUS="${TOY_GPUS:-1,2,3}"
IFS=',' read -ra toy_gpu_arr <<< "${TOY_GPUS}"
TOY_NPROC="${TOY_NPROC:-${#toy_gpu_arr[@]}}"
TOY_PRIMARY_GPU="${TOY_PRIMARY_GPU:-${toy_gpu_arr[0]}}"
TOY_REASONING_GPUS="${TOY_REASONING_GPUS:-${TOY_PRIMARY_GPU}}"
TOY_EVAL_GPUS="${TOY_EVAL_GPUS:-${TOY_PRIMARY_GPU}}"

toy_root="${VIDEOSEARCH_WORKSPACE}/toy_activitynet"
verified_root="${toy_root}/verified_annotations/activitynet-fig"
state_dir="${toy_root}/state"
mkdir -p "${verified_root}" "${state_dir}" "${VIDEOSEARCH_OUTPUT_ROOT}/configs"

download_verified_file() {
  local name="$1"
  local url="${VERIFIED_RAW_BASE_URL:-https://raw.githubusercontent.com/hlchen23/VERIFIED/main/fine-grained-anno}/activitynet-fig/${name}"
  local dst="${verified_root}/${name}"
  if [[ -s "${dst}" && "${OVERWRITE_VERIFIED:-0}" != "1" ]]; then
    echo "[toy_activitynet] exists: ${dst}"
    return 0
  fi
  echo "[toy_activitynet] download ${url}"
  curl -L --fail --retry 4 --retry-delay 3 --connect-timeout 20 --max-time 300 -o "${dst}.tmp" "${url}"
  mv "${dst}.tmp" "${dst}"
}

default_video_source=""
if [[ -d "/hub_data2/jinyoungkim/DATAS/ActivityNet/v1-2" ]]; then
  default_video_source="/hub_data2/jinyoungkim/DATAS/ActivityNet/v1-2"
fi
TOY_ACTIVITYNET_VIDEO_SOURCE="${TOY_ACTIVITYNET_VIDEO_SOURCE:-${default_video_source}}"
TOY_ACTIVITYNET_LINK_MODE="${TOY_ACTIVITYNET_LINK_MODE:-symlink}"

echo "[toy_activitynet] repo=${repo_root}"
echo "[toy_activitynet] workspace=${VIDEOSEARCH_WORKSPACE}"
echo "[toy_activitynet] data_root=${VIDEOSEARCH_DATA_ROOT}"
echo "[toy_activitynet] output_root=${VIDEOSEARCH_OUTPUT_ROOT}"
echo "[toy_activitynet] gpus=${TOY_GPUS}"
echo "[toy_activitynet] python=${PYTHON_BIN}"

RUN_DATA_CONSTRUCT="${RUN_DATA_CONSTRUCT:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_REPORT="${RUN_REPORT:-1}"

if [[ "${RUN_DATA_CONSTRUCT}" == "1" ]]; then
  download_verified_file activitynet_fig_train.jsonl
  download_verified_file activitynet_fig_val_1.jsonl
  download_verified_file activitynet_fig_val_2.jsonl

  video_source_args=()
  if [[ -n "${TOY_ACTIVITYNET_VIDEO_SOURCE}" ]]; then
    IFS=':' read -ra video_sources <<< "${TOY_ACTIVITYNET_VIDEO_SOURCE}"
    for src in "${video_sources[@]}"; do
      [[ -n "${src}" ]] && video_source_args+=(--video-source "${src}")
    done
  fi
  download_args=()
  if [[ "${TOY_ACTIVITYNET_DOWNLOAD_MISSING:-0}" == "1" ]]; then
    download_args+=(--download-missing)
  fi
  allow_args=()
  if [[ "${TOY_ACTIVITYNET_ALLOW_SMALLER:-0}" == "1" ]]; then
    allow_args+=(--allow-smaller)
  fi

  "${PYTHON_BIN}" "${repo_root}/scripts/data_construct/build_activitynet_toy_subset.py" \
    --verified-root "${verified_root}" \
    --output-data-root "${VIDEOSEARCH_DATA_ROOT}" \
    --train-videos "${TOY_ACTIVITYNET_TRAIN_VIDEOS:-10}" \
    --val-videos "${TOY_ACTIVITYNET_VAL_VIDEOS:-10}" \
    --test-videos "${TOY_ACTIVITYNET_TEST_VIDEOS:-10}" \
    --rows-per-video "${TOY_ACTIVITYNET_ROWS_PER_VIDEO:-1}" \
    --link-mode "${TOY_ACTIVITYNET_LINK_MODE}" \
    --overwrite \
    "${video_source_args[@]}" \
    "${download_args[@]}" \
    "${allow_args[@]}"

  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EMBED_DEVICE="${TOY_PRIMARY_GPU}" \
    GPU_IDS="${TOY_GPUS}" \
    NUM_SHARDS="${TOY_NUM_SHARDS:-${TOY_NPROC}}" \
    SEGMENT_BATCH_SIZE="${TOY_SEGMENT_BATCH_SIZE:-1}" \
    NUM_WORKERS="${TOY_EXTRACT_WORKERS:-4}" \
    LOG_EVERY=1 \
    OVERWRITE="${TOY_OVERWRITE_NPY:-1}" \
    SKIP_EXISTING="${TOY_SKIP_EXISTING_EMBEDS:-0}" \
    SAMPLE_MAX_FRAMES="${TOY_SAMPLE_MAX_FRAMES:-32}" \
    VIDEO_MAXLEN="${TOY_VIDEO_MAXLEN:-32}" \
    VIDEO_MAX_PIXELS="${TOY_VIDEO_MAX_PIXELS:-200704}" \
    SHARD_GPUS="${TOY_REASONING_GPUS}" \
    MODEL_NAME="${TOY_REASONING_MODEL:-${VIDEOSEARCH_BASE_MODEL}}" \
    BACKEND="${TOY_REASONING_BACKEND:-local_vllm}" \
    WORKERS="${TOY_REASONING_WORKERS:-1}" \
    LOCAL_BATCH_SIZE="${TOY_REASONING_BATCH_SIZE:-1}" \
    LOCAL_GPU_MEMORY_UTILIZATION="${TOY_REASONING_GPU_MEMORY_UTILIZATION:-0.45}" \
    LOCAL_MAX_MODEL_LEN="${TOY_REASONING_MAX_MODEL_LEN:-8192}" \
    MAX_TOKENS="${TOY_REASONING_MAX_TOKENS:-1024}" \
    LIMIT="${TOY_REASONING_LIMIT:-0}" \
    TOPK_POOL="${TOY_TOPK_POOL:-10}" \
    MAX_NEGATIVES="${TOY_MAX_NEGATIVES:-10}" \
    HARD_NEGATIVE_TOPK="${TOY_HARD_NEGATIVE_TOPK:-8}" \
    HARD_NEGATIVE_DEPTH="${TOY_HARD_NEGATIVE_DEPTH:-32}" \
    MAX_ROWS="${TOY_GRPO_MAX_ROWS:-0}" \
    REGISTER_CONFIG=0 \
    RUN_FROM_STEP="${RUN_FROM_STEP:-1}" \
    bash "${repo_root}/scripts/data_construct/start_from_scratch.bash" activitynet
fi

latest_model_dir() {
  local run_dir="$1"
  local latest=""
  if [[ -d "${run_dir}" ]]; then
    latest="$(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  fi
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
  else
    echo "${run_dir}"
  fi
}

stage1_output_root="${TOY_STAGE1_OUTPUT_ROOT:-${VIDEOSEARCH_OUTPUT_ROOT}/toy_activitynet/stage1}"
stage2_output_root="${TOY_STAGE2_OUTPUT_ROOT:-${VIDEOSEARCH_OUTPUT_ROOT}/toy_activitynet/stage2}"
mkdir -p "${stage1_output_root}" "${stage2_output_root}"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  stage1_output_dir="${TOY_STAGE1_OUTPUT_DIR:-${stage1_output_root}/$(date +%Y%m%d%H%M%S)}"
  echo "[toy_activitynet] stage1_output=${stage1_output_dir}"
  env \
    GPUS="${TOY_GPUS}" \
    NPROC_PER_NODE="${TOY_NPROC}" \
    MASTER_PORT="${TOY_STAGE1_MASTER_PORT:-29143}" \
    MODEL_PATH="${TOY_BASE_MODEL:-${VIDEOSEARCH_BASE_MODEL}}" \
    OUTPUT_DIR="${stage1_output_dir}" \
    REPORT_TO=none \
    WANDB_DISABLED=true \
    MAX_SAMPLES="${TOY_STAGE1_MAX_SAMPLES:-8}" \
    MAX_STEPS="${TOY_STAGE1_MAX_STEPS:-1}" \
    SAVE_STEPS="${TOY_STAGE1_SAVE_STEPS:-1}" \
    NUM_TRAIN_EPOCHS="${TOY_STAGE1_EPOCHS:-1}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${TOY_STAGE1_BATCH_SIZE:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${TOY_STAGE1_GRAD_ACC:-1}" \
    DATALOADER_NUM_WORKERS="${TOY_DATALOADER_WORKERS:-0}" \
    EVAL_RECALL_ON_EVAL=False \
    USE_VERIFIED_TEST_EVAL=False \
    bash "${repo_root}/scripts/training/stage1/train.bash" activitynet
  echo "${stage1_output_dir}" > "${state_dir}/stage1_output_dir.txt"
elif [[ -f "${state_dir}/stage1_output_dir.txt" ]]; then
  stage1_output_dir="$(cat "${state_dir}/stage1_output_dir.txt")"
else
  stage1_output_dir="${TOY_STAGE1_OUTPUT_DIR:-}"
fi

stage1_model_path="${TOY_STAGE1_MODEL_PATH:-}"
if [[ -z "${stage1_model_path}" && -n "${stage1_output_dir:-}" ]]; then
  stage1_model_path="$(latest_model_dir "${stage1_output_dir}")"
fi
if [[ -n "${stage1_model_path}" ]]; then
  echo "${stage1_model_path}" > "${state_dir}/stage1_model_path.txt"
  echo "[toy_activitynet] stage1_model=${stage1_model_path}"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  if [[ -z "${stage1_model_path:-}" || ! -d "${stage1_model_path}" ]]; then
    echo "[toy_activitynet][error] stage1 model path not found: ${stage1_model_path:-<empty>}" >&2
    exit 1
  fi
  echo "[toy_activitynet] stage2_out_root=${stage2_output_root}"
  env \
    GPUS="${TOY_GPUS}" \
    TRAIN_GPUS="${TOY_GPUS}" \
    NPROC_PER_NODE="${TOY_NPROC}" \
    MASTER_PORT="${TOY_STAGE2_MASTER_PORT:-29943}" \
    MODEL_PATH="${stage1_model_path}" \
    OUT_ROOT="${stage2_output_root}" \
    EXP_NAME="${TOY_STAGE2_EXP_NAME:-toy_activitynet_stage2}" \
    WANDB_NAME="${TOY_STAGE2_EXP_NAME:-toy_activitynet_stage2}" \
    REPORT_TO=none \
    WANDB_DISABLED=true \
    MAX_STEPS="${TOY_STAGE2_MAX_STEPS:-1}" \
    SAVE_STEPS="${TOY_STAGE2_SAVE_STEPS:-1}" \
    SAVE_ONLY_MODEL="${TOY_STAGE2_SAVE_ONLY_MODEL:-True}" \
    NUM_EPOCHS="${TOY_STAGE2_EPOCHS:-1}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${TOY_STAGE2_BATCH_SIZE:-1}" \
    NUM_GENERATIONS="${TOY_STAGE2_NUM_GENERATIONS:-2}" \
    STEPS_PER_GENERATION="${TOY_STAGE2_STEPS_PER_GENERATION:-1}" \
    MIN_FREE_MEM_MB="${TOY_MIN_FREE_MEM_MB:-10000}" \
    VLLM_GPU_MEMORY_UTILIZATION="${TOY_STAGE2_VLLM_GPU_MEMORY_UTILIZATION:-0.18}" \
    USE_SQR_LATENT_LOSS="${TOY_STAGE2_USE_SQR_LATENT_LOSS:-False}" \
    USE_INFONCE_LATENT_AUX_LOSS="${TOY_STAGE2_USE_INFONCE_LATENT_AUX_LOSS:-False}" \
    SEARCH_DEBUG=False \
    REWARD_DEBUG=False \
    bash "${repo_root}/scripts/training/stage2/train.bash" activitynet
fi

stage2_model_path="${TOY_STAGE2_MODEL_PATH:-}"
if [[ -z "${stage2_model_path}" ]]; then
  latest_stage2_run="$(find "${stage2_output_root}" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${latest_stage2_run}" ]]; then
    stage2_model_path="$(latest_model_dir "${latest_stage2_run}")"
  fi
fi
if [[ -n "${stage2_model_path}" ]]; then
  echo "${stage2_model_path}" > "${state_dir}/stage2_model_path.txt"
  echo "[toy_activitynet] stage2_model=${stage2_model_path}"
fi

inference_jsonl=""
if [[ "${RUN_INFERENCE}" == "1" ]]; then
  if [[ -z "${stage2_model_path:-}" || ! -d "${stage2_model_path}" ]]; then
    echo "[toy_activitynet][error] stage2 model path not found: ${stage2_model_path:-<empty>}" >&2
    exit 1
  fi
  logs_dir="${TOY_INFERENCE_LOGS_DIR:-${VIDEOSEARCH_OUTPUT_ROOT}/toy_activitynet/inference_logs}"
  mkdir -p "${logs_dir}"
  env \
    EVAL_GPUS="${TOY_EVAL_GPUS}" \
    NUM_PROCESSES_PER_GPU="${TOY_EVAL_PROCESSES_PER_GPU:-1}" \
    LOGS_DIR="${logs_dir}" \
    MAX_SAMPLES="${TOY_INFERENCE_MAX_SAMPLES:-10}" \
    MAX_TURN="${TOY_INFERENCE_MAX_TURN:-1}" \
    TOPK="${TOY_INFERENCE_TOPK:-1,5,10}" \
    USE_VLLM="${TOY_INFERENCE_USE_VLLM:-False}" \
    EXTERNAL_EVAL_MAX_NEW_TOKENS="${TOY_INFERENCE_MAX_NEW_TOKENS:-128}" \
    RESUME_FROM_JSONL=False \
    bash "${repo_root}/scripts/inference/inference.bash" activitynet --checkpoint "${stage2_model_path}"
  inference_jsonl="$(find "${logs_dir}" -maxdepth 1 -type f -name 'external_verified_test_temporal_grounding_*.jsonl' | sort | tail -n 1 || true)"
  if [[ -n "${inference_jsonl}" ]]; then
    echo "${inference_jsonl}" > "${state_dir}/inference_jsonl.txt"
  fi
elif [[ -f "${state_dir}/inference_jsonl.txt" ]]; then
  inference_jsonl="$(cat "${state_dir}/inference_jsonl.txt")"
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
  if [[ -z "${inference_jsonl:-}" || ! -f "${inference_jsonl}" ]]; then
    echo "[toy_activitynet][error] inference jsonl not found: ${inference_jsonl:-<empty>}" >&2
    exit 1
  fi
  bash "${repo_root}/scripts/inference/report.bash" "${inference_jsonl}"
fi

echo "[toy_activitynet] done"
echo "[toy_activitynet] state=${state_dir}"
