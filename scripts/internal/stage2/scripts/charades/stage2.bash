#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"

# Stage 2: stable GRPO baseline without the optional latent auxiliary loss.
export GPUS="${GPUS:-1,2,3}"
export TRAIN_GPUS="${TRAIN_GPUS:-${GPUS}}"
export MASTER_PORT="${MASTER_PORT:-29810}"

export EXP_NAME="${EXP_NAME:-charades_v1_stable_noaux}"
export WANDB_NAME="${WANDB_NAME:-charades_v1_stable_noaux}"

# Stable reward balance (avoid over-bias to not_matched)
export W_ACC="${W_ACC:-1.6}"
export W_THINK_REWARD="${W_THINK_REWARD:-0.05}"
export W_TIME_FMT="${W_TIME_FMT:-0.05}"
export W_TIME_IOU="${W_TIME_IOU:-0.35}"
export W_FMT="${W_FMT:-0.10}"
export W_MARGIN="${W_MARGIN:-0.8}"
export NO_REFINE_QUALITY_PENALTY="${NO_REFINE_QUALITY_PENALTY:-0.0}"
export SEARCH_FORCE_REFINE_TOKEN="${SEARCH_FORCE_REFINE_TOKEN:-False}"

# Aux losses off
export USE_SQR_LATENT_LOSS="${USE_SQR_LATENT_LOSS:-False}"
export USE_INFONCE_LATENT_AUX_LOSS="${USE_INFONCE_LATENT_AUX_LOSS:-False}"

# Runtime defaults
export DS_STAGE="${DS_STAGE:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"

export MODEL_PATH="${MODEL_PATH:-charades-stage1}"
set_grpo_dataset_defaults "charades"
unset DATASET_NAME || true
export QUERY_EMBEDDER_PATH="${QUERY_EMBEDDER_PATH:-${MODEL_PATH}/query_embedder}"
export WANDB_PROJECT="${WANDB_PROJECT:-videosearch-r1}"

exec "${script_dir}/../../train_stage2.bash" "$@"
