#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"
cd "${repo_root}"

# Reset legacy runtime toggles so this script runs with baseline defaults unless explicitly overridden.
unset VLLM_DISABLE_CUSTOM_ALL_REDUCE || true
unset VLLM_ENFORCE_EAGER || true
unset VLLM_USE_V1 || true
unset VLLM_WORKER_MULTIPROC_METHOD || true
unset GRPO_LOGPS_NO_CHUNK || true

timestamp="$(date +%Y%m%d%H%M)"
exp="${EXP_NAME:-stage2_grpo}"

MODEL_PATH="${MODEL_PATH:-${VIDEOSEARCH_BASE_MODEL}}"
MODEL_PATH="$(videosearch_local_model_path "${MODEL_PATH}")"
if [[ -z "${QUERY_EMBEDDER_PATH:-}" ]]; then
  if [[ -d "${MODEL_PATH}/query_embedder" ]]; then
    QUERY_EMBEDDER_PATH="${MODEL_PATH}/query_embedder"
  else
    QUERY_EMBEDDER_PATH="${VIDEOSEARCH_EMBED_MODEL}"
  fi
fi
USE_QUERY_EMBEDDER_PATH="${USE_QUERY_EMBEDDER_PATH:-True}"
OUT_ROOT="${OUT_ROOT:-${VIDEOSEARCH_OUTPUT_ROOT}/grpo}"
output_dir="${OUT_ROOT}/${timestamp}-${exp}"
mkdir -p "${output_dir}"

REFINE_TOKEN="${REFINE_TOKEN:-<REFINE>}"
REFINE_ROLLOUT_DEPTH="${REFINE_ROLLOUT_DEPTH:-8}"

python - "${MODEL_PATH}" "${REFINE_TOKEN}" <<'PY'
import os
import sys
from transformers import AutoProcessor, AutoTokenizer

model_path = sys.argv[1]
refine_token = (sys.argv[2] or "<REFINE>").strip() or "<REFINE>"

load_errors = []
tokenizer = None
loaded_from = ""
candidates = [model_path, os.path.dirname(os.path.normpath(model_path))]
seen = set()
for cand in candidates:
    cand = str(cand or "").strip()
    if not cand or cand in seen:
        continue
    seen.add(cand)
    try:
        proc = AutoProcessor.from_pretrained(cand, trust_remote_code=True)
        tokenizer = proc.tokenizer
        loaded_from = f"AutoProcessor:{cand}"
        break
    except Exception as exc:
        load_errors.append(f"{cand} (processor): {exc}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
        loaded_from = f"AutoTokenizer:{cand}"
        break
    except Exception as exc:
        load_errors.append(f"{cand} (tokenizer): {exc}")

if tokenizer is None:
    print("[refine_guard][error] failed to load tokenizer for checkpoint validation.", file=sys.stderr)
    for msg in load_errors:
        print(f"  - {msg}", file=sys.stderr)
    sys.exit(1)

vocab = tokenizer.get_vocab()
if refine_token not in vocab:
    print(
        f"[refine_guard][error] refine token '{refine_token}' not found in checkpoint tokenizer: {model_path}",
        file=sys.stderr,
    )
    sys.exit(2)

print(
    f"[refine_guard] ok: refine_token={refine_token}, "
    f"loaded_from={loaded_from}"
)
PY

default_activitynet_root="$(videosearch_dataset_dir activitynet)"
DATASET_INFO="${DATASET_INFO:-}"
DATASET_NAME="${DATASET_NAME:-}"
VIDEO_ROOT="${VIDEO_ROOT:-${default_activitynet_root}/train/video_npy_with_meta}"
VIDEO_META_PATH="${VIDEO_META_PATH:-${default_activitynet_root}/train/video_npy_with_meta/meta.jsonl}"
VERIFIED_JSONL="${VERIFIED_JSONL:-${default_activitynet_root}/raw_annotation/train.jsonl}"
QUERY_EMBEDDINGS_PATH="${QUERY_EMBEDDINGS_PATH:-${default_activitynet_root}/train/query_embedding/query_embeddings.train.npy}"
QUERY_META_PATH="${QUERY_META_PATH:-${default_activitynet_root}/train/query_embedding/query_meta.train.jsonl}"
INDEX_FAISS_PATH="${INDEX_FAISS_PATH:-${default_activitynet_root}/train/index/index.faiss}"
INDEX_ID_MAP_PATH="${INDEX_ID_MAP_PATH:-${default_activitynet_root}/train/index/id_map.json}"
VIDEO_EMBEDDINGS_PATH="${VIDEO_EMBEDDINGS_PATH:-${default_activitynet_root}/train/video_embedding_1fps/segment_embeds.npy}"
VIDEO_DOCID2ROW_PATH="${VIDEO_DOCID2ROW_PATH:-${default_activitynet_root}/train/video_embedding_1fps/docid2row.json}"
MINIMAL_JSONL="${MINIMAL_JSONL:-${default_activitynet_root}/grpo_data/train.minimal_top1.jsonl}"
MINIMAL_STATS_JSON="${MINIMAL_STATS_JSON:-${default_activitynet_root}/grpo_data/stats.minimal_top1.json}"
HARD_NEGATIVE_TOPK="${HARD_NEGATIVE_TOPK:-24}"
HARD_NEGATIVE_DEPTH="${HARD_NEGATIVE_DEPTH:-200}"
AUGMENT_MATCH_TO_BALANCE="${AUGMENT_MATCH_TO_BALANCE:-1}"
TARGET_MATCH_RATIO="${TARGET_MATCH_RATIO:-1.0}"
AUGMENT_SEED="${AUGMENT_SEED:-42}"
FORCE_REBUILD_MINIMAL="${FORCE_REBUILD_MINIMAL:-False}"
QUERY_KEY="${QUERY_KEY:-fig_desc}"
GT_KEY="${GT_KEY:-video}"
GT_TIME_KEY="${GT_TIME_KEY:-time}"
GT_DURATION_KEY="${GT_DURATION_KEY:-duration}"
BOOTSTRAP_KEY="${BOOTSTRAP_KEY:-retrieved_video}"

if [[ -z "${DATASET_NAME}" ]]; then
  dataset_name_suffix="${GRPO_DATASET_TAG:-auto}"
  DATASET_NAME="Verified-Search-Minimal-Top1-${dataset_name_suffix}"
  echo "[stage2] DATASET_NAME auto-set to ${DATASET_NAME}"
fi

need_build="False"
if [[ ! -f "${MINIMAL_JSONL}" ]]; then
  need_build="True"
elif [[ "${FORCE_REBUILD_MINIMAL}" == "True" ]]; then
  need_build="True"
else
  if ! python - "${MINIMAL_JSONL}" <<'PY'
import json, sys
path = sys.argv[1]
ok = False
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ok = (
            isinstance(row.get("hard_negative_ids", None), list)
            and "gt_time" in row
            and "gt_duration" in row
            and "is_augmented" in row
        )
        break
print("ok" if ok else "missing")
sys.exit(0 if ok else 1)
PY
  then
    need_build="True"
  fi
fi

if [[ "${need_build}" == "True" ]]; then
  echo "[stage2] building minimal dataset with hard negative pool..."
  python "${repo_root}/scripts/internal/stage2/build_minimal_match_dataset.py" \
    --verified_jsonl "${VERIFIED_JSONL}" \
    --query_embeddings_path "${QUERY_EMBEDDINGS_PATH}" \
    --query_meta_path "${QUERY_META_PATH}" \
    --index_faiss_path "${INDEX_FAISS_PATH}" \
    --index_id_map_path "${INDEX_ID_MAP_PATH}" \
    --output_jsonl "${MINIMAL_JSONL}" \
    --output_stats_json "${MINIMAL_STATS_JSON}" \
    --video_root "${VIDEO_ROOT}" \
    --query_key "${QUERY_KEY}" \
    --gt_key "${GT_KEY}" \
    --time_key "${GT_TIME_KEY}" \
    --duration_key "${GT_DURATION_KEY}" \
    --hard_negative_topk "${HARD_NEGATIVE_TOPK}" \
    --hard_negative_depth "${HARD_NEGATIVE_DEPTH}" \
    --augment_match_to_balance "${AUGMENT_MATCH_TO_BALANCE}" \
    --target_match_ratio "${TARGET_MATCH_RATIO}" \
    --augment_seed "${AUGMENT_SEED}"
fi

AUTO_DATASET_INFO="${AUTO_DATASET_INFO:-True}"
if [[ "${AUTO_DATASET_INFO}" == "True" ]]; then
  dataset_cfg_dir="${output_dir}/dataset_config"
  mkdir -p "${dataset_cfg_dir}"
  DATASET_INFO="${dataset_cfg_dir}/data_config.auto.yaml"
  cat > "${DATASET_INFO}" <<EOF
${DATASET_NAME}:
  anno_path: ${MINIMAL_JSONL}
  video_root: ${VIDEO_ROOT}
  video_meta_path: ${VIDEO_META_PATH}
  query_key: query
  gt_key: gt_video
  bootstrap_key: ${BOOTSTRAP_KEY}
EOF
fi

if [[ -z "${DATASET_INFO}" || ! -f "${DATASET_INFO}" ]]; then
  echo "[stage2][error] dataset_info not found: ${DATASET_INFO:-<empty>}" >&2
  echo "[stage2][hint] set AUTO_DATASET_INFO=True or pass DATASET_INFO=/path/to/config.yaml" >&2
  exit 1
fi
echo "[stage2] dataset_info=${DATASET_INFO}"
echo "[stage2] dataset_name=${DATASET_NAME}"

TRAIN_GPUS="${TRAIN_GPUS:-${GPUS:-1,2,3}}"
IFS=',' read -ra GPU_ARR <<< "${TRAIN_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARR[@]}}"
MASTER_PORT="${MASTER_PORT:-29800}"

MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-22000}"
for gpu_id in "${GPU_ARR[@]}"; do
  free_mb="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ -z "${free_mb}" ]]; then
    echo "[stage2][error] failed to query free memory for GPU ${gpu_id}" >&2
    exit 1
  fi
  if (( free_mb < MIN_FREE_MEM_MB )); then
    echo "[stage2][error] GPU ${gpu_id} free memory is too low: ${free_mb} MiB (< ${MIN_FREE_MEM_MB} MiB)." >&2
    echo "[stage2][error] choose different TRAIN_GPUS or stop other jobs on this GPU." >&2
    exit 1
  fi
done

# Qwen3-VL-2B uses 16 heads. Pick a valid TP that divides both world size and 16.
VLLM_TP="${VLLM_TP:-auto}"
if [[ "${VLLM_TP}" == "auto" ]]; then
  VLLM_TP=""
fi
if [[ -z "${VLLM_TP}" ]]; then
  for cand in 8 4 2 1; do
    if (( cand <= NPROC_PER_NODE )) && (( NPROC_PER_NODE % cand == 0 )) && (( 16 % cand == 0 )); then
      VLLM_TP="${cand}"
      break
    fi
  done
fi
if [[ -z "${VLLM_TP}" ]]; then
  echo "[stage2][error] failed to infer valid VLLM_TP for NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
echo "[stage2] NPROC_PER_NODE=${NPROC_PER_NODE}, VLLM_TP=${VLLM_TP}"
echo "[stage2] video_meta_path=${VIDEO_META_PATH:-auto}"

# Match SFT/eval prompt style.
export SEARCH_SYSTEM_PROMPT_STYLE="${SEARCH_SYSTEM_PROMPT_STYLE:-sft_eval}"
export SEARCH_LOG_BASE_DIR="${SEARCH_LOG_BASE_DIR:-${output_dir}/search_logs}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
refine_suffix="${REFINE_TOKEN}"
export SEARCH_REFINE_SUFFIX="${SEARCH_REFINE_SUFFIX:-${refine_suffix}}"

video_min_pixels=$((16 * 32 * 32))
video_max_pixels=$((64 * 32 * 32))
video_total_pixels=$((1280 * 32 * 32))
max_frames=64
image_min_pixels=$((4 * 32 * 32))
image_max_pixels=$((768 * 32 * 32))

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
USE_VLLM="${USE_VLLM:-True}"
OPTIM_NAME="${OPTIM_NAME:-adamw_torch_fused}"
DS_STAGE="${DS_STAGE:-1}"
DS_CONFIG="${DS_CONFIG:-${repo_root}/scripts/internal/dsconfig/zero${DS_STAGE}.json}"
TUNE_MM_LLM="${TUNE_MM_LLM:-True}"
TUNE_MM_MLP="${TUNE_MM_MLP:-False}"
TUNE_MM_VISION="${TUNE_MM_VISION:-False}"
# Keep flash attention, but disable vLLM custom all-reduce by default for stability on this node.
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-True}"
# Optional escape hatch if CUDA graph capture is unstable.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"
echo "[stage2] use_vllm=${USE_VLLM}, vllm_mem_util=${VLLM_GPU_MEMORY_UTILIZATION}, attn_impl=${ATTN_IMPL}, disable_custom_all_reduce=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}, enforce_eager=${VLLM_ENFORCE_EAGER}"
if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "[stage2][error] deepspeed config not found: ${DS_CONFIG}" >&2
  echo "[stage2][hint] set DS_STAGE=1|2|3 or DS_CONFIG=/abs/path/to/config.json" >&2
  exit 1
fi
echo "[stage2] deepspeed_config=${DS_CONFIG}"

# Reward weights:
# - W_ACC: answer correctness
# - W_THINK_REWARD: think-tag format reward
# - W_TIME_FMT: start/end tag format reward
# - W_TIME_IOU: temporal IoU reward on matched positives
# - W_FMT: refine presence format reward (has <REFINE>)
# - W_MARGIN: query-refine quality reward
W_ACC="${W_ACC:-1.6}"
W_THINK_REWARD="${W_THINK_REWARD:-${w_think_reward:-0.05}}"
W_TIME_FMT="${W_TIME_FMT:-0.05}"
W_TIME_IOU="${W_TIME_IOU:-0.35}"
W_FMT="${W_FMT:-0.8}"
W_MARGIN="${W_MARGIN:-0.5}"
MARGIN_SCALE="${MARGIN_SCALE:-1.0}"
QUERY_REFINE_TEMP="${QUERY_REFINE_TEMP:-0.1}"
USE_SQR_LATENT_LOSS="${USE_SQR_LATENT_LOSS:-True}"
SQR_LATENT_SIGMA="${SQR_LATENT_SIGMA:-0.05}"
SQR_LATENT_LOSS_WEIGHT="${SQR_LATENT_LOSS_WEIGHT:-1.0}"
SQR_LATENT_CLIP_EPSILON="${SQR_LATENT_CLIP_EPSILON:-0.2}"
SQR_LATENT_TRAIN_DEPTH="${SQR_LATENT_TRAIN_DEPTH:--1}"
SQR_LATENT_EVERY_N_STEPS="${SQR_LATENT_EVERY_N_STEPS:-1}"
USE_INFONCE_LATENT_AUX_LOSS="${USE_INFONCE_LATENT_AUX_LOSS:-True}"
INFONCE_LATENT_LOSS_WEIGHT="${INFONCE_LATENT_LOSS_WEIGHT:-1.0}"
INFONCE_LATENT_TEMPERATURE="${INFONCE_LATENT_TEMPERATURE:-0.1}"
INFONCE_LATENT_MODE="${INFONCE_LATENT_MODE:-abs}" # abs|delta
INFONCE_LATENT_TRAIN_DEPTH="${INFONCE_LATENT_TRAIN_DEPTH:--1}"
INFONCE_LATENT_EVERY_N_STEPS="${INFONCE_LATENT_EVERY_N_STEPS:-1}"
# Behavior when <REFINE> is absent for r_query_refine_quality:
#   0.0  => no contribution
#  -1.0  => explicit penalty
NO_REFINE_QUALITY_PENALTY="${NO_REFINE_QUALITY_PENALTY:--0.0}"
SEARCH_FORCE_REFINE_TOKEN="${SEARCH_FORCE_REFINE_TOKEN:-False}"
SEARCH_DEBUG="${SEARCH_DEBUG:-True}"
REWARD_DEBUG="${REWARD_DEBUG:-True}"
REWARD_DEBUG_EVERY="${REWARD_DEBUG_EVERY:-1}"
REWARD_DEBUG_MAX_SAMPLES="${REWARD_DEBUG_MAX_SAMPLES:-0}"
REWARD_DEBUG_STEPS="${REWARD_DEBUG_STEPS:-}"
GRPO_STAGE_DEBUG="${GRPO_STAGE_DEBUG:-False}"
GRPO_STAGE_DEBUG_EVERY="${GRPO_STAGE_DEBUG_EVERY:-1}"
GRPO_STAGE_WATCHDOG_SEC="${GRPO_STAGE_WATCHDOG_SEC:-0}"

export GRPO_STAGE_DEBUG
export GRPO_STAGE_DEBUG_EVERY
export GRPO_STAGE_WATCHDOG_SEC
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export GRPO_LATENT_FORWARD_USE_VISION="${GRPO_LATENT_FORWARD_USE_VISION:-True}"
export GRPO_QUERY_REFINE_TEMPERATURE="${QUERY_REFINE_TEMP}"
export GRPO_NO_REFINE_QUALITY_PENALTY="${NO_REFINE_QUALITY_PENALTY}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE
export VLLM_ENFORCE_EAGER

gen_batch_size=$((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * STEPS_PER_GENERATION))
if (( gen_batch_size % NUM_GENERATIONS != 0 )); then
  fixed_steps="${STEPS_PER_GENERATION}"
  while (( (PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * fixed_steps) % NUM_GENERATIONS != 0 )); do
    fixed_steps=$((fixed_steps + 1))
  done
  echo "[stage2][warn] generation_batch_size=${gen_batch_size} is not divisible by num_generations=${NUM_GENERATIONS}."
  echo "[stage2][warn] auto-adjust steps_per_generation: ${STEPS_PER_GENERATION} -> ${fixed_steps}"
  STEPS_PER_GENERATION="${fixed_steps}"
fi

# Keep GRPO generation/update cadence aligned by default.
# This avoids entering old_logps branch caused by grad_accum_steps % (steps_per_generation*num_iterations) != 0.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${STEPS_PER_GENERATION}}"
echo "[stage2] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"

reward_debug_steps_args=()
if [[ -n "${REWARD_DEBUG_STEPS}" ]]; then
  reward_debug_steps_args=(--reward_debug_steps "${REWARD_DEBUG_STEPS}")
fi
max_steps_args=()
if [[ -n "${MAX_STEPS}" ]]; then
  max_steps_args=(--max_steps "${MAX_STEPS}")
fi
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-False}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1 --master_port="${MASTER_PORT}" \
  videosearch_r1/train_grpo_qwen3_vl.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${output_dir}" \
  --dataset_info "${DATASET_INFO}" \
  --dataset_name "${DATASET_NAME}" \
  --video_root_override "${VIDEO_ROOT}" \
  --video_meta_path "${VIDEO_META_PATH}" \
  --video_min_pixels "${video_min_pixels}" \
  --video_max_pixels "${video_max_pixels}" \
  --video_total_pixels "${video_total_pixels}" \
  --max_frames "${max_frames}" \
  --image_min_pixels "${image_min_pixels}" \
  --image_max_pixels "${image_max_pixels}" \
  --bf16 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "constant_with_warmup" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --deepspeed "${DS_CONFIG}" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_only_model "${SAVE_ONLY_MODEL}" \
  --logging_steps 1 \
  --eval_strategy "no" \
  --report_to "${REPORT_TO:-wandb}" \
  --optim "${OPTIM_NAME}" \
  --weight_decay 0.01 \
  --tune_mm_llm "${TUNE_MM_LLM}" \
  --tune_mm_mlp "${TUNE_MM_MLP}" \
  --tune_mm_vision "${TUNE_MM_VISION}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --num_generations "${NUM_GENERATIONS}" \
  --steps_per_generation "${STEPS_PER_GENERATION}" \
  --beta 0.01 \
  --reward_funcs "r_answer_binary" "r_think_format" "r_time_format" "r_time_iou" "r_refine_presence" "r_query_refine_quality" \
  --reward_weights "${W_ACC}" "${W_THINK_REWARD}" "${W_TIME_FMT}" "${W_TIME_IOU}" "${W_FMT}" "${W_MARGIN}" \
  --apply_monkey_patch "enforce_image_video" \
  --rl_mode "cot_rl" \
  --save_total_limit 2 \
  --use_vllm "${USE_VLLM}" \
  --vllm_mode "colocate" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_tensor_parallel_size "${VLLM_TP}" \
  --use_search True \
  --search_max_turns 1 \
  --search_topk 1 \
  --search_use_instruction False \
  --search_use_original_query True \
  --search_use_bootstrap_video True \
  --search_force_refine_token "${SEARCH_FORCE_REFINE_TOKEN}" \
  --refine_token "${REFINE_TOKEN}" \
  --refine_rollout_depth "${REFINE_ROLLOUT_DEPTH}" \
  --search_rank_k 0 \
  --search_debug "${SEARCH_DEBUG}" \
  --reward_debug "${REWARD_DEBUG}" \
  --reward_debug_every "${REWARD_DEBUG_EVERY}" \
  --reward_debug_max_samples "${REWARD_DEBUG_MAX_SAMPLES}" \
  "${max_steps_args[@]}" \
  "${reward_debug_steps_args[@]}" \
  --use_latent_improve_reward True \
  --margin_reward_scale "${MARGIN_SCALE}" \
  --use_refine_gate False \
  --use_query_embedder_path "${USE_QUERY_EMBEDDER_PATH}" \
  --query_embedder_model_path "${QUERY_EMBEDDER_PATH}" \
  --qfinal_pooling "latent_last" \
  --qfinal_normalize True \
  --query_embedder_max_length 128 \
  --query_embeddings_path "${QUERY_EMBEDDINGS_PATH}" \
  --query_meta_path "${QUERY_META_PATH}" \
  --video_embeddings_path "${VIDEO_EMBEDDINGS_PATH}" \
  --video_docid2row_path "${VIDEO_DOCID2ROW_PATH}" \
  --use_sqr_latent_loss "${USE_SQR_LATENT_LOSS}" \
  --sqr_latent_sigma "${SQR_LATENT_SIGMA}" \
  --sqr_latent_loss_weight "${SQR_LATENT_LOSS_WEIGHT}" \
  --sqr_latent_clip_epsilon "${SQR_LATENT_CLIP_EPSILON}" \
  --sqr_latent_train_depth "${SQR_LATENT_TRAIN_DEPTH}" \
  --sqr_latent_every_n_steps "${SQR_LATENT_EVERY_N_STEPS}" \
  --use_infonce_latent_aux_loss "${USE_INFONCE_LATENT_AUX_LOSS}" \
  --infonce_latent_loss_weight "${INFONCE_LATENT_LOSS_WEIGHT}" \
  --infonce_latent_temperature "${INFONCE_LATENT_TEMPERATURE}" \
  --infonce_latent_mode "${INFONCE_LATENT_MODE}" \
  --infonce_latent_train_depth "${INFONCE_LATENT_TRAIN_DEPTH}" \
  --infonce_latent_every_n_steps "${INFONCE_LATENT_EVERY_N_STEPS}"
