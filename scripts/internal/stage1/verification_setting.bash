#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

timestamp=$(date +%Y%m%d%H%M)
exp="${EXP_NAME:-qwen3_vl_sft_oneturn_soft_verify_strong_temporal}"
export LOSS_TYPE="${LOSS_TYPE:-all_assistant}"

# Centralize outputs/caches under VIDEOSEARCH_WORKSPACE.
HUB2_ROOT="${HUB2_ROOT:-${VIDEOSEARCH_WORKSPACE}}"
mkdir -p \
  "${VIDEOSEARCH_OUTPUT_ROOT}/sft" \
  "${VIDEOSEARCH_OUTPUT_ROOT}/log" \
  "${HUB2_ROOT}/cache/hf" \
  "${HUB2_ROOT}/cache/torch_extensions" \
  "${HUB2_ROOT}/cache/triton" \
  "${HUB2_ROOT}/tmp" \
  "${HUB2_ROOT}/wandb"
export HF_HOME="${HF_HOME:-${HUB2_ROOT}/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUB2_ROOT}/cache/hf/transformers}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HUB2_ROOT}/cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HUB2_ROOT}/cache/triton}"
export WANDB_DIR="${WANDB_DIR:-${HUB2_ROOT}/wandb}"
export TMPDIR="${TMPDIR:-${HUB2_ROOT}/tmp}"
# CUDA allocator fragmentation 완화(필요 시 env로 override 가능)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ShareGPTDataArguments
image_min_pixels=$((4 * 28 * 28))
image_max_pixels=$((16384 * 28 * 28))
video_min_pixels=$((128 * 28 * 28))
# video_max_pixels=$((768 * 28 * 28))
video_max_pixels=$((256 * 28 * 28))
video_total_pixels=$((115200 * 28 * 28))
max_frames=64
fps=1.0
video_root_override="${VIDEO_ROOT_OVERRIDE:-}"


# ModelArguments
model_max_length=46384

# Verification setting: default freeze backbone (override with env if needed).
tune_mm_llm="${TUNE_MM_LLM:-True}"
tune_mm_mlp="${TUNE_MM_MLP:-True}"
tune_mm_vision="${TUNE_MM_VISION:-False}"

# TrainingArguments
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
learning_rate=2e-5
lr_scheduler_type="linear"
warmup_ratio=0.0
warmup_steps=0
weight_decay=0.0
max_grad_norm=1.0
optim="adamw_torch_fused"
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8
bf16=True
fp16=False
tf32=False
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-True}"
ddp_find_unused_parameters="${DDP_FIND_UNUSED_PARAMETERS:-False}"
max_length="$model_max_length"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
dataloader_pin_memory="${DATALOADER_PIN_MEMORY:-True}"
seed=42
save_strategy="steps"
save_steps="${SAVE_STEPS:-0.05}"
save_total_limit=10
save_only_model="${SAVE_ONLY_MODEL:-True}"
logging_steps="${LOGGING_STEPS:-5}"
eval_strategy="${EVAL_STRATEGY:-no}"
eval_steps="${EVAL_STEPS:-0.25}"
load_best_model_at_end=True
metric_for_best_model="eval_loss"
greater_is_better=False
report_to="${REPORT_TO:-none}"
run_name="${RUN_NAME:-}"
logging_dir="${LOGGING_DIR:-${VIDEOSEARCH_OUTPUT_ROOT}/log/}"
deepspeed="${DEEPSPEED_CONFIG:-}"
max_samples="${MAX_SAMPLES:-}"
max_steps="${MAX_STEPS:-}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"

if [ "${eval_strategy}" = "no" ]; then
  load_best_model_at_end=False
fi

# Verification defaults (dataset/path can be overridden via env).
model_path="${MODEL_PATH:-${VIDEOSEARCH_BASE_MODEL}}"
output_dir="${OUTPUT_DIR:-${VIDEOSEARCH_OUTPUT_ROOT}/sft/${timestamp}-${exp}/}"
dataset_info="${DATASET_INFO:-data/data_config.yaml}"
dataset_name_csv="${DATASET_NAME_CSV:-ActivityNet-Oneturn_temporal}"
IFS=',' read -ra dataset_name_raw <<< "${dataset_name_csv}"
dataset_name=()
for raw_name in "${dataset_name_raw[@]}"; do
  name="$(echo "${raw_name}" | xargs)"
  if [ -n "${name}" ]; then
    dataset_name+=("${name}")
  fi
done
if [ "${#dataset_name[@]}" -eq 0 ]; then
  echo "[verification][error] DATASET_NAME_CSV resolved to empty list: ${dataset_name_csv}" >&2
  exit 1
fi
# Retrieval optimization defaults (override via env for quick sweeps).
enable_retrieval_optimization="${ENABLE_RETRIEVAL_OPTIMIZATION:-True}"
retrieval_loss_weight="${RETRIEVAL_LOSS_WEIGHT:-0.3}"
retrieval_temperature="${RETRIEVAL_TEMPERATURE:-0.1}"
negative_pool_size="${NEGATIVE_POOL_SIZE:-24}"
retrieval_ignore_ambiguous_negatives="${RETRIEVAL_IGNORE_AMBIGUOUS_NEGATIVES:-True}"
ambiguous_negative_margin="${AMBIGUOUS_NEGATIVE_MARGIN:-0.0005}"
strict_negative_topk="${STRICT_NEGATIVE_TOPK:-0}"
projector_lr="${PROJECTOR_LR:-5e-5}"
use_refine_gate="${USE_REFINE_GATE:-False}"
zero_init_refine="${ZERO_INIT_REFINE:-True}"
retrieval_on_eval="${RETRIEVAL_ON_EVAL:-True}"
only_neg="${ONLY_NEG:-False}"
external_eval_on_gpu0="${EXTERNAL_EVAL_ON_GPU0:-False}"
external_eval_gpu="${EXTERNAL_EVAL_GPU:-0}"
eval_split_ratio="${EVAL_SPLIT_RATIO:-0}"
eval_recall_on_eval="${EVAL_RECALL_ON_EVAL:-True}"
eval_recall_ks="${EVAL_RECALL_KS:-1,5,10,100}"
eval_detail_dir="${EVAL_DETAIL_DIR:-logs}"
refine_token="${REFINE_TOKEN:-<REFINE>}"
refine_token_count="${REFINE_TOKEN_COUNT:-8}"
default_activitynet_root="$(videosearch_dataset_dir activitynet)"
query_embeddings_path="${QUERY_EMBEDDINGS_PATH:-${default_activitynet_root}/train/query_embedding/query_embeddings.train.npy}"
query_meta_path="${QUERY_META_PATH:-${default_activitynet_root}/train/query_embedding/query_meta.train.jsonl}"
video_embeddings_path="${VIDEO_EMBEDDINGS_PATH:-${default_activitynet_root}/train/video_embedding_1fps/segment_embeds.npy}"
video_docid2row_path="${VIDEO_DOCID2ROW_PATH:-${default_activitynet_root}/train/video_embedding_1fps/docid2row.json}"
hard_negatives_path="${HARD_NEGATIVES_PATH:-${default_activitynet_root}/train/hard_negatives.json}"
hard_negative_refresh_steps="${HARD_NEGATIVE_REFRESH_STEPS:-1}"
video_meta_path="${VIDEO_META_PATH:-${default_activitynet_root}/train/video_npy_with_meta/meta.jsonl}"
use_query_embedder_path="${USE_QUERY_EMBEDDER_PATH:-True}"
query_embedder_model_path="${QUERY_EMBEDDER_MODEL_PATH:-${VIDEOSEARCH_EMBED_MODEL}}"
qfinal_pooling="${QFINAL_POOLING:-latent_last}"
qfinal_normalize="${QFINAL_NORMALIZE:-True}"
tune_query_embedder="${TUNE_QUERY_EMBEDDER:-False}"
query_embedder_lr="${QUERY_EMBEDDER_LR:-5e-6}"
query_embedder_max_length="${QUERY_EMBEDDER_MAX_LENGTH:-128}"
query_embedder_input_prefix="${QUERY_EMBEDDER_INPUT_PREFIX:-}"
retrieval_debug="${DEBUG:-True}"
retrieval_debug_max_logs="${DEBUG_MAX_LOGS:-3000}"
retrieval_debug_jsonl="${DEBUG_JSONL:-logs/retrieval_debug_verification.jsonl}"

# Optional strict external test eval (verified_test + generate answer + refine rerank).
use_verified_test_eval="${USE_VERIFIED_TEST_EVAL:-False}"
external_eval_mode="${EXTERNAL_EVAL_MODE:-standard}"
external_eval_verified_test_jsonl="${EXTERNAL_EVAL_VERIFIED_TEST_JSONL:-${default_activitynet_root}/raw_annotation/test.jsonl}"
external_eval_video_root="${EXTERNAL_EVAL_VIDEO_ROOT:-${default_activitynet_root}/test/video_npy_with_meta}"
external_eval_video_meta_path="${EXTERNAL_EVAL_VIDEO_META_PATH:-}"
external_eval_query_embeddings_path="${EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH:-${default_activitynet_root}/test/query_embedding/query_embeddings.test.npy}"
external_eval_query_meta_path="${EXTERNAL_EVAL_QUERY_META_PATH:-${default_activitynet_root}/test/query_embedding/query_meta.test.jsonl}"
external_eval_video_embeddings_path="${EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH:-${default_activitynet_root}/test/video_embedding_1fps/segment_embeds.npy}"
external_eval_video_docid2row_path="${EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH:-${default_activitynet_root}/test/video_embedding_1fps/docid2row.json}"
external_eval_topk="${EXTERNAL_EVAL_TOPK:-1,5,10,100}"
external_eval_max_samples="${EXTERNAL_EVAL_MAX_SAMPLES:-0}"
external_eval_limit_ratio="${EXTERNAL_EVAL_LIMIT_RATIO:-1}"
external_eval_max_new_tokens="${EXTERNAL_EVAL_MAX_NEW_TOKENS:-128}"
external_eval_output_jsonl="${EXTERNAL_EVAL_OUTPUT_JSONL:-True}"

if [ "${use_verified_test_eval}" = "True" ]; then
  eval_strategy="${EVAL_STRATEGY:-no}"
  load_best_model_at_end=False
  eval_recall_on_eval=False
  external_eval_on_gpu0=True
  external_eval_mode="verified_test_rerank"
fi

# Optional GPU selection.
GPUS="${GPUS:-1,2,3}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
if [ -n "${GPUS}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  if [ -z "${NPROC_PER_NODE}" ]; then
    IFS=',' read -ra GPU_ARR <<< "${GPUS}"
    NPROC_PER_NODE="${#GPU_ARR[@]}"
  fi
fi
if [ -z "${NPROC_PER_NODE}" ]; then
  NPROC_PER_NODE=8
fi
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[verification] output_dir=${output_dir}"
echo "[verification] dataset_info=${dataset_info} dataset_name=${dataset_name[*]}"
echo "[verification] batch train/eval=${per_device_train_batch_size}/${per_device_eval_batch_size}, grad_acc=${gradient_accumulation_steps}, workers=${dataloader_num_workers}, grad_ckpt=${gradient_checkpointing}"
echo "[verification] enable_retrieval_optimization=${enable_retrieval_optimization}"
echo "[verification] retrieval_loss_weight=${retrieval_loss_weight}, projector_lr=${projector_lr}, negative_pool_size=${negative_pool_size}"
echo "[verification] retrieval_ignore_ambiguous_negatives=${retrieval_ignore_ambiguous_negatives}, ambiguous_negative_margin=${ambiguous_negative_margin}, strict_negative_topk=${strict_negative_topk}"
echo "[verification] use_refine_gate=${use_refine_gate}, zero_init_refine=${zero_init_refine}, only_neg=${only_neg}, tune_mm_llm/mlp/vision=${tune_mm_llm}/${tune_mm_mlp}/${tune_mm_vision}"
echo "[verification] eval_strategy=${eval_strategy}, eval_steps=${eval_steps}, eval_split_ratio=${eval_split_ratio}"
echo "[verification] save_only_model=${save_only_model}"
echo "[verification] eval_recall_on_eval=${eval_recall_on_eval}, eval_recall_ks=${eval_recall_ks}"
echo "[verification] use_verified_test_eval=${use_verified_test_eval}, external_eval_mode=${external_eval_mode}, external_eval_on_gpu0=${external_eval_on_gpu0}"
echo "[verification] use_query_embedder_path=${use_query_embedder_path}, query_embedder_model_path=${query_embedder_model_path}"
echo "[verification] hard_negative_refresh_steps=${hard_negative_refresh_steps}"
echo "[verification] query_embedder_input_prefix=${query_embedder_input_prefix:-none}"
echo "[verification] refine_token=${refine_token}, refine_token_count=${refine_token_count}"
echo "[verification] qfinal_pooling=${qfinal_pooling}, qfinal_normalize=${qfinal_normalize}, tune_query_embedder=${tune_query_embedder}"
echo "[verification] video_root_override=${video_root_override:-none}"
echo "[verification] video_meta_path=${video_meta_path:-none}, external_eval_video_meta_path=${external_eval_video_meta_path:-auto}"
echo "[verification] resume_from_checkpoint=${resume_from_checkpoint:-none}"

extra_args=(
  --per_device_train_batch_size "$per_device_train_batch_size"
  --per_device_eval_batch_size "$per_device_eval_batch_size"
  --gradient_accumulation_steps "$gradient_accumulation_steps"
  --num_train_epochs "$num_train_epochs"
  --learning_rate "$learning_rate"
  --lr_scheduler_type "$lr_scheduler_type"
  --warmup_ratio "$warmup_ratio"
  --warmup_steps "$warmup_steps"
  --weight_decay "$weight_decay"
  --max_grad_norm "$max_grad_norm"
  --optim "$optim"
  --adam_beta1 "$adam_beta1"
  --adam_beta2 "$adam_beta2"
  --adam_epsilon "$adam_epsilon"
  --gradient_checkpointing "$gradient_checkpointing"
  --ddp_find_unused_parameters "$ddp_find_unused_parameters"
  --max_length "$max_length"
  --dataloader_num_workers "$dataloader_num_workers"
  --dataloader_pin_memory "$dataloader_pin_memory"
  --seed "$seed"
  --bf16 "$bf16"
  --fp16 "$fp16"
  --tf32 "$tf32"
  --save_strategy "$save_strategy"
  --save_steps "$save_steps"
  --save_total_limit "$save_total_limit"
  --save_only_model "$save_only_model"
  --logging_steps "$logging_steps"
  --eval_strategy "$eval_strategy"
  --eval_steps "$eval_steps"
  --load_best_model_at_end "$load_best_model_at_end"
  --metric_for_best_model "$metric_for_best_model"
  --greater_is_better "$greater_is_better"
  --report_to "$report_to"
  --logging_dir "$logging_dir"
  --enable_retrieval_optimization "$enable_retrieval_optimization"
  --retrieval_loss_weight "$retrieval_loss_weight"
  --retrieval_temperature "$retrieval_temperature"
  --negative_pool_size "$negative_pool_size"
  --retrieval_ignore_ambiguous_negatives "$retrieval_ignore_ambiguous_negatives"
  --ambiguous_negative_margin "$ambiguous_negative_margin"
  --strict_negative_topk "$strict_negative_topk"
  --projector_lr "$projector_lr"
  --use_refine_gate "$use_refine_gate"
  --zero_init_refine "$zero_init_refine"
  --retrieval_on_eval "$retrieval_on_eval"
  --only_neg "$only_neg"
  --external_eval_on_gpu0 "$external_eval_on_gpu0"
  --external_eval_gpu "$external_eval_gpu"
  --external_eval_mode "$external_eval_mode"
  --external_eval_verified_test_jsonl "$external_eval_verified_test_jsonl"
  --external_eval_video_root "$external_eval_video_root"
  --external_eval_video_meta_path "$external_eval_video_meta_path"
  --external_eval_query_embeddings_path "$external_eval_query_embeddings_path"
  --external_eval_query_meta_path "$external_eval_query_meta_path"
  --external_eval_video_embeddings_path "$external_eval_video_embeddings_path"
  --external_eval_video_docid2row_path "$external_eval_video_docid2row_path"
  --external_eval_topk "$external_eval_topk"
  --external_eval_max_samples "$external_eval_max_samples"
  --external_eval_limit_ratio "$external_eval_limit_ratio"
  --external_eval_max_new_tokens "$external_eval_max_new_tokens"
  --external_eval_output_jsonl "$external_eval_output_jsonl"
  --eval_split_ratio "$eval_split_ratio"
  --eval_recall_on_eval "$eval_recall_on_eval"
  --eval_recall_ks "$eval_recall_ks"
  --eval_detail_dir "$eval_detail_dir"
  --refine_token "$refine_token"
  --refine_token_count "$refine_token_count"
  --query_embeddings_path "$query_embeddings_path"
  --query_meta_path "$query_meta_path"
  --video_embeddings_path "$video_embeddings_path"
  --video_docid2row_path "$video_docid2row_path"
  --hard_negatives_path "$hard_negatives_path"
  --hard_negative_refresh_steps "$hard_negative_refresh_steps"
  --use_query_embedder_path "$use_query_embedder_path"
  --query_embedder_model_path "$query_embedder_model_path"
  --qfinal_pooling "$qfinal_pooling"
  --qfinal_normalize "$qfinal_normalize"
  --tune_query_embedder "$tune_query_embedder"
  --query_embedder_lr "$query_embedder_lr"
  --query_embedder_max_length "$query_embedder_max_length"
  --retrieval_debug "$retrieval_debug"
  --retrieval_debug_max_logs "$retrieval_debug_max_logs"
  --retrieval_debug_jsonl "$retrieval_debug_jsonl"
)

if [ -n "$run_name" ]; then
  extra_args+=(--run_name "$run_name")
fi
if [ -n "$deepspeed" ]; then
  extra_args+=(--deepspeed "$deepspeed")
fi
if [ -n "$max_samples" ]; then
  extra_args+=(--max_samples "$max_samples")
fi
if [ -n "$max_steps" ]; then
  extra_args+=(--max_steps "$max_steps")
fi
if [ -n "$resume_from_checkpoint" ]; then
  extra_args+=(--resume_from_checkpoint "$resume_from_checkpoint")
fi

cd "${repo_root}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --master_port="${MASTER_PORT}" \
  "${repo_root}/videosearch_r1/train_sft_qwen3_vl_share_gpt.py" \
  --model_path "$model_path" \
  --output_dir "$output_dir" \
  --dataset_info "$dataset_info" \
  --dataset_name "${dataset_name[@]}" \
  --image_min_pixels "$image_min_pixels" \
  --image_max_pixels "$image_max_pixels" \
  --video_min_pixels "$video_min_pixels" \
  --video_max_pixels "$video_max_pixels" \
  --video_total_pixels "$video_total_pixels" \
  --max_frames "$max_frames" \
  --fps "$fps" \
  --video_root_override "$video_root_override" \
  --video_meta_path "$video_meta_path" \
  --model_max_length "$model_max_length" \
  --tune_mm_llm "$tune_mm_llm" \
  --tune_mm_mlp "$tune_mm_mlp" \
  --tune_mm_vision "$tune_mm_vision" \
  "${extra_args[@]}"
