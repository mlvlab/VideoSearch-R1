#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash test_checkpoint.bash /path/to/checkpoint-XXX
  CHECKPOINT_PATH=/path/to/checkpoint-XXX bash test_checkpoint.bash
  DEFAULT_CHECKPOINT=didemo-stage2 bash test_checkpoint.bash

Env knobs:
  EVAL_GPU=0
  RERANKS=all            # e.g. all,10,5
  RESUME_FROM_JSONL=True # continue from existing jsonl if present
  QUERY_EMBEDDER_MODEL_PATH=...  # override query embedder model path
  EXTERNAL_EVAL_VIDEO_META_PATH=/path/to/meta.jsonl  # optional; only used for npy/npz mode
  REFINE_TOKEN_COUNT=0      # 0=auto-detect, 1=single, N=multi (<REFINE_1>...<REFINE_N>)
  TOPK=1,5,10,100
  EVAL_LIMIT_RATIO=1.0
  MAX_SAMPLES=0

Examples:
  RERANKS=all bash test_checkpoint.bash /hub_data2/.../checkpoint-115
  RERANKS=all,10,5 EVAL_GPU=2 bash test_checkpoint.bash /hub_data2/.../checkpoint-115
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

checkpoint_path="${1:-${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT:-}}}"
if [[ -z "${checkpoint_path}" ]]; then
  echo "checkpoint path is required. Pass an argument, CHECKPOINT_PATH, or DEFAULT_CHECKPOINT." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source "${repo_root}/scripts/common/env.bash"
if [[ ! -d "${checkpoint_path}" ]]; then
  checkpoint_path="$(videosearch_local_model_path "${checkpoint_path}")"
fi
if [[ ! -d "${checkpoint_path}" ]]; then
  echo "checkpoint dir not found: ${checkpoint_path}" >&2
  exit 1
fi
eval_script="${repo_root}/scripts/internal/stage1/eval_verified_test_rerank.py"
if [[ ! -f "${eval_script}" ]]; then
  echo "eval script not found: ${eval_script}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"

eval_gpu="${EVAL_GPU:-3}"
reranks="${RERANKS:-all}"
resume_from_jsonl="${RESUME_FROM_JSONL:-True}"

override_logs_dir="${LOGS_DIR:-}"
run_dir="$(dirname "${checkpoint_path}")"
if [[ -n "${override_logs_dir}" ]]; then
  logs_dir="${override_logs_dir}"
else
  logs_dir="${run_dir}/logs"
fi
mkdir -p "${logs_dir}"

# Defaults aligned with verification_setting.bash
default_activitynet_root="$(videosearch_dataset_dir activitynet)"
verified_test_jsonl="${EXTERNAL_EVAL_VERIFIED_TEST_JSONL:-${default_activitynet_root}/raw_annotation/test.jsonl}"
video_root="${EXTERNAL_EVAL_VIDEO_ROOT:-${default_activitynet_root}/test/video_npy_with_meta}"
video_meta_path="${EXTERNAL_EVAL_VIDEO_META_PATH:-${default_activitynet_root}/test/video_npy_with_meta/meta.jsonl}"
query_embeddings_path="${EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH:-${default_activitynet_root}/test/query_embedding/query_embeddings.test.npy}"
query_meta_path="${EXTERNAL_EVAL_QUERY_META_PATH:-${default_activitynet_root}/test/query_embedding/query_meta.test.jsonl}"
video_embeddings_path="${EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH:-${default_activitynet_root}/test/video_embedding_1fps/segment_embeds.npy}"
video_docid2row_path="${EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH:-${default_activitynet_root}/test/video_embedding_1fps/docid2row.json}"

refine_token="${REFINE_TOKEN:-<REFINE>}"
refine_token_count="${REFINE_TOKEN_COUNT:-0}"
use_refine_gate="${USE_REFINE_GATE:-False}"
use_query_embedder_path="${USE_QUERY_EMBEDDER_PATH:-True}"
query_embedder_model_path="${QUERY_EMBEDDER_MODEL_PATH:-Qwen/Qwen3-VL-Embedding-2B}"
qfinal_pooling="${QFINAL_POOLING:-latent_last}"
qfinal_normalize="${QFINAL_NORMALIZE:-True}"
query_embedder_max_length="${QUERY_EMBEDDER_MAX_LENGTH:-128}"
model_max_length="${MODEL_MAX_LENGTH:-46384}"
max_new_tokens="${EXTERNAL_EVAL_MAX_NEW_TOKENS:-128}"
topk="${TOPK:-1,5,10,100}"
max_samples="${MAX_SAMPLES:-0}"
eval_limit_ratio="${EVAL_LIMIT_RATIO:-1.0}"
seed="${SEED:-42}"
bf16="${BF16:-True}"

ckpt_name="$(basename "${checkpoint_path}")"
checkpoint_query_embedder_dir="${checkpoint_path}/query_embedder"
if [[ -d "${checkpoint_query_embedder_dir}" ]]; then
  query_embedder_model_path="${checkpoint_query_embedder_dir}"
  echo "[test_checkpoint] detected tuned query embedder: ${query_embedder_model_path}"
else
  echo "[test_checkpoint] query embedder path: ${query_embedder_model_path}"
fi

echo "[test_checkpoint] checkpoint=${checkpoint_path}"
echo "[test_checkpoint] logs_dir=${logs_dir}"
echo "[test_checkpoint] eval_gpu=${eval_gpu} reranks=${reranks} resume_from_jsonl=${resume_from_jsonl}"
echo "[test_checkpoint] video_meta_path=${video_meta_path:-auto}"

IFS=',' read -ra RK_ARR <<< "${reranks}"
for raw_rk in "${RK_ARR[@]}"; do
  rk="$(echo "${raw_rk}" | xargs)"
  if [[ -z "${rk}" ]]; then
    continue
  fi

  suffix=""
  rk_lc="$(echo "${rk}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${rk_lc}" != "all" ]]; then
    suffix="_rerank${rk}"
  fi

  output_json="${logs_dir}/external_verified_test_${ckpt_name}${suffix}.json"
  output_jsonl="${logs_dir}/external_verified_test_${ckpt_name}${suffix}.jsonl"

  echo "[test_checkpoint] rerank_topk=${rk} ->"
  echo "  json=${output_json}"
  echo "  jsonl=${output_jsonl}"

  CUDA_VISIBLE_DEVICES="${eval_gpu}" "${python_bin}" "${eval_script}" \
    --model_path "${checkpoint_path}" \
    --processor_path "${checkpoint_path}" \
    --verified_test_jsonl "${verified_test_jsonl}" \
    --video_root "${video_root}" \
    --video_meta_path "${video_meta_path}" \
    --query_embeddings_path "${query_embeddings_path}" \
    --query_meta_path "${query_meta_path}" \
    --video_embeddings_path "${video_embeddings_path}" \
    --video_docid2row_path "${video_docid2row_path}" \
    --refine_token "${refine_token}" \
    --refine_token_count "${refine_token_count}" \
    --use_refine_gate "${use_refine_gate}" \
    --use_query_embedder_path "${use_query_embedder_path}" \
    --query_embedder_model_path "${query_embedder_model_path}" \
    --qfinal_pooling "${qfinal_pooling}" \
    --qfinal_normalize "${qfinal_normalize}" \
    --query_embedder_max_length "${query_embedder_max_length}" \
    --model_max_length "${model_max_length}" \
    --max_new_tokens "${max_new_tokens}" \
    --topk "${topk}" \
    --rerank_topk "${rk}" \
    --max_samples "${max_samples}" \
    --eval_limit_ratio "${eval_limit_ratio}" \
    --seed "${seed}" \
    --bf16 "${bf16}" \
    --output_json "${output_json}" \
    --output_jsonl "${output_jsonl}" \
    --resume_from_jsonl "${resume_from_jsonl}"
done
