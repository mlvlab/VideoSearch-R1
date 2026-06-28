#!/usr/bin/env bash
set -euo pipefail

# EVAL_GPUS=1,2 NUM_PROCESSES_PER_GPU=1,1 USE_VLLM=True VLLM_GPU_MEMORY_UTILIZATION=0.25 VLLM_MAX_NUM_SEQS=8 VLLM_EVAL_BATCH_SIZE=8 VLLM_ENFORCE_EAGER=True bash ./scripts/inference/inference.bash didemo

usage() {
  cat <<'EOF'
Usage:
  bash test_checkpoint_temporal_grounding.bash /path/to/checkpoint-XXX
  CHECKPOINT_PATH=/path/to/checkpoint-XXX bash test_checkpoint_temporal_grounding.bash
  DEFAULT_CHECKPOINT=didemo-stage2 bash test_checkpoint_temporal_grounding.bash
  bash test_checkpoint_temporal_grounding.bash /path/to/checkpoint-XXX --use_updated_query

Env knobs:
  EVAL_GPU=0
  EVAL_GPUS=0            # optional CSV list, e.g. 0,2
  NUM_PROCESSES_PER_GPU=1 # CSV counts aligned with EVAL_GPUS, e.g. 2,3
  RERANKS=all            # temporal eval currently supports only 'all'
  RESUME_FROM_JSONL=True
  QUERY_EMBEDDER_MODEL_PATH=...  # override query embedder model path
  QUERY_EMBEDDER_INPUT_PREFIX="Represent the user's input"
  EXTERNAL_EVAL_VIDEO_META_PATH=/path/to/meta.jsonl  # optional; only used for npy/npz mode
  REFINE_TOKEN_COUNT=0      # 0=auto-detect, 1=single, N=multi (<REFINE_1>...<REFINE_N>)
  TOPK=1,5,10,100
  EVAL_LIMIT_RATIO=1.0
  MAX_SAMPLES=0
  PROGRESS_INTERVAL_SEC=2.0
  USE_TQDM=True
  TQDM_MININTERVAL=0.2
  USE_VLLM=True
  VLLM_MODEL_PATH=      # optional; if empty, eval auto-builds _vllm_compat from checkpoint
  VLLM_TENSOR_PARALLEL_SIZE=1
  VLLM_GPU_MEMORY_UTILIZATION=0.85
  VLLM_MAX_NUM_SEQS=1
  VLLM_EVAL_BATCH_SIZE=0
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=False
  VLLM_ENFORCE_EAGER=False
  VLLM_BASE_PORT=39000
  VLLM_PORT_STRIDE=100
  SAVE_QUERY_VECTORS=False
  QUERY_VECTORS_DIR=/path/to/query_vectors
  SAVE_QUERY_VECTORS_MAX_GB=200
  SAVE_QUERY_VECTORS_ONLY_REFINED=False
  PROFILE_ROLLOUT_DEPTHS=1,2,4,8,16

Temporal knobs:
  TEMPORAL=True
  MAX_TURN=2
  IOU_THRESHOLDS=0.3,0.5,0.7
  USE_UPDATED_QUERY=False # if True, re-embed initial test queries with query_embedder_model_path
  # Turn semantics in temporal eval:
  # one turn = VLM 1x + (if not_matched and refine token exists) rerank 1x

Examples:
  MAX_TURN=2 bash test_checkpoint_temporal_grounding.bash /hub_data2/.../checkpoint-115
  EVAL_GPU=2 MAX_TURN=3 RESUME_FROM_JSONL=True bash test_checkpoint_temporal_grounding.bash /hub_data2/.../checkpoint-115
  bash test_checkpoint_temporal_grounding.bash /hub_data2/.../checkpoint-115 --use_updated_query
EOF
}

use_updated_query="${USE_UPDATED_QUERY:-False}"
checkpoint_path="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT:-}}"
cli_checkpoint_path=""
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --use_updated_query)
      use_updated_query="True"
      ;;
    --no-use_updated_query)
      use_updated_query="False"
      ;;
    --*)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "${cli_checkpoint_path}" ]]; then
        cli_checkpoint_path="${arg}"
      else
        echo "Unexpected positional argument: ${arg}" >&2
        usage
        exit 1
      fi
      ;;
  esac
done
if [[ -n "${cli_checkpoint_path}" ]]; then
  checkpoint_path="${cli_checkpoint_path}"
fi
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
eval_script="${repo_root}/scripts/internal/stage1/eval_verified_test_rerank_temporal_grounding.py"
if [[ ! -f "${eval_script}" ]]; then
  echo "eval script not found: ${eval_script}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"

eval_gpu="${EVAL_GPU:-0}"
eval_gpus="${EVAL_GPUS:-${eval_gpu}}"
num_processes_per_gpu="${NUM_PROCESSES_PER_GPU:-1}"
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

default_activitynet_root="$(videosearch_dataset_dir activitynet)"
verified_test_jsonl="${EXTERNAL_EVAL_VERIFIED_TEST_JSONL:-${default_activitynet_root}/raw_annotation/test.jsonl}"
video_root="${EXTERNAL_EVAL_VIDEO_ROOT:-${default_activitynet_root}/test/video_npy_with_meta}"
video_meta_path="${EXTERNAL_EVAL_VIDEO_META_PATH:-${default_activitynet_root}/test/video_npy_with_meta/meta.jsonl}"
query_embeddings_path="${EXTERNAL_EVAL_QUERY_EMBEDDINGS_PATH:-${default_activitynet_root}/test/query_embedding/query_embeddings.test.npy}"
query_meta_path="${EXTERNAL_EVAL_QUERY_META_PATH:-${default_activitynet_root}/test/query_embedding/query_meta.test.jsonl}"
video_embeddings_path="${EXTERNAL_EVAL_VIDEO_EMBEDDINGS_PATH:-${default_activitynet_root}/test/video_embedding_1fps/segment_embeds.npy}"
video_docid2row_path="${EXTERNAL_EVAL_VIDEO_DOCID2ROW_PATH:-${default_activitynet_root}/test/video_embedding_1fps/docid2row.json}"

temporal="${TEMPORAL:-True}"
max_turn="${MAX_TURN:-3}"
iou_thresholds="${IOU_THRESHOLDS:-0.3,0.5,0.7}"

refine_token="${REFINE_TOKEN:-<REFINE>}"
refine_token_count="${REFINE_TOKEN_COUNT:-8}"
use_refine_gate="${USE_REFINE_GATE:-False}"
use_query_embedder_path="${USE_QUERY_EMBEDDER_PATH:-True}"
query_embedder_model_path="${QUERY_EMBEDDER_MODEL_PATH:-${VIDEOSEARCH_EMBED_MODEL}}"
qfinal_pooling="${QFINAL_POOLING:-latent_last}"
qfinal_normalize="${QFINAL_NORMALIZE:-True}"
query_embedder_max_length="${QUERY_EMBEDDER_MAX_LENGTH:-128}"
query_embedder_input_prefix="${QUERY_EMBEDDER_INPUT_PREFIX:-}"
model_max_length="${MODEL_MAX_LENGTH:-20384}"
max_new_tokens="${EXTERNAL_EVAL_MAX_NEW_TOKENS:-1024}"
topk="${TOPK:-1,5,10,100}"
max_samples="${MAX_SAMPLES:-0}"
eval_limit_ratio="${EVAL_LIMIT_RATIO:-1.0}"
progress_interval_sec="${PROGRESS_INTERVAL_SEC:-2.0}"
use_tqdm="${USE_TQDM:-True}"
tqdm_mininterval="${TQDM_MININTERVAL:-0.2}"
seed="${SEED:-42}"
bf16="${BF16:-True}"
use_vllm="${USE_VLLM:-True}"
vllm_model_path="${VLLM_MODEL_PATH:-}"
vllm_tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-1}"
vllm_eval_batch_size="${VLLM_EVAL_BATCH_SIZE:-0}"
vllm_disable_custom_all_reduce="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-False}"
vllm_enforce_eager="${VLLM_ENFORCE_EAGER:-False}"
vllm_base_port="${VLLM_BASE_PORT:-39000}"
vllm_port_stride="${VLLM_PORT_STRIDE:-100}"
save_query_vectors="${SAVE_QUERY_VECTORS:-False}"
query_vectors_dir="${QUERY_VECTORS_DIR:-${logs_dir}/query_vectors}"
save_query_vectors_max_gb="${SAVE_QUERY_VECTORS_MAX_GB:-200}"
save_query_vectors_only_refined="${SAVE_QUERY_VECTORS_ONLY_REFINED:-False}"
profile_rollout_depths="${PROFILE_ROLLOUT_DEPTHS:-}"

IFS=',' read -ra GPU_RAW_ARR <<< "${eval_gpus}"
gpu_arr=()
for raw in "${GPU_RAW_ARR[@]}"; do
  g="$(echo "${raw}" | xargs)"
  if [[ -n "${g}" ]]; then
    gpu_arr+=("${g}")
  fi
done
if [[ ${#gpu_arr[@]} -eq 0 ]]; then
  echo "No valid GPU id found from EVAL_GPUS='${eval_gpus}'" >&2
  exit 1
fi

IFS=',' read -ra PROC_RAW_ARR <<< "${num_processes_per_gpu}"
proc_arr=()
for raw in "${PROC_RAW_ARR[@]}"; do
  p="$(echo "${raw}" | xargs)"
  if [[ -n "${p}" ]]; then
    proc_arr+=("${p}")
  fi
done
if [[ ${#proc_arr[@]} -eq 0 ]]; then
  echo "No valid process count found from NUM_PROCESSES_PER_GPU='${num_processes_per_gpu}'" >&2
  exit 1
fi
if [[ ${#proc_arr[@]} -eq 1 && ${#gpu_arr[@]} -gt 1 ]]; then
  single_p="${proc_arr[0]}"
  proc_arr=()
  for _ in "${gpu_arr[@]}"; do
    proc_arr+=("${single_p}")
  done
fi
if [[ ${#proc_arr[@]} -ne ${#gpu_arr[@]} ]]; then
  echo "NUM_PROCESSES_PER_GPU count (${#proc_arr[@]}) must match EVAL_GPUS count (${#gpu_arr[@]})" >&2
  exit 1
fi

total_workers=0
for p in "${proc_arr[@]}"; do
  if ! [[ "${p}" =~ ^[0-9]+$ ]] || [[ "${p}" -lt 1 ]]; then
    echo "Invalid process count '${p}' in NUM_PROCESSES_PER_GPU; must be positive integer." >&2
    exit 1
  fi
  total_workers=$((total_workers + p))
done
if [[ "${total_workers}" -lt 1 ]]; then
  echo "total_workers must be >=1" >&2
  exit 1
fi

if ! [[ "${vllm_base_port}" =~ ^[0-9]+$ ]] || [[ "${vllm_base_port}" -lt 1024 ]] || [[ "${vllm_base_port}" -gt 65535 ]]; then
  echo "Invalid VLLM_BASE_PORT='${vllm_base_port}'. Expected integer in [1024, 65535]." >&2
  exit 1
fi
if ! [[ "${vllm_port_stride}" =~ ^[0-9]+$ ]] || [[ "${vllm_port_stride}" -lt 1 ]]; then
  echo "Invalid VLLM_PORT_STRIDE='${vllm_port_stride}'. Expected integer >= 1." >&2
  exit 1
fi
max_vllm_port="$((vllm_base_port + (total_workers - 1) * vllm_port_stride))"
if [[ "${max_vllm_port}" -gt 65535 ]]; then
  echo "VLLM_BASE_PORT (${vllm_base_port}) + workers/stride exceeds 65535. Adjust VLLM_BASE_PORT or VLLM_PORT_STRIDE." >&2
  exit 1
fi

save_query_vectors_max_gb_per_worker="${save_query_vectors_max_gb}"
if [[ "${total_workers}" -gt 1 ]]; then
  save_query_vectors_max_gb_per_worker="$(
    python - "${save_query_vectors_max_gb}" "${total_workers}" <<'PY'
import sys
cap = float(sys.argv[1])
n = int(sys.argv[2])
if cap <= 0 or n <= 1:
    print(cap)
else:
    print(cap / n)
PY
  )"
fi

ckpt_name="$(basename "${checkpoint_path}")"
checkpoint_query_embedder_dir="${checkpoint_path}/query_embedder"
if [[ -d "${checkpoint_query_embedder_dir}" ]]; then
  query_embedder_model_path="${checkpoint_query_embedder_dir}"
  echo "[test_checkpoint_temporal_grounding] detected tuned query embedder: ${query_embedder_model_path}"
else
  echo "[test_checkpoint_temporal_grounding] query embedder path: ${query_embedder_model_path}"
fi

echo "[test_checkpoint_temporal_grounding] checkpoint=${checkpoint_path}"
echo "[test_checkpoint_temporal_grounding] logs_dir=${logs_dir}"
echo "[test_checkpoint_temporal_grounding] eval_gpus=${eval_gpus} num_processes_per_gpu=${num_processes_per_gpu} total_workers=${total_workers}"
echo "[test_checkpoint_temporal_grounding] reranks=${reranks} resume_from_jsonl=${resume_from_jsonl}"
echo "[test_checkpoint_temporal_grounding] temporal=${temporal} max_turn=${max_turn} iou_thresholds=${iou_thresholds}"
echo "[test_checkpoint_temporal_grounding] progress_interval_sec=${progress_interval_sec}"
echo "[test_checkpoint_temporal_grounding] use_tqdm=${use_tqdm} tqdm_mininterval=${tqdm_mininterval}"
echo "[test_checkpoint_temporal_grounding] use_vllm=${use_vllm} tp=${vllm_tensor_parallel_size} mem_util=${vllm_gpu_memory_utilization} max_num_seqs=${vllm_max_num_seqs}"
echo "[test_checkpoint_temporal_grounding] vllm_eval_batch_size=${vllm_eval_batch_size}"
echo "[test_checkpoint_temporal_grounding] vllm_model_path=${vllm_model_path:-auto(_vllm_compat)}"
echo "[test_checkpoint_temporal_grounding] vllm_base_port=${vllm_base_port} vllm_port_stride=${vllm_port_stride}"
echo "[test_checkpoint_temporal_grounding] use_updated_query=${use_updated_query}"
echo "[test_checkpoint_temporal_grounding] query_embedder_input_prefix=${query_embedder_input_prefix:-none}"
echo "[test_checkpoint_temporal_grounding] video_meta_path=${video_meta_path:-auto}"
echo "[test_checkpoint_temporal_grounding] save_query_vectors=${save_query_vectors}"
echo "[test_checkpoint_temporal_grounding] query_vectors_dir=${query_vectors_dir}"
echo "[test_checkpoint_temporal_grounding] save_query_vectors_max_gb(global)=${save_query_vectors_max_gb}"
echo "[test_checkpoint_temporal_grounding] save_query_vectors_max_gb(per_worker)=${save_query_vectors_max_gb_per_worker}"
echo "[test_checkpoint_temporal_grounding] save_query_vectors_only_refined=${save_query_vectors_only_refined}"
echo "[test_checkpoint_temporal_grounding] profile_rollout_depths=${profile_rollout_depths:-none}"

is_true() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${v}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_eval_worker() {
  local gpu="$1"
  local output_json="$2"
  local output_jsonl="$3"
  local num_shards="$4"
  local shard_id="$5"
  local resume_skip_jsonl="$6"
  local worker_vllm_port
  worker_vllm_port="$((vllm_base_port + shard_id * vllm_port_stride))"

  CUDA_VISIBLE_DEVICES="${gpu}" VLLM_PORT="${worker_vllm_port}" PYTHONUNBUFFERED=1 "${python_bin}" "${eval_script}" \
    --model_path "${checkpoint_path}" \
    --processor_path "${checkpoint_path}" \
    --verified_test_jsonl "${verified_test_jsonl}" \
    --video_root "${video_root}" \
    --video_meta_path "${video_meta_path}" \
    --query_embeddings_path "${query_embeddings_path}" \
    --query_meta_path "${query_meta_path}" \
    --video_embeddings_path "${video_embeddings_path}" \
    --video_docid2row_path "${video_docid2row_path}" \
    --temporal "${temporal}" \
    --max_turn "${max_turn}" \
    --iou_thresholds "${iou_thresholds}" \
    --refine_token "${refine_token}" \
    --refine_token_count "${refine_token_count}" \
    --use_refine_gate "${use_refine_gate}" \
    --use_query_embedder_path "${use_query_embedder_path}" \
    --query_embedder_model_path "${query_embedder_model_path}" \
    --use_updated_query "${use_updated_query}" \
    --qfinal_pooling "${qfinal_pooling}" \
    --qfinal_normalize "${qfinal_normalize}" \
    --query_embedder_max_length "${query_embedder_max_length}" \
    --query_embedder_input_prefix "${query_embedder_input_prefix}" \
    --model_max_length "${model_max_length}" \
    --max_new_tokens "${max_new_tokens}" \
    --use_vllm "${use_vllm}" \
    --vllm_model_path "${vllm_model_path}" \
    --vllm_tensor_parallel_size "${vllm_tensor_parallel_size}" \
    --vllm_gpu_memory_utilization "${vllm_gpu_memory_utilization}" \
    --vllm_max_num_seqs "${vllm_max_num_seqs}" \
    --vllm_eval_batch_size "${vllm_eval_batch_size}" \
    --vllm_disable_custom_all_reduce "${vllm_disable_custom_all_reduce}" \
    --vllm_enforce_eager "${vllm_enforce_eager}" \
    --topk "${topk}" \
    --rerank_topk "${rk}" \
    --max_samples "${max_samples}" \
    --eval_limit_ratio "${eval_limit_ratio}" \
    --num_shards "${num_shards}" \
    --shard_id "${shard_id}" \
    --resume_skip_jsonl "${resume_skip_jsonl}" \
    --progress_interval_sec "${progress_interval_sec}" \
    --use_tqdm "${use_tqdm}" \
    --tqdm_mininterval "${tqdm_mininterval}" \
    --seed "${seed}" \
    --bf16 "${bf16}" \
    --output_json "${output_json}" \
    --output_jsonl "${output_jsonl}" \
    --save_query_vectors "${save_query_vectors}" \
    --save_query_vectors_dir "${query_vectors_dir}" \
    --save_query_vectors_max_gb "${save_query_vectors_max_gb_per_worker}" \
    --save_query_vectors_only_refined "${save_query_vectors_only_refined}" \
    --profile_rollout_depths "${profile_rollout_depths}" \
    --resume_from_jsonl "${resume_from_jsonl}"
}

merge_jsonl_sources_into_base() {
  local base_jsonl="$1"
  local shard_glob="$2"
  local tag="${3:-merge}"

  local -a merge_sources=()
  local -A seen_merge_sources=()
  if [[ -f "${base_jsonl}" ]]; then
    merge_sources+=("${base_jsonl}")
    seen_merge_sources["${base_jsonl}"]=1
  fi

  shopt -s nullglob
  for f in ${shard_glob}; do
    if [[ -f "${f}" && -z "${seen_merge_sources["$f"]:-}" ]]; then
      merge_sources+=("${f}")
      seen_merge_sources["${f}"]=1
    fi
  done
  shopt -u nullglob

  if [[ ${#merge_sources[@]} -eq 0 ]]; then
    return 0
  fi

  local merge_sources_csv
  merge_sources_csv="$(IFS=,; echo "${merge_sources[*]}")"
  local check_jsonl="${base_jsonl%.jsonl}.check.jsonl"
  MERGE_SOURCES_CSV="${merge_sources_csv}" MERGED_JSONL_PATH="${base_jsonl}" CHECK_JSONL_PATH="${check_jsonl}" python - <<'PY'
import json
import os

sources = [p.strip() for p in os.environ.get("MERGE_SOURCES_CSV", "").split(",") if p.strip()]
dst = os.environ["MERGED_JSONL_PATH"]
check_path = os.environ.get("CHECK_JSONL_PATH", "").strip()

by_index = {}
fallback_rows = []
for path in sources:
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            idx = row.get("index")
            if isinstance(idx, int):
                by_index[int(idx)] = row
            else:
                fallback_rows.append(row)

def write_rows(path):
    with open(path, "w", encoding="utf-8") as f:
        for idx in sorted(by_index.keys()):
            f.write(json.dumps(by_index[idx], ensure_ascii=False) + "\n")
        for row in fallback_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_rows(dst)
if check_path:
    write_rows(check_path)
PY
  echo "[test_checkpoint_temporal_grounding] ${tag} merged jsonl: ${base_jsonl} (sources=${#merge_sources[@]}, check=${check_jsonl})"
}

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

  output_json="${logs_dir}/external_verified_test_temporal_grounding_${ckpt_name}${suffix}.json"
  output_jsonl="${logs_dir}/external_verified_test_temporal_grounding_${ckpt_name}${suffix}.jsonl"

  echo "[test_checkpoint_temporal_grounding] rerank_topk=${rk} ->"
  echo "  json=${output_json}"
  echo "  jsonl=${output_jsonl}"

  if is_true "${resume_from_jsonl}"; then
    merge_jsonl_sources_into_base "${output_jsonl}" "${output_jsonl%.jsonl}.shard*.jsonl" "prestart"
  fi

  resume_skip_jsonl=""
  if is_true "${resume_from_jsonl}"; then
    declare -A seen_resume_paths=()
    resume_paths=()
    if [[ -f "${output_jsonl}" ]]; then
      resume_paths+=("${output_jsonl}")
      seen_resume_paths["${output_jsonl}"]=1
    fi
    shopt -s nullglob
    for f in "${output_jsonl%.jsonl}".shard*.jsonl; do
      if [[ -f "${f}" && -z "${seen_resume_paths["$f"]:-}" ]]; then
        resume_paths+=("${f}")
        seen_resume_paths["${f}"]=1
      fi
    done
    shopt -u nullglob
    if [[ ${#resume_paths[@]} -gt 0 ]]; then
      resume_skip_jsonl="$(IFS=,; echo "${resume_paths[*]}")"
    fi
  fi

  if [[ "${total_workers}" -eq 1 ]]; then
    run_eval_worker "${gpu_arr[0]}" "${output_json}" "${output_jsonl}" 1 0 "${resume_skip_jsonl}"
    continue
  fi

  echo "[test_checkpoint_temporal_grounding] launching ${total_workers} workers (sharded)"
  pids=()
  worker_shard_id=0
  for gpu_idx in "${!gpu_arr[@]}"; do
    gpu="${gpu_arr[$gpu_idx]}"
    nproc="${proc_arr[$gpu_idx]}"
    for ((local_i=0; local_i<nproc; local_i++)); do
      shard_id="${worker_shard_id}"
      shard_output_json="${output_json%.json}.shard${shard_id}of${total_workers}.json"
      shard_output_jsonl="${output_jsonl%.jsonl}.shard${shard_id}of${total_workers}.jsonl"
      shard_log="${output_jsonl%.jsonl}.shard${shard_id}of${total_workers}.log"
      echo "  worker shard=${shard_id}/${total_workers} gpu=${gpu} -> ${shard_output_jsonl}"
      run_eval_worker "${gpu}" "${shard_output_json}" "${shard_output_jsonl}" "${total_workers}" "${shard_id}" "${resume_skip_jsonl}" \
        > "${shard_log}" 2>&1 &
      pids+=($!)
      worker_shard_id=$((worker_shard_id + 1))
    done
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  merge_jsonl_sources_into_base "${output_jsonl}" "${output_jsonl%.jsonl}.shard*.jsonl" "postrun"
  if [[ "${failed}" -ne 0 ]]; then
    echo "[test_checkpoint_temporal_grounding] one or more shard workers failed (partial progress merged)." >&2
    exit 1
  fi
done
