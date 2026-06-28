#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir didemo)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
STRUCTURED_ROOT="${STRUCTURED_ROOT:-${DATASET_ROOT}}"
SPLIT="${SPLIT:-test}"

OUT_SUFFIX="${OUT_SUFFIX:-1fps}"
EMBED_SUBDIR="${EMBED_SUBDIR:-video_embedding_${OUT_SUFFIX}}"

TOPKS="${TOPKS:-1,5,10,100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
NORMALIZE="${NORMALIZE:-1}"

QUERY_EMB="${QUERY_EMB:-${STRUCTURED_ROOT}/${SPLIT}/query_embedding/query_embeddings.${SPLIT}.npy}"
QUERY_META="${QUERY_META:-${STRUCTURED_ROOT}/${SPLIT}/query_embedding/query_meta.${SPLIT}.jsonl}"
VIDEO_EMB="${VIDEO_EMB:-${STRUCTURED_ROOT}/${SPLIT}/${EMBED_SUBDIR}/segment_embeds.npy}"
VIDEO_DOCID2ROW="${VIDEO_DOCID2ROW:-${STRUCTURED_ROOT}/${SPLIT}/${EMBED_SUBDIR}/docid2row.json}"
QUERIES_JSONL="${QUERIES_JSONL:-${STRUCTURED_ROOT}/raw_annotation/${SPLIT}.jsonl}"
OUT_PATH="${OUT_PATH:-${STRUCTURED_ROOT}/${SPLIT}/metrics_npy.json}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

args=(
  --query_embeddings "${QUERY_EMB}"
  --query_meta "${QUERY_META}"
  --video_embeddings "${VIDEO_EMB}"
  --video_docid2row "${VIDEO_DOCID2ROW}"
  --topk_list "${TOPKS}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --normalize "${NORMALIZE}"
  --out_path "${OUT_PATH}"
)

if [[ -n "${QUERIES_JSONL}" ]]; then
  if [[ -f "${QUERIES_JSONL}" ]]; then
    args+=(--queries "${QUERIES_JSONL}")
  else
    echo "[07_eval_npy][warn] QUERIES_JSONL not found: ${QUERIES_JSONL}" >&2
    echo "[07_eval_npy][warn] evaluating full query_meta rows instead." >&2
  fi
fi

echo "[07_eval_npy] split=${SPLIT} topks=${TOPKS}"
echo "[07_eval_npy] query_emb=${QUERY_EMB}"
echo "[07_eval_npy] query_meta=${QUERY_META}"
echo "[07_eval_npy] video_emb=${VIDEO_EMB}"
echo "[07_eval_npy] video_docid2row=${VIDEO_DOCID2ROW}"
if [[ -f "${QUERIES_JSONL}" ]]; then
  echo "[07_eval_npy] queries=${QUERIES_JSONL}"
else
  echo "[07_eval_npy] queries=(none, using query_meta)"
fi

echo "[07_eval_npy] out=${OUT_PATH}"
PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.eval_retrieval_npy "${args[@]}"

echo "[07_eval_npy] done"
