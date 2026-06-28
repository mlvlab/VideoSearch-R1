#!/usr/bin/env bash
set -euo pipefail

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${_repo_root}/scripts/common/env.bash"
DATASET_ROOT="$(videosearch_dataset_dir activitynet)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOSEARCH_DATA_ROOT}}"
STRUCTURED_ROOT="${STRUCTURED_ROOT:-${DATASET_ROOT}}"
SPLIT="${SPLIT:-train}"

OUT_SUFFIX="${OUT_SUFFIX:-1fps}"
EMBED_SUBDIR="${EMBED_SUBDIR:-video_embedding_${OUT_SUFFIX}}"

# Match the provided example: retrieve top-50 and drop GT without refill -> typically 49~50 per qid.
TOPK_POOL="${TOPK_POOL:-50}"
MAX_NEGATIVES="${MAX_NEGATIVES:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
NORMALIZE="${NORMALIZE:-1}"
DEDUP="${DEDUP:-1}"

QUERY_EMB="${QUERY_EMB:-${STRUCTURED_ROOT}/${SPLIT}/query_embedding/query_embeddings.${SPLIT}.npy}"
QUERY_META="${QUERY_META:-${STRUCTURED_ROOT}/${SPLIT}/query_embedding/query_meta.${SPLIT}.jsonl}"
VIDEO_EMB="${VIDEO_EMB:-${STRUCTURED_ROOT}/${SPLIT}/${EMBED_SUBDIR}/segment_embeds.npy}"
VIDEO_DOCID2ROW="${VIDEO_DOCID2ROW:-${STRUCTURED_ROOT}/${SPLIT}/${EMBED_SUBDIR}/docid2row.json}"

OUT_PATH="${OUT_PATH:-${STRUCTURED_ROOT}/${SPLIT}/hard_negatives.json}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "[08_hard_neg] dataset=activitynet split=${SPLIT}"
echo "[08_hard_neg] query_emb=${QUERY_EMB}"
echo "[08_hard_neg] query_meta=${QUERY_META}"
echo "[08_hard_neg] video_emb=${VIDEO_EMB}"
echo "[08_hard_neg] video_docid2row=${VIDEO_DOCID2ROW}"
echo "[08_hard_neg] topk_pool=${TOPK_POOL} max_negatives=${MAX_NEGATIVES}"
echo "[08_hard_neg] out=${OUT_PATH}"

PYTHONPATH=. "${PYTHON_BIN}" -m annotation_process.build_hard_negatives_npy \
  --query_embeddings "${QUERY_EMB}" \
  --query_meta "${QUERY_META}" \
  --video_embeddings "${VIDEO_EMB}" \
  --video_docid2row "${VIDEO_DOCID2ROW}" \
  --out_path "${OUT_PATH}" \
  --topk_pool "${TOPK_POOL}" \
  --max_negatives "${MAX_NEGATIVES}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --normalize "${NORMALIZE}" \
  --dedup "${DEDUP}"

echo "[08_hard_neg] done"
