#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval_all_checkpoints_temporal_grounding.bash /path/to/run_dir
  RUN_DIR=/path/to/run_dir bash eval_all_checkpoints_temporal_grounding.bash

What it does:
  1) Finds checkpoint-* dirs under RUN_DIR
  2) Optionally includes final model at RUN_DIR
  3) Runs temporal eval in parallel across GPUs (checkpoint-level sharding)
  4) Merges per-checkpoint outputs into one final summary + merged jsonl

Env knobs:
  SHARD_GPUS=0,1,2,3              # fallback to GPUS, then 0
  LOGS_DIR=/path/to/logs          # default: RUN_DIR/logs
  INCLUDE_FINAL_MODEL=1           # 1: evaluate RUN_DIR final model too
  SKIP_DONE=1                     # 1: skip checkpoint if summary json already exists
  FAIL_FAST=0                     # 1: stop worker queue on first failure
  SHARD_KEEP_TMP=1                # 1: keep temp shard logs/queues
  MERGE_DETAIL_JSONL=0            # 1: merge per-checkpoint detail jsonl into one file
  EVAL_GPU (ignored here)         # each shard worker sets EVAL_GPU automatically

Pass-through to underlying eval script:
  MAX_TURN, IOU_THRESHOLDS, EXTERNAL_EVAL_MAX_NEW_TOKENS, TOPK, MAX_SAMPLES,
  EVAL_LIMIT_RATIO, RESUME_FROM_JSONL, QUERY_EMBEDDER_MODEL_PATH, ...

Outputs (in LOGS_DIR):
  - external_verified_test_temporal_grounding_<checkpoint>.json(.jsonl)
  - all_checkpoints_temporal_grounding_summary.csv
  - all_checkpoints_temporal_grounding_summary.md
  - all_checkpoints_temporal_grounding_summary.json
  - (optional) all_checkpoints_temporal_grounding_merged_detail.jsonl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"
SFT_FORCE_DATASET_DEFAULTS="True"
set_sft_dataset_defaults "didemo"

run_dir="${1:-${RUN_DIR:-}}"
if [[ -z "${run_dir}" ]]; then
  echo "[eval_all][error] run_dir is required." >&2
  usage
  exit 1
fi
if [[ ! -d "${run_dir}" ]]; then
  echo "[eval_all][error] run_dir not found: ${run_dir}" >&2
  exit 1
fi

single_eval_script="${script_dir}/test_checkpoint_temporal_grounding.bash"
if [[ ! -f "${single_eval_script}" ]]; then
  echo "[eval_all][error] missing script: ${single_eval_script}" >&2
  exit 1
fi

shard_gpus_raw="${SHARD_GPUS:-${GPUS:-0}}"
logs_dir="${LOGS_DIR:-${run_dir}/logs}"
include_final_model="${INCLUDE_FINAL_MODEL:-1}"
skip_done="${SKIP_DONE:-1}"
fail_fast="${FAIL_FAST:-0}"
shard_keep_tmp="${SHARD_KEEP_TMP:-1}"
merge_detail_jsonl="${MERGE_DETAIL_JSONL:-0}"

mkdir -p "${logs_dir}"
tmp_dir="${logs_dir}/.eval_all_ckpt_temporal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${tmp_dir}"

cleanup() {
  if [[ "${shard_keep_tmp}" != "1" ]]; then
    rm -rf "${tmp_dir}"
  fi
}
trap cleanup EXIT

declare -a shard_gpus=()
IFS=',' read -ra _gpu_arr <<< "${shard_gpus_raw}"
for raw in "${_gpu_arr[@]}"; do
  gpu="$(echo "${raw}" | xargs)"
  if [[ -n "${gpu}" ]]; then
    shard_gpus+=("${gpu}")
  fi
done
if [[ "${#shard_gpus[@]}" -eq 0 ]]; then
  echo "[eval_all][error] no valid GPUs in SHARD_GPUS/GPUS: ${shard_gpus_raw}" >&2
  exit 1
fi

has_final_model=0
if [[ -f "${run_dir}/config.json" ]] && compgen -G "${run_dir}/model*.safetensors" > /dev/null; then
  has_final_model=1
elif [[ -f "${run_dir}/config.json" ]] && compgen -G "${run_dir}/pytorch_model*.bin" > /dev/null; then
  has_final_model=1
fi

declare -a ckpt_paths=()
while IFS= read -r path; do
  [[ -n "${path}" ]] && ckpt_paths+=("${path}")
done < <(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" | sort -V)

if [[ "${#ckpt_paths[@]}" -eq 0 ]] && [[ "${include_final_model}" != "1" || "${has_final_model}" != "1" ]]; then
  echo "[eval_all][error] no checkpoint-* found and no final model found in: ${run_dir}" >&2
  exit 1
fi

targets_tsv="${tmp_dir}/targets.tsv"
: > "${targets_tsv}"

add_target() {
  local label="$1"
  local path="$2"
  local state="$3"
  local base summary_json detail_jsonl
  base="$(basename "${path}")"
  summary_json="${logs_dir}/external_verified_test_temporal_grounding_${base}.json"
  detail_jsonl="${logs_dir}/external_verified_test_temporal_grounding_${base}.jsonl"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${path}" "${base}" "${summary_json}" "${detail_jsonl}" "${state}" >> "${targets_tsv}"
}

pending_count=0
done_count=0

for ckpt in "${ckpt_paths[@]}"; do
  label="$(basename "${ckpt}")"
  base="$(basename "${ckpt}")"
  summary_json="${logs_dir}/external_verified_test_temporal_grounding_${base}.json"
  if [[ "${skip_done}" == "1" && -s "${summary_json}" ]]; then
    add_target "${label}" "${ckpt}" "done"
    ((done_count += 1))
  else
    add_target "${label}" "${ckpt}" "pending"
    ((pending_count += 1))
  fi
done

if [[ "${include_final_model}" == "1" && "${has_final_model}" == "1" ]]; then
  base="$(basename "${run_dir}")"
  summary_json="${logs_dir}/external_verified_test_temporal_grounding_${base}.json"
  if [[ "${skip_done}" == "1" && -s "${summary_json}" ]]; then
    add_target "final_model" "${run_dir}" "done"
    ((done_count += 1))
  else
    add_target "final_model" "${run_dir}" "pending"
    ((pending_count += 1))
  fi
fi

echo "[eval_all] run_dir=${run_dir}"
echo "[eval_all] logs_dir=${logs_dir}"
echo "[eval_all] shard_gpus=${shard_gpus[*]}"
echo "[eval_all] targets_total=$((pending_count + done_count)) pending=${pending_count} done=${done_count}"
echo "[eval_all] tmp_dir=${tmp_dir}"

declare -a queue_files=()
for gpu in "${shard_gpus[@]}"; do
  qf="${tmp_dir}/queue.gpu${gpu}.tsv"
  : > "${qf}"
  queue_files+=("${qf}")
done

task_idx=0
while IFS=$'\t' read -r label path base summary_json detail_jsonl state; do
  [[ -z "${path}" ]] && continue
  if [[ "${state}" != "pending" ]]; then
    continue
  fi
  shard_idx=$((task_idx % ${#shard_gpus[@]}))
  gpu="${shard_gpus[$shard_idx]}"
  qf="${tmp_dir}/queue.gpu${gpu}.tsv"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${path}" "${base}" "${summary_json}" "${detail_jsonl}" >> "${qf}"
  ((task_idx += 1))
done < "${targets_tsv}"

: > "${tmp_dir}/failures.tsv"
: > "${tmp_dir}/success.tsv"

run_worker() {
  local gpu="$1"
  local queue_file="$2"
  local worker_rc=0
  while IFS=$'\t' read -r label ckpt_path base summary_json detail_jsonl; do
    [[ -z "${ckpt_path}" ]] && continue
    local run_log="${tmp_dir}/run.${label}.gpu${gpu}.log"
    echo "[eval_all][start] label=${label} gpu=${gpu} ckpt=${ckpt_path}"
    (
      set +e
      env "${SFT_DATASET_ENV[@]}" \
        EVAL_GPU="${gpu}" LOGS_DIR="${logs_dir}" \
        bash "${single_eval_script}" "${ckpt_path}"
    ) > "${run_log}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
      echo "[eval_all][error] label=${label} gpu=${gpu} rc=${rc} log=${run_log}" >&2
      printf "%s\t%s\t%s\t%s\t%d\n" "${label}" "${ckpt_path}" "${gpu}" "${run_log}" "${rc}" >> "${tmp_dir}/failures.tsv"
      worker_rc=1
      if [[ "${fail_fast}" == "1" ]]; then
        break
      fi
      continue
    fi
    echo "[eval_all][done] label=${label} gpu=${gpu} summary=${summary_json}"
    printf "%s\t%s\t%s\t%s\t%d\n" "${label}" "${ckpt_path}" "${gpu}" "${run_log}" "${rc}" >> "${tmp_dir}/success.tsv"
  done < "${queue_file}"
  return "${worker_rc}"
}

declare -a pids=()
declare -a pid_gpus=()
for gpu in "${shard_gpus[@]}"; do
  qf="${tmp_dir}/queue.gpu${gpu}.tsv"
  if [[ ! -s "${qf}" ]]; then
    continue
  fi
  run_worker "${gpu}" "${qf}" &
  pids+=("$!")
  pid_gpus+=("${gpu}")
done

overall_rc=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  gpu="${pid_gpus[$i]}"
  if wait "${pid}"; then
    echo "[eval_all] worker gpu=${gpu} finished"
  else
    echo "[eval_all][error] worker gpu=${gpu} failed" >&2
    overall_rc=1
  fi
done

summary_csv="${logs_dir}/all_checkpoints_temporal_grounding_summary.csv"
summary_md="${logs_dir}/all_checkpoints_temporal_grounding_summary.md"
summary_json="${logs_dir}/all_checkpoints_temporal_grounding_summary.json"
merged_jsonl="${logs_dir}/all_checkpoints_temporal_grounding_merged_detail.jsonl"

TARGETS_TSV="${targets_tsv}" \
SUMMARY_CSV="${summary_csv}" \
SUMMARY_MD="${summary_md}" \
SUMMARY_JSON="${summary_json}" \
MERGED_JSONL="${merged_jsonl}" \
MERGE_DETAIL_JSONL="${merge_detail_jsonl}" \
python - <<'PY'
import csv
import json
import os
from datetime import datetime

targets_tsv = os.environ["TARGETS_TSV"]
summary_csv = os.environ["SUMMARY_CSV"]
summary_md = os.environ["SUMMARY_MD"]
summary_json = os.environ["SUMMARY_JSON"]
merged_jsonl = os.environ["MERGED_JSONL"]
merge_detail_jsonl = str(os.environ.get("MERGE_DETAIL_JSONL", "0")).strip().lower() in {"1", "true", "yes", "on"}

targets = []
with open(targets_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        label, path, base, summary_path, detail_path, state = line.split("\t")
        targets.append(
            {
                "label": label,
                "path": path,
                "base": base,
                "summary_path": summary_path,
                "detail_path": detail_path,
                "state": state,
            }
        )

rows = []
missing = []
for t in targets:
    sp = t["summary_path"]
    if not os.path.isfile(sp) or os.path.getsize(sp) == 0:
        missing.append({"label": t["label"], "path": t["path"], "summary_path": sp})
        continue
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        missing.append({"label": t["label"], "path": t["path"], "summary_path": sp, "error": str(exc)})
        continue

    retrieval_final = (data.get("retrieval") or {}).get("final") or {}
    temporal_final = (data.get("temporal") or {}).get("final") or {}
    retrieval_delta = (data.get("retrieval") or {}).get("delta_final_minus_turn1") or {}
    temporal_delta = (data.get("temporal") or {}).get("delta_final_minus_turn1") or {}
    answer_counts = data.get("answer_counts") or {}

    row = {
        "label": t["label"],
        "base": t["base"],
        "model_path": data.get("model_path", t["path"]),
        "rows_valid": int(data.get("rows_valid", 0) or 0),
        "matched": int(answer_counts.get("matched", 0) or 0),
        "not_matched": int(answer_counts.get("not_matched", 0) or 0),
        "unknown": int(answer_counts.get("unknown", 0) or 0),
        "R@1": float(retrieval_final.get("R@1", 0.0) or 0.0),
        "R@5": float(retrieval_final.get("R@5", 0.0) or 0.0),
        "R@10": float(retrieval_final.get("R@10", 0.0) or 0.0),
        "MRR": float(retrieval_final.get("mrr", 0.0) or 0.0),
        "IoU@0.3@R1": float(temporal_final.get("IoU@0.3@R1", 0.0) or 0.0),
        "IoU@0.5@R1": float(temporal_final.get("IoU@0.5@R1", 0.0) or 0.0),
        "IoU@0.7@R1": float(temporal_final.get("IoU@0.7@R1", 0.0) or 0.0),
        "mIoU@R1": float(temporal_final.get("mIoU@R1", 0.0) or 0.0),
        "mIoU_matched_only": float(temporal_final.get("mIoU_matched_only", 0.0) or 0.0),
        "dR@1": float(retrieval_delta.get("R@1", 0.0) or 0.0),
        "dMRR": float(retrieval_delta.get("mrr", 0.0) or 0.0),
        "dIoU@0.5@R1": float(temporal_delta.get("IoU@0.5@R1", 0.0) or 0.0),
        "dmIoU@R1": float(temporal_delta.get("mIoU@R1", 0.0) or 0.0),
        "summary_json": sp,
        "detail_jsonl": t["detail_path"],
    }
    rows.append(row)

rows.sort(key=lambda r: (r["IoU@0.5@R1"], r["mIoU@R1"], r["R@1"]), reverse=True)

fieldnames = [
    "label",
    "base",
    "model_path",
    "rows_valid",
    "matched",
    "not_matched",
    "unknown",
    "R@1",
    "R@5",
    "R@10",
    "MRR",
    "IoU@0.3@R1",
    "IoU@0.5@R1",
    "IoU@0.7@R1",
    "mIoU@R1",
    "mIoU_matched_only",
    "dR@1",
    "dMRR",
    "dIoU@0.5@R1",
    "dmIoU@R1",
    "summary_json",
    "detail_jsonl",
]

with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Temporal Grounding Eval Summary (All Checkpoints)\n\n")
    f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"- Targets: {len(targets)}\n")
    f.write(f"- Available summaries: {len(rows)}\n")
    f.write(f"- Missing summaries: {len(missing)}\n\n")
    f.write("| label | rows | R@1 | MRR | IoU@0.5@R1 | mIoU@R1 | matched | not_matched | unknown |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['label']} | {row['rows_valid']} | {row['R@1']:.4f} | {row['MRR']:.4f} | "
            f"{row['IoU@0.5@R1']:.4f} | {row['mIoU@R1']:.4f} | {row['matched']} | "
            f"{row['not_matched']} | {row['unknown']} |\n"
        )
    if missing:
        f.write("\n## Missing\n")
        for item in missing:
            err = item.get("error", "")
            if err:
                f.write(f"- {item['label']}: {item['summary_path']} (error: {err})\n")
            else:
                f.write(f"- {item['label']}: {item['summary_path']}\n")

merge_written = 0
merge_bad = 0
merged_jsonl_artifact = ""
if merge_detail_jsonl:
    with open(merged_jsonl, "w", encoding="utf-8") as out:
        for row in rows:
            detail_path = row["detail_jsonl"]
            if not os.path.isfile(detail_path):
                continue
            with open(detail_path, "r", encoding="utf-8") as inp:
                for line in inp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        merge_bad += 1
                        continue
                    obj["_checkpoint_label"] = row["label"]
                    obj["_checkpoint_base"] = row["base"]
                    obj["_checkpoint_model_path"] = row["model_path"]
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    merge_written += 1
    merged_jsonl_artifact = merged_jsonl

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "targets_total": len(targets),
            "summaries_available": len(rows),
            "summaries_missing": len(missing),
            "merge_detail_jsonl": bool(merge_detail_jsonl),
            "merged_jsonl_rows": merge_written,
            "merged_jsonl_bad_lines": merge_bad,
            "ranked": rows,
            "missing": missing,
            "artifacts": {
                "summary_csv": summary_csv,
                "summary_md": summary_md,
                "merged_jsonl": merged_jsonl_artifact,
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"[eval_all][merge] summary_csv={summary_csv}")
print(f"[eval_all][merge] summary_md={summary_md}")
print(f"[eval_all][merge] summary_json={summary_json}")
if merge_detail_jsonl:
    print(f"[eval_all][merge] merged_jsonl={merged_jsonl} rows={merge_written} bad={merge_bad}")
else:
    print("[eval_all][merge] merged_jsonl=disabled")
PY

if [[ -s "${tmp_dir}/failures.tsv" ]]; then
  echo "[eval_all][error] failures detected:"
  cat "${tmp_dir}/failures.tsv"
  overall_rc=1
fi

echo "[eval_all] completed with rc=${overall_rc}"
echo "[eval_all] summary_csv=${summary_csv}"
echo "[eval_all] summary_md=${summary_md}"
echo "[eval_all] summary_json=${summary_json}"
if [[ "${merge_detail_jsonl}" == "1" ]]; then
  echo "[eval_all] merged_jsonl=${merged_jsonl}"
else
  echo "[eval_all] merged_jsonl=disabled (MERGE_DETAIL_JSONL=0)"
fi
echo "[eval_all] tmp_dir=${tmp_dir}"

exit "${overall_rc}"
