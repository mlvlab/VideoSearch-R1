#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"
SFT_FORCE_DATASET_DEFAULTS="True"
set_sft_dataset_defaults "didemo"

if [[ $# -eq 0 && -z "${JSONL:-}" ]]; then
  maybe_set_default_report_jsonl temporal
fi

exec env "${SFT_DATASET_ENV[@]}" "${script_dir}/../../temp_report_temporal_grounding.bash" "$@"
