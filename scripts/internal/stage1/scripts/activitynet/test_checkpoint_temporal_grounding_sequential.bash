#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"
SFT_FORCE_DATASET_DEFAULTS="${SFT_FORCE_DATASET_DEFAULTS:-True}"
set_sft_dataset_defaults "activitynet"

exec env "${SFT_DATASET_ENV[@]}" DEFAULT_CHECKPOINT="${DEFAULT_CHECKPOINT:-activitynet-stage2}" "${script_dir}/../../test_checkpoint_temporal_grounding_sequential.bash" "$@"
