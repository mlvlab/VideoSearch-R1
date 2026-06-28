#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"
user_master_port="${MASTER_PORT:-}"
user_report_to="${REPORT_TO:-}"
user_wandb_project="${WANDB_PROJECT:-}"
user_run_name="${RUN_NAME:-}"
user_attn_impl="${ATTN_IMPL:-}"
SFT_FORCE_DATASET_DEFAULTS="${SFT_FORCE_DATASET_DEFAULTS:-True}"
set_sft_dataset_defaults "activitynet"
master_port="${user_master_port:-29103}"
report_to="${user_report_to:-wandb}"
wandb_project="${user_wandb_project:-eccv27}"
run_name="${user_run_name:-real_soft}"
attn_impl="${user_attn_impl:-flash_attention_2}"

exec env "${SFT_DATASET_ENV[@]}" \
  MASTER_PORT="${master_port}" \
  REPORT_TO="${report_to}" \
  WANDB_PROJECT="${wandb_project}" \
  RUN_NAME="${run_name}" \
  ATTN_IMPL="${attn_impl}" \
  "${script_dir}/../../verification_setting.bash" "$@"
