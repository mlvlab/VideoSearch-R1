#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../_dataset_defaults.bash"
user_report_to="${REPORT_TO:-}"
user_wandb_project="${WANDB_PROJECT:-}"
user_run_name="${RUN_NAME:-}"
user_attn_impl="${ATTN_IMPL:-}"
user_master_port="${MASTER_PORT:-}"
SFT_FORCE_DATASET_DEFAULTS="${SFT_FORCE_DATASET_DEFAULTS:-True}"
set_sft_dataset_defaults "didemo"

# Didemo default experiment naming (can be overridden via env).
report_to="${user_report_to:-wandb}"
wandb_project="${user_wandb_project:-eccv27}"
run_name="${user_run_name:-full_sft}"
attn_impl="${user_attn_impl:-flash_attention_2}"
master_port="${user_master_port:-29107}"
#export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-True}"

exec env "${SFT_DATASET_ENV[@]}" \
  REPORT_TO="${report_to}" \
  WANDB_PROJECT="${wandb_project}" \
  RUN_NAME="${run_name}" \
  ATTN_IMPL="${attn_impl}" \
  MASTER_PORT="${master_port}" \
  "${script_dir}/../../verification_setting.bash" "$@"
