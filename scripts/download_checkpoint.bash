#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${repo_root}/scripts/common/env.bash"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_checkpoint.bash didemo-stage1
  bash scripts/download_checkpoint.bash didemo-stage2
  bash scripts/download_checkpoint.bash charades-stage1
  bash scripts/download_checkpoint.bash charades-stage2
  bash scripts/download_checkpoint.bash activitynet-stage2
  bash scripts/download_checkpoint.bash VideoSearchR1/didemo-stage2

Env:
  VIDEOSEARCH_CHECKPOINT_ROOT=/path/to/checkpoints
  HF_TOKEN=...  # optional for public repos, required for private repos
EOF
}

model="${1:-}"
case "${model}" in
  -h|--help|"")
    usage
    exit 0
    ;;
esac

if [[ $# -ge 2 ]]; then
  repo_id="$(videosearch_resolve_hf_model "${model}")"
  target="$2"
  mkdir -p "${target}"
  echo "[download_checkpoint] ${repo_id} -> ${target}"
  python - "${repo_id}" "${target}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
  echo "${target}"
else
  videosearch_local_model_path "${model}"
fi
