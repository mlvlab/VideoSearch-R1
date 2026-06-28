#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

dataset="${1:-all}"
exec bash "${repo_root}/scripts/download_data.bash" "${dataset}"
