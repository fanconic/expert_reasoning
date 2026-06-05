#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="original_base"
RUN_NAME="${RUN_NAME:-qwen7b_interval_original_base_kl0p01_ebs256_ddp2}"
DEFAULT_GPUS="${DEFAULT_GPUS:-0,1}"
DEFAULT_MASTER_PORT="${DEFAULT_MASTER_PORT:-29601}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen7b_ddp_common.sh"
