#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="original_sft"
RUN_NAME="${RUN_NAME:-qwen7b_interval_original_sft_kl0p01_ebs256_ddp2}"
DEFAULT_GPUS="${DEFAULT_GPUS:-2,3}"
DEFAULT_MASTER_PORT="${DEFAULT_MASTER_PORT:-29602}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen7b_ddp_common.sh"
