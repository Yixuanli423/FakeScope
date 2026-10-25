#!/usr/bin/env bash
# Activate your LLaVA environment first. Extra arguments go to its image-chat CLI.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m llava.serve.cli \
  --model-path "${repo_root}/checkpoints/llava_1.5_fakeinstruct_v3" \
  --device cuda:0 \
  --conv-mode llava_v1 \
  "$@"
