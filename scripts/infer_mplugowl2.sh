#!/usr/bin/env bash
# Activate your mPLUG-Owl2 environment first. Extra arguments go to its image-chat CLI.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" -m mplug_owl2.serve.cli \
  --model-path "${repo_root}/checkpoints/mplug_owl2_fakeinstruct_v3" \
  --device cuda:0 \
  --conv-mode mplug_owl2 \
  "$@"
