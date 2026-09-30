#!/bin/bash
set -euo pipefail

source "activate_env.sh"
source "lib.sh"


CONFIG="${1:?usage: submit_mia.sh <config.yaml>}"
resolve_paths

python "${HOME_DIR}/src/prepare/get_manifest.py" --config "$CONFIG"   # reuses eval.txt

N=$(wc -l < "${HOME_DIR}/manifests/${DATASET}/eval.txt")
MIA=$(sbatch --parsable --array=0-$((N - 1)) run_mia.sh "$CONFIG")
echo "mia array: $MIA  ($N tasks)"