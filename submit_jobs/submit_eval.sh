#!/bin/bash
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: submit_eval.sh <config.yaml>}"
resolve_paths

# build manifests from the (already generated) synthetic tree
python "${HOME_DIR}/src/prepare/get_manifest.py" --config "$CONFIG"

N_EVAL=$(wc -l < "${HOME_DIR}/manifests/${DATASET}/eval.txt")
N_PAIRS=$(wc -l < "${HOME_DIR}/manifests/${DATASET}/pairs.txt")

EVAL=$(sbatch --parsable --array=0-$((N_EVAL - 1))%60 run_eval.sh "$CONFIG" "$DATASET")
echo "eval array: $EVAL  ($N_EVAL tasks)"

COMBINE=$(sbatch --parsable --dependency=afterok:$EVAL \
    --array=0-$((N_PAIRS - 1)) combine_eval.sh "$CONFIG" "$DATASET")
echo "combine array: $COMBINE  ($N_PAIRS tasks, after $EVAL)"