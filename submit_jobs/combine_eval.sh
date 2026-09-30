#!/bin/bash
#SBATCH --job-name=combine_eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs-combine/combine_eval_%A_%a.out
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: combine_eval.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))

resolve_paths

read -r GEN EXP < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/pairs.txt")

out="${RES_ROOT}/${GEN}/${EXP}/evaluation_results.csv"
if [[ -f "$out" ]]; then
    echo "skip: $out exists"; exit 0
fi

python "${SRC_DIR}/evaluation/evaluate.py" combine-results "$GEN" "$EXP" \
     --configfile "$CONFIG"