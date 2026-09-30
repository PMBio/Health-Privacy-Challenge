#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=[YOUR-PARTION]
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs-eval/eval_%A_%a.out

set -euo pipefail
source activate_env.sh  
source "lib.sh"

CONFIG="${1:?usage: run_eval.sh <config> <dataset>}"
DATASET="${2:?missing dataset}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))     # array is 0-based; manifest is 1-based
resolve_paths

read -r GEN EXP SPLIT < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/eval.txt")


out="${RES_ROOT}/${GEN}/${EXP}/evaluation_split_${SPLIT}.csv"
if [[ -f "$out" ]]; then
    echo "skip: $out exists"; exit 0
fi

python "${SRC_DIR}/evaluation/evaluate.py" run-evaluator "$SPLIT" "$GEN" "$EXP" \
       --configfile "$CONFIG"