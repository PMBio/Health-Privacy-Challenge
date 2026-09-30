#!/bin/bash
#SBATCH --job-name=combine_de
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs-combine-de/combine_de_%A_%a.out
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: combine_de.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))
resolve_paths

read -r GEN EXP LFC < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/pairs_lfc.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/DE_lfc=${LFC}_results_fpr.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

python "${SRC_DIR}/evaluation/evaluate.py" combine-diffexpress-results "$LFC" "$GEN" "$EXP" --configfile "$CONFIG"