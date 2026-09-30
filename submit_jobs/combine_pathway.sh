#!/bin/bash
#SBATCH --job-name=combine_pathway
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs-combine-pathway/combine_pathway_%A_%a.out
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: combine_pathway.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))

resolve_paths

read -r GEN EXP < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/pairs.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/pathway_metrics_results.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

python "${SRC_DIR}/evaluation/evaluate.py" combine-pathway-results "$GEN" "$EXP" --configfile "$CONFIG"