#!/bin/bash
#SBATCH --job-name=pathway
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs-pathway/pathway_%A_%a.out
set -euo pipefail
source "lib.sh"

CONFIG="${1:?usage: run_pathway.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))
resolve_paths


read -r GEN EXP SPLIT < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/eval.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/pathway_metrics_split_${SPLIT}.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

module load R/4.3.2-gfbf-2023a
Rscript "${SRC_DIR}/evaluation/bio/pathway.R" \
    "$HOME_DIR" "$SPLIT" "$DATASET" "$GEN" "$EXP"