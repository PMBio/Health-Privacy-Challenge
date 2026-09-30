#!/bin/bash
#SBATCH --job-name=de
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs-de/de_%A_%a.out
set -euo pipefail
source "lib.sh"

CONFIG="${1:?usage: run_de.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))
resolve_paths


read -r GEN EXP SPLIT LFC < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/de.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/DE_lfc=${LFC}_split_${SPLIT}_fpr.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

module load R/4.3.2-gfbf-2023a
Rscript "${SRC_DIR}/evaluation/bio/diffexpression.R" \
    "$HOME_DIR" "$SPLIT" "$DATASET" "$GEN" "$EXP" "$P_VALUE" "$LFC"