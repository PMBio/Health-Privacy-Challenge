#!/bin/bash
#SBATCH --job-name=coexpr
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs-coexpr/coexpr_%A_%a.out

set -euo pipefail
source "lib.sh"

CONFIG="${1:?usage: run_coexpr.sh <config>}"
resolve_paths
HCOCENA_DIR=$(cfg dir_list hcocena_dir)

TASK=$((SLURM_ARRAY_TASK_ID + 1))
read -r GEN EXP SPLIT CUTOFF < <(sed -n "${TASK}p" "${MANIFEST_DIR}/coexpr.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/coexpr_cutoff=${CUTOFF}_split_${SPLIT}.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

module load R/4.3.2-gfbf-2023a
Rscript "${SRC_DIR}/evaluation/bio/coexpression.R" \
    "$HOME_DIR" "$SPLIT" "$DATASET" "$GEN" "$EXP" "$HCOCENA_DIR" "$CUTOFF"