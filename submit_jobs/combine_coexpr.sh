#!/bin/bash
#SBATCH --job-name=combine_coexpr
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs-combine-coexpr/combine_coexpr_%A_%a.out
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: combine_coexpr.sh <config>}"
resolve_paths

TASK=$((SLURM_ARRAY_TASK_ID + 1))
read -r GEN EXP CUTOFF < <(sed -n "${TASK}p" "${MANIFEST_DIR}/pairs_cutoff.txt")

out="${BIO_ROOT}/${GEN}/${EXP}/coexpr_cutoff=${CUTOFF}_results.csv"
if [[ -s "$out" ]]; then echo "skip: $out"; exit 0; fi

python "${SRC_DIR}/evaluation/evaluate.py" combine-coexpress-results "$CUTOFF" "$GEN" "$EXP" --configfile "$CONFIG"