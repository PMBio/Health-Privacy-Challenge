#!/bin/bash
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: submit_bio.sh <config.yaml>}"
resolve_paths

python "${HOME_DIR}/src/prepare/get_manifest.py" --config "$CONFIG"

M="${HOME_DIR}/manifests/${DATASET}"
N_DE=$(wc -l < "$M/de.txt")
N_DE_PL=$(wc -l < "$M/pairs_lfc.txt")
N_PW=$(wc -l < "$M/eval.txt")
N_PW_PR=$(wc -l < "$M/pairs.txt")
N_CX=$(wc -l < "$M/coexpr.txt")
N_CX_PR=$(wc -l < "$M/pairs_cutoff.txt")

# --- DE: array -> combine ---
DE=$(sbatch --parsable --array=0-$((N_DE - 1)) run_de.sh "$CONFIG")
echo "DE: $DE ($N_DE tasks)"
sbatch --parsable --dependency=afterok:$DE --array=0-$((N_DE_PL - 1)) combine_de.sh "$CONFIG"

# --- Pathway: array -> combine ---
PW=$(sbatch --parsable --array=0-$((N_PW - 1)) run_pathway.sh "$CONFIG")
echo "pathway: $PW ($N_PW tasks)"
sbatch --parsable --dependency=afterok:$PW --array=0-$((N_PW_PR - 1)) combine_pathway.sh "$CONFIG"


# --- Coexpression: array -> combine ---
CX=$(sbatch --parsable --array=0-$((N_CX - 1)) run_coexpr.sh "$CONFIG")
echo "coexpr: $CX"
sbatch --dependency=afterok:$CX --array=0-$((N_CX_PR - 1)) combine_coexpr.sh "$CONFIG"

echo "submitted DE + pathway + coexpr for $DATASET"