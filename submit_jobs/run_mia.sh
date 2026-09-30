#!/bin/bash
#SBATCH --job-name=mia
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs-mia/mia_%A_%a.out
set -euo pipefail
source "activate_env.sh"
source "lib.sh"

CONFIG="${1:?usage: run_mia.sh <config>}"
TASK=$((SLURM_ARRAY_TASK_ID + 1))
resolve_paths

read -r GEN EXP SPLIT < <(sed -n "${TASK}p" "${HOME_DIR}/manifests/${DATASET}/eval.txt")

SYN_DS="${DATA_SPLIT_DIR}/${DATASET}/synthetic/${GEN}/${EXP}/synthetic_data_split_${SPLIT}.csv"
MIA_LBL="${DATA_SPLIT_DIR}/${DATASET}/real/MIA_lbl_split_${SPLIT}.csv"
MIA_TEST="${HOME_DIR}/data/processed/${DATASET}_primary_tumor_star_deseq_VST_lmgenes.tsv"

out="${RES_ROOT}/${GEN}/${EXP}/split_${SPLIT}/pos_loss.npy"
if [[ -f "$out" ]]; then
    echo "skip: $out exists"; exit 0
fi

## comment out for TCGA-COMBINED
# REFERENCE="${HOME_DIR}/data/processed/${DATASET}_primary_tumor_star_deseq_VST_lmgenes_reference.tsv"

python "${SRC_DIR}/mia/red_team.py" run-mia \
    "$SYN_DS" "$MIA_TEST" "${GEN}_${EXP}_split_${SPLIT}" "$GEN" "$EXP" \
    --mmb_labels_file "$MIA_LBL" \
    --test_on_real False \
    --configfile "$CONFIG" 
    #--reference_file "$REFERENCE"