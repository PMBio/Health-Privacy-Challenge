#!/bin/bash
#SBATCH --job-name=train-generate
#SBATCH --partition=gpu #[YOUR CLUSTER SETTING]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-5                      
#SBATCH --output=logs-train/gen_%x_%A_%a.out

set -euo pipefail
source activate_env.sh  
source lib.sh

# --- args first ---
GENERATOR="${1:?usage: sbatch run_generators.sh <generator> <experiment>}"
EXPERIMENT="${2:?missing experiment}"
CONFIG="${3:?missing config}"
resolve_paths

SPLIT=$SLURM_ARRAY_TASK_ID

OUT="${DATA_SPLIT_DIR}/${DATASET}/synthetic/${GENERATOR}/${EXPERIMENT}"
data_out="${OUT}/synthetic_data_split_${SPLIT}.csv"
lbl_out="${OUT}/synthetic_labels_split_${SPLIT}.csv"


if [[ -f "$data_out" && -f "$lbl_out" ]]; then
    echo "skip: $GENERATOR/$EXPERIMENT split $SPLIT already done"
    exit 0
fi

python "${SRC_DIR}/generators/blue_team.py" run-generator "$SPLIT" \
    --configfile "$CONFIG" \
    --generator_name "$GENERATOR" \
    --experiment_name "$EXPERIMENT"