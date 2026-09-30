#!/bin/bash
set -euo pipefail
CONFIG="${1:?usage: submit_generators.sh <config.yaml>}"

SPLIT_NUM=5
NOISE_LEVEL=0.5

# generator -> experiment_name  
declare -A EXPERIMENT=(
    #["ctgan"]="no_dp"
    #["dpctgan"]="dp"
    #["cvae"]="no_dp"
    ["cvae_gmm"]="no_dp"
    #["dpcvae"]="dp"
    ["wgan_gp"]="no_dp"
    #["multivariate"]="noise_${NOISE_LEVEL}"    
)

for gen in "${!EXPERIMENT[@]}"; do
    exp="${EXPERIMENT[$gen]}"
    echo "submitting $gen ($exp), splits 1-${SPLIT_NUM}"
    sbatch --job-name="gen_${gen}" \
           --array=1-"${SPLIT_NUM}" \
           run_generators.sh "$gen" "$exp" "$CONFIG"
done