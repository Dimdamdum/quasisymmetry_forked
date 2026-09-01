#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=logs/cluster_metrics_%A_%a.out
#SBATCH --error=logs/cluster_metrics_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --partition small
#SBATCH --cpus-per-task=4  # Give it a few cores if DMRG needs them
#SBATCH --mem=32G # 32 GB for medium cost DMRG or if 3- and 4-rdms are needed

# # # block2-related fixes: start # # #
# 1. Prevent thread oversubscription / hangs when using 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# 2. Bypass Intel MKL CPU vendor lock-in for AMD EPYC nodes
export LD_PRELOAD="/project/theorie/d/Damiano.Aliverti/quasisymmetry_forked/libfakeintel.so"
# # # block2-related fixes: end # # #

source quasisym/bin/activate

# Define bond lengths based on SLURM_ARRAY_TASK_ID
minbondlength=1.7; maxbondlength=2.9; steps=2
bondlengths=($(seq $minbondlength $(awk "BEGIN {print ($maxbondlength - $minbondlength) / ($steps - 1)}") $maxbondlength))
b=${bondlengths[$SLURM_ARRAY_TASK_ID]}

# Parse CLI arguments
molecule="$1"
basis="$2"
angle="$3"
matrix="$4"
skip_k_param="${5:-}"  # Optional 5th parameter (defaults to empty string)

# Initialize an array for optional Python flags
OPT_ARGS=()

# If the 5th argument is provided and isn't "false" or "0", add the flag
if [[ -n "$skip_k_param" && "$skip_k_param" != "false" && "$skip_k_param" != "0" ]]; then
    OPT_ARGS+=("--skip-K-states")
fi

# tell the system where mkl package is - Gemini's latest fix
# export LD_LIBRARY_PATH="/project/theorie/d/Damiano.Aliverti/quasisymmetry_forked/quasisym/lib:$LD_LIBRARY_PATH"

# Run the Python script over the metrics
for metric in "variance" "commutator"; do
    python cluster_numbers_metrics.py "$molecule" "$basis" "$b" "$metric" \
        --bond-angle "$angle" \
        --cluster-matrix "$matrix" \
        --max-transfers 1 2 3 4 5 6 \
        "${OPT_ARGS[@]}"
done