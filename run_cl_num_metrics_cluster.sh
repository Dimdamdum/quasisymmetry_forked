#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=logs/cluster_metrics_%A_%a.out
#SBATCH --error=logs/cluster_metrics_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --partition small
#SBATCH --cpus-per-task=1  # Give it a few cores if DMRG needs them
#SBATCH --mem=32G # 32 GB for medium cost DMRG or if 3- and 4-rdms are needed

# # # block2-related fixes: start # # #
# 1. Prevent thread oversubscription / hangs when using 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# 2. Fix Intel MKL shared library loading and CPU detection on compute nodes
export LD_LIBRARY_PATH="/project/theorie/d/Damiano.Aliverti/quasisymmetry_forked/quasisym/lib/python3.11/site-packages/mkl/lib/intel64:/project/theorie/d/Damiano.Aliverti/quasisymmetry_forked/quasisym/lib/python3.11/site-packages/block2.libs:$LD_LIBRARY_PATH"
export MKL_DEBUG_CPU_TYPE=5
export MKL_ENABLE_INSTRUCTIONS=AVX2
# # # block2-related fixes: end # # #

source quasisym/bin/activate

minbondlength=0.8; maxbondlength=2.9; steps=8
bondlengths=($(seq $minbondlength $(awk "BEGIN {print ($maxbondlength - $minbondlength) / ($steps - 1)}") $maxbondlength))
b=${bondlengths[$SLURM_ARRAY_TASK_ID]}

python cluster_numbers_metrics.py n2 sto3g $b "variance" \
    --cluster-matrix '[[1,1,1,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,1,1,0,0]]' \
    --max-transfers 1 2 3 4 5 6

python cluster_numbers_metrics.py n2 sto3g $b "commutator" \
    --cluster-matrix '[[1,1,1,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,1,1,0,0]]' \
    --max-transfers 1 2 3 4 5 6
