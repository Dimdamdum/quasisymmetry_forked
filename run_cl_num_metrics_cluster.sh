#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=logs/cluster_metrics_%A_%a.out
#SBATCH --error=logs/cluster_metrics_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --partition cluster

#SBATCH --cpus-per-task=4   # Give it a few cores if DMRG needs them

# --- Prevent thread oversubscription & deadlocks ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
# --------------------------------------------------

source quasisym/bin/activate

# Create a unique scratch directory for THIS specific array task ID.
# This means each script runs dmrg. To be made more efficient later (e.g., first run dmrg, then get metrics by loading mps)
export TMP_WAVEFUNCTION_DIR="wavefunctions_job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP_WAVEFUNCTION_DIR"

types=('variance' 'extremality')
t=${types[$SLURM_ARRAY_TASK_ID]}

python cluster_numbers_metrics.py h2o sto3g 1.0 "$t" \
    --cluster-matrix '[[1,1,0,0,0,0,0],[0,0,1,1,0,0,0]]' \
    --max-transfers 2 --bond-angle 104.5 \
    --wavefunction-dir "$TMP_WAVEFUNCTION_DIR"

# clean up wavefunction files after job finishes to save disk space
rm -rf "$TMP_WAVEFUNCTION_DIR"