#!/bin/bash
#SBATCH --job-name=cl_num_sector_search
#SBATCH --output=logs/cl_num_sector_search_%A_%a.out
#SBATCH --error=logs/cl_num_sector_search_%A_%a.err
#SBATCH --time=20:00:00
# 0: h2o 6-31g   bond 0.96, angle 104.0
# 1: h2o 6-31g   bond 2.00, angle 104.0
# 2: h2o sto-3g  bond 2.00, angle 104.5
# 3: h4_linear 6-311++g bond 2.00
# 4: lih 6-31g   bond 2.50
# 5: n2 sto-3g   bond 2.50
#SBATCH --array=0-5
#SBATCH --partition cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=th-cl-uv[201-203,301-302],met-cl-lx[017-020,022-025]

# # # block2-related fixes: start # # #
# 1. Prevent thread oversubscription / hangs when using 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# 2. Bypass Intel MKL CPU vendor lock-in for AMD EPYC nodes
export LD_PRELOAD="/project/theorie/d/Damiano.Aliverti/quasisymmetry_forked/libfakeintel.so"
# 3. Force MKL to use the AVX2 kernel directly, bypassing runtime CPU dispatch
#    (avoids mis-detection on certain AMD EPYC steppings after the vendor spoof)
export MKL_ENABLE_INSTRUCTIONS=AVX2
# 4. Check and print whether the node has avx2
if grep -q 'avx2' /proc/cpuinfo; then
    echo "AVX2 check: SUPPORTED on node $(hostname). No action needed"
else
    echo "AVX2 check: NOT SUPPORTED on node $(hostname) -> add to excluded nodes at the top of the .sh script"
fi
# # # block2-related fixes: end # # #

source quasisym/bin/activate

case "$SLURM_ARRAY_TASK_ID" in
    0)
        molecule=h2o
        basis=6-31g
        bondlength=0.96
        bondangle=104.0
        ;;
    1)
        molecule=h2o
        basis=6-31g
        bondlength=2.0
        bondangle=104.0
        ;;
    2)
        molecule=h2o
        basis=sto-3g
        bondlength=2.0
        bondangle=104.5
        ;;
    3)
        molecule=h4_linear
        basis=6-311++g
        bondlength=2.0
        ;;
    4)
        molecule=lih
        basis=6-31g
        bondlength=2.5
        ;;
    5)
        molecule=n2
        basis=sto-3g
        bondlength=2.5
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

python_args=(
    "$molecule" "$basis" "$bondlength" variance
    --K-sector-analysis --num-sectors-to-retain 40 --max-elec-transfer 2
)

if [[ -n "${bondangle:-}" ]]; then
    python_args+=(--bond-angle "$bondangle")
fi

python cluster_number_sector_search_exact_weights.py "${python_args[@]}"