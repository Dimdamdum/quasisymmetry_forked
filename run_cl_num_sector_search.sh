#!/bin/bash
#SBATCH --job-name=decomp_opt
#SBATCH --output=logs/cluster_decomp_opt_%A_%a.out
#SBATCH --error=logs/cluster_decomp_opt_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --array=2
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

molecules=( h2o h2o h4_linear lih n2)
bases=(  6-31g 6-31g 6-311++g 6-31g sto-3g)
bondlengths=( 0.96 2.00 2.00  2.5 2.5)
angles=(   104  104   ""     ""      ""    )

molecule="${molecules[$SLURM_ARRAY_TASK_ID]}"
basis="${bases[$SLURM_ARRAY_TASK_ID]}"
bondlength="${bondlengths[$SLURM_ARRAY_TASK_ID]}"
angle="${angles[$SLURM_ARRAY_TASK_ID]}"

OPT_ARGS=()
if [[ -n "$angle" ]]; then
    OPT_ARGS+=(--bond-angle "$angle")
fi

python cluster_number_sector_search.py "$molecule" "$basis" "$bondlength" variance "${OPT_ARGS[@]}" --K-sector-analysis --num-sectors-to-retain 40 --min-child-cluster-size 3