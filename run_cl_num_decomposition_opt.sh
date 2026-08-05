#!/bin/bash
#SBATCH --job-name=decomp_opt
#SBATCH --output=logs/cluster_decomp_opt_%A_%a.out
#SBATCH --error=logs/cluster_decomp_opt_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --partition cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G # small FCI spaces here (<= ~14k det.s) and cost=variance (no 3-/4-rdms), so far below the 32G used for commutator-cost/medium DMRG runs

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

# Ready-to-run: no CLI parameters needed at submission (sbatch run_cl_num_decomposition_opt.sh).
# One (molecule, basis, bond_length, angle) combination per array task -- a
# non-exhaustive spread of a few geometries per system, just enough to see
# how the K-sectors/K-states plots look, not a real scan.
#
# H2O is run in sto-3g (FCI dim = 441), NOT 6-31g (FCI dim ~= 1.66M): the
# K-states sector analysis (get_ordered_decoupled_states, on by default) does
# a DENSE diagonalization of every symmetry sector, built via one full-space
# matrix-vector product per column of that sector. For a coarse decomposition
# of a ~1.66M-dimensional space, a single sector can already need tens of GB
# just to store as a dense matrix -- a memory wall no amount of --time raises,
# so the system was downgraded instead. N2/sto-3g (dim 14,400) and LiH/6-31g
# (dim 3,025) stay three orders of magnitude below that and are safe even in
# the worst case of one dominant sector.
molecules=(   n2      n2      n2      lih     lih     lih     h2o     h2o     h2o    )
bases=(       sto-3g  sto-3g  sto-3g  6-31g   6-31g   6-31g   sto-3g  sto-3g  sto-3g )
bondlengths=( 1.10    1.80    2.50    1.60    2.50    4.00    0.96    1.50    2.00   )
angles=(      ""      ""      ""      ""      ""      ""      104.5   104.5   104.5  )

molecule="${molecules[$SLURM_ARRAY_TASK_ID]}"
basis="${bases[$SLURM_ARRAY_TASK_ID]}"
bondlength="${bondlengths[$SLURM_ARRAY_TASK_ID]}"
angle="${angles[$SLURM_ARRAY_TASK_ID]}"

# Initialize an array for optional Python flags
OPT_ARGS=()
if [[ -n "$angle" ]]; then
    OPT_ARGS+=(--bond-angle "$angle")
fi

# cost=variance only for this glimpse run: it's the cheap, 1-/2-RDM-only cost
# meant to drive the search (see the module docstring); commutator additionally
# needs the 3-/4-RDM at every beam-search node and is meant for polishing a
# chosen winner afterward, not for a broad first look.
python cluster_number_decomposition_optimization.py "$molecule" "$basis" "$bondlength" variance \
    "${OPT_ARGS[@]}" \
    --initial-basis both
