#!/bin/bash
#SBATCH --job-name=cl_num_sector_search
#SBATCH --output=logs/cl_num_sector_search_%A_%a.out
#SBATCH --error=logs/cl_num_sector_search_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --array=0-29
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

# One array task per FCIDUMP Hamiltonian from
# Ai_exploration_3/generate_extra_hamiltonians.py's batch (30 total:
# butadiene, C2, O2, H2O, H6 chain, H6 ring x 5 geometries each). Job
# metadata is read from that run's own summary JSON instead of being
# hardcoded here, so this script stays correct if the Hamiltonians are ever
# regenerated with different geometries/counts. molecule/basis are used by
# cluster_number_sector_search.py only to label the output/plots directory
# tree when --fcidump is given; bondlength/angle are not used at all in that
# mode (still required/accepted positionally) -- all are passed through
# anyway purely so the launch command and SLURM logs stay self-documenting
# (e.g. butadiene's torsion angle in place of a literal bond length).
SUMMARY_JSON="hamiltonians/cluster_number_hamiltonians/generate_extra_hamiltonians_summary.json"

JOB_FIELDS=$(python -c "
import json
with open('$SUMMARY_JSON') as f:
    data = json.load(f)
r = data['summary'][$SLURM_ARRAY_TASK_ID]
fcidump = r.get('fcidump_path') or r['full_space_fcidump_path']
angle = r.get('bond_angle')
fields = [str(r['mol_name']), str(r['basis']), str(r['geom_param']),
          '' if angle is None else str(angle), fcidump]
print('|'.join(fields))
")
IFS='|' read -r molecule basis bondlength angle fcidump <<< "$JOB_FIELDS"

echo "Task $SLURM_ARRAY_TASK_ID: molecule=$molecule basis=$basis bondlength=$bondlength angle=$angle fcidump=$fcidump"

OPT_ARGS=(--fcidump "$fcidump")
if [[ -n "$angle" ]]; then
    OPT_ARGS+=(--bond-angle "$angle")
fi

python cluster_number_sector_search.py "$molecule" "$basis" "$bondlength" variance "${OPT_ARGS[@]}" --K-sector-analysis --num-sectors-to-retain 40