"""Orbital optimization with MPS-native block2 costs (FCIDUMP-friendly).

Same role as ``optimize_symmetries.py --reference dmrg``, but imports no
pyscf/ffsim so it runs on machines that only have block2 (e.g. native
Windows). Output is an ``x_opt`` text file (JSON metadata header + parameter
vector) consumed by ``metrics.py``, ``rotate_fcidump.py`` and
``solve_dmrg.py --U``.

Supports ``--orbital_rotation {full,irrep}`` (default ``full``). Irrep mode
needs a symmetry-adapted FCIDUMP/chk with distinct ORBSYM / point-group labels.

Example::

    python optimize_dmrg.py hamiltonians/sentest_5_d754.FCIDUMP parity.txt \\
        --cost_function NC --bond_dim 200 --maxiter 20
    python optimize_dmrg.py mol.chk parity.txt --orbital_rotation irrep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.optimize

from src.dmrg_costs import MultiplyConfig, build_dmrg_orbital_costs
from src.dmrg_solver import DMRGConfig
from src.orbital_rotation import n_params, resolve_orbital_rotation
from src.workflow_cli import add_orbital_rotation_arg


def callback(intermediate_result):
    print(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        end=" ",
    )
    # SciPy < 1.11 passes the parameter vector; newer versions pass OptimizeResult.
    if hasattr(intermediate_result, "fun"):
        print("{0:4.6f}".format(intermediate_result.fun))
    else:
        print("(iter)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize orbital rotations with MPS-native NC/variance costs"
    )
    parser.add_argument("molpath", help="Hamiltonian (.FCIDUMP or .chk)")
    parser.add_argument("parity", help="parity-matrix path")
    parser.add_argument("--cost_function", choices=("NC", "variance"), default="NC")
    parser.add_argument("--x0", default=None, help="initial rotation parameters")
    parser.add_argument("--bond_dim", type=int, default=250)
    parser.add_argument("--n_sweeps", type=int, default=20)
    parser.add_argument(
        "--dmrg_seed", type=int, default=None,
        help="reseed block2's RNG before the reference DMRG solve, for a "
             "reproducible reference energy across reruns (default: "
             "unseeded, matching prior behavior; without this, reruns can "
             "drift by ~1e-5-1e-4 Ha)",
    )
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--wavefunction_dir", default=None)
    parser.add_argument(
        "--multiply_bond_dim", type=int, default=None,
        help="literal fit bond dimension for MPO-MPS multiplies; default "
             "derives it from the reference bond dim instead "
             "(ceil(reference_bond_dim * bra_bond_dim_factor))",
    )
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument(
        "--finite_difference", action="store_true",
        help="use L-BFGS-B's own finite-difference gradient instead of "
             "DMRGOrbitalCosts.commutator_and_gradient/variance_and_gradient "
             "(src/dmrg_costs.py), which is the default. The analytic "
             "gradient is ~20-60x fewer DMRG cost evaluations per step and "
             "avoids the truncated-multiply bias/noise that made finite "
             "differences unsafe here (see diagnostics/phase2_analytic_gradient/); "
             "this flag exists for comparison/debugging. Support-1 parity "
             "rows only (analytic path raises NotImplementedError otherwise); "
             "see diagnostics/phase2_analytic_gradient/tier1_tier2_FINDINGS.md "
             "for validation and known bond-dimension-margin caveats.",
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None)
    add_orbital_rotation_arg(parser)
    args = parser.parse_args()

    parity_matrix = np.atleast_2d(np.loadtxt(args.parity, dtype=int))
    store_dir = args.wavefunction_dir or str(
        Path("wavefunctions") / Path(args.molpath).stem
    )

    costs, result, solver = build_dmrg_orbital_costs(
        args.molpath,
        parity_matrix,
        store_dir=store_dir,
        config=DMRGConfig(
            max_bond_dim=args.bond_dim, n_sweeps=args.n_sweeps, seed=args.dmrg_seed,
        ),
        multiply=MultiplyConfig(
            bond_dim=args.multiply_bond_dim,
            n_sweeps=args.multiply_sweeps,
        ),
        reuse=not args.no_reuse,
        n_threads=args.n_threads,
    )
    rotation_pairs, rotation_irreps = resolve_orbital_rotation(
        args.orbital_rotation, args.molpath, solver.n_sites
    )
    costs.pairs = rotation_pairs
    n_free = n_params(solver.n_sites, None)
    n_sym = n_params(solver.n_sites, rotation_pairs)
    print(
        f"orbital_rotation={args.orbital_rotation}: "
        f"N_free={n_free}, N_sym={n_sym}"
        + (f", reduced={n_free - n_sym}" if rotation_pairs is not None else "")
    )
    print("DMRG reference energy: {0:4.6f}".format(result.energy))
    print("wavefunction store: {}".format(result.store_dir))

    use_analytic_gradient = not args.finite_difference
    if use_analytic_gradient:
        f = costs.cost_function_and_gradient(args.cost_function)
    else:
        f = costs.cost_function(args.cost_function)
    x0 = (
        np.zeros(n_params(solver.n_sites, rotation_pairs))
        if args.x0 is None
        else np.loadtxt(args.x0)
    )
    cost_before = f(x0)[0] if use_analytic_gradient else f(x0)
    print("before optimization: {0:4.6f}".format(cost_before))
    res = scipy.optimize.minimize(
        f,
        x0,
        method="L-BFGS-B",
        jac=True if use_analytic_gradient else None,
        options={"maxiter": args.maxiter},
        callback=callback if args.verbose else None,
    )
    print(res.message)
    print("optimized: {0:4.6f}".format(res.fun))
    print("cost evaluations: {}".format(costs.n_evaluations))

    outname = args.outname or (
        time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_x_opt.txt"
    )
    meta = dict(vars(args))
    meta["orbital_rotation"] = args.orbital_rotation
    if rotation_irreps is not None:
        meta["irreps"] = np.asarray(rotation_irreps, dtype=int).tolist()
    with open(outname, "a", newline="", encoding="utf-8") as fp:
        fp.write(json.dumps(meta) + "\n")
        np.savetxt(fp, res.x)
    print("wrote", outname)


if __name__ == "__main__":
    main()
