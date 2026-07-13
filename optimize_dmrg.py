"""Orbital optimization with MPS-native block2 costs (FCIDUMP-friendly).

Same role as ``optimize_symmetries.py --backend dmrg``, but imports no
pyscf/ffsim so it runs on machines that only have block2 (e.g. native
Windows). Output is the same ``x_opt`` text file consumed by
``metrics.py --U``, ``rotate_fcidump.py`` and ``solve_dmrg.py --U``.

Example::

    python optimize_dmrg.py hamiltonians/sentest_5_d754.FCIDUMP parity.txt \\
        --cost_function NC --bond_dim 200 --maxiter 20
"""

from __future__ import annotations

import argparse
import time
from math import comb
from pathlib import Path

import numpy as np
import scipy.optimize

from src.dmrg_costs import MultiplyConfig, build_dmrg_orbital_costs
from src.dmrg_solver import DMRGConfig


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
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--wavefunction_dir", default=None)
    parser.add_argument("--multiply_bond_dim", type=int, default=None)
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None)
    args = parser.parse_args()

    parity_matrix = np.atleast_2d(np.loadtxt(args.parity, dtype=int))
    store_dir = args.wavefunction_dir or str(
        Path("wavefunctions") / Path(args.molpath).stem
    )

    costs, result, solver = build_dmrg_orbital_costs(
        args.molpath,
        parity_matrix,
        store_dir=store_dir,
        config=DMRGConfig(max_bond_dim=args.bond_dim, n_sweeps=args.n_sweeps),
        multiply=MultiplyConfig(
            bond_dim=args.multiply_bond_dim,
            n_sweeps=args.multiply_sweeps,
        ),
        reuse=not args.no_reuse,
        n_threads=args.n_threads,
    )
    print("DMRG reference energy: {0:4.6f}".format(result.energy))
    print("wavefunction store: {}".format(result.store_dir))

    f = costs.cost_function(args.cost_function)
    x0 = (
        np.zeros(comb(solver.n_sites, 2))
        if args.x0 is None
        else np.loadtxt(args.x0)
    )
    print("before optimization: {0:4.6f}".format(f(x0)))
    res = scipy.optimize.minimize(
        f,
        x0,
        method="L-BFGS-B",
        options={"maxiter": args.maxiter},
        callback=callback if args.verbose else None,
    )
    print(res.message)
    print("optimized: {0:4.6f}".format(res.fun))
    print("cost evaluations: {}".format(costs.n_evaluations))

    outname = args.outname or (
        time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_x_opt.txt"
    )
    with open(outname, "a", newline="", encoding="utf-8") as fp:
        fp.write(str(vars(args)) + "\n")
        np.savetxt(fp, res.x)
    print("wrote", outname)


if __name__ == "__main__":
    main()
