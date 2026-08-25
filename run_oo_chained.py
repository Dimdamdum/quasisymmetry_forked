"""Attempt-chaining wrapper around ``optimize_symmetries.py``.

Reproduces the pattern used for the Fe2S2 ``seniority_oo`` FCI run: call
``optimize_symmetries.py`` repeatedly with a fixed ``--optimizer_maxiter``,
feeding each attempt's optimized rotation back in as the next attempt's
``--x0``, until L-BFGS-B reports real convergence (not just "hit the
iteration limit") or ``--max_attempts`` is reached.

Per-attempt raw stdout goes to ``<outdir>/logs/<tag>_attemptN.log``. A
``[tag]``-prefixed copy of everything, plus a one-line cost/convergence
summary per attempt, is appended to ``<outdir>/driver.log``. A final
``<outdir>/summary.json`` records the outcome across all attempts.

Example::

    python run_oo_chained.py hamiltonians/Fe2S2/fe2s2_10e10o_FCIDUMP \\
        hamiltonians/Fe2S2/parity_10_sens.txt \\
        --tag fe2s2_10e10o_FCIDUMP_local_seniority_NC_dmrg \\
        --outdir hamiltonians/Fe2S2/seniority_oo \\
        --reference dmrg --cost_function NC --bond_dim 20 \\
        --wavefunction_dir wavefunctions/fe2s2_10e10o_FCIDUMP_bd20 \\
        --attempt_maxiter 100 --max_attempts 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "darwin":
    # On macOS, block2 links against Apple's Accelerate framework for BLAS/LAPACK,
    # which is itself internally multithreaded. Combined with block2's own
    # --n_threads OpenMP parallelism, that nests two thread pools and causes
    # oversubscription. Capping Accelerate to 1 thread/call measured ~11% faster
    # wall-clock in an isolated A/B test (bd=20, single DMRG solve + NC eval,
    # M5 Pro, 2026-08-11); setdefault so an explicit override still wins.
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# scipy L-BFGS-B's ftol-triggered stop message: relative cost reduction fell
# below ftol. Unlike a gtol stop (small gradient), this can fire after just a
# handful of iterations on a noisy/biased cost (e.g. truncated DMRG), so it
# is not accepted as real convergence below --min_converged_iter.
FTOL_MESSAGE_PREFIX = "CONVERGENCE: RELATIVE REDUCTION OF F"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("molpath")
    parser.add_argument("parity", nargs="?", default=None)
    parser.add_argument("--seniority", action="store_true")
    parser.add_argument("--tag", required=True, help="label used in log lines and output filenames")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--reference", choices=("fci", "hf", "dmrg"), default="fci")
    parser.add_argument("--cost_function", default="NC")
    parser.add_argument("--orbital_rotation", choices=("full", "irrep"), default="full")
    parser.add_argument("--bond_dim", type=int, default=250)
    parser.add_argument("--wavefunction_dir", default=None)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--multiply_bond_dim", type=int, default=None)
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument(
        "--dmrg_seed", type=int, default=None,
        help="forwarded to optimize_symmetries.py --dmrg_seed (--reference dmrg "
             "only): reseed block2's RNG before the reference DMRG solve for a "
             "reproducible reference energy across attempts/reruns",
    )
    parser.add_argument("--sector_backend", choices=("determinant", "clifford"), default="determinant")
    parser.add_argument("--symmetry_manifest", default=None)
    parser.add_argument("--fixed_sector", default=None)
    parser.add_argument("--sector_switch_maxiter", type=int, default=None)
    parser.add_argument("--attempt_maxiter", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument(
        "--min_converged_iter", type=int, default=10,
        help="an attempt reporting converged=True via the ftol message "
             "('CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH') with "
             "fewer than this many L-BFGS-B iterations is treated as a "
             "premature/noise-driven stop, not real convergence, and the "
             "chain keeps going; gtol-triggered convergence is always "
             "accepted regardless of iteration count",
    )
    parser.add_argument("--initial_x0", default=None)
    parser.add_argument("--ftol", type=float, default=None, help="forwarded to optimize_symmetries.py --ftol")
    parser.add_argument("--gtol", type=float, default=None, help="forwarded to optimize_symmetries.py --gtol")
    parser.add_argument("--eps", type=float, default=None, help="forwarded to optimize_symmetries.py --eps")
    parser.add_argument(
        "--finite_difference", action="store_true",
        help="forwarded to optimize_symmetries.py --finite_difference (--reference "
             "dmrg only): use L-BFGS-B's own finite-difference gradient instead of "
             "the analytic gradient, which is the default",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    driver_log_path = outdir / "driver.log"

    def log(line: str, prefix: bool = True) -> None:
        text = f"[{args.tag}] {line}" if prefix else line
        print(text, flush=True)
        with open(driver_log_path, "a", encoding="utf-8") as fp:
            fp.write(text + "\n")

    x0_path = args.initial_x0
    attempt = 1
    per_attempt = []

    while True:
        outname = outdir / f"{args.tag}_attempt{attempt}_result.json"
        output_fcidump = outdir / f"{args.tag}_attempt{attempt}_rotated_FCIDUMP"
        orbene_npy = outdir / f"{args.tag}_attempt{attempt}_orbenes.npy"
        attempt_log_path = logs_dir / f"{args.tag}_attempt{attempt}.log"

        cmd = [args.python, "-u", "optimize_symmetries.py", args.molpath]
        if args.parity:
            cmd.append(args.parity)
        if args.seniority:
            cmd.append("--seniority")
        cmd += [
            "--reference", args.reference,
            "--cost_function", args.cost_function,
            "--orbital_rotation", args.orbital_rotation,
            "--optimizer_maxiter", str(args.attempt_maxiter),
            "--outname", str(outname),
            "--output_fcidump", str(output_fcidump),
            "--orbene_npy", str(orbene_npy),
            "--verbose",
        ]
        if args.reference == "dmrg":
            cmd += [
                "--bond_dim", str(args.bond_dim),
                "--n_threads", str(args.n_threads),
                "--multiply_sweeps", str(args.multiply_sweeps),
            ]
            if args.wavefunction_dir:
                cmd += ["--wavefunction_dir", args.wavefunction_dir]
            if args.multiply_bond_dim is not None:
                cmd += ["--multiply_bond_dim", str(args.multiply_bond_dim)]
            if args.dmrg_seed is not None:
                cmd += ["--dmrg_seed", str(args.dmrg_seed)]
            if args.finite_difference:
                cmd += ["--finite_difference"]
        else:
            if args.sector_backend != "determinant":
                cmd += ["--sector_backend", args.sector_backend]
            if args.symmetry_manifest:
                cmd += ["--symmetry_manifest", args.symmetry_manifest]
            if args.fixed_sector:
                cmd += ["--fixed_sector", args.fixed_sector]
            if args.sector_switch_maxiter is not None:
                cmd += ["--sector_switch_maxiter", str(args.sector_switch_maxiter)]
        if x0_path:
            cmd += ["--x0", str(x0_path)]
        if args.ftol is not None:
            cmd += ["--ftol", str(args.ftol)]
        if args.gtol is not None:
            cmd += ["--gtol", str(args.gtol)]
        if args.eps is not None:
            cmd += ["--eps", str(args.eps)]

        x0_desc = x0_path if x0_path else "zeros"
        log(f"=== {args.tag}: attempt {attempt} (maxiter={args.attempt_maxiter}, x0={x0_desc}) ===", prefix=False)

        with open(attempt_log_path, "w", encoding="utf-8", buffering=1) as attempt_log_fp, \
             open(driver_log_path, "a", encoding="utf-8", buffering=1) as driver_log_fp:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                print(line, flush=True)
                attempt_log_fp.write(line + "\n")
                driver_log_fp.write(f"[{args.tag}] {line}\n")
            returncode = proc.wait()

        if returncode != 0:
            log(f"attempt {attempt} FAILED (exit code {returncode}); aborting chain")
            raise SystemExit(returncode)

        with open(outname) as fp:
            result = json.load(fp)

        cost_before = result["cost_before"]
        cost_after = result["cost_after"]
        nit = result["nit"]
        converged = result["converged"]
        message = result["message"]

        premature = (
            converged
            and message.startswith(FTOL_MESSAGE_PREFIX)
            and nit < args.min_converged_iter
        )
        # A premature ftol stop after nit<min_converged_iter iterations is
        # normally treated as a noise-driven false convergence, not real
        # convergence. But if this attempt was warm-started at (essentially)
        # the cost the previous attempt ended at, and didn't move, it isn't
        # noise -- it's L-BFGS-B correctly re-confirming an already-converged
        # point. Rejecting that just burns another full attempt for an
        # identical result (see diagnostics/phase5_end_to_end/FINDINGS.md,
        # n2_1_50's chain).
        resumed_at_prior_optimum = (
            premature
            and per_attempt
            and abs(cost_before - per_attempt[-1]["cost_after"]) < 1e-10
            and abs(cost_after - cost_before) < 1e-10
        )
        accepted_converged = converged and (not premature or resumed_at_prior_optimum)

        log(
            f"cost_before={cost_before:.6e}  cost_after={cost_after:.6e}  "
            f"nit={nit}  converged={converged}  message={message!r}"
            + ("  [resumed at prior attempt's optimum; accepted despite nit<{}]".format(
                   args.min_converged_iter)
               if resumed_at_prior_optimum else
               "  [premature ftol stop at nit<{}, not accepted as real "
               "convergence; continuing chain]".format(args.min_converged_iter)
               if premature else "")
        )
        per_attempt.append({
            "attempt": attempt,
            "outname": str(outname),
            "output_fcidump": str(output_fcidump),
            "orbene_npy": str(orbene_npy),
            "cost_before": cost_before,
            "cost_after": cost_after,
            "nit": nit,
            "converged": converged,
            "accepted_converged": accepted_converged,
            "message": message,
        })

        x_opt_path = outdir / f"{args.tag}_attempt{attempt}_x_opt.txt"
        np.savetxt(x_opt_path, np.asarray(result["rotation"]))
        x0_path = str(x_opt_path)

        if accepted_converged or attempt >= args.max_attempts:
            break
        attempt += 1

    total_nit = sum(a["nit"] for a in per_attempt)
    summary = {
        "tag": args.tag,
        "converged": per_attempt[-1]["accepted_converged"],
        "attempts": len(per_attempt),
        "cost_before": per_attempt[0]["cost_before"],
        "cost_after": per_attempt[-1]["cost_after"],
        "total_nit": total_nit,
        "fcidump": per_attempt[-1]["output_fcidump"],
        "orbene_npy": per_attempt[-1]["orbene_npy"],
        "per_attempt": per_attempt,
    }
    log("", prefix=False)
    log("=== summary ===", prefix=False)
    log(
        f"{args.tag}: converged={summary['converged']} attempts={summary['attempts']} "
        f"cost_before={summary['cost_before']:.6e} cost_after={summary['cost_after']:.6e} "
        f"total_nit={summary['total_nit']} fcidump={summary['fcidump']}",
        prefix=False,
    )

    summary_path = outdir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    log("", prefix=False)
    log(f"full summary written to {summary_path}", prefix=False)


if __name__ == "__main__":
    main()
