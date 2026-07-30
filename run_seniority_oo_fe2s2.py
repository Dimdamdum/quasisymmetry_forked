"""
Batch orbital optimization driver: local-seniority quasisymmetries, NC
(non-commutator) cost against the FCI reference state, for the Fe2S2
10e10o FCIDUMP Hamiltonian.

Local (per-orbital) seniority operators are passed via --parity with
hamiltonians/Fe2S2/parity_10_sens.txt, a 10x10 identity matrix — each row
picks out one orbital's local parity/seniority operator, so
parity_matrix_to_quasisymmetries yields 10 separate quasisymmetries
(one per orbital) rather than a single combined/global operator. (The
--seniority CLI flag instead sums all orbitals into one operator before
computing the commutator, which is a *global* seniority quasisymmetry —
not what we want here.)

Runs optimize_symmetries.py as a subprocess, tees its --verbose
per-iteration monitoring output to a log file, and if L-BFGS-B hasn't
converged after --optimizer_maxiter cycles, restarts from the last
rotation (via --x0) for further attempts (up to MAX_ATTEMPTS) so the run
is given every chance to actually converge rather than just stopping at
an iteration cap.

Mirrors run_seniority_oo_n2_minimal.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MOLPATHS = [
    REPO_ROOT / "hamiltonians/Fe2S2/fe2s2_10e10o_FCIDUMP",
]
PARITY_MATRIX = REPO_ROOT / "hamiltonians/Fe2S2/parity_10_sens.txt"  # 10x10 identity -> local seniority per orbital
OUT_DIR = REPO_ROOT / "hamiltonians/Fe2S2/seniority_oo"
LOG_DIR = OUT_DIR / "logs"

MAXITER_PER_ATTEMPT = 100
MAX_ATTEMPTS = 4
PARALLEL_WORKERS = 1

# Sequential runs, but cap BLAS threads: unset, OpenBLAS auto-detects nproc and
# spawns that many workers per process (verified: OPENBLAS_NUM_THREADS alone
# took thread count at import from 23 down to 3; RAYON_NUM_THREADS had no
# effect, so ffsim routes through OpenBLAS rather than its own pool). With one
# worker at a time we can give it nearly the whole machine.
THREAD_CAP = str(max(1, os.cpu_count() - 2))
SUBPROCESS_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": THREAD_CAP,
    "OMP_NUM_THREADS": THREAD_CAP,
    "MKL_NUM_THREADS": THREAD_CAP,
}

_print_lock = threading.Lock()


def log_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def descriptive_base(molpath: Path) -> str:
    return f"{molpath.stem}_local_seniority_NC_fci"


def run_attempt(molpath: Path, base: str, attempt: int, x0_path: "Path | None") -> dict:
    outname = OUT_DIR / f"{base}_attempt{attempt}_result.json"
    output_fcidump = OUT_DIR / f"{base}_attempt{attempt}_rotated_FCIDUMP"
    orbene_npy = OUT_DIR / f"{base}_attempt{attempt}_orbenes.npy"
    logfile = LOG_DIR / f"{base}_attempt{attempt}.log"

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "optimize_symmetries.py"),
        str(molpath),
        str(PARITY_MATRIX),
        "--reference", "fci",
        "--cost_function", "NC",
        "--optimizer_maxiter", str(MAXITER_PER_ATTEMPT),
        "--verbose",
        "--output_fcidump", str(output_fcidump),
        "--orbene_npy", str(orbene_npy),
        "--outname", str(outname),
    ]
    if x0_path is not None:
        cmd += ["--x0", str(x0_path)]

    log_print(f"=== {base}: attempt {attempt} (maxiter={MAXITER_PER_ATTEMPT}, "
              f"x0={x0_path if x0_path else 'zeros'}) ===")
    t0 = time.time()
    with open(logfile, "w") as fp:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=SUBPROCESS_ENV)
        for line in proc.stdout:
            fp.write(line)
            fp.flush()
            with _print_lock:
                sys.stdout.write(f"[{base}] {line}")
        proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"optimize_symmetries.py failed for {molpath} (see {logfile})")

    with open(outname) as fp:
        result = json.load(fp)
    result["_elapsed_wall"] = elapsed
    result["_output_fcidump"] = str(output_fcidump)
    result["_orbene_npy"] = str(orbene_npy)
    result["_logfile"] = str(logfile)
    return result


def write_x0(base: str, attempt: int, rotation: list) -> Path:
    x0_path = OUT_DIR / f"{base}_attempt{attempt}_x_opt.txt"
    with open(x0_path, "w") as fp:
        fp.write("\n".join(repr(v) for v in rotation))
    return x0_path


def optimize_one(molpath: Path) -> dict:
    base = descriptive_base(molpath)
    x0_path = None
    result = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = run_attempt(molpath, base, attempt, x0_path)
        converged = bool(result["converged"])
        log_print(f"[{base}] cost_before={result['cost_before']:.6e}  "
                  f"cost_after={result['cost_after']:.6e}  "
                  f"nit={result['nit']}  converged={converged}  "
                  f"message={result['message']!r}")
        if converged:
            break
        x0_path = write_x0(base, attempt, result["rotation"])
    else:
        log_print(f"[{base}] WARNING: did not converge after {MAX_ATTEMPTS} attempts "
                  f"({MAX_ATTEMPTS * MAXITER_PER_ATTEMPT} total cycles)")
    result["_attempts"] = attempt
    result["_base"] = base
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        summary = list(pool.map(optimize_one, MOLPATHS))

    print("\n=== summary ===")
    for r in summary:
        print(f"{r['_base']}: converged={r['converged']} attempts={r['_attempts']} "
              f"cost_before={r['cost_before']:.6e} cost_after={r['cost_after']:.6e} "
              f"total_nit={r['nit']} fcidump={r['_output_fcidump']} "
              f"orbenes={r['_orbene_npy']}")

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nfull summary written to {summary_path}")


if __name__ == "__main__":
    main()
