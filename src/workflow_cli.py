"""Shared CLI vocabulary for optimize and metrics.

``--reference`` (both scripts)
    Which wavefunction / energy is treated as truth.

``--backend`` (metrics only)
    Which sector eigensolver to use. Optimize does not take ``--backend``:
    ``--reference`` alone picks the cost engine
    (``fci`` / ``hf`` → CI statevector costs; ``dmrg`` → Block2 MPS costs).

metrics.py backends
    ``fci``       scipy eigsh / dense eigh on each sector block
    ``davidson``  PySCF Davidson on the same sector blocks
    ``dmrg``      Block2 sector-targeted DMRG (E_dec / K)

``dmrg`` always means Block2. Shared flags: ``--bond_dim``,
``--wavefunction_dir``, ``--n_threads``.
"""

from __future__ import annotations

import argparse

REFERENCE_CHOICES = ("fci", "hf", "dmrg")
METRICS_REFERENCE_CHOICES = ("fci", "dmrg")
METRICS_BACKEND_CHOICES = ("fci", "dmrg", "davidson")

OPTIMIZE_EPILOG = """
--reference picks both the wavefunction and the cost engine
-----------------------------------------------------------
  --reference fci     PySCF FCI CI vector + ffsim costs (default)
  --reference hf      Hartree-Fock CI vector + ffsim costs
  --reference dmrg    Block2 MPS + MPS-native NC/variance

  Sector energy costs (decoupled / fixed_sector / switching_sector)
  require --reference fci or hf (CI / ffsim path).

examples
--------
  python optimize_symmetries.py mol.FCIDUMP parity.txt
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference hf
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference dmrg --bond_dim 250
"""

METRICS_EPILOG = """
valid combinations
------------------
  --reference fci    --backend fci        # eigsh/eigh sectors (default)
  --reference fci    --backend davidson   # PySCF Davidson on same blocks
  --reference dmrg   --backend dmrg       # Block2 sector DMRG

  Omitting --reference uses: dmrg if --backend dmrg, else fci.
  --solver is an alias of --backend.

examples
--------
  python metrics.py oo.json
  python metrics.py oo.json --backend davidson --davidson_tol 1e-10
  python metrics.py oo.json --backend dmrg --bond_dim 250 --penalty 30
"""


def add_dmrg_common_args(parser: argparse.ArgumentParser) -> None:
    """Bond dimension / store / threads used by any ``dmrg`` path."""
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=250,
        help="Block2 DMRG bond dimension (only when reference or backend is dmrg)",
    )
    parser.add_argument(
        "--wavefunction_dir",
        default=None,
        help="directory for Block2 MPS files (reuse across runs)",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Block2 OpenMP thread count",
    )


def add_optimize_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` for ``optimize_symmetries.py`` (no ``--backend``)."""
    parser.add_argument(
        "--reference",
        choices=REFERENCE_CHOICES,
        default="fci",
        metavar="{fci,hf,dmrg}",
        help=(
            "REFERENCE STATE and cost engine: "
            "fci=PySCF FCI + ffsim costs (default); "
            "hf=Hartree-Fock + ffsim costs; "
            "dmrg=Block2 MPS + MPS-native NC/variance. "
            "Sector energy costs need fci or hf."
        ),
    )
    add_dmrg_common_args(parser)
    _attach_epilog(parser, OPTIMIZE_EPILOG)


def add_metrics_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` / ``--backend`` for ``metrics.py``."""
    parser.add_argument(
        "--backend",
        "--solver",
        dest="backend",
        choices=METRICS_BACKEND_CHOICES,
        default="fci",
        metavar="{fci,davidson,dmrg}",
        help=(
            "SECTOR SOLVER: fci=eigsh/eigh on each sector block (default); "
            "davidson=PySCF Davidson on the same blocks; "
            "dmrg=Block2 sector-targeted DMRG. "
            "--solver is a deprecated alias of --backend."
        ),
    )
    parser.add_argument(
        "--reference",
        choices=METRICS_REFERENCE_CHOICES,
        default=None,
        metavar="{fci,dmrg}",
        help=(
            "REFERENCE ENERGY/STATE for dE and reference-ordered K "
            "(fci=PySCF FCI, dmrg=Block2 GS). "
            "Default: dmrg when --backend dmrg, else fci. "
            "Must match the backend (fci with fci|davidson; dmrg with dmrg)."
        ),
    )
    add_dmrg_common_args(parser)
    _attach_epilog(parser, METRICS_EPILOG)


def resolve_metrics_reference(backend: str, reference: str | None) -> str:
    """Default metrics reference from backend when the user omits --reference."""
    if reference is not None:
        return reference
    return "dmrg" if backend == "dmrg" else "fci"


def optimize_cost_engine(reference: str) -> str:
    """Derived cost engine label for optimize banners / JSON."""
    return "dmrg" if reference == "dmrg" else "statevector"


def validate_metrics_workflow(parser: argparse.ArgumentParser, args) -> None:
    """Reject illegal --reference/--backend pairs for metrics."""
    args.reference = resolve_metrics_reference(args.backend, args.reference)
    allowed = {
        "fci": {"fci"},
        "davidson": {"fci"},
        "dmrg": {"dmrg"},
    }
    if args.reference not in allowed[args.backend]:
        need = " or ".join(sorted(allowed[args.backend]))
        parser.error(
            f"invalid combination: --reference {args.reference} --backend {args.backend}\n"
            f"  With --backend {args.backend} you must use --reference {need}.\n"
            "  Valid pairs:\n"
            "    --reference fci   --backend fci|davidson\n"
            "    --reference dmrg  --backend dmrg"
        )


def print_workflow_banner(script: str, reference: str, backend: str | None = None, **extra) -> None:
    """Print a short resolved-settings banner so the run mode is obvious."""
    lines = [
        f"[workflow] reference={reference}  (wavefunction / energy used as truth)",
    ]
    if script == "optimize":
        engine = backend or optimize_cost_engine(reference)
        lines.append(f"[workflow] cost_engine={engine}  (from --reference)")
    elif backend is not None:
        lines.append(f"[workflow] backend={backend}  (sector solver)")
    for key, value in extra.items():
        if value is not None:
            lines.append(f"[workflow] {key}={value}")
    print("\n".join(lines), flush=True)


def _attach_epilog(parser: argparse.ArgumentParser, epilog: str) -> None:
    """Append recipes to the parser epilog without clobbering an existing one."""
    existing = parser.epilog or ""
    parser.epilog = (existing + "\n" + epilog).strip()
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
