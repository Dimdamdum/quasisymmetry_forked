"""MPS-native orbital-optimization costs for the quasisymmetry pipeline.

The statevector cost in ``optimize_symmetries`` is

    NC(x) = sum_k ||[H(U), S_k] U|ψ⟩||²
          = sum_k ||[H, U^dagger S_k U] |ψ⟩||²

with ``U = expm(A(x))``. This module evaluates the right-hand side entirely
with block2 MPO/MPS operations on a **fixed** DMRG reference ``|ψ⟩``:

* DMRG is run once (or reloaded from a local wavefunction store);
* ``η = H|ψ⟩`` is cached;
* each optimizer step builds the rotated parity operators
  ``S̃_k = U^dagger S_k U`` as factor MPOs and applies them by multiply;
* the NC residual is ``||H S̃|ψ⟩ - S̃|η⟩||²`` and the variance cost is
  ``1 - |⟨ψ|S̃|ψ⟩|²``.

Output of the optimizer is still the rotation vector ``x``, so
``rotate_fcidump.py``, ``metrics.py`` and ``solve_dmrg.py --U`` stay
unchanged.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import block2
import numpy as np

from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    DMRGResult,
    rotation_from_parameters,
    solve_or_load_ground_state,
)
from src.orbital_rotation import params_to_U_and_frechet

logger = logging.getLogger(__name__)

# Process-wide, monotonically increasing -- gives every DMRGOrbitalCosts
# instance a unique tag namespace. Sharing a store_dir across sequential
# instances (e.g. one per bond-dim factor in a sweep script) is a real,
# already-used pattern; a per-instance-only counter starting at 0 every
# time made every instance's "COST_KET"/first-eta tag collide with every
# other instance's, silently corrupting whichever one was still alive when
# a sibling was constructed (found by adversarial review, see
# diagnostics/phase2_analytic_gradient/adversarial_review_round1.md finding 1).
_next_instance_id = itertools.count()


@dataclass(frozen=True)
class MultiplyConfig:
    """Sweep settings for MPO–MPS multiplies used inside the cost."""

    bond_dim: int | None = None
    """Literal fit bond dimension for every MPO-MPS multiply. When left at
    the default ``None``, the fit bond dimension is instead derived from the
    reference MPS as ``ceil(reference_bond_dim * bra_bond_dim_factor)``."""
    n_sweeps: int = 8
    tol: float = 1e-10
    bra_bond_dim_factor: float = 1.5
    """Extra room for ``H|φ⟩`` / ``S̃|η⟩`` relative to the reference bond dim,
    used only when ``bond_dim`` is left at its default ``None``."""


class DMRGOrbitalCosts:
    """Callable NC / variance costs over orbital-rotation parameters ``x``.

    Parameters
    ----------
    solver:
        A :class:`Block2DMRGSolver` whose ground-state MPS is already solved
        (or will be loaded via ``mps_tag``).
    parity_matrix:
        Quasi-symmetry incidence matrix (``n_sym × norb`` or
        ``n_sym × 2 norb``), same convention as ``optimize_symmetries``.
    mps_tag:
        Tag of the reference MPS inside ``solver.store_dir``.
    multiply:
        Fit settings for intermediate MPO–MPS multiplies.
    """

    def __init__(
        self,
        solver: Block2DMRGSolver,
        parity_matrix: np.ndarray,
        mps_tag: str = "GS",
        multiply: MultiplyConfig | None = None,
        pairs=None,
        rng_seed: int = 12345,
    ) -> None:
        self.solver = solver
        self.parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
        self.mps_tag = mps_tag
        self.multiply = multiply or MultiplyConfig()
        self.pairs = pairs
        self._eval_count = 0
        self._tag_serial = 0
        self._instance_id = next(_next_instance_id)
        # get_random_mps (src/dmrg_solver.py's apply_mpo) draws from block2's
        # global RNG on every MPO-MPS multiply, with no seeding anywhere in
        # this module. Without a fixed reseed here, commutator()/variance()
        # are not pure functions of x: the same x evaluated twice can return
        # different values, since the RNG stream position depends on how many
        # prior multiplies happened this session (including the one-time,
        # lazily-cached eta build). That silently injects noise into every
        # finite-difference gradient the optimizer takes. Reseeding
        # identically at the top of every call makes the cost reproducible
        # for a repeated x (validated: bit-identical across repeated calls).
        self._rng_seed = rng_seed

        self.solver._activate()
        # Working copy of the reference MPS. block2 multiply may rewrite the
        # ket's on-disk tensors while sweeping; never point this at the stored
        # "GS" tag or later reloads / simultaneous multiplies will corrupt it.
        stored = self.solver.get_mps(mps_tag)
        self.ket = stored.deep_copy(f"COST_KET_{self._instance_id}")
        self._h_mpo = self.solver.hamiltonian_mpo()
        self._eta = None  # cached H|ψ⟩ (also a working copy)
        self._ref_bond_dim = self._bond_dim_of(self.ket)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _bond_dim_of(mps) -> int:
        try:
            return max(int(mps.info.get_max_bond_dimension()), 2)
        except Exception:
            try:
                return max(int(mps.info.bond_dim), 2)
            except Exception:
                return 200

    def _next_tag(self, prefix: str) -> str:
        self._tag_serial += 1
        return f"{prefix}_{self._instance_id}_{self._tag_serial}"

    def _bra_bond_dim(self) -> int:
        if self.multiply.bond_dim is not None:
            # Explicit override: use it as the literal final fit bond
            # dimension. bra_bond_dim_factor applies only to the
            # reference-derived default below, so it is never applied twice.
            return max(2, int(self.multiply.bond_dim))
        return max(
            2, int(np.ceil(self._ref_bond_dim * self.multiply.bra_bond_dim_factor))
        )

    def _apply(self, mpo, ket, prefix: str, bond_dim: int | None = None):
        # Defensive copy so the source ket's disk files are not rewritten.
        ket_tag = self._next_tag(f"{prefix}_SRC")
        ket_work = ket.deep_copy(ket_tag)
        try:
            return self.solver.apply_mpo(
                mpo,
                ket=ket_work,
                tag=self._next_tag(prefix),
                bond_dim=bond_dim or self._bra_bond_dim(),
                n_sweeps=self.multiply.n_sweeps,
                tol=self.multiply.tol,
            )
        finally:
            # The defensive copy is single-use: consumed entirely inside the
            # multiply call above, never referenced again. Deleting it here
            # (rather than leaving every one of them on disk forever) is the
            # fix for the ~2.5M-orphaned-file growth found in
            # diagnostics/REMEDIATION_PLAN.md's Phase 2 item 4 / Phase 2
            # item 4 wavefunction-store-hygiene finding.
            self.solver.delete_mps_tag(ket_work.info.tag)

    def _apply_symmetry(self, row: np.ndarray, rotation: np.ndarray, ket, prefix: str):
        ket_work = ket.deep_copy(self._next_tag(f"{prefix}_SRC"))
        try:
            return self.solver.apply_rotated_parity(
                row,
                rotation,
                ket=ket_work,
                tag=self._next_tag(prefix),
                bond_dim=self._bra_bond_dim(),
                n_sweeps=self.multiply.n_sweeps,
                tol=self.multiply.tol,
            )
        finally:
            self.solver.delete_mps_tag(ket_work.info.tag)

    def _ensure_eta(self) -> None:
        if self._eta is None:
            logger.info("caching H|psi> for MPS-native costs")
            # Reseed before this one-time build too: it is not covered by
            # commutator()/variance()'s own reseed (which only runs after
            # this method returns), so without this line eta's random
            # initial guess would depend on whatever RNG state preceded the
            # first cost evaluation on this object — different across fresh
            # processes and across sequential DMRGOrbitalCosts objects in
            # one process, even at the same x.
            block2.Random.rand_seed(self._rng_seed)
            self._eta = self._apply(self._h_mpo, self.ket, "ETA")

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def _cleanup(self, *mps_objs) -> None:
        """Delete the on-disk files for one or more transient MPS results.

        Every argument must be a within-this-call scratch result (an
        ``_apply``/``_apply_symmetry`` return value) that nothing needs
        after this row's contribution has been read off it -- never
        ``self.ket``/``self._eta``, which live for the whole object.
        """
        for mps in mps_objs:
            if mps is not None:
                self.solver.delete_mps_tag(mps.info.tag)

    def variance(self, x: np.ndarray) -> float:
        """``sum_k (1 - |<ψ|U^dagger S_k U|ψ>|^2)``."""
        self._eval_count += 1
        # Reseed so every call starts its MPO-MPS multiplies (get_random_mps)
        # from the same RNG state, making this a pure function of x. See the
        # note on ``rng_seed`` in __init__.
        block2.Random.rand_seed(self._rng_seed)
        rotation = rotation_from_parameters(x, self.solver.n_sites, self.pairs)
        total = 0.0
        for row in self.parity_matrix:
            created: list = []
            try:
                phi = self._apply_symmetry(row, rotation, self.ket, "VPHI")
                created.append(phi)
                # S̃ is Hermitian and involutory, so <ψ|S̃|ψ> = <ψ|φ>
                expectation = np.real(self.solver.mps_overlap(self.ket, phi))
                total += 1.0 - expectation ** 2
            finally:
                self._cleanup(*created)
        return float(total)

    def commutator(self, x: np.ndarray) -> float:
        """``sum_k ||[H, U^dagger S_k U] |ψ>||^2``."""
        self._eval_count += 1
        self._ensure_eta()
        # Reseed again after _ensure_eta (which reseeds and consumes RNG
        # state for its own one-time build, see the comment there) so the
        # row loop below always starts from the same state, on the first
        # call and every subsequent one alike.
        block2.Random.rand_seed(self._rng_seed)
        rotation = rotation_from_parameters(x, self.solver.n_sites, self.pairs)
        total = 0.0
        for row in self.parity_matrix:
            created: list = []
            try:
                # Each intermediate is appended to `created` immediately, so
                # a mid-row exception (a real block2 multiply failure, e.g.)
                # still cleans up whatever succeeded before it -- appending
                # only after the whole chain finished (the original shape
                # here) left earlier successes leaked on any later failure
                # in the same row; found by adversarial review, see
                # diagnostics/phase2_analytic_gradient/
                # adversarial_review_round1.md finding 2.
                phi = self._apply_symmetry(row, rotation, self.ket, "CPHI")
                created.append(phi)
                xi = self._apply_symmetry(row, rotation, self._eta, "CXI")
                created.append(xi)
                chi = self._apply(self._h_mpo, phi, "CCHI")
                created.append(chi)
                # ||chi - xi||^2 = <chi|chi> + <xi|xi> - 2 Re <chi|xi>
                c2 = self.solver.mps_norm2(chi)
                x2 = self.solver.mps_norm2(xi)
                cx = self.solver.mps_overlap(chi, xi)
                total += c2 + x2 - 2.0 * float(np.real(cx))
            finally:
                self._cleanup(*created)
        return float(total)

    # ------------------------------------------------------------------
    # Analytic gradients (Phase 2 item 3 of diagnostics/REMEDIATION_PLAN.md)
    # ------------------------------------------------------------------
    #
    # Support-1 spatial parity rows only -- exactly Fe2S2/N2's real
    # production shape (both use the 10x10 identity parity matrix), and the
    # only shape validated so far. See diagnostics/phase2_analytic_gradient/
    # for the derivation (three independently-developed plans, two hostile
    # critiques, one dropped for a proven control-flow error), the
    # synthesis, and the validation ladder (Tier 0 dense/exact: passed;
    # Tier 1/2 block2 MPS: in progress). Wider support (multiple orbitals
    # per row) is validated mathematically (Tier 0) but not implemented
    # here yet -- raises NotImplementedError rather than silently computing
    # something wrong.

    def _support1_spatial_orbital(self, row: np.ndarray) -> int:
        row = np.asarray(row)
        if row.shape[0] != self.solver.n_sites:
            raise NotImplementedError(
                "analytic gradient only supports spatial (norb-length) parity "
                "rows so far (alpha=beta=True per orbital); spin-resolved rows "
                "are not yet implemented"
            )
        nonzero = np.flatnonzero(row)
        if len(nonzero) != 1:
            raise NotImplementedError(
                "analytic gradient only supports support-1 parity rows so far "
                f"(this row has support {len(nonzero)}); see diagnostics/"
                "phase2_analytic_gradient/SYNTHESIS.md risk 3 for the general "
                "(mathematically validated at Tier 0, not yet implemented) case"
            )
        return int(nonzero[0])

    @staticmethod
    def _density_derivative_kernel(
        D: np.ndarray, dm1: np.ndarray, dm2: np.ndarray
    ) -> np.ndarray:
        """``M[m,n] = d<a|F_p(D)|b>/dD_mn`` given the transition 1pdm/2pdm
        between MPS ``a`` (bra) and ``b`` (ket), for
        ``F_p(D) = 1 - 2 n_a(D) - 2 n_b(D) + 4 n_a(D) n_b(D)`` (the rotated
        single-orbital parity factor -- see
        ``Block2DMRGSolver._spin_parity_factor_mpo``). Index convention
        empirically verified against exact dense arithmetic in
        ``diagnostics/phase2_analytic_gradient/archive/tier1_convention_check.py``.
        """
        t1_alpha, t1_beta = dm1[0], dm1[1]
        t2_ab = dm2[1]  # block2's "ab" (mixed-spin) 2pdm block
        M = -2.0 * (t1_alpha + t1_beta)
        M = M + 4.0 * np.einsum("st,mnst->mn", D, t2_ab)
        M = M + 4.0 * np.einsum("qr,qrmn->mn", D, t2_ab)
        return M

    def _rotation_and_frechet(self, x: np.ndarray):
        return params_to_U_and_frechet(x, self.solver.n_sites, self.pairs)

    def commutator_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Analytic ``(commutator(x), grad)`` pair. Support-1 rows only."""
        self._eval_count += 1
        self._ensure_eta()
        block2.Random.rand_seed(self._rng_seed)
        x = np.asarray(x, dtype=float)
        rotation, L = self._rotation_and_frechet(x)
        n_par = L.shape[0]

        total_cost = 0.0
        total_grad = np.zeros(n_par)
        for row in self.parity_matrix:
            p = self._support1_spatial_orbital(row)
            D_p = np.outer(rotation[p, :], rotation[p, :])

            created: list = []
            try:
                phi = self._apply_symmetry(row, rotation, self.ket, "GPHI")
                created.append(phi)
                xi = self._apply_symmetry(row, rotation, self._eta, "GXI")
                created.append(xi)
                chi = self._apply(self._h_mpo, phi, "GCHI")
                created.append(chi)
                h2phi = self._apply(self._h_mpo, chi, "GH2PHI")
                created.append(h2phi)
                hxi = self._apply(self._h_mpo, xi, "GHXI")
                created.append(hxi)

                c2 = self.solver.mps_norm2(chi)
                x2 = self.solver.mps_norm2(xi)
                cx = self.solver.mps_overlap(chi, xi)
                total_cost += c2 + x2 - 2.0 * float(np.real(cx))

                dm1_a, dm2_a = self.solver.transition_density_matrices(h2phi, self.ket)
                dm1_b, dm2_b = self.solver.transition_density_matrices(hxi, self.ket)
                dm1_c, dm2_c = self.solver.transition_density_matrices(chi, self._eta)
                dm1_d, dm2_d = self.solver.transition_density_matrices(xi, self._eta)

                M_total = (
                    self._density_derivative_kernel(D_p, dm1_a, dm2_a)
                    - self._density_derivative_kernel(D_p, dm1_b, dm2_b)
                    - self._density_derivative_kernel(D_p, dm1_c, dm2_c)
                    + self._density_derivative_kernel(D_p, dm1_d, dm2_d)
                )

                for i in range(n_par):
                    dD_p_i = (
                        np.outer(L[i, p, :], rotation[p, :])
                        + np.outer(rotation[p, :], L[i, p, :])
                    )
                    total_grad[i] += 2.0 * float(np.real(np.sum(M_total * dD_p_i)))
            finally:
                self._cleanup(*created)

        return float(total_cost), total_grad

    def variance_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Analytic ``(variance(x), grad)`` pair. Support-1 rows only."""
        self._eval_count += 1
        block2.Random.rand_seed(self._rng_seed)
        x = np.asarray(x, dtype=float)
        rotation, L = self._rotation_and_frechet(x)
        n_par = L.shape[0]

        # <psi|...|psi> is x-independent (ordinary, not transition, RDM) --
        # computed once and reused across every row, unlike commutator's
        # per-row transition RDMs.
        dm1_psi, dm2_psi = self.solver.transition_density_matrices(self.ket, self.ket)

        total_cost = 0.0
        total_grad = np.zeros(n_par)
        for row in self.parity_matrix:
            p = self._support1_spatial_orbital(row)
            D_p = np.outer(rotation[p, :], rotation[p, :])

            created: list = []
            try:
                phi = self._apply_symmetry(row, rotation, self.ket, "VGPHI")
                created.append(phi)
                e = float(np.real(self.solver.mps_overlap(self.ket, phi)))
                total_cost += 1.0 - e ** 2

                M = self._density_derivative_kernel(D_p, dm1_psi, dm2_psi)
                for i in range(n_par):
                    dD_p_i = (
                        np.outer(L[i, p, :], rotation[p, :])
                        + np.outer(rotation[p, :], L[i, p, :])
                    )
                    de_i = float(np.real(np.sum(M * dD_p_i)))
                    total_grad[i] += -2.0 * e * de_i
            finally:
                self._cleanup(*created)

        return float(total_cost), total_grad

    def cost_function(self, kind: str = "NC") -> Callable[[np.ndarray], float]:
        """Return a scipy-optimize-compatible objective ``f(x)``."""
        kind = kind.lower()
        if kind in ("nc", "commutator"):
            return self.commutator
        if kind == "variance":
            return self.variance
        raise ValueError("cost kind must be 'NC' or 'variance'")

    def cost_function_and_gradient(
        self, kind: str = "NC"
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        """Return a ``scipy.optimize.minimize(..., jac=True)``-compatible
        ``f(x) -> (value, grad)`` objective. Support-1 parity rows only
        (see ``commutator_and_gradient``/``variance_and_gradient``)."""
        kind = kind.lower()
        if kind in ("nc", "commutator"):
            return self.commutator_and_gradient
        if kind == "variance":
            return self.variance_and_gradient
        raise ValueError("cost kind must be 'NC' or 'variance'")

    @property
    def n_evaluations(self) -> int:
        return self._eval_count


#: Active-space CI dimension above which the reference-quality print check
#: doesn't bother computing exact FCI. Every real system in this repo (Fe2S2
#: and all three N2 geometries) sits at dim=63,504, ~30x under this -- this
#: is just a cheap sanity cap, not a physics threshold.
_REFERENCE_QUALITY_MAX_FCI_DIMENSION = 2_000_000


def _print_reference_quality(solver: Block2DMRGSolver, result: DMRGResult) -> None:
    """Print ``E_DMRG`` vs. exact FCI, when the active space is tractable.

    Diagnostic only -- never blocks or raises. An earlier version of this
    check (``src/reference_quality.py``, Phase 3 of
    ``diagnostics/REMEDIATION_PLAN.md``) also had a bond-dimension
    extrapolation fallback plus raise/warn/skip enforcement modes, but every
    real system here has a tractable FCI space, so the extrapolation tier
    was never exercised organically across the whole investigation (only by
    forcing ``max_fci_dimension=0`` in its own tests) and the enforcement
    modes never changed an outcome. Simplified down to what was actually
    used.
    """
    from math import comb

    n_alpha = (solver.n_elec + solver.spin) // 2
    n_beta = solver.n_elec - n_alpha
    dim = comb(int(solver.n_sites), n_alpha) * comb(int(solver.n_sites), n_beta)
    if dim > _REFERENCE_QUALITY_MAX_FCI_DIMENSION:
        print(f"[reference quality] CI dimension={dim} too large for an FCI check; skipping")
        return
    try:
        import optimize_symmetries

        if optimize_symmetries.pyscf is None:
            print("[reference quality] pyscf not installed; skipping FCI comparison")
            return
        dumpdata = {
            "H1": np.asarray(solver.h1e),
            "H2": np.asarray(solver.g2e),
            "NORB": int(solver.n_sites),
            "NELEC": (n_alpha, n_beta),
            "ECORE": float(solver.ecore),
        }
        e_fci, _ = optimize_symmetries.get_fci(dumpdata)
    except Exception as exc:
        print(f"[reference quality] FCI check failed ({exc!r}); skipping")
        return
    gap = abs(result.energy - float(e_fci))
    print(
        f"[reference quality] E_DMRG={result.energy:.10f}  E_FCI={float(e_fci):.10f}  "
        f"gap={gap:.6e} Ha  (CI dim={dim})"
    )


def build_dmrg_orbital_costs(
    molpath: str,
    parity_matrix: np.ndarray,
    store_dir: str | None = None,
    config: DMRGConfig | None = None,
    multiply: MultiplyConfig | None = None,
    reuse: bool = True,
    rotation: np.ndarray | None = None,
    n_threads: int = 4,
    pairs=None,
) -> tuple[DMRGOrbitalCosts, DMRGResult, Block2DMRGSolver]:
    """Solve (or reload) the reference MPS and wrap it as orbital costs.

    ``molpath`` may be an FCIDUMP (pyscf-free) or a ``.chk`` (needs pyscf).

    Before construction, prints the reference DMRG energy vs. exact FCI when
    the active space is small enough for FCI to be tractable -- see
    ``_print_reference_quality``. Diagnostic only; never blocks.
    """
    from pathlib import Path

    path = Path(molpath)
    config = config or DMRGConfig()
    if path.suffix == ".chk":
        from chemistry import fcidump_data

        solver = Block2DMRGSolver.from_dumpdata(
            fcidump_data(str(path)),
            store_dir=store_dir,
            n_threads=n_threads,
        )
    else:
        solver = Block2DMRGSolver.from_fcidump(
            path, store_dir=store_dir, n_threads=n_threads
        )

    if rotation is not None:
        from src.dmrg_solver import rotate_integrals

        h1e, g2e = rotate_integrals(solver.h1e, solver.g2e, rotation)
        solver = Block2DMRGSolver(
            h1e=h1e,
            g2e=g2e,
            ecore=solver.ecore,
            n_elec=solver.n_elec,
            spin=solver.spin,
            store_dir=store_dir or solver.store_dir,
            n_threads=n_threads,
        )

    result = solve_or_load_ground_state(solver, config=config, reuse=reuse)

    try:
        _print_reference_quality(solver, result)
    except Exception as exc:
        print(f"[reference quality] check errored ({exc!r}); skipping")

    costs = DMRGOrbitalCosts(
        solver,
        parity_matrix,
        mps_tag=result.mps_tag,
        multiply=multiply,
        pairs=pairs,
    )
    return costs, result, solver


def commutator_scores_by_row(
    costs,
    x,
    candidate_numbers=None,
    total_candidates=None,
    worker_label=None,
):
    """Return one MPS-native NC score for every parity row.

    This is the per-generator form of :meth:`DMRGOrbitalCosts.commutator`.
    The expensive reference action ``H|psi>`` is cached once and reused for
    all rows, which is required when ranking a seniority/quartet pool.
    """
    costs._eval_count += 1
    costs._ensure_eta()
    rotation = rotation_from_parameters(
        np.asarray(x, dtype=float), costs.solver.n_sites, costs.pairs
    )
    scores = []

    total_rows = len(costs.parity_matrix)
    if candidate_numbers is None:
        candidate_numbers = list(range(1, total_rows + 1))
    if total_candidates is None:
        total_candidates = total_rows

    for local_index, row in enumerate(costs.parity_matrix):
        candidate_number = int(candidate_numbers[local_index])
        prefix = f"[{worker_label}] " if worker_label else ""
        print(
            f"[NC score] {prefix}candidate "
            f"{candidate_number}/{int(total_candidates)}",
            flush=True,
        )
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SCORE_PHI")
        xi = costs._apply_symmetry(row, rotation, costs._eta, "SCORE_XI")
        chi = costs._apply(costs._h_mpo, phi, "SCORE_CHI")
        c2 = costs.solver.mps_norm2(chi)
        x2 = costs.solver.mps_norm2(xi)
        cx = costs.solver.mps_overlap(chi, xi)
        score = c2 + x2 - 2.0 * float(np.real(cx))
        scores.append(max(0.0, float(score)))

    return np.asarray(scores, dtype=float)


def candidate_index_chunks(n_candidates, n_workers):
    """Split candidate indices into balanced, nonempty worker chunks."""
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError("at least one candidate is required")
    n_workers = max(1, min(int(n_workers), n_candidates))
    indices = np.arange(n_candidates, dtype=int)
    chunks = np.array_split(indices, n_workers)
    return [chunk.tolist() for chunk in chunks if len(chunk)]


def _score_candidate_chunk(task):
    """Load one private MPS store and score one candidate chunk."""
    worker_number = int(task["worker_number"])
    worker_threads = int(task["worker_threads"])
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)

    solver = Block2DMRGSolver.load(
        task["store_dir"],
        n_threads=worker_threads,
    )
    costs = DMRGOrbitalCosts(
        solver,
        np.asarray(task["rows"], dtype=int),
        mps_tag=task["mps_tag"],
        multiply=MultiplyConfig(
            bond_dim=task["multiply_bond_dim"],
            n_sweeps=int(task["multiply_sweeps"]),
            tol=float(task["multiply_tol"]),
            bra_bond_dim_factor=float(task["bra_bond_dim_factor"]),
        ),
        pairs=task["pairs"],
    )
    indices = [int(index) for index in task["indices"]]
    scores = commutator_scores_by_row(
        costs,
        np.asarray(task["x"], dtype=float),
        candidate_numbers=[index + 1 for index in indices],
        total_candidates=int(task["total_candidates"]),
        worker_label=f"worker {worker_number}",
    )
    return indices, scores.tolist()


def _copy_mps_store_for_tag(source_dir, target_dir, mps_tag):
    """Copy only one saved MPS and the integrals needed to reload it."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    exact_names = {
        "integrals.npz",
        "metadata.json",
        f"{mps_tag}-mps_info.bin",
    }
    mps_prefix = f"F.MPS.{mps_tag}."
    info_prefix = f"F.MPS.INFO.{mps_tag}."
    copied = 0
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if (
            source.name in exact_names
            or source.name.startswith(mps_prefix)
            or source.name.startswith(info_prefix)
        ):
            shutil.copy2(source, target_dir / source.name)
            copied += 1
    if not (target_dir / f"{mps_tag}-mps_info.bin").exists():
        raise FileNotFoundError(
            f"MPS tag {mps_tag!r} was not found in {source_dir}"
        )
    return copied


def parallel_commutator_scores_from_store(
    store_dir,
    parity_matrix,
    x,
    pairs=None,
    multiply=None,
    n_threads=1,
    n_workers=1,
    scratch_root=None,
    mps_tag="GS",
):
    """Score parity rows in separate Block2 worker processes.

    Each process receives a private copy of the parent MPS store.  Block2 uses
    files inside that store for temporary MPS tensors, so sharing one directory
    among concurrent workers can corrupt intermediate states.
    """
    rows = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    chunks = candidate_index_chunks(len(rows), n_workers)
    if len(chunks) == 1:
        solver = Block2DMRGSolver.load(store_dir, n_threads=int(n_threads))
        costs = DMRGOrbitalCosts(
            solver,
            rows,
            mps_tag=mps_tag,
            multiply=multiply,
            pairs=pairs,
        )
        return commutator_scores_by_row(costs, x)

    multiply = multiply or MultiplyConfig()
    worker_threads = max(1, int(n_threads) // len(chunks))
    scratch_parent = Path(scratch_root or Path(store_dir).parent)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    worker_root = Path(
        tempfile.mkdtemp(prefix="nc_candidate_workers_", dir=scratch_parent)
    )

    print(
        f"[NC score] parallel workers={len(chunks)}, "
        f"threads_per_worker={worker_threads}",
        flush=True,
    )
    tasks = []
    try:
        for worker_number, indices in enumerate(chunks, start=1):
            worker_store = worker_root / f"worker_{worker_number}" / "mps"
            print(
                f"[NC score] preparing worker {worker_number}/{len(chunks)} "
                f"with {len(indices)} candidates",
                flush=True,
            )
            copied = _copy_mps_store_for_tag(store_dir, worker_store, mps_tag)
            print(
                f"[NC score] worker {worker_number} copied "
                f"{copied} parent-MPS files",
                flush=True,
            )
            tasks.append(
                {
                    "worker_number": worker_number,
                    "worker_threads": worker_threads,
                    "store_dir": str(worker_store),
                    "rows": rows[indices].tolist(),
                    "indices": indices,
                    "total_candidates": len(rows),
                    "x": np.asarray(x, dtype=float).tolist(),
                    "pairs": None if pairs is None else list(pairs),
                    "mps_tag": str(mps_tag),
                    "multiply_bond_dim": multiply.bond_dim,
                    "multiply_sweeps": multiply.n_sweeps,
                    "multiply_tol": multiply.tol,
                    "bra_bond_dim_factor": multiply.bra_bond_dim_factor,
                }
            )

        scores = np.empty(len(rows), dtype=float)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(tasks),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_score_candidate_chunk, task)
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                indices, chunk_scores = future.result()
                scores[np.asarray(indices, dtype=int)] = np.asarray(
                    chunk_scores, dtype=float
                )
                print(
                    f"[NC score] completed {len(indices)} candidates; "
                    f"first candidate={indices[0] + 1}",
                    flush=True,
                )
        return scores
    finally:
        shutil.rmtree(worker_root, ignore_errors=True)


def parity_expectations_by_row(costs, x, parity_matrix):
    """Return ``<psi|U^dagger S_k U|psi>`` for selected parity rows."""
    rotation = rotation_from_parameters(
        np.asarray(x, dtype=float), costs.solver.n_sites, costs.pairs
    )
    expectations = []
    rows = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    for index, row in enumerate(rows, start=1):
        print(f"[generator sign] selected {index}/{len(rows)}", flush=True)
        phi = costs._apply_symmetry(row, rotation, costs.ket, "SIGN_PHI")
        value = costs.solver.mps_overlap(costs.ket, phi)
        expectations.append(float(np.real(value)))
    return np.asarray(expectations, dtype=float)
