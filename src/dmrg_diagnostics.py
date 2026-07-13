"""MPS-native energy-sector and entanglement diagnostics.

Phase-3 replacement for the dense ``metrics.subspace_matrix`` path:

* ``E_decoupled`` from sector-targeted DMRG on the exactly decoupled
  Hamiltonian (see :meth:`Block2DMRGSolver.sector_ground_state`);
* coupled-energy ``K`` via PT-screened greedy selection over DMRG sector
  eigenstates, using ``<phi_i|H|phi_j>`` MPO expectations (no CI vector);
* orbital entropies and mutual information from the stored ground-state MPS.

Parity matrices given in the *original* orbital order are remapped through
``solver.remap_parity_matrix`` when the solver was built with Fiedler (or
other) orbital reordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.coupled_energy_core import (
    COUPLED_ENERGY_DEGENERACY_FLOOR,
    CoupledSpanState,
    augment_h_proj,
    ground_from_h_proj,
    improves_toward_fci,
    perturbation_may_improve,
    trial_ground_energy_incremental,
)
from src.dmrg_solver import Block2DMRGSolver, DMRGConfig

logger = logging.getLogger(__name__)

CHEMICAL_PRECISION = 0.0016


@dataclass
class SectorState:
    """One DMRG eigenstate of a parity sector."""

    energy: float
    sector_label: tuple[int, ...]
    block_index: int
    mps_tag: str


@dataclass
class DecoupledDiagnostic:
    e_reference: float
    e_decoupled: float
    best_sector: tuple[int, ...]
    sector_energies: dict[tuple[int, ...], float]
    dE: float
    k_equals_one: bool


@dataclass
class CoupledDiagnostic:
    e_coupled: float | None
    k: int
    converged: bool
    chosen: list[tuple[tuple[int, ...], int]]
    e_reference: float
    e_decoupled: float


@dataclass
class EntanglementDiagnostic:
    bipartite: np.ndarray
    orbital_s1: np.ndarray
    mutual_information: np.ndarray


@dataclass
class DMRGMetricsReport:
    e_reference: float
    decoupled: DecoupledDiagnostic
    coupled: CoupledDiagnostic | None = None
    entanglement: EntanglementDiagnostic | None = None
    symmetry_expectations: np.ndarray | None = None
    orbital_permutation: tuple[int, ...] = field(default_factory=tuple)


def prepare_parity_matrix(
    solver: Block2DMRGSolver, parity_matrix: np.ndarray
) -> np.ndarray:
    """Parity matrix in the solver's (possibly reordered) orbital basis."""
    return solver.remap_parity_matrix(np.atleast_2d(np.asarray(parity_matrix, dtype=int)))


def decoupled_energy_dmrg(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    e_reference: float,
    *,
    config: DMRGConfig | None = None,
    penalty: float = 30.0,
    max_sectors: int = 16,
    chemical_precision: float = CHEMICAL_PRECISION,
) -> DecoupledDiagnostic:
    """Scan dominant sectors and return the lowest single-sector energy."""
    parity = prepare_parity_matrix(solver, parity_matrix)
    config = config or DMRGConfig()
    labels = solver.dominant_sector_labels(parity, max_sectors=max_sectors)
    if not labels:
        raise RuntimeError("no sector weight found in the reference MPS")

    sector_energies: dict[tuple[int, ...], float] = {}
    best_energy, best_label = np.inf, labels[0][0]
    for label, weight in labels:
        result = solver.sector_ground_state(
            parity, label, penalty=penalty, config=config
        )
        sector_energies[label] = result.energy
        logger.info(
            "sector %s weight=%.4f E=%.10f", label, weight, result.energy
        )
        if result.energy < best_energy:
            best_energy, best_label = result.energy, label

    dE = best_energy - e_reference
    return DecoupledDiagnostic(
        e_reference=e_reference,
        e_decoupled=best_energy,
        best_sector=best_label,
        sector_energies=sector_energies,
        dE=dE,
        k_equals_one=dE < chemical_precision,
    )


def collect_sector_states(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    sector_labels: Sequence[tuple[int, ...]],
    *,
    nroots: int = 5,
    penalty: float = 30.0,
    config: DMRGConfig | None = None,
) -> list[SectorState]:
    """DMRG eigenstates for each requested sector, sorted by energy."""
    parity = prepare_parity_matrix(solver, parity_matrix)
    config = config or DMRGConfig()
    states: list[SectorState] = []
    for label in sector_labels:
        roots = solver.sector_excited_states(
            parity, label, nroots=nroots, penalty=penalty, config=config
        )
        for block_index, (energy, tag) in enumerate(roots):
            states.append(SectorState(
                energy=energy,
                sector_label=tuple(label),
                block_index=block_index,
                mps_tag=tag,
            ))
    states.sort(key=lambda s: s.energy)
    return states


def _hamiltonian_element(
    solver: Block2DMRGSolver, bra_tag: str, ket_tag: str
) -> complex:
    bra = solver.get_mps(bra_tag)
    ket = solver.get_mps(ket_tag) if ket_tag != bra_tag else bra
    return complex(solver.expectation(solver.hamiltonian_mpo(), ket=ket, bra=bra))


def greedy_coupled_energy_mps(
    solver: Block2DMRGSolver,
    states: Sequence[SectorState],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    max_total_vectors: int | None = None,
    coupling_tol: float = 1e-12,
    energy_change_tol: float = 1e-12,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> tuple[float | None, int, bool, list[tuple[tuple[int, ...], int]]]:
    """PT-screened greedy K using MPS Hamiltonian matrix elements."""
    if not states:
        return None, 0, False, []
    if max_total_vectors is None:
        max_total_vectors = len(states)

    # Cache <i|H|j> for chosen/candidate pairs as we go.
    h_cache: dict[tuple[str, str], complex] = {}

    def h_ij(tag_a: str, tag_b: str) -> complex:
        key = (tag_a, tag_b)
        key_rev = (tag_b, tag_a)
        if key in h_cache:
            return h_cache[key]
        if key_rev in h_cache:
            return np.conjugate(h_cache[key_rev])
        value = _hamiltonian_element(solver, tag_a, tag_b)
        h_cache[key] = value
        return value

    chosen_keys: list[tuple[tuple[int, ...], int]] = []
    chosen_indices: set[int] = set()
    chosen_tags: list[str] = []
    state: CoupledSpanState | None = None
    converged = False

    while True:
        added_this_pass = False
        for index, cand in enumerate(states):
            if index in chosen_indices:
                continue
            if len(chosen_keys) >= max_total_vectors:
                break

            if state is None:
                e_new = float(cand.energy)
                h_cache[(cand.mps_tag, cand.mps_tag)] = complex(e_new)
                h_proj = np.array([[e_new]], dtype=np.complex128)
                state = CoupledSpanState(
                    chosen_vecs=[],  # unused in MPS path
                    h_vecs=[],
                    h_proj=h_proj,
                    psi0=np.array([1.0], dtype=np.complex128),
                    e_proj=e_new,
                )
                chosen_tags = [cand.mps_tag]
            else:
                h_cols = [
                    h_ij(tag, cand.mps_tag) for tag in chosen_tags
                ]
                max_coupling = max((abs(z) for z in h_cols), default=0.0)
                if max_coupling <= coupling_tol:
                    continue

                if not perturbation_may_improve(
                    state.psi0,
                    h_cols,
                    state.e_proj,
                    float(cand.energy),
                    state.e_proj,
                    e_exact,
                    coupling_tol=coupling_tol,
                    energy_change_tol=energy_change_tol,
                    degeneracy_floor=degeneracy_floor,
                ):
                    continue

                e_new = trial_ground_energy_incremental(
                    state.h_proj, h_cols, float(cand.energy)
                )
                if not improves_toward_fci(
                    e_new, state.e_proj, e_exact, energy_change_tol
                ):
                    continue

                h_trial = augment_h_proj(state.h_proj, h_cols, float(cand.energy))
                e_accept, psi0 = ground_from_h_proj(h_trial)
                state = CoupledSpanState(
                    chosen_vecs=[],
                    h_vecs=[],
                    h_proj=h_trial,
                    psi0=psi0,
                    e_proj=e_accept,
                )
                chosen_tags.append(cand.mps_tag)

            chosen_indices.add(index)
            chosen_keys.append((cand.sector_label, cand.block_index))
            added_this_pass = True

            if e_exact is not None and abs(state.e_proj - e_exact) <= tol:
                converged = True
                break

        if converged:
            break
        if not added_this_pass or len(chosen_keys) >= max_total_vectors:
            break

    if state is None:
        return None, 0, False, []
    if e_exact is not None and abs(state.e_proj - e_exact) <= tol:
        converged = True
    return state.e_proj, len(chosen_keys), converged, chosen_keys


def coupled_energy_dmrg(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    e_reference: float,
    e_decoupled: float,
    *,
    sector_labels: Sequence[tuple[int, ...]] | None = None,
    nroots: int = 5,
    penalty: float = 30.0,
    config: DMRGConfig | None = None,
    max_sectors: int = 8,
    chemical_precision: float = CHEMICAL_PRECISION,
) -> CoupledDiagnostic:
    """Build sector spectra with DMRG and run the PT-screened K diagnostic.

    ``parity_matrix`` must be in the *original* orbital order; remapping for a
    reordered solver is handled inside :func:`collect_sector_states`.
    """
    if sector_labels is None:
        remapped = prepare_parity_matrix(solver, parity_matrix)
        sector_labels = [
            label for label, _ in solver.dominant_sector_labels(
                remapped, max_sectors=max_sectors
            )
        ]
    states = collect_sector_states(
        solver,
        parity_matrix,
        sector_labels,
        nroots=nroots,
        penalty=penalty,
        config=config,
    )
    e_coupled, k, converged, chosen = greedy_coupled_energy_mps(
        solver,
        states,
        e_exact=e_reference,
        tol=chemical_precision,
    )
    return CoupledDiagnostic(
        e_coupled=e_coupled,
        k=k,
        converged=converged,
        chosen=chosen,
        e_reference=e_reference,
        e_decoupled=e_decoupled,
    )


def entanglement_diagnostic(
    solver: Block2DMRGSolver, ket=None
) -> EntanglementDiagnostic:
    """Bipartite cut entropies, 1-orbital entropies and mutual information."""
    if ket is None:
        ket = solver.get_mps()
    return EntanglementDiagnostic(
        bipartite=solver.bipartite_entanglement(ket=ket),
        orbital_s1=solver.orbital_entropies(ket=ket, orb_type=1),
        mutual_information=solver.mutual_information(ket=ket),
    )


def run_dmrg_metrics(
    solver: Block2DMRGSolver,
    parity_matrix: np.ndarray,
    *,
    config: DMRGConfig | None = None,
    penalty: float = 30.0,
    max_sectors: int = 16,
    states_per_sector: int = 5,
    compute_k: bool = True,
    compute_entanglement: bool = True,
    chemical_precision: float = CHEMICAL_PRECISION,
    reuse_ground_state: bool = True,
) -> DMRGMetricsReport:
    """Full MPS-native diagnostics: E_ref, E_dec, optional K and entropies."""
    from src.dmrg_solver import solve_or_load_ground_state

    config = config or DMRGConfig()
    gs = solve_or_load_ground_state(
        solver, config=config, reuse=reuse_ground_state
    )
    parity = prepare_parity_matrix(solver, parity_matrix)
    expectations = solver.symmetry_expectations(parity)

    decoupled = decoupled_energy_dmrg(
        solver,
        parity_matrix,  # original order; remapped inside
        gs.energy,
        config=config,
        penalty=penalty,
        max_sectors=max_sectors,
        chemical_precision=chemical_precision,
    )

    coupled = None
    if compute_k and not decoupled.k_equals_one:
        coupled = coupled_energy_dmrg(
            solver,
            parity_matrix,
            gs.energy,
            decoupled.e_decoupled,
            sector_labels=list(decoupled.sector_energies.keys()),
            nroots=states_per_sector,
            penalty=penalty,
            config=config,
            chemical_precision=chemical_precision,
        )
    elif decoupled.k_equals_one:
        coupled = CoupledDiagnostic(
            e_coupled=decoupled.e_decoupled,
            k=1,
            converged=True,
            chosen=[(decoupled.best_sector, 0)],
            e_reference=gs.energy,
            e_decoupled=decoupled.e_decoupled,
        )

    ent = entanglement_diagnostic(solver) if compute_entanglement else None
    return DMRGMetricsReport(
        e_reference=gs.energy,
        decoupled=decoupled,
        coupled=coupled,
        entanglement=ent,
        symmetry_expectations=expectations,
        orbital_permutation=tuple(solver.orbital_permutation),
    )


def format_metrics_report(report: DMRGMetricsReport) -> list[str]:
    """Plain-text lines matching the style of ``metrics.py`` result files."""
    lines = [
        f"solver dmrg",
        f"orbital_permutation {list(report.orbital_permutation)}",
        f"E_FCI {report.e_reference:4.6f}",
        f"E_decoupled {report.decoupled.e_decoupled:4.6f}",
        f"dE {report.decoupled.dE:4.6f}",
        f"best_sector {report.decoupled.best_sector}",
    ]
    if report.symmetry_expectations is not None:
        lines.append(
            f"symmetry expectations {np.round(report.symmetry_expectations, 6)}"
        )
    if report.coupled is not None:
        lines.append("coupled_energy_method perturbation")
        if report.coupled.e_coupled is not None:
            lines.append(f"E_coupled {report.coupled.e_coupled:4.6f}")
        lines.append(f"K {report.coupled.k}")
        lines.append(f"converged {report.coupled.converged}")
        for key in report.coupled.chosen:
            lines.append(str(key))
    if report.entanglement is not None:
        lines.append(
            f"bipartite entanglement {np.round(report.entanglement.bipartite, 6)}"
        )
        lines.append(
            f"orbital entropies {np.round(report.entanglement.orbital_s1, 6)}"
        )
        mi = report.entanglement.mutual_information
        lines.append(f"mutual_information_max {float(np.max(mi)):.6f}")
    return lines
