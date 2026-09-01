from __future__ import annotations

"""
Cluster Number Selected-Sector Lanczos

Two-part pipeline built on top of cluster_number_sector_search.py:

  Part 1 (reused verbatim, not reimplemented): run the same beam search +
  cluster-number sector ranking as cluster_number_sector_search.py, for one
  or more trajectory decompositions. Output: a list of (Decomposition,
  ranked_sectors) pairs, where ranked_sectors is Part 1's own ordered list
  of SectorRelevance entries.

  Part 2 (new): for each such pair, run a matrix-free selected-sector
  Lanczos/Davidson energy solve restricted to the union of the retained
  sectors' determinant supports, using src/selected_sector_lanczos.py's own
  Krylov/coupling machinery wherever it applies unchanged. That module was
  built around binary "cluster-number-parity" sectors (N_K mod 2, via GF(2)
  syndromes); this file adapts only the address-finding front end for full
  integer sectors (see integer_sector_supports) -- everything downstream of
  a (support, label, vector) triple in that module is already label-format-
  agnostic and is reused as-is.

Part 2's coupling architecture: exactly one sector -- the t=0 "anchor"
sector from Part 1's own ranking -- is solved exactly (its true ground
state within its own restricted Hilbert space). Every other retained sector
contributes a small Krylov subspace seeded not by its own ground state, but
by however it actually couples to the current best coupled state (the
"leakage" of H|state> into that sector) -- a more targeted use of a limited
per-sector subspace budget than diagonalizing each sector in isolation
would be. This repeats, growing the coupled subspace one H-action-cheap
round at a time, until the energy stops moving, no sector shows further
leakage, or a safety cap on iterations is hit.

Because H has body-rank <= 2, a single H-application can only connect
sectors within transportation distance t<=2 of wherever the state currently
has support (the same structural fact cluster_number_sector_search.py's own
q(delta) scoring relies on). This is why leakage detection is re-run every
iteration against Part 1's *entire* retained candidate list, not just the
sectors already touched so far: a sector at t=3 or t=4 (only possible when
--max-elec-transfer is raised above the default 2) is invisible to a single
leakage sweep from the anchor alone, but becomes visible once the coupled
state has already picked up some t<=2 character -- exactly what the
iterative extension is for.

CLI usage (mirrors cluster_number_sector_search.py's own CLI -- reuses its
full flag set for the DMRG + beam-search + sector-ranking stage, adding a
new argument group for the Lanczos solve itself):
    python cluster_number_selected_sector_lanczos.py h2o sto-3g 2.0 commutator --bond-angle 104.5
    python cluster_number_selected_sector_lanczos.py lih sto-3g 1.6 variance --analyze-num-clusters-lanczos 2

Library usage (bring your own Decomposition + RDMData + ranked_sectors,
e.g. from cluster_number_sector_search.rank_relevant_sectors):
    from cluster_number_selected_sector_lanczos import (
        solve_decomposition_selected_sectors, LanczosSearchConfig,
    )
    result = solve_decomposition_selected_sectors(
        deco, rdm_data, nelec, ecore, ranked_sectors, LanczosSearchConfig(),
    )

See tests/test_cluster_number_selected_sector_lanczos.py for from-scratch
verification of the new address-finding and orchestration logic against
exact FCI ground truth.
"""

# AI implemented

import argparse
import json
import logging
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pyscf.fci.cistring
import pyscf.fci.direct_spin1
import scipy.sparse.linalg

from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    partition_to_cluster_matrix,
    rotate_rdm_data,
)
from cluster_number_sector_search import (
    SectorRelevance,
    SectorSearchConfig,
    _geometry_output_subpath,
    _output_dir_for,
    _sector_relevance_to_json,
    normalize_cluster_family,
    rank_relevant_sectors,
)
from cluster_number_sector_search import create_parser as _sector_search_create_parser
from src.selected_sector_lanczos import (
    coupled_ground_residual,
    coupling_seeded_krylov_basis,
    extend_coupled_matrix,
    krylov_depth_curve,
    residual_seeded_krylov_extension,
    solve_selected_sector,
)

logger = logging.getLogger(__name__)


# =============================================================================
# New primitive: matrix-free full-space Hamiltonian (human verified pending)
# =============================================================================


def build_full_hamiltonian_operator(
    h1e: np.ndarray, eri: np.ndarray, norb: int, nelec: tuple[int, int]
) -> tuple[scipy.sparse.linalg.LinearOperator, int]:
    """Wrap pyscf's FCI sigma-vector routine (fci.direct_spin1.absorb_h1e +
    contract_2e -- this codebase's own established idiom for a matrix-free
    full-space Hamiltonian action, already used the same way in
    sector_metrics_k_mp.py for an unrelated pipeline) as a flat-vector
    LinearOperator over the full fixed-spin Hilbert space, matching
    src/selected_sector_lanczos.py's own addressing convention: flat address
    = alpha_address * n_beta_strings + beta_address, matching a C-order
    reshape of pyscf's native (n_alpha_strings, n_beta_strings) CI array.

    Real (float64) dtype: h1e/eri are real in every actual pipeline in this
    codebase (orbital-rotation parametrization only ever emits real
    orthogonal matrices), so this halves the memory/FLOPs of every one of
    the (many) full-space matrix-vector products this pipeline performs,
    relative to a naive complex128 default -- restricted_linear_operator in
    src/selected_sector_lanczos.py already propagates this dtype via
    np.result_type, so nothing there needs to change for the saving to take
    effect. rmatvec=matvec is correct (not a shortcut) because H is
    real-symmetric here, the same justification restricted_linear_operator
    itself already relies on for its own rmatvec=matvec.
    """
    n_alpha = comb(norb, int(nelec[0]))
    n_beta = comb(norb, int(nelec[1]))
    full_dimension = n_alpha * n_beta
    h2e = pyscf.fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)

    def matvec(flat_vec: np.ndarray) -> np.ndarray:
        vec2d = np.asarray(flat_vec).reshape(n_alpha, n_beta)
        sigma2d = pyscf.fci.direct_spin1.contract_2e(h2e, vec2d, norb, nelec)
        return np.asarray(sigma2d).reshape(-1)

    operator = scipy.sparse.linalg.LinearOperator(
        shape=(full_dimension, full_dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    return operator, full_dimension


# =============================================================================
# New: integer cluster-number-sector determinant addressing (human verified pending)
# =============================================================================


def cluster_occupation_counts(strings: np.ndarray, cluster_masks: list[int]) -> np.ndarray:
    """(n_strings, K) int array of per-string, per-cluster occupation counts
    -- the integer analog of bitstring_syndrome in
    src/selected_sector_lanczos.py (a full popcount per cluster, rather than
    one GF(2) parity bit). cluster_masks[k] is the int bitmask of orbitals
    in cluster k (bit p set iff orbital p is in C_k), built the same way
    sector_metrics_k_mp.py's own _popcount_parity already does -- precedent
    already established in this codebase, not invention."""
    strings = np.asarray(strings, dtype=np.int64)
    counts = np.zeros((len(strings), len(cluster_masks)), dtype=np.int64)
    for k, mask in enumerate(cluster_masks):
        counts[:, k] = [bin(int(s) & int(mask)).count("1") for s in strings]
    return counts


def label_key(label: tuple[int, ...]) -> str:
    """Multi-digit-safe compact label formatter (e.g. "2-0-3"). Replaces
    src/selected_sector_lanczos.py's label_text, which joins bits directly
    with no separator -- fine for single-digit binary parity bits, but
    ambiguous for integer N_K > 9. Used only for progress/log messages;
    JSON output always uses labels as plain lists, never as string keys."""
    return "-".join(str(int(x)) for x in label)


def integer_sector_supports(
    indicator: np.ndarray,
    labels: list[tuple[int, ...]],
    norb: int,
    nelec: tuple[int, int],
    print_progress: bool = True,
) -> dict[tuple[int, ...], dict]:
    """Integer-sector analog of
    src/selected_sector_lanczos.py's selected_sector_supports: for the
    requested full cluster-number labels (N_0, ..., N_{K-1}), returns the
    complete set of fixed-(Nalpha,Nbeta) determinant addresses in each
    sector, without enumerating the full Hilbert space.

    Mirrors the original's real structure: only BETA strings are grouped
    (once, by per-cluster occupation-count tuple) -- alpha strings are
    walked exactly once each in the outer loop regardless, so grouping them
    separately would add cost for no benefit, same as the original. For
    each alpha string and each requested label, the complementary beta
    count is needed_beta = label - alpha_count (vector SUBTRACTION
    replacing the original's XOR, since N_K = N_K^alpha + N_K^beta is a
    true sum -- parity only needs XOR because (a+b) mod 2 = (a mod 2) XOR
    (b mod 2)). Any count with a negative or otherwise-unrealized entry
    simply matches nothing (dict.get(..., ())) -- no separate capacity
    check is needed. Labels are assumed unique (guaranteed by Part 1's own
    dict-based candidate generation).

    Returns {label: {"label": label, "full_addresses": int64 array,
    "dimension": int}} -- exactly selected_sector_supports's own return
    shape, so every reused function downstream (which only ever consumes
    "full_addresses") works unchanged.
    """
    n_alpha, n_beta = int(nelec[0]), int(nelec[1])
    K = indicator.shape[0]
    labels = [tuple(int(x) for x in label) for label in labels]
    if any(len(label) != K for label in labels):
        raise ValueError("integer_sector_supports: every label must have one entry per cluster")

    cluster_masks = [int(np.sum(indicator[k] * (1 << np.arange(norb)))) for k in range(K)]

    alpha_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_alpha), dtype=np.int64)
    beta_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_beta), dtype=np.int64)
    n_beta_strings = len(beta_strings)

    alpha_counts = cluster_occupation_counts(alpha_strings, cluster_masks)
    beta_counts = cluster_occupation_counts(beta_strings, cluster_masks)

    beta_groups: dict[tuple[int, ...], list[int]] = {}
    for beta_address, count_row in enumerate(beta_counts):
        beta_groups.setdefault(tuple(int(x) for x in count_row), []).append(beta_address)

    entries: dict[tuple[int, ...], list[int]] = {label: [] for label in labels}
    for alpha_address, count_row in enumerate(alpha_counts):
        alpha_count = tuple(int(x) for x in count_row)
        for label in labels:
            needed_beta = tuple(label[k] - alpha_count[k] for k in range(K))
            for beta_address in beta_groups.get(needed_beta, ()):
                entries[label].append(alpha_address * n_beta_strings + beta_address)

    supports: dict[tuple[int, ...], dict] = {}
    for label in labels:
        addresses = np.asarray(sorted(entries[label]), dtype=np.int64)
        supports[label] = {
            "label": label,
            "full_addresses": addresses,
            "dimension": int(len(addresses)),
        }
        if print_progress:
            print(
                f"[support] sector {label_key(label)}: {len(addresses):,} physical determinants",
                flush=True,
            )
    return supports


# =============================================================================
# New: candidate-restricted leakage (human verified pending)
# =============================================================================


def candidate_leakage_weights(
    residual: np.ndarray,
    supports: dict[tuple[int, ...], dict],
    candidate_labels: list[tuple[int, ...]],
) -> tuple[list[tuple[tuple[int, ...], float]], float]:
    """Given an already-computed full-space residual vector, sums
    |residual[supports[label]["full_addresses"]]|^2 for each of Part 1's
    already-short candidate labels -- NOT a generic sector-classification
    sweep over every nonzero residual entry the way
    src/selected_sector_lanczos.py's sector_leakage_weights/
    sector_vector_weights do (we already know exactly which labels we care
    about, so there is no need to classify the rest of the residual at
    all). Returns (ranked (label, weight) pairs sorted descending, total
    ||residual||^2). The gap between sum(weights) and the total is a
    coverage diagnostic: how much of the true leakage lands outside Part
    1's retained candidates -- should be ~0 whenever max_elec_transfer>=2
    (see module docstring's structural-fact discussion); a material gap
    flags a bug or a too-small max_elec_transfer.

    Deliberately takes `residual` as a plain array rather than computing it
    internally from an anchor/energy pair: the orchestration below calls
    this once per iteration with residuals from different sources (the
    anchor's own leakage on the trivial first call, then
    coupled_ground_residual's residual on every later call).
    """
    residual = np.asarray(residual)
    weights: list[tuple[tuple[int, ...], float]] = []
    for label in candidate_labels:
        addresses = supports[label]["full_addresses"]
        weight = float(np.sum(np.abs(residual[addresses]) ** 2))
        weights.append((label, weight))
    weights.sort(key=lambda item: -item[1])
    total = float(np.vdot(residual, residual).real)
    return weights, total


# =============================================================================
# New: Part-2 orchestration (human verified pending)
# =============================================================================


@dataclass
class LanczosSearchConfig:
    """Hyperparameters for solve_decomposition_selected_sectors."""

    krylov_seed_depth: int = 4  # vectors requested the first time a sector is touched
    max_iterations: int = 5  # rounds of leakage-detection + extension
    vectors_per_iteration: int = 2  # new vectors per already-active sector per later round
    energy_tolerance: float = 1e-6  # Hartree; stop once |delta E| between rounds is below this
    leakage_threshold: float = 1e-10  # ignore sectors with negligible weight this round


def solve_decomposition_selected_sectors(
    deco: Decomposition,
    rdm_data: RDMData,
    nelec: tuple[int, int],
    ecore: float,
    ranked_sectors: list[SectorRelevance],
    config: LanczosSearchConfig | None = None,
) -> dict:
    """Selected-sector Lanczos/Davidson energy solve for one decomposition,
    restricted to the union of `ranked_sectors`'s determinant supports (the
    output of cluster_number_sector_search.rank_relevant_sectors for the
    same `deco`). See module docstring for the algorithm and its rationale.

    `rdm_data` is in the same reference basis deco.U is relative to (as for
    rank_relevant_sectors); it is rotated into deco's own basis internally
    -- a second time, independently of whatever rank_relevant_sectors did
    internally for the same deco, since that function doesn't expose its
    own rotated copy. `ecore` (the nuclear-repulsion/core-energy constant,
    orbital-rotation-invariant) is only used to report a total energy
    alongside the electronic-only energy the FCI machinery itself computes.

    `ranked_sectors` must contain exactly one entry with elec_transfer==0
    (the anchor label) -- guaranteed by rank_relevant_sectors's own BFS
    candidate generation, which always starts from and includes the main
    label at t=0.
    """
    if config is None:
        config = LanczosSearchConfig()

    started = time.perf_counter()
    norb = rdm_data.norb
    rdm_data_cur = rotate_rdm_data(rdm_data, deco.U)
    full_operator, full_dimension = build_full_hamiltonian_operator(
        rdm_data_cur.h1e, rdm_data_cur.g2e_full, norb, nelec
    )

    clean_partition, _cluster_map, _had_ghost = normalize_cluster_family(deco.partition, norb)
    indicator = partition_to_cluster_matrix(clean_partition, norb)

    labels = [r.label for r in ranked_sectors]
    supports = integer_sector_supports(indicator, labels, norb, nelec, print_progress=False)

    anchor_entries = [r for r in ranked_sectors if r.elec_transfer == 0]
    if not anchor_entries:
        raise ValueError(
            "solve_decomposition_selected_sectors: ranked_sectors has no elec_transfer==0 "
            "(anchor) entry -- was it produced by rank_relevant_sectors?"
        )
    anchor_label = anchor_entries[0].label
    anchor_support = supports[anchor_label]
    if anchor_support["dimension"] == 0:
        raise ValueError(
            f"solve_decomposition_selected_sectors: anchor sector {anchor_label} is empty"
        )

    matvec_count = 0
    matvec_seconds = 0.0

    anchor_result = solve_selected_sector(
        full_operator, full_dimension, anchor_support["full_addresses"], n_roots=1,
    )
    matvec_count += int(anchor_result["matvec_count"])
    matvec_seconds += float(anchor_result["matvec_seconds"])
    anchor_vector = anchor_result["vectors"][:, 0]
    anchor_energy = float(np.real(anchor_result["energies"][0]))

    candidates: list[dict] = [
        {
            "label": anchor_label,
            "support": anchor_support["full_addresses"],
            "vector": anchor_vector,
            "kind": "anchor",
            "depth": 0,
            "sector_column": 0,
        }
    ]
    matrix = np.asarray([[anchor_energy]], dtype=np.complex128)
    sector_bases: dict[tuple[int, ...], np.ndarray] = {}
    non_anchor_labels = [r.label for r in ranked_sectors if r.label != anchor_label]
    energy_history: list[float] = []
    stop_reason = "max_iterations_reached"
    leakage: list[tuple[tuple[int, ...], float]] = []
    total_leak = 0.0

    for iteration in range(config.max_iterations):
        energy, _coeffs, _state, residual = coupled_ground_residual(
            full_operator, full_dimension, matrix, candidates
        )
        matvec_count += 1  # coupled_ground_residual does exactly one H-action
        energy_history.append(float(np.real(energy)))
        if len(energy_history) > 1 and abs(energy_history[-1] - energy_history[-2]) < config.energy_tolerance:
            stop_reason = "energy_converged"
            break

        leakage, total_leak = candidate_leakage_weights(residual, supports, non_anchor_labels)
        new_candidates: list[dict] = []
        for label, weight in leakage:
            if weight <= config.leakage_threshold:
                continue
            seed = residual[supports[label]["full_addresses"]]
            if label not in sector_bases:
                result = coupling_seeded_krylov_basis(
                    full_operator, full_dimension, supports[label]["full_addresses"],
                    seed, config.krylov_seed_depth, print_every=0,
                )
                sector_bases[label] = result["basis"]
                start_col, kind = 0, "krylov"
                matvec_count += int(result["matvec_count"])
                matvec_seconds += float(result["matvec_seconds"])
            else:
                extension = residual_seeded_krylov_extension(
                    full_operator, full_dimension, supports[label]["full_addresses"],
                    seed, sector_bases[label], config.vectors_per_iteration,
                )
                start_col = sector_bases[label].shape[1]
                sector_bases[label] = np.column_stack([sector_bases[label], extension])
                kind = "residual_krylov"
                # residual_seeded_krylov_extension doesn't expose matvec stats (it
                # returns a bare array, unlike coupling_seeded_krylov_basis/
                # solve_selected_sector) -- approximate its H-action count as the
                # number of vectors it actually returned (undercounts by at most 1,
                # in the case where the very last H-action hit an invariant
                # subspace and was discarded rather than yielding a new vector).
                matvec_count += int(extension.shape[1])

            for offset in range(sector_bases[label].shape[1] - start_col):
                column = start_col + offset
                new_candidates.append(
                    {
                        "label": label,
                        "support": supports[label]["full_addresses"],
                        "vector": sector_bases[label][:, column],
                        "kind": kind,
                        "depth": column + 1,
                        "sector_column": column,
                    }
                )
        if not new_candidates:
            stop_reason = "leakage_exhausted"
            break
        matrix, candidates, extend_seconds = extend_coupled_matrix(
            full_operator, full_dimension, matrix, candidates, new_candidates, print_every=0,
        )
        matvec_count += len(new_candidates)  # one H-action per new candidate, per its own docstring
        matvec_seconds += float(extend_seconds)
    else:
        # Loop ran to config.max_iterations without an explicit break -- the final
        # round's extend_coupled_matrix grew (matrix, candidates), but no energy was
        # computed for that just-extended state yet. One more solve (cheap: no new
        # H-actions, matrix/candidates already built) keeps energy_history[-1]
        # consistent with the returned matrix/candidates. Also correctly handles
        # max_iterations=0: the for body never runs, this else still does, so
        # energy_history ends up = [anchor_energy].
        energy, _coeffs, _state, _residual = coupled_ground_residual(
            full_operator, full_dimension, matrix, candidates
        )
        matvec_count += 1
        energy_history.append(float(np.real(energy)))

    converged = stop_reason == "energy_converged"

    depths = sorted({c["depth"] for c in candidates})
    curve = krylov_depth_curve(
        matrix, candidates, depths, reference_energy=energy_history[-1], tolerance=config.energy_tolerance,
    )

    final_energy = energy_history[-1]
    captured_fraction = float(sum(w for _l, w in leakage) / total_leak) if total_leak > 0 else 1.0

    return {
        "anchor": {
            "label": list(anchor_label),
            "energy": anchor_energy,
            "dimension": anchor_support["dimension"],
        },
        "leakage": {
            # From whichever round most recently computed leakage/total_leak -- can be
            # one round "behind" final.energy when stop_reason=="energy_converged",
            # since the convergence check (and its break) happens BEFORE that round's
            # own leakage computation. Still a meaningful snapshot (the leakage
            # picture that was already small enough to be about to converge), just
            # not literally "the final iteration's" leakage.
            "total_norm_sq": total_leak,
            "captured_fraction": captured_fraction,
            "per_label": [{"label": list(label), "weight": weight} for label, weight in leakage],
        },
        "iterations": [
            {
                "iteration": i,
                "energy": e,
                "delta_e": (e - energy_history[i - 1]) if i > 0 else None,
            }
            for i, e in enumerate(energy_history)
        ],
        "final": {
            "energy": final_energy,
            "total_energy_with_ecore": final_energy + float(ecore),
            "dimension": int(matrix.shape[0]),
            "converged": converged,
            "stop_reason": stop_reason,
        },
        "krylov_depth_curve": curve,
        "timing": {
            "matvec_count": matvec_count,
            "matvec_seconds": matvec_seconds,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }


# =============================================================================
# CLI Interface (human verified pending)
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Argument parser reusing cluster_number_sector_search.create_parser()'s
    full flag set (this module runs that same beam search + sector ranking
    as its own Part 1), plus a new argument group for the selected-sector
    Lanczos solve itself (Part 2)."""
    parser = _sector_search_create_parser()
    parser.description = (
        "Beam-search decomposition + cluster-number sector ranking (as "
        "cluster_number_sector_search.py), followed by a selected-sector "
        "Lanczos/Davidson energy solve restricted to the union of the "
        "ranked sectors' determinant supports."
    )

    group = parser.add_argument_group(
        "selected-sector Lanczos (Part 2)",
        description="Note: --force-h1e and --force-full-rdms (above) are both no-ops in this "
        "script -- full RDMs/integrals are always extracted regardless of either flag, since "
        "the Lanczos solve below needs h1e/g2e_full no matter what. Use --max-sector-tier to "
        "control Part 1's own ranking tier independently of that.",
    )
    group.add_argument(
        "--analyze-num-clusters-lanczos", type=int, nargs="+", default=None,
        help="Which trajectory cluster counts to run the Lanczos solve on (default: "
        "only the single entry with the largest num_clusters -- this stage is much "
        "more expensive than the sector ranking above, so it deliberately does not "
        "default to 'all entries' the way --analyze-num-clusters does). Must be a "
        "subset of whatever --analyze-num-clusters selected.",
    )
    group.add_argument(
        "--max-sector-tier", type=int, choices=[0, 1, 2], default=None,
        help="Cap Part 1's sector-ranking energy-score tier below the full Tier-2 data always "
        "extracted above for Part 2 -- Part 2's Lanczos solve is unaffected by this flag, it "
        "always gets the full, uncapped data. Default: None, i.e. the ranking also uses the "
        "full Tier-2 data. 0 disables energy scoring for the ranking entirely (weight_score-only, "
        "i.e. Tier 0); 1 caps it at Tier 1 (h1e only). Useful since Tier 0 sometimes outranks "
        "Tier 2 despite being cheaper (see cluster_number_sector_search.py's module docstring).",
    )
    group.add_argument("--krylov-seed-depth", type=int, default=4)
    group.add_argument("--max-iterations", type=int, default=5)
    group.add_argument("--vectors-per-iteration", type=int, default=2)
    group.add_argument("--energy-tolerance", type=float, default=1e-6)
    group.add_argument("--leakage-threshold", type=float, default=1e-10)
    group.add_argument(
        "--lanczos-plots-dir", type=str, default=None,
        help="Directory for convergence plots (default: <part2_output_dir>/plots/). Named "
        "distinctly from the inherited --plots-dir (Part 1's own --K-sector-analysis plots) "
        "to avoid a flag collision, since this parser is built on top of that one.",
    )
    group.add_argument("--no-plots", action="store_true")
    return parser


def _part2_output_dir_for(args: argparse.Namespace) -> Path:
    return Path("outputs_") / "cluster_number_selected_sector_lanczos" / _geometry_output_subpath(args)


def _plot_convergence(result: dict, dmrg_energy: float, output_path: Path) -> None:
    """Primary convergence curve: energy_history (iteration-indexed), the
    exact record of what this pipeline's own optimization actually did.
    krylov_depth_curve isn't used here -- its depth values are per-sector-
    local column indices, not synchronized iteration numbers (see module/
    plan discussion), so it's a supplementary diagnostic, not this plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [it["iteration"] for it in result["iterations"]]
    energies = [it["energy"] for it in result["iterations"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iterations, energies, marker="o", color="tab:blue", label="coupled energy")
    ax.axhline(dmrg_energy, color="gray", linestyle="--", label="DMRG energy")
    ax.axhline(result["anchor"]["energy"], color="tab:orange", linestyle=":", label="anchor-only energy")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy (Ha)")
    ax.set_title(f"Selected-sector Lanczos convergence ({result['final']['stop_reason']})")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    from cluster_number_decomposition_optimization import (
        DecompositionOptimizerConfig,
        _build_cost_function_constructor,
        _build_initial_bases,
        _run_dmrg_and_build_rdm_data,
        run_decomposition_optimizer,
    )
    from cluster_numbers_metrics import get_git_hash, get_timestamp

    _known_molecules = {"h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"}
    if args.fcidump is None and args.molecule.lower() not in _known_molecules:
        logger.error(f"Unsupported molecule: {args.molecule}. Supported: {sorted(_known_molecules)}")
        exit(1)

    logger.info(f"Running beam search for {args.molecule}/{args.basis_set} (always with full RDMs)...")
    # force_full_rdms is hardcoded True here (not args.force_full_rdms) -- see create_parser's
    # "--force-full-rdms is a no-op in this script" note: the Lanczos solve below always needs
    # h1e/g2e_full/rdm3/rdm4, regardless of that flag.
    rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(
        args, force_full_rdms=True,
    )
    norb = rdm_data.norb

    # rotate_rdm_data (called on every beam-search split/round, and once per initial
    # basis in _build_initial_bases's --fiedler-reorder path) rotates every populated
    # field unconditionally, including rdm3/rdm4 (O(norb^6)/O(norb^8) tensor
    # contractions) -- expensive, and completely wasted whenever the cost function
    # doesn't read them (every cost function except "commutator"). rdm_data.rdm3/.rdm4/
    # .h1e/.g2e_full are populated here purely for Part 1 (rank_relevant_sectors) and
    # Part 2 (solve_decomposition_selected_sectors) below, both of which run once, after
    # the beam search entirely -- so the beam search itself must only see them when it
    # actually needs them (see the same fix in cluster_number_sector_search.py).
    if args.cost_function == "commutator":
        beam_rdm_data = rdm_data # doesn't allocate: just binds a new name to the same object
    else:
        beam_rdm_data = RDMData(D=rdm_data.D, Gamma=rdm_data.Gamma)

    cost_function_constructor = _build_cost_function_constructor(args.cost_function, args.var_exponent)
    initial_bases = _build_initial_bases(args, beam_rdm_data, norb)

    opt_config = DecompositionOptimizerConfig(
        num_decos=args.num_decos, num_subdecos=args.num_subdecos,
        min_parent_cluster_size=args.min_parent_cluster_size,
        min_child_cluster_size=args.min_child_cluster_size,
        naturalize_children=not args.no_naturalize_children,
        target_num_clusters=args.target_num_clusters, max_rounds=args.max_rounds, maxiter=args.maxiter,
        optimize_rotation_in_beam_search=not args.no_orb_opt_in_beam_search,
    )
    trajectory = run_decomposition_optimizer(cost_function_constructor, beam_rdm_data, opt_config, initial_bases)

    if args.analyze_num_clusters is not None:
        wanted = set(args.analyze_num_clusters)
        entries = [d for d in trajectory if d.num_clusters in wanted]
        missing = wanted - {d.num_clusters for d in entries}
        if missing:
            logger.warning(f"--analyze-num-clusters {sorted(missing)} not present in trajectory; skipping.")
    else:
        entries = [d for d in trajectory if d.num_clusters >= 2]

    if not entries:
        logger.info("No trajectory entries selected for sector search (see --analyze-num-clusters).")
        return

    if args.analyze_num_clusters_lanczos is not None:
        wanted_lanczos = set(args.analyze_num_clusters_lanczos)
        lanczos_entries = [d for d in entries if d.num_clusters in wanted_lanczos]
        missing_lanczos = wanted_lanczos - {d.num_clusters for d in lanczos_entries}
        if missing_lanczos:
            logger.warning(
                f"--analyze-num-clusters-lanczos {sorted(missing_lanczos)} not present "
                "among the sector-search entries; skipping."
            )
    else:
        lanczos_entries = [max(entries, key=lambda d: d.num_clusters)]
    lanczos_cluster_counts = {d.num_clusters for d in lanczos_entries}

    search_config = SectorSearchConfig(
        num_sectors_to_retain=args.num_sectors_to_retain,
        max_cum_dim_to_retain=args.max_cum_dim_to_retain,
        max_elec_transfer=args.max_elec_transfer,
    )

    # rank_relevant_sectors derives its own energy-score tier purely from what's
    # populated on whatever RDMData it's handed -- so --max-sector-tier is implemented
    # by simply stripping fields off a copy, the same trick beam_rdm_data above uses.
    # solve_decomposition_selected_sectors (Part 2) always gets the untouched, full
    # rdm_data regardless -- it needs h1e/g2e_full no matter what --max-sector-tier says.
    if args.max_sector_tier is None or args.max_sector_tier >= 2:
        ranking_rdm_data = rdm_data
    elif args.max_sector_tier == 1:
        ranking_rdm_data = RDMData(D=rdm_data.D, Gamma=rdm_data.Gamma, h1e=rdm_data.h1e)
    else:
        ranking_rdm_data = RDMData(D=rdm_data.D, Gamma=rdm_data.Gamma)

    lanczos_config = LanczosSearchConfig(
        krylov_seed_depth=args.krylov_seed_depth,
        max_iterations=args.max_iterations,
        vectors_per_iteration=args.vectors_per_iteration,
        energy_tolerance=args.energy_tolerance,
        leakage_threshold=args.leakage_threshold,
    )

    part1_output_dir = _output_dir_for(args)
    part1_output_dir.mkdir(parents=True, exist_ok=True)
    part2_output_dir = _part2_output_dir_for(args)
    plots_dir = Path(args.lanczos_plots_dir) if args.lanczos_plots_dir is not None else part2_output_dir / "plots"

    timestamp = get_timestamp()
    git_hash = get_git_hash()

    for deco in entries:
        logger.info(
            f"Ranking sectors for {deco.num_clusters}-cluster decomposition "
            f"(sizes={[len(c) for c in deco.partition]})..."
        )
        ranked = rank_relevant_sectors(deco, ranking_rdm_data, nelec, search_config)
        for r in ranked[:10]:
            logger.info(
                f"  label={r.label} logw={r.weight_score:.4e} energy={r.energy_score} "
                f"(tier {r.energy_tier}) t={r.elec_transfer} dim={r.dimension}"
            )

        metadata = {
            "molecule": args.molecule, "basis_set": args.basis_set,
            "bond_length": args.bond_length if args.fcidump is None else None,
            "bond_angle": args.bond_angle if args.fcidump is None else None,
            "fcidump": args.fcidump, "timestamp": timestamp, "git_hash": git_hash,
            "norb": norb, "nelec": [int(x) for x in nelec], "dmrg_energy": float(dmrg_energy),
            "cost": args.cost_function, "num_clusters": deco.num_clusters,
            "cluster_sizes": [len(c) for c in deco.partition],
            "max_elec_transfer": args.max_elec_transfer,
            "num_sectors_to_retain": args.num_sectors_to_retain,
            "max_cum_dim_to_retain": args.max_cum_dim_to_retain,
            "max_sector_tier": args.max_sector_tier,
        }
        part1_output = {"metadata": metadata, "ranked_sectors": _sector_relevance_to_json(ranked)}
        part1_filename = f"sectors_{deco.num_clusters}clusters_{args.cost_function}_{timestamp}_{git_hash}.json"
        part1_filepath = part1_output_dir / part1_filename
        with open(part1_filepath, "w") as f:
            json.dump(part1_output, f, indent=2)
        logger.info(f"Sector ranking saved to {part1_filepath}")

        if deco.num_clusters not in lanczos_cluster_counts:
            continue

        logger.info(f"Running selected-sector Lanczos solve for {deco.num_clusters}-cluster decomposition...")
        lanczos_result = solve_decomposition_selected_sectors(
            deco, rdm_data, nelec, ecore, ranked, lanczos_config,
        )
        logger.info(
            f"  anchor energy={lanczos_result['anchor']['energy']:.8f} -> "
            f"final energy={lanczos_result['final']['energy']:.8f} "
            f"(dim={lanczos_result['final']['dimension']}, {lanczos_result['final']['stop_reason']})"
        )

        part2_output_dir.mkdir(parents=True, exist_ok=True)
        part2_metadata = dict(metadata)
        part2_metadata.update(
            {
                "krylov_seed_depth": args.krylov_seed_depth,
                "max_iterations": args.max_iterations,
                "vectors_per_iteration": args.vectors_per_iteration,
                "energy_tolerance": args.energy_tolerance,
                "leakage_threshold": args.leakage_threshold,
                "ecore": float(ecore),
            }
        )
        part2_output = {"metadata": part2_metadata, **lanczos_result}
        part2_filename = f"lanczos_{deco.num_clusters}clusters_{args.cost_function}_{timestamp}_{git_hash}.json"
        part2_filepath = part2_output_dir / part2_filename
        with open(part2_filepath, "w") as f:
            json.dump(part2_output, f, indent=2)
        logger.info(f"Lanczos result saved to {part2_filepath}")

        if not args.no_plots:
            plot_path = (
                plots_dir
                / f"convergence_{deco.num_clusters}clusters_{args.cost_function}_{timestamp}_{git_hash}.png"
            )
            _plot_convergence(lanczos_result, dmrg_energy, plot_path)
            logger.info(f"Convergence plot saved to {plot_path}")

    logger.info("Computation completed successfully!")


if __name__ == "__main__":
    main()
