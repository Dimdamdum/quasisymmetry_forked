"""Energy-sector diagnostics: decoupled and coupled (PT or reference) K."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import bisect

from src.coupled_energy_core import (
    CHEMICAL_PRECISION,
    COUPLED_ENERGY_DEGENERACY_FLOOR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TAU_PT,
    all_sector_eigenpair_candidates,
    one_shot_coupled_energy,
    reference_coupled_energy,
)


def find_first_negative(f, N):
    domain = range(1, N + 1)
    index = bisect.bisect_left(domain, x=True, key=lambda x: f(x) < 0)
    if index < len(domain):
        return domain[index]
    return -1

def diagonalize_sector_blocks(h_apply, sectors_dict, full_dim: int):
    """
    Diagonalize each symmetry block independently.

    Returns sector_data mapping sector key -> {idxs, evals, evecs_full}.
    """
    sector_data = {}
    for key, idxs in sectors_dict.items():
        h_local = _subspace_matrix(h_apply, idxs, full_dim)
        h_local = 0.5 * (h_local + h_local.conj().T)
        evals, evecs = np.linalg.eigh(h_local)

        evecs_full = []
        for j in range(evecs.shape[1]):
            v = np.zeros(full_dim, dtype=np.complex128)
            v[np.asarray(idxs, dtype=int)] = evecs[:, j]
            evecs_full.append(v)

        sector_data[key] = {
            "idxs": idxs,
            "evals": evals,
            "evecs_full": evecs_full,
        }
    return sector_data


def sector_data_from_gs_pairs(sectors_dict, sector_gs_pairs, full_dim: int):
    """Build sector_data from precomputed (evals, local_evecs) per sector."""
    sector_data = {}
    for key, idxs in sectors_dict.items():
        evals, evecs_local = sector_gs_pairs[key]
        evecs_full = []
        for j in range(evecs_local.shape[1]):
            v = np.zeros(full_dim, dtype=np.complex128)
            v[idxs] = evecs_local[:, j]
            evecs_full.append(v)
        sector_data[key] = {
            "idxs": idxs,
            "evals": evals,
            "evecs_full": evecs_full,
        }
    return sector_data


def decoupled_energy_test(sectors_dict, sector_gs_pairs):
    """E_dec_min = min_s lambda_min(H(s))."""
    best_e = None
    best_key = None
    best_dim = 0
    for key, idxs in sectors_dict.items():
        e0 = float(np.min(sector_gs_pairs[key][0]))
        if best_e is None or e0 < best_e:
            best_e = e0
            best_key = key
            best_dim = len(idxs)
    return best_e, best_key, best_dim


def coupled_energy_perturbation(
    h_apply: Callable[[np.ndarray], np.ndarray],
    sector_data,
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    max_total_vectors: int | None = None,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    backward_prune: bool = True,
):
    """One-shot PT ordering + nested variational coupled dimension."""
    candidates = all_sector_eigenpair_candidates(sector_data)
    return one_shot_coupled_energy(
        candidates,
        h_apply,
        e_exact=e_exact,
        tol=tol,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=degeneracy_floor,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    ).as_tuple()


def reference_coupled_energy_K_(
    h_apply: Callable[[np.ndarray], np.ndarray],
    sectors: dict,
    sector_eigs: dict,
    reference_vector: np.ndarray,
    e_exact: float,
    *,
    precision: float = CHEMICAL_PRECISION,
):
    # sector_reference_projection_amplitudes = {}
    # sector_reference_weights = {}
    # for sector_label, sector_bitstrings in sectors.items():
    #     projected_reference = reference_vector[sector_bitstrings]
    #     projection_amplitudes = sector_eigs[sector_label][1].T.conj() @ projected_reference
    #     sector_reference_projection_amplitudes[sector_label] = projection_amplitudes
    #     sector_reference_weights[sector_label] = abs(projection_amplitudes)**2

    all_state_labels = []
    sector_reference_projection_amplitudes = np.zeros((0,))
    for sector_label, sector_gs in sector_eigs.items():
        sector_bitstrings = sectors[sector_label]
        labels = [(sector_label, i) for i in range(sector_gs[1].shape[1])]
        all_state_labels.extend(labels)
        projected_reference = reference_vector[sector_bitstrings]
        projection_amplitudes = sector_eigs[sector_label][1].T.conj() @ projected_reference
        sector_reference_projection_amplitudes = np.concatenate(
            (sector_reference_projection_amplitudes, projection_amplitudes))

    weights_order = np.argsort(abs(sector_reference_projection_amplitudes))[::-1]

    def f(K):
        compressed_reference_vector = np.zeros_like(reference_vector, dtype="complex")

        for i in range(K):
            sector_label, sector_eigenstate_index = all_state_labels[weights_order[i]]
            support = sectors[sector_label]
            compressed_reference_vector[support] += (
                sector_eigs[sector_label][1][:, sector_eigenstate_index]
                * sector_reference_projection_amplitudes[weights_order[i]])

        compressed_reference_vector /= np.linalg.norm(compressed_reference_vector)

        e_K = compressed_reference_vector.T.conj() @ h_apply(compressed_reference_vector)
        return (e_K - e_exact - precision).real

    K = find_first_negative(f, len(all_state_labels))
    kept_states = [all_state_labels[i] for i in weights_order[:K]]
    kept_state_weights = abs(sector_reference_projection_amplitudes[weights_order[:K]]).tolist()
    return K, kept_states, kept_state_weights


def reference_coupled_energy_k(
    h_apply: Callable[[np.ndarray], np.ndarray],
    full_space_vectors_cat: np.ndarray,
    reference_vector: np.ndarray,
    e_exact: float,
    *,
    chemical_precision: float = CHEMICAL_PRECISION,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """
    Reference-overlap ordering + nested variational ``K``.

    ``full_space_vectors_cat`` columns are sector eigenstates. Returns
    ``(K, e_coupled, converged, order_indices)`` where ``order_indices`` indexes
    those columns in decreasing ``|<psi|Psi_ref>|^2`` order.
    """
    n = full_space_vectors_cat.shape[1]
    candidates = [
        (0.0, j, full_space_vectors_cat[:, j], 0) for j in range(n)
    ]
    result = reference_coupled_energy(
        candidates,
        h_apply,
        reference_vector,
        e_exact=e_exact,
        tol=chemical_precision,
        block_size=block_size,
    )
    order = np.asarray(result.order_indices, dtype=int)
    if result.K is None:
        return None, result.e_coupled, False, order
    return result.K, result.e_coupled, result.converged, order



def state_labels_for_columns(sector_gs_pairs):
    """(sector_label, block_index) for each column of the concatenated basis."""
    labels = []
    for sector_label, sector_gs in sector_gs_pairs.items():
        for i in range(sector_gs[1].shape[1]):
            labels.append((sector_label, i))
    return labels


def _subspace_matrix(h_apply, support, full_dim: int):
    dim = len(support)
    h_sub = np.zeros((dim, dim), dtype=np.complex128)
    for i, big_index in enumerate(support):
        x = np.zeros(full_dim, dtype=np.complex128)
        x[big_index] = 1.0
        y = h_apply(x)
        h_sub[:, i] = y[support]
    return h_sub
