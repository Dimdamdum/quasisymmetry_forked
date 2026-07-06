"""Energy-sector diagnostics: decoupled and coupled (PT or reference) K."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.coupled_energy_core import (
    COUPLED_ENERGY_DEGENERACY_FLOOR,
    all_sector_eigenpair_candidates,
    greedy_coupled_energy,
)

CHEMICAL_PRECISION = 0.0016


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
    tol: float = 1e-8,
    max_total_vectors: int | None = None,
    coupling_tol: float = 1e-12,
    energy_change_tol: float = 1e-12,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
):
    """PT-screened greedy coupled-energy selection over sector eigenvectors."""
    candidates = all_sector_eigenpair_candidates(sector_data)
    return greedy_coupled_energy(
        candidates,
        h_apply,
        e_exact=e_exact,
        tol=tol,
        max_total_vectors=max_total_vectors,
        coupling_tol=coupling_tol,
        energy_change_tol=energy_change_tol,
        degeneracy_floor=degeneracy_floor,
    )


def reference_coupled_energy_k(
    h_apply: Callable[[np.ndarray], np.ndarray],
    full_space_vectors_cat: np.ndarray,
    reference_vector: np.ndarray,
    e_exact: float,
    *,
    chemical_precision: float = CHEMICAL_PRECISION,
):
    """
    Greedy K from FCI (reference) coefficients on sector eigenvectors.

    Returns (K, e_coupled, converged, chosen_column_indices).
    """
    coefficients = full_space_vectors_cat.T.conj() @ reference_vector
    weights_order = np.argsort(abs(coefficients))[::-1]

    projected = full_space_vectors_cat @ coefficients
    projected /= np.linalg.norm(projected)
    e_full = float(np.real(projected.T.conj() @ h_apply(projected)))
    if e_full > e_exact + chemical_precision:
        return None, e_full, False, weights_order

    def f(k):
        compressed = np.zeros_like(coefficients, dtype=np.complex128)
        compressed[weights_order[:k]] = coefficients[weights_order[:k]]
        compressed /= np.linalg.norm(compressed)
        vec = full_space_vectors_cat @ compressed
        e_k = float(np.real(vec.T.conj() @ h_apply(vec)))
        return (e_k - e_exact - chemical_precision).real

    k_min = _find_first_negative(f, full_space_vectors_cat.shape[1])
    if k_min < 0 or k_min >= full_space_vectors_cat.shape[1]:
        return None, e_full, False, weights_order

    converged = True
    compressed = np.zeros_like(coefficients, dtype=np.complex128)
    compressed[weights_order[:k_min]] = coefficients[weights_order[:k_min]]
    compressed /= np.linalg.norm(compressed)
    vec = full_space_vectors_cat @ compressed
    e_coupled = float(np.real(vec.T.conj() @ h_apply(vec)))
    return k_min, e_coupled, converged, weights_order


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


def _find_first_negative(f, n):
    for k in range(1, n + 1):
        if f(k) < 0:
            return k
    return -1
