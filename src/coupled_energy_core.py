"""Incremental coupled-energy greedy selection (dense backend)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

COUPLED_ENERGY_DEGENERACY_FLOOR = 1e-8


def all_sector_eigenpair_candidates(
    sector_data,
) -> list[tuple[float, object, np.ndarray, int]]:
    """All block eigenpairs (energy, sector key, full-space vector, block index)."""
    candidates: list[tuple[float, object, np.ndarray, int]] = []
    for key, data in sector_data.items():
        for block_index, (energy, vector) in enumerate(
            zip(data["evals"], data["evecs_full"])
        ):
            candidates.append((float(energy), key, vector, int(block_index)))
    candidates.sort(key=lambda item: item[0])
    return candidates


@dataclass
class CoupledSpanState:
    chosen_vecs: list[np.ndarray]
    h_vecs: list[np.ndarray]
    h_proj: np.ndarray
    psi0: np.ndarray
    e_proj: float


def augment_h_proj(
    h_proj: np.ndarray,
    h_cols: list[complex] | np.ndarray,
    h_new_new: float,
) -> np.ndarray:
    k = h_proj.shape[0]
    h_trial = np.zeros((k + 1, k + 1), dtype=np.complex128)
    h_trial[:k, :k] = h_proj
    h_cols_arr = np.asarray(h_cols, dtype=np.complex128)
    h_trial[:k, k] = h_cols_arr
    h_trial[k, :k] = h_cols_arr.conj()
    h_trial[k, k] = h_new_new
    return 0.5 * (h_trial + h_trial.conj().T)


def ground_from_h_proj(h_proj: np.ndarray) -> tuple[float, np.ndarray]:
    evals, evecs = np.linalg.eigh(h_proj)
    index = int(np.argmin(evals))
    return float(evals[index]), np.asarray(evecs[:, index], dtype=np.complex128)


def two_state_ground_energy(e0: float, e_new: float, v: complex) -> float:
    gap = e0 - e_new
    return float(0.5 * (e0 + e_new - np.sqrt(gap * gap + 4.0 * abs(v) ** 2)))


def h_cols_from_h_vecs(h_vecs: list[np.ndarray], cand_vec: np.ndarray) -> list[complex]:
    return [np.vdot(h_vec, cand_vec) for h_vec in h_vecs]


def max_coupling_from_h_vecs(h_vecs: list[np.ndarray], cand_vec: np.ndarray) -> float:
    if not h_vecs:
        return float("inf")
    return max(float(abs(np.vdot(h_vec, cand_vec))) for h_vec in h_vecs)


def trial_ground_energy_incremental(
    h_proj: np.ndarray,
    h_cols: list[complex],
    h_new_new: float,
) -> float:
    if h_proj.shape[0] == 1:
        return two_state_ground_energy(
            float(np.real(h_proj[0, 0])), h_new_new, complex(h_cols[0])
        )
    h_trial = augment_h_proj(h_proj, h_cols, h_new_new)
    return float(np.linalg.eigvalsh(h_trial)[0])


def improves_toward_fci(
    e_new: float,
    e_proj: float,
    e_exact: float | None,
    energy_change_tol: float,
) -> bool:
    if e_exact is not None:
        return abs(e_new - e_exact) < abs(e_proj - e_exact) - energy_change_tol
    return e_new < e_proj - energy_change_tol


def perturbation_may_improve(
    psi0: np.ndarray,
    h_cols: list[complex],
    e0: float,
    e_new: float,
    e_proj: float,
    e_exact: float | None,
    *,
    coupling_tol: float,
    energy_change_tol: float,
    degeneracy_floor: float,
) -> bool:
    """Return True when a full (k+1)-dim trial should run; False to skip."""
    h_cols_arr = np.asarray(h_cols, dtype=np.complex128)
    max_coupling = float(np.max(np.abs(h_cols_arr))) if h_cols_arr.size else 0.0
    v0 = complex(np.vdot(psi0, h_cols_arr))

    if max_coupling > coupling_tol and abs(v0) <= coupling_tol:
        return True

    denom = abs(e0 - e_new)
    if denom < degeneracy_floor:
        return True

    delta_e = abs(v0) ** 2 / denom
    e_est = e_proj - delta_e
    return improves_toward_fci(e_est, e_proj, e_exact, energy_change_tol)


def projected_ground_energy_dense(h_dense: np.ndarray, vecs: list[np.ndarray]) -> float:
    """Reference implementation for tests (full V† H V rebuild)."""
    v = np.column_stack(vecs)
    h_proj = v.conj().T @ h_dense @ v
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    return float(np.linalg.eigvalsh(h_proj)[0])


def greedy_coupled_energy(
    candidates: list[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    *,
    e_exact: float | None = None,
    tol: float = 1e-8,
    max_total_vectors: int | None = None,
    coupling_tol: float = 1e-12,
    energy_change_tol: float = 1e-12,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> tuple[float | None, int, bool, list[tuple[object, int]]]:
    if not candidates:
        return None, 0, False, []

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_keys: list[tuple[object, int]] = []
    chosen_indices: set[int] = set()
    state: CoupledSpanState | None = None
    converged = False

    while True:
        added_this_pass = False
        for index, (energy, key, vec, block_index) in enumerate(candidates):
            if index in chosen_indices:
                continue
            if len(chosen_keys) >= max_total_vectors:
                break

            if state is None:
                hcand = apply_h(vec)
                e_new = float(energy)
                h_proj = np.array([[e_new]], dtype=np.complex128)
                psi0 = np.array([1.0], dtype=np.complex128)
                state = CoupledSpanState(
                    chosen_vecs=[vec],
                    h_vecs=[hcand],
                    h_proj=h_proj,
                    psi0=psi0,
                    e_proj=e_new,
                )
            else:
                hcand = apply_h(vec)
                if max_coupling_from_h_vecs(state.h_vecs, vec) <= coupling_tol:
                    continue

                h_cols = h_cols_from_h_vecs(state.h_vecs, vec)
                if not perturbation_may_improve(
                    state.psi0,
                    h_cols,
                    state.e_proj,
                    float(energy),
                    state.e_proj,
                    e_exact,
                    coupling_tol=coupling_tol,
                    energy_change_tol=energy_change_tol,
                    degeneracy_floor=degeneracy_floor,
                ):
                    continue

                e_new = trial_ground_energy_incremental(state.h_proj, h_cols, float(energy))
                if not improves_toward_fci(
                    e_new, state.e_proj, e_exact, energy_change_tol
                ):
                    continue

                h_trial = augment_h_proj(state.h_proj, h_cols, float(energy))
                e_accept, psi0 = ground_from_h_proj(h_trial)
                state = CoupledSpanState(
                    chosen_vecs=[*state.chosen_vecs, vec],
                    h_vecs=[*state.h_vecs, hcand],
                    h_proj=h_trial,
                    psi0=psi0,
                    e_proj=e_accept,
                )

            chosen_indices.add(index)
            chosen_keys.append((key, block_index))
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
