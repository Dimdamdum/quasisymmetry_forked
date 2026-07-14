"""Coupled-dimension selection over sector eigenstates.

Provides:

* one-shot Epstein--Nesbet PT ordering + nested variational benchmark
  (recommended default for determining ``K``);
* legacy multi-pass PT-screened greedy selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

COUPLED_ENERGY_DEGENERACY_FLOOR = 1e-8
CHEMICAL_PRECISION = 0.0016
DEFAULT_TAU_PT = 1e-12
DEFAULT_BLOCK_SIZE = 1


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


# ---------------------------------------------------------------------------
# One-shot PT ordering + nested variational K (recommended default)
# ---------------------------------------------------------------------------


@dataclass
class OneShotCoupledResult:
    """Result of the one-shot PT + nested variational protocol."""

    e_coupled: float | None
    K: int
    K_pt: int
    converged: bool
    chosen_keys: list[tuple[object, int]]
    order_indices: list[int] = field(default_factory=list)
    pt_weights: np.ndarray = field(default_factory=lambda: np.asarray([]))
    energies: list[float] = field(default_factory=list)


def one_shot_pt_weight(
    coupling: complex,
    delta: float,
    *,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> float:
    """Epstein--Nesbet second-order importance ``|v|^2 / Delta``.

    A vanishing (or near-vanishing) denominator with nonzero coupling yields
    ``+inf``, so the state is ranked at the front of the ordered list.
    """
    numer = float(abs(coupling) ** 2)
    if numer == 0.0:
        return 0.0
    if abs(delta) <= degeneracy_floor:
        return float("inf")
    return numer / abs(delta)


def one_shot_pt_order(
    energies: Sequence[float],
    couplings_to_ref: Sequence[complex],
    ref_index: int,
    *,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
) -> tuple[list[int], np.ndarray]:
    """Order candidates by decreasing one-shot PT importance relative to ``ref``.

    Returns ``(order, weights)`` where ``order[0] == ref_index``, the remaining
    entries are external candidates sorted by ``w_alpha``, and ``weights[i]`` is
    the PT weight of candidate ``i`` (``weights[ref_index] == +inf`` sentinel).
    """
    n = len(energies)
    if n == 0:
        return [], np.asarray([])
    if len(couplings_to_ref) != n:
        raise ValueError("couplings_to_ref must match energies length")

    e0 = float(energies[ref_index])
    weights = np.zeros(n, dtype=np.float64)
    weights[ref_index] = float("inf")
    external: list[tuple[float, int]] = []
    for index in range(n):
        if index == ref_index:
            continue
        weight = one_shot_pt_weight(
            complex(couplings_to_ref[index]),
            float(energies[index]) - e0,
            degeneracy_floor=degeneracy_floor,
        )
        weights[index] = weight
        external.append((weight, index))

    # Stable tie-break: higher weight first, then lower energy, then index.
    external.sort(key=lambda item: (-item[0], float(energies[item[1]]), item[1]))
    order = [ref_index] + [index for _weight, index in external]
    return order, weights


def k_pt_from_ordered_weights(
    ordered_external_weights: Sequence[float],
    tau_pt: float,
) -> int:
    """``K_PT = 1 + max{r : w_{alpha_r} >= tau_PT}`` (or 1 if none pass)."""
    k_pt = 1
    for rank, weight in enumerate(ordered_external_weights, start=1):
        if weight >= tau_pt:
            k_pt = 1 + rank
        else:
            break
    return k_pt


def build_candidate_hamiltonian(
    candidates: Sequence[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    *,
    order: Sequence[int] | None = None,
) -> np.ndarray:
    """Dense Hamiltonian matrix in the (optionally reordered) candidate basis."""
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    indices = list(range(n)) if order is None else list(order)
    if len(indices) != n:
        raise ValueError("order must permute all candidates")

    vecs = [candidates[i][2] for i in indices]
    h_vecs = [apply_h(vec) for vec in vecs]
    h_mat = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        for i in range(j + 1):
            value = complex(np.vdot(vecs[i], h_vecs[j]))
            h_mat[i, j] = value
            h_mat[j, i] = np.conjugate(value)
    return 0.5 * (h_mat + h_mat.conj().T)


def nested_ground_energy(h_ordered: np.ndarray, k: int) -> float:
    """Lowest eigenvalue of the leading principal ``k x k`` submatrix."""
    if k <= 0:
        raise ValueError("k must be positive")
    sub = h_ordered[:k, :k]
    return float(np.linalg.eigvalsh(sub)[0])


def find_k_epsilon(
    h_ordered: np.ndarray,
    e_ref: float,
    epsilon: float,
    k_start: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int | None, list[float], bool]:
    """Smallest nested dimension with ``E_0^(K) - e_ref <= epsilon``.

    Starts at ``k_start`` (typically ``K_PT``), expands or contracts in blocks
    of ``block_size`` to bracket the threshold, then binary-searches the
    bracket.  Returns ``(K_epsilon, prefix_energies, converged)`` where
    ``prefix_energies[k-1]`` is ``E_0^(k)`` for every evaluated ``k`` that was
    needed (sparse cache expanded to a dense prefix list up to the returned K,
    or up to ``n`` on failure).
    """
    n = h_ordered.shape[0]
    if n == 0:
        return None, [], False
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    cache: dict[int, float] = {}

    def energy_at(k: int) -> float:
        if k not in cache:
            cache[k] = nested_ground_energy(h_ordered, k)
        return cache[k]

    def acceptable(k: int) -> bool:
        return energy_at(k) - e_ref <= epsilon

    k_start = min(max(1, k_start), n)

    if acceptable(k_start):
        # Shrink until the failure boundary is bracketed: lo fails, hi succeeds.
        hi = k_start
        lo = 0
        k = k_start - block_size
        while k >= 1:
            if acceptable(k):
                hi = k
                k -= block_size
            else:
                lo = k
                break
        if hi == 1 and acceptable(1):
            lo = 0
    else:
        # Grow until the success boundary is bracketed: lo fails, hi succeeds.
        lo = k_start
        hi = None
        k = k_start + block_size
        while k <= n:
            if acceptable(k):
                hi = k
                break
            lo = k
            k += block_size
        if hi is None:
            if acceptable(n):
                hi = n
            else:
                energies = [energy_at(k) for k in range(1, n + 1)]
                return None, energies, False

    # Binary search for the smallest acceptable K in (lo, hi].
    left = lo + 1
    right = hi
    while left < right:
        mid = (left + right) // 2
        if acceptable(mid):
            right = mid
        else:
            left = mid + 1

    k_eps = left
    energies = [energy_at(k) for k in range(1, k_eps + 1)]
    return k_eps, energies, True


def one_shot_coupled_energy(
    candidates: list[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    max_total_vectors: int | None = None,
) -> OneShotCoupledResult:
    """One-shot PT ranking + nested variational determination of ``K``.

    1. Identify the lowest sector eigenstate ``psi_0``.
    2. Rank all other candidates by ``w = |<psi|V|psi_0>|^2 / Delta``.
    3. Form ``K_PT`` from the individual threshold ``tau_pt``.
    4. Build the Hamiltonian once in this fixed order and locate the minimal
       nested dimension ``K`` with variational error ``<= tol`` relative to
       ``e_exact`` (when provided).
    """
    if not candidates:
        return OneShotCoupledResult(
            e_coupled=None, K=0, K_pt=0, converged=False, chosen_keys=[]
        )

    pool = list(candidates)
    energies = [float(item[0]) for item in pool]
    ref_index = int(np.argmin(energies))
    ref_vec = pool[ref_index][2]
    h_ref = apply_h(ref_vec)
    couplings = [complex(np.vdot(item[2], h_ref)) for item in pool]

    order, weights = one_shot_pt_order(
        energies,
        couplings,
        ref_index,
        degeneracy_floor=degeneracy_floor,
    )
    ordered_external_weights = [float(weights[i]) for i in order[1:]]
    k_pt = k_pt_from_ordered_weights(ordered_external_weights, tau_pt)

    if max_total_vectors is not None:
        order = order[: max(1, max_total_vectors)]
        k_pt = min(k_pt, len(order))

    h_ordered = build_candidate_hamiltonian(pool, apply_h, order=order)
    n = h_ordered.shape[0]
    k_pt = min(max(1, k_pt), n)

    def keys_for(k: int) -> list[tuple[object, int]]:
        return [(pool[order[i]][1], pool[order[i]][3]) for i in range(k)]

    if e_exact is None:
        e_coupled = nested_ground_energy(h_ordered, k_pt)
        return OneShotCoupledResult(
            e_coupled=e_coupled,
            K=k_pt,
            K_pt=k_pt,
            converged=False,
            chosen_keys=keys_for(k_pt),
            order_indices=order,
            pt_weights=weights,
            energies=[nested_ground_energy(h_ordered, k) for k in range(1, k_pt + 1)],
        )

    k_eps, energies_curve, converged = find_k_epsilon(
        h_ordered,
        float(e_exact),
        tol,
        k_pt,
        block_size=block_size,
    )
    if k_eps is None:
        e_coupled = nested_ground_energy(h_ordered, n)
        return OneShotCoupledResult(
            e_coupled=e_coupled,
            K=n,
            K_pt=k_pt,
            converged=False,
            chosen_keys=keys_for(n),
            order_indices=order,
            pt_weights=weights,
            energies=energies_curve,
        )

    e_coupled = energies_curve[k_eps - 1]
    return OneShotCoupledResult(
        e_coupled=e_coupled,
        K=k_eps,
        K_pt=k_pt,
        converged=converged,
        chosen_keys=keys_for(k_eps),
        order_indices=order,
        pt_weights=weights,
        energies=energies_curve,
    )


def one_shot_coupled_energy_tuple(
    candidates: list[tuple[float, object, np.ndarray, int]],
    apply_h: Callable[[np.ndarray], np.ndarray],
    **kwargs,
) -> tuple[float | None, int, bool, list[tuple[object, int]]]:
    """Same as :func:`one_shot_coupled_energy` with the legacy 4-tuple return."""
    result = one_shot_coupled_energy(candidates, apply_h, **kwargs)
    return result.e_coupled, result.K, result.converged, result.chosen_keys
