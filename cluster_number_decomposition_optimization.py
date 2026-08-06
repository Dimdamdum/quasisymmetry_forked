from __future__ import annotations

# This file is piecewise AI-written and human-verified

"""
Cluster Number Decomposition Optimizer

Jointly optimizes BOTH the orbital-cluster partition and the orbital basis of
a quasisymmetry decomposition of the single-orbital space, via beam search
over cluster splits combined with continuous orbital-rotation optimization
(as in cluster_numbers_metrics.py) restricted to the subspace being split.

The object being optimized is a decomposition deco = (P, U):
  - P: a partition of range(norb) into clusters (list of lists of orbital
    indices) -- which orbitals are grouped into which cluster (each cluster
    defines a quasisymmetry operator, i.e. the number operator for that cluster).
  - U: a norb x norb orthogonal matrix, the orbital basis relative to a fixed
    reference basis (e.g. MOs or NatOs).

Algorithm (see Decomposition, run_decomposition_optimizer):
  1. Start from `num_decos` copies of the trivial one-cluster decomposition
     (all orbitals in a single cluster), optionally seeded from different
     reference bases (`initial_bases`).
  2. Each round, every decomposition in the beam produces up to `num_subdecos`
     children: pick a cluster (largest current number-variance first), split
     it into two by spectral (Fiedler) bisection of its pairwise
     number-operator covariance graph, then re-optimize the orbital rotation
     *restricted to that cluster's subspace* (all other clusters, and their
     orbitals, are left untouched) against `cost_function_constructor`.
  3. Keep the best `num_decos` children (by cost) as the new beam.
  4. Repeat until every deco in the beam is down to singleton clusters (or
     `min_cluster_size`), or `target_num_clusters` is reached.

The discrete part of the search (which cluster to split, how to bisect it) is
always driven by the cheap 1-/2-RDM covariance structure, independent of
`cost_function_constructor`; only the continuous per-split optimization and
the beam ranking use the supplied cost function. This matters because the
commutator-based cost needs the 3-/4-RDM rotated at every candidate (expensive
for a search touching many nodes) -- prefer make_variance_cost_constructor (or
another cheap, 1-/2-RDM-only cost) to drive the search itself, and reserve
make_commutator_cost_constructor for polish_decomposition on the trajectory
endpoint(s) you actually care about.

`run_decomposition_optimizer` returns the full "trajectory" (one winner decomposition
per cluster count) rather than a single winner. This is useful for
the following empirical fact (valid for variance cost). The raw cost
turns out to favor fewer clusters, but fewer clusters result in large sectors,
so which point on the trajectory is "best" really depends on the downstream
metrics (sectors or decoupled states-based analysis). Use
best_decomposition(trajectory, num_clusters=...) to pick one out once
you know what you're optimizing for.

Other important empirical fact (again, valid for variance cost): at eachr round,
raw cost prefers splitting the biggest cluster C into (|C| - 1) + 1 subclusters. 
But again, this preference may not correspond to low values of downstream metrics.
Therefore, the minimal cluster size and target number of clusters are probably two
important hyperparameters (CLI options --min-cluster-size and --target-num-clusters).

CLI usage (mirrors cluster_numbers_metrics.py -- runs DMRG via that script's
own compute_dmrg/extract_rdms, then runs the beam search on the resulting
RDMs):
    python cluster_number_decomposition_optimization.py h2o sto-3g 2.0 variance --bond-angle 104.5
    python cluster_number_decomposition_optimization.py h4_square 6-31g 1.0 commutator --num-decos 6 --num-subdecos 8 --fiedler-reorder

Run with --help for the full list of options. Most physically meaningful ones are probably
--min-cluster-size (default: 1)
--target-num-clusters (default: run down to singleton clusters)
--initial-basis (default: MOs)
--fiedler-reorder (on initial basis - default: False)
--max-transfers (default: 1 2 3)
Main hyperparameters are probably
--bond-dim (default: 150)
--n-sweeps (default: 50)
--num-decos (default: 4)
--num-subdecos (default: 4)

Library usage (skips DMRG entirely -- bring your own RDMs, e.g. computed
elsewhere, or via cluster_numbers_metrics.py's compute_dmrg/extract_rdms):
    from cluster_number_decomposition_optimization import (
        RDMData, DecompositionOptimizerConfig, run_decomposition_optimizer,
        make_variance_cost_constructor, make_commutator_cost_constructor,
        polish_decomposition, best_decomposition,
    )

    # e.g. from cluster_numbers_metrics.py's own pipeline:
    #   solver, dmrg_energy, h1e, g2e, ecore, nelec, result = compute_dmrg(config)
    #   D, Gamma = extract_rdms(solver, result.mps_tag)
    rdm_data = RDMData(D=D, Gamma=Gamma)

    config = DecompositionOptimizerConfig(num_decos=4, num_subdecos=6, min_cluster_size=2)
    trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), rdm_data, config)

    deco = best_decomposition(trajectory, num_clusters=3)
    deco = polish_decomposition(deco, rdm_data, make_variance_cost_constructor())

Output (CLI only): one JSON file per run (partition + orbital rotation U for
every trajectory entry, polished or not) plus an optional cost-vs-#clusters
plot, both under a molecule/basis_set/bond_.../[angle_.../] directory tree
matching cluster_numbers_metrics.py's outputs_/plots/ convention:
    outputs_/cluster_number_decomposition/{molecule}/{basis_set}/bond_{length}/[angle_{angle}/]
        decomposition_{cost}_{timestamp}_{git_hash}.json
    plots/cluster_number_decomposition/{molecule}/{basis_set}/bond_{length}/[angle_{angle}/]
        cost_vs_clusters_{cost}_{timestamp}.png

Unless --skip-sector-analysis is passed, each "winner" decomposition (each
trajectory entry, polished if enabled) additionally gets a full K-sectors /
K-states sector analysis, using cluster_numbers_metrics.py's OWN
compute_sector_analysis-equivalent logic, get_K_sectors_values_energies /
get_K_states_values_energies, save_metrics and generate_plots verbatim. The
only difference from cluster_numbers_metrics.py's version is that there each
curve is a different orbital basis U for one FIXED cluster_matrix, while here
each curve is a different trajectory entry, each with its OWN cluster_matrix
(so `sectors` is rebuilt per entry rather than shared). Output nests under
max_transfers_{N}/ exactly like cluster_numbers_metrics.py:
    outputs_/cluster_number_decomposition/{molecule}/.../max_transfers_{N}/results_{cost}_{timestamp}_{git_hash}.json
    plots/cluster_number_decomposition/{molecule}/.../max_transfers_{N}/energy_vs_sectors_..., retained_sectors_bar_chart_..., energy_vs_states_..., retained_states_bar_chart_...
By default only trajectory entries with >= 2 clusters are analyzed (the
1-cluster decomposition carries no symmetry information and is the most
expensive case for K-states mode to diagonalize); use --analyze-num-clusters
to pick specific cluster counts instead.

See cluster_numbers_metrics.py for the fixed-cluster version this generalizes,
and tests/test_cluster_number_decomposition_optimization.py for correctness
checks of the block-local optimization and Fiedler splitting.
"""

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass
from functools import cache
from math import comb
from pathlib import Path
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize

jax.config.update("jax_enable_x64", True)

from src.cluster_number_operators import (
    build_loc_number_evaluator,
    get_cluster_indices,
    number_and_parity_symmetry_sectors,
    params_to_U_jax,
    number_variance_cost,
    number_eval_eq_cost,
    extremality_cost,
    number_commutator_cost_v1,
    number_commutator_cost_v2,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class RDMData:
    """Bundle of (spin-summed) RDMs and, optionally, electronic integrals in a
    fixed reference orbital basis. Which fields beyond D, Gamma are populated
    depends on which cost function will be used -- only the commutator cost
    (make_commutator_cost_constructor) needs rdm3, rdm4, h1e, g2e_full."""

    D: np.ndarray
    Gamma: np.ndarray
    rdm3: np.ndarray | None = None
    rdm4: np.ndarray | None = None
    h1e: np.ndarray | None = None
    g2e_full: np.ndarray | None = None

    @property
    def norb(self) -> int:
        return self.D.shape[0]


@dataclass
class Decomposition:
    """A direct-sum decomposition deco = (P, U) of the single-orbital space:
    P is a partition of range(norb) into clusters (list of lists of orbital
    indices), U is the accumulated norb x norb orthogonal rotation from the
    fixed reference basis to this decomposition's orbital basis."""

    partition: list[list[int]]
    U: np.ndarray
    cost: float

    @property
    def num_clusters(self) -> int:
        return len(self.partition)


CostFunctionConstructor = Callable[[RDMData, np.ndarray], Callable[[np.ndarray], float]]


@dataclass
class DecompositionOptimizerConfig:
    """Hyperparameters for run_decomposition_optimizer."""

    num_decos: int = 4  # beam width
    num_subdecos: int = 4  # splits attempted per deco per round
    min_cluster_size: int = 1  # clusters at or below this size are never split
    target_num_clusters: int | None = None  # stop once the beam reaches this many clusters
    max_rounds: int | None = None  # extra safety cap on rounds
    maxiter: int = 200  # L-BFGS-B iterations for each per-split local optimization

    def __post_init__(self):
        if self.num_decos < 1:
            raise ValueError("num_decos must be >= 1")
        if self.num_subdecos < 1:
            raise ValueError("num_subdecos must be >= 1")
        if self.num_subdecos < 2:
            logger.warning(
                "num_subdecos=1: each deco produces only one child per round, so the beam "
                "never actually selects among alternatives (no real search happening)."
            )
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1")


# =============================================================================
# Partition <-> cluster_matrix helpers
# =============================================================================


def partition_to_cluster_matrix(partition: list[list[int]], norb: int) -> np.ndarray:
    """Binary (n_clusters x norb) matrix representation of a partition."""
    matrix = np.zeros((len(partition), norb), dtype=int)
    for i, cluster in enumerate(partition):
        matrix[i, list(cluster)] = 1
    return matrix


def validate_partition(partition: list[list[int]], norb: int) -> None:
    """Check that `partition` covers range(norb) exactly once, with no empty clusters."""
    seen: set[int] = set()
    for cluster in partition:
        if len(cluster) == 0:
            raise ValueError("Partition contains an empty cluster.")
        overlap = seen & set(cluster)
        if overlap:
            raise ValueError(f"Orbital(s) {sorted(overlap)} appear in more than one cluster.")
        seen.update(cluster)
    if seen != set(range(norb)):
        missing = set(range(norb)) - seen
        raise ValueError(f"Partition does not cover all orbitals; missing {sorted(missing)}.")


# =============================================================================
# RDM / integral rotation
# =============================================================================


def rotate_rdm_data(rdm_data: RDMData, U: np.ndarray) -> RDMData:
    """Rotate all populated tensors of `rdm_data` from the reference basis into
    the basis defined by the orthogonal matrix U (rows = new basis, columns =
    reference basis, matching params_to_U_jax's convention). Mirrors the
    rotation formulas already used for NatOs in
    cluster_numbers_metrics.optimize_orbital_basis, and internally in
    number_commutator_cost_v1/v2 (and validated in
    tests/test_cluster_number_operators.py:TestRdmRotationsLocalNumbers)."""
    U = np.asarray(U)
    Uc = U.conj()

    D_new = Uc @ rdm_data.D @ U.T
    Gamma_new = np.einsum("pi,qj,rk,sl,ijkl->pqrs", Uc, Uc, U, U, rdm_data.Gamma, optimize=True)

    rdm3_new = None
    if rdm_data.rdm3 is not None:
        rdm3_new = np.einsum(
            "pi,qj,rk,sl,tm,un,ijklmn->pqrstu",
            Uc, Uc, Uc, U, U, U, rdm_data.rdm3, optimize=True,
        )

    rdm4_new = None
    if rdm_data.rdm4 is not None:
        rdm4_new = np.einsum(
            "pi,qj,rk,sl,tm,un,vo,wa,ijklmnoa->pqrstuvw",
            Uc, Uc, Uc, Uc, U, U, U, U, rdm_data.rdm4, optimize=True,
        )

    h1e_new = None
    if rdm_data.h1e is not None:
        h1e_new = np.einsum("pr,qs,rs->pq", U, Uc, rdm_data.h1e, optimize=True)

    g2e_new = None
    if rdm_data.g2e_full is not None:
        g2e_new = np.einsum(
            "pr,tu,qs,vz,rsuz->pqtv", U, U, Uc, Uc, rdm_data.g2e_full, optimize=True,
        )

    return RDMData(D=D_new, Gamma=Gamma_new, rdm3=rdm3_new, rdm4=rdm4_new, h1e=h1e_new, g2e_full=g2e_new)


# =============================================================================
# Cluster statistics and Fiedler splitting
# =============================================================================


def cluster_number_stats(rdm_data: RDMData, partition: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """One-shot <n_p> and <n_p n_q> (restricted to pairs within the same
    cluster of `partition`, zero elsewhere) for RDMs already expressed in the
    basis of interest -- call rotate_rdm_data first if starting from a
    reference basis. A single call computes stats for every cluster in
    `partition` at once (reused for both split-priority ranking and Fiedler
    bisection of any of its clusters)."""
    norb = rdm_data.norb
    cluster_matrix = partition_to_cluster_matrix(partition, norb)
    evaluator = build_loc_number_evaluator(
        jnp.array(rdm_data.D), jnp.array(rdm_data.Gamma), cluster_matrix=cluster_matrix, with_ghost=False
    )
    n1, n2 = evaluator(jnp.eye(norb))
    return np.asarray(n1), np.asarray(n2)


def cluster_variance(n1: np.ndarray, n2: np.ndarray, cluster: list[int]) -> float:
    """Var(N_cluster) given the n1, n2 statistics from cluster_number_stats."""
    idx = np.asarray(cluster)
    expected_n = n1[idx].sum()
    expected_n_sq = n2[np.ix_(idx, idx)].sum()
    return float(expected_n_sq - expected_n**2)


def fiedler_bipartition(n1: np.ndarray, n2: np.ndarray, cluster: list[int]) -> tuple[list[int], list[int]]:
    """Split `cluster` into two groups by spectral (Fiedler) bisection of the
    pairwise number-operator covariance graph |Cov(n_p, n_q)|: orbitals whose
    occupations fluctuate together are kept on the same side. This is a cheap
    proxy for "the split that will end up costing the least", used to seed the
    subsequent continuous optimization over Var(N_V1) + Var(N_V2)."""
    cluster = list(cluster)
    k = len(cluster)
    if k < 2:
        raise ValueError("Cannot split a cluster of size < 2.")
    if k == 2:
        return [cluster[0]], [cluster[1]]

    idx = np.asarray(cluster)
    n1_c = n1[idx]
    n2_c = n2[np.ix_(idx, idx)]
    W = np.abs(n2_c - np.outer(n1_c, n1_c))
    np.fill_diagonal(W, 0.0)

    if not np.any(W > 1e-12):
        # No resolvable correlation structure: fall back to a balanced index split.
        half = k // 2
        return cluster[:half], cluster[half:]

    L = np.diag(W.sum(axis=1)) - W
    _, evecs = np.linalg.eigh(L)
    fiedler_vec = evecs[:, 1]  # eigenvector of the second-smallest eigenvalue

    side = fiedler_vec > 0
    if side.all() or not side.any():
        # Degenerate/near-zero Fiedler component: fall back to a median split.
        order = np.argsort(fiedler_vec)
        half = k // 2
        side = np.zeros(k, dtype=bool)
        side[order[half:]] = True

    V1 = idx[side].tolist()
    V2 = idx[~side].tolist()
    return V1, V2


def fiedler_orbital_order(D: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """Permutation of range(norb) obtained by sorting orbitals along the
    Fiedler vector of the full pairwise number-operator covariance graph, so
    strongly-correlated orbitals end up close together. Useful for building an
    `initial_bases` entry, e.g.:
        perm = fiedler_orbital_order(D_MOs, Gamma_MOs)
        U0 = permutation_to_unitary(perm)
    """
    norb = D.shape[0]
    n1, n2 = cluster_number_stats(RDMData(D=D, Gamma=Gamma), [list(range(norb))])
    W = np.abs(n2 - np.outer(n1, n1))
    np.fill_diagonal(W, 0.0)
    if not np.any(W > 1e-12):
        return np.arange(norb)
    L = np.diag(W.sum(axis=1)) - W
    _, evecs = np.linalg.eigh(L)
    return np.argsort(evecs[:, 1])


def permutation_to_unitary(perm) -> np.ndarray:
    """Permutation matrix U with U[p, perm[p]] = 1, usable as an initial_bases entry."""
    norb = len(perm)
    U = np.zeros((norb, norb))
    U[np.arange(norb), np.asarray(perm)] = 1.0
    return U


# =============================================================================
# Cost function adapters
#
# run_decomposition_optimizer always calls cost_function_constructor(rdm_data,
# cluster_matrix), where rdm_data holds tensors already rotated into the basis
# being scored. The existing cost-function builders in
# src.cluster_number_operators take differing, cost-specific argument lists
# (e.g. extremality_cost only needs D; number_commutator_cost_v1 needs h1e,
# g2e_full, and up to the 4-RDM); these factories adapt each of them to the
# uniform (rdm_data, cluster_matrix) -> Callable[[params], float] signature.
# =============================================================================


def make_variance_cost_constructor(var_exponent: int = 1) -> CostFunctionConstructor:
    def constructor(rdm_data: RDMData, cluster_matrix: np.ndarray):
        return number_variance_cost(
            rdm_data.D, rdm_data.Gamma, cluster_matrix, with_ghost=False, var_exponent=var_exponent
        )

    return constructor


def make_extremality_cost_constructor() -> CostFunctionConstructor:
    def constructor(rdm_data: RDMData, cluster_matrix: np.ndarray):
        return extremality_cost(rdm_data.D, cluster_matrix, with_ghost=False)

    return constructor


def make_eval_eq_cost_constructor() -> CostFunctionConstructor:
    def constructor(rdm_data: RDMData, cluster_matrix: np.ndarray):
        norb = rdm_data.norb
        clusters = get_cluster_indices(cluster_matrix, norb, with_ghost=False)
        n1_diag = np.diag(rdm_data.D).real
        evals = [round(float(n1_diag[c].sum())) for c in clusters]
        return number_eval_eq_cost(rdm_data.D, rdm_data.Gamma, cluster_matrix, evals, with_ghost=False)

    return constructor


def make_mixed_cost_constructor(c1: float = 1.0, c2: float = 0.5, var_exponent: int = 1) -> CostFunctionConstructor:
    def constructor(rdm_data: RDMData, cluster_matrix: np.ndarray):
        f1 = number_variance_cost(
            rdm_data.D, rdm_data.Gamma, cluster_matrix, with_ghost=False, var_exponent=var_exponent
        )
        f2 = extremality_cost(rdm_data.D, cluster_matrix, with_ghost=False)

        def f(x):
            return c1 * f1(x) + c2 * f2(x)

        return f

    return constructor


def make_commutator_cost_constructor(version: str = "v2") -> CostFunctionConstructor:
    impl = {"v1": number_commutator_cost_v1, "v2": number_commutator_cost_v2}[version]

    def constructor(rdm_data: RDMData, cluster_matrix: np.ndarray):
        if rdm_data.h1e is None or rdm_data.g2e_full is None or rdm_data.rdm3 is None or rdm_data.rdm4 is None:
            raise ValueError(
                "The commutator cost needs h1e, g2e_full, rdm3 and rdm4 to be populated on RDMData."
            )
        return impl(
            rdm_data.h1e, rdm_data.g2e_full, rdm_data.D, rdm_data.Gamma,
            rdm_data.rdm3, rdm_data.rdm4, cluster_matrix, with_ghost=False,
        )

    return constructor


# =============================================================================
# Restricted-rotation optimization core
# =============================================================================


@cache
def _pair_to_flat_index(norb: int) -> dict[tuple[int, int], int]:
    rows, cols = np.triu_indices(norb, k=1)
    return {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}


def _block_flat_indices(block: list[int], norb: int) -> np.ndarray:
    """Flat positions (in params_to_U_jax's triu_indices(norb, k=1) ordering)
    of the rotation-generator entries that mix pairs of orbitals within `block`."""
    pair_map = _pair_to_flat_index(norb)
    block_sorted = sorted(block)
    indices = [pair_map[(p, q)] for i, p in enumerate(block_sorted) for q in block_sorted[i + 1 :]]
    return np.array(indices, dtype=int)


def _optimize_restricted_rotation(
    f_full: Callable[[jnp.ndarray], float],
    norb: int,
    free_indices: np.ndarray | None,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    """Minimize f_full over the rotation parameters at `free_indices` only (all
    others held at 0, i.e. that part of the rotation stays the identity).
    free_indices=None frees all comb(norb, 2) parameters (a full joint
    reoptimization). Returns (x_opt_full, cost_opt); x_opt_full always has the
    full comb(norb, 2) length, zero-padded outside free_indices."""
    n_full = comb(norb, 2)
    if free_indices is None:
        free_indices = np.arange(n_full)
    free_indices = np.asarray(free_indices)
    free_indices_jax = jnp.array(free_indices)

    def g(x_free):
        x_full = jnp.zeros(n_full, dtype=jnp.float64).at[free_indices_jax].set(x_free)
        return f_full(x_full)

    g_val_and_grad = jax.jit(jax.value_and_grad(g))

    def scipy_g(x):
        val, grad = g_val_and_grad(x)
        return float(val), np.asarray(grad, dtype=np.float64)

    x0 = np.zeros(len(free_indices))
    result = scipy.optimize.minimize(
        fun=scipy_g, x0=x0, method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "disp": False},
    )

    x_opt_full = np.zeros(n_full)
    x_opt_full[free_indices] = result.x
    return x_opt_full, float(result.fun)


# =============================================================================
# Splitting / beam expansion
# =============================================================================


def _split_one_cluster(
    deco: Decomposition,
    rdm_data_cur: RDMData,
    n1: np.ndarray,
    n2: np.ndarray,
    cluster_idx: int,
    cost_function_constructor: CostFunctionConstructor,
    maxiter: int,
) -> Decomposition:
    """Split deco.partition[cluster_idx] via Fiedler bisection and re-optimize
    the rotation restricted to that cluster's orbitals, starting from deco.U."""
    norb = rdm_data_cur.norb
    V = deco.partition[cluster_idx]
    V1, V2 = fiedler_bipartition(n1, n2, V)

    child_partition = deco.partition[:cluster_idx] + deco.partition[cluster_idx + 1 :] + [V1, V2]
    child_cluster_matrix = partition_to_cluster_matrix(child_partition, norb)

    f_full = cost_function_constructor(rdm_data_cur, child_cluster_matrix)
    free_indices = _block_flat_indices(V, norb)
    x_opt_full, cost_opt = _optimize_restricted_rotation(f_full, norb, free_indices, maxiter)

    U_block = np.asarray(params_to_U_jax(jnp.array(x_opt_full), norb))
    U_child = U_block @ deco.U

    return Decomposition(partition=child_partition, U=U_child, cost=cost_opt)


def expand_decomposition(
    deco: Decomposition,
    rdm_data_ref: RDMData,
    cost_function_constructor: CostFunctionConstructor,
    num_subdecos: int,
    min_cluster_size: int,
    maxiter: int,
) -> list[Decomposition]:
    """Produce up to `num_subdecos` child decompositions of `deco`: pick the
    (up to num_subdecos) clusters with the largest current number-variance
    (clusters at or below min_cluster_size are never chosen), and split each
    independently via _split_one_cluster."""
    rdm_data_cur = rotate_rdm_data(rdm_data_ref, deco.U)
    n1, n2 = cluster_number_stats(rdm_data_cur, deco.partition)

    splittable = [i for i, c in enumerate(deco.partition) if len(c) > min_cluster_size]
    if not splittable:
        return []

    splittable.sort(key=lambda i: -cluster_variance(n1, n2, deco.partition[i]))
    chosen = splittable[:num_subdecos]

    return [
        _split_one_cluster(deco, rdm_data_cur, n1, n2, cluster_idx, cost_function_constructor, maxiter)
        for cluster_idx in chosen
    ]


# =============================================================================
# Main beam search
# =============================================================================


def _initial_beam(
    rdm_data_ref: RDMData,
    cost_function_constructor: CostFunctionConstructor,
    num_decos: int,
    initial_bases: list[np.ndarray] | None,
) -> list[Decomposition]:
    norb = rdm_data_ref.norb
    if initial_bases is None:
        initial_bases = [np.eye(norb)]

    beam = []
    for i in range(num_decos):
        U0 = np.asarray(initial_bases[i % len(initial_bases)])
        partition0 = [list(range(norb))]
        cluster_matrix0 = partition_to_cluster_matrix(partition0, norb)
        rdm_data_0 = rotate_rdm_data(rdm_data_ref, U0)
        f0 = cost_function_constructor(rdm_data_0, cluster_matrix0)
        cost0 = float(f0(jnp.zeros(comb(norb, 2))))
        beam.append(Decomposition(partition=partition0, U=U0, cost=cost0))
    return beam


def run_decomposition_optimizer(
    cost_function_constructor: CostFunctionConstructor,
    rdm_data: RDMData,
    config: DecompositionOptimizerConfig | None = None,
    initial_bases: list[np.ndarray] | None = None,
) -> list[Decomposition]:
    """Jointly optimize the cluster partition and orbital basis by beam search
    (see module docstring for the full algorithm). Returns the trajectory of
    best decompositions, one per cluster count: trajectory[i] is the best
    decomposition found with i + 1 clusters. norb is inferred from rdm_data.D.
    """
    if config is None:
        config = DecompositionOptimizerConfig()

    norb = rdm_data.norb
    beam = _initial_beam(rdm_data, cost_function_constructor, config.num_decos, initial_bases)
    for deco in beam:
        validate_partition(deco.partition, norb)

    trajectory = [min(beam, key=lambda d: d.cost)]
    round_idx = 0

    while True:
        if config.target_num_clusters is not None and beam[0].num_clusters >= config.target_num_clusters:
            break
        if all(len(c) <= config.min_cluster_size for deco in beam for c in deco.partition):
            break
        if config.max_rounds is not None and round_idx >= config.max_rounds:
            break

        candidates: list[Decomposition] = []
        for deco in beam:
            candidates.extend(
                expand_decomposition(
                    deco, rdm_data, cost_function_constructor,
                    config.num_subdecos, config.min_cluster_size, config.maxiter,
                )
            )
        if not candidates:
            logger.info("Round %d: no deco could be split further, stopping.", round_idx)
            break

        for cand in candidates:
            validate_partition(cand.partition, norb)

        candidates.sort(key=lambda d: d.cost)
        beam = candidates[: config.num_decos]
        trajectory.append(beam[0])

        logger.info(
            "Round %d: %d candidates from %d decos -> kept %d, best cost=%.6e "
            "(%d clusters, sizes=%s)",
            round_idx, len(candidates), len(beam), len(beam), beam[0].cost,
            beam[0].num_clusters, [len(c) for c in beam[0].partition],
        )
        round_idx += 1

    return trajectory


def polish_decomposition(
    deco: Decomposition,
    rdm_data_ref: RDMData,
    cost_function_constructor: CostFunctionConstructor,
    maxiter: int = 500,
) -> Decomposition:
    """Final full joint reoptimization of the orbital basis for a FIXED
    partition (deco.partition), starting from deco.U. Unlike the incremental,
    per-split block-restricted updates used during the beam search, this frees
    the whole norb x norb rotation at once, which can polish away suboptimality
    left over from the greedy/hierarchical splitting process."""
    norb = rdm_data_ref.norb
    rdm_data_cur = rotate_rdm_data(rdm_data_ref, deco.U)
    cluster_matrix = partition_to_cluster_matrix(deco.partition, norb)
    f_full = cost_function_constructor(rdm_data_cur, cluster_matrix)

    x_opt_full, cost_opt = _optimize_restricted_rotation(f_full, norb, None, maxiter)
    U_new = np.asarray(params_to_U_jax(jnp.array(x_opt_full), norb)) @ deco.U

    return Decomposition(partition=deco.partition, U=U_new, cost=cost_opt)


def best_decomposition(trajectory: list[Decomposition], num_clusters: int | None = None) -> Decomposition:
    """Convenience accessor: trajectory[-1] (finest decomposition reached) by
    default, or the trajectory entry with exactly `num_clusters` clusters."""
    if num_clusters is None:
        return trajectory[-1]
    for deco in trajectory:
        if deco.num_clusters == num_clusters:
            return deco
    raise ValueError(
        f"No decomposition with {num_clusters} clusters in trajectory "
        f"(have {[d.num_clusters for d in trajectory]})."
    )


# =============================================================================
# CLI Interface
# =============================================================================


def _build_cost_function_constructor(name: str, var_exponent: int) -> CostFunctionConstructor:
    if name == "variance":
        return make_variance_cost_constructor(var_exponent=var_exponent)
    if name == "eval_eq":
        return make_eval_eq_cost_constructor()
    if name == "extremality":
        return make_extremality_cost_constructor()
    if name == "mixed":
        return make_mixed_cost_constructor(var_exponent=var_exponent)
    if name == "commutator":
        return make_commutator_cost_constructor()
    raise ValueError(f"Unsupported cost function: {name}")


def _build_initial_bases(args: argparse.Namespace, rdm_data: RDMData, norb: int) -> list[np.ndarray]:
    """MOs and/or NatOs (per --initial-basis), optionally each reordered by
    the Fiedler vector of its own correlation graph (per --fiedler-reorder)."""
    bases = []
    if args.initial_basis in ("MOs", "both"):
        bases.append(np.eye(norb))
    if args.initial_basis in ("NatOs", "both"):
        _, evecs = np.linalg.eigh(rdm_data.D)
        bases.append(evecs.T)

    if args.fiedler_reorder:
        reordered = []
        for U0 in bases:
            rdm_data_U0 = rotate_rdm_data(rdm_data, U0)
            perm = fiedler_orbital_order(rdm_data_U0.D, rdm_data_U0.Gamma)
            reordered.append(permutation_to_unitary(perm) @ U0)
        bases = reordered

    return bases


def _run_dmrg_and_build_rdm_data(
    args: argparse.Namespace,
) -> tuple[RDMData, dict, Any, float, np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Reuses cluster_numbers_metrics.py's DMRG + RDM-extraction pipeline to
    turn CLI molecule/basis/geometry arguments into an RDMData bundle, plus
    everything (solver, integrals) later needed to rerun the sector analysis.
    Imported lazily so that using this module as a library (with RDMs already
    in hand) doesn't pull in the DMRG solver / plotting stack.

    Returns: rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec
    """
    import pyscf.ao2mo
    from cluster_numbers_metrics import MetricsConfig, compute_dmrg, extract_rdms

    metrics_config = MetricsConfig(
        molecule=args.molecule,
        basis_set=args.basis_set,
        bond_length=args.bond_length,
        type_cost_function=args.cost_function,  # reuses MetricsConfig's own validation
        bond_angle=args.bond_angle,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        wavefunction_dir=args.wavefunction_dir,
        n_threads=args.n_threads,
        reuse_wavefunction=not args.no_reuse,
    )

    logger.info("Running DMRG to get ground state MPS and integrals...")
    solver, dmrg_energy, h1e, g2e, ecore, nelec, result = compute_dmrg(metrics_config)
    norb = solver.n_sites

    if args.cost_function == "commutator":
        D, Gamma, rdm3, rdm4 = extract_rdms(solver, result.mps_tag, with_34_rdms=True)
        g2e_full = pyscf.ao2mo.restore(1, g2e, norb)
        rdm_data = RDMData(D=D, Gamma=Gamma, rdm3=rdm3, rdm4=rdm4, h1e=h1e, g2e_full=g2e_full)
    else:
        D, Gamma = extract_rdms(solver, result.mps_tag, with_34_rdms=False)
        rdm_data = RDMData(D=D, Gamma=Gamma)

    dmrg_metadata = {"nelec": [int(x) for x in nelec], "dmrg_energy": float(dmrg_energy)}
    return rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec


# -----------------------------------------------------------------------------
# K-sectors / K-states sector analysis for the winning decompositions
#
# This section is the per-trajectory-entry analog of
# cluster_numbers_metrics.py's compute_sector_analysis + generate_plots: it
# reuses SingleMaxelectransferResult, BasisResult, MetricsOutput,
# get_ordered_state_projections_in_sectors, get_ordered_decoupled_states,
# save_metrics and generate_plots from that module UNCHANGED, and
# get_K_sectors_values_energies /
# get_K_states_values_energies / plot_energy_vs_K / plot_dual_bar_chart from
# src.K_sectors_plots UNCHANGED. The only real difference from
# compute_sector_analysis is that there ONE cluster_matrix is shared across
# several orbital bases U (so `sectors` is computed once); here every
# trajectory entry has its OWN partition, so `sectors` is rebuilt per entry.
# -----------------------------------------------------------------------------


def _build_analysis_entries(
    trajectory: list[Decomposition],
    polished_trajectory: list[Decomposition] | None,
    args: argparse.Namespace,
) -> list[tuple[str, Decomposition]]:
    """Which trajectory entries to run sector analysis on, and their curve
    labels (analogous to cluster_numbers_metrics.py's data_label). Defaults to
    every entry with >= 2 clusters: the 1-cluster decomposition carries no
    symmetry information (every determinant falls into the same, full-Hilbert
    -space sector) and is the most expensive case for K-states mode to
    diagonalize, so it's excluded unless explicitly requested."""
    final_trajectory = polished_trajectory if polished_trajectory is not None else trajectory

    if args.analyze_num_clusters is not None:
        wanted = set(args.analyze_num_clusters)
        selected = [d for d in final_trajectory if d.num_clusters in wanted]
        missing = wanted - {d.num_clusters for d in selected}
        if missing:
            logger.warning(f"--analyze-num-clusters {sorted(missing)} not present in trajectory; skipping.")
    else:
        selected = [d for d in final_trajectory if d.num_clusters >= 2]

    label_suffix = " (polished)" if polished_trajectory is not None else ""
    return [(f"{d.num_clusters} clusters{label_suffix}", d) for d in selected]


def _compute_sector_analysis_for_trajectory(
    entries: list[tuple[str, Decomposition]],
    solver: Any,
    dmrg_energy: float,
    h1e: np.ndarray,
    g2e: np.ndarray,
    ecore: float,
    nelec: tuple[int, int],
    norb: int,
    max_elec_transfers: list[int],
    max_K_sectors: int,
    max_K_states: int,
    skip_K_sectors: bool,
    skip_K_states: bool,
) -> list:
    """Returns a list[BasisResult] (cluster_numbers_metrics.py's own
    dataclass), one entry per (data_label, deco) in `entries`, each covering
    every value in `max_elec_transfers` -- exactly the shape
    compute_sector_analysis produces, ready for save_metrics/generate_plots."""
    import ffsim
    import pyscf.ao2mo
    from cluster_numbers_metrics import (
        BasisResult,
        SingleMaxelectransferResult,
        get_ordered_decoupled_states,
        get_ordered_state_projections_in_sectors,
    )
    from src.K_sectors_plots import get_K_sectors_values_energies, get_K_states_values_energies

    mps = solver.get_mps()
    psi_MOs = solver.to_ci_vector(ket=mps)
    g2e_full = pyscf.ao2mo.restore(1, g2e, norb)
    hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor=h1e, two_body_tensor=g2e_full, constant=ecore)

    basis_results = []
    for data_label, deco in entries:
        max_cluster_size = max(len(c) for c in deco.partition)
        if not skip_K_states and max_cluster_size > 10:
            logger.warning(
                f"{data_label}: largest cluster has {max_cluster_size} orbitals -- "
                "diagonalizing its symmetry sector (K-states mode) may be very slow or "
                "infeasible. Consider --skip-K-states, a narrower --analyze-num-clusters, "
                "or a larger --min-cluster-size in the search."
            )

        cluster_matrix = partition_to_cluster_matrix(deco.partition, norb)
        # every trajectory partition already fully covers range(norb) (see
        # validate_partition), so there is never a ghost cluster to add here,
        # unlike compute_sector_analysis
        sectors = number_and_parity_symmetry_sectors(cluster_matrix, [], norb, nelec)

        psi = ffsim.apply_orbital_rotation(psi_MOs, np.asarray(deco.U), norb, nelec)
        h_rotated = hamiltonian.rotated(np.asarray(deco.U))
        h_linop = ffsim.linear_operator(h_rotated, norb, nelec)

        # sanity check, as in compute_sector_analysis
        assert np.isclose(dmrg_energy, np.vdot(psi, h_linop @ psi).real, atol=1e-8)
        assert np.isclose(0, np.vdot(psi, h_linop @ psi).imag, atol=1e-8)

        logger.info(f"Analyzing decomposition: {data_label} ({len(sectors)} symmetry sectors)")

        # Objects independent of max_elec_transfers, computed once per entry and
        # reused across the whole max_elec loop below (same pattern as the
        # current compute_sector_analysis in cluster_numbers_metrics.py).
        ordered_state_projections_in_sectors = None
        if not skip_K_sectors:
            ordered_state_projections_in_sectors = get_ordered_state_projections_in_sectors(sectors, h_linop, psi)

        ordered_decoupled_states = None
        if not skip_K_states:
            ordered_decoupled_states = get_ordered_decoupled_states(sectors, h_linop, psi)

        transfer_results = []
        for max_elec in max_elec_transfers:
            transfer_result = SingleMaxelectransferResult(
                max_elec_transfers=max_elec,
                K_sectors_values=[], K_sectors_energies=[], K_sectors_retained_dimensions=[],
                num_retained_sectors=0, retained_dim=None, chem_accuracy_reached_sectors=False,
                K_states_values=[], K_states_energies=[], K_states_num_retained_state_sectors=[],
                num_retained_states=0, num_retained_state_sectors=0, chem_accuracy_reached_states=False,
            )

            if not skip_K_sectors:
                (K_sectors_values, K_sectors_energies, K_sectors_retained_dimensions,
                 chem_accuracy_reached_sectors) = get_K_sectors_values_energies(
                    psi, h_linop, dmrg_energy, sectors, ordered_state_projections_in_sectors, max_elec, "projected",
                    max_K_sectors=max_K_sectors, verbose=0,
                )
                transfer_result.K_sectors_values = K_sectors_values
                transfer_result.K_sectors_energies = K_sectors_energies
                transfer_result.K_sectors_retained_dimensions = K_sectors_retained_dimensions
                transfer_result.num_retained_sectors = len(K_sectors_values)
                transfer_result.retained_dim = (
                    K_sectors_retained_dimensions[-1] if K_sectors_retained_dimensions else None
                )
                transfer_result.chem_accuracy_reached_sectors = chem_accuracy_reached_sectors

            if not skip_K_states:
                (K_states_values, K_states_energies, K_states_num_retained_state_sectors,
                 chem_accuracy_reached_states, _retained_sector_labels) = get_K_states_values_energies(
                    psi, h_linop, dmrg_energy, ordered_decoupled_states, max_elec, "projected",
                    max_K_states=max_K_states, verbose=0,
                )
                transfer_result.K_states_values = K_states_values
                transfer_result.K_states_energies = K_states_energies
                transfer_result.K_states_num_retained_state_sectors = K_states_num_retained_state_sectors
                transfer_result.num_retained_states = len(K_states_values)
                transfer_result.num_retained_state_sectors = (
                    K_states_num_retained_state_sectors[-1] if K_states_num_retained_state_sectors else 0
                )
                transfer_result.chem_accuracy_reached_states = chem_accuracy_reached_states

            transfer_results.append(transfer_result)

        basis_results.append(BasisResult(data_label=data_label, transfer_results=transfer_results))

    return basis_results


def _max_transfers_dirs(args: argparse.Namespace) -> dict[int, tuple[Path, Path]]:
    """Per max_elec_transfers output/plots directories for the sector
    analysis, nested under this run's own output/plots directory -- mirrors
    cluster_numbers_metrics.create_output_dirs's max_transfers_{N} convention."""
    base_output_dir = _output_dir_for(args)
    base_plots_dir = _plots_dir_for(args)
    dirs = {}
    for max_elec in args.max_transfers:
        output_dir = base_output_dir / f"max_transfers_{max_elec}"
        plots_dir = base_plots_dir / f"max_transfers_{max_elec}"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        dirs[max_elec] = (output_dir, plots_dir)
    return dirs


def _run_sector_analysis(
    args: argparse.Namespace,
    trajectory: list[Decomposition],
    polished_trajectory: list[Decomposition] | None,
    solver: Any,
    dmrg_energy: float,
    h1e: np.ndarray,
    g2e: np.ndarray,
    ecore: float,
    nelec: tuple[int, int],
    norb: int,
    computation_time: float,
) -> None:
    """For the selected "winner" decompositions, reruns
    cluster_numbers_metrics.py's own K-sectors/K-states sector analysis and
    saves/plots it with that module's OWN save_metrics/generate_plots -- same
    JSON shape, same four PNGs per max_elec_transfers value -- just with each
    trajectory entry (instead of each orbital basis) as one curve."""
    from cluster_numbers_metrics import MetricsOutput, generate_plots, get_git_hash, get_timestamp, save_metrics

    entries = _build_analysis_entries(trajectory, polished_trajectory, args)
    if not entries:
        logger.info("No trajectory entries selected for sector analysis (see --analyze-num-clusters).")
        return

    logger.info(f"Running K-sectors/K-states sector analysis for: {[label for label, _ in entries]}")
    basis_results = _compute_sector_analysis_for_trajectory(
        entries, solver, dmrg_energy, h1e, g2e, ecore, nelec, norb,
        max_elec_transfers=args.max_transfers,
        max_K_sectors=args.max_K_sectors,
        max_K_states=args.max_K_states,
        skip_K_sectors=args.skip_K_sectors,
        skip_K_states=args.skip_K_states,
    )

    metadata = {
        "molecule": args.molecule,
        "basis_set": args.basis_set,
        "bond_length": args.bond_length,
        "bond_angle": args.bond_angle,
        "max_elec_transfers": args.max_transfers,
        "timestamp": get_timestamp(),
        "git_hash": get_git_hash(),
        "norb": norb,
        "nelec": [int(x) for x in nelec],
        "dmrg_energy": float(dmrg_energy),
        "cost": args.cost_function,
        # unlike compute_cluster_number_metrics, cluster_matrix/cluster_sizes
        # aren't a single value here -- each curve has its own partition
        "cluster_sizes": "varies per curve -- see data_label",
        "analyzed_num_clusters": [deco.num_clusters for _, deco in entries],
        "computation_time_seconds": computation_time,
    }
    output = MetricsOutput(metadata=metadata, basis_results=basis_results)

    for max_elec, (output_dir, plots_dir) in _max_transfers_dirs(args).items():
        filepath = save_metrics(output, output_dir, max_elec=max_elec)
        logger.info(f"Sector analysis results saved to {filepath}")
        if not args.no_plots:
            generate_plots(
                output, plots_dir, max_elec=max_elec, show=args.show_plots, save=True,
                skip_K_sectors=args.skip_K_sectors, skip_K_states=args.skip_K_states,
            )


def _geometry_output_subpath(args: argparse.Namespace) -> Path:
    molecule = args.molecule.lower()
    basis_set = args.basis_set.lower()
    bond_length_str = f"{args.bond_length:.4f}".replace(".", "_")
    path = Path(molecule) / basis_set / f"bond_{bond_length_str}"
    if args.bond_angle is not None:
        angle_str = f"{args.bond_angle:.4f}".replace(".", "_")
        path = path / f"angle_{angle_str}"
    return path


def _output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs_") / "cluster_number_decomposition" / _geometry_output_subpath(args)


def _plots_dir_for(args: argparse.Namespace) -> Path:
    if args.plots_dir is not None:
        return Path(args.plots_dir)
    return Path("plots") / "cluster_number_decomposition" / _geometry_output_subpath(args)


def _trajectory_to_json(
    trajectory: list[Decomposition], polished_trajectory: list[Decomposition] | None
) -> list[dict]:
    entries = []
    for i, deco in enumerate(trajectory):
        entry = {
            "num_clusters": deco.num_clusters,
            "cluster_sizes": [len(c) for c in deco.partition],
            "partition": deco.partition,
            "cost": deco.cost,
            "U": deco.U.tolist(),
        }
        if polished_trajectory is not None:
            entry["polished_cost"] = polished_trajectory[i].cost
            entry["polished_U"] = polished_trajectory[i].U.tolist()
        entries.append(entry)
    return entries


def _plot_trajectory(
    trajectory: list[Decomposition],
    polished_trajectory: list[Decomposition] | None,
    metadata: dict,
    plots_dir: Path,
    show: bool = False,
    save: bool = True,
) -> None:
    import matplotlib

    if not save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [d.num_clusters for d in trajectory]
    ys = [d.cost for d in trajectory]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", label="beam search")
    if polished_trajectory is not None:
        ax.plot(xs, [d.cost for d in polished_trajectory], "s--", label="polished")
        ax.legend()
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(f"Cost ({metadata['cost']})")
    ax.set_yscale("log")
    ax.set_title(
        f"{metadata['molecule']} / {metadata['basis_set']}, norb={metadata['norb']}, cost={metadata['cost']}"
    )
    fig.tight_layout()

    if save:
        filename = f"cost_vs_clusters_{metadata['cost']}_{metadata['timestamp']}.png"
        filepath = plots_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Trajectory plot saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Jointly optimize the cluster partition and orbital basis of a quasisymmetry decomposition"
    )

    # Required arguments (same molecule/basis_set/bond_length/cost_function
    # convention as cluster_numbers_metrics.py)
    parser.add_argument(
        "molecule", type=str,
        choices=["h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"],
        help="Molecule to analyze",
    )
    parser.add_argument("basis_set", type=str, help="Basis set (e.g., sto-3g, 6-31g)")
    parser.add_argument("bond_length", type=float, help="Bond length in Angstrom")
    parser.add_argument(
        "cost_function", type=str,
        help="Cost function type (variance, eval_eq, extremality, mixed, commutator)",
    )

    parser.add_argument("--bond-angle", type=float, default=None, help="Bond angle in degrees (for H2O)")

    # DMRG parameters
    parser.add_argument("--bond-dim", type=int, default=150, help="DMRG bond dimension (default: 150)")
    parser.add_argument("--n-sweeps", type=int, default=50, help="Number of DMRG sweeps (default: 50)")

    # Cost function parameters
    parser.add_argument("--var-exponent", type=int, default=1, help="Variance exponent (default: 1)")

    # Beam search hyperparameters
    parser.add_argument("--num-decos", type=int, default=4, help="Beam width (default: 4)")
    parser.add_argument(
        "--num-subdecos", type=int, default=4, help="Splits attempted per deco per round (default: 4)"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=1,
        help="Clusters at or below this size are never split (default: 1)",
    )
    parser.add_argument(
        "--target-num-clusters", type=int, default=None,
        help="Stop once this many clusters is reached (default: run down to singleton clusters)",
    )
    parser.add_argument("--max-rounds", type=int, default=None, help="Extra safety cap on rounds (default: none)")
    parser.add_argument(
        "--maxiter", type=int, default=200, help="L-BFGS-B iterations per split, block-restricted (default: 200)"
    )

    # Initial basis
    parser.add_argument(
        "--initial-basis", type=str, choices=["MOs", "NatOs", "both"], default="MOs",
        help="Reference basis (or bases) the beam is seeded from (default: MOs)",
    )
    parser.add_argument(
        "--fiedler-reorder", action="store_true",
        help="Reorder each initial basis by the Fiedler vector of its own correlation graph",
    )

    # Polish
    parser.add_argument(
        "--no-polish", action="store_true",
        help="Skip the full joint-rotation polish step on each trajectory entry",
    )
    parser.add_argument(
        "--polish-maxiter", type=int, default=500, help="L-BFGS-B iterations for polishing (default: 500)"
    )

    # Sector analysis (K-sectors / K-states) for selected trajectory entries,
    # analogous to cluster_numbers_metrics.py's own sector analysis
    parser.add_argument(
        "--max-transfers", type=int, nargs="+", default=[1, 2, 3],
        help="Maximum electron transfers for the sector analysis (default: 1 2 3)",
    )
    parser.add_argument(
        "--analyze-num-clusters", type=int, nargs="+", default=None,
        help="Which trajectory cluster counts to run sector analysis on (default: all with >= 2 clusters)",
    )
    parser.add_argument(
        "--skip-sector-analysis", action="store_true",
        help="Skip the K-sectors/K-states sector analysis entirely (only produce the trajectory JSON/plot)",
    )
    parser.add_argument("--skip-K-sectors", action="store_true", help="Disable sector-based (K-sectors) analysis")
    parser.add_argument(
        "--skip-K-states", action="store_true", help="Disable decoupled state-based (K-states) analysis"
    )
    parser.add_argument("--max-K-sectors", type=int, default=50, help="Maximum number of sectors (default: 50)")
    parser.add_argument(
        "--max-K-states", type=int, default=50, help="Maximum number of decoupled states (default: 50)"
    )

    # Output options
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--plots-dir", type=str, default=None, help="Plots directory")
    parser.add_argument(
        "--wavefunction-dir", type=str, default="wavefunctions",
        help="MPS wavefunction directory (for input and output)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")

    # HPC options
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads (default: 1)")
    parser.add_argument("--no-reuse", action="store_true", help="Don't reuse existing wavefunction")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"Starting joint decomposition optimization for {args.molecule} in {args.basis_set} basis set")
    logger.info(
        f"cost function={args.cost_function}, num_decos={args.num_decos}, "
        f"num_subdecos={args.num_subdecos}, min_cluster_size={args.min_cluster_size}"
    )
    if args.bond_angle is not None:
        logger.info(f"Bond angle={args.bond_angle}")

    start_time = time.time()

    try:
        rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(args)
        norb = rdm_data.norb

        cost_function_constructor = _build_cost_function_constructor(args.cost_function, args.var_exponent)
        initial_bases = _build_initial_bases(args, rdm_data, norb)

        opt_config = DecompositionOptimizerConfig(
            num_decos=args.num_decos,
            num_subdecos=args.num_subdecos,
            min_cluster_size=args.min_cluster_size,
            target_num_clusters=args.target_num_clusters,
            max_rounds=args.max_rounds,
            maxiter=args.maxiter,
        )

        logger.info("Running beam search over cluster partitions and orbital rotations...")
        trajectory = run_decomposition_optimizer(cost_function_constructor, rdm_data, opt_config, initial_bases)

        polished_trajectory = None
        if not args.no_polish:
            logger.info("Polishing each trajectory entry with a full joint-rotation reoptimization...")
            polished_trajectory = [
                polish_decomposition(deco, rdm_data, cost_function_constructor, maxiter=args.polish_maxiter)
                for deco in trajectory
            ]

        computation_time = time.time() - start_time

        from cluster_numbers_metrics import get_git_hash, get_timestamp

        metadata = {
            "molecule": args.molecule,
            "basis_set": args.basis_set,
            "bond_length": args.bond_length,
            "bond_angle": args.bond_angle,
            "cost": args.cost_function,
            "timestamp": get_timestamp(),
            "git_hash": get_git_hash(),
            "norb": norb,
            "num_decos": args.num_decos,
            "num_subdecos": args.num_subdecos,
            "min_cluster_size": args.min_cluster_size,
            "target_num_clusters": args.target_num_clusters,
            "initial_basis": args.initial_basis,
            "fiedler_reorder": args.fiedler_reorder,
            "polished": not args.no_polish,
            "computation_time_seconds": computation_time,
            **dmrg_metadata,
        }

        output_dir = _output_dir_for(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"decomposition_{args.cost_function}_{metadata['timestamp']}_{metadata['git_hash']}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(
                {"metadata": metadata, "trajectory": _trajectory_to_json(trajectory, polished_trajectory)},
                f, indent=2,
            )
        logger.info(f"Results saved to {filepath}")

        if not args.no_plots:
            plots_dir = _plots_dir_for(args)
            plots_dir.mkdir(parents=True, exist_ok=True)
            _plot_trajectory(trajectory, polished_trajectory, metadata, plots_dir, show=args.show_plots, save=True)

        logger.info("Computation completed successfully!")
        for i, deco in enumerate(trajectory):
            cost_str = f"{deco.cost:.6e}"
            if polished_trajectory is not None:
                cost_str += f" (polished: {polished_trajectory[i].cost:.6e})"
            logger.info(f"  {deco.num_clusters} clusters, sizes={[len(c) for c in deco.partition]}: cost={cost_str}")

        if not args.skip_sector_analysis:
            _run_sector_analysis(
                args, trajectory, polished_trajectory, solver, dmrg_energy, h1e, g2e, ecore, nelec, norb,
                computation_time,
            )

    except Exception as e:
        logger.error(f"Computation failed: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
