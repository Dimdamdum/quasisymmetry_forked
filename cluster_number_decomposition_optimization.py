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
     orbitals, are left untouched) against `cost_function_constructor`, then
     (by default) diagonalize each child's own 1-RDM block to that child's own
     natural orbitals -- exactly cost-neutral (see `--no-naturalize-children`).
  3. Keep the best `num_decos` children (by cost) as the new beam.
  4. Repeat until every deco in the beam is down to singleton clusters (or
     `min_parent_cluster_size`), or `target_num_clusters` is reached.

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

Other important empirical fact (again, valid for variance cost): at each round,
raw cost prefers splitting the biggest cluster C into (|C| - 1) + 1 subclusters.
But again, this preference may not always correspond to low values of downstream metrics.
Therefore, the minimal cluster size and target number of clusters are probably two
important hyperparameters (CLI options --min-parent-cluster-size and --target-num-clusters).
Note --min-parent-cluster-size only controls which clusters remain eligible to be split
further -- it does NOT prevent a single split from producing a small subcluster
(a cluster larger than --min-parent-cluster-size can still be split like (|C|-1)+1). To
directly rule out that pattern (or any subcluster below a given size), use
--min-child-cluster-size instead: it constrains the split itself, so neither
resulting subcluster can ever be smaller than the given value.

CLI usage (mirrors cluster_numbers_metrics.py -- runs DMRG via that script's
own compute_dmrg/extract_rdms, then runs the beam search on the resulting
RDMs):
    python cluster_number_decomposition_optimization.py h2o sto-3g 2.0 variance --bond-angle 104.5
    python cluster_number_decomposition_optimization.py h4_square 6-31g 1.0 commutator --num-decos 6 --num-subdecos 8 --fiedler-reorder

Run with --help for the full list of options. Most physically meaningful ones are probably
--min-parent-cluster-size (default: 1)
--min-child-cluster-size (default: 1)
--no-naturalize-children (default: False, i.e. naturalization ON)
--no-orb-opt-in-beam-search (default: False, i.e. per-split rotation optimization ON)
--target-num-clusters (default: run down to singleton clusters)
--initial-basis (default: MOs)
--fiedler-reorder (on initial basis - default: False)
--max-transfers (default: 1 2 3)
Main hyperparameters are probably
--bond-dim (default: 150)
--n-sweeps (default: 50)
--num-decos (default: 4)
--num-subdecos (default: 4)
and possibly the orbital optimization max iterations
--maxiter (default: 200)
--polish-maxiter (default: 500)

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

    config = DecompositionOptimizerConfig(num_decos=4, num_subdecos=6, min_parent_cluster_size=2)
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
# Data structures (human verified ✓)
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

# Defining a useful type alias
CostFunctionConstructor = Callable[[RDMData, np.ndarray], Callable[[np.ndarray], float]]


@dataclass
class DecompositionOptimizerConfig:
    """Hyperparameters for run_decomposition_optimizer."""

    num_decos: int = 4  # beam width
    num_subdecos: int = 4  # splits attempted per deco per round
    min_parent_cluster_size: int = 1  # clusters at or below this size are never split
    min_child_cluster_size: int = 1  # neither child of a split may end up smaller than this
    naturalize_children: bool = True  # diagonalize each split's two children's own 1-RDM blocks
    target_num_clusters: int | None = None  # stop once the beam reaches this many clusters
    max_rounds: int | None = None  # extra safety cap on rounds
    maxiter: int = 200  # L-BFGS-B iterations for each per-split local optimization
    optimize_rotation_in_beam_search: bool = True  # continuously re-optimize each split's rotation block (off: leave it at the identity; Fiedler bisection still happens)

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
        if self.min_parent_cluster_size < 1:
            raise ValueError("min_parent_cluster_size must be >= 1")
        if self.min_child_cluster_size < 1:
            raise ValueError("min_child_cluster_size must be >= 1")


# =============================================================================
# Partition <-> cluster_matrix helpers (human verified ✓)
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
# RDM / integral rotation (human verified ✓)
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
# Cluster statistics and Fiedler splitting (human verified ✓)
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


def fiedler_bipartition(
    n1: np.ndarray, n2: np.ndarray, cluster: list[int], min_child_size: int = 1
) -> tuple[list[int], list[int]]:
    """Split `cluster` into two groups by spectral (Fiedler) bisection of the
    pairwise number-operator covariance graph |Cov(n_p, n_q)|: orbitals whose
    occupations fluctuate together are kept on the same side. This is a cheap
    proxy for "the split that will end up costing the least", used to seed the
    subsequent continuous optimization over Var(N_V1) + Var(N_V2).

    min_child_size: neither returned group will have fewer than this many
    orbitals -- the natural spectral cut point is clamped towards the middle
    just enough to satisfy this (a no-op whenever the natural cut already
    does, e.g. always at the default min_child_size=1). Raises ValueError if
    `cluster` is too small for any split to satisfy this
    (len(cluster) < 2 * min_child_size).
    
    Returns a couple of lists of integers that bipartition the initial cluster."""
    cluster = list(cluster)
    k = len(cluster)
    if k < 2:
        raise ValueError("Cannot split a cluster of size < 2.")
    if k < 2 * min_child_size:
        raise ValueError(
            f"Cannot split a cluster of size {k} into two groups each >= "
            f"min_child_size={min_child_size} (need size >= {2 * min_child_size})."
        )
    if k == 2:
        return [cluster[0]], [cluster[1]]

    idx = np.asarray(cluster)
    n1_c = n1[idx]
    n2_c = n2[np.ix_(idx, idx)]
    W = np.abs(n2_c - np.outer(n1_c, n1_c))
    np.fill_diagonal(W, 0.0)

    if not np.any(W > 1e-12):
        # No resolvable correlation structure: fall back to a balanced index split.
        order = np.arange(k)
        cut = k // 2
    else:
        L = np.diag(W.sum(axis=1)) - W
        _, evecs = np.linalg.eigh(L)
        fiedler_vec = evecs[:, 1]  # eigenvector of the second-smallest eigenvalue
        order = np.argsort(fiedler_vec)

        side = fiedler_vec > 0
        if side.all() or not side.any():
            # Degenerate/near-zero Fiedler component: fall back to a median split.
            cut = k // 2
        else:
            cut = k - int(side.sum())  # size of the non-positive side, in `order`

    # Clamp the cut so both sides respect min_child_size (a no-op whenever the
    # natural cut already does).
    cut = min(max(cut, min_child_size), k - min_child_size)

    cluster_sorted = idx[order]
    V1 = cluster_sorted[cut:].tolist()
    V2 = cluster_sorted[:cut].tolist()
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
# Cost function adapters (human verified ✓)
#
# run_decomposition_optimizer always calls cost_function_constructor(rdm_data,
# cluster_matrix), where rdm_data holds tensors already rotated into the basis
# being scored. The existing cost-function builders in
# src.cluster_number_operators take differing, cost-specific argument lists
# (e.g. extremality_cost only needs D; number_commutator_cost_v1 needs h1e,
# g2e_full, and up to the 4-RDM); the factories below adapt each of them to the
# uniform (rdm_data, cluster_matrix) -> Callable[[params], float] signature.
# =============================================================================

# Tldr: The following are functions that construct functions that construct functions,
# i.e., constructors of function constructors; they uniformize conventions of
# existing function constructors, so they can all be called in the same way

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
# Restricted-rotation optimization core (human verified ✓)
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
    others held at 0, i.e. that part of the rotation stays the identity;
    take this into account when building the input).
    free_indices is typically obtained with _block_flat_indices.
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
# Splitting / beam expansion (human verified ✓)
# =============================================================================


def _local_natural_orbital_rotation(
    U_step: np.ndarray, D_cur: np.ndarray, clusters: list[list[int]], norb: int
) -> np.ndarray:
    """Rotation R (identity outside `clusters`' orbitals) that, composed on top of U_step
    (i.e. R @ U_step), diagonalizes D_cur's own block for each cluster in `clusters`
    independently -- each cluster's orbitals become that cluster's own natural orbitals.
    D_cur is the 1-RDM in U_step's PARENT basis (not yet rotated by U_step); same
    np.linalg.eigh(...).T convention as the global NatOs seeding in _build_initial_bases.
    Exactly cost-neutral for every cost function in this module: N_C is basis-independent
    within its own subspace, so this changes which orbital combination represents a cluster,
    never the cost."""
    Uc = U_step.conj()
    D_in_target_basis = Uc @ D_cur @ U_step.T
    R = np.eye(norb, dtype=D_in_target_basis.dtype)
    for cluster in clusters:
        idx = np.asarray(cluster)
        block = D_in_target_basis[np.ix_(idx, idx)]
        _, evecs = np.linalg.eigh(block)
        R[np.ix_(idx, idx)] = evecs.T # this transposition matches the way we transform
        # the 1RDMs; compare with U_NatOs = evecs.T in cluster_numbers_metrics.py
    return R

def _split_one_cluster(
    deco: Decomposition,
    rdm_data_cur: RDMData,
    n1: np.ndarray,
    n2: np.ndarray,
    cluster_idx: int,
    cost_function_constructor: CostFunctionConstructor,
    maxiter: int,
    min_child_cluster_size: int = 1,
    naturalize_children: bool = True,
    optimize_rotation_in_beam_search: bool = True,
) -> Decomposition:
    """Split deco.partition[cluster_idx] via Fiedler bisection (never leaving
    either resulting subcluster smaller than min_child_cluster_size); if
    optimize_rotation_in_beam_search, re-optimize the rotation restricted to
    that cluster's orbitals starting from deco.U, else leave that block at the
    identity (Fiedler bisection is still applied); and (if naturalize_children)
    diagonalize each child's own 1-RDM block afterward -- exactly cost-neutral
    (N_C is basis-independent within its own subspace), so this only changes
    which orbital combination represents each cluster, never the recorded
    cost."""
    norb = rdm_data_cur.norb
    V = deco.partition[cluster_idx]
    V1, V2 = fiedler_bipartition(n1, n2, V, min_child_size=min_child_cluster_size)

    child_partition = deco.partition[:cluster_idx] + deco.partition[cluster_idx + 1 :] + [V1, V2]
    child_cluster_matrix = partition_to_cluster_matrix(child_partition, norb)

    f_full = cost_function_constructor(rdm_data_cur, child_cluster_matrix)
    if optimize_rotation_in_beam_search:
        free_indices = _block_flat_indices(V, norb)
        x_opt_full, cost_opt = _optimize_restricted_rotation(f_full, norb, free_indices, maxiter)
    else:
        x_opt_full = np.zeros(comb(norb, 2))
        cost_opt = float(f_full(jnp.array(x_opt_full)))

    U_block = np.asarray(params_to_U_jax(jnp.array(x_opt_full), norb))
    U_child = U_block @ deco.U

    if naturalize_children:
        R = _local_natural_orbital_rotation(U_block, rdm_data_cur.D, [V1, V2], norb)
        U_child = R @ U_child

    return Decomposition(partition=child_partition, U=U_child, cost=cost_opt)

def _can_split(cluster_size: int, min_parent_cluster_size: int, min_child_cluster_size: int) -> bool:
    """Whether a cluster of this size is eligible to be split at all: strictly
    larger than min_parent_cluster_size (else it's already "small enough" and never
    chosen), and large enough that both children of a split can respect
    min_child_cluster_size (necessary and sufficient for
    fiedler_bipartition(..., min_child_size=min_child_cluster_size) to
    succeed on a cluster of this size)."""
    return cluster_size > min_parent_cluster_size and cluster_size >= 2 * min_child_cluster_size

def expand_decomposition(
    deco: Decomposition,
    rdm_data_ref: RDMData,
    cost_function_constructor: CostFunctionConstructor,
    num_subdecos: int,
    min_parent_cluster_size: int,
    maxiter: int,
    min_child_cluster_size: int = 1,
    naturalize_children: bool = True,
    optimize_rotation_in_beam_search: bool = True,
) -> list[Decomposition]:
    """Produce up to `num_subdecos` child decompositions of `deco`: pick the
    (up to num_subdecos) clusters with the largest current number-variance
    (clusters not satisfying _can_split -- e.g. at or below min_parent_cluster_size,
    or too small to respect min_child_cluster_size on both sides of a
    split -- are never chosen), and split each independently via
    _split_one_cluster (see that function for naturalize_children and
    optimize_rotation_in_beam_search)."""
    rdm_data_cur = rotate_rdm_data(rdm_data_ref, deco.U)
    n1, n2 = cluster_number_stats(rdm_data_cur, deco.partition) # notice: using current/rotated rdm data

    splittable = [
        i
        for i, c in enumerate(deco.partition)
        if _can_split(len(c), min_parent_cluster_size, min_child_cluster_size)
    ]
    if not splittable:
        return []

    splittable.sort(key=lambda i: -cluster_variance(n1, n2, deco.partition[i]))
    chosen = splittable[:num_subdecos]

    return [
        _split_one_cluster(
            deco, rdm_data_cur, n1, n2, cluster_idx, cost_function_constructor, maxiter,
            min_child_cluster_size=min_child_cluster_size,
            naturalize_children=naturalize_children,
            optimize_rotation_in_beam_search=optimize_rotation_in_beam_search,
        )
        for cluster_idx in chosen
    ]

# =============================================================================
# Main beam search (human verified ✓)
# =============================================================================

def _initial_beam(
    rdm_data_ref: RDMData,
    cost_function_constructor: CostFunctionConstructor,
    num_decos: int,
    initial_bases: list[np.ndarray] | None, # list unitaries; in main workflow, built with _initial_bases and contains identity (for MOs) and/or unitary MOs -> NatOs
) -> list[Decomposition]:
    norb = rdm_data_ref.norb
    if initial_bases is None:
        initial_bases = [np.eye(norb)]

    beam = []
    for i in range(num_decos):
        U0 = np.asarray(initial_bases[i % len(initial_bases)]) # use initial_bases equally often
        partition0 = [list(range(norb))] # start with trivial coarsest partition
        cluster_matrix0 = partition_to_cluster_matrix(partition0, norb)
        rdm_data_0 = rotate_rdm_data(rdm_data_ref, U0)
        f0 = cost_function_constructor(rdm_data_0, cluster_matrix0)
        cost0 = float(f0(jnp.zeros(comb(norb, 2))))
        beam.append(Decomposition(partition=partition0, U=U0, cost=cost0))
    return beam


def _log_search_stopped(
    reason: str, round_idx: int, beam: list[Decomposition], config: DecompositionOptimizerConfig
) -> None:
    """Log why the beam search's round loop is terminating. If a requested
    target_num_clusters was not reached, this is worth a warning naming the
    likely cause (min_parent_cluster_size and/or min_child_cluster_size blocking
    every remaining cluster) rather than a silent/plain info line."""
    if config.target_num_clusters is not None and beam[0].num_clusters < config.target_num_clusters:
        logger.warning(
            "Round %d: %s (stuck at %d clusters), but target_num_clusters=%d was not reached "
            "-- likely caused by min_parent_cluster_size=%d and/or min_child_cluster_size=%d.",
            round_idx, reason, beam[0].num_clusters, config.target_num_clusters,
            config.min_parent_cluster_size, config.min_child_cluster_size,
        )
    else:
        logger.info("Round %d: %s, stopping.", round_idx, reason)


def run_decomposition_optimizer(
    cost_function_constructor: CostFunctionConstructor,
    rdm_data: RDMData, # of the initial (DMRG) GS; in ref basis (typically MOs)
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
    if norb < 2 * config.min_child_cluster_size:
        logger.warning(
            "min_child_cluster_size=%d requires at least %d orbitals to perform even one "
            "split, but norb=%d: the beam search will not be able to split the initial "
            "single-cluster decomposition at all.",
            config.min_child_cluster_size, 2 * config.min_child_cluster_size, norb,
        )

    beam = _initial_beam(rdm_data, cost_function_constructor, config.num_decos, initial_bases)
    for deco in beam:
        validate_partition(deco.partition, norb) # currently redundant

    trajectory = [min(beam, key=lambda d: d.cost)] # here beam contains only trivial decos
    # with same cost. The minimization therefore just picks one
    round_idx = 0

    while True:
        if config.target_num_clusters is not None and beam[0].num_clusters >= config.target_num_clusters:
            break
        if all(
            not _can_split(len(c), config.min_parent_cluster_size, config.min_child_cluster_size)
            for deco in beam
            for c in deco.partition
        ):
            _log_search_stopped("no cluster in the beam is splittable further", round_idx, beam, config)
            break
        if config.max_rounds is not None and round_idx >= config.max_rounds:
            break

        candidates: list[Decomposition] = []
        for deco in beam:
            candidates.extend(
                expand_decomposition(
                    deco, rdm_data, cost_function_constructor,
                    config.num_subdecos, config.min_parent_cluster_size, config.maxiter,
                    min_child_cluster_size=config.min_child_cluster_size,
                    naturalize_children=config.naturalize_children,
                    optimize_rotation_in_beam_search=config.optimize_rotation_in_beam_search,
                )
            )
        if not candidates:
            _log_search_stopped("no deco could be split further", round_idx, beam, config)
            break

        for cand in candidates:
            validate_partition(cand.partition, norb)

        candidates.sort(key=lambda d: d.cost)
        beam = candidates[: config.num_decos]
        trajectory.append(beam[0]) # append to output trajectory the best decomposition
        # beam[0] for the current number of clusters

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
    left over from the greedy/hierarchical splitting process.

    Note: the cost surface is exactly flat along within-cluster rotation
    directions at any base point, but params_to_U_jax's coordinates are only
    aligned with that flat direction at the exponential map's origin (x=0, i.e.
    exactly a naturalize_children=True deco.U) -- once L-BFGS-B moves other
    coordinates away from 0, there's no optimization pressure to drift off a
    prior per-cluster naturalization, but no hard guarantee against incidental
    drift either."""
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
# CLI Interface (human verified ✓)
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


def _run_dmrg_and_build_rdm_data_from_fcidump(
    args: argparse.Namespace,
) -> tuple[RDMData, dict, Any, float, np.ndarray, np.ndarray, float, tuple[int, int]]:
    """FCIDUMP counterpart of _run_dmrg_and_build_rdm_data: builds the DMRG
    solver directly from a precomputed Hamiltonian (e.g. a CASSCF
    active-space FCIDUMP written by another workflow) via
    Block2DMRGSolver.from_fcidump, instead of building a molecule + HF
    reference from --molecule/--basis-set/--bond-length -- the physics comes
    entirely from the FCIDUMP. Of those three CLI arguments, --molecule and
    --basis-set are then only used as free-form labels for the output/plots
    directory tree (see _geometry_output_subpath) and metadata; --bond-length
    is not used at all (still required positionally, but discarded -- see
    _geometry_output_subpath and main()'s metadata construction). Unlike
    compute_dmrg, this skips the DMRG-vs-HF variational sanity check: an
    active-space FCIDUMP has no comparable whole-system HF energy to check
    against.

    Returns: rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec
    """
    import pyscf.ao2mo
    from cluster_numbers_metrics import extract_rdms
    from src.dmrg_solver import Block2DMRGSolver, DMRGConfig, restore_g2e, solve_or_load_ground_state

    fcidump_path = Path(args.fcidump)
    logger.info(f"Loading Hamiltonian from FCIDUMP: {fcidump_path}")

    store_dir = (
        Path(args.wavefunction_dir)
        / f"dmrg_fcidump_{fcidump_path.stem}_bd{args.bond_dim}_sw{args.n_sweeps}"
    )
    solver = Block2DMRGSolver.from_fcidump(fcidump_path, store_dir=store_dir, n_threads=args.n_threads)
    norb = solver.n_sites
    nelec = ((solver.n_elec + solver.spin) // 2, (solver.n_elec - solver.spin) // 2)

    logger.info(f"Number of orbitals: {norb}")
    logger.info(f"Number of electrons: {nelec}")
    logger.info(f"Space dimension: {comb(norb, nelec[0]) * comb(norb, nelec[1])}")

    logger.info("Running DMRG...")
    result = solve_or_load_ground_state(
        solver,
        config=DMRGConfig(max_bond_dim=args.bond_dim, n_sweeps=args.n_sweeps),
        reuse=not args.no_reuse,
    )
    dmrg_energy = result.energy
    logger.info(f"DMRG Energy: {dmrg_energy:.10f} Ha")

    h1e = solver.h1e
    ecore = solver.ecore
    g2e_full = restore_g2e(solver.g2e, norb)
    g2e = pyscf.ao2mo.restore(4, g2e_full, norb)  # match compute_dmrg's packed convention

    if args.cost_function == "commutator":
        D, Gamma, rdm3, rdm4 = extract_rdms(solver, result.mps_tag, with_34_rdms=True)
        rdm_data = RDMData(D=D, Gamma=Gamma, rdm3=rdm3, rdm4=rdm4, h1e=h1e, g2e_full=g2e_full)
    else:
        D, Gamma = extract_rdms(solver, result.mps_tag, with_34_rdms=False)
        rdm_data = RDMData(D=D, Gamma=Gamma)

    dmrg_metadata = {
        "nelec": [int(x) for x in nelec], "dmrg_energy": float(dmrg_energy), "fcidump": str(fcidump_path),
    }
    return rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec


# -----------------------------------------------------------------------------
# K-sectors / K-states sector analysis for the winning decompositions
#
# This section is the per-trajectory-entry analog of
# cluster_numbers_metrics.py's compute_sector_analysis + generate_plots: it
# reuses SingleMaxelectransferResult, BasisResult, MetricsOutput,
# get_ordered_state_projections_in_sectors, get_ordered_decoupled_states,
# save_metrics from that module UNCHANGED, and get_K_sectors_values_energies /
# get_K_states_values_energies from src.K_sectors_plots UNCHANGED. generate_plots
# (and, through it, plot_energy_vs_K / plot_dual_bar_chart) is called with the
# additional from_beam_search=True / min_child_cluster_size / target_num_clusters /
# initial_basis passthrough kwargs, which only add an extra title line -- no
# other difference from compute_sector_analysis's own call. The only other
# real difference from
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
                "or a larger --min-child-cluster-size in the search (--min-parent-cluster-size "
                "alone only controls split eligibility, not how balanced a split is, so it "
                "doesn't reliably shrink the largest cluster)."
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
        "bond_length": args.bond_length if args.fcidump is None else None,
        "bond_angle": args.bond_angle if args.fcidump is None else None,
        "fcidump": args.fcidump,
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
                from_beam_search=True,
                min_child_cluster_size=args.min_child_cluster_size,
                target_num_clusters=args.target_num_clusters,
                initial_basis=args.initial_basis,
            )


def _geometry_output_subpath(args: argparse.Namespace) -> Path:
    molecule = args.molecule.lower()
    basis_set = args.basis_set.lower()
    if args.fcidump is not None:
        # No molecule/bond-length geometry to key on here -- nest under the
        # source FCIDUMP's filename instead. molecule/basis_set are then just
        # free-form labels the caller chose for this Hamiltonian (bond_length
        # is unused in this mode).
        return Path(molecule) / basis_set / f"fcidump_{Path(args.fcidump).stem}"
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
        if deco.num_clusters == 1:
            continue
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

    xs = [d.num_clusters for d in trajectory if d.num_clusters > 1]
    ys = [d.cost for d in trajectory if d.num_clusters > 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", label="beam search")
    if polished_trajectory is not None:
        ys_polished = [d.cost for d in polished_trajectory if d.num_clusters > 1]
        ax.plot(xs, ys_polished, "s--", label="polished")
        ax.legend()
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(f"Cost ({metadata['cost']})")
    ax.set_yscale("log")
    ax.set_title(
        f"{metadata['molecule']} / {metadata['basis_set']}, norb={metadata['norb']}, cost={metadata['cost']}\n"
        f"min. child cl. size = {metadata['min_child_cluster_size']}, "
        f"target num. clusters = {metadata['target_num_clusters']}, in. basis = {metadata['initial_basis']}"
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
    # convention as cluster_numbers_metrics.py). When --fcidump is given,
    # molecule/basis_set/bond_length are NOT used to build a molecule/HF
    # reference: molecule/basis_set instead become free-form labels for the
    # output/plots directory tree (see _geometry_output_subpath) and
    # metadata, while bond_length is discarded entirely (still required
    # positionally, but unused -- pass any placeholder float). E.g.
    # `fe2s2 casscf_10e10o 0 commutator --fcidump path/to/FCIDUMP`.
    parser.add_argument(
        "molecule", type=str,
        help=(
            "Molecule to analyze (one of h2, h2o, n2, lih, h4_linear, h4_square, h4_rectangle), "
            "or a free-form label when --fcidump is given"
        ),
    )
    parser.add_argument(
        "basis_set", type=str,
        help="Basis set (e.g., sto-3g, 6-31g), or a free-form label when --fcidump is given",
    )
    parser.add_argument(
        "bond_length", type=float,
        help="Bond length in Angstrom (unused, but still required, when --fcidump is given)",
    )
    parser.add_argument(
        "cost_function", type=str,
        help="Cost function type (variance, eval_eq, extremality, mixed, commutator)",
    )

    parser.add_argument("--bond-angle", type=float, default=None, help="Bond angle in degrees (for H2O)")
    parser.add_argument(
        "--fcidump", type=str, default=None,
        help=(
            "Path to an FCIDUMP file with a precomputed Hamiltonian (e.g. a CASSCF active-space "
            "Hamiltonian). When given, skips building a molecule/HF reference from "
            "molecule/basis_set/bond_length entirely and runs DMRG directly on this Hamiltonian "
            "(via Block2DMRGSolver.from_fcidump)."
        ),
    )

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
        "--min-parent-cluster-size", type=int, default=1,
        help="Clusters at or below this size are never split (default: 1)",
    )
    parser.add_argument(
        "--min-child-cluster-size", type=int, default=1,
        help=(
            "Never split a cluster if either resulting subcluster would have fewer than this "
            "many orbitals -- unlike --min-parent-cluster-size, this constrains the split itself, not "
            "just which clusters are eligible to be split (default: 1, i.e. no extra restriction)"
        ),
    )
    parser.add_argument(
        "--no-naturalize-children", action="store_true",
        help=(
            "After each split, skip diagonalizing the two new clusters' own 1-RDM blocks "
            "(natural orbitals within each cluster) -- on by default; this is exactly "
            "cost-neutral for the split just made (every cost in this module only depends on "
            "each cluster's aggregate occupation statistics, never on which specific orbital "
            "combination represents it), so disabling it only affects which orbital basis "
            "represents each cluster, not the search's cost values"
        ),
    )
    parser.add_argument(
        "--target-num-clusters", type=int, default=None,
        help="Stop once this many clusters is reached (default: run down to singleton clusters)",
    )
    parser.add_argument("--max-rounds", type=int, default=None, help="Extra safety cap on rounds (default: none)")
    parser.add_argument(
        "--maxiter", type=int, default=200, help="L-BFGS-B iterations per split, block-restricted (default: 200)"
    )
    parser.add_argument(
        "--no-orb-opt-in-beam-search", action="store_true",
        help=(
            "Skip the per-split continuous orbital-rotation reoptimization during the beam "
            "search -- the discrete Fiedler bisection of the chosen cluster still happens, "
            "but each split's new rotation block is left at the identity relative to the "
            "parent decomposition's U (on by default). Independent of "
            "--no-naturalize-children, which still runs by default regardless of this flag. "
            "Does not affect the separate final polish step (--no-polish/--polish-maxiter), "
            "which always performs a full joint reoptimization on each trajectory entry."
        ),
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
        f"num_subdecos={args.num_subdecos}, min_parent_cluster_size={args.min_parent_cluster_size}, "
        f"min_child_cluster_size={args.min_child_cluster_size}, "
        f"naturalize_children={not args.no_naturalize_children}, "
        f"optimize_rotation_in_beam_search={not args.no_orb_opt_in_beam_search}"
    )
    if args.bond_angle is not None:
        logger.info(f"Bond angle={args.bond_angle}")

    _known_molecules = {"h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"}
    if args.fcidump is None and args.molecule.lower() not in _known_molecules:
        logger.error(
            f"Unsupported molecule: {args.molecule}. Supported: {sorted(_known_molecules)} "
            "(or pass --fcidump to load a precomputed Hamiltonian instead)."
        )
        exit(1)

    start_time = time.time()

    try:
        if args.fcidump is not None:
            rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = (
                _run_dmrg_and_build_rdm_data_from_fcidump(args)
            )
        else:
            rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(args)
        norb = rdm_data.norb

        cost_function_constructor = _build_cost_function_constructor(args.cost_function, args.var_exponent)
        initial_bases = _build_initial_bases(args, rdm_data, norb)

        opt_config = DecompositionOptimizerConfig(
            num_decos=args.num_decos,
            num_subdecos=args.num_subdecos,
            min_parent_cluster_size=args.min_parent_cluster_size,
            min_child_cluster_size=args.min_child_cluster_size,
            naturalize_children=not args.no_naturalize_children,
            target_num_clusters=args.target_num_clusters,
            max_rounds=args.max_rounds,
            maxiter=args.maxiter,
            optimize_rotation_in_beam_search=not args.no_orb_opt_in_beam_search,
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
            "bond_length": args.bond_length if args.fcidump is None else None,
            "bond_angle": args.bond_angle if args.fcidump is None else None,
            "cost": args.cost_function,
            "timestamp": get_timestamp(),
            "git_hash": get_git_hash(),
            "norb": norb,
            "num_decos": args.num_decos,
            "num_subdecos": args.num_subdecos,
            "min_parent_cluster_size": args.min_parent_cluster_size,
            "min_child_cluster_size": args.min_child_cluster_size,
            "naturalize_children": not args.no_naturalize_children,
            "optimize_rotation_in_beam_search": not args.no_orb_opt_in_beam_search,
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
