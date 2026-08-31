from __future__ import annotations

"""
Cluster Number Sector Search

Cheap (polynomial in norb and in the number of clusters K -- never
exponential in the many-body Hilbert space dimension comb(norb,Nalpha)*
comb(norb,Nbeta)) ranking of candidate cluster-number sectors by
"relevance", for one optimized decomposition deco = (partition, U) out of
cluster_number_decomposition_optimization.py's trajectory.

A "sector" here is labeled by a tuple (N_0, ..., N_{K-1}) of eigenvalues of the K
cluster number operators N_K = sum_{p in C_K} n_p (same convention as
number_and_parity_symmetry_sectors's own sector labels, minus the parity
sub-label, which this module doesn't use). Here, the clusters C_K are defined
by the partition, and the orbitals by U. The output is meant to feed a
downstream variational subspace solver (e.g. a matrix-free Lanczos/Davidson
over the union of retained sectors' determinant supports, in the style of
src/selected_sector_lanczos.py; or a QSENSE-like method) that will improve
on the beam search's own initial low-bond-dimension DMRG energy.

Algorithm (see rank_relevant_sectors for the orchestration):
  0. normalize_cluster_family: tolerate messy input partitions (clusters
     sharing an orbital get merged; orbitals covered by none get folded
     into an implicit trailing "ghost" cluster) rather than assuming the
     strict, disjoint, fully-covering partition that
     cluster_number_decomposition_optimization.validate_partition enforces.
  A. cluster_number_moments: mean vector and full K x K covariance matrix
     of the cluster number operators, from the 1-/2-RDM alone.
  B. main_sector_label / gaussian_weight: a cheap, closed-form estimate of
     the main sector holding most of the state's weight, and how much weight
     neighbouring sectors plausibly hold, via a multivariate-Gaussian
     model of the joint cluster-count distribution (maximum-entropy, given
     only the first two RDM-derived moments).
  C. enumerate_candidate_labels: which OTHER sectors are even worth
     considering -- a bounded graph search (BFS over elementary
     single-electron cluster-to-cluster moves), not an exhaustive sweep
     over all comb(norb,Nalpha)*comb(norb,Nbeta) determinants.
  D. coupling_strength_{complex,real}: a PT2-flavoured "how strongly does H
     couple the main sector to this candidate" sector-relevance score, computed directly
     from RDM contractions of h1e/g2e-derived tensors MASKED to the
     specific electron-transfer pattern (example of a pattern: "move one electron
     from cluster I to cluster J"). This score is exactly zero whenever
     the pattern would move 3 or more electrons across clusters, as an exact
     consequence of H having body-rank <= 2. Therefore, only max_elec_transfer<=2
     patterns ever get a direct energy score; max_elec_transfer>=2 candidates fall back
     to the weight score alone. Two independent implementations are
     provided -- fully general complex, and a real-only fast path (this
     codebase's own h1e/g2e/RDMs are always real) -- verified
     to agree with each other and with a from-scratch Fock-space ground
     truth in tests/test_cluster_number_sector_search.py.
  E. sector_dimension: the exact size of a sector's dimension,
     via a simple calculation, without ever enumerating determinants.
  F. rank_relevant_sectors: ties 0-E together into a single ranked,
     greedily-truncated (num_sectors_to_retain / max_cum_dim_to_retain)
     list of SectorRelevance entries.

CLI usage (mirrors cluster_number_decomposition_optimization.py: runs its
own beam search via the same DMRG + cost-function + initial-basis + config
pipeline, then runs the sector search on the resulting trajectory):
    python cluster_number_sector_search.py h2o sto-3g 2.0 commutator --bond-angle 104.5
    python cluster_number_sector_search.py h4_square 6-31g 1.0 variance --force-full-rdms --analyze-num-clusters 3
    python cluster_number_sector_search.py h2o sto-3g 2.0 commutator --K-sector-analysis

Library usage (bring your own Decomposition + RDMData, e.g. from
cluster_number_decomposition_optimization.py's own trajectory):
    from cluster_number_sector_search import rank_relevant_sectors, SectorSearchConfig
    entries = rank_relevant_sectors(deco, rdm_data, nelec, SectorSearchConfig(max_elec_transfer=2))

See tests/test_cluster_number_sector_search.py for the from-scratch
Fock-space verification of the RDM-contraction formulas in Step D.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    cluster_number_stats,
    partition_to_cluster_matrix,
    rotate_rdm_data,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures (human verified pending)
# =============================================================================


@dataclass
class SectorSearchConfig:
    """Hyperparameters for rank_relevant_sectors."""

    num_sectors_to_retain: int | None = None  # (max) number of sectors to retain; None = unlimited
    max_cum_dim_to_retain: int | None = None  # max sum of retained sectors' dimensions; None = unlimited
    max_elec_transfer: int = 2  # candidate labels are generated out to this many electrons moved


@dataclass
class SectorRelevance:
    """One ranked candidate sector."""

    label: tuple[int, ...]  # (N_0, ..., N_{K-1}), K = len(deco.partition) (ghost cluster, if any, stripped)
    weight_score: float  # Gaussian/max-entropy estimate of the state's weight in this sector
    energy_score: float | None  # q(delta): RDM-computable coupling strength to the main sector; None if not scored
    energy_tier: int | None  # 1 (h1e only) or 2 (+g2e/rdm3/rdm4); None if energy_score is None
    elec_transfer: int  # t = (1/2) * sum(|label - main_label|)
    dimension: int  # exact size of this sector's determinant support


# =============================================================================
# Step 0: cluster-family normalization (human verified pending)
# =============================================================================


def normalize_cluster_family(
    partition: list[list[int]], norb: int
) -> tuple[list[list[int]], list[int], bool]:
    """Turn an arbitrary list of orbital-index lists into a genuine partition
    of range(norb): clusters sharing >= 1 orbital are merged (transitively,
    via union-find) into one effective cluster; orbitals covered by none are
    collected into one trailing "ghost" cluster. Neither condition is
    treated as an error -- both are logged and handled -- so callers with
    messy input (e.g. a cluster_matrix-style specification that doesn't
    happen to be a strict partition, unlike Decomposition.partition, which
    cluster_number_decomposition_optimization.validate_partition enforces
    to always be one) still get a usable result rather than a crash.

    Returns:
        clean_partition: disjoint, covers range(norb) exactly.
        cluster_map: cluster_map[i] is the index into clean_partition that
            original cluster i was merged into (identity if nothing merged).
        had_ghost: whether a trailing ghost cluster was appended.
    """
    if any(len(c) == 0 for c in partition):
        raise ValueError("normalize_cluster_family: partition contains an empty cluster.")

    n = len(partition)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    owner: dict[int, int] = {}
    for i, cluster in enumerate(partition):
        for p in cluster:
            if p in owner:
                union(owner[p], i)
            else:
                owner[p] = i

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clean_partition: list[list[int]] = []
    cluster_map = [0] * n
    merged_report = []
    for members in groups.values():
        merged_orbitals: set[int] = set()
        for i in members:
            merged_orbitals.update(partition[i])
            cluster_map[i] = len(clean_partition)
        clean_partition.append(sorted(merged_orbitals))
        if len(members) > 1:
            merged_report.append(members)

    if merged_report:
        logger.warning(
            "normalize_cluster_family: input clusters overlap (share >=1 orbital) -- "
            "merged the following groups of 0-indexed input cluster indices into one "
            "effective cluster each: %s",
            merged_report,
        )

    covered: set[int] = set()
    for cluster in clean_partition:
        covered.update(cluster)
    uncovered = sorted(set(range(norb)) - covered)
    had_ghost = bool(uncovered)
    if had_ghost:
        logger.info(
            "normalize_cluster_family: orbitals %s are covered by no input cluster -- "
            "appended as a trailing ghost cluster for internal bookkeeping.",
            uncovered,
        )
        clean_partition.append(uncovered)

    return clean_partition, cluster_map, had_ghost


# =============================================================================
# Step A: cluster-number moments (human verified pending)
# =============================================================================


def cluster_number_moments(
    rdm_data_cur: RDMData, clean_partition: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector mu_K = <N_K> and full K x K covariance matrix
    Sigma_KL = Cov(N_K, N_L) of the cluster number operators, from the
    1-/2-RDM of rdm_data_cur (already rotated into the decomposition's own
    basis -- call rotate_rdm_data(rdm_data, deco.U) first). Sigma is exactly
    singular along the all-ones direction (total electron number has zero
    variance in a DMRG state) -- see gaussian_weight for how that's used.

    Reuses cluster_number_stats with the "single all-encompassing cluster"
    trick already used by fiedler_orbital_order in
    cluster_number_decomposition_optimization.py to get the FULL (not just
    intra-cluster) pairwise <n_p n_q> matrix cheaply -- O(norb^2) overall.
    """
    norb = rdm_data_cur.norb
    n1, n2 = cluster_number_stats(rdm_data_cur, [list(range(norb))])

    idx = [np.asarray(c) for c in clean_partition]
    K = len(clean_partition)
    mu = np.array([float(n1[c].sum()) for c in idx])
    Sigma = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            block = n2[np.ix_(idx[i], idx[j])] - np.outer(n1[idx[i]], n1[idx[j]])
            Sigma[i, j] = float(block.sum())
    return mu, Sigma


# =============================================================================
# Step B: main sector label + Gaussian weight (human verified pending)
# =============================================================================


def main_sector_label(
    mu: np.ndarray, nelec: tuple[int, int], cluster_sizes: list[int]
) -> tuple[int, ...]:
    """Largest-remainder rounding of the real-valued mean vector mu to the
    nearest integer point on {N : sum(N_K) = Nalpha+Nbeta}, clipped to each
    cluster's capacity [0, 2*|C_K|]. For a good decomposition mu should
    already be very close to integer (that's what the beam search's own
    cost function was optimizing for); this rounding mainly guarantees
    sum-exactness, which a naive per-cluster round() (as in
    make_eval_eq_cost_constructor) doesn't."""
    N_tot = nelec[0] + nelec[1]
    capacities = [2 * s for s in cluster_sizes]

    floor_vals = np.clip(np.floor(mu).astype(int), 0, capacities)
    remainder = int(N_tot - floor_vals.sum())
    fractional = mu - np.floor(mu)
    order = np.argsort(-fractional)  # largest fractional part first

    label = floor_vals.copy()
    if remainder > 0:
        for k in order:
            if remainder == 0:
                break
            if label[k] < capacities[k]:
                label[k] += 1
                remainder -= 1
    elif remainder < 0:
        for k in order[::-1]:  # smallest fractional part first -- take from these
            if remainder == 0:
                break
            if label[k] > 0:
                label[k] -= 1
                remainder += 1

    if remainder != 0:
        raise ValueError(
            f"main_sector_label: could not place all {N_tot} electrons within cluster "
            f"capacities {capacities} (mu={mu.tolist()}) -- nelec inconsistent with cluster_sizes?"
        )
    return tuple(int(x) for x in label)


def gaussian_weight(label: tuple[int, ...], mu: np.ndarray, sigma_pinv: np.ndarray) -> float:
    """exp(-1/2 (label-mu)^T Sigma^+ (label-mu)) -- unnormalized max-entropy/
    multivariate-Gaussian weight estimate (see module docstring): relative
    ranking signal only, not a calibrated probability. sigma_pinv should be
    precomputed once per decomposition via np.linalg.pinv(Sigma) and reused
    across every candidate -- Sigma's null direction (all-ones) is exactly
    the direction pinv correctly treats as forbidden, and every label/mu we
    evaluate this on already lies in the sum(N_K)=Nalpha+Nbeta hyperplane."""
    diff = np.asarray(label, dtype=float) - mu
    exponent = -0.5 * float(diff @ sigma_pinv @ diff)
    return float(np.exp(exponent))


# =============================================================================
# Step C: candidate generation (human verified pending)
# =============================================================================


def enumerate_candidate_labels(
    main_label: tuple[int, ...], cluster_sizes: list[int], max_elec_transfer: int
) -> dict[tuple[int, ...], int]:
    """BFS over elementary single-electron moves N -> N + e_I - e_J (any
    I != J with room to do so), out to max_elec_transfer graph-distance from
    main_label. Graph distance in this graph exactly equals the
    transportation distance t = (1/2)*sum|N_K - N_K^(0)| (moving electrons
    one at a time is always optimal), so plain single-step BFS already
    generates the full, correct candidate set -- no separate "2-electron
    edge" is needed for generation (only for scoring, where t determines
    which tier of q(delta) is even attempted -- see build_transfer_masks).

    Returns {label: t}, including main_label itself at t=0.
    """
    K = len(main_label)
    caps = [2 * s for s in cluster_sizes]

    visited: dict[tuple[int, ...], int] = {main_label: 0}
    frontier = [main_label]
    for t in range(1, max_elec_transfer + 1):
        next_frontier = []
        for label in frontier:
            for i in range(K):
                if label[i] >= caps[i]:
                    continue
                for j in range(K):
                    if j == i or label[j] <= 0:
                        continue
                    new_label = list(label)
                    new_label[i] += 1
                    new_label[j] -= 1
                    new_label = tuple(new_label)
                    if new_label not in visited:
                        visited[new_label] = t
                        next_frontier.append(new_label)
        frontier = next_frontier
        if not frontier:
            break
    return visited


# =============================================================================
# Step D: coupling-strength q(delta) -- Tier 1 (h1e) / Tier 2 (+g2e/rdm3/rdm4) (human verified pending)
#
# Formulas verified (symbolically, via a from-scratch normal-ordering
# derivation, and numerically, against an explicit Fock-space ground truth
# built independently of this module) for the general spinful, complex case.
# D, Gamma, rdm3, rdm4 use this codebase's spin-summed conventions:
#   D_pq        = sum_sigma           D^spin_{p sigma, q sigma}
#   Gamma_pqrs  = sum_{sigma,tau}      Gamma^spin_{p sigma, q tau, r tau, s sigma}
#   rdm3[p,q,r, s,t,u]    = sum_{s1,s2,s3}   <c+_{p,s1} c+_{q,s2} c+_{r,s3}  a_{s,s3} a_{t,s2} a_{u,s1}>
#   rdm4[p,q,r,s, t,u,v,w] = sum_{s1..s4}    <c+_{p,s1}..c+_{s,s4}  a_{t,s4} a_{u,s3} a_{v,s2} a_{w,s1}>
# (the "nested" extension of the given 2-RDM pattern -- creator i pairs,
# same spin, with annihilator N+1-i -- confirmed directly against
# extract_rdms's own spin-block-combining code in cluster_numbers_metrics.py).
# =============================================================================


def build_transfer_masks(
    indicator: np.ndarray, norb: int, delta: np.ndarray, need_g: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Boolean masks selecting exactly the h1e / g2e-derived entries whose
    net cluster-flow equals delta (a length-K vector):
        h_mask[p,q]     iff  indicator[:,p] - indicator[:,q]                       == delta
        g_mask[p,q,r,s] iff (indicator[:,p]+indicator[:,q]) - (indicator[:,r]+indicator[:,s]) == delta
    (indicator: (K, norb) one-hot cluster membership, e.g.
    partition_to_cluster_matrix(clean_partition, norb)). One recipe covers
    direct 1-electron hops, density-assisted (spectator-mediated)
    1-electron hops, and 2-electron pair-/double-hops uniformly, and is
    identically all-False whenever t = (1/2)*sum|delta| >= 3, since a
    1-body term touches at most 2 cluster slots and a 2-body term at most 4
    -- the exact reason direct q(delta) scoring is only ever attempted for
    t<=2 (see module docstring).
    """
    diff_pq = indicator[:, :, None] - indicator[:, None, :]  # (K, norb, norb)
    h_mask = np.all(diff_pq == delta[:, None, None], axis=0)

    g_mask = None
    if need_g:
        flow = (
            indicator[:, :, None, None, None]
            + indicator[:, None, :, None, None]
            - indicator[:, None, None, :, None]
            - indicator[:, None, None, None, :]
        )  # (K, norb, norb, norb, norb)
        g_mask = np.all(flow == delta[:, None, None, None, None], axis=0)
    return h_mask, g_mask


def coupling_strength_complex(
    h1e: np.ndarray,
    g_derived: np.ndarray | None,
    D: np.ndarray,
    Gamma: np.ndarray,
    rdm3: np.ndarray | None,
    rdm4: np.ndarray | None,
    h_mask: np.ndarray,
    g_mask: np.ndarray | None,
) -> tuple[float | None, int | None]:
    """q(delta) = <psi|V_delta^dagger V_delta|psi>, fully general complex
    path. Tier 1 (TermA, h1e masked by h_mask only) is always attempted;
    Tier 2 (adds TermB + conj(TermB) + TermD, g_derived masked by g_mask) is
    attempted only when g_mask/rdm3/rdm4 are all available and g_mask
    selects something.

    Returns (score, tier_used); (None, None) if neither mask selects
    anything (delta unreachable at whatever tier(s) the caller could supply
    data for -- e.g. t>=3, or t==2 with only Tier-1 data on hand).
    """
    h = h1e * h_mask
    have_g = g_mask is not None and g_derived is not None and rdm3 is not None and rdm4 is not None
    g = g_derived * g_mask if have_g else None

    h_active = np.any(h)
    g_active = g is not None and np.any(g)
    if not h_active and not g_active:
        return None, None

    hc = h.conj()
    # TermA = sum_{p,q,q'} conj(h[p,q])*h[p,q']*D[q,q']  +  sum_{p,p',q,q'} conj(h[p,q])*h[p',q']*Gamma[q,p',q',p]
    termA = np.einsum("pq,pr,qr->", hc, h, D, optimize=True) + np.einsum(
        "pq,ab,qabp->", hc, h, Gamma, optimize=True
    )

    if not g_active:
        return float(termA.real), 1

    gc = g.conj()
    # TermB = 1/2 * sum conj(h[p,q])*g[p',q',r',s'] * ( D3[q,p',q',r',s',p]
    #                                                    + delta(p,q')*Gamma[q,p',s',r']
    #                                                    + delta(p,p')*Gamma[q,q',r',s'] )
    termB = 0.5 * (
        np.einsum("pq,abcd,qabcdp->", hc, g, rdm3, optimize=True)
        + np.einsum("pq,apcd,qadc->", hc, g, Gamma, optimize=True)
        + np.einsum("pq,pbcd,qbcd->", hc, g, Gamma, optimize=True)
    )
    # TermD = 1/4 * sum conj(g[p,q,r,s])*g[p',q',r',s'] * ( D4[s,r,p',q',r',s',q,p]
    #                                                        + delta(q,q')*D3[s,r,p',s',r',p]
    #                                                        + delta(p,q')*D3[s,r,p',s',q,r']
    #                                                        + delta(q,p')*D3[s,r,q',r',s',p]
    #                                                        + delta(p,q')*delta(q,p')*Gamma[s,r,s',r']
    #                                                        + delta(p,p')*D3[s,r,q',r',q,s']
    #                                                        + delta(p,p')*delta(q,q')*Gamma[s,r,r',s'] )
    termD = 0.25 * (
        np.einsum("pqrs,abcd,srabcdqp->", gc, g, rdm4, optimize=True)
        + np.einsum("pqrs,aqcd,sradcp->", gc, g, rdm3, optimize=True)
        + np.einsum("pqrs,apcd,sradqc->", gc, g, rdm3, optimize=True)
        + np.einsum("pqrs,qbcd,srbcdp->", gc, g, rdm3, optimize=True)
        + np.einsum("pqrs,qpcd,srdc->", gc, g, Gamma, optimize=True)
        + np.einsum("pqrs,pbcd,srbcqd->", gc, g, rdm3, optimize=True)
        + np.einsum("pqrs,pqcd,srcd->", gc, g, Gamma, optimize=True)
    )
    total = termA + termB + termB.conj() + termD
    return float(total.real), 2


def coupling_strength_real(
    h1e: np.ndarray,
    g_derived: np.ndarray | None,
    D: np.ndarray,
    Gamma: np.ndarray,
    rdm3: np.ndarray | None,
    rdm4: np.ndarray | None,
    h_mask: np.ndarray,
    g_mask: np.ndarray | None,
) -> tuple[float | None, int | None]:
    """Real-only specialization of coupling_strength_complex: no conj(), and
    TermB + conj(TermB) collapses to 2*TermB (already real when every input
    is real). Deliberately a separate, independently-written implementation
    rather than the complex path fed real-dtype arrays, so that agreement
    between the two (checked extensively in
    tests/test_cluster_number_sector_search.py) is a meaningful
    cross-check, not a tautology. Same (score, tier_used) contract as
    coupling_strength_complex.
    """
    h = h1e * h_mask
    have_g = g_mask is not None and g_derived is not None and rdm3 is not None and rdm4 is not None
    g = g_derived * g_mask if have_g else None

    h_active = np.any(h)
    g_active = g is not None and np.any(g)
    if not h_active and not g_active:
        return None, None

    termA = np.einsum("pq,pr,qr->", h, h, D, optimize=True) + np.einsum(
        "pq,ab,qabp->", h, h, Gamma, optimize=True
    )

    if not g_active:
        return float(termA), 1

    termB = 0.5 * (
        np.einsum("pq,abcd,qabcdp->", h, g, rdm3, optimize=True)
        + np.einsum("pq,apcd,qadc->", h, g, Gamma, optimize=True)
        + np.einsum("pq,pbcd,qbcd->", h, g, Gamma, optimize=True)
    )
    termD = 0.25 * (
        np.einsum("pqrs,abcd,srabcdqp->", g, g, rdm4, optimize=True)
        + np.einsum("pqrs,aqcd,sradcp->", g, g, rdm3, optimize=True)
        + np.einsum("pqrs,apcd,sradqc->", g, g, rdm3, optimize=True)
        + np.einsum("pqrs,qbcd,srbcdp->", g, g, rdm3, optimize=True)
        + np.einsum("pqrs,qpcd,srdc->", g, g, Gamma, optimize=True)
        + np.einsum("pqrs,pbcd,srbcqd->", g, g, rdm3, optimize=True)
        + np.einsum("pqrs,pqcd,srcd->", g, g, Gamma, optimize=True)
    )
    total = termA + 2.0 * termB + termD
    return float(total), 2


# =============================================================================
# Step E: sector dimension (human verified pending)
# =============================================================================


def sector_dimension(label: tuple[int, ...], cluster_sizes: list[int], nelec: tuple[int, int]) -> int:
    """Exact dimension of sector `label` (number of Slater determinants with
    fixed (Nalpha, Nbeta) such that cluster K has exactly label[K] total
    (alpha+beta) electrons, for every K) via a knapsack-style DP over
    clusters -- O(K*norb^2), no enumeration. Returns a plain Python int
    (dimensions routinely exceed 2**63)."""
    Nalpha, Nbeta = nelec
    dp: dict[tuple[int, int], int] = {(0, 0): 1}
    for size, N_K in zip(cluster_sizes, label):
        per_cluster = []
        for a in range(max(0, N_K - size), min(size, N_K) + 1):
            b = N_K - a
            per_cluster.append((a, b, comb(size, a) * comb(size, b)))
        new_dp: dict[tuple[int, int], int] = {}
        for (a_cum, b_cum), ways_cum in dp.items():
            for a, b, ways in per_cluster:
                a_new, b_new = a_cum + a, b_cum + b
                if a_new > Nalpha or b_new > Nbeta:
                    continue
                key = (a_new, b_new)
                new_dp[key] = new_dp.get(key, 0) + ways_cum * ways
        dp = new_dp
    return dp.get((Nalpha, Nbeta), 0)


# =============================================================================
# Step F: orchestration (human verified pending)
# =============================================================================


def rank_relevant_sectors(
    deco: Decomposition,
    rdm_data: RDMData,
    nelec: tuple[int, int],
    config: SectorSearchConfig | None = None,
    use_real_path: bool | None = None,
) -> list[SectorRelevance]:
    """Cheaply rank candidate cluster-number sectors of `deco` by relevance
    (see module docstring for the algorithm). `rdm_data` is in the same
    reference basis deco.U is relative to (e.g. MOs); it gets rotated into
    deco's own basis internally. `nelec` must be given explicitly --
    RDMData has no nelec field, and it can't be recovered from spin-summed
    D/Gamma alone (only the total Nalpha+Nbeta could be, not the split).

    use_real_path: None (default) auto-detects from whether rdm_data's
    populated arrays are real-dtype (true for every actual pipeline in this
    codebase, since params_to_U_jax only emits real orthogonal rotations).
    """
    if config is None:
        config = SectorSearchConfig()

    norb = rdm_data.norb
    clean_partition, _cluster_map, had_ghost = normalize_cluster_family(deco.partition, norb)
    if had_ghost:
        logger.warning(
            "rank_relevant_sectors: deco.partition does not cover all orbitals; an "
            "implicit ghost cluster was added for internal bookkeeping and its "
            "coordinate is stripped from every label returned here."
        )
    cluster_sizes = [len(c) for c in clean_partition]
    K_real = len(clean_partition) - 1 if had_ghost else len(clean_partition)

    rdm_data_cur = rotate_rdm_data(rdm_data, deco.U)

    mu, Sigma = cluster_number_moments(rdm_data_cur, clean_partition)
    sigma_pinv = np.linalg.pinv(Sigma)
    main_label = main_sector_label(mu, nelec, cluster_sizes)

    if use_real_path is None:
        populated = [
            a
            for a in (
                rdm_data_cur.D,
                rdm_data_cur.Gamma,
                rdm_data_cur.h1e,
                rdm_data_cur.g2e_full,
                rdm_data_cur.rdm3,
                rdm_data_cur.rdm4,
            )
            if a is not None
        ]
        use_real_path = not any(np.iscomplexobj(a) for a in populated)

    have_tier2_source = (
        rdm_data_cur.h1e is not None
        and rdm_data_cur.g2e_full is not None
        and rdm_data_cur.rdm3 is not None
        and rdm_data_cur.rdm4 is not None
    )
    if rdm_data_cur.h1e is None:
        logger.warning(
            "rank_relevant_sectors: rdm_data has no h1e -- no candidate will get an "
            "energy_score; ranking will fall back to weight_score alone for all of them."
        )
    elif not have_tier2_source:
        logger.info(
            "rank_relevant_sectors: rdm_data lacks g2e_full/rdm3/rdm4 -- only Tier 1 "
            "(single-electron-transfer) energy scores will be available; t=2 candidates "
            "will fall back to weight_score alone."
        )

    indicator = partition_to_cluster_matrix(clean_partition, norb)
    candidates = enumerate_candidate_labels(main_label, cluster_sizes, config.max_elec_transfer)
    coupling_fn = coupling_strength_real if use_real_path else coupling_strength_complex
    g_derived = rdm_data_cur.g2e_full.transpose(0, 2, 3, 1) if have_tier2_source else None

    results: list[SectorRelevance] = []
    for label, t in candidates.items():
        delta = np.asarray(label, dtype=int) - np.asarray(main_label, dtype=int)
        weight_score = gaussian_weight(label, mu, sigma_pinv)

        energy_score: float | None = None
        energy_tier: int | None = None
        if t <= 2 and rdm_data_cur.h1e is not None:
            h_mask, g_mask = build_transfer_masks(indicator, norb, delta, need_g=have_tier2_source)
            energy_score, energy_tier = coupling_fn(
                rdm_data_cur.h1e,
                g_derived,
                rdm_data_cur.D,
                rdm_data_cur.Gamma,
                rdm_data_cur.rdm3,
                rdm_data_cur.rdm4,
                h_mask,
                g_mask,
            )

        reported_label = label[:K_real] if had_ghost else label
        results.append(
            SectorRelevance(
                label=reported_label,
                weight_score=weight_score,
                energy_score=energy_score,
                energy_tier=energy_tier,
                elec_transfer=t,
                dimension=sector_dimension(label, cluster_sizes, nelec),
            )
        )

    # Candidates with a direct energy score outrank those without one (the
    # downstream consumer is a variational diagonalization, so
    # energy-lowering potential matters more than raw weight -- see module
    # docstring); within each group, higher score first.
    results.sort(
        key=lambda r: (
            r.energy_score is not None,
            r.energy_score if r.energy_score is not None else 0.0,
            r.weight_score,
        ),
        reverse=True,
    )

    retained: list[SectorRelevance] = []
    cum_dim = 0
    for r in results:
        if config.num_sectors_to_retain is not None and len(retained) >= config.num_sectors_to_retain:
            break
        if (
            config.max_cum_dim_to_retain is not None
            and retained
            and cum_dim + r.dimension > config.max_cum_dim_to_retain
        ):
            break
        if (
            config.max_cum_dim_to_retain is not None
            and not retained
            and r.dimension > config.max_cum_dim_to_retain
        ):
            logger.warning(
                "rank_relevant_sectors: the top-ranked sector's dimension (%d) alone "
                "exceeds max_cum_dim_to_retain (%d); retaining it anyway so the result "
                "isn't empty.",
                r.dimension,
                config.max_cum_dim_to_retain,
            )
        retained.append(r)
        cum_dim += r.dimension

    return retained

# =============================================================================
# Sector analysis using rank_relevant_sectors output (for flag --K-sector-analysis) (human verified pending)
# =============================================================================

def get_relevance_ranked_K_sectors_values_energies(
    psi: np.ndarray,
    h_linop: Any,
    ref_energy: float,
    sectors: dict[tuple, list[int]],
    ranked: list[SectorRelevance],
    chemical_precision: float,
) -> tuple[list[int], list[float], list[int], bool]:
    """K-sweep for --K-sector-analysis: the relevance-ranked analogue of
    src.K_sectors_plots.get_K_sectors_values_energies's psi-weight-ranked
    K-sweep. Walks `ranked` in the order rank_relevant_sectors already put
    it in -- no re-ranking, and no re-deriving a main-sector/electron-transfer
    cap from scratch the way that sibling function does (rank_relevant_sectors's
    own config already determined which and how many candidates are in
    `ranked`).

    At each K = 1, 2, ..., len(ranked): accumulate the K highest-score
    sectors' determinant supports, project psi onto their direct sum,
    renormalize to get psi', and record <psi'|H|psi'> and the cumulative
    dimension sum(r.dimension for r in ranked[:K]). Sectors are disjoint by
    construction, so a plain index assignment (not +=) into a running buffer
    suffices, and ||psi'||^2 is non-decreasing in K.

    Stops the first time energy - ref_energy < chemical_precision (same
    non-abs comparison as get_K_sectors_values_energies), or once every
    sector in `ranked` has been retained.

    sectors: a number_and_parity_symmetry_sectors(...)-style dict keyed by
    (label, ()) -- the parity part is always an empty tuple here, unlike
    SectorRelevance.label, which is the bare label tuple alone.

    Returns (K_values, energies, retained_dimensions, chem_accuracy_reached).
    """
    K_values: list[int] = []
    energies: list[float] = []
    retained_dimensions: list[int] = []
    chem_accuracy_reached = False

    compressed_coeffs = np.zeros_like(psi, dtype="complex")
    cum_dim = 0
    K = 0
    for r in ranked:
        indices = sectors.get((r.label, ()))
        if indices is None:
            logger.warning(
                "get_relevance_ranked_K_sectors_values_energies: label %s from "
                "rank_relevant_sectors not found among this decomposition's own symmetry "
                "sectors -- skipping it (doesn't count towards K).",
                r.label,
            )
            continue
        K += 1  # K = number of sectors actually retained so far, not the loop position
        compressed_coeffs[indices] = psi[indices]
        cum_dim += r.dimension

        norm = np.linalg.norm(compressed_coeffs)
        if norm < 1e-15:
            continue
        psi_prime = compressed_coeffs / norm
        # np.vdot(psi_prime, h_linop @ psi_prime) rather than the (mathematically
        # equivalent) psi_prime.conj() @ h_linop @ psi_prime: the latter left-multiplies
        # a LinearOperator, which needs rmatvec/__rmatmul__ support that isn't guaranteed
        # -- h_linop @ psi_prime alone only ever needs the forward matvec every
        # LinearOperator provides.
        e_K = float(np.vdot(psi_prime, h_linop @ psi_prime).real)

        K_values.append(K)
        energies.append(e_K)
        retained_dimensions.append(cum_dim)

        if e_K - ref_energy < chemical_precision:
            chem_accuracy_reached = True
            break

    return K_values, energies, retained_dimensions, chem_accuracy_reached


@dataclass
class KSectorAnalysisState:
    """Accumulator threaded through main()'s trajectory loop for --K-sector-analysis."""

    hamiltonian: Any
    psi_MOs: np.ndarray
    data_label_list: list[str]
    K_values_list: list[list[int]]
    energies_list: list[list[float]]
    retained_dims_list: list[list[int]]


def _setup_K_sector_analysis(solver: Any, h1e: np.ndarray, g2e: np.ndarray, ecore: float, norb: int) -> KSectorAnalysisState:
    """One-time --K-sector-analysis setup, called once before main()'s trajectory
    loop: extracts the full CI vector (in the MO basis) from the DMRG MPS and
    builds the matrix-free Hamiltonian it will later be rotated into each
    trajectory entry's own basis for (see _run_K_sector_analysis_for_entry)."""
    import ffsim
    import pyscf.ao2mo

    mps = solver.get_mps()
    psi_MOs = solver.to_ci_vector(ket=mps)
    g2e_full = pyscf.ao2mo.restore(1, g2e, norb)
    hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor=h1e, two_body_tensor=g2e_full, constant=ecore)
    return KSectorAnalysisState(
        hamiltonian=hamiltonian, psi_MOs=psi_MOs,
        data_label_list=[], K_values_list=[], energies_list=[], retained_dims_list=[],
    )


def _run_K_sector_analysis_for_entry(
    state: KSectorAnalysisState,
    deco: Decomposition,
    ranked: list[SectorRelevance],
    norb: int,
    nelec: tuple[int, int],
    dmrg_energy: float,
) -> None:
    """Per-trajectory-entry --K-sector-analysis step, called once per `deco` in
    main()'s trajectory loop: runs the relevance-ranked K-sweep and appends its
    curve into `state` (a no-op, beyond a warning, if `ranked` is empty)."""
    if not ranked:
        logger.warning(
            f"{deco.num_clusters} clusters: rank_relevant_sectors retained no sectors -- "
            "skipping its K-sector-analysis curve."
        )
        return

    import ffsim
    from chemistry import CHEMICAL_PRECISION
    from src.cluster_number_operators import number_and_parity_symmetry_sectors

    # deco.partition always already fully covers range(norb) with no overlaps (see
    # validate_partition), so rank_relevant_sectors's own normalize_cluster_family
    # never needed to add a ghost cluster for it -- sector labels built here from
    # deco.partition directly therefore already match the label ordering `ranked`'s
    # entries use.
    cluster_matrix = partition_to_cluster_matrix(deco.partition, norb)
    sectors = number_and_parity_symmetry_sectors(cluster_matrix, [], norb, nelec)
    psi = ffsim.apply_orbital_rotation(state.psi_MOs, np.asarray(deco.U), norb, nelec)
    h_rotated = state.hamiltonian.rotated(np.asarray(deco.U))
    h_linop = ffsim.linear_operator(h_rotated, norb, nelec)
    h_psi = h_linop @ psi
    assert np.isclose(dmrg_energy, np.vdot(psi, h_psi).real, atol=1e-8)
    assert np.isclose(0, np.vdot(psi, h_psi).imag, atol=1e-8)

    K_values, energies, retained_dims, chem_accuracy_reached = (
        get_relevance_ranked_K_sectors_values_energies(
            psi, h_linop, dmrg_energy, sectors, ranked, CHEMICAL_PRECISION,
        )
    )
    logger.info(
        f"{deco.num_clusters} clusters: K-sector-analysis "
        f"{'reached' if chem_accuracy_reached else 'did not reach'} chemical accuracy "
        f"(stopped at K={K_values[-1] if K_values else 0})."
    )
    state.data_label_list.append(f"{deco.num_clusters} clusters")
    state.K_values_list.append(K_values)
    state.energies_list.append(energies)
    state.retained_dims_list.append(retained_dims)


def _finalize_K_sector_analysis(
    state: KSectorAnalysisState,
    args: argparse.Namespace,
    dmrg_energy: float,
    norb: int,
    timestamp: str,
    git_hash: str,
) -> None:
    """Saves the combined double plot for --K-sector-analysis, called once after
    main()'s trajectory loop ends (a no-op, beyond a log message, if no entry
    produced a curve)."""
    if not state.data_label_list:
        logger.info("No K-sector-analysis curves to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.K_sectors_plots import plot_energy_vs_K

    plots_dir = _plots_dir_for(args)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_energy_vs_K(
        state.data_label_list, state.K_values_list, state.energies_list, state.retained_dims_list,
        dmrg_energy,
        molecule=args.molecule, basis_set=args.basis_set, norb=norb,
        cluster_sizes="varies per curve -- see data_label",
        max_elec_transfers=args.max_elec_transfer, cost=args.cost_function,
        sectors_or_states="sectors", from_beam_search=True,
        min_child_cluster_size=args.min_child_cluster_size,
        target_num_clusters=args.target_num_clusters,
        initial_basis=args.initial_basis,
    )
    filename = f"K_sector_analysis_{args.cost_function}_{timestamp}_{git_hash}.png"
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"K-sector-analysis plot saved to {filepath}")

# =============================================================================
# CLI Interface (human verified pending)
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Argument parser mirroring cluster_number_decomposition_optimization.py's
    own create_parser() for the shared molecule/DMRG/beam-search flags (this
    module runs that same beam search to obtain a trajectory to search),
    plus new flags specific to the sector search itself."""
    parser = argparse.ArgumentParser(
        description="Cheap, non-exponential cluster-number sector relevance ranking "
        "for a beam-search decomposition's trajectory."
    )
    parser.add_argument(
        "molecule", type=str,
        help="Molecule to analyze (one of h2, h2o, n2, lih, h4_linear, h4_square, h4_rectangle), "
        "or a free-form label when --fcidump is given",
    )
    parser.add_argument(
        "basis_set", type=str, help="Basis set (e.g., sto-3g, 6-31g), or a free-form label when --fcidump is given"
    )
    parser.add_argument(
        "bond_length", type=float, help="Bond length in Angstrom (unused, but still required, when --fcidump is given)"
    )
    parser.add_argument(
        "cost_function", type=str,
        help="Cost function type for the beam search (variance, eval_eq, extremality, mixed, commutator)",
    )

    parser.add_argument("--bond-angle", type=float, default=None, help="Bond angle in degrees (for H2O)")
    parser.add_argument(
        "--fcidump", type=str, default=None,
        help="Path to an FCIDUMP file with a precomputed Hamiltonian, as in cluster_number_decomposition_optimization.py",
    )

    # DMRG parameters
    parser.add_argument("--bond-dim", type=int, default=150, help="DMRG bond dimension (default: 150)")
    parser.add_argument("--n-sweeps", type=int, default=50, help="Number of DMRG sweeps (default: 50)")

    # Cost function parameters
    parser.add_argument("--var-exponent", type=int, default=1, help="Variance exponent (default: 1)")

    # Beam search hyperparameters
    parser.add_argument("--num-decos", type=int, default=4, help="Beam width (default: 4)")
    parser.add_argument("--num-subdecos", type=int, default=4, help="Splits attempted per deco per round (default: 4)")
    parser.add_argument("--min-parent-cluster-size", type=int, default=1)
    parser.add_argument("--min-child-cluster-size", type=int, default=1)
    parser.add_argument("--no-naturalize-children", action="store_true")
    parser.add_argument("--target-num-clusters", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--no-orb-opt-in-beam-search", action="store_true")

    # Initial basis
    parser.add_argument("--initial-basis", type=str, choices=["MOs", "NatOs", "both"], default="MOs")
    parser.add_argument("--fiedler-reorder", action="store_true")

    # Polish
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--polish-maxiter", type=int, default=500)

    # Sector search (new)
    parser.add_argument(
        "--analyze-num-clusters", type=int, nargs="+", default=None,
        help="Which trajectory cluster counts to run the sector search on (default: all with >= 2 clusters)",
    )
    parser.add_argument("--num-sectors-to-retain", type=int, default=None)
    parser.add_argument("--max-cum-dim-to-retain", type=int, default=None)
    parser.add_argument("--max-elec-transfer", type=int, default=2)
    parser.add_argument(
        "--force-full-rdms", action="store_true",
        help="Extract h1e/g2e_full/rdm3/rdm4 (needed for Tier 2) directly, even if --cost-function "
        "isn't 'commutator' -- otherwise Tier 2 is only available when it is, since that's the only "
        "case the beam search's own RDM extraction populates them.",
    )
    parser.add_argument(
        "--K-sector-analysis", action="store_true",
        help="Produce one combined double plot (energy vs K, cumulative retained dimension vs K), "
        "covering every selected trajectory entry, where K is the number of top-ranked sectors "
        "retained per rank_relevant_sectors's own ranking (not psi-weight): at each K, the energy "
        "is <psi'|H|psi'> for psi' = the normalized projection of the true wavefunction onto the "
        "direct sum of the top-K sectors' determinant supports. A horizontal chemical-accuracy line "
        "(from the DMRG energy) is drawn, and each entry's curve stops once it first crosses that "
        "line. Off by default.",
    )

    # Output options
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--plots-dir", type=str, default=None, help="Plots directory (for --K-sector-analysis)")
    parser.add_argument(
        "--wavefunction-dir", type=str, default="wavefunctions", help="MPS wavefunction directory (for input and output)"
    )

    # HPC options
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads (default: 1)")
    parser.add_argument("--no-reuse", action="store_true", help="Don't reuse existing wavefunction")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def _geometry_output_subpath(args: argparse.Namespace) -> Path:
    molecule = args.molecule.lower()
    basis_set = args.basis_set.lower()
    if args.fcidump is not None:
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
    return Path("outputs_") / "cluster_number_sector_search" / _geometry_output_subpath(args)


def _plots_dir_for(args: argparse.Namespace) -> Path:
    if args.plots_dir is not None:
        return Path(args.plots_dir)
    return Path("plots") / "cluster_number_sector_search" / _geometry_output_subpath(args)


def _sector_relevance_to_json(entries: list[SectorRelevance]) -> list[dict]:
    return [
        {
            "label": list(r.label),
            "weight_score": r.weight_score,
            "energy_score": r.energy_score,
            "energy_tier": r.energy_tier,
            "elec_transfer": r.elec_transfer,
            "dimension": r.dimension,
        }
        for r in entries
    ]


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

    logger.info(f"Running beam search for {args.molecule}/{args.basis_set} to feed the sector search...")
    rdm_data, dmrg_metadata, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(
        args, force_full_rdms=args.force_full_rdms,
    )
    norb = rdm_data.norb

    cost_function_constructor = _build_cost_function_constructor(args.cost_function, args.var_exponent)
    initial_bases = _build_initial_bases(args, rdm_data, norb)

    opt_config = DecompositionOptimizerConfig(
        num_decos=args.num_decos, num_subdecos=args.num_subdecos,
        min_parent_cluster_size=args.min_parent_cluster_size,
        min_child_cluster_size=args.min_child_cluster_size,
        naturalize_children=not args.no_naturalize_children,
        target_num_clusters=args.target_num_clusters, max_rounds=args.max_rounds, maxiter=args.maxiter,
        optimize_rotation_in_beam_search=not args.no_orb_opt_in_beam_search,
    )
    trajectory = run_decomposition_optimizer(cost_function_constructor, rdm_data, opt_config, initial_bases)

    if rdm_data.h1e is None:
        # ensure rank_relevant_sectors's Tier-1 (h1e-only) energy_score is always available
        rdm_data.h1e = h1e

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

    search_config = SectorSearchConfig(
        num_sectors_to_retain=args.num_sectors_to_retain,
        max_cum_dim_to_retain=args.max_cum_dim_to_retain,
        max_elec_transfer=args.max_elec_transfer,
    )

    output_dir = _output_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    git_hash = get_git_hash()

    if args.K_sector_analysis:
        K_sector_state = _setup_K_sector_analysis(solver, h1e, g2e, ecore, norb)

    for deco in entries:
        logger.info(f"Ranking sectors for {deco.num_clusters}-cluster decomposition (sizes={[len(c) for c in deco.partition]})...")
        ranked = rank_relevant_sectors(deco, rdm_data, nelec, search_config)
        for r in ranked[:10]:
            logger.info(
                f"  label={r.label} weight={r.weight_score:.4e} energy={r.energy_score} "
                f"(tier {r.energy_tier}) t={r.elec_transfer} dim={r.dimension}"
            )

        if args.K_sector_analysis:
            _run_K_sector_analysis_for_entry(K_sector_state, deco, ranked, norb, nelec, dmrg_energy)

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
        }
        output = {"metadata": metadata, "ranked_sectors": _sector_relevance_to_json(ranked)}
        filename = f"sectors_{deco.num_clusters}clusters_{args.cost_function}_{timestamp}_{git_hash}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {filepath}")

    if args.K_sector_analysis:
        _finalize_K_sector_analysis(K_sector_state, args, dmrg_energy, norb, timestamp, git_hash)

    logger.info("Computation completed successfully!")


if __name__ == "__main__":
    main()
