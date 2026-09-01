# AI-generated

"""Tests for cluster_number_sector_search.py.

Two kinds of ground truth are used:
  - `_rdm_data_from_random_mps` (mirroring
    test_cluster_number_decomposition_optimization.py's own helper of the
    same name): a random MPS (bond_dim=5, no DMRG sweep) gives cheap,
    valid-but-not-physical REAL RDMs, used for anything that doesn't need
    to probe the delicate RDM-contraction formulas themselves (Step 0
    guard, moments, main-sector rounding, end-to-end wiring).
  - A from-scratch Fock-space construction (`_ladder_operators` +
    `_rdm_tensors_from_state`, via openfermion's jordan_wigner +
    get_sparse_operator, exactly as tests/test_dmrg_costs.py already does
    for its own exact-diagonalization cross-checks), used for
    TestCouplingStrengthGroundTruth: q(delta) = <psi|V_delta^dagger
    V_delta|psi> is an operator identity, so it must hold for ANY state
    psi -- a fully random (not even N-representable-restricted) complex
    vector is the most stringent test, and is how these formulas were
    originally derived and verified.
"""

import logging
import shutil
import sys
import tempfile
from itertools import product
from math import comb
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import openfermion as of
import pytest
import scipy.sparse.linalg

from src.dmrg_solver import Block2DMRGSolver

from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    cluster_number_stats,
    cluster_variance,
    partition_to_cluster_matrix,
)
from cluster_number_sector_search import (
    SectorRelevance,
    SectorSearchConfig,
    build_transfer_masks,
    cluster_number_moments,
    coupling_strength_complex,
    coupling_strength_real,
    enumerate_candidate_labels,
    gaussian_weight,
    main_sector_label,
    normalize_cluster_family,
    rank_relevant_sectors,
    sector_dimension,
)


def _rdm_data_from_random_mps(norb, nelec, bond_dim=5, with_34=False, seed=0):
    """RDMData built from a random MPS -- valid tensors, not a ground state.
    Mirrors test_cluster_number_decomposition_optimization.py's own helper
    of the same name."""
    tmp_dir = tempfile.mkdtemp(prefix="sector_search_test_")
    try:
        solver = Block2DMRGSolver(
            h1e=np.zeros((norb, norb)),
            g2e=np.zeros((norb, norb, norb, norb)),
            ecore=0.0,
            n_elec=nelec,
            spin=None,
            store_dir=tmp_dir,
            n_threads=1,
            save_integrals=False,
        )
        mps = solver.driver.get_random_mps(tag=f"RAND{seed}", bond_dim=bond_dim, nroots=1)

        rdm1_a, rdm1_b = solver.driver.get_1pdm(mps)
        D = rdm1_a + rdm1_b
        rdm2_aa, rdm2_ab, rdm2_bb = solver.driver.get_2pdm(mps)
        Gamma = rdm2_aa + rdm2_bb + rdm2_ab + rdm2_ab.transpose(1, 0, 3, 2)

        if not with_34:
            return RDMData(D=D, Gamma=Gamma)

        rdm3_aaa, rdm3_aab, rdm3_abb, rdm3_bbb = solver.driver.get_3pdm(mps)
        rdm3 = (
            rdm3_aaa
            + rdm3_aab + rdm3_aab.transpose(0, 2, 1, 4, 3, 5) + rdm3_aab.transpose(2, 1, 0, 5, 4, 3)
            + rdm3_abb + rdm3_abb.transpose(1, 0, 2, 3, 5, 4) + rdm3_abb.transpose(2, 1, 0, 5, 4, 3)
            + rdm3_bbb
        )
        rdm4_aaaa, rdm4_aaab, rdm4_aabb, rdm4_abbb, rdm4_bbbb = solver.driver.get_4pdm(mps)
        rdm4 = (
            rdm4_aaaa
            + rdm4_aaab
            + rdm4_aaab.transpose(0, 1, 3, 2, 5, 4, 6, 7)
            + rdm4_aaab.transpose(0, 3, 2, 1, 6, 5, 4, 7)
            + rdm4_aaab.transpose(3, 1, 2, 0, 7, 5, 6, 4)
            + rdm4_aabb
            + rdm4_aabb.transpose(0, 2, 1, 3, 4, 6, 5, 7)
            + rdm4_aabb.transpose(0, 3, 2, 1, 6, 5, 4, 7)
            + rdm4_aabb.transpose(2, 1, 0, 3, 4, 7, 6, 5)
            + rdm4_aabb.transpose(3, 1, 2, 0, 7, 5, 6, 4)
            + rdm4_aabb.transpose(2, 3, 0, 1, 6, 7, 4, 5)
            + rdm4_abbb
            + rdm4_abbb.transpose(1, 0, 2, 3, 4, 5, 7, 6)
            + rdm4_abbb.transpose(2, 1, 0, 3, 4, 7, 6, 5)
            + rdm4_abbb.transpose(3, 1, 2, 0, 7, 5, 6, 4)
            + rdm4_bbbb
        )
        rng = np.random.default_rng(seed)
        C = rng.standard_normal((norb, norb, 3))
        C = C + C.transpose(1, 0, 2)
        g2e_full = np.einsum("pqk,rsk->pqrs", C, C)
        h1e = rng.standard_normal((norb, norb))
        h1e = h1e + h1e.T

        return RDMData(D=D, Gamma=Gamma, rdm3=rdm3, rdm4=rdm4, h1e=h1e, g2e_full=g2e_full)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# From-scratch Fock-space ground truth (openfermion, matching this repo's
# own idiom in tests/test_dmrg_costs.py -- interleaved spin-orbital indexing
# 2*p+spin, spin=0 alpha, spin=1 beta)
# =============================================================================


def _so(p, spin):
    return 2 * p + spin


def _ladder_operators(n_modes):
    """cre[i], des[i] as scipy sparse matrices over the 2**n_modes Fock space."""
    cre = [of.get_sparse_operator(of.FermionOperator(((i, 1),)), n_qubits=n_modes) for i in range(n_modes)]
    des = [of.get_sparse_operator(of.FermionOperator(((i, 0),)), n_qubits=n_modes) for i in range(n_modes)]
    return cre, des


def _random_state(dim, rng, real):
    vec = rng.standard_normal(dim)
    if not real:
        vec = vec + 1j * rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def _rdm_tensors_from_state(psi, des, norb, with_34=True):
    """D, Gamma, rdm3, rdm4 (spin-summed, "nested" convention -- see
    cluster_number_sector_search.py's module docstring) computed directly
    from an explicit state vector, via precomputed a_X...a_Z|psi> vectors
    (fast: avoids rebuilding a full operator product per tensor entry).
    psi need not be N-representable/physical -- these are just matrix
    elements of a given vector, well-defined for any psi."""
    n_modes = 2 * norb
    v1 = {X: des[X] @ psi for X in range(n_modes)}
    v2 = {(X, Y): des[X] @ v1[Y] for X in range(n_modes) for Y in range(n_modes)}

    D = np.zeros((norb, norb), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            D[p, q] = sum(np.vdot(v1[_so(p, s)], v1[_so(q, s)]) for s in (0, 1))

    Gamma = np.zeros((norb,) * 4, dtype=complex)
    for p, q, r, s in np.ndindex(norb, norb, norb, norb):
        Gamma[p, q, r, s] = sum(
            np.vdot(v2[(_so(q, tt), _so(p, sg))], v2[(_so(r, tt), _so(s, sg))])
            for sg in (0, 1)
            for tt in (0, 1)
        )

    if not with_34:
        return D, Gamma, None, None

    v3 = {(X, Y, Z): des[X] @ v2[(Y, Z)] for X in range(n_modes) for Y in range(n_modes) for Z in range(n_modes)}
    v4 = {}
    for W in range(n_modes):
        for X in range(n_modes):
            for Y in range(n_modes):
                for Z in range(n_modes):
                    v4[(W, X, Y, Z)] = des[W] @ v3[(X, Y, Z)]

    rdm3 = np.zeros((norb,) * 6, dtype=complex)
    for p, q, r, s, t, u in np.ndindex(norb, norb, norb, norb, norb, norb):
        rdm3[p, q, r, s, t, u] = sum(
            np.vdot(v3[(_so(r, s3), _so(q, s2), _so(p, s1))], v3[(_so(s, s3), _so(t, s2), _so(u, s1))])
            for s1 in (0, 1)
            for s2 in (0, 1)
            for s3 in (0, 1)
        )

    rdm4 = np.zeros((norb,) * 8, dtype=complex)
    for p, q, r, s in np.ndindex(norb, norb, norb, norb):
        for t, u, v, w in np.ndindex(norb, norb, norb, norb):
            rdm4[p, q, r, s, t, u, v, w] = sum(
                np.vdot(
                    v4[(_so(s, s4), _so(r, s3), _so(q, s2), _so(p, s1))],
                    v4[(_so(t, s4), _so(u, s3), _so(v, s2), _so(w, s1))],
                )
                for s1 in (0, 1)
                for s2 in (0, 1)
                for s3 in (0, 1)
                for s4 in (0, 1)
            )

    return D, Gamma, rdm3, rdm4


def _build_v_delta(cre, des, norb, h_masked, g_masked):
    """V_delta = sum h_masked[p,q] sum_sigma cre[p,s]@des[q,s]
              + 1/2 sum g_masked[p,q,r,s] sum_{sig,tau} cre[p,sig]@cre[q,tau]@des[r,tau]@des[s,sig]
    built directly as an explicit sparse matrix -- the ground truth that
    coupling_strength_{complex,real} are checked against via
    <psi|V_delta^dagger V_delta|psi>."""
    dim = cre[0].shape[0]
    V = 0 * cre[0]
    for p, q in np.ndindex(norb, norb):
        coeff = h_masked[p, q]
        if coeff == 0:
            continue
        for s in (0, 1):
            V = V + coeff * (cre[_so(p, s)] @ des[_so(q, s)])
    if g_masked is not None:
        for p, q, r, s in np.ndindex(norb, norb, norb, norb):
            coeff = g_masked[p, q, r, s]
            if coeff == 0:
                continue
            for sig in (0, 1):
                for tau in (0, 1):
                    V = V + (0.5 * coeff) * (
                        cre[_so(p, sig)] @ cre[_so(q, tau)] @ des[_so(r, tau)] @ des[_so(s, sig)]
                    )
    return V


def _block_h_mask(norb, C_I, C_J):
    mask = np.zeros((norb, norb), dtype=bool)
    for p in C_I:
        for q in C_J:
            mask[p, q] = True
    return mask


def _block_g_mask(norb, C_I, C_J):
    mask = np.zeros((norb,) * 4, dtype=bool)
    for p in C_I:
        for q in C_I:
            for r in C_J:
                for s in C_J:
                    mask[p, q, r, s] = True
    return mask


def _random_tensor(shape, rng, real):
    arr = rng.standard_normal(shape)
    if not real:
        arr = arr + 1j * rng.standard_normal(shape)
    return arr


# =============================================================================
# Step 0
# =============================================================================


class TestNormalizeClusterFamily:
    def test_disjoint_partition_unchanged(self):
        partition = [[0, 1], [2], [3, 4]]
        clean, cluster_map, had_ghost = normalize_cluster_family(partition, norb=5)
        assert clean == [[0, 1], [2], [3, 4]]
        assert cluster_map == [0, 1, 2]
        assert had_ghost is False

    def test_overlapping_clusters_merged_and_logged(self, caplog):
        partition = [[0, 1], [1, 2], [3]]
        with caplog.at_level(logging.WARNING):
            clean, cluster_map, had_ghost = normalize_cluster_family(partition, norb=4)
        assert clean == [[0, 1, 2], [3]]
        assert cluster_map[0] == cluster_map[1]
        assert cluster_map[2] != cluster_map[0]
        assert had_ghost is False
        assert any("merged" in rec.message for rec in caplog.records)

    def test_missing_coverage_gets_ghost_cluster_and_logged(self, caplog):
        partition = [[0, 1]]
        with caplog.at_level(logging.INFO):
            clean, cluster_map, had_ghost = normalize_cluster_family(partition, norb=4)
        assert clean == [[0, 1], [2, 3]]
        assert had_ghost is True
        assert any("ghost" in rec.message for rec in caplog.records)

    def test_overlap_and_missing_coverage_together(self):
        partition = [[0, 1], [1, 2]]
        clean, cluster_map, had_ghost = normalize_cluster_family(partition, norb=5)
        assert clean == [[0, 1, 2], [3, 4]]
        assert had_ghost is True

    def test_transitive_merge_of_three_clusters(self):
        # 0-1 share orbital 2; 1-2 share orbital 5 -> all three must merge
        partition = [[0, 2], [2, 5, 6], [5, 9]]
        clean, cluster_map, had_ghost = normalize_cluster_family(partition, norb=10)
        assert len(clean) == 2  # merged cluster + ghost
        assert cluster_map[0] == cluster_map[1] == cluster_map[2]

    def test_empty_cluster_rejected(self):
        with pytest.raises(ValueError):
            normalize_cluster_family([[0, 1], []], norb=3)


# =============================================================================
# Step A
# =============================================================================


class TestClusterNumberMoments:
    @pytest.fixture(scope="class")
    def rdm_data(self):
        return _rdm_data_from_random_mps(norb=5, nelec=(3, 2), bond_dim=5, seed=1)

    def test_mu_matches_brute_force_1rdm_sum(self, rdm_data):
        partition = [[0, 1], [2], [3, 4]]
        mu, _ = cluster_number_moments(rdm_data, partition)
        expected = [np.diag(rdm_data.D)[c].sum().real for c in partition]
        assert np.allclose(mu, expected, atol=1e-10)

    def test_sigma_diagonal_matches_cluster_variance(self, rdm_data):
        partition = [[0, 1], [2], [3, 4]]
        n1, n2 = cluster_number_stats(rdm_data, [list(range(rdm_data.norb))])
        _, Sigma = cluster_number_moments(rdm_data, partition)
        for i, cluster in enumerate(partition):
            assert Sigma[i, i] == pytest.approx(cluster_variance(n1, n2, cluster), abs=1e-10)

    def test_sigma_is_symmetric(self, rdm_data):
        partition = [[0, 1], [2], [3, 4]]
        _, Sigma = cluster_number_moments(rdm_data, partition)
        assert np.allclose(Sigma, Sigma.T, atol=1e-10)

    def test_sigma_row_sums_zero(self, rdm_data):
        # total electron number has exactly zero variance -- Sigma is exactly
        # singular along the all-ones direction (DMRG conserves N exactly)
        partition = [[0, 1], [2], [3, 4]]
        _, Sigma = cluster_number_moments(rdm_data, partition)
        assert np.allclose(Sigma.sum(axis=1), 0.0, atol=1e-8)


# =============================================================================
# Step B
# =============================================================================


class TestMainSectorLabel:
    def test_rounds_to_nearest_when_already_near_integer(self):
        label = main_sector_label(np.array([2.001, 1.999, 3.0]), nelec=(4, 3), cluster_sizes=[3, 3, 3])
        assert label == (2, 2, 3)

    def test_largest_remainder_sums_exactly(self):
        # naive per-cluster round() gives (0,0,3) or (1,1,3) depending on the
        # 0.5 tie -- neither sums to nelec's 4 unless the remainder logic fires
        label = main_sector_label(np.array([0.5, 0.5, 3.0]), nelec=(2, 2), cluster_sizes=[2, 2, 3])
        assert sum(label) == 4
        assert label[0] in (0, 1) and label[1] in (0, 1) and label[0] + label[1] == 1
        assert label[2] == 3

    def test_respects_capacity_bounds(self):
        label = main_sector_label(np.array([5.0, 0.0]), nelec=(4, 1), cluster_sizes=[2, 3])
        assert label[0] <= 4
        assert sum(label) == 5

    def test_negative_remainder_case(self):
        # sum(floor(mu)) can exceed N_tot when several clusters round up
        # from just below an integer; remainder < 0 must also be handled
        label = main_sector_label(np.array([1.6, 1.6, 0.8]), nelec=(2, 2), cluster_sizes=[2, 2, 2])
        assert sum(label) == 4


# =============================================================================
# Gaussian weight
# =============================================================================


class TestGaussianWeight:
    # gaussian_weight returns the raw log-weight (-1/2 * Mahalanobis-distance^2),
    # not exp(...) of it -- see gaussian_weight's own docstring for why (exp()
    # underflows to an uninformative exact 0.0 for essentially every non-main
    # candidate in practice). Peak value is now 0.0 (attained at label==mu),
    # not 1.0.
    def test_peaks_at_mu(self):
        mu = np.array([2.0, 1.0])
        sigma_pinv = np.linalg.pinv(np.array([[1.0, 0.3], [0.3, 1.0]]))
        assert gaussian_weight((2, 1), mu, sigma_pinv) == pytest.approx(0.0, abs=1e-12)
        assert gaussian_weight((3, 0), mu, sigma_pinv) < gaussian_weight((2, 1), mu, sigma_pinv)

    def test_handles_singular_sigma_via_pinv(self):
        # singular along the all-ones direction, as Sigma always is in practice
        Sigma = np.array([[1.0, -1.0], [-1.0, 1.0]])
        sigma_pinv = np.linalg.pinv(Sigma)
        mu = np.array([2.0, 1.0])
        w_null_dir = gaussian_weight((3, 2), mu, sigma_pinv)  # displacement (1,1): along the null direction
        w_other_dir = gaussian_weight((3, 0), mu, sigma_pinv)  # displacement (1,-1): orthogonal to it
        assert np.isfinite(w_null_dir) and np.isfinite(w_other_dir)
        assert w_null_dir == pytest.approx(0.0, abs=1e-10)  # unpenalized along the null direction
        assert w_other_dir < w_null_dir

    def test_underflow_case_no_longer_collapses_to_a_single_value(self):
        # Regression test for the actual bug: a well-converged state's tiny
        # cluster-number variance used to make exp(...) underflow to 0.0 for
        # every candidate beyond the main label, making them indistinguishable.
        # Reproduce that regime directly (tiny variance -> huge Sigma^+
        # eigenvalues) and confirm distinct integer displacements still give
        # distinct, ordered scores.
        mu = np.array([2.0, 2.0])
        Sigma = np.array([[1.3e-4, -1.3e-4], [-1.3e-4, 1.3e-4]])
        sigma_pinv = np.linalg.pinv(Sigma)
        w0 = gaussian_weight((2, 2), mu, sigma_pinv)  # t=0
        w1 = gaussian_weight((1, 3), mu, sigma_pinv)  # t=1
        w2 = gaussian_weight((0, 4), mu, sigma_pinv)  # t=2
        # In the old exp() version these would all have been exactly 0.0
        # (confirmed: raw exponents here are on the order of -3.7e3/-1.5e4,
        # far below float64's underflow floor of about -745) except w0.
        assert w0 > w1 > w2
        assert all(np.isfinite(w) for w in (w0, w1, w2))


# =============================================================================
# Step C
# =============================================================================


class TestEnumerateCandidateLabels:
    def test_includes_main_label_at_t0(self):
        candidates = enumerate_candidate_labels((2, 1, 1), [2, 2, 2], max_elec_transfer=2)
        assert candidates[(2, 1, 1)] == 0

    def test_respects_max_elec_transfer_radius(self):
        candidates = enumerate_candidate_labels((2, 1, 1), [3, 3, 3], max_elec_transfer=1)
        assert all(t <= 1 for t in candidates.values())
        assert (2, 1, 1) in candidates
        assert (3, 0, 1) in candidates  # reachable in one hop 1<-0? check below instead
        assert (1, 2, 1) in candidates  # one electron 0->1

    def test_reported_t_equals_transportation_distance(self):
        main_label = (2, 1, 1)
        candidates = enumerate_candidate_labels(main_label, [4, 4, 4], max_elec_transfer=3)
        for label, t in candidates.items():
            expected_t = sum(abs(a - b) for a, b in zip(label, main_label)) // 2
            assert t == expected_t

    def test_respects_capacity_bounds(self):
        # cluster 0 has size 1 (capacity 2); main label already saturates it
        candidates = enumerate_candidate_labels((2, 0), [1, 3], max_elec_transfer=2)
        assert all(0 <= label[0] <= 2 for label in candidates)
        assert all(0 <= label[1] <= 6 for label in candidates)

    def test_no_duplicate_labels(self):
        candidates = enumerate_candidate_labels((1, 1, 1, 1), [2, 2, 2, 2], max_elec_transfer=2)
        assert len(candidates) == len(set(candidates))  # dict keys are already unique; sanity check anyway


# =============================================================================
# Mask building
# =============================================================================


class TestBuildTransferMasks:
    def test_h_mask_matches_two_cluster_block_for_t1(self):
        partition = [[0, 1], [2], [3, 4]]
        indicator = partition_to_cluster_matrix(partition, norb=5)
        delta = np.array([1, -1, 0])  # one electron: cluster 1 -> cluster 0
        h_mask, _ = build_transfer_masks(indicator, norb=5, delta=delta, need_g=False)
        expected = np.zeros((5, 5), dtype=bool)
        for p in (0, 1):
            expected[p, 2] = True
        assert np.array_equal(h_mask, expected)

    def test_g_mask_zero_for_t_geq_3(self):
        partition = [[0], [1], [2], [3]]
        indicator = partition_to_cluster_matrix(partition, norb=4)
        delta = np.array([1, 1, 1, -3])  # t = 3
        h_mask, g_mask = build_transfer_masks(indicator, norb=4, delta=delta, need_g=True)
        assert not np.any(h_mask)
        assert not np.any(g_mask)

    def test_g_mask_includes_density_assisted_and_pair_hop_patterns(self):
        partition = [[0], [1], [2]]  # C0, C1, C2 singleton clusters
        indicator = partition_to_cluster_matrix(partition, norb=3)
        # t=1, density-assisted hop C1 -> C0 mediated by spectator C2:
        # p=0 (C0, created), q=r=2 (C2, created+destroyed, net 0), s=1 (C1, destroyed)
        delta_t1 = np.array([1, -1, 0])
        _, g_mask_t1 = build_transfer_masks(indicator, norb=3, delta=delta_t1, need_g=True)
        assert g_mask_t1[0, 2, 2, 1]
        # t=2, pair hop: both electrons from C1 to C0
        delta_t2 = np.array([2, -2, 0])
        _, g_mask_t2 = build_transfer_masks(indicator, norb=3, delta=delta_t2, need_g=True)
        assert g_mask_t2[0, 0, 1, 1]


# =============================================================================
# Step D -- coupling strength, ground-truth cross-check
# =============================================================================

_CONFIGS = {
    "single_orbital_clusters": dict(norb=2, C_I=[0], C_J=[1]),
    "multi_orbital_clusters": dict(norb=3, C_I=[0, 1], C_J=[2]),
}


class TestCouplingStrengthGroundTruth:
    """Cross-checks coupling_strength_{complex,real} against
    <psi|V_delta^dagger V_delta|psi> built directly from Jordan-Wigner
    operators, independent of everything in cluster_number_sector_search.py
    itself."""

    @staticmethod
    def _run(config_name, real, tier2, seed):
        cfg = _CONFIGS[config_name]
        norb, C_I, C_J = cfg["norb"], cfg["C_I"], cfg["C_J"]
        rng = np.random.default_rng(seed)
        cre, des = _ladder_operators(2 * norb)
        psi = _random_state(2 ** (2 * norb), rng, real=real)
        D, Gamma, rdm3, rdm4 = _rdm_tensors_from_state(psi, des, norb, with_34=tier2)

        h1e = np.zeros((norb, norb), dtype=D.dtype)
        for p in C_I:
            for q in C_J:
                h1e[p, q] = _random_tensor((), rng, real)

        g_derived = None
        if tier2:
            g_derived = np.zeros((norb,) * 4, dtype=D.dtype)
            for p in C_I:
                for q in C_I:
                    for r in C_J:
                        for s in C_J:
                            g_derived[p, q, r, s] = _random_tensor((), rng, real)

        h_mask = _block_h_mask(norb, C_I, C_J)
        g_mask = _block_g_mask(norb, C_I, C_J) if tier2 else None

        V = _build_v_delta(cre, des, norb, h1e * h_mask, g_derived * g_mask if tier2 else None)
        Vpsi = V @ psi
        direct = complex(np.vdot(Vpsi, Vpsi))  # <psi|V^dagger V|psi> = <V psi|V psi>, NOT <psi|V|psi>

        coupling_fn = coupling_strength_real if real else coupling_strength_complex
        h1e_in = h1e.real if real else h1e
        g_in = (g_derived.real if real else g_derived) if tier2 else None
        D_in = D.real if real else D
        Gamma_in = Gamma.real if real else Gamma
        rdm3_in = (rdm3.real if real else rdm3) if tier2 else None
        rdm4_in = (rdm4.real if real else rdm4) if tier2 else None

        score, tier = coupling_fn(h1e_in, g_in, D_in, Gamma_in, rdm3_in, rdm4_in, h_mask, g_mask)
        return direct, score, tier

    @pytest.mark.parametrize("config_name", list(_CONFIGS))
    @pytest.mark.parametrize("real", [False, True])
    def test_tier1_matches_ground_truth(self, config_name, real):
        for seed in range(3):
            direct, score, tier = self._run(config_name, real=real, tier2=False, seed=seed)
            assert tier == 1
            assert score == pytest.approx(direct.real, abs=1e-9)
            assert abs(direct.imag) < 1e-9  # <V^dagger V> is a norm-squared: manifestly real

    @pytest.mark.parametrize("config_name", list(_CONFIGS))
    @pytest.mark.parametrize("real", [False, True])
    def test_tier2_matches_ground_truth(self, config_name, real):
        for seed in range(3):
            direct, score, tier = self._run(config_name, real=real, tier2=True, seed=seed)
            assert tier == 2
            assert score == pytest.approx(direct.real, abs=1e-7)
            assert abs(direct.imag) < 1e-7

    def test_real_and_complex_paths_agree_on_real_input(self):
        # the explicit agreement check: same real-valued data fed to BOTH
        # independently-written implementations must match each other, not
        # just each match ground truth separately
        cfg = _CONFIGS["multi_orbital_clusters"]
        norb, C_I, C_J = cfg["norb"], cfg["C_I"], cfg["C_J"]
        rng = np.random.default_rng(7)
        cre, des = _ladder_operators(2 * norb)
        psi = _random_state(2 ** (2 * norb), rng, real=True)
        D, Gamma, rdm3, rdm4 = _rdm_tensors_from_state(psi, des, norb, with_34=True)
        D, Gamma, rdm3, rdm4 = D.real, Gamma.real, rdm3.real, rdm4.real

        h1e = rng.standard_normal((norb, norb))
        g_derived = rng.standard_normal((norb,) * 4)
        h_mask = _block_h_mask(norb, C_I, C_J)
        g_mask = _block_g_mask(norb, C_I, C_J)

        score_real, tier_real = coupling_strength_real(h1e, g_derived, D, Gamma, rdm3, rdm4, h_mask, g_mask)
        score_complex, tier_complex = coupling_strength_complex(
            h1e.astype(complex), g_derived.astype(complex), D.astype(complex), Gamma.astype(complex),
            rdm3.astype(complex), rdm4.astype(complex), h_mask, g_mask,
        )
        assert tier_real == tier_complex == 2
        assert score_real == pytest.approx(score_complex, abs=1e-10)

    def test_tier2_falls_back_to_tier1_when_g_data_missing(self):
        cfg = _CONFIGS["multi_orbital_clusters"]
        norb, C_I, C_J = cfg["norb"], cfg["C_I"], cfg["C_J"]
        rng = np.random.default_rng(3)
        cre, des = _ladder_operators(2 * norb)
        psi = _random_state(2 ** (2 * norb), rng, real=True)
        D, Gamma, _, _ = _rdm_tensors_from_state(psi, des, norb, with_34=False)
        D, Gamma = D.real, Gamma.real
        h1e = rng.standard_normal((norb, norb))
        h_mask = _block_h_mask(norb, C_I, C_J)

        score, tier = coupling_strength_real(h1e, None, D, Gamma, None, None, h_mask, None)
        assert tier == 1
        assert score is not None

    def test_result_is_real_and_nonnegative(self):
        # q(delta) = ||V_delta psi||^2 -- must be real and >= 0 regardless
        # of h/g being complex, even though nothing in the formula enforces
        # this by an explicit .real/clip -- it's a structural consequence
        for config_name in _CONFIGS:
            for seed in range(2):
                _, score, _ = self._run(config_name, real=False, tier2=True, seed=seed)
                assert score >= -1e-9  # allow tiny negative from floating-point noise at 0


# =============================================================================
# Step E
# =============================================================================


class TestSectorDimension:
    def test_matches_brute_force_enumeration_tiny_system(self):
        # 2 clusters of size 2 each (4 spin-orbitals total per cluster
        # capacity 4); brute-force over all (a,b) alpha/beta occupations
        cluster_sizes = [2, 2]
        nelec = (2, 1)
        for label in np.ndindex(5, 5):  # each N_K in [0,4]
            expected = 0
            for a0 in range(3):
                for b0 in range(3):
                    if a0 + b0 != label[0]:
                        continue
                    a1, b1 = nelec[0] - a0, nelec[1] - b0
                    if not (0 <= a1 <= 2 and 0 <= b1 <= 2 and a1 + b1 == label[1]):
                        continue
                    expected += comb(2, a0) * comb(2, b0) * comb(2, a1) * comb(2, b1)
            assert sector_dimension(label, cluster_sizes, nelec) == expected

    def test_sums_to_full_hilbert_space_dimension_over_all_labels(self):
        cluster_sizes = [2, 3]
        nelec = (3, 2)
        total = sum(
            sector_dimension((n0, n1), cluster_sizes, nelec)
            for n0 in range(5)
            for n1 in range(7)
        )
        expected = comb(5, 3) * comb(5, 2)
        assert total == expected

    def test_returns_python_int_not_numpy_scalar(self):
        result = sector_dimension((3, 2), [3, 3], (3, 2))
        assert isinstance(result, int)
        assert not isinstance(result, np.integer)


# =============================================================================
# Step F -- integration
# =============================================================================


class TestRankRelevantSectors:
    @pytest.fixture(scope="class")
    def rdm_data(self):
        return _rdm_data_from_random_mps(norb=4, nelec=(2, 2), bond_dim=5, with_34=True, seed=2)

    @pytest.fixture
    def deco(self):
        return Decomposition(partition=[[0, 1], [2, 3]], U=np.eye(4), cost=0.0)

    def test_respects_num_sectors_to_retain(self, deco, rdm_data):
        config = SectorSearchConfig(num_sectors_to_retain=2, max_elec_transfer=2)
        results = rank_relevant_sectors(deco, rdm_data, nelec=(2, 2), config=config)
        assert len(results) <= 2

    def test_respects_max_cum_dim_to_retain(self, deco, rdm_data):
        unrestricted = rank_relevant_sectors(deco, rdm_data, nelec=(2, 2), config=SectorSearchConfig(max_elec_transfer=2))
        cap = unrestricted[0].dimension  # only the top sector should fit
        results = rank_relevant_sectors(
            deco, rdm_data, nelec=(2, 2), config=SectorSearchConfig(max_cum_dim_to_retain=cap, max_elec_transfer=2)
        )
        assert sum(r.dimension for r in results) <= cap or len(results) == 1

    def test_energy_scored_candidates_outrank_weight_only_candidates(self, deco, rdm_data):
        config = SectorSearchConfig(max_elec_transfer=3)  # includes t=3, which never gets an energy score
        results = rank_relevant_sectors(deco, rdm_data, nelec=(2, 2), config=config)
        scored = [r for r in results if r.energy_score is not None]
        unscored = [r for r in results if r.energy_score is None]
        if scored and unscored:
            first_unscored_rank = results.index(unscored[0])
            last_scored_rank = max(results.index(r) for r in scored)
            assert last_scored_rank < first_unscored_rank

    def test_end_to_end_on_synthetic_decomposition(self, deco, rdm_data):
        results = rank_relevant_sectors(deco, rdm_data, nelec=(2, 2))
        assert len(results) > 0
        assert all(isinstance(r, SectorRelevance) for r in results)
        assert all(len(r.label) == 2 for r in results)  # 2 clusters, no ghost
        assert all(r.dimension > 0 for r in results)
        # descending order (by the documented sort key)
        scores = [(r.energy_score is not None, r.energy_score or 0.0, r.weight_score) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_handles_overlapping_or_incomplete_partition(self, rdm_data):
        messy_deco = Decomposition(partition=[[0, 1], [1, 2]], U=np.eye(4), cost=0.0)  # overlap + orbital 3 uncovered
        results = rank_relevant_sectors(messy_deco, rdm_data, nelec=(2, 2))
        assert len(results) > 0
        # merged partition has 1 real cluster ([0,1,2]) + a ghost for {3};
        # only the real cluster's coordinate should appear in reported labels
        assert all(len(r.label) == 1 for r in results)


# =============================================================================
# =============================================================================
# SECOND, FULLY INDEPENDENT verification suite for the energy score q(delta) (human verified pending)
#
# Everything below is written from scratch: it does NOT call any helper
# function defined above in this file (not _ladder_operators, not
# _rdm_tensors_from_state, not _build_v_delta, nothing). It re-derives its
# own from-scratch Jordan-Wigner Fock-space machinery, independently, and
# uses it to compute two numbers for many sampled scenarios:
#
#   (1) brute_force_energy_score(...): the *definitional* ground truth
#       ||proj_{N'} H |psi>||^2, computed the most literal way possible --
#       apply the full many-body Hamiltonian H to psi, then zero out every
#       component of the result that does not sit in the target sector N',
#       then take the squared norm of what's left. This function never
#       touches an RDM of any order; it only needs H and psi.
#
#   (2) coupling_strength_complex / coupling_strength_real (the actual
#       functions under test in cluster_number_sector_search.py): the cheap,
#       RDM-contraction-based estimate of the same quantity, which is only
#       equal to (1) under the "psi is concentrated in sector N" assumption
#       -- an assumption that is EXACTLY satisfied here, since psi is built
#       by construction to have ALL of its weight in sector N (not just
#       approximately concentrated there), so the two numbers must match to
#       numerical precision, with no wiggle room from that approximation.
#
# Why build psi and H completely independently of the module under test,
# rather than reusing e.g. build_transfer_masks to construct H directly in
# "already masked" form? Because that would only test whether
# coupling_strength_* is self-consistent with its own masking helper -- it
# would never catch a bug in the *physics* (wrong sign, wrong index, wrong
# RDM slice) of coupling_strength_* itself, since both sides of the
# comparison would be built from the same masked pieces. Building H from the
# FULL, unmasked, unrestricted h/g tensors and only ever masking on the
# coupling_strength_* side (via build_transfer_masks, which is production
# code, not a test helper, and is exercised on its own in
# TestBuildTransferMasks above) means the brute-force side is a genuinely
# independent oracle: it doesn't know or care which piece of H is
# "supposed" to matter for a given delta, it just applies the WHOLE H and
# lets the projection onto N' sort out which piece actually contributed.
#
# Deliberate choice made after investigation (see the conversation this
# suite was written in, or ask if you want the rationale restated): psi and
# H live in the FULL, unrestricted 2**(2*norb) Fock space (every particle
# number sector at once, spin-orbital index 2*p+spin), not in ffsim's
# fixed-(Nalpha,Nbeta) CI-vector convention. ffsim's own RDM utility
# (ffsim.rdms) does not support rank 3 or 4 ("NotImplementedError:
# Computing the rank 3/4 reduced density matrix is currently not
# supported" -- checked directly against the installed version), and the
# two natural guesses at bridging ffsim's internal CI-address/sign
# convention into an independent Jordan-Wigner Fock space (interleaved
# 2*p+spin ordering, and block alpha-then-beta ordering) both failed a
# direct numerical cross-check against ffsim.rdms's own rank-1/2 output.
# Rather than depend on an unresolved convention, everything here is kept
# self-consistent within one from-scratch construction, exactly as
# literally described in the task ("a full-Fock-space vector psi", "a
# full-Fock-space Hamiltonian H").
# =============================================================================


def _bf_so(p: int, spin: int) -> int:
    """Spin-orbital index for spatial orbital p, spin in {0=alpha, 1=beta}.
    This is the ONLY place the bit layout of a full-Fock-space basis index
    is defined for this second suite; every other function below goes
    through this one function so the convention only has to be stated once."""
    return 2 * p + spin


def _bf_ladder_operators(n_modes: int):
    """Build creation/annihilation operators cre[i], des[i] (i = spin-orbital
    index, 0..n_modes-1) as explicit scipy sparse matrices over the full
    2**n_modes-dimensional Fock space, via OpenFermion's own
    Jordan-Wigner transform (of.jordan_wigner) -- i.e. we are NOT hand-
    rolling the fermionic sign bookkeeping ourselves; we trust OpenFermion's
    implementation of {a_i, a_j^dagger} = delta_ij (a widely used, heavily
    tested library primitive), and use it completely independently of the
    OTHER Jordan-Wigner helper earlier in this file (which happens to make
    the same library call, but that's the only thing shared -- no state,
    no functions, no imports between the two)."""
    cre = [of.get_sparse_operator(of.FermionOperator(((i, 1),)), n_qubits=n_modes) for i in range(n_modes)]
    des = [of.get_sparse_operator(of.FermionOperator(((i, 0),)), n_qubits=n_modes) for i in range(n_modes)]
    return cre, des


def _bf_bit_position(p: int, spin: int, norb: int) -> int:
    """The actual bit position, in a basis-index integer, that corresponds
    to spin-orbital (p, spin) -- NOT the same thing as the mode LABEL
    _bf_so(p, spin) passed to cre[.]/des[.].

    Verified directly (not assumed): of.get_sparse_operator labels its
    modes 0..n_modes-1 the natural way for building cre[i]/des[i], but its
    internal bit layout is big-endian relative to that labeling -- mode 0
    ends up as the MOST significant bit, mode n_modes-1 as the least
    significant one. Checked with cre[mode] applied to the vacuum state for
    every mode of an 8-mode system: cre[mode]|vac> always lands on basis
    index 2**(n_modes-1-mode), never 2**mode. _bf_cluster_occupation is the
    only place in this suite that inspects basis-index bits directly (every
    other function only ever composes cre[.]/des[.] as operators, or takes
    inner products of the resulting vectors -- operations that are
    self-consistent regardless of which physical bit a mode label maps to),
    so this is the one place that mapping has to be stated correctly."""
    n_modes = 2 * norb
    return n_modes - 1 - _bf_so(p, spin)


def _bf_cluster_occupation(basis_index: int, partition: list[list[int]], norb: int) -> tuple[int, ...]:
    """Decode ONE basis state (an integer whose bit _bf_bit_position(p, spin,
    norb) is 1 iff that spin-orbital is occupied in this determinant) into
    its per-cluster occupation-number label (N_0, ..., N_{K-1}): for each
    cluster, just count how many of its 2*|cluster| spin-orbital bits are
    set. This is the single, central piece of bookkeeping that everything
    else in this suite (building psi restricted to a sector, and projecting
    H|psi> onto a target sector) is built on top of."""
    label = []
    for cluster in partition:
        count = 0
        for p in cluster:
            if (basis_index >> _bf_bit_position(p, 0, norb)) & 1:
                count += 1
            if (basis_index >> _bf_bit_position(p, 1, norb)) & 1:
                count += 1
        label.append(count)
    return tuple(label)


def _bf_build_hamiltonian_sparse(cre, des, norb: int, h1e: np.ndarray, g2e_full: np.ndarray):
    """Build the full many-body Hamiltonian H = H_1 + H_2 as an explicit
    sparse matrix over the full Fock space, directly from the 1-body matrix
    h1e and the 2-body tensor g2e_full (chemist-notation two-electron
    integrals g2e_full[p,s,q,r] = (ps|qr), exactly RDMData.g2e_full's own
    convention -- h1e/g2e_full here are NOT pre-masked to any cluster
    pattern; the whole point of this suite is that coupling_strength_*
    itself is responsible for picking out, via masking, only the piece of
    this full H that matters for a given transfer):

        H_1 = sum_{p,q} h1e[p,q] * sum_sigma  a^dagger_{p,sigma} a_{q,sigma}

        H_2 = 1/2 * sum_{p,q,r,s} g2e_full[p,s,q,r] *
                sum_{sigma,tau}  a^dagger_{p,sigma} a^dagger_{q,tau} a_{r,tau} a_{s,sigma}

    This is exactly the second-quantized convention already pinned down and
    cross-checked (independently, three separate ways) earlier in this
    project: the LaTeX docstring in src/cluster_number_operators.py, the
    validated RDM/energy cross-check comment in
    tests/test_cluster_number_operators.py, and the Hamiltonian-building
    code in tests/test_dmrg_costs.py (build_h_sparse) -- see
    cluster_number_sector_search.py's own module docstring for how
    coupling_strength_* uses the SAME convention internally (via
    g_derived = g2e_full.transpose(0, 2, 3, 1)).
    """
    H = 0 * cre[0]  # zero sparse matrix, correct shape/dtype, without hand-picking a constructor

    # H_1: one-body hopping term.
    for p in range(norb):
        for q in range(norb):
            coeff = h1e[p, q]
            if coeff == 0:
                continue
            for sigma in (0, 1):
                H = H + coeff * (cre[_bf_so(p, sigma)] @ des[_bf_so(q, sigma)])

    # H_2: two-body (Coulomb) term.
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    coeff = g2e_full[p, s, q, r]
                    if coeff == 0:
                        continue
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            H = H + (0.5 * coeff) * (
                                cre[_bf_so(p, sigma)] @ cre[_bf_so(q, tau)]
                                @ des[_bf_so(r, tau)] @ des[_bf_so(s, sigma)]
                            )
    return H


def _bf_as_linear_operator(H_sparse) -> scipy.sparse.linalg.LinearOperator:
    """Wrap the explicit sparse Hamiltonian matrix as a genuine
    scipy.sparse.linalg.LinearOperator -- the type brute_force_energy_score
    is specified to accept, and the same base type ffsim.linear_operator
    itself returns in production code, so this is a faithful stand-in for
    "a full-Fock-space Hamiltonian H as a LinearOperator" even though it is
    built by completely different (and, for a small system, exactly
    equivalent) means."""
    return scipy.sparse.linalg.LinearOperator(
        shape=H_sparse.shape, dtype=complex, matvec=lambda v: H_sparse @ v
    )


def brute_force_energy_score(
    norb: int,
    partition: list[list[int]],
    N: tuple[int, ...],
    N_prime: tuple[int, ...],
    psi: np.ndarray,
    H: scipy.sparse.linalg.LinearOperator,
) -> float:
    """THE GROUND-TRUTH FUNCTION requested for this suite.

    Computes ||proj_{N'} H |psi>||^2 by brute force, i.e. by literally doing
    what that expression says rather than via any RDM/kernel shortcut:

      1. Apply H to psi (a single LinearOperator matvec).
      2. Enumerate every basis state of the full 2**(2*norb)-dimensional
         Fock space, decode its cluster-occupation label, and zero out every
         entry of the result from step 1 whose label is not exactly N'.
         (This is the "proj_{N'}" -- projection onto the sector-N' subspace
         -- spelled out one basis state at a time, which is only feasible
         because norb is kept small in this suite; that exponential cost is
         exactly what cluster_number_sector_search.py exists to avoid in
         the actual algorithm, and exactly why this function is only ever
         used here, as a slow-but-certainly-correct test oracle.)
      3. Return the squared norm of what remains.

    N is not actually needed to COMPUTE the projection (only N' is), but is
    used to defensively verify the caller's contract -- "psi is guaranteed
    to be in the sector labelled by N" -- since silently getting that wrong
    would make every other check in this suite meaningless.
    """
    dim = 2 ** (2 * norb)
    N = tuple(N)
    N_prime = tuple(N_prime)

    # Defensive check of the stated precondition: every nonzero amplitude of
    # psi must sit on a basis state whose cluster occupation is exactly N.
    for basis_index in range(dim):
        if abs(psi[basis_index]) > 1e-12:
            actual_label = _bf_cluster_occupation(basis_index, partition, norb)
            if actual_label != N:
                raise ValueError(
                    f"brute_force_energy_score: psi has nonzero amplitude on basis state "
                    f"{basis_index}, whose cluster occupation is {actual_label}, not the "
                    f"claimed sector N={N}."
                )

    H_psi = H @ psi  # step 1

    projected = np.zeros(dim, dtype=complex)  # step 2
    for basis_index in range(dim):
        if _bf_cluster_occupation(basis_index, partition, norb) == N_prime:
            projected[basis_index] = H_psi[basis_index]

    return float(np.vdot(projected, projected).real)  # step 3; .real discards O(1e-16) noise only


def _bf_rdm_tensors_from_state(psi: np.ndarray, des, norb: int):
    """Compute D, Gamma, rdm3, rdm4 (spin-summed, in the SAME "nested"
    convention coupling_strength_* expects -- see
    cluster_number_sector_search.py's module docstring) directly from the
    full-Fock-space state vector psi, via annihilation-operator chains
    applied to psi ONCE and reused (precomputing a_X|psi>, a_X a_Y|psi>,
    etc. is far cheaper than rebuilding a fresh operator product for every
    single tensor entry, which is what makes this "brute-force in Hilbert
    space size" approach still tractable at the small norb used here).

    This duplicates the *idea* used earlier in this file
    (_rdm_tensors_from_state), but is a separate, independently-written
    implementation -- no code or state is shared between the two.
    """
    n_modes = 2 * norb

    # v1[X] = a_X |psi>;  v2[(X,Y)] = a_X a_Y |psi>;  v3, v4 likewise, one
    # more annihilation operator applied each time.
    v1 = {X: des[X] @ psi for X in range(n_modes)}
    v2 = {(X, Y): des[X] @ v1[Y] for X in range(n_modes) for Y in range(n_modes)}
    v3 = {(X, Y, Z): des[X] @ v2[(Y, Z)] for X in range(n_modes) for Y in range(n_modes) for Z in range(n_modes)}
    v4 = {}
    for W in range(n_modes):
        for X in range(n_modes):
            for Y in range(n_modes):
                for Z in range(n_modes):
                    v4[(W, X, Y, Z)] = des[W] @ v3[(X, Y, Z)]

    # D_pq = sum_sigma <psi| a+_{p,sigma} a_{q,sigma} |psi>
    #      = sum_sigma <a_{p,sigma} psi | a_{q,sigma} psi>   (since (a+_X)^dagger = a_X)
    #      = sum_sigma <v1[so(p,sigma)], v1[so(q,sigma)]>
    D = np.zeros((norb, norb), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            D[p, q] = sum(np.vdot(v1[_bf_so(p, s)], v1[_bf_so(q, s)]) for s in (0, 1))

    # Gamma_pqrs = sum_{sigma,tau} <psi| a+_{p,sigma} a+_{q,tau} a_{r,tau} a_{s,sigma} |psi>
    #            = sum_{sigma,tau} <a_{q,tau} a_{p,sigma} psi | a_{r,tau} a_{s,sigma} psi>
    #            = sum_{sigma,tau} <v2[(so(q,tau), so(p,sigma))], v2[(so(r,tau), so(s,sigma))]>
    # (bra side picks up (a+_p a+_q)^dagger = a_q a_p, hence the swapped
    # argument order relative to the ket side -- this is the exact identity
    # already verified independently earlier in this project.)
    Gamma = np.zeros((norb,) * 4, dtype=complex)
    for p, q, r, s in np.ndindex(norb, norb, norb, norb):
        Gamma[p, q, r, s] = sum(
            np.vdot(v2[(_bf_so(q, tau), _bf_so(p, sigma))], v2[(_bf_so(r, tau), _bf_so(s, sigma))])
            for sigma in (0, 1)
            for tau in (0, 1)
        )

    # rdm3[p,q,r,s,t,u] = sum_{s1,s2,s3} <c+_{p,s1} c+_{q,s2} c+_{r,s3}  a_{s,s3} a_{t,s2} a_{u,s1}>
    #   bra side: (c+_p c+_q c+_r)^dagger = c_r c_q c_p  ->  v3[(so(r,s3), so(q,s2), so(p,s1))]
    #   ket side: a_{s,s3} a_{t,s2} a_{u,s1} |psi>        ->  v3[(so(s,s3), so(t,s2), so(u,s1))]
    rdm3 = np.zeros((norb,) * 6, dtype=complex)
    for p, q, r, s, t, u in np.ndindex(norb, norb, norb, norb, norb, norb):
        rdm3[p, q, r, s, t, u] = sum(
            np.vdot(
                v3[(_bf_so(r, s3), _bf_so(q, s2), _bf_so(p, s1))],
                v3[(_bf_so(s, s3), _bf_so(t, s2), _bf_so(u, s1))],
            )
            for s1 in (0, 1)
            for s2 in (0, 1)
            for s3 in (0, 1)
        )

    # rdm4[p,q,r,s,t,u,v,w] = sum_{s1..s4} <c+_p c+_q c+_r c+_s  a_t a_u a_v a_w>  (matching spins as documented)
    #   bra side: (c+_p c+_q c+_r c+_s)^dagger = c_s c_r c_q c_p  ->  v4[(so(s,s4), so(r,s3), so(q,s2), so(p,s1))]
    #   ket side: a_t a_u a_v a_w |psi>                            ->  v4[(so(t,s4), so(u,s3), so(v,s2), so(w,s1))]
    rdm4 = np.zeros((norb,) * 8, dtype=complex)
    for p, q, r, s in np.ndindex(norb, norb, norb, norb):
        for t, u, v, w in np.ndindex(norb, norb, norb, norb):
            rdm4[p, q, r, s, t, u, v, w] = sum(
                np.vdot(
                    v4[(_bf_so(s, s4), _bf_so(r, s3), _bf_so(q, s2), _bf_so(p, s1))],
                    v4[(_bf_so(t, s4), _bf_so(u, s3), _bf_so(v, s2), _bf_so(w, s1))],
                )
                for s1 in (0, 1)
                for s2 in (0, 1)
                for s3 in (0, 1)
                for s4 in (0, 1)
            )

    return D, Gamma, rdm3, rdm4


def _bf_random_state_in_sector(norb: int, partition: list[list[int]], N: tuple[int, ...], rng, real: bool) -> np.ndarray:
    """A normalized, random full-Fock-space vector supported ONLY on basis
    states whose cluster occupation is exactly N (i.e. "a dense state psi
    ... that lives in the sector N", built directly rather than by
    projecting a generic random vector -- projecting would need this exact
    same enumeration anyway)."""
    dim = 2 ** (2 * norb)
    N = tuple(N)
    psi = np.zeros(dim, dtype=complex)
    for basis_index in range(dim):
        if _bf_cluster_occupation(basis_index, partition, norb) == N:
            amplitude = rng.standard_normal()
            if not real:
                amplitude = amplitude + 1j * rng.standard_normal()
            psi[basis_index] = amplitude
    norm = np.linalg.norm(psi)
    if norm < 1e-14:
        raise ValueError(f"_bf_random_state_in_sector: sector N={N} is empty for partition {partition}.")
    return psi / norm


def _bf_valid_initial_N(cluster_sizes: list[int], destinations: list[int], sources: list[int], spectator_fill: int = 1) -> list[int]:
    """Build an initial sector label N that is guaranteed to be a valid,
    non-trivial starting point for the requested transfer (destinations,
    sources -- lists of cluster indices, repeats allowed, e.g.
    destinations=[I, I] when both incoming electrons land on cluster I):

      - every DESTINATION cluster starts at 0 electrons: always room to
        receive whatever it's about to gain.
      - every SOURCE cluster starts with EXACTLY as many electrons as it is
        about to lose (1, or 2 if it appears twice in `sources`): enough to
        supply the transfer, landing at exactly 0 afterward.
      - every OTHER ("spectator") cluster -- not touched by this transfer at
        all -- gets a small nonzero occupation (`spectator_fill` electrons,
        capped at that cluster's capacity). This matters: with NO
        spectator electrons, the total electron count in psi can be as low
        as 1 or 2 (just enough for the transfer itself), and rdm4 in
        particular is then trivially zero on many index combinations
        (you cannot annihilate 4 electrons out of a 1- or 2-electron
        state) -- a degenerate, much-less-informative test of exactly the
        Tier-2 terms this suite most wants to exercise. A few spectator
        electrons elsewhere keep the total electron count comfortably
        above the bare minimum without touching the transfer bookkeeping
        above.
    """
    assert set(destinations).isdisjoint(sources), (
        "a cluster cannot be both a destination and a source of the same transfer -- "
        "that is not one of the four transfer shapes this suite builds."
    )
    loss_count: dict[int, int] = {}
    for k in sources:
        loss_count[k] = loss_count.get(k, 0) + 1

    N = [0] * len(cluster_sizes)
    for k in range(len(cluster_sizes)):
        if k in loss_count:
            N[k] = loss_count[k]
        elif k in destinations:
            N[k] = 0
        else:
            N[k] = min(spectator_fill, 2 * cluster_sizes[k])
    return N


def _bf_apply_transfer(N: list[int], destinations: list[int], sources: list[int]) -> list[int]:
    """N' = N with +1 for every cluster index in `destinations` and -1 for
    every cluster index in `sources` (a repeated index accumulates
    correctly: destinations=[I, I] adds +2 to N'[I], exactly capturing
    "both electrons arrive at the same cluster")."""
    N_prime = list(N)
    for k in destinations:
        N_prime[k] += 1
    for k in sources:
        N_prime[k] -= 1
    return N_prime


def _bf_two_electron_shapes(cluster_indices: list[int]) -> dict[str, tuple[list[int], list[int]]]:
    """The four transfer shapes named in the task, as concrete
    (destinations, sources) index-pairs built from the given list of
    available cluster indices -- each shape answers "are the two
    destination clusters the same cluster?" x "are the two source clusters
    the same cluster?" independently:

        shape             destinations   sources    clusters needed
        ----------------  -------------  ---------  ---------------
        all_different      [I, J]         [K, L]     4 (I,J,K,L distinct)
        sources_same       [I, J]         [K, K]     3 (single source K)
        destinations_same  [I, I]         [J, K]     3 (single destination I)
        both_same          [I, I]         [K, K]     2 (the "canonical" pair hop)

    Only the shapes for which enough distinct cluster indices are available
    are returned.
    """
    shapes: dict[str, tuple[list[int], list[int]]] = {}
    if len(cluster_indices) >= 2:
        I, K = cluster_indices[0], cluster_indices[1]
        shapes["both_same"] = ([I, I], [K, K])
    if len(cluster_indices) >= 3:
        I, J, K = cluster_indices[0], cluster_indices[1], cluster_indices[2]
        shapes["sources_same"] = ([I, J], [K, K])
        shapes["destinations_same"] = ([I, I], [J, K])
    if len(cluster_indices) >= 4:
        I, J, K, L = cluster_indices[:4]
        shapes["all_different"] = ([I, J], [K, L])
    return shapes


def _bf_two_electron_shapes_with_doubled_cluster(
    cluster_indices: list[int], doubled_cluster: int
) -> dict[str, tuple[list[int], list[int]]]:
    """The three "doubled-role" shapes (both_same, sources_same,
    destinations_same -- all_different has no doubled role, so it is not
    generated here) with the doubled role EXPLICITLY routed through
    `doubled_cluster`, once as the doubled destination and once as the
    doubled source, rather than left to fall wherever
    _bf_two_electron_shapes' fixed cluster_indices[0]/[1]/[2] construction
    happens to put it.

    This matters specifically when `doubled_cluster` has >= 2 orbitals: a
    singleton cluster holding 2 electrons has exactly one way to do it
    (both spin-orbitals of its single orbital filled), so if a
    multi-orbital cluster is never actually placed in the doubled role,
    the RDM contractions that COULD involve two genuinely different
    orbitals within the same cluster (as opposed to the same orbital,
    different spin) never get exercised. Concretely, for both_same,
    sources_same, destinations_same, this generates:

        shape                        destinations         sources
        ---------------------------  -------------------  -------------------
        both_same_doubled_dest       [doubled, doubled]    [other0, other0]
        both_same_doubled_src        [other0, other0]      [doubled, doubled]
        sources_same_doubled         [other0, other1]       [doubled, doubled]
        destinations_same_doubled    [doubled, doubled]     [other0, other1]

    where other0, other1 are two cluster indices other than `doubled_cluster`
    (destinations_same_doubled/sources_same_doubled need 2 of them; the
    both_same variants need only 1).
    """
    others = [c for c in cluster_indices if c != doubled_cluster]
    shapes: dict[str, tuple[list[int], list[int]]] = {}
    if len(others) >= 1:
        shapes["both_same_doubled_dest"] = ([doubled_cluster, doubled_cluster], [others[0], others[0]])
        shapes["both_same_doubled_src"] = ([others[0], others[0]], [doubled_cluster, doubled_cluster])
    if len(others) >= 2:
        shapes["sources_same_doubled"] = ([others[0], others[1]], [doubled_cluster, doubled_cluster])
        shapes["destinations_same_doubled"] = ([doubled_cluster, doubled_cluster], [others[0], others[1]])
    return shapes


# ---- scenario generation: (norb, partition) pairs, plus per-scenario cases ----
#
# norb=4 costs ~1s per sampled state's full RDM tensor set (D, Gamma, rdm3,
# rdm4) on this machine; norb=5 costs ~12s; norb=6 costs ~100s. To honor
# "larger, more thorough" without the suite taking many tens of minutes, the
# heaviest scenario is capped at norb=5, used for a full sweep of shapes
# rather than norb=6 used for only one or two.

_BF_SCENARIOS = {
    # scenario name: (norb, partition)
    "norb4_four_singletons": (4, [[0], [1], [2], [3]]),
    "norb4_mixed_clusters": (4, [[0, 1], [2], [3]]),
    "norb5_larger_mixed": (5, [[0], [1], [2], [3, 4]]),
}


def _bf_generate_cases():
    """Build the full list of (case_id, norb, partition, N, N_prime) test
    cases: for every scenario above, every two-electron shape its cluster
    count allows, plus a couple of one-electron (I, J) transfers, each
    coupled with an initial N built to be valid and non-degenerate for that
    specific transfer (see _bf_valid_initial_N)."""
    cases = []
    for scenario_name, (norb, partition) in _BF_SCENARIOS.items():
        cluster_sizes = [len(c) for c in partition]
        K = len(partition)
        cluster_indices = list(range(K))

        # Two-electron cases: all four shapes this scenario's cluster count allows.
        for shape_name, (destinations, sources) in _bf_two_electron_shapes(cluster_indices).items():
            N = _bf_valid_initial_N(cluster_sizes, destinations, sources)
            N_prime = _bf_apply_transfer(N, destinations, sources)
            cases.append((f"{scenario_name}__2e_{shape_name}", norb, partition, tuple(N), tuple(N_prime)))

        # Two-electron cases, take 2: for every genuinely multi-orbital
        # cluster (>= 2 orbitals) in this scenario, explicitly route the
        # doubled role through it (both as doubled destination and as
        # doubled source) via _bf_two_electron_shapes_with_doubled_cluster
        # -- see that function's docstring for why this is not already
        # covered by the block above (the generic shapes only exercise a
        # multi-orbital cluster's doubled role by accident of which index
        # it happens to have, never guaranteed, and never as a source).
        multi_orbital_clusters = [k for k, cluster in enumerate(partition) if len(cluster) >= 2]
        for doubled_cluster in multi_orbital_clusters:
            extra_shapes = _bf_two_electron_shapes_with_doubled_cluster(cluster_indices, doubled_cluster)
            for shape_name, (destinations, sources) in extra_shapes.items():
                N = _bf_valid_initial_N(cluster_sizes, destinations, sources)
                N_prime = _bf_apply_transfer(N, destinations, sources)
                cases.append((f"{scenario_name}__2e_{shape_name}_c{doubled_cluster}", norb, partition, tuple(N), tuple(N_prime)))

        # One-electron cases: a couple of distinct (I, J) destination/source
        # pairs (reusing the same machinery with single-element lists).
        one_electron_pairs = [(cluster_indices[0], cluster_indices[1])]
        if K >= 3:
            one_electron_pairs.append((cluster_indices[2], cluster_indices[0]))
        for pair_idx, (I, J) in enumerate(one_electron_pairs):
            N = _bf_valid_initial_N(cluster_sizes, [I], [J])
            N_prime = _bf_apply_transfer(N, [I], [J])
            cases.append((f"{scenario_name}__1e_pair{pair_idx}", norb, partition, tuple(N), tuple(N_prime)))

    return cases


# Cross every generated case with real=True/False -- exactly half of all
# resulting sub-cases use real h/g/psi (+ coupling_strength_real), half use
# complex (+ coupling_strength_complex), as requested.
_BF_CASES_X_DTYPE = [
    (case_id, norb, partition, N, N_prime, real)
    for (case_id, norb, partition, N, N_prime) in _bf_generate_cases()
    for real in (True, False)
]


class TestEnergyScoreBruteForceCrossCheck:
    """For each sampled (partition, N, N', h, g, psi) scenario, checks that
    coupling_strength_{complex,real} (the cheap, RDM-based estimate of
    ||proj_{N'} H psi||^2) agrees with brute_force_energy_score (the same
    quantity computed by literally applying H and projecting) -- to
    numerical precision, since psi is built to have ALL of its weight in
    sector N, not just approximately. See the module-level comment block
    above this class for the full rationale and the ffsim investigation
    that led to using a from-scratch Fock-space construction here."""

    @pytest.mark.parametrize(
        "case_id, norb, partition, N, N_prime, real", _BF_CASES_X_DTYPE, ids=[c[0] + ("_real" if c[5] else "_complex") for c in _BF_CASES_X_DTYPE]
    )
    def test_energy_score_matches_brute_force(self, case_id, norb, partition, N, N_prime, real):
        rng = np.random.default_rng(hash((case_id, real)) % (2**32))
        n_modes = 2 * norb
        cre, des = _bf_ladder_operators(n_modes)

        # ---- sample h (1-body) and g (2-body): fully general/unconstrained,
        # no Hermiticity or permutation symmetry imposed (the strongest test
        # of the underlying operator identity, and the formulas were derived
        # to hold regardless of any such symmetry). ----
        dtype = float if real else complex

        def rand(shape):
            arr = rng.standard_normal(shape)
            if not real:
                arr = arr + 1j * rng.standard_normal(shape)
            return arr

        h1e = rand((norb, norb)).astype(dtype)
        g2e_full = rand((norb, norb, norb, norb)).astype(dtype)

        # ---- H as a LinearOperator, built directly from the full (unmasked) h1e/g2e_full ----
        H_sparse = _bf_build_hamiltonian_sparse(cre, des, norb, h1e, g2e_full)
        H = _bf_as_linear_operator(H_sparse)

        # ---- psi: guaranteed (by construction) to live entirely in sector N ----
        psi = _bf_random_state_in_sector(norb, partition, N, rng, real=real)

        # ---- ground truth ----
        brute_force_score = brute_force_energy_score(norb, partition, N, N_prime, psi, H)

        # ---- coupling_strength_{complex,real}: build its inputs ----
        # RDMs of psi, via this suite's own from-scratch builder.
        D, Gamma, rdm3, rdm4 = _bf_rdm_tensors_from_state(psi, des, norb)
        if real:
            D, Gamma, rdm3, rdm4 = D.real, Gamma.real, rdm3.real, rdm4.real

        # g_derived: the transposed convention coupling_strength_* itself
        # expects internally (see cluster_number_sector_search.py's module
        # docstring) -- computed here exactly as rank_relevant_sectors does.
        g_derived = g2e_full.transpose(0, 2, 3, 1)

        # Masks selecting the h1e/g_derived entries whose net cluster-flow
        # equals delta = N' - N -- built via the module's OWN production
        # mask-builder (not re-derived here), since this step is already
        # covered independently by TestBuildTransferMasks above, and using
        # it here tests coupling_strength_* the way rank_relevant_sectors
        # actually calls it.
        indicator = partition_to_cluster_matrix(partition, norb)
        delta = np.array(N_prime, dtype=int) - np.array(N, dtype=int)
        h_mask, g_mask = build_transfer_masks(indicator, norb, delta, need_g=True)

        coupling_fn = coupling_strength_real if real else coupling_strength_complex
        h1e_in = h1e.real if real else h1e
        coupling_score, tier = coupling_fn(h1e_in, g_derived, D, Gamma, rdm3, rdm4, h_mask, g_mask)

        # ---- the actual cross-check ----
        assert coupling_score is not None, (
            f"coupling_strength_{'real' if real else 'complex'} returned no score for "
            f"delta={delta.tolist()} -- expected a Tier-1 or Tier-2 score since h1e/g2e_full "
            f"are fully dense (h_mask/g_mask should always select something for any "
            f"t<=2 transfer, and every case generated here has t<=2 by construction)."
        )
        assert coupling_score == pytest.approx(brute_force_score, abs=1e-6, rel=1e-6), (
            f"case={case_id} real={real}: brute_force_energy_score={brute_force_score!r} vs. "
            f"coupling_strength_{'real' if real else 'complex'}={coupling_score!r} (tier={tier}), "
            f"N={N} N'={N_prime} delta={delta.tolist()}"
        )
        # q(delta) is always a squared norm: both sides must be non-negative
        # (brute force manifestly is, by construction of the sum-of-squares
        # norm; coupling_strength_* is not manifestly so from its formula
        # alone -- this is a structural consequence being checked, same as
        # in TestCouplingStrengthGroundTruth above).
        assert brute_force_score >= -1e-9
        assert coupling_score >= -1e-9
