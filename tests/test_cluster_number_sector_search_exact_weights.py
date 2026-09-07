# AI-generated

"""Tests for cluster_number_sector_search_exact_weights.py.

Two kinds of ground truth are used, at two different levels:

  - The FULL-STATE ROUTE, reused directly from this codebase's own
    production modules rather than reimplemented here: get_sector_weight
    and number_and_parity_symmetry_sectors, the exact functions
    cluster_number_decomposition_optimization.py's own sector-analysis path
    (_compute_sector_analysis_for_trajectory, via
    get_ordered_state_projections_in_sectors) is built on. Given a full CI
    vector psi (from solver.to_ci_vector, optionally rotated by
    ffsim.apply_orbital_rotation) and a partition, this literally builds
    every symmetry sector's determinant-index list and sums |psi|^2 over it
    -- exponential in norb, but exact and completely independent of
    anything in this module (no MPS, no block2 decoding, no environment
    sweep). This is the module the task asked this suite to compare
    against, and is used both directly (TestSectorWeightNativeVsFullStateRoute,
    TestRankRelevantSectorsRealSystem) and to build a convenience wrapper
    (_full_state_sector_weights) reused throughout this file.

  - Two kinds of MPS feed that ground truth: a RANDOM MPS (bond_dim=5, no
    DMRG sweep -- mirrors test_cluster_number_sector_search.py's own
    _rdm_data_from_random_mps helper, extended here to also return the
    solver+mps this module's functions need), used for FAST, EXTENSIVE
    coverage across many (norb, nelec, bond_dim, partition, seed, proper/
    improper rotation) combinations -- valid because sector_weight_native's
    exactness claim is a property of ANY U(1)-symmetric MPS, not something
    that requires a converged ground state (this was explicitly checked
    against a deliberately truncated-bond-dimension MPS during this
    module's own development; TestSectorWeightNativeVsFullStateRoute
    exercises the same point via a genuinely random MPS, an even stronger
    check); and a REAL ground-state MPS from an actual small-molecule DMRG
    run (via cluster_number_decomposition_optimization._run_dmrg_and_build_
    rdm_data), used where the real thing matters: end-to-end integration
    through an actual beam-search trajectory (whose partitions are, in
    practice, NOT contiguous chain-order blocks -- see
    TestFullEnvironmentSweepNonContiguousPartitions and
    TestRankRelevantSectorsRealSystem), and locking in the specific
    (molecule, bond_dim) combinations that used to crash or silently
    mis-decode before this module's from_block2 patch was fixed (see
    TestDecoderBoundaryConfigurationRegression).
"""

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pytest

import ffsim

from src.dmrg_solver import Block2DMRGSolver
from src.cluster_number_operators import number_and_parity_symmetry_sectors
from src.orbital_rotation import params_to_U
from cluster_numbers_metrics import get_sector_weight
from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    DecompositionOptimizerConfig,
    _run_dmrg_and_build_rdm_data,
    make_variance_cost_constructor,
    partition_to_cluster_matrix,
    run_decomposition_optimizer,
)
from cluster_number_sector_search import create_parser as create_tier_parser  # for _run_dmrg_and_build_rdm_data's args
from cluster_number_sector_search_exact_weights import (
    SectorRelevance,
    SectorSearchConfig,
    _full_environment_sweep,
    _orbital_to_cluster_map,
    _patched_from_block2,
    create_parser,
    decode_mps,
    native_sweep_all_labels,
    rank_relevant_sectors,
    rotate_mps_to_basis,
    sector_weight_native,
)


# =============================================================================
# Shared helpers
# =============================================================================


def _random_mps_solver(norb, nelec, bond_dim=5, seed=0):
    """A Block2DMRGSolver + random MPS -- valid block-sparse tensors, not a
    ground state. Mirrors test_cluster_number_sector_search.py's own
    _rdm_data_from_random_mps, extended to return the solver+mps this
    module's own functions (rotate_mps_to_basis, decode_mps, ...) need.
    Caller owns cleaning up solver.store_dir (a fresh tempdir each call)."""
    tmp_dir = tempfile.mkdtemp(prefix="exact_weights_test_")
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
    return solver, mps, tmp_dir


def _full_state_sector_weights(psi, norb, nelec, partition):
    """The full-state route, reused verbatim from cluster_numbers_metrics.py
    (get_sector_weight) and src/cluster_number_operators.py
    (number_and_parity_symmetry_sectors) -- see module docstring. Returns
    {label: weight} for every symmetry sector of `partition`, label = the
    bare per-cluster-count tuple (the parity sub-label is always () here,
    since no cluster_parity_matrix is passed)."""
    cluster_matrix = partition_to_cluster_matrix(partition, norb)
    sectors = number_and_parity_symmetry_sectors(cluster_matrix, [], norb, nelec)
    return {label: get_sector_weight(sectors, (label, ()), psi) for (label, _par) in sectors}


def _random_proper_or_improper_U(norb, seed, improper):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.8, 0.8, size=norb * (norb - 1) // 2)
    U = params_to_U(x, norb)
    if improper:
        U = U.copy()
        U[:, -1] *= -1
    assert np.allclose(U @ U.T, np.eye(norb), atol=1e-10)
    assert (np.linalg.det(U) < 0) == improper
    return U


# =============================================================================
# _patched_from_block2 / decode_mps: normalization sanity
# =============================================================================


class TestPatchedDecoderNormalization:
    """<psi|psi> == 1 after decode (labels=None, i.e. no sector restriction
    at all -- see _full_environment_sweep) is the cheapest possible
    end-to-end check that decode_mps's block-sparse tensors are internally
    consistent (chain correctly bond-to-bond all the way through) --
    exercised across many bond dimensions/electron counts/rotations,
    including both proper and improper ones, entirely on fast random MPS."""

    @pytest.mark.parametrize("norb,nelec", [(4, (2, 2)), (5, (3, 2)), (6, (3, 3)), (6, (4, 2))])
    @pytest.mark.parametrize("bond_dim", [1, 3, 5, 10])
    @pytest.mark.parametrize("improper", [False, True])
    def test_full_norm_is_one(self, norb, nelec, bond_dim, improper):
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=bond_dim, seed=1)
        try:
            U = _random_proper_or_improper_U(norb, seed=2, improper=improper)
            mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=20, ket0=mps)
            weights = native_sweep_all_labels(mps_rot, [[p] for p in range(norb)], labels=None)
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_identity_rotation_is_a_no_op(self):
        norb, nelec = 5, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=3)
        try:
            mps_rot = rotate_mps_to_basis(solver, np.eye(norb), ket0=mps)
            assert mps_rot is mps  # early-exit path: identity generator, no td_dmrg run at all
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# sector_weight_native / native_sweep_all_labels vs. the full-state route
# =============================================================================


class TestSectorWeightNativeVsFullStateRoute:
    """The core correctness suite: native_sweep_all_labels (this module's
    exact-weight machinery) against _full_state_sector_weights (the
    codebase's own production full-state route -- see module docstring),
    on random MPS across many partitions -- CONTIGUOUS (chain-order blocks)
    and NON-CONTIGUOUS (shuffled cluster order / interleaved orbitals,
    exactly the shape a real beam-search partition has -- see
    TestRankRelevantSectorsRealSystem) -- many bond dimensions, and both
    proper and improper rotations."""

    _PARTITIONS_BY_NORB = {
        6: {
            "contiguous_2clu": [[0, 1, 2], [3, 4, 5]],
            "contiguous_singleton": [[0], [1], [2], [3], [4], [5]],
            "noncontiguous_shuffled_singleton": [[3], [0], [5], [1], [4], [2]],
            "noncontiguous_interleaved": [[0, 2, 4], [1, 3, 5]],
            "noncontiguous_mixed": [[0, 3], [1, 4, 5], [2]],
        },
        7: {
            "contiguous_3clu": [[0, 1], [2, 3, 4], [5, 6]],
            "noncontiguous_shuffled": [[6], [0], [4], [1], [5], [2], [3]],
            "noncontiguous_interleaved": [[0, 3, 6], [1, 4], [2, 5]],
        },
    }

    @pytest.mark.parametrize("norb,nelec,bond_dim", [(6, (3, 3), 4), (6, (4, 2), 8), (7, (4, 3), 6)])
    @pytest.mark.parametrize("seed", [1, 2])
    @pytest.mark.parametrize("improper", [False, True])
    def test_matches_full_state_route(self, norb, nelec, bond_dim, seed, improper):
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=bond_dim, seed=seed)
        try:
            U = _random_proper_or_improper_U(norb, seed=seed + 100, improper=improper)
            mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=20, ket0=mps)
            psi_rot = solver.to_ci_vector(ket=mps_rot)

            for name, partition in self._PARTITIONS_BY_NORB[norb].items():
                exact = _full_state_sector_weights(psi_rot, norb, nelec, partition)
                native = native_sweep_all_labels(mps_rot, partition, list(exact.keys()))
                max_diff = max(abs(exact[l] - native[l]) for l in exact)
                assert max_diff < 1e-9, (
                    f"norb={norb} nelec={nelec} bd={bond_dim} seed={seed} improper={improper} "
                    f"partition={name}: max|native-full_state|={max_diff:.3e}"
                )
                assert sum(native.values()) == pytest.approx(1.0, abs=1e-8)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_matches_full_state_route_for_truncated_nonconverged_mps(self):
        """Exactness must hold for WHATEVER state the MPS holds, not just a
        converged one -- a bond_dim=1 random MPS is about as far from a
        physical ground state as a valid U(1)-symmetric MPS can get, and is
        used here specifically because it's cheap and exercises the same
        "sector weight is exact for the state on hand, regardless of
        convergence" claim already checked (during this module's
        development, against a deliberately truncated real DMRG state) on a
        real molecule -- see the module docstring."""
        norb, nelec = 6, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=1, seed=5)
        try:
            psi = solver.to_ci_vector(ket=mps)
            partition = [[4], [0], [2, 5], [1, 3]]
            exact = _full_state_sector_weights(psi, norb, nelec, partition)
            native = native_sweep_all_labels(mps, partition, list(exact.keys()))
            max_diff = max(abs(exact[l] - native[l]) for l in exact)
            assert max_diff < 1e-9
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_labels_none_returns_every_reachable_label(self):
        norb, nelec = 5, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=6, seed=7)
        try:
            partition = [[1, 3], [0], [2, 4]]
            all_labels = native_sweep_all_labels(mps, partition, labels=None)
            exact = _full_state_sector_weights(solver.to_ci_vector(ket=mps), norb, nelec, partition)
            assert set(all_labels.keys()) == set(exact.keys())
            for label in exact:
                assert all_labels[label] == pytest.approx(exact[label], abs=1e-9)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_unreachable_label_returns_zero_not_a_crash(self):
        norb, nelec = 5, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=8)
        try:
            partition = [[0, 1], [2, 3, 4]]
            # (5, 5) sums to 10 > nelec_total=5: unreachable for ANY state.
            out = native_sweep_all_labels(mps, partition, [(5, 5)])
            assert out[(5, 5)] == 0.0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestOrbitalToClusterMap:
    def test_basic_mapping(self):
        mapping = _orbital_to_cluster_map([[2, 0], [1, 3]])
        assert mapping == {2: 0, 0: 0, 1: 1, 3: 1}

    def test_singleton_clusters(self):
        mapping = _orbital_to_cluster_map([[3], [1], [0], [2]])
        assert mapping == {3: 0, 1: 1, 0: 2, 2: 3}


class TestSectorWeightNativeSingleLabelMatchesBatch:
    def test_single_label_call_matches_native_sweep_all_labels(self):
        norb, nelec = 5, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=9)
        try:
            partition = [[0, 2], [1], [3, 4]]
            decoded = decode_mps(mps)
            orbital_to_cluster = _orbital_to_cluster_map(partition)
            batch = native_sweep_all_labels(mps, partition, labels=None)
            for label in list(batch.keys())[:5]:
                single = sector_weight_native(decoded, orbital_to_cluster, len(partition), label)
                assert single == pytest.approx(batch[label], abs=1e-12)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# rotate_mps_to_basis
# =============================================================================


class TestRotateMpsToBasis:
    def test_rotation_converges_with_more_td_steps(self):
        """The rotation step (td_dmrg/TDVP) is the one place this module's
        overall weight has finite-precision error (unlike the sweep itself,
        which is exact for whatever state it's handed -- see module
        docstring): the discrepancy against an EXACTLY rotated reference
        (via ffsim.apply_orbital_rotation on the un-rotated CI vector, not
        self-consistently re-extracted from the same rotated MPS) must
        shrink as n_td_steps grows.

        Needs a real, moderately-entangled ground state, not a random MPS:
        tried first on a random bond_dim=6/norb=5 MPS, whose rotation
        turned out to already be exact to ~1e-15 even at n_td_steps=1 (the
        max_bd=max(actual_bd,200) rotation cap so vastly exceeds a 100-
        dimensional Hilbert space's own true bond dimension that there is no
        truncation error at all, apparently leaving TDVP's own integration
        error immeasurably small too for a system this simple) -- i.e. not
        a case that exercises this behavior at all. A real LiH ground state
        with a large, deliberately aggressive rotation (matching the
        magnitude cluster_number_sector_search_exact_weights.py's own
        module docstring calls out as the "worst case" this convergence
        knob exists for) reproduces the effect this test needs to check.
        """
        solver, rdm_data, dmrg_energy, h1e, g2e, ecore, nelec = _build_real_solver_and_rdm_data(
            "lih", "sto-3g", 1.6, bond_dim=20, n_sweeps=14,
        )
        norb = rdm_data.norb
        psi0 = solver.to_ci_vector(ket=solver.get_mps())
        U = _random_proper_or_improper_U(norb, seed=42, improper=False)
        psi_exact = ffsim.apply_orbital_rotation(psi0, U, norb, nelec)
        partition = [[0, 1, 2], [3, 4, 5]]
        exact = _full_state_sector_weights(psi_exact, norb, nelec, partition)

        errors = []
        for n_td in (10, 40):
            mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=n_td)
            native = native_sweep_all_labels(mps_rot, partition, list(exact.keys()))
            errors.append(max(abs(exact[l] - native[l]) for l in exact))
        assert errors[0] > 1e-8, (
            f"n_td_steps=10 error {errors[0]:.3e} is already negligible -- this rotation no "
            f"longer exercises meaningful convergence behavior; errors={errors}"
        )
        assert errors[1] < errors[0], errors

    def test_improper_rotation_matches_proper_plus_reflection_manually(self):
        """Sanity-checks rotate_mps_to_basis's own U = R @ F factoring
        against a hand-rolled equivalent using the SAME public
        Block2DMRGSolver.apply_rotated_parity API it calls internally, to
        catch a regression in which factor is applied first independent of
        whatever native_sweep_all_labels/ground truth comparison would
        otherwise mask."""
        norb, nelec = 5, (3, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=13)
        try:
            U = _random_proper_or_improper_U(norb, seed=14, improper=True)
            mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=20, ket0=mps)

            F = np.eye(norb)
            F[-1, -1] = -1.0
            R = U @ F
            parity_row = np.zeros(norb)
            parity_row[-1] = 1.0
            reflected = solver.apply_rotated_parity(parity_row, np.eye(norb), ket=mps, tag="MANUAL_REFL")
            import scipy.linalg
            A = scipy.linalg.logm(R)
            A = 0.5 * (np.real(A) - np.real(A).T)
            builder = solver.driver.expr_builder()
            builder.add_sum_term("cd", A)
            builder.add_sum_term("CD", A)
            mpo_A = solver.driver.get_mpo(builder.finalize(), iprint=0)
            max_bd = max(int(reflected.info.get_max_bond_dimension()), 200)
            manual = solver.driver.td_dmrg(
                mpo_A, reflected, delta_t=-1.0 / 20, target_t=-1.0, n_steps=20,
                te_type="tdvp", bond_dims=[max_bd] * 20, final_mps_tag="MANUAL_ROT",
                normalize_mps=False, hermitian=False, iprint=0,
            )
            partition = [[i] for i in range(norb)]
            a = native_sweep_all_labels(mps_rot, partition, labels=None)
            b = native_sweep_all_labels(manual, partition, labels=None)
            assert set(a.keys()) == set(b.keys())
            for label in a:
                assert a[label] == pytest.approx(b[label], abs=1e-8)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# SectorSearchConfig / SectorRelevance
# =============================================================================


class TestDataclassDefaults:
    def test_sector_search_config_defaults(self):
        config = SectorSearchConfig()
        assert config.num_sectors_to_retain is None
        assert config.max_cum_dim_to_retain is None
        assert config.max_elec_transfer == 2
        assert config.n_td_steps == 20

    def test_sector_relevance_fields(self):
        r = SectorRelevance(label=(2, 1), weight=0.5, elec_transfer=1, dimension=6)
        assert r.label == (2, 1)
        assert r.weight == 0.5
        assert r.elec_transfer == 1
        assert r.dimension == 6


# =============================================================================
# rank_relevant_sectors -- synthetic (random-MPS) integration tests
# =============================================================================


class TestRankRelevantSectorsSynthetic:
    @pytest.fixture
    def solver_and_data(self):
        norb, nelec = 4, (2, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=21)
        rdm1_a, rdm1_b = solver.driver.get_1pdm(mps)
        D = rdm1_a + rdm1_b
        rdm2_aa, rdm2_ab, rdm2_bb = solver.driver.get_2pdm(mps)
        Gamma = rdm2_aa + rdm2_bb + rdm2_ab + rdm2_ab.transpose(1, 0, 3, 2)
        rdm_data = RDMData(D=D, Gamma=Gamma)
        yield solver, mps, rdm_data, nelec
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.fixture
    def deco(self):
        return Decomposition(partition=[[0, 1], [2, 3]], U=np.eye(4), cost=0.0)

    def test_end_to_end_descending_order_and_weight_sum(self, deco, solver_and_data):
        solver, mps, rdm_data, nelec = solver_and_data
        results = rank_relevant_sectors(
            deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=3), mps=mps
        )
        assert len(results) > 0
        assert all(isinstance(r, SectorRelevance) for r in results)
        assert all(len(r.label) == 2 for r in results)
        assert all(r.dimension > 0 for r in results)
        assert all(0.0 <= r.weight <= 1.0 + 1e-9 for r in results)
        weights = [r.weight for r in results]
        assert weights == sorted(weights, reverse=True)
        # every candidate the BFS could reach was scored -- sum over the FULL
        # candidate set (not just the retained/truncated one) must be <= 1
        # (equality only if max_elec_transfer covers every reachable sector)
        assert sum(weights) <= 1.0 + 1e-8

    def test_respects_num_sectors_to_retain(self, deco, solver_and_data):
        solver, mps, rdm_data, nelec = solver_and_data
        config = SectorSearchConfig(num_sectors_to_retain=2, max_elec_transfer=2)
        results = rank_relevant_sectors(deco, rdm_data, nelec, solver, config, mps=mps)
        assert len(results) <= 2

    def test_respects_max_cum_dim_to_retain(self, deco, solver_and_data):
        solver, mps, rdm_data, nelec = solver_and_data
        unrestricted = rank_relevant_sectors(
            deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=2), mps=mps
        )
        cap = unrestricted[0].dimension
        results = rank_relevant_sectors(
            deco, rdm_data, nelec, solver, SectorSearchConfig(max_cum_dim_to_retain=cap, max_elec_transfer=2),
            mps=mps,
        )
        assert sum(r.dimension for r in results) <= cap or len(results) == 1

    def test_handles_overlapping_or_incomplete_partition(self, solver_and_data):
        solver, mps, rdm_data, nelec = solver_and_data
        messy_deco = Decomposition(partition=[[0, 1], [1, 2]], U=np.eye(4), cost=0.0)  # overlap + orbital 3 uncovered
        results = rank_relevant_sectors(
            messy_deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=2), mps=mps
        )
        assert len(results) > 0
        assert all(len(r.label) == 1 for r in results)  # merged partition has 1 real cluster; ghost's coord stripped

    def test_weight_is_exact_not_just_ranked(self, deco, solver_and_data):
        """Cross-checks rank_relevant_sectors's own reported weight against
        the full-state route directly (not just internal self-consistency),
        for the identity-rotation deco used throughout this class."""
        solver, mps, rdm_data, nelec = solver_and_data
        results = rank_relevant_sectors(
            deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=3), mps=mps
        )
        psi = solver.to_ci_vector(ket=mps)  # deco.U is identity: reference basis IS deco's basis
        exact = _full_state_sector_weights(psi, 4, nelec, deco.partition)
        for r in results:
            assert r.weight == pytest.approx(exact[r.label], abs=1e-8)


# =============================================================================
# Real small-molecule integration: the actual beam search + the actual
# full-state route from cluster_number_decomposition_optimization.py /
# cluster_numbers_metrics.py.
# =============================================================================


def _build_real_solver_and_rdm_data(molecule, basis, bond_length, bond_dim, n_sweeps, bond_angle=None):
    parser = create_tier_parser()
    argv = [molecule, basis, str(bond_length), "variance", "--bond-dim", str(bond_dim), "--n-sweeps", str(n_sweeps)]
    if bond_angle is not None:
        argv += ["--bond-angle", str(bond_angle)]
    args = parser.parse_args(argv)
    rdm_data, meta, solver, dmrg_energy, h1e, g2e, ecore, nelec = _run_dmrg_and_build_rdm_data(args, force_full_rdms=False)
    return solver, rdm_data, dmrg_energy, h1e, g2e, ecore, nelec


class TestRankRelevantSectorsRealSystem:
    """End-to-end on a REAL molecule through the ACTUAL beam search
    (run_decomposition_optimizer), which -- unlike every partition
    constructed by hand elsewhere in this file -- routinely produces
    partitions whose clusters are NOT contiguous, chain-ordered orbital
    blocks (confirmed directly: see the partition asserted on below), the
    exact scenario TestSectorWeightNativeVsFullStateRoute's noncontiguous_*
    cases exist to cover. Ground truth is
    cluster_numbers_metrics.get_sector_weight, called through EXACTLY the
    same ffsim.apply_orbital_rotation + number_and_parity_symmetry_sectors
    pipeline cluster_number_decomposition_optimization.py's own
    _compute_sector_analysis_for_trajectory uses -- i.e. this is a
    comparison against that module's actual production code path, not a
    re-derivation of the same idea."""

    @pytest.fixture(scope="class")
    @classmethod
    def lih_setup(cls):
        solver, rdm_data, dmrg_energy, h1e, g2e, ecore, nelec = _build_real_solver_and_rdm_data(
            "lih", "sto-3g", 1.6, bond_dim=25, n_sweeps=16,
        )
        beam_rdm_data = RDMData(D=rdm_data.D, Gamma=rdm_data.Gamma)
        opt_config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, maxiter=30)
        trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), beam_rdm_data, opt_config)
        return solver, rdm_data, nelec, trajectory

    def test_trajectory_has_a_noncontiguous_partition(self, lih_setup):
        """Documents (and pins) the premise this whole class exists to
        cover: a real beam search's own partition is generally not
        chain-ordered, so this test suite MUST exercise that shape to be a
        genuine end-to-end check, not just a happy-path one."""
        _solver, _rdm_data, _nelec, trajectory = lih_setup
        finest = trajectory[-1]
        assert finest.num_clusters >= 3
        orbital_order = [cluster[0] for cluster in finest.partition]  # each cluster is a singleton here
        assert orbital_order != sorted(orbital_order), (
            f"partition {finest.partition} came back chain-ordered -- this class's premise "
            "(real beam-search partitions are shuffled relative to orbital order) wasn't "
            "exercised; pick a different num_decos/seed/molecule to restore coverage."
        )

    @pytest.mark.parametrize("which", ["finest", "middle"])
    def test_ranked_weights_match_the_real_full_state_route(self, lih_setup, which):
        solver, rdm_data, nelec, trajectory = lih_setup
        norb = rdm_data.norb
        deco = trajectory[-1] if which == "finest" else trajectory[len(trajectory) // 2]
        if deco.num_clusters < 2:
            pytest.skip("trajectory entry has only 1 cluster")

        config = SectorSearchConfig(max_elec_transfer=2, n_td_steps=40)
        ranked = rank_relevant_sectors(deco, rdm_data, nelec, solver, config)
        assert len(ranked) > 0

        # The real full-state route, exactly as
        # cluster_number_decomposition_optimization._compute_sector_analysis_
        # for_trajectory computes it: rotate the REFERENCE-basis CI vector by
        # deco.U via ffsim, then read off sector weights with get_sector_weight.
        psi_MOs = solver.to_ci_vector(ket=solver.get_mps())
        psi_rot = ffsim.apply_orbital_rotation(psi_MOs, np.asarray(deco.U), norb, nelec)
        cluster_matrix = partition_to_cluster_matrix(deco.partition, norb)
        sectors = number_and_parity_symmetry_sectors(cluster_matrix, [], norb, nelec)

        max_diff = 0.0
        for r in ranked:
            exact = get_sector_weight(sectors, (r.label, ()), psi_rot)
            max_diff = max(max_diff, abs(exact - r.weight))
        # n_td_steps=40's own rotation-convergence error (see
        # TestRotateMpsToBasis) dominates this tolerance, not
        # sector_weight_native's own exactness.
        assert max_diff < 5e-3, f"deco={which} ({deco.num_clusters} clusters): max_diff={max_diff:.3e}"

    def test_weights_sum_towards_one_as_max_elec_transfer_grows(self, lih_setup):
        """Not a strict equality (the BFS candidate set is still bounded),
        but the retained weight should visibly increase as more of the
        Hilbert space is covered -- a sanity check on both the BFS
        candidate generation (reused unchanged from the tier module) and
        the weights themselves."""
        solver, rdm_data, nelec, trajectory = lih_setup
        deco = trajectory[-1]
        sums = []
        for t in (1, 2, 4):
            ranked = rank_relevant_sectors(deco, rdm_data, nelec, solver, SectorSearchConfig(max_elec_transfer=t))
            sums.append(sum(r.weight for r in ranked))
        assert sums[0] <= sums[1] + 1e-9 <= sums[2] + 1e-9
        assert sums[-1] > 0.9  # LiH/sto-3g is a weakly correlated system at this geometry


# =============================================================================
# Decoder boundary-configuration regression (see
# cluster_number_sector_search_exact_weights._patched_from_block2's own
# docstring for the full derivation of both cases this locks in).
# =============================================================================


class TestDecoderBoundaryConfigurationRegression:
    """H2O/sto-3g at bond dimensions that used to either hard-crash
    (AssertionError: no matching dispatch branch) or silently return
    all-zero sector weights (wrong quantum-number convention on the fix's
    first attempt -- see the module docstring) before this module's
    from_block2 patch was completed. These need a real DMRG run (the
    boundary configuration is a property of how td_dmrg's own TDVP sweep
    interacts with the state's actual entanglement structure, not something
    a uniform-bond-dimension random MPS reliably reproduces), so they're
    slower than the rest of this file and deliberately kept to a small,
    representative subset rather than the full sweep this bug was
    originally diagnosed with (see AI_exploration_2/step5_decoder_fix_
    validation.py in this project's own exploration history for that
    broader sweep)."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o_solver(cls):
        solver, rdm_data, dmrg_energy, h1e, g2e, ecore, nelec = _build_real_solver_and_rdm_data(
            "h2o", "sto-3g", 2.0, bond_dim=8, n_sweeps=14, bond_angle=104.5,
        )
        return solver, rdm_data.norb, nelec

    @pytest.mark.parametrize("bond_dim", [8, 9, 10, 11, 13])
    @pytest.mark.parametrize("improper", [False, True])
    def test_previously_crashing_bond_dimensions(self, bond_dim, improper):
        solver, rdm_data, dmrg_energy, h1e, g2e, ecore, nelec = _build_real_solver_and_rdm_data(
            "h2o", "sto-3g", 2.0, bond_dim=bond_dim, n_sweeps=14, bond_angle=104.5,
        )
        norb = rdm_data.norb
        U = _random_proper_or_improper_U(norb, seed=1, improper=improper)
        mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=20)
        psi_rot = solver.to_ci_vector(ket=mps_rot)

        for partition in ([[0, 1, 2], [3, 4, 5, 6]], [[i] for i in range(norb)]):
            exact = _full_state_sector_weights(psi_rot, norb, nelec, partition)
            native = native_sweep_all_labels(mps_rot, partition, list(exact.keys()))
            max_diff = max(abs(exact[l] - native[l]) for l in exact)
            assert max_diff < 1e-9, f"bd={bond_dim} improper={improper} partition={partition}: {max_diff:.3e}"
            assert sum(native.values()) == pytest.approx(1.0, abs=1e-8)

    def test_left_boundary_configuration_is_actually_exercised(self, h2o_solver):
        """Directly confirms this test class is hitting the configuration
        it exists to cover (center==0, dot==2, wavefunction at the literal
        boundary site -- see _patched_from_block2's docstring), rather than
        silently passing for an unrelated reason (e.g. a bond dimension
        that no longer reaches any boundary-touching configuration at all
        after an unrelated future change elsewhere in the pipeline)."""
        solver, norb, nelec = h2o_solver
        U = _random_proper_or_improper_U(norb, seed=1, improper=False)
        mps_rot = rotate_mps_to_basis(solver, U, n_td_steps=20)
        assert mps_rot.dot == 2
        assert mps_rot.center in (0, mps_rot.n_sites - 2)


# =============================================================================
# K-sector-analysis reuse (get_relevance_ranked_K_sectors_values_energies
# and friends, imported unchanged from cluster_number_sector_search.py)
# =============================================================================


class TestKSectorAnalysisReuse:
    def test_get_relevance_ranked_K_sectors_values_energies_accepts_new_dataclass(self):
        from cluster_number_sector_search import get_relevance_ranked_K_sectors_values_energies

        norb, nelec = 4, (2, 2)
        solver, mps, tmp_dir = _random_mps_solver(norb, nelec, bond_dim=5, seed=31)
        try:
            partition = [[0, 1], [2, 3]]
            psi = solver.to_ci_vector(ket=mps)
            cluster_matrix = partition_to_cluster_matrix(partition, norb)
            sectors = number_and_parity_symmetry_sectors(cluster_matrix, [], norb, nelec)

            weights = native_sweep_all_labels(mps, partition, labels=None)
            ranked = [
                SectorRelevance(label=label, weight=w, elec_transfer=0, dimension=len(sectors[(label, ())]))
                for label, w in sorted(weights.items(), key=lambda kv: -kv[1])
            ]

            h_linop = ffsim.linear_operator(
                ffsim.MolecularHamiltonian(
                    one_body_tensor=np.zeros((norb, norb)), two_body_tensor=np.zeros((norb,) * 4),
                ),
                norb, nelec,
            )
            K_values, energies, retained_dims, _reached = get_relevance_ranked_K_sectors_values_energies(
                psi, h_linop, 0.0, sectors, ranked, chemical_precision=1e-6,
            )
            assert K_values == list(range(1, len(K_values) + 1))
            assert retained_dims == sorted(retained_dims)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================


class TestCLI:
    def test_parser_accepts_required_positional_args(self):
        parser = create_parser()
        args = parser.parse_args(["h2o", "sto-3g", "2.0", "variance"])
        assert args.molecule == "h2o"
        assert args.n_td_steps == 20

    def test_tier_only_flags_are_gone(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["h2o", "sto-3g", "2.0", "variance", "--force-full-rdms"])
        with pytest.raises(SystemExit):
            parser.parse_args(["h2o", "sto-3g", "2.0", "variance", "--force-h1e"])

    def test_n_td_steps_is_configurable(self):
        parser = create_parser()
        args = parser.parse_args(["h2o", "sto-3g", "2.0", "variance", "--n-td-steps", "60"])
        assert args.n_td_steps == 60


# =============================================================================
# main() end-to-end (direct call, not a subprocess -- matches this
# codebase's own convention of testing CLI modules by calling their
# functions directly rather than spawning `python module.py ...`).
# =============================================================================


class TestMainEndToEnd:
    def test_main_produces_valid_json_output(self, monkeypatch, tmp_path, caplog):
        import cluster_number_sector_search_exact_weights as ew

        output_dir = tmp_path / "out"
        wavefunction_dir = tmp_path / "wavefunctions"
        argv = [
            "prog", "lih", "sto-3g", "1.6", "variance",
            "--bond-dim", "20", "--n-sweeps", "12",
            "--num-decos", "2", "--num-subdecos", "2", "--maxiter", "20",
            "--target-num-clusters", "3",
            "--n-td-steps", "10",
            "--output-dir", str(output_dir),
            "--wavefunction-dir", str(wavefunction_dir),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with caplog.at_level(logging.INFO):
            ew.main()

        files = list(output_dir.glob("sectors_*clusters_variance_*.json"))
        assert len(files) >= 1
        with open(files[0]) as f:
            output = json.load(f)

        assert "metadata" in output and "ranked_sectors" in output
        assert output["metadata"]["molecule"] == "lih"
        assert output["metadata"]["n_td_steps"] == 10
        assert len(output["ranked_sectors"]) > 0
        for entry in output["ranked_sectors"]:
            assert set(entry.keys()) == {"label", "weight", "elec_transfer", "dimension"}
            assert 0.0 <= entry["weight"] <= 1.0 + 1e-6
        weights = [entry["weight"] for entry in output["ranked_sectors"]]
        assert weights == sorted(weights, reverse=True)
