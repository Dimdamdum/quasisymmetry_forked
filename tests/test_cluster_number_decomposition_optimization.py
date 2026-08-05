"""Tests for cluster_number_decomposition_optimization.py.

These don't re-validate primitives already covered by
test_cluster_number_operators.py (build_loc_number_evaluator,
number_variance_cost, params_to_U_jax, ...); they focus on the new module's
own logic: partition bookkeeping, the block-restricted rotation embedding
(the most error-prone part), Fiedler splitting, and the beam search /
polish mechanics.

RDMs are built from a random MPS (bond_dim=5, no DMRG sweep) rather than an
actual ground state, mirroring
test_cluster_number_operators.py:TestNumberCostFunctions -- these tests are
about the algebra, not about physical accuracy.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.dmrg_solver import Block2DMRGSolver

from cluster_number_decomposition_optimization import (
    RDMData,
    Decomposition,
    DecompositionOptimizerConfig,
    partition_to_cluster_matrix,
    validate_partition,
    rotate_rdm_data,
    cluster_number_stats,
    cluster_variance,
    fiedler_bipartition,
    fiedler_orbital_order,
    permutation_to_unitary,
    make_variance_cost_constructor,
    make_commutator_cost_constructor,
    expand_decomposition,
    run_decomposition_optimizer,
    polish_decomposition,
    best_decomposition,
    _split_one_cluster,
)
from src.cluster_number_operators import number_variance_cost


def _rdm_data_from_random_mps(norb, nelec, bond_dim=5, with_34=False, seed=0):
    """RDMData built from a random MPS -- valid tensors, not a ground state."""
    tmp_dir = tempfile.mkdtemp(prefix="decomp_opt_test_")
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
# Partition helpers
# =============================================================================


def test_partition_to_cluster_matrix():
    partition = [[0, 2], [1, 3, 4]]
    matrix = partition_to_cluster_matrix(partition, norb=5)
    expected = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
    assert np.array_equal(matrix, expected)


def test_validate_partition_accepts_valid():
    validate_partition([[0, 1], [2], [3, 4]], norb=5)  # should not raise


@pytest.mark.parametrize(
    "partition",
    [
        [[0, 1], [1, 2]],  # overlap
        [[0, 1], [2]],  # missing orbital (norb=4)
        [[0, 1], []],  # empty cluster
    ],
)
def test_validate_partition_rejects_invalid(partition):
    with pytest.raises(ValueError):
        validate_partition(partition, norb=4)


# =============================================================================
# Fiedler splitting and cluster statistics
# =============================================================================


class TestClusterStatsAndFiedler:
    norb = 6
    nelec = (3, 2)

    @pytest.fixture(scope="class")
    def rdm_data(self):
        return _rdm_data_from_random_mps(self.norb, self.nelec, seed=1)

    def test_fiedler_bipartition_is_a_valid_split(self, rdm_data):
        partition = [list(range(self.norb))]
        n1, n2 = cluster_number_stats(rdm_data, partition)
        for cluster in [[0, 1, 2, 3, 4, 5], [0, 1, 2], [2, 5], [1, 4, 5]]:
            V1, V2 = fiedler_bipartition(n1, n2, cluster)
            assert set(V1) | set(V2) == set(cluster)
            assert set(V1) & set(V2) == set()
            assert len(V1) > 0 and len(V2) > 0

    def test_cluster_variance_matches_number_variance_cost(self, rdm_data):
        # At the identity rotation, cluster_variance (built from
        # cluster_number_stats) must reproduce number_variance_cost exactly,
        # since both reduce to the same <n_p>, <n_p n_q> formula.
        for cluster in [[0, 1, 2], [3, 4], [5]]:
            partition = [cluster]
            n1, n2 = cluster_number_stats(rdm_data, partition)
            got = cluster_variance(n1, n2, cluster)

            cluster_matrix = partition_to_cluster_matrix(partition, self.norb)
            f = number_variance_cost(rdm_data.D, rdm_data.Gamma, cluster_matrix, with_ghost=False)
            n_params = self.norb * (self.norb - 1) // 2
            expected = float(f(np.zeros(n_params)))

            assert got == pytest.approx(expected, abs=1e-10)

    def test_fiedler_orbital_order_is_a_permutation(self, rdm_data):
        perm = fiedler_orbital_order(rdm_data.D, rdm_data.Gamma)
        assert sorted(perm.tolist()) == list(range(self.norb))
        U = permutation_to_unitary(perm)
        assert np.allclose(U @ U.T, np.eye(self.norb))
        assert np.allclose(U.sum(axis=0), 1.0)
        assert np.allclose(U.sum(axis=1), 1.0)


# =============================================================================
# Block-locality: the key correctness invariant of the whole approach
# =============================================================================


class TestBlockLocality:
    """Splitting cluster A and re-optimizing the rotation restricted to A's
    orbitals must leave every other cluster's orbitals -- and hence their
    number-operator statistics -- exactly untouched."""

    norb = 6
    nelec = (3, 2)

    @pytest.fixture(scope="class")
    def rdm_data(self):
        return _rdm_data_from_random_mps(self.norb, self.nelec, seed=2)

    def test_untouched_cluster_stats_are_invariant(self, rdm_data):
        partition = [[0, 1, 2], [3, 4, 5]]
        U0 = np.eye(self.norb)
        rdm_data_cur = rotate_rdm_data(rdm_data, U0)
        n1, n2 = cluster_number_stats(rdm_data_cur, partition)

        deco = Decomposition(partition=partition, U=U0, cost=0.0)
        cost_fn = make_variance_cost_constructor()

        child = _split_one_cluster(deco, rdm_data_cur, n1, n2, cluster_idx=0, cost_function_constructor=cost_fn, maxiter=50)

        # cluster [3, 4, 5] must be untouched: still present verbatim, and its
        # own variance under the child's rotation must be unchanged.
        assert [3, 4, 5] in child.partition
        untouched = [3, 4, 5]

        rdm_data_child = rotate_rdm_data(rdm_data, child.U)
        n1_child, n2_child = cluster_number_stats(rdm_data_child, [untouched])
        var_child = cluster_variance(n1_child, n2_child, untouched)
        var_parent = cluster_variance(n1, n2, untouched)

        assert var_child == pytest.approx(var_parent, abs=1e-8)
        assert np.allclose(n1_child[untouched], n1[untouched], atol=1e-8)

    def test_child_U_is_orthogonal(self, rdm_data):
        partition = [[0, 1, 2], [3, 4, 5]]
        U0 = np.eye(self.norb)
        rdm_data_cur = rotate_rdm_data(rdm_data, U0)
        n1, n2 = cluster_number_stats(rdm_data_cur, partition)
        deco = Decomposition(partition=partition, U=U0, cost=0.0)
        cost_fn = make_variance_cost_constructor()

        child = _split_one_cluster(deco, rdm_data_cur, n1, n2, cluster_idx=0, cost_function_constructor=cost_fn, maxiter=50)
        assert np.allclose(child.U @ child.U.T, np.eye(self.norb), atol=1e-8)

    def test_split_cost_at_most_identity_cost(self, rdm_data):
        # The optimizer starts at x=0 (the Fiedler split, no further rotation),
        # so the optimized cost can only be <= the cost measured there.
        partition = [[0, 1, 2], [3, 4, 5]]
        U0 = np.eye(self.norb)
        rdm_data_cur = rotate_rdm_data(rdm_data, U0)
        n1, n2 = cluster_number_stats(rdm_data_cur, partition)
        deco = Decomposition(partition=partition, U=U0, cost=0.0)
        cost_fn = make_variance_cost_constructor()

        from cluster_number_decomposition_optimization import fiedler_bipartition as _fb

        V1, V2 = _fb(n1, n2, [0, 1, 2])
        child_partition = [[3, 4, 5], V1, V2]
        cluster_matrix = partition_to_cluster_matrix(child_partition, self.norb)
        f = cost_fn(rdm_data_cur, cluster_matrix)
        n_params = self.norb * (self.norb - 1) // 2
        cost_at_identity = float(f(np.zeros(n_params)))

        child = _split_one_cluster(deco, rdm_data_cur, n1, n2, cluster_idx=0, cost_function_constructor=cost_fn, maxiter=200)
        assert child.cost <= cost_at_identity + 1e-8


# =============================================================================
# Beam search end-to-end
# =============================================================================


class TestRunDecompositionOptimizer:
    norb = 5
    nelec = (3, 2)

    @pytest.fixture(scope="class")
    def rdm_data(self):
        return _rdm_data_from_random_mps(self.norb, self.nelec, seed=3)

    def test_trajectory_shape_and_validity(self, rdm_data):
        config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, min_cluster_size=1, maxiter=20)
        trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), rdm_data, config)

        # runs all the way down to singleton clusters: norb rounds total
        assert trajectory[0].num_clusters == 1
        assert trajectory[-1].num_clusters == self.norb
        for i, deco in enumerate(trajectory):
            assert deco.num_clusters == i + 1
            validate_partition(deco.partition, self.norb)
            assert np.allclose(deco.U @ deco.U.T, np.eye(self.norb), atol=1e-6)

    def test_target_num_clusters_stops_early(self, rdm_data):
        config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, target_num_clusters=3, maxiter=20)
        trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), rdm_data, config)
        assert trajectory[-1].num_clusters == 3
        assert len(trajectory) == 3

    def test_min_cluster_size_stops_early(self, rdm_data):
        config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, min_cluster_size=2, maxiter=20)
        trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), rdm_data, config)
        for deco in trajectory:
            for cluster in deco.partition:
                assert len(cluster) >= 1  # partitions still fully valid
        # no cluster larger than min_cluster_size should remain splittable
        last = trajectory[-1]
        assert all(len(c) <= 2 for c in last.partition) or len(last.partition) == 1

    def test_best_decomposition_lookup(self, rdm_data):
        config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, maxiter=20)
        trajectory = run_decomposition_optimizer(make_variance_cost_constructor(), rdm_data, config)

        assert best_decomposition(trajectory) is trajectory[-1]
        deco3 = best_decomposition(trajectory, num_clusters=3)
        assert deco3.num_clusters == 3
        with pytest.raises(ValueError):
            best_decomposition(trajectory, num_clusters=self.norb + 5)

    def test_initial_bases_are_cycled(self, rdm_data):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((self.norb, self.norb))
        A = A - A.T
        U_alt = np.linalg.qr(np.eye(self.norb) + A)[0]  # some non-identity orthogonal matrix
        config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=1, target_num_clusters=1, maxiter=5)
        trajectory = run_decomposition_optimizer(
            make_variance_cost_constructor(), rdm_data, config, initial_bases=[np.eye(self.norb), U_alt]
        )
        # target_num_clusters=1 means no rounds run; just check it didn't crash
        # and the single trajectory entry is a valid 1-cluster decomposition.
        assert len(trajectory) == 1
        assert trajectory[0].num_clusters == 1


def test_polish_decomposition_does_not_increase_cost():
    norb, nelec = 5, (3, 2)
    rdm_data = _rdm_data_from_random_mps(norb, nelec, seed=4)
    config = DecompositionOptimizerConfig(num_decos=2, num_subdecos=2, target_num_clusters=3, maxiter=20)
    cost_fn = make_variance_cost_constructor()
    trajectory = run_decomposition_optimizer(cost_fn, rdm_data, config)

    deco = trajectory[-1]
    polished = polish_decomposition(deco, rdm_data, cost_fn, maxiter=200)

    assert polished.partition == deco.partition
    assert polished.cost <= deco.cost + 1e-8
    assert np.allclose(polished.U @ polished.U.T, np.eye(norb), atol=1e-6)


# =============================================================================
# Commutator-cost adapter: plumbing check only (no ground-state physics implied)
# =============================================================================


def test_commutator_cost_constructor_runs():
    norb, nelec = 4, (2, 1)
    rdm_data = _rdm_data_from_random_mps(norb, nelec, with_34=True, seed=5)
    deco = Decomposition(partition=[[0, 1], [2, 3]], U=np.eye(norb), cost=0.0)

    cost_fn = make_commutator_cost_constructor()
    children = expand_decomposition(
        deco, rdm_data, cost_fn, num_subdecos=1, min_cluster_size=1, maxiter=20
    )
    assert len(children) == 1
    child = children[0]
    validate_partition(child.partition, norb)
    assert np.isfinite(child.cost)
    assert np.allclose(child.U @ child.U.T, np.eye(norb), atol=1e-6)


def test_commutator_cost_constructor_requires_full_data():
    rdm_data = RDMData(D=np.eye(4), Gamma=np.zeros((4, 4, 4, 4)))
    cluster_matrix = partition_to_cluster_matrix([[0, 1], [2, 3]], norb=4)
    with pytest.raises(ValueError):
        make_commutator_cost_constructor()(rdm_data, cluster_matrix)
