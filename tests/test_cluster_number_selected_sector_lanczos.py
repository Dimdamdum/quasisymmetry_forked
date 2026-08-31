# AI-generated

"""Tests for cluster_number_selected_sector_lanczos.py.

Ground truth strategy, chosen deliberately to avoid a cross-representation
risk this project has been bitten by before (a basis/bit-ordering mismatch
between two independently-built representations of "the same" physical
system): everything here that needs an exact answer -- the full-space
Hamiltonian's ground energy, sector dimensions, sector membership -- is
computed entirely within pyscf's OWN representation (pyscf.fci.cistring
string addresses, pyscf.fci.direct_spin1.FCI().kernel()), the same
representation cluster_number_selected_sector_lanczos.py itself uses. No
openfermion / Jordan-Wigner cross-check is used here (unlike
tests/test_cluster_number_sector_search.py's own ground truth, which
specifically needed to probe delicate RDM-contraction *formulas* and so
built an independent Fock-space representation on purpose) -- this file is
about the wiring and Krylov/coupling logic on top of RDM-contraction
scoring that's already covered elsewhere, not about re-deriving RDM
formulas, so staying inside one consistent representation is both simpler
and removes an entire class of possible bugs that isn't the point here.
"""

import sys
from itertools import product
from math import comb
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pyscf.fci.cistring
import pyscf.fci.direct_spin1
import pytest

from cluster_number_decomposition_optimization import (
    Decomposition,
    RDMData,
    partition_to_cluster_matrix,
)
from cluster_number_sector_search import (
    SectorRelevance,
    SectorSearchConfig,
    rank_relevant_sectors,
    sector_dimension,
)
from cluster_number_selected_sector_lanczos import (
    LanczosSearchConfig,
    build_full_hamiltonian_operator,
    candidate_leakage_weights,
    cluster_occupation_counts,
    integer_sector_supports,
    label_key,
    solve_decomposition_selected_sectors,
)


# =============================================================================
# Shared fixtures
# =============================================================================


def _random_h1e_eri(norb, seed):
    """Small random real Hamiltonian integrals: symmetric h1e, and eri with
    full 8-fold permutational symmetry via the C@C^T-style construction
    already used elsewhere in this codebase's own tests
    (tests/test_cluster_number_sector_search.py's _rdm_data_from_random_mps)."""
    rng = np.random.default_rng(seed)
    h1e = rng.standard_normal((norb, norb))
    h1e = h1e + h1e.T
    C = rng.standard_normal((norb, norb, norb))
    C = C + C.transpose(1, 0, 2)
    eri = np.einsum("pqk,rsk->pqrs", C, C)
    return h1e, eri


def _all_integer_sectors(cluster_sizes, nelec):
    """Every (N_0,...,N_{K-1}) with 0<=N_k<=2*cluster_sizes[k] summing to
    nelec[0]+nelec[1] -- brute-force enumeration, fine for the tiny cluster
    counts used in these tests."""
    n_tot = sum(nelec)
    ranges = [range(0, 2 * s + 1) for s in cluster_sizes]
    return [combo for combo in product(*ranges) if sum(combo) == n_tot]


def _transportation_distance(label, main_label):
    return 0.5 * sum(abs(a - b) for a, b in zip(label, main_label))


def _build_ranked_sectors(labels, main_label, cluster_sizes, nelec):
    """Hand-built SectorRelevance list (bypassing rank_relevant_sectors
    entirely) -- gives full, explicit control over exactly which sectors
    solve_decomposition_selected_sectors is offered, independent of any
    RDM-based ranking quality. weight_score/energy_score/energy_tier are
    irrelevant to solve_decomposition_selected_sectors (only .label and
    .elec_transfer are read), so arbitrary placeholders are used."""
    main_label = tuple(main_label)
    entries = []
    for label in labels:
        t = _transportation_distance(label, main_label)
        assert t == int(t)
        entries.append(
            SectorRelevance(
                label=tuple(label),
                weight_score=1.0,
                energy_score=None,
                energy_tier=None,
                elec_transfer=int(t),
                dimension=sector_dimension(tuple(label), cluster_sizes, nelec),
            )
        )
    return entries


# =============================================================================
# integer_sector_supports / cluster_occupation_counts / label_key
# =============================================================================


class TestClusterOccupationCounts:
    def test_matches_hand_popcount(self):
        # 4 modes, cluster masks: {0,1} -> 0b0011=3, {2,3} -> 0b1100=12
        strings = np.array([0b0000, 0b0001, 0b0110, 0b1111, 0b1010], dtype=np.int64)
        counts = cluster_occupation_counts(strings, [0b0011, 0b1100])
        expected = np.array(
            [
                [0, 0],  # 0000
                [1, 0],  # 0001
                [1, 1],  # 0110 -> bit1 in mask0, bit2 in mask1
                [2, 2],  # 1111
                [1, 1],  # 1010 -> bit1 in mask0, bit3 in mask1
            ]
        )
        assert np.array_equal(counts, expected)


class TestLabelKey:
    def test_multi_digit_labels_unambiguous(self):
        assert label_key((2, 10, 3)) == "2-10-3"
        assert label_key((0,)) == "0"
        # would be ambiguous ("2103" could be (2,1,0,3) or (21,0,3) etc.) without
        # the separator -- exactly what this replaces label_text for.
        assert label_key((21, 0, 3)) != label_key((2, 10, 3))


class TestIntegerSectorSupports:
    def _brute_force(self, cluster_masks, labels, norb, nelec):
        n_alpha, n_beta = nelec
        alpha_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_alpha))
        beta_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_beta))
        n_beta_strings = len(beta_strings)
        result = {label: [] for label in labels}
        for a_addr, a_str in enumerate(alpha_strings):
            for b_addr, b_str in enumerate(beta_strings):
                counts = tuple(
                    bin(int(a_str) & mask).count("1") + bin(int(b_str) & mask).count("1")
                    for mask in cluster_masks
                )
                if counts in result:
                    result[counts].append(a_addr * n_beta_strings + b_addr)
        return {label: sorted(addrs) for label, addrs in result.items()}

    @pytest.mark.parametrize(
        "norb,partition,nelec",
        [
            (3, [[0], [1, 2]], (1, 1)),
            (4, [[0], [1], [2], [3]], (2, 2)),
            (5, [[0], [1], [2], [3, 4]], (3, 2)),
        ],
    )
    def test_matches_brute_force_enumeration(self, norb, partition, nelec):
        cluster_sizes = [len(c) for c in partition]
        labels = _all_integer_sectors(cluster_sizes, nelec)
        indicator = partition_to_cluster_matrix(partition, norb)
        cluster_masks = [sum(1 << p for p in cluster) for cluster in partition]

        got = integer_sector_supports(indicator, labels, norb, nelec, print_progress=False)
        expected = self._brute_force(cluster_masks, labels, norb, nelec)

        for label in labels:
            assert got[label]["full_addresses"].tolist() == expected[label]
            assert got[label]["dimension"] == len(expected[label])
            assert got[label]["dimension"] == sector_dimension(label, cluster_sizes, nelec)

    def test_supports_are_disjoint_and_cover_full_space(self):
        norb, partition, nelec = 4, [[0], [1], [2], [3]], (2, 2)
        cluster_sizes = [len(c) for c in partition]
        labels = _all_integer_sectors(cluster_sizes, nelec)
        indicator = partition_to_cluster_matrix(partition, norb)

        supports = integer_sector_supports(indicator, labels, norb, nelec, print_progress=False)
        all_addresses = np.concatenate([s["full_addresses"] for s in supports.values()])
        full_dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
        assert len(all_addresses) == len(set(all_addresses.tolist())) == full_dim

    def test_rejects_wrong_width_label(self):
        indicator = partition_to_cluster_matrix([[0], [1]], 2)
        with pytest.raises(ValueError):
            integer_sector_supports(indicator, [(1,)], 2, (1, 1), print_progress=False)


# =============================================================================
# build_full_hamiltonian_operator
# =============================================================================


class TestBuildFullHamiltonianOperator:
    @pytest.mark.parametrize("norb,nelec,seed", [(3, (1, 1), 0), (4, (2, 1), 1)])
    def test_lowest_eigenvalue_matches_fci_kernel(self, norb, nelec, seed):
        h1e, eri = _random_h1e_eri(norb, seed)
        exact_energy, _civec = pyscf.fci.direct_spin1.FCI().kernel(h1e, eri, norb, nelec, verbose=0)

        operator, full_dimension = build_full_hamiltonian_operator(h1e, eri, norb, nelec)
        assert full_dimension == comb(norb, nelec[0]) * comb(norb, nelec[1])
        assert operator.dtype == np.float64

        identity = np.eye(full_dimension)
        dense = np.column_stack([operator @ identity[:, i] for i in range(full_dimension)])
        lowest = np.linalg.eigvalsh(dense)[0]
        assert lowest == pytest.approx(exact_energy, abs=1e-9)

    def test_operator_is_symmetric(self):
        norb, nelec = 4, (2, 2)
        h1e, eri = _random_h1e_eri(norb, seed=2)
        operator, full_dimension = build_full_hamiltonian_operator(h1e, eri, norb, nelec)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(full_dimension)
        y = rng.standard_normal(full_dimension)
        assert np.vdot(y, operator @ x) == pytest.approx(np.vdot(x, operator @ y), abs=1e-9)


# =============================================================================
# candidate_leakage_weights
# =============================================================================


class TestCandidateLeakageWeights:
    def test_weights_match_direct_summation(self):
        residual = np.array([1.0, 2.0, 0.0, -3.0, 0.5, 1j], dtype=np.complex128)
        supports = {
            (0,): {"full_addresses": np.array([0, 1])},
            (1,): {"full_addresses": np.array([3, 4])},
            (2,): {"full_addresses": np.array([5])},
        }
        weights, total = candidate_leakage_weights(residual, supports, [(0,), (1,), (2,)])
        weight_dict = dict(weights)
        assert weight_dict[(0,)] == pytest.approx(1.0**2 + 2.0**2)
        assert weight_dict[(1,)] == pytest.approx(3.0**2 + 0.5**2)
        assert weight_dict[(2,)] == pytest.approx(1.0)
        assert total == pytest.approx(float(np.vdot(residual, residual).real))

    def test_ranked_descending(self):
        residual = np.array([0.1, 5.0, 0.2])
        supports = {(0,): {"full_addresses": np.array([0])}, (1,): {"full_addresses": np.array([1])}}
        weights, _total = candidate_leakage_weights(residual, supports, [(0,), (1,)])
        assert [label for label, _w in weights] == [(1,), (0,)]

    def test_only_scores_requested_candidates(self):
        residual = np.array([1.0, 1.0, 1.0])
        supports = {
            (0,): {"full_addresses": np.array([0])},
            (1,): {"full_addresses": np.array([1])},
            (2,): {"full_addresses": np.array([2])},
        }
        weights, total = candidate_leakage_weights(residual, supports, [(0,)])
        assert [label for label, _w in weights] == [(0,)]
        assert total == pytest.approx(3.0)  # total is the FULL residual norm, not just requested labels


# =============================================================================
# solve_decomposition_selected_sectors
# =============================================================================


class TestSolveDecompositionSelectedSectors:
    @pytest.fixture
    def tiny_system(self):
        """norb=3, 3 singleton clusters, nelec=(1,1): full space dim 9,
        6 valid integer sectors (hand-verified: dims 1,1,1,2,2,2 sum to 9)."""
        norb = 3
        partition = [[0], [1], [2]]
        nelec = (1, 1)
        h1e, eri = _random_h1e_eri(norb, seed=7)
        exact_energy, _civec = pyscf.fci.direct_spin1.FCI().kernel(h1e, eri, norb, nelec, verbose=0)
        deco = Decomposition(partition=partition, U=np.eye(norb), cost=0.0)
        rdm_data = RDMData(D=np.eye(norb), Gamma=np.zeros((norb,) * 4), h1e=h1e, g2e_full=eri)
        cluster_sizes = [len(c) for c in partition]
        labels = _all_integer_sectors(cluster_sizes, nelec)
        main_label = max(labels, key=lambda label: sector_dimension(label, cluster_sizes, nelec))
        ranked_sectors = _build_ranked_sectors(labels, main_label, cluster_sizes, nelec)
        return {
            "deco": deco, "rdm_data": rdm_data, "nelec": nelec, "exact_energy": exact_energy,
            "ranked_sectors": ranked_sectors, "full_dim": comb(norb, 1) * comb(norb, 1),
        }

    def test_final_energy_variational_and_close_to_exact_with_full_coverage(self, tiny_system):
        config = LanczosSearchConfig(
            krylov_seed_depth=10, max_iterations=30, vectors_per_iteration=5,
            energy_tolerance=1e-10, leakage_threshold=1e-14,
        )
        result = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
            tiny_system["ranked_sectors"], config,
        )
        exact = tiny_system["exact_energy"]
        # Variational: a subspace-restricted ground energy can never undercut the
        # true minimum.
        assert result["final"]["energy"] >= exact - 1e-8
        assert result["anchor"]["energy"] >= exact - 1e-8
        # With every one of the tiny system's 6 sectors offered and generous
        # depth/iterations, the coupled subspace should reach (close to) the full
        # 9-dimensional space and hence the exact answer.
        assert result["final"]["energy"] == pytest.approx(exact, abs=1e-5)
        assert result["final"]["dimension"] <= tiny_system["full_dim"]

    def test_energy_improves_monotonically_with_more_iterations(self, tiny_system):
        energies = []
        for max_iterations in (1, 3, 8):
            config = LanczosSearchConfig(
                krylov_seed_depth=2, max_iterations=max_iterations, vectors_per_iteration=1,
                energy_tolerance=1e-12, leakage_threshold=1e-14,
            )
            result = solve_decomposition_selected_sectors(
                tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
                tiny_system["ranked_sectors"], config,
            )
            energies.append(result["final"]["energy"])
        # Rayleigh-Ritz monotonicity: each run's coupled subspace strictly
        # contains the previous (shorter) run's -- growing a variational
        # subspace can only lower (or match) the ground-state estimate.
        assert energies[0] >= energies[1] - 1e-9
        assert energies[1] >= energies[2] - 1e-9

    def test_anchor_alone_when_no_other_sectors_offered(self, tiny_system):
        anchor_only = [r for r in tiny_system["ranked_sectors"] if r.elec_transfer == 0]
        config = LanczosSearchConfig(max_iterations=5)
        result = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
            anchor_only, config,
        )
        assert result["final"]["stop_reason"] == "leakage_exhausted"
        assert result["final"]["dimension"] == 1
        assert result["final"]["energy"] == pytest.approx(result["anchor"]["energy"])

    def test_max_iterations_zero_returns_anchor_only(self, tiny_system):
        config = LanczosSearchConfig(max_iterations=0)
        result = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
            tiny_system["ranked_sectors"], config,
        )
        assert result["final"]["stop_reason"] == "max_iterations_reached"
        assert result["final"]["dimension"] == 1
        assert result["final"]["energy"] == pytest.approx(result["anchor"]["energy"])
        assert result["iterations"] == [{"iteration": 0, "energy": result["anchor"]["energy"], "delta_e": None}]

    def test_stop_reason_max_iterations_matrix_energy_consistency(self, tiny_system):
        # Regression test for the design-correction-#2 fix: force exhaustion
        # (max_iterations=1, so the for loop's single allowed round runs and
        # exhausts range(1) without ever hitting a break -- confirmed
        # empirically to reliably land on "max_iterations_reached" for this
        # fixture, unlike a larger max_iterations here, which this tiny
        # system's small sectors can fully saturate before the cap is even
        # reached, landing on "leakage_exhausted" instead) and confirm the
        # reported final energy is NOT stale relative to what a fresh
        # diagonalization of the implied final subspace would give. We can't
        # access `matrix`/`candidates` directly (not returned), but we CAN
        # confirm dimension growth actually happened (so a real extension DID
        # occur this run) and that the last iteration's own energy record
        # matches the reported final energy exactly -- the strongest
        # externally observable proxy for "the returned final state is the
        # one that was actually last computed, not a round behind."
        config_short = LanczosSearchConfig(
            krylov_seed_depth=1, max_iterations=1, vectors_per_iteration=1,
            energy_tolerance=1e-15, leakage_threshold=1e-14,
        )
        result_short = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
            tiny_system["ranked_sectors"], config_short,
        )
        assert result_short["final"]["stop_reason"] == "max_iterations_reached"
        assert result_short["final"]["dimension"] > 1  # a real extension happened
        # The last iteration's own energy record must equal the reported final
        # energy exactly -- this IS the consistency the fix restores.
        assert result_short["iterations"][-1]["energy"] == pytest.approx(
            result_short["final"]["energy"], abs=1e-12
        )

    def test_multi_round_sector_discovery_via_intermediate(self):
        # 4 singleton clusters, nelec=(2,2): anchor=(2,2,0,0), intermediate
        # bridge=(0,2,2,0) is t=2 from anchor (single H-application reach),
        # far target=(0,0,2,2) is t=4 from anchor (unreachable in one step,
        # by the body-rank<=2 structural fact) but t=2 from the intermediate
        # (reachable in a SECOND leakage round once the coupled state has
        # picked up intermediate character). This is exactly the scenario
        # design correction #1 fixes: without re-running leakage detection
        # against the full candidate list every round, the far target would
        # never be discovered at all.
        norb, partition, nelec = 4, [[0], [1], [2], [3]], (2, 2)
        cluster_sizes = [len(c) for c in partition]
        h1e, eri = _random_h1e_eri(norb, seed=11)
        deco = Decomposition(partition=partition, U=np.eye(norb), cost=0.0)
        rdm_data = RDMData(D=np.eye(norb), Gamma=np.zeros((norb,) * 4), h1e=h1e, g2e_full=eri)

        anchor = (2, 2, 0, 0)
        intermediate = (0, 2, 2, 0)
        far_target = (0, 0, 2, 2)
        assert _transportation_distance(intermediate, anchor) == 2
        assert _transportation_distance(far_target, anchor) == 4
        assert _transportation_distance(far_target, intermediate) == 2

        config = LanczosSearchConfig(
            krylov_seed_depth=6, max_iterations=15, vectors_per_iteration=3,
            energy_tolerance=1e-12, leakage_threshold=1e-12,
        )

        # (a) Without the intermediate: the far target can never show leakage
        # from the anchor alone (t=4 > single-H-application reach) and so must
        # never be reached.
        ranked_without = _build_ranked_sectors([anchor, far_target], anchor, cluster_sizes, nelec)
        result_without = solve_decomposition_selected_sectors(
            deco, rdm_data, nelec, 0.0, ranked_without, config
        )
        assert result_without["final"]["dimension"] == sector_dimension(anchor, cluster_sizes, nelec)

        # (b) With the intermediate bridging anchor -> far_target: the far
        # target should eventually get discovered and pulled in.
        ranked_with = _build_ranked_sectors(
            [anchor, intermediate, far_target], anchor, cluster_sizes, nelec
        )
        result_with = solve_decomposition_selected_sectors(
            deco, rdm_data, nelec, 0.0, ranked_with, config
        )
        anchor_dim = sector_dimension(anchor, cluster_sizes, nelec)
        intermediate_dim = sector_dimension(intermediate, cluster_sizes, nelec)
        assert result_with["final"]["dimension"] > anchor_dim + intermediate_dim, (
            "far_target's dimension was never folded in -- the multi-round "
            "sector-discovery fix appears not to be working"
        )

    def test_missing_anchor_raises(self, tiny_system):
        non_anchor = [r for r in tiny_system["ranked_sectors"] if r.elec_transfer != 0]
        with pytest.raises(ValueError, match="anchor"):
            solve_decomposition_selected_sectors(
                tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
                non_anchor, LanczosSearchConfig(),
            )

    def test_total_energy_with_ecore_adds_ecore(self, tiny_system):
        config = LanczosSearchConfig(max_iterations=1)
        result = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 1.2345,
            tiny_system["ranked_sectors"], config,
        )
        assert result["final"]["total_energy_with_ecore"] == pytest.approx(
            result["final"]["energy"] + 1.2345
        )

    def test_result_is_json_serializable(self, tiny_system):
        import json

        config = LanczosSearchConfig(max_iterations=3)
        result = solve_decomposition_selected_sectors(
            tiny_system["deco"], tiny_system["rdm_data"], tiny_system["nelec"], 0.0,
            tiny_system["ranked_sectors"], config,
        )
        json.dumps(result)  # raises TypeError on any stray numpy scalar/array


# =============================================================================
# End-to-end integration with the real rank_relevant_sectors (Part 1)
# =============================================================================


class TestEndToEndWithRankRelevantSectors:
    def test_pipeline_wiring_produces_variational_result(self):
        norb, partition, nelec = 4, [[0, 1], [2], [3]], (2, 1)
        h1e, eri = _random_h1e_eri(norb, seed=42)
        exact_energy, _civec = pyscf.fci.direct_spin1.FCI().kernel(h1e, eri, norb, nelec, verbose=0)

        deco = Decomposition(partition=partition, U=np.eye(norb), cost=0.0)
        # D/Gamma need not be physical for Part 2's correctness (it only consumes
        # h1e/g2e_full via rotate_rdm_data) -- an identity-like placeholder is
        # enough for rank_relevant_sectors to run without crashing and produce
        # *some* reasonable candidate list; h1e/eri are the real, fixed
        # Hamiltonian whose exact energy is checked against below.
        D = np.diag([float(nelec[0] + nelec[1]) / norb] * norb)
        rdm_data = RDMData(D=D, Gamma=np.zeros((norb,) * 4), h1e=h1e, g2e_full=eri)

        search_config = SectorSearchConfig(max_elec_transfer=4)
        ranked = rank_relevant_sectors(deco, rdm_data, nelec, search_config)
        assert any(r.elec_transfer == 0 for r in ranked)

        lanczos_config = LanczosSearchConfig(
            krylov_seed_depth=6, max_iterations=15, vectors_per_iteration=3, energy_tolerance=1e-10,
        )
        result = solve_decomposition_selected_sectors(deco, rdm_data, nelec, 0.0, ranked, lanczos_config)

        assert result["final"]["energy"] >= exact_energy - 1e-8
        assert result["final"]["energy"] <= result["anchor"]["energy"] + 1e-9
