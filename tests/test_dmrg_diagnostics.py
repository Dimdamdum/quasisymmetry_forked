"""Tests for MPS-native diagnostics (Phase 3) and orbital reordering (Phase 4)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openfermion as of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coupled_energy_core import one_shot_coupled_energy
from src.dmrg_diagnostics import (
    collect_sector_states,
    decoupled_energy_dmrg,
    entanglement_diagnostic,
    one_shot_coupled_energy_mps,
    prepare_parity_matrix,
)
from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    permute_integrals,
    permute_parity_matrix,
    restore_g2e,
)

FCIDUMP_PATH = (
    Path(__file__).resolve().parents[1] / "hamiltonians" / "sentest_5_d754.FCIDUMP"
)


def build_h_sparse(solver: Block2DMRGSolver):
    norb = solver.n_sites
    g2e = restore_g2e(solver.g2e, norb)
    op = of.FermionOperator((), float(solver.ecore))
    for p in range(norb):
        for q in range(norb):
            if abs(solver.h1e[p, q]) > 1e-14:
                for s in (0, 1):
                    op += of.FermionOperator(
                        ((2 * p + s, 1), (2 * q + s, 0)), float(solver.h1e[p, q])
                    )
    for p, q, r, s in np.ndindex(norb, norb, norb, norb):
        v = float(g2e[p, q, r, s])
        if abs(v) <= 1e-14:
            continue
        for s1 in (0, 1):
            for s2 in (0, 1):
                op += of.FermionOperator(
                    ((2 * p + s1, 1), (2 * r + s2, 1),
                     (2 * s + s2, 0), (2 * q + s1, 0)),
                    0.5 * v,
                )
    return of.get_sparse_operator(of.jordan_wigner(op), n_qubits=2 * norb)


def occupation_of_index(index, n_qubits):
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


def sector_support(n_qubits, n_alpha, n_beta):
    return np.array([
        i for i in range(1 << n_qubits)
        if sum(occupation_of_index(i, n_qubits)[0::2]) == n_alpha
        and sum(occupation_of_index(i, n_qubits)[1::2]) == n_beta
    ])


def parity_bits(parity_row, index, n_qubits, norb):
    parity_row = np.asarray(parity_row, dtype=int)
    occ = occupation_of_index(index, n_qubits)
    if parity_row.shape[0] == norb:
        spin_orbitals = [
            q for p in np.flatnonzero(parity_row) for q in (2 * p, 2 * p + 1)
        ]
    else:
        spin_orbitals = list(np.flatnonzero(parity_row))
    return sum(occ[q] for q in spin_orbitals) % 2


class TestPermuteHelpers(unittest.TestCase):
    def test_permute_parity_spatial_and_spin_resolved(self):
        perm = (2, 0, 1)
        spatial = np.array([[1, 0, 1], [0, 1, 0]])
        self.assertTrue(np.array_equal(
            permute_parity_matrix(spatial, perm),
            np.array([[1, 1, 0], [0, 0, 1]]),
        ))
        spin = np.array([[1, 1, 0, 0, 0, 0]])
        out = permute_parity_matrix(spin, perm)
        # orbital 0 -> new position 1
        self.assertEqual(out[0, 2], 1)
        self.assertEqual(out[0, 3], 1)

    def test_energy_invariant_under_fiedler_reorder(self):
        store = Path(tempfile.mkdtemp(prefix="dmrg_reorder_"))
        try:
            plain = Block2DMRGSolver.from_fcidump(
                FCIDUMP_PATH, store_dir=store / "plain"
            )
            e0 = plain.run_ground_state(
                DMRGConfig(max_bond_dim=80, n_sweeps=10)
            ).energy
            reordered = Block2DMRGSolver.from_fcidump(
                FCIDUMP_PATH, store_dir=store / "fiedler", reorder="fiedler"
            )
            e1 = reordered.run_ground_state(
                DMRGConfig(max_bond_dim=80, n_sweeps=10)
            ).energy
            self.assertNotEqual(
                reordered.orbital_permutation, tuple(range(reordered.n_sites))
            )
            self.assertAlmostEqual(e0, e1, delta=1e-6)
        finally:
            shutil.rmtree(store, ignore_errors=True)


class TestDMRGDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Path(tempfile.mkdtemp(prefix="dmrg_diag_"))
        cls.solver = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH, store_dir=cls.store / "gs"
        )
        cls.config = DMRGConfig(max_bond_dim=100, n_sweeps=12)
        cls.gs = cls.solver.run_ground_state(cls.config)
        cls.parity = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        cls.H = build_h_sparse(cls.solver)
        cls.psi = cls.solver.to_statevector()
        cls.norb = cls.solver.n_sites
        cls.n_qubits = 2 * cls.norb
        cls.n_alpha = (cls.solver.n_elec + cls.solver.spin) // 2
        cls.n_beta = cls.solver.n_elec - cls.n_alpha
        cls.support = sector_support(cls.n_qubits, cls.n_alpha, cls.n_beta)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.store, ignore_errors=True)

    def _exact_sector_ground(self, label):
        mask = np.ones(len(self.support), dtype=bool)
        for row, bit in zip(self.parity, label):
            bits = np.array([
                parity_bits(row, int(idx), self.n_qubits, self.norb)
                for idx in self.support
            ])
            mask &= bits == bit
        idxs = self.support[mask]
        h_sub = self.H[np.ix_(idxs, idxs)].toarray()
        return float(np.linalg.eigvalsh(h_sub)[0])

    def test_decoupled_matches_exact(self):
        report = decoupled_energy_dmrg(
            self.solver,
            self.parity,
            self.gs.energy,
            config=self.config,
            penalty=30.0,
            max_sectors=4,
        )
        exact = min(
            self._exact_sector_ground(label)
            for label in report.sector_energies
        )
        self.assertAlmostEqual(report.e_decoupled, exact, delta=1e-5)

    def test_sector_excited_states(self):
        label = (0, 0)
        roots = self.solver.sector_excited_states(
            self.parity, label, nroots=3, penalty=30.0,
            config=DMRGConfig(max_bond_dim=120, n_sweeps=16),
        )
        self.assertEqual(len(roots), 3)
        self.assertLessEqual(roots[0][0], roots[1][0])
        self.assertAlmostEqual(
            roots[0][0], self._exact_sector_ground(label), delta=1e-5
        )

    def test_coupled_energy_mps_matches_dense_pt(self):
        # Build a few sector eigenstates with DMRG and compare K path to
        # dense one-shot PT on the same vectors reconstructed as statevectors.
        labels = [(0, 0), (0, 1), (1, 0)]
        states = collect_sector_states(
            self.solver, self.parity, labels,
            nroots=2, penalty=30.0,
            config=DMRGConfig(max_bond_dim=120, n_sweeps=14),
        )
        e_mps, k_mps, conv_mps, chosen_mps = one_shot_coupled_energy_mps(
            self.solver, states, e_exact=self.gs.energy, tol=1e-3
        )

        candidates = []
        for state in states:
            ket = self.solver.get_mps(state.mps_tag)
            vec = self.solver.to_statevector(ket)
            candidates.append(
                (state.energy, state.sector_label, vec, state.block_index)
            )

        def apply_h(v):
            return self.H @ v

        result = one_shot_coupled_energy(
            candidates, apply_h, e_exact=self.gs.energy, tol=1e-3
        )
        e_dense, k_dense, conv_dense, chosen_dense = result.as_tuple()
        self.assertEqual(k_mps, k_dense)
        self.assertEqual(conv_mps, conv_dense)
        self.assertAlmostEqual(e_mps, e_dense, delta=1e-4)
        self.assertEqual(chosen_mps, chosen_dense)

    def test_entanglement_shapes(self):
        ent = entanglement_diagnostic(self.solver)
        self.assertEqual(len(ent.bipartite), self.norb - 1)
        self.assertEqual(ent.orbital_s1.shape, (self.norb,))
        self.assertEqual(ent.mutual_information.shape, (self.norb, self.norb))
        self.assertTrue(np.allclose(np.diag(ent.mutual_information), 0.0))
        self.assertTrue(np.all(ent.mutual_information >= -1e-8))

    def test_prepare_parity_with_reorder(self):
        reordered = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH,
            store_dir=self.store / "prep",
            reorder="fiedler",
        )
        remapped = prepare_parity_matrix(reordered, self.parity)
        self.assertEqual(remapped.shape, self.parity.shape)
        # Remap is a column permutation of the original.
        expected = permute_parity_matrix(
            self.parity, reordered.orbital_permutation
        )
        self.assertTrue(np.array_equal(remapped, expected))

    def test_decoupled_with_fiedler_uses_original_parity(self):
        """Original-order parity must not be double-remapped under reorder."""
        reordered = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH,
            store_dir=self.store / "dec_fiedler",
            reorder="fiedler",
        )
        config = DMRGConfig(max_bond_dim=100, n_sweeps=12)
        e_ref = reordered.run_ground_state(config).energy
        report = decoupled_energy_dmrg(
            reordered,
            self.parity,
            e_ref,
            config=config,
            penalty=30.0,
            max_sectors=4,
        )
        plain = decoupled_energy_dmrg(
            self.solver,
            self.parity,
            self.gs.energy,
            config=self.config,
            penalty=30.0,
            max_sectors=4,
        )
        self.assertAlmostEqual(
            report.e_decoupled, plain.e_decoupled, delta=1e-5
        )


if __name__ == "__main__":
    unittest.main()
