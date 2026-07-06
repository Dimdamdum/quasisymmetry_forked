"""Sanity checks for PT-screened coupled-energy diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.coupled_energy_core import (
    all_sector_eigenpair_candidates,
    augment_h_proj,
    h_cols_from_h_vecs,
    perturbation_may_improve,
    projected_ground_energy_dense,
    trial_ground_energy_incremental,
)
from src.energy_diagnostics import (
    coupled_energy_perturbation,
    decoupled_energy_test,
    diagonalize_sector_blocks,
)


def _toy_hamiltonian_and_sectors():
    dim = 6
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.05
    h[2, 2] = 0.1
    h[3, 3] = 0.11
    h[4, 4] = 0.2
    h[5, 5] = 0.3
    coupling = 0.02
    h[0, 2] = h[2, 0] = coupling

    sectors = {"A": [0, 1], "B": [2, 3], "C": [4], "D": [5]}
    sector_data = diagonalize_sector_blocks(lambda v: h @ v, sectors, dim)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sectors, sector_data, e_exact


def _toy_with_excited_cross_coupling():
    dim = 3
    h = np.zeros((dim, dim), dtype=np.complex128)
    h[0, 0] = 0.0
    h[1, 1] = 0.08
    h[2, 2] = 0.12
    h[0, 2] = h[2, 0] = 0.02
    h[1, 2] = h[2, 1] = 0.05

    sectors = {"A": [0, 1], "B": [2]}
    sector_data = diagonalize_sector_blocks(lambda v: h @ v, sectors, dim)
    e_exact = float(np.linalg.eigvalsh(0.5 * (h + h.conj().T))[0])
    return h, sectors, sector_data, e_exact


class CoupledEnergySanityTests(unittest.TestCase):
    def test_pt_matches_exact_on_toy(self):
        h, _sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        e_pt, k_pt, converged, _ = coupled_energy_perturbation(
            lambda v: h @ v, sector_data, e_exact=e_exact, tol=1e-10
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(e_pt, e_exact, places=10)
        self.assertGreaterEqual(k_pt, 2)

    def test_pt_finds_excited_cross_coupling(self):
        h, _sectors, sector_data, e_exact = _toy_with_excited_cross_coupling()
        e_pt, k_pt, converged, keys = coupled_energy_perturbation(
            lambda v: h @ v, sector_data, e_exact=e_exact, tol=1e-10
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(e_pt, e_exact, places=10)
        self.assertGreater(k_pt, 2)
        block_indices = {idx for _key, idx in keys}
        self.assertIn(1, block_indices)

    def test_incremental_matches_dense_projection(self):
        h, _sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        candidates = all_sector_eigenpair_candidates(sector_data)
        vecs = [candidates[i][2] for i in range(3)]
        e_dense = projected_ground_energy_dense(h, vecs)
        h_vecs = [h @ v for v in vecs[:2]]
        h_cols = h_cols_from_h_vecs(h_vecs, vecs[2])
        e0 = float(np.real(np.vdot(vecs[0], h @ vecs[0])))
        e1 = float(np.real(np.vdot(vecs[1], h @ vecs[1])))
        e2 = float(np.real(np.vdot(vecs[2], h @ vecs[2])))
        h_proj = np.array([[e0]], dtype=np.complex128)
        h_proj = augment_h_proj(h_proj, [complex(np.vdot(h @ vecs[0], vecs[1]))], e1)
        e_incr = trial_ground_energy_incremental(
            h_proj, h_cols, e2
        )
        self.assertAlmostEqual(e_incr, e_dense, places=10)

    def test_decoupled_energy(self):
        h, sectors, sector_data, e_exact = _toy_hamiltonian_and_sectors()
        sector_gs_pairs = {k: (d["evals"], np.column_stack(d["evecs_full"])) for k, d in sector_data.items()}
        e_dec, best_key, _dim = decoupled_energy_test(sectors, sector_gs_pairs)
        self.assertAlmostEqual(e_dec, 0.0, places=10)
        self.assertEqual(best_key, "A")
        self.assertGreater(e_dec, e_exact)


if __name__ == "__main__":
    unittest.main()
