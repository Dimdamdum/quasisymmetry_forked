"""Unit tests for PySCF Davidson on sector Hamiltonian blocks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.davidson_solver import solve_sector_davidson


class TestDavidsonSolver(unittest.TestCase):
    def test_davidson_roots_match_eigh(self):
        rng = np.random.default_rng(0)
        dim = 24
        a = rng.normal(size=(dim, dim))
        h = 0.5 * (a + a.T)
        np.fill_diagonal(h, np.arange(dim, dtype=float) * 0.3)

        nroots = 3
        energies, vectors, meta = solve_sector_davidson(
            h, nroots=nroots, tol=1e-10, max_cycle=80, max_space=16
        )
        ref_e, ref_v = np.linalg.eigh(h)

        self.assertEqual(meta["solver"], "davidson")
        self.assertTrue(meta["converged"])
        self.assertEqual(meta["nroots"], nroots)
        self.assertEqual(meta["dimension"], dim)
        self.assertGreater(meta["n_cycle"], 0)
        self.assertGreater(meta["elapsed_seconds"], 0.0)
        np.testing.assert_allclose(energies, ref_e[:nroots], atol=1e-8, rtol=0)

        for i in range(nroots):
            overlap = abs(np.vdot(vectors[:, i], ref_v[:, i]))
            self.assertGreater(overlap, 1.0 - 1e-6)

    def test_small_block_falls_back_to_dense(self):
        h = np.array([[2.0, 0.1], [0.1, 3.0]], dtype=float)
        energies, vectors, meta = solve_sector_davidson(h, nroots=1)
        ref_e, ref_v = np.linalg.eigh(h)

        self.assertEqual(meta["solver"], "dense")
        self.assertTrue(meta["converged"])
        self.assertEqual(meta["n_cycle"], 0)
        np.testing.assert_allclose(energies, ref_e[:1], atol=1e-12)
        self.assertGreater(abs(np.vdot(vectors[:, 0], ref_v[:, 0])), 1.0 - 1e-12)


if __name__ == "__main__":
    unittest.main()
