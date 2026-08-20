"""Validate MPS-native orbital costs against exact diagonalization."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

import numpy as np
import openfermion as of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dmrg_costs import DMRGOrbitalCosts, MultiplyConfig
from src.dmrg_solver import (
    Block2DMRGSolver,
    DMRGConfig,
    restore_g2e,
    rotation_from_parameters,
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


def build_stilde_sparse(rotation_row: np.ndarray, norb: int, alpha=True, beta=True):
    D = np.outer(rotation_row, rotation_row)
    st = of.FermionOperator((), 1.0)
    for q in range(norb):
        for r in range(norb):
            c = float(-2.0 * D[q, r])
            if abs(c) <= 1e-14:
                continue
            if alpha:
                st += of.FermionOperator(((2 * q, 1), (2 * r, 0)), c)
            if beta:
                st += of.FermionOperator(((2 * q + 1, 1), (2 * r + 1, 0)), c)
    if alpha and beta:
        for q, r, s, t in np.ndindex(norb, norb, norb, norb):
            c = float(4.0 * D[q, r] * D[s, t])
            if abs(c) > 1e-14:
                st += of.FermionOperator(
                    ((2 * q, 1), (2 * s + 1, 1), (2 * t + 1, 0), (2 * r, 0)), c
                )
    return of.get_sparse_operator(of.jordan_wigner(st), n_qubits=2 * norb)


def product_stilde(rotation: np.ndarray, parity_row: np.ndarray, norb: int):
    """Sparse ``U^dagger S U`` for a (possibly multi-orbital) parity row."""
    parity_row = np.asarray(parity_row, dtype=int)
    eye = of.get_sparse_operator(of.FermionOperator((), 1.0), n_qubits=2 * norb)
    op = eye
    if parity_row.shape[0] == norb:
        for p in np.flatnonzero(parity_row):
            op = op @ build_stilde_sparse(rotation[int(p)], norb)
    elif parity_row.shape[0] == 2 * norb:
        for p in range(norb):
            alpha = bool(parity_row[2 * p])
            beta = bool(parity_row[2 * p + 1])
            if alpha or beta:
                op = op @ build_stilde_sparse(
                    rotation[p], norb, alpha=alpha, beta=beta
                )
    else:
        raise ValueError("bad parity row")
    return op


class TestDMRGOrbitalCosts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Path(tempfile.mkdtemp(prefix="dmrg_costs_"))
        cls.solver = Block2DMRGSolver.from_fcidump(
            FCIDUMP_PATH, store_dir=cls.store / "gs"
        )
        cls.solver.run_ground_state(DMRGConfig(max_bond_dim=100, n_sweeps=12))
        cls.psi = cls.solver.to_statevector()
        cls.H = build_h_sparse(cls.solver)
        cls.norb = cls.solver.n_sites
        cls.parity = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ])
        cls.costs = DMRGOrbitalCosts(
            cls.solver,
            cls.parity,
            multiply=MultiplyConfig(bond_dim=120, n_sweeps=10, bra_bond_dim_factor=1.25),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.store, ignore_errors=True)

    def _exact_costs(self, x):
        U = rotation_from_parameters(x, self.norb)
        nc = 0.0
        var = 0.0
        for row in self.parity:
            St = product_stilde(U, row, self.norb)
            nc += float(np.linalg.norm((self.H @ St - St @ self.H) @ self.psi) ** 2)
            exp = float(np.vdot(self.psi, St @ self.psi).real)
            var += 1.0 - exp ** 2
        return nc, var

    def test_variance_matches_exact_at_identity(self):
        x = np.zeros(comb(self.norb, 2))
        _, var_exact = self._exact_costs(x)
        var = self.costs.variance(x)
        self.assertAlmostEqual(var, var_exact, places=6)

    def test_commutator_matches_exact_at_identity(self):
        x = np.zeros(comb(self.norb, 2))
        nc_exact, _ = self._exact_costs(x)
        nc = self.costs.commutator(x)
        self.assertAlmostEqual(nc, nc_exact, delta=1e-5)

    def test_costs_match_exact_at_finite_rotation(self):
        rng = np.random.default_rng(3)
        x = rng.normal(scale=0.12, size=comb(self.norb, 2))
        nc_exact, var_exact = self._exact_costs(x)
        var = self.costs.variance(x)
        nc = self.costs.commutator(x)
        self.assertAlmostEqual(var, var_exact, places=5)
        self.assertAlmostEqual(nc, nc_exact, delta=5e-5)

    def test_spin_resolved_row(self):
        row = np.zeros(2 * self.norb, dtype=int)
        row[0] = 1  # alpha parity on orbital 0
        costs = DMRGOrbitalCosts(
            self.solver,
            row.reshape(1, -1),
            multiply=MultiplyConfig(bond_dim=100, n_sweeps=8),
        )
        rng = np.random.default_rng(5)
        x = rng.normal(scale=0.1, size=comb(self.norb, 2))
        U = rotation_from_parameters(x, self.norb)
        St = product_stilde(U, row, self.norb)
        var_exact = 1.0 - float(np.vdot(self.psi, St @ self.psi).real) ** 2
        nc_exact = float(np.linalg.norm((self.H @ St - St @ self.H) @ self.psi) ** 2)
        self.assertAlmostEqual(costs.variance(x), var_exact, places=5)
        self.assertAlmostEqual(costs.commutator(x), nc_exact, delta=5e-5)

    def test_cost_function_dispatch(self):
        x = np.zeros(comb(self.norb, 2))
        self.assertAlmostEqual(
            self.costs.cost_function("NC")(x), self.costs.commutator(x), places=10
        )
        self.assertAlmostEqual(
            self.costs.cost_function("variance")(x), self.costs.variance(x), places=10
        )

    def test_explicit_multiply_bond_dim_not_rescaled(self):
        # multiply.bond_dim, when set explicitly, is the literal fit bond
        # dimension -- bra_bond_dim_factor must not also be applied on top of
        # it (that was a Phase 1 double-scaling bug: an explicit override
        # silently came out ~1.5x larger than requested).
        costs = DMRGOrbitalCosts(
            self.solver, self.parity,
            multiply=MultiplyConfig(bond_dim=42, bra_bond_dim_factor=1.5),
        )
        self.assertEqual(costs._bra_bond_dim(), 42)

    def test_two_instances_same_store_do_not_interfere(self):
        # Every DMRGOrbitalCosts instance tags its working MPS copies with a
        # unique per-instance id (see the module-level comment on
        # _next_instance_id). Before that fix, a second instance sharing the
        # same store_dir could collide with the first's "COST_KET" tag and
        # silently corrupt it mid-evaluation.
        x = np.zeros(comb(self.norb, 2))
        multiply = MultiplyConfig(bond_dim=60, n_sweeps=4)
        c1 = DMRGOrbitalCosts(self.solver, self.parity, multiply=multiply)
        v1_before = c1.commutator(x)
        c2 = DMRGOrbitalCosts(self.solver, self.parity, multiply=multiply)
        _ = c2.commutator(x)
        v1_after = c1.commutator(x)
        self.assertAlmostEqual(v1_before, v1_after, places=8)

    def test_analytic_gradient_matches_finite_difference(self):
        # Regression test for the Tier 1a validation done ad hoc during Phase
        # 2 (diagnostics/phase2_analytic_gradient/): commutator_and_gradient's
        # analytic gradient should agree with a (large-step, truncation-noise
        # -tolerant) central finite difference of commutator() itself.
        row = np.array([[1, 0, 0, 0, 0]])  # support-1: required by the analytic path
        costs = DMRGOrbitalCosts(
            self.solver, row,
            multiply=MultiplyConfig(bond_dim=100, n_sweeps=8),
        )
        rng = np.random.default_rng(7)
        x = rng.normal(scale=0.1, size=comb(self.norb, 2))
        _, grad = costs.commutator_and_gradient(x)

        h = 1e-4
        for i in (0, comb(self.norb, 2) // 2, comb(self.norb, 2) - 1):
            e_i = np.zeros_like(x)
            e_i[i] = 1.0
            fd = (costs.commutator(x + h * e_i) - costs.commutator(x - h * e_i)) / (2 * h)
            self.assertAlmostEqual(
                grad[i], fd, delta=max(abs(fd), abs(grad[i])) * 0.05 + 1e-6
            )


if __name__ == "__main__":
    unittest.main()
