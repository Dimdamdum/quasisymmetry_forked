"""Unit tests for full vs irrep-restricted orbital-rotation packing."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from src.orbital_rotation import (
    full_pairs,
    irrep_pairs,
    n_params,
    pairs_from_oo_data,
    params_to_U,
    rotation_from_oo_data,
)


def _block_sizes_to_irreps(sizes):
    labels = []
    for g, size in enumerate(sizes):
        labels.extend([g] * size)
    return np.asarray(labels, dtype=int)


class TestOrbitalRotationPacking:
    def test_full_param_count_and_identity(self):
        norb = 7
        assert n_params(norb) == comb(norb, 2)
        assert len(full_pairs(norb)) == comb(norb, 2)
        U = params_to_U(np.zeros(comb(norb, 2)), norb)
        np.testing.assert_allclose(U, np.eye(norb), atol=1e-14)

    def test_full_orthogonal(self):
        norb = 5
        rng = np.random.default_rng(0)
        x = rng.normal(scale=0.2, size=comb(norb, 2))
        U = params_to_U(x, norb)
        np.testing.assert_allclose(U @ U.T, np.eye(norb), atol=1e-12)

    def test_h2o_sto3g_irrep_count(self):
        # A1 x4, B1 x2, B2 x1 → N_sym = 6 + 1 + 0 = 7
        irreps = _block_sizes_to_irreps((4, 2, 1))
        pairs = irrep_pairs(irreps)
        assert n_params(len(irreps), pairs) == 7
        assert len(pairs) == 7

    def test_n2_sto3g_irrep_count(self):
        # Ag x3, B1u x3, then four singlets → N_sym = 3 + 3 = 6
        irreps = _block_sizes_to_irreps((3, 3, 1, 1, 1, 1))
        pairs = irrep_pairs(irreps)
        assert n_params(len(irreps), pairs) == 6

    def test_h2o_631g_and_n2_631g_counts(self):
        assert n_params(13, irrep_pairs(_block_sizes_to_irreps((7, 4, 2)))) == 28
        assert (
            n_params(18, irrep_pairs(_block_sizes_to_irreps((5, 5, 2, 2, 2, 2))))
            == 24
        )

    def test_restricted_is_block_diagonal(self):
        irreps = _block_sizes_to_irreps((4, 2, 1))
        norb = len(irreps)
        pairs = irrep_pairs(irreps)
        rng = np.random.default_rng(1)
        x = rng.normal(scale=0.3, size=len(pairs))
        U = params_to_U(x, norb, pairs)
        np.testing.assert_allclose(U @ U.T, np.eye(norb), atol=1e-12)
        for i in range(norb):
            for j in range(norb):
                if irreps[i] != irreps[j]:
                    assert abs(U[i, j]) < 1e-12

    def test_params_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="parameter length"):
            params_to_U(np.zeros(3), 5)

    def test_oo_data_roundtrip(self):
        irreps = _block_sizes_to_irreps((3, 3, 1, 1))
        norb = len(irreps)
        pairs = irrep_pairs(irreps)
        x = np.linspace(0.01, 0.1, len(pairs))
        data = {
            "orbital_rotation": "irrep",
            "irreps": irreps.tolist(),
            "rotation": x.tolist(),
        }
        assert pairs_from_oo_data(data, norb) == pairs
        U = rotation_from_oo_data(data, norb)
        np.testing.assert_allclose(U, params_to_U(x, norb, pairs))

    def test_oo_data_full_default(self):
        norb = 4
        x = np.zeros(comb(norb, 2))
        data = {"rotation": x.tolist()}
        assert pairs_from_oo_data(data, norb) is None
        np.testing.assert_allclose(rotation_from_oo_data(data, norb), np.eye(norb))
