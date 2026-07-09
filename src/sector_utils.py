""" utilities for symmetry sectors."""

from math import comb

import ffsim
import numpy as np


def symmetry_sectors(parity_matrix, norb, nelec):
    """Group determinant addresses by their symmetry-sector labels."""
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])

    if parity_matrix.shape[1] == norb:
        bitstrings = ffsim.addresses_to_strings(
            range(dim),
            norb,
            nelec,
            bitstring_type=ffsim.BitstringType.INT,
            concatenate=False,
        )
        bit_powers = 2 ** np.arange(norb - 1, -1, -1)
        bit_masks = parity_matrix[:, ::-1] @ bit_powers

        sectors = {}
        for address in range(dim):
            alpha_bits = bitstrings[0][address]
            beta_bits = bitstrings[1][address]
            pair_parity_bits = alpha_bits ^ beta_bits
            label = tuple(
                int.bit_count(int(pair_parity_bits & mask)) % 2
                for mask in bit_masks
            )
            sectors.setdefault(label, []).append(address)
        return sectors

    if parity_matrix.shape[1] == 2 * norb:
        bit_powers = 2 ** np.arange(2 * norb - 1, -1, -1)
        reversed_interleaved_order = np.concatenate(
            (
                np.arange(2 * norb - 2, -1, -2),
                np.arange(2 * norb - 1, -1, -2),
            )
        )
        bit_masks = parity_matrix[:, reversed_interleaved_order] @ bit_powers
        bitstrings = ffsim.addresses_to_strings(
            range(dim),
            norb,
            nelec,
            bitstring_type=ffsim.BitstringType.INT,
            concatenate=True,
        )

        sectors = {}
        for address in range(dim):
            label = tuple(
                int.bit_count(int(bitstrings[address] & mask)) % 2
                for mask in bit_masks
            )
            sectors.setdefault(label, []).append(address)
        return sectors

    raise ValueError("parity_matrix must have norb or 2*norb columns")


def subspace_matrix(A, support):
    """Build the dense matrix A[support, support] using matrix-vector products."""
    dim = len(support)
    support = np.array(support, dtype=int)
    A_sub = np.zeros((dim, dim), dtype=complex)

    for small_col, big_col in enumerate(support):
        x = np.zeros(A.shape[0], dtype=complex)
        x[big_col] = 1.0
        y = A @ x
        A_sub[:, small_col] = y[support]

    return A_sub

