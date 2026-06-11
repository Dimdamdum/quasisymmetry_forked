import argparse
import pyscf
import ffsim
import numpy as np
import scipy
import json
from tqdm import tqdm
from pathlib import Path
from itertools import product
from math import comb

from chemistry import load_moldata, fcidump_data
from new_optimize import parity_matrix_to_quasisymmetries, x_to_rotation, get_fci


def symmetry_sectors(parity_matrix, norb, nelec):
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    bitstrings = ffsim.addresses_to_strings(range(dim), norb, nelec,
        bitstring_type=ffsim.BitstringType.INT, concatenate=False)

    bit_powers = 2**(np.arange(norb - 1, -1, -1))
    bit_masks = parity_matrix[:, ::-1] @ bit_powers

    sectors = {}
    for i in range(dim):
        ab_parities = bitstrings[0][i] ^ bitstrings[1][i]
        sector_label = tuple(
            (int.bit_count(int(ab_parities & q)) % 2
             for q in bit_masks)
        )
        sectors.setdefault(sector_label, []).append(i)

    return sectors


def subspace_matrix(A, support):
    # dim = support.shape[0]
    dim = len(support)

    A_sub = np.zeros((dim, dim), dtype="complex")

    for i, big_index in enumerate(support):
        x = np.zeros(A.shape[0], dtype="complex")
        x[big_index] = 1
        y = A @ x
        A_sub[:, i] = y[support]

    return A_sub


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the metrics")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    parity_matrix = np.loadtxt(args.parity_matrix, dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(parity_matrix,
                                                  moldata.norb,
                                                  moldata.nelec)

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    if args.U is not None:
        x = np.loadtxt(args.U)
        U = x_to_rotation(x, moldata.norb)
    else:
        U = np.eye(moldata.norb)

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(rotated_h,
                                            norb=moldata.norb,
                                            nelec=moldata.nelec)

    e_fci, fcivec = get_fci(dumpdata)
    print("FCI ", e_fci)

    print("qty of sectors ", len(sectors.keys()))

    print("Creating subspace Hamiltonians")

    sector_hamiltonians = {}
    for sector_label, sector_bitstrings in tqdm(sectors.items()):
        sector_hamiltonians[sector_label] = subspace_matrix(rotated_h_linop,
                                                            sector_bitstrings)

    sector_gs_pairs = {}

    smallest = 0
    lowest_sector_label = None
    print("Calculating sector eigenvalues")
    for sector_label, h_local in tqdm(sector_hamiltonians.items()):
        sector_gs_pairs[sector_label] = scipy.sparse.linalg.eigsh(
            h_local, which="SA", k=1)
        if sector_gs_pairs[sector_label][0] < smallest:
            smallest = sector_gs_pairs[sector_label][0]
            lowest_sector_label = sector_label
    print("Lowest sector energy and label")
    print(smallest, lowest_sector_label)

    full_space_vectors = np.zeros((rotated_h_linop.shape[0],
                                   len(sectors.keys())), dtype="complex")

    for i, (k, v) in enumerate(sectors.items()):
        full_space_vectors[v, i] = sector_gs_pairs[k][1].flatten()

    h_bo = full_space_vectors.T.conj() @ rotated_h_linop @ full_space_vectors

    w_bo, v_bo = np.linalg.eigh(h_bo)
    e_bo = np.min(w_bo)

    print("BO energy ", e_bo)






