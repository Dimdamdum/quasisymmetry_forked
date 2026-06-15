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


def submatrix_eigenvalues_to_target(A: np.ndarray, e_target: float) -> int:
    """Start in the upper left corner of A, take a KxK block and calculate its
    lowest eignvalue. Return the smallest K that yields energy below e_target
    or -1 if no such thing can be found"""
    e, _ = scipy.sparse.linalg.eigsh(A, which="SA", k=1)

    if e > e_target:
        return -1
    elif A[0, 0] < e_target:
        return 1
    else:
        for vec_count in tqdm(range(2, A.shape[0])):
            submatrix = A[::vec_count, :][:, ::vec_count]
            e, _ = scipy.sparse.linalg.eigsh(A, which="SA", k=1)
            if e < e_target:
                return vec_count
        else:
            raise ValueError("this should never happen")




if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the metrics")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    parser.add_argument("--states_per_sector", type=int, default=10)
    parser.add_argument("--K_method", default="PT")
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
        if args.states_per_sector <= h_local.shape[0] - 2:
            sector_gs_pairs[sector_label] = scipy.sparse.linalg.eigsh(
                h_local, which="SA", k=args.states_per_sector)
        else:
            sector_gs_pairs[sector_label] = np.linalg.eigh(h_local)
        if np.min(sector_gs_pairs[sector_label][0]) < smallest:
            smallest = np.min(sector_gs_pairs[sector_label][0])
            lowest_sector_label = sector_label
    print("Lowest sector energy and label")
    print(smallest, lowest_sector_label)



    # joint_space_dimension = sum([w[0].shape[0] for w in sector_gs_pairs.values()])
    #
    # full_space_vectors = np.zeros((rotated_h_linop.shape[0],
    #                                joint_space_dimension), dtype="complex")

    full_space_vectors = []
    for k, v in sectors.items():
        full_space_vectors_in_sector = np.zeros((rotated_h_linop.shape[0],
                                                 sector_gs_pairs[k][0].shape[0]),
                                                dtype="complex")
        full_space_vectors_in_sector[v, :] = sector_gs_pairs[k][1]
        full_space_vectors.append(full_space_vectors_in_sector)

    full_space_vectors_cat = np.concatenate(full_space_vectors, axis=1)

    h_subspace = full_space_vectors_cat.T.conj() @ rotated_h_linop @ full_space_vectors_cat

    w_subspace, _ = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")
    print("Coupled energy", w_subspace)
    if w_subspace - e_fci > 0.0016:
        print("Not enough states to reach chemical accuracy")
        quit()
    else:
        lowest_energy_vector_index = np.argmin(np.diag(h_subspace))
        pt_coefficients_numerator = abs(
            h_subspace[:, lowest_energy_vector_index])**2
        pt_coefficients_denominator = (np.diag(h_subspace)
            - h_subspace[lowest_energy_vector_index, lowest_energy_vector_index])
        pt_coefficients = pt_coefficients_numerator / pt_coefficients_denominator
        pt_coefficients = np.nan_to_num(pt_coefficients, posinf=0, neginf=0,
                                        nan=0)
        pt_coeffs_order = np.argsort(abs(pt_coefficients))[::-1]
        h_subspace_reordered = h_subspace[:, pt_coeffs_order][pt_coeffs_order, :]
        K = submatrix_eigenvalues_to_target(h_subspace_reordered,
                                            e_fci + 0.0016)
        print("K ", K)


    #
    # for i, (k, v) in enumerate(sectors.items()):
    #     full_space_vectors[v, i] = sector_gs_pairs[k][1].flatten()
    #
    # h_bo = full_space_vectors.T.conj() @ rotated_h_linop @ full_space_vectors
    #
    # w_bo, v_bo = np.linalg.eigh(h_bo)
    # e_bo = np.min(w_bo)
    #
    # print("BO energy ", e_bo)






