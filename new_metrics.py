import argparse
import pyscf
import ffsim
import numpy as np
import scipy
import scipy.sparse.linalg
import json
from tqdm import tqdm
from pathlib import Path
from itertools import product
from math import comb
import matplotlib.pyplot as plt
import bisect
from uuid import uuid4

from chemistry import load_moldata, fcidump_data, CHEMICAL_PRECISION
from new_optimize import parity_matrix_to_quasisymmetries, x_to_rotation, get_fci, commutator_cost


def symmetry_sectors(parity_matrix, norb, nelec):
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    if parity_matrix.shape[1] == norb:
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
    elif parity_matrix.shape[1] == 2 * norb:
        bit_powers = 2 ** (np.arange(2 * norb - 1, -1, -1))
        reversed_interleaved_order = np.concatenate(
            (np.arange(2 * norb - 2, -1, -2),
             np.arange(2 * norb - 1, -1, -2),
            )
        )
        bit_masks = parity_matrix[:, reversed_interleaved_order] @ bit_powers
        bitstrings = ffsim.addresses_to_strings(range(dim), norb, nelec,
            bitstring_type=ffsim.BitstringType.INT, concatenate=True)

        sectors = {}
        for i in range(dim):
            sector_label = tuple(
                (int.bit_count(int(bitstrings[i] & q)) % 2
                 for q in bit_masks)
            )
            sectors.setdefault(sector_label, []).append(i)

        return sectors
    else:
        raise ValueError()


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


def submatrix_eigenvalues_to_target(A: np.ndarray, e_target: float):
    """Start in the upper left corner of A, take a KxK block and calculate its
    lowest eignvalue. Return the smallest K that yields energy below e_target
    or -1 if no such thing can be found, and the vector that does it"""
    e_full, v_full = scipy.sparse.linalg.eigsh(A, which="SA", k=1)
    energies = np.zeros(A.shape[0])
    energies[0] = A[0, 0].real
    # energies[0] = np.nan

    if e_full > e_target:
        return -1, v_full
    elif A[0, 0] < e_target:
        v = np.zeros(A.shape[0])
        v[0] = 1
        return 1, v
    else:
        order = np.argsort(abs(v_full.flatten()))[::-1]
        B = A[np.ix_(order, order)]
        for vec_count in tqdm(range(2, B.shape[0] + 1)):
            # submatrix = A[:vec_count, :][:, :vec_count]
            submatrix = B[:vec_count, :vec_count]
            # e, _ = scipy.sparse.linalg.eigsh(submatrix, which="SA", k=1)
            e, v = np.linalg.eigh(submatrix)
            energies[vec_count - 1] = e[0]
            if e[0] < e_target:
                y = np.zeros(B.shape[0], dtype="complex")
                y[:vec_count] = v[:, 0]
                return vec_count, y

        else:
            plt.plot(energies - e_target)
            plt.yscale("log")
            plt.axhline(e_full - e_target)
            plt.show()
            raise ValueError("this should never happen")


def selected_column_solver(A: np.ndarray, e_target, thr=1e-8, start="zero"):
    if start == "zero":
        starting_index = 0
    elif start == "energy":
        starting_index = np.argmin(np.diag(A))
    else:
        raise ValueError()
    vector_count = -1
    current_vector = np.zeros(A.shape[0])
    current_vector[starting_index] = 1
    current_round = 0
    current_dimension = 1
    if current_vector.T.conj() @ A @ current_vector < e_target:
        return 1, current_vector
    while vector_count == -1:
        current_round += 1
        if current_round > 1000:
            raise ValueError("MaxIter")
        print("SCI-like round ", current_round)
        current_indices = np.where(abs(A @ current_vector) + abs(current_vector) > thr)
        print("dimension ", len(current_indices[0]))
        if len(current_indices[0]) == current_dimension:
            print("stopping as nothing new found within thr")
            break
        current_dimension = len(current_indices[0])
        submatrix = A[np.ix_(current_indices[0], current_indices[0])]
        vector_count, v = submatrix_eigenvalues_to_target(submatrix, e_target)
        current_vector = np.zeros(A.shape[0], dtype="complex")
        current_vector[current_indices] = v.flatten()
        print("SCI-like energy", current_vector.T.conj() @ A @ current_vector)
    return vector_count, current_vector


def orthogonalize_degenerate(w, V, tol=1e-10):
    V_orth = V.copy()

    start = 0
    while start < len(w):
        end = start + 1
        while end < len(w) and abs(w[end] - w[start]) < tol:
            end += 1

        # Orthogonalize this degenerate block
        Q, _ = scipy.linalg.qr(V[:, start:end], mode='economic')
        V_orth[:, start:end] = Q

        start = end
    return V_orth


def find_first_negative(f, N):
    # We create a range object from 1 to N.
    # Note: range(1, N + 1) is lazy and takes O(1) memory.
    domain = range(1, N + 1)

    # We use a key function that returns True (1) when negative
    # and False (0) when positive/zero.
    # Because False < True, this creates a virtual sorted array: [0, 0, ..., 1, 1]
    index = bisect.bisect_left(domain, x=True, key=lambda x: f(x) < 0)

    # bisect_left returns the index in the 'domain' range object.
    # If it returns N, it means it ran off the end and never found a negative.
    if index < len(domain):
        return domain[index]

    return -1


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the metrics")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    parser.add_argument("--states_per_sector", type=int, default=20)
    parser.add_argument("--K_start", default="energy")
    parser.add_argument("--check_if_enough", action="store_true")
    parser.add_argument("--born_huang", action="store_true",
                        help="calculate 'K' by increasing the number of states per sector,"
                             "up to --states_per_sector")
    parser.add_argument("--min_born_huang", type=int, default=1)
    parser.add_argument("--direct_K", action="store_true")
    args = parser.parse_args()

    outname = str(uuid4()) + ".txt"
    with open(outname, "a") as fp:
        fp.write(str(vars(args)) + "\n")

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    parity_matrix = np.loadtxt(args.parity_matrix, dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(parity_matrix,
                                                  moldata.norb,
                                                  moldata.nelec)

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    if args.U is not None:
        x = np.loadtxt(args.U, comments=["#", "{"])
        U = x_to_rotation(x, moldata.norb)
    else:
        U = np.eye(moldata.norb)
        x = np.zeros(comb(moldata.norb, 2))

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(rotated_h,
                                            norb=moldata.norb,
                                            nelec=moldata.nelec)

    e_fci, fcivec = get_fci(dumpdata)
    print("FCI ", e_fci)
    with open(outname, "a") as fp:
        fp.write("E_FCI {0:4.6f}\n".format(e_fci))
    rotated_fcivec = ffsim.apply_orbital_rotation(fcivec, U, norb=moldata.norb,
                                                  nelec=moldata.nelec)

    f = commutator_cost(moldata, symmetries, fcivec)
    print("fci NC cost", f(x))
    with open(outname, "a") as fp:
        fp.write("fci NC cost {0:4.6f}\n".format(f(x)))

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
            w, v = scipy.sparse.linalg.eigsh(
                h_local, which="SA", k=args.states_per_sector)
            v = v[:, np.argsort(w)]
            w = np.sort(w)
            v_orth = orthogonalize_degenerate(w, v)
            sector_gs_pairs[sector_label] = w, v_orth
        else:
            sector_gs_pairs[sector_label] = np.linalg.eigh(h_local)
        if np.min(sector_gs_pairs[sector_label][0]) < smallest:
            smallest = np.min(sector_gs_pairs[sector_label][0])
            lowest_sector_label = sector_label
    print("Lowest sector energy and label")
    print(smallest, lowest_sector_label)
    de_dec = smallest - e_fci
    print("Decoupled error ", smallest - e_fci)
    with open(outname, "a") as fp:
        fp.write("E_decoupled {0:4.6f}\n".format(smallest))
        fp.write("dE {0:4.6f}\n".format(de_dec))
    if de_dec < 0.0016:
        print("K = 1")
        with open(outname, "a") as fp:
            fp.write("K 1")
        quit()

    maxdim = np.max([h.shape[0] for h in sector_hamiltonians.values()])
    print("Largest subspace dimension", maxdim)
    with open(outname, "a") as fp:
        fp.write("maxdim {0:}\n".format(maxdim))
    try:
        zerodim = sector_hamiltonians[tuple([0] * parity_matrix.shape[0])].shape[0]
        print("Zero parity subspace dimension", zerodim)
    except KeyError:
        print([(k, v.shape[0]) for k, v in sector_hamiltonians.items()])

    full_space_vectors = []
    for k, v in sectors.items():
        full_space_vectors_in_sector = np.zeros((rotated_h_linop.shape[0],
                                                 sector_gs_pairs[k][0].shape[0]),
                                                dtype="complex")
        full_space_vectors_in_sector[v, :] = sector_gs_pairs[k][1]
        full_space_vectors.append(full_space_vectors_in_sector)


    full_space_vectors_cat = np.concatenate(full_space_vectors, axis=1)



    if args.direct_K:
        print("Calculating K directly from FCI")
        coefficients = full_space_vectors_cat.T.conj() @ rotated_fcivec
        weights_order = np.argsort(abs(coefficients))[::-1]
        projected_fcivec = full_space_vectors_cat @ coefficients
        projected_fcivec /= np.linalg.norm(projected_fcivec)
        e_full = projected_fcivec.T.conj() @ rotated_h_linop @ projected_fcivec
        print(e_full)
        if e_full > e_fci + CHEMICAL_PRECISION:
            print(e_full)
            with open(outname, "a") as fp:
                fp.write("Not enough states per sector")
            print("Not enough states per sector")
            quit()

        def f(K):
            compressed_coeffs = np.zeros_like(coefficients, dtype="complex")
            compressed_coeffs[weights_order[:K]] = coefficients[weights_order[:K]]
            compressed_coeffs /= np.linalg.norm(compressed_coeffs)
            compressed_fcivec = full_space_vectors_cat @ compressed_coeffs
            e_K = compressed_fcivec.T.conj() @ rotated_h_linop @ compressed_fcivec
            return (e_K - e_fci - CHEMICAL_PRECISION).real

        K_min = find_first_negative(f, full_space_vectors_cat.shape[1])
        if K_min < full_space_vectors_cat.shape[1] and K_min != -1:
            print("K ", K_min)
            with open(outname, "a") as fp:
                fp.write("K {0:}\n".format(K_min))

            all_state_labels = []
            for sector_label, sector_gs in tqdm(sector_gs_pairs.items()):
                labels = [(sector_label, i) for i in range(sector_gs[1].shape[1])]
                all_state_labels.extend(labels)
            print("Sector eigenstates used (sector and excitation level):")
            with open(outname, "a") as fp:
                for i in range(K_min):
                    print(all_state_labels[weights_order[i]])
                    fp.write(str(all_state_labels[weights_order[i]]) + "\n")
            quit()
        else:
            print("not enough?")
            quit()

    if args.born_huang:
        print("Picking L states per sector and finding the energy")
        for L in range(args.min_born_huang, args.states_per_sector + 1):
            print("L = {0:}".format(L), end=" ")
            vectors_stacked = np.concatenate(
                [w[:, :L] for w in full_space_vectors],
                                             axis=1)
            print("dim = {0:}".format(vectors_stacked.shape[1]), end=" ")
            # subspace_op = scipy.sparse.linalg.LinearOperator(
            #     (vectors_stacked.shape[1], vectors_stacked.shape[1]),
            #     matvec=lambda x: vectors_stacked.T.conj() @ rotated_h_linop @ vectors_stacked @ x,
            # )
            subspace_op = vectors_stacked.T.conj() @ rotated_h_linop @ vectors_stacked
            w, v = scipy.sparse.linalg.eigsh(subspace_op, which="SA", k=1)
            print("dE ", w - e_fci)
            if w[0] < e_fci + CHEMICAL_PRECISION:
                full_space_solution = vectors_stacked @ v[:, 0]
                eeeee = full_space_solution.T.conj() @ rotated_h_linop @ full_space_solution
                print(eeeee)
                print(np.linalg.norm(full_space_solution))
                print("{0:} states per sector is enough".format(L))
                abs_coeffs_sorted = np.argsort(np.abs(v[:, 0]**2))[::-1]
                for k in range(1, v.shape[0]):
                    compressed_vector = np.zeros_like(v[:, 0], dtype="complex")
                    compressed_vector[abs_coeffs_sorted[:k]] = v[abs_coeffs_sorted[:k], 0]
                    compressed_vector /= np.linalg.norm(compressed_vector)
                    e_compressed = compressed_vector.T.conj() @ subspace_op @ compressed_vector
                    print(e_compressed - e_fci)
                    if e_compressed - e_fci < 0.0016:
                        print("K = {0:}".format(k))
                        quit()
                quit()
        else:
            print("Chemical accuracy not reached ")
            quit()

    h_subspace = full_space_vectors_cat.T.conj() @ rotated_h_linop @ full_space_vectors_cat

    if args.check_if_enough:
        w_subspace, _ = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")
        print("Coupled energy", w_subspace)
        if w_subspace - e_fci > 0.0016:
            print("Not enough states to reach chemical accuracy, increase states_per_sector")
            quit()

    if args.K_start == "zero":
        print("Using the first sector for the start")
        print(list(sectors.keys())[0])
        K, v = selected_column_solver(h_subspace, e_fci + 0.0016, start="zero")
    elif args.K_start == "energy":
        K, v = selected_column_solver(h_subspace, e_fci + 0.0016, start="energy")
    else:
        raise ValueError()
    print("K ", K)
    # print("variance ", v.T.conj() @ h_subspace @ h_subspace @ v - (v.T.conj() @ h_subspace @ v) ** 2)




