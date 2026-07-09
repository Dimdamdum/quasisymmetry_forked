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
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


from chemistry import load_moldata, fcidump_data, CHEMICAL_PRECISION
from optimize_symmetries import parity_matrix_to_quasisymmetries, x_to_rotation, get_fci, commutator_cost
from src.energy_diagnostics import (
    coupled_energy_perturbation,
    reference_coupled_energy_k,
    sector_data_from_gs_pairs,
    state_labels_for_columns,
)
from src.sector_utils import symmetry_sectors, subspace_matrix


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
    """scipy.sparse.linalg.eigsh sometimes returns non-orthogonal eigenvectors if they have
    degenerate eigenvalues. This function rectifies that."""
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

def solve_eigs(data):
    # mpi4py can't pickle the rotated_h_linop, so we will be constructing it on each worker?
    moldata = data["moldata"]
    rotated_h = data["rotated_h"]
    sector_bitstrings = data["sector_bitstrings"]
    rotated_h_linop = ffsim.linear_operator(rotated_h,
                                            norb=moldata.norb,
                                            nelec=moldata.nelec)

    h_subspace = subspace_matrix(rotated_h_linop, sector_bitstrings)
    if data["states_per_sector"] <= h_subspace.shape[0] - 2:
        w, v = scipy.sparse.linalg.eigsh(
            h_subspace, which="SA", k=data["states_per_sector"])
        v = v[:, np.argsort(w)]
        w = np.sort(w)
        v_orth = orthogonalize_degenerate(w, v)
        sector_eigs = w, v_orth
    else:
        sector_eigs = np.linalg.eigh(h_subspace)

    return {"sector_label": data["sector_label"],
            "sector_eigs": sector_eigs,
            "rank": MPI.COMM_WORLD.Get_rank(),
            "hostname": MPI.Get_processor_name()}


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the metrics")
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    parser.add_argument("--states_per_sector", type=int, default=500)
    parser.add_argument("--check_if_enough", action="store_true")
    parser.add_argument(
        "--coupled_energy_method",
        choices=("reference", "perturbation"),
        default="reference",
        help="K_coupled selection: FCI-coefficient greedy (reference) or PT-screened greedy (perturbation)",
    )
    args = parser.parse_args()

    p = Path(args.molpath)

    outname = "result_" + p.parts[-1] + "_" + str(uuid4())[:6] + ".txt"
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # this process's ID, 0..size-1
    size = comm.Get_size()  # total number of processes
    print("rank and size", rank, size)

    tasks = [{"moldata": moldata,
              "rotated_h": rotated_h,
              "states_per_sector": args.states_per_sector,
              "sector_label": k,
              "sector_bitstrings": v
              }
             for k, v in sectors.items()]

    sector_eigs = {}

    # maybe restore the old way and run it if size == 1?
    with MPIPoolExecutor() as executor:
        for r in executor.map(solve_eigs, tasks):
            label = tuple(r["sector_label"])
            sector_eigs[label]= r["sector_eigs"]


    sector_gs_energies = []
    for w, v in sector_eigs.items():
        sector_gs_energies.append(np.min(v[0]))

    smallest = np.min(sector_gs_energies)

    # sector_dims = [len(sector_bistrings) for sector_bistrings in sectors.values()]
    # maxdim = max(sector_dims)
    # print("Largest subspace dimension", maxdim)
    # with open(outname, "a") as fp:
    #     fp.write("maxdim {0:}\n".format(maxdim))

    de_dec = smallest - e_fci
    print("Decoupled error ", smallest - e_fci)
    with open(outname, "a") as fp:
        fp.write("E_decoupled {0:4.6f}\n".format(smallest))
        fp.write("dE {0:4.6f}\n".format(de_dec))
    # if de_dec < 0.0016:
    #     print("K = 1")
    #     with open(outname, "a") as fp:
    #         fp.write("K 1")
    #     quit()

    h_apply = lambda v: rotated_h_linop @ v

    if args.coupled_energy_method == "perturbation":
        print("Calculating K via PT-screened coupled-energy greedy selection")
        sector_data = sector_data_from_gs_pairs(
            sectors, sector_eigs, rotated_h_linop.shape[0]
        )
        e_coupled, k_coupled, converged, chosen_keys = coupled_energy_perturbation(
            h_apply,
            sector_data,
            e_exact=e_fci,
            tol=CHEMICAL_PRECISION,
        )
        print("E_coupled", e_coupled)
        print("K", k_coupled)
        print("converged", converged)
        with open(outname, "a") as fp:
            fp.write("E_coupled {0:4.6f}\n".format(e_coupled))
            fp.write("K {0:}\n".format(k_coupled))
            fp.write("converged {0:}\n".format(converged))
        # if converged:
            # print("Sector eigenstates used (sector and excitation level):")
            # with open(outname, "a") as fp:
            #     for key in chosen_keys:
            #         print(key)
            #         fp.write(str(key) + "\n")
        if not converged:
            print("PT coupled-energy did not converge within chemical precision")

    elif args.coupled_energy_method == "reference":

        print("Calculating K directly from FCI (reference wavefunction)")

        full_space_vectors = []
        for k, v in sectors.items():
            full_space_vectors_in_sector = np.zeros((rotated_h_linop.shape[0],
                                                     sector_eigs[k][0].shape[0]),
                                                    dtype="complex")
            full_space_vectors_in_sector[v, :] = sector_eigs[k][1]
            full_space_vectors.append(full_space_vectors_in_sector)
        full_space_vectors_cat = np.concatenate(full_space_vectors, axis=1)

        k_min, e_coupled, converged, weights_order = reference_coupled_energy_k(
            h_apply,
            full_space_vectors_cat,
            rotated_fcivec,
            e_fci,
            chemical_precision=CHEMICAL_PRECISION,
        )
        print("E_coupled (full projection)", e_coupled)
        if k_min is None:
            with open(outname, "a") as fp:
                fp.write("coupled_energy_method reference\n")
                fp.write("Not enough states per sector\n")
            print("Not enough states per sector")
            quit()

        print("K ", k_min)
        with open(outname, "a") as fp:
            fp.write("K {0:}\n".format(k_min))

        all_state_labels = state_labels_for_columns(sector_eigs)

        # with open(outname, "a") as fp:
        #     for i in range(k_min):
        #         print(all_state_labels[weights_order[i]])
        #         fp.write(str(all_state_labels[weights_order[i]]) + "\n")

        chosen_keys = [all_state_labels[weights_order[i]] for i in range(k_min)]

    print("Sector eigenstates used (sector and excitation level):")
    with open(outname, "a") as fp:
        fp.write("Sector eigenstates used (sector and excitation level):\n")
        for key in chosen_keys:
            print(key)
            fp.write(str(key) + "\n")

        unique_sectors_used = set([w[0] for w in chosen_keys])
        total_dim_of_relevant_sectors = 0
        print("Relevant sectors and their dimensions:")
        fp.write("Relevant sectors and their dimensions:\n")
        for s in unique_sectors_used:
            print(s, len(sectors[s]))
            fp.write(str(s) + " " + str(len(sectors[s])) + "\n")
            total_dim_of_relevant_sectors += len(sectors[s])
        print("{0:} sectors in total".format(len(unique_sectors_used)))
        fp.write("{0:} sectors in total\n".format(len(unique_sectors_used)))
        print("Total dimension: {0:}".format(total_dim_of_relevant_sectors))
        fp.write("Total dimension: {0:}\n".format(total_dim_of_relevant_sectors))



