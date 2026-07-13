import argparse
import bisect
import json
from itertools import product
from math import comb
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg
from tqdm import tqdm
<<<<<<< HEAD

try:
    import ffsim
    import pyscf
    from chemistry import CHEMICAL_PRECISION, fcidump_data, load_moldata
    from optimize_symmetries import (
        commutator_cost,
        get_fci,
        parity_matrix_to_quasisymmetries,
        x_to_rotation,
    )
    from src.energy_diagnostics import (
        coupled_energy_perturbation,
        reference_coupled_energy_k,
        sector_data_from_gs_pairs,
        state_labels_for_columns,
    )
except ImportError as _FCI_IMPORT_ERROR:  # noqa: N816 — kept for CLI messaging
    ffsim = None
    pyscf = None
    fcidump_data = None
    load_moldata = None
    CHEMICAL_PRECISION = 0.0016
else:
    _FCI_IMPORT_ERROR = None


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
=======
from pathlib import Path
from itertools import product
from math import comb
import matplotlib.pyplot as plt
import bisect
from uuid import uuid4
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import time


from chemistry import load_moldata, fcidump_data, CHEMICAL_PRECISION
from optimize_symmetries import parity_matrix_to_quasisymmetries, x_to_rotation, get_fci, commutator_cost
from src.energy_diagnostics import (
    coupled_energy_perturbation,
    reference_coupled_energy_k,
    sector_data_from_gs_pairs,
    state_labels_for_columns,
)
from src.sector_utils import symmetry_sectors, subspace_matrix
>>>>>>> f09fee37df2e44a547fbf022c745bee435e5c8cf


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
<<<<<<< HEAD
    parser.add_argument("molpath",
        help="path to the Hamiltonian (PySCF .chk or .FCIDUMP)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--U", help="x as orbital rotation",
                        default=None)
    parser.add_argument("--solver", choices=("fci", "dmrg"), default="fci",
                        help="reference / diagnostics backend (default: fci)")
    parser.add_argument("--bond_dim", type=int, default=250,
                        help="DMRG bond dimension (only with --solver dmrg)")
    parser.add_argument("--wavefunction_dir", default=None,
                        help="local DMRG wavefunction store to reuse/create "
                             "(only with --solver dmrg)")
    parser.add_argument("--n_threads", type=int, default=4,
                        help="block2 threads (dmrg solver only)")
    parser.add_argument("--penalty", type=float, default=30.0,
                        help="sector penalty for DMRG E_decoupled / K")
    parser.add_argument("--max_sectors", type=int, default=16,
                        help="max sectors to scan in DMRG diagnostics")
    parser.add_argument("--reorder", choices=("fiedler", "gaopt"), default=None,
                        help="optional orbital reordering before DMRG")
    parser.add_argument("--entanglement", action="store_true",
                        help="with --solver dmrg, also report orbital entropies")
=======
    parser.add_argument("input_data", help="JSON you got from optimize_symmetries.py")
>>>>>>> f09fee37df2e44a547fbf022c745bee435e5c8cf
    parser.add_argument("--states_per_sector", type=int, default=500)
    parser.add_argument("--check_if_enough", action="store_true")
    parser.add_argument(
        "--coupled_energy_method",
        choices=("reference", "perturbation"),
        default="reference",
        help="K_coupled selection: FCI-coefficient greedy (reference) or PT-screened greedy (perturbation)",
    )
    args = parser.parse_args()

    with open(args.input_data, "r") as fp:
        input_data = json.load(fp)

    p = Path(input_data["molpath"])

<<<<<<< HEAD
    parity_matrix = np.loadtxt(args.parity_matrix, dtype=int)

    if args.solver == "dmrg":
        from src.dmrg_diagnostics import format_metrics_report, run_dmrg_metrics
        from src.dmrg_solver import (
            Block2DMRGSolver,
            DMRGConfig,
            rotate_integrals,
            rotation_from_parameters,
        )

        store_dir = args.wavefunction_dir
        if store_dir is None:
            store_dir = str(Path("wavefunctions") / (p.stem + "_metrics"))

        if p.suffix == ".chk":
            if fcidump_data is None:
                raise SystemExit(
                    ".chk input requires pyscf (chemistry.fcidump_data). "
                    "Pass an FCIDUMP instead, or install pyscf."
                )
            dumpdata = fcidump_data(args.molpath)
            base = Block2DMRGSolver.from_dumpdata(
                dumpdata, store_dir=None, n_threads=args.n_threads,
                save_integrals=False,
            )
        else:
            base = Block2DMRGSolver.from_fcidump(
                args.molpath, store_dir=None, n_threads=args.n_threads,
                save_integrals=False,
            )
        h1e, g2e, ecore = base.h1e, base.g2e, base.ecore
        n_elec, spin = base.n_elec, base.spin
        if args.U is not None:
            x = np.loadtxt(args.U, comments=["#", "{"])
            h1e, g2e = rotate_integrals(
                h1e, g2e, rotation_from_parameters(x, h1e.shape[0])
            )

        solver = Block2DMRGSolver(
            h1e=h1e, g2e=g2e, ecore=ecore, n_elec=n_elec, spin=spin,
            store_dir=store_dir, n_threads=args.n_threads,
            reorder=args.reorder,
        )
        # For DMRG, the PT path is the only scalable K method.
        compute_k = True
        states_per_sector = (
            args.states_per_sector if args.states_per_sector < 50
            else 5
        )
        report = run_dmrg_metrics(
            solver,
            parity_matrix,
            config=DMRGConfig(max_bond_dim=args.bond_dim),
            penalty=args.penalty,
            max_sectors=args.max_sectors,
            states_per_sector=states_per_sector,
            compute_k=compute_k,
            compute_entanglement=args.entanglement,
        )
        lines = format_metrics_report(report)
        for line in lines:
            print(line)
        with open(outname, "a") as fp:
            fp.write("\n".join(lines) + "\n")
        print("results written to", outname)
        raise SystemExit(0)

    if _FCI_IMPORT_ERROR is not None:
        raise SystemExit(
            f"--solver fci requires pyscf/ffsim ({_FCI_IMPORT_ERROR}). "
            "Use --solver dmrg on FCIDUMP files without those packages."
        )

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

=======
    outname = ("metrics_" +
               p.parts[-1] + "_" + str(uuid4())[:8] + ".json")
    out_data = {}
    out_data["args"] = vars(args)
    out_data["OO_data"] = input_data

    moldata = load_moldata(input_data["molpath"])
    dumpdata = fcidump_data(input_data["molpath"])

    parity_matrix = np.loadtxt(input_data["parity"], dtype=int)
>>>>>>> f09fee37df2e44a547fbf022c745bee435e5c8cf
    symmetries = parity_matrix_to_quasisymmetries(parity_matrix,
                                                  moldata.norb,
                                                  moldata.nelec)

    print(parity_matrix)

    sectors = symmetry_sectors(parity_matrix, moldata.norb, moldata.nelec)

    x = np.array(input_data["rotation"])
    U = x_to_rotation(x, moldata.norb)

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(rotated_h,
                                            norb=moldata.norb,
                                            nelec=moldata.nelec)

    e_fci, fcivec = get_fci(dumpdata)
<<<<<<< HEAD
    print("reference (fci) ", e_fci)
    with open(outname, "a") as fp:
        fp.write("solver fci\n")
        fp.write("E_FCI {0:4.6f}\n".format(e_fci))
=======
    print("FCI ", e_fci)
    out_data["E_FCI"] = e_fci
>>>>>>> f09fee37df2e44a547fbf022c745bee435e5c8cf
    rotated_fcivec = ffsim.apply_orbital_rotation(fcivec, U, norb=moldata.norb,
                                                  nelec=moldata.nelec)

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

    de_dec = smallest - e_fci
    print("Decoupled error ", smallest - e_fci)
    out_data["E_decoupled"] = smallest
    out_data["dE"] = de_dec

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
        out_data["E_coupled"] = e_coupled
        out_data["K"] = k_coupled
        out_data["converged"] = converged
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
        out_data["K"] = k_min
        if k_min is None:
            print("Not enough states per sector")
            quit()

        print("K ", k_min)

        all_state_labels = state_labels_for_columns(sector_eigs)

        chosen_keys = [all_state_labels[weights_order[i]] for i in range(k_min)]

    print("Sector eigenstates used (sector and excitation level):")
    for key in chosen_keys:
        print(key)
    out_data["sector_eigenstates"] = chosen_keys

    unique_sectors_used = list(set([w[0] for w in chosen_keys]))
    total_dim_of_relevant_sectors = 0
    print("Relevant sectors and their dimensions:")
    for s in unique_sectors_used:
        print(s, len(sectors[s]))
        total_dim_of_relevant_sectors += len(sectors[s])
    print("{0:} sectors in total".format(len(unique_sectors_used)))
    print("Total dimension: {0:}".format(total_dim_of_relevant_sectors))

    out_data["relevant_sectors"] = unique_sectors_used
    out_data["relevant_sectors_count"] = len(unique_sectors_used)
    out_data["relevant_sectors_total_dim"] = total_dim_of_relevant_sectors
    with open(outname, "a") as fp:
        json.dump(out_data, fp, indent=2)



