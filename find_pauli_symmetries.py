import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import pyscf.fci
import openfermion as of
import openfermionpyscf

from typing import Callable
from math import comb
from functools import cache, reduce
from itertools import combinations

from chemistry import load_moldata, fcidump_data

from external_imports import get_cisd_gs, get_hf_occ, get_hf_wfn
from external_imports import beam_search_symmetries, BeamSearch_Symmetries
from external_imports import mask_to_qubit_operator
from external_imports import variance, molecular_data_from_fcidump


from optimize_symmetries import get_fci, expand_state, comm_sq_exp_fast
from src.clifford_sectors import save_symmetry_manifest


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--cost_function", default="NC")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None,
                        help="Name of the output file. If none specified, a time stamp will be used.")
    parser.add_argument("--senquart", action="store_true")
    parser.add_argument("--parity_output", default="parity_matrix.txt",
                        help="output path for the legacy parity matrix")
    parser.add_argument("--symmetry_manifest", default="symmetry_manifest.json",
                        help="output path for ordered signed Pauli symmetries")

    args = parser.parse_args()

    mol = molecular_data_from_fcidump(args.molpath)


    H = of.get_fermion_operator(mol.get_molecular_hamiltonian())
    n_qubits = of.count_qubits(H)
    qubit_hamiltonian = of.jordan_wigner(H)
    sparse_qubit_op = of.get_sparse_operator(qubit_hamiltonian, n_qubits)

    dumpdata = fcidump_data(args.molpath)
    if args.reference == "fci":
        # e, gs, gs_info = get_fci_state_openfermion(mol)
        e, state = get_fci(dumpdata, flatten=False)
        ref_state = expand_state(mol, state)
    elif args.reference == "hf":
        hf_occ = get_hf_occ(mol.n_electrons, mol.n_orbitals, as_str=True)
        ref_state = get_hf_wfn([int(s) for s in hf_occ])
    elif args.reference == "cisd":
        hf_occ = get_hf_occ(mol.n_electrons, mol.n_orbitals, as_str=True)
        e, ref_state = get_cisd_gs(hf_occ, qubit_hamiltonian, n_qubits, 'wfs', tf='jw')
    else:
        raise ValueError('reference can be fci, cisd, hf')

    if args.cost_function == "NC":
        cost = lambda s_list: comm_sq_exp_fast(s_list, sparse_qubit_op,
                                                          ref_state, n_qubits)
    elif args.cost_function == "variance":
        cost = lambda s_list: variance(s_list, ref_state, n_qubits)
    else:
        raise NotImplementedError()

    beam_score = lambda s: (-1) * cost(s)

    n_sym = n_qubits // 2

    if args.senquart:
        seniorities = [(0, 2**(2 * i) + 2**(2 * i + 1)) for i in range(n_qubits // 2)]
        quartets = [(0, s[0][1] + s[1][1]) for s in combinations(seniorities, 2)]

        symmetry_costs = []
        for s in seniorities + quartets:
            symmetry_costs.append(cost([mask_to_qubit_operator(s, n_qubits)]))

        print("Symmetries (not) sorted by their cost value")
        # for i in np.argsort(symmetry_costs):
        for i in range(len(seniorities + quartets)):
            symm_z_mask = (seniorities + quartets)[i][1]
            print(i, format(symm_z_mask, "0" + str(n_qubits) + "b")[::-1],
                  symmetry_costs[i])


        beam_symmetries = beam_search_symmetries(
            qubit_hamiltonian,
            seniorities + quartets,
            target_rank=n_sym,
            n_qubits=n_qubits,
            beam_width=16,
            heavy_core_fraction=0.95,
            initial_generators=None,
            score_func=beam_score
        )

    else:

        beam_symmetries = BeamSearch_Symmetries(qubit_hamiltonian,
                                                     target_rank=n_sym,
                                                     beam_width=16,
                                                     heavy_core_fraction=0.95,
                                                     include_pairwise_products=True,
                                                     pairwise_seed_terms=12,
                                                     seed_with_exact_symmetries=True,
                                                     score_func=beam_score
                                                     )

    parity_matrix = np.zeros((len(beam_symmetries), n_qubits), dtype=int)

    print("Kept symmetries:")
    for i, s in enumerate(beam_symmetries):
        pauli_keys = list(s.terms.keys())
        assert len(pauli_keys) == 1
        key = pauli_keys[0]
        string_letters = "".join([w[1] for w in key])
        pauli_positions = [w[0] for w in key]
        if string_letters.find("X") == -1 and string_letters.find("Y") == -1:
            parity_matrix[i, pauli_positions] = 1
        print(s)
    print("Parity matrix from the Z symmetries")
    print(parity_matrix)
    np.savetxt(args.parity_output, parity_matrix, fmt='%d')
    save_symmetry_manifest(
        args.symmetry_manifest,
        beam_symmetries,
        parity_matrix,
        metadata={
            "molpath": args.molpath,
            "reference": args.reference,
            "cost_function": args.cost_function,
            "candidate_family": "seniority_plus_quartet" if args.senquart else "beam_pauli",
            "selected_set_cost": float(np.real(cost(beam_symmetries))),
            "individual_costs": [float(np.real(cost([symmetry]))) for symmetry in beam_symmetries],
        },
    )
    print("Saved parity matrix to", args.parity_output)
    print("Saved symmetry manifest to", args.symmetry_manifest)
