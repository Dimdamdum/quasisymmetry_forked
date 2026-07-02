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

from chemistry import load_moldata, fcidump_data

from src.state_utils import get_cisd_gs, get_fci_state_openfermion
from src.bs import beam
import fcidump_openfermion

from optimize_symmetries import get_fci, expand_state, comm_sq_exp_fast


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

    args = parser.parse_args()

    mol = fcidump_openfermion.molecular_data_from_fcidump(args.molpath)


    H = of.get_fermion_operator(mol.get_molecular_hamiltonian())
    n_qubits = of.count_qubits(H)
    qubit_hamiltonian = of.jordan_wigner(H)
    sparse_qubit_op = of.get_sparse_operator(qubit_hamiltonian, n_qubits)

    dumpdata = fcidump_data(args.molpath)
    if args.reference == "fci":
        # e, gs, gs_info = get_fci_state_openfermion(mol)
        e, state = get_fci(dumpdata, flatten=False)
        ref_state = expand_state(mol, state)
    else:
        raise NotImplementedError()

    if args.cost_function == "NC":
        cost = lambda s_list: comm_sq_exp_fast(s_list, sparse_qubit_op,
                                                          ref_state, n_qubits)
    else:
        raise NotImplementedError()

    beam_score = lambda s: (-1) * cost(s)

    n_sym = n_qubits // 2
    beam_symmetries = beam.BeamSearch_Symmetries(qubit_hamiltonian,
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
    np.savetxt("parity_matrix.txt", parity_matrix, fmt='%d')