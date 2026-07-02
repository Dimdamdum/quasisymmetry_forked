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


def commutator_cost(moldata: ffsim.MolecularData,
                    symmetries: list,
                    reference_state: np.ndarray) -> Callable:
    def f(x):
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                  norb=moldata.norb, nelec=moldata.nelec)
        total_nc = 0
        for s in symmetries:
            commutator = h @ s - s @ h
            total_nc += np.linalg.norm(commutator @ rotated_state)**2
        return total_nc
    return f


@cache
def parities(norb, nelec):
    local_parities = []
    for i in range(norb):
        s_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): -2,
                (): 1
            }
        )
        s_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): -2,
                (): 1
            }
        )
        s = s_alpha * s_beta
        local_parities.append(ffsim.linear_operator(s, norb, nelec))
    return local_parities


def parity_matrix_to_quasisymmetries(parity_matrix: np.ndarray,
                                     norb,
                                     nelec):
    local_parities = parities(norb, nelec)
    if parity_matrix.shape[1] == norb:
        operators = []
        for i in range(parity_matrix.shape[0]):
            relevant_parities = [local_parities[j] for j in range(norb)
                                 if parity_matrix[i][j] == 1]
            quasisymmetry = reduce(lambda a, b: a @ b,
                relevant_parities
            )
            operators.append(quasisymmetry)
        return operators
    elif parity_matrix.shape[1] == norb * 2:
        operators = []
        for i in range(parity_matrix.shape[0]):
            current_ops = []
            for j in range(norb):
                if parity_matrix[i][2 * j] == parity_matrix[i][2 * j + 1] == 1:
                    current_ops.append(local_parities[j])
                elif parity_matrix[i][2 * j] == 1:
                    s_alpha = ffsim.FermionOperator(
                        {
                            (ffsim.cre_a(j), ffsim.des_a(j)): -2,
                            (): 1
                        }
                    )
                    current_ops.append(ffsim.linear_operator(s_alpha, norb, nelec))
                elif parity_matrix[i][2 * j + 1] == 1:
                    s_beta = ffsim.FermionOperator(
                        {
                            (ffsim.cre_b(j), ffsim.des_b(j)): -2,
                            (): 1
                        }
                    )
                    current_ops.append(ffsim.linear_operator(s_beta, norb, nelec))

            quasisymmetry = reduce(lambda a, b: a @ b,
                                   current_ops
                                   )
            operators.append(quasisymmetry)
        return operators
    else:
        raise ValueError("shape[1] must be norb or 2 * norb")


def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)


def get_fci(dumpdata, flatten=True):
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 500
    cisolver.conv_tol = 1e-10
    e_fci, fcivec = cisolver.kernel(
        dumpdata["H1"],
        dumpdata["H2"],
        dumpdata["NORB"],
        dumpdata["NELEC"],
        ecore=dumpdata["ECORE"],
    )
    if not cisolver.converged:
        raise RuntimeError("FCI didn't converge!")
    if flatten:
        return e_fci, np.array(fcivec.reshape((-1,)), dtype="complex")
    else:
        return e_fci, fcivec


def expand_state(mol:of.MolecularData, ci, threshold=1e-12):
    """Given a pyscf/ffsim representation of a CI state, expand it into a (2**n_qubits, 1) vector"""
    norb = mol.n_orbitals
    n_qubits = 2 * norb
    dim = 1 << n_qubits

    nelec = int(mol.n_electrons)
    spin = int(mol.multiplicity - 1)   # = n_alpha - n_beta
    n_alpha = (nelec + spin) // 2
    n_beta = nelec - n_alpha

    alpha_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_alpha), dtype=np.int64)
    beta_strings = np.asarray(pyscf.fci.cistring.make_strings(range(norb), n_beta), dtype=np.int64)

    expected_shape = (len(alpha_strings), len(beta_strings))
    if ci.shape != expected_shape:
        raise ValueError(
            f"Unexpected CI tensor shape {ci.shape}; expected {expected_shape}."
        )


    def occupied_orbitals(det, norb):
        return [p for p in range(norb) if (det >> p) & 1]

    def pyscf_det_to_openfermion_index_and_phase(alpha_det, beta_det, norb):
        """
        PySCF CI basis is organized by separate alpha/beta strings.
        We embed into OpenFermion spin-orbital order:
            [a0, b0, a1, b1, ..., a_{norb-1}, b_{norb-1}]
        and build the basis index using qubit 0 as the leftmost tensor factor.

        Returns
        -------
        basis_index : int
        phase : +1 or -1
        """
        occ_alpha = occupied_orbitals(alpha_det, norb)
        occ_beta = occupied_orbitals(beta_det, norb)

        # Fermionic sign from reordering:
        # starting from all-alpha then all-beta ordering
        # into interleaved spin-orbital ordering.
        inversions = 0
        for p in occ_alpha:
            for q in occ_beta:
                if q < p:
                    inversions += 1
        phase = -1.0 if (inversions % 2) else 1.0

        # Build OpenFermion computational-basis index.
        # qubit 0 is the leftmost tensor factor, so it contributes to the
        # most-significant bit of the basis index.
        idx = 0
        for p in occ_alpha:
            q = 2 * p
            idx |= (1 << (n_qubits - 1 - q))
        for p in occ_beta:
            q = 2 * p + 1
            idx |= (1 << (n_qubits - 1 - q))

        return idx, phase

    rows = []
    data = []

    for ia, alpha_det in enumerate(alpha_strings):
        for ib, beta_det in enumerate(beta_strings):
            coeff = ci[ia, ib]
            if abs(coeff) > threshold:
                idx, phase = pyscf_det_to_openfermion_index_and_phase(
                    int(alpha_det), int(beta_det), norb
                )
                rows.append(idx)
                data.append(phase * coeff)

    if rows:
        cols = np.zeros(len(rows), dtype=np.int64)
        state = scipy.sparse.csr_matrix(
            (np.asarray(data, dtype=np.complex128), (np.asarray(rows), cols)),
            shape=(dim, 1),
            dtype=np.complex128,
        )
    else:
        state = scipy.sparse.csr_matrix((dim, 1), dtype=np.complex128)
    return state.todense()


def callback(intermediate_result):
    print(time.strftime("%a, %d %b %Y %H:%M:%S",
                        time.localtime()), end=" ")
    print("{0:4.6f}".format(intermediate_result.fun))


def comm_sq_exp_fast(sym_ops, H, state, n_qubits, verbose=False):
    """
    Compute sum_k <state| ( i[H, S_k] )^2 |state> efficiently.

    Parameters
    ----------
    sym_ops : list[QubitOperator]
        Symmetry operators (Pauli products).
    H : sparse operator
        Hamiltonian.
    state : np.ndarray
        State vector.
    n_qubits : int

    Returns
    -------
    float or complex
    """

    psi = np.asarray(state)

    # Reused for every symmetry operator
    Hpsi = H @ psi

    total = 0.0 + 0.0j
    for sym in sym_ops:
        S = of.get_sparse_operator(sym, n_qubits).tocsr()

        Spsi = S @ psi
        delta = 1j * ((H @ Spsi) - (S @ Hpsi))  # delta = i[H,S]|psi>

        # <psi| (i[H,S])^2 |psi> = || delta ||^2
        total += np.vdot(delta, delta)

    nc_exp = np.real_if_close(total)
    if verbose: print("Exp(non-commutator^2): ", nc_exp)
    return nc_exp



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # mandatory arguments
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("parity",
                        help="path to the incidence matrix of symmetries")

    # optional arguments
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--cost_function", default="NC")
    parser.add_argument("--x0",
                        help="path to the initial guess for the orbital rotation (either U or x)",
                        default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None,
                        help="Name of the output file. If none specified, a time stamp will be used.")

    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    parity_matrix = np.loadtxt(args.parity, dtype=int)
    symmetries = parity_matrix_to_quasisymmetries(parity_matrix,
                                                  moldata.norb,
                                                  moldata.nelec)
    if args.reference == "fci":
        _, state = get_fci(dumpdata)
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("reference must be fci or hf")

    f = commutator_cost(moldata, symmetries, state)

    if args.x0 is None:
        x0 = np.zeros(comb(moldata.norb, 2))
    else:
        x0 = np.loadtxt(args.x0)

    print("before optimization: {0:4.6f}".format(f(x0)))
    res = scipy.optimize.minimize(f, x0, method="L-BFGS-B",
                                  options={"maxiter": 100},
                                  callback=callback if args.verbose else None)
    print(res.message)
    print("optimized: {0:4.6f}".format(res.fun))
    if args.outname is not None:
        outname = args.outname
    else:
        outname = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".txt"

    with open(outname,
              "a", newline="") as fp:
        fp.write(str(vars(args)) + "\n")
        np.savetxt(fp, res.x)