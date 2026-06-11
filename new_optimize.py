import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import pyscf.fci

from typing import Callable
from math import comb
from functools import cache, reduce

from chemistry import load_moldata, fcidump_data


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
        raise NotImplementedError()
    else:
        raise ValueError("shape[1] must be norb or 2 * norb")


def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)


def get_fci(dumpdata):
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
    return e_fci, np.array(fcivec.reshape((-1,)), dtype="complex")


def callback(intermediate_result):
    print(time.strftime("%a, %d %b %Y %H:%M:%S",
                        time.localtime()), end=" ")
    print("{0:4.6f}".format(intermediate_result.fun))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("parity_matrix",
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--x0",
                        help="path to the initial guess for the orbital rotation (either U or x)",
                        default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    args = parser.parse_args()

    moldata = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    parity_matrix = np.loadtxt(args.parity_matrix, dtype=int)
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
    np.savetxt("x_opt.txt", res.x)