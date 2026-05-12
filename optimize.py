"""Find approximate quasisymmetries for a given Hamiltonian and save their parameters."""

import argparse
from itertools import combinations
import numpy as np
import time
import ffsim
import scipy
import pyscf
from typing import Tuple, Callable

SENIORITY_ANGLES = (np.arccos(-2.0 / np.sqrt(6.0)), np.pi / 4.0)


def callback(intermediate_result):
    print(intermediate_result.fun)
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


def commutator_cost(moldata: ffsim.MolecularData, reference="fci") -> Callable:
    h_linop = ffsim.linear_operator(moldata.hamiltonian,
                                    norb=moldata.norb,
                                    nelec=moldata.nelec)

    if reference == "fci":
        _, ref_state = scipy.sparse.linalg.eigsh(h_linop, which="SA", k=1)
    elif reference == "hf":
        ref_state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("reference can be 'fci' or 'hf'")

    iu = np.triu_indices(moldata.norb, k=1)

    def f(x):
        linops = make_quasiymmetries(x, moldata.norb, moldata.nelec)

        rotation_generator = np.zeros((moldata.norb, moldata.norb))
        rotation_generator[iu] = x[:-2]
        rotation_generator -= rotation_generator.T
        U = scipy.linalg.expm(rotation_generator)

        u_psi = ffsim.apply_orbital_rotation(ref_state, U, moldata.norb, moldata.nelec)
        s_u_psis = [linop @ u_psi for linop in linops]
        udag_s_u_psis = [ffsim.apply_orbital_rotation(psi, U.T.conj(),
                                                     moldata.norb, moldata.nelec)
                            for psi in s_u_psis]
        h_psi = h_linop @ ref_state

        u_h_psi = ffsim.apply_orbital_rotation(h_psi, U,
                                               moldata.norb, moldata.nelec)
        symmetry_u_h_psis = [linop @ u_h_psi for linop in linops]
        udag_s_u_h_psis = [ffsim.apply_orbital_rotation(psi, U.T.conj(),
                                                     moldata.norb, moldata.nelec)
                            for psi in symmetry_u_h_psis]
        final_states = [h_linop @ udag_s_u_psis[i] -  udag_s_u_h_psis[i]
                        for i in range(len(udag_s_u_psis))]
        return np.sum([np.linalg.norm(psi) ** 2 for psi in final_states])

    # xdim = iu[0].shape[0] + 2

    return f


def variance_cost(moldata: ffsim.MolecularData, reference="fci") -> Callable:
    if reference == "fci":
        h_linop = ffsim.linear_operator(moldata.hamiltonian,
                                        norb=moldata.norb,
                                        nelec=moldata.nelec)
        _, state = scipy.sparse.linalg.eigsh(h_linop, which="SA", k=1)
    elif reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("reference can be 'fci' or 'hf'")

    iu = np.triu_indices(moldata.norb, k=1)

    def f(x):
        linops = make_quasiymmetries(x, moldata.norb, moldata.nelec)
        rotation_generator = np.zeros((moldata.norb, moldata.norb))
        rotation_generator[iu] = x[:-2]
        rotation_generator -= rotation_generator.T
        U = scipy.linalg.expm(rotation_generator)
        u_psi = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)
        s_u_psi = [linop @ u_psi for linop in linops]
        s_s_u_psi = [linop @ linop @ u_psi for linop in linops]
        udag_s_u_psi = [ffsim.apply_orbital_rotation(psi, U.T.conj(),
                                                     moldata.norb, moldata.nelec)
                            for psi in s_u_psi]
        udag_s_s_u_psi = [ffsim.apply_orbital_rotation(psi, U.T.conj(),
                                                     moldata.norb, moldata.nelec)
                        for psi in s_s_u_psi]
        expected_s = [state.T.conj() @ psi for psi in udag_s_u_psi]
        expected_s_s = [state.T.conj() @ psi for psi in udag_s_s_u_psi]
        return np.sum(expected_s_s) - np.sum(np.array(expected_s) ** 2)

    return f


def make_quasiymmetries(x, norb, nelec):
    a = np.sin(x[-2]) * np.cos(x[-1])
    b = np.sin(x[-2]) * np.sin(x[-1])
    c = np.cos(x[-2])
    ops = []
    for i in range(norb):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): a,
                (ffsim.cre_b(i), ffsim.des_b(i)): b,
                (ffsim.cre_a(i), ffsim.des_a(i), ffsim.cre_b(i), ffsim.des_b(i)): c
            }
        )
        ops.append(op)
    linops = [ffsim.linear_operator(op, norb, nelec) for op in ops]
    return linops

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("initialguesses",
                        help="path to file with initial guesses (one line = one point)")
    parser.add_argument("cost_function", help="what to optimize over")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--seniority", action="store_true",
                        help="fix the symmetry parameters to (1, 1, -2) and only optimize U")

    args = parser.parse_args()

    mol = pyscf.lib.chkfile.load_mol(args.molpath)
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(args.molpath)

    moldata = ffsim.MolecularData.from_scf(mf)

    xs_filename = (time.strftime("%Y%m%d_%H%M%S", time.localtime())
                   + "_x_opt.txt")
    with open(xs_filename,
              "a", newline="") as fp:
        fp.write(str(vars(args)) + "\n")

    if args.cost_function == "commutator":
        if args.reference == "fci":
            # f = commutator_cost_fci(moldata)
            f = commutator_cost(moldata, "fci")
        elif args.reference == "hf":
            # f = commutator_cost_hf(moldata)
            f = commutator_cost(moldata, "hf")
        else:
            raise ValueError()
    else:
        raise NotImplementedError()


    def foo_abc_112(y):
        x = np.concatenate([y, np.array(SENIORITY_ANGLES)])
        return f(x)


    initial_guesses = np.loadtxt(args.initialguesses)
    n_points = initial_guesses.shape[0]
    for i in range(n_points):
        x_0 = initial_guesses[i, :]
        if args.seniority:
            y_0 = x_0[:-2]
            res = scipy.optimize.minimize(foo_abc_112, y_0,
                                          method="L-BFGS-B",
                                          options={"maxiter": 100},
                                          callback=callback if args.verbose else None)

            print(res.message)
            res_x_extended = np.concatenate([res.x, np.array(SENIORITY_ANGLES)])
            with open(xs_filename, "ab") as fp:
                np.savetxt(fp, res_x_extended.reshape(1, -1))

        else:
            print("x0", x_0)
            res = scipy.optimize.minimize(f, x_0,
                                          method="L-BFGS-B",
                                          options={"maxiter": 100},
                                          callback=callback if args.verbose else None)


            print(res.message)
            with open(xs_filename, "ab") as fp:
                np.savetxt(fp, res.x.reshape(1, res.x.shape[0]))

