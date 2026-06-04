from math import comb

import ffsim
import numpy as np
import argparse
from scipy.stats import special_ortho_group
from uuid import uuid4
import scipy.sparse.linalg as spla
import pyscf

from itertools import product

from cost_functions import rotation_to_x, x_to_rotation
from optimize import SENIORITY_ANGLES, commutator_cost, commutator_cost_2


# def generic_h(parity_table, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()
#     n_spin_orb = parity_table.shape[1]
#     one_body_tensor = np.zeros((n_spin_orb, n_spin_orb))
#     two_body_tensor = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
#
#     for p, q in product(range(n_spin_orb), repeat=2):
#         if (parity_table[:, p] == parity_table[:, q]).all():
#             one_body_tensor[p, q] = rng.normal()
#
#     one_body_tensor += one_body_tensor.T
#     one_body_tensor = one_body_tensor / 2
#
#     for p, q, r, s in product(range(n_spin_orb), repeat=4):
#         if ((parity_table[:, p] + parity_table[:, r]) % 2
#             == (parity_table[:, s] + parity_table[:, q]) % 2).all():
#             two_body_tensor[p, q, r, s] = rng.normal()
#
#     two_body_tensor += np.transpose(two_body_tensor)
#     two_body_tensor = two_body_tensor / 2
#
#     h = ffsim.hamiltonians.MolecularHamiltonianSpinless(
#         one_body_tensor, two_body_tensor
#     )
#
#
#     return h


def generic_spinful_h(condensed_parity_table, rng=None):
    """Assuming that every alpha column of the parity table is equal to its beta counterpart"""
    if rng is None:
        rng = np.random.default_rng()
    norb = condensed_parity_table.shape[1]
    one_body_tensor = np.zeros((norb, norb))
    two_body_tensor = np.zeros((norb, norb, norb, norb))

    for p, q in product(range(norb), repeat=2):
        if (condensed_parity_table[:, p] == condensed_parity_table[:, q]).all():
            one_body_tensor[p, q] = rng.normal()
            # print(p, q)

    one_body_tensor += one_body_tensor.T
    one_body_tensor = one_body_tensor / 2

    # print(one_body_tensor)

    for p, q, r, s in product(range(norb), repeat=4):
        if ((condensed_parity_table[:, p] + condensed_parity_table[:, r]) % 2
            == (condensed_parity_table[:, s] + condensed_parity_table[:, q]) % 2).all():
            two_body_tensor[p, q, r, s] = rng.normal()
            # print(p, q, r, s)

    two_body_tensor += np.transpose(two_body_tensor)
    two_body_tensor = two_body_tensor / 2

    tbt_compressed = pyscf.ao2mo.restore(8, two_body_tensor.real, norb)

    tbt_uncompressed = pyscf.ao2mo.restore(1, tbt_compressed, args.norb)

    h = ffsim.hamiltonians.MolecularHamiltonian(
        one_body_tensor, tbt_uncompressed
    )


    # f = ffsim.fermion_operator(h)
    # f += f.adjoint()

    # h_hermitized = ffsim.MolecularHamiltonian.from_fermion_operator(f)

    # return h_hermitized

    return h

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate test Hamiltonians with known orbital parity symmetries",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("norb", help="Number of atomic orbitals; the number of spin-orbitals is 2x that",
                        type=int)
    parser.add_argument("--h_type",
                        help="type of Hamiltonian to generate",
                        default="seniority")
    parser.add_argument("--seed", help="RNG seed",
                        default=None, type=int)
    parser.add_argument("--no_U", action="store_true")
    parser.add_argument("--U_close_to_id", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.h_type == "seniority":
        # parity_table = np.zeros((args.norb, args.norb * 2), dtype=int)
        # for i in range(args.norb):
        #     parity_table[i][2 * i] = 1
        #     parity_table[i][2 * i + 1] = 1
        condensed_parity = np.eye(args.norb, dtype=int)
    else:
        raise ValueError("h_type must be 'seniority'")

    # print(condensed_parity)

    h = generic_spinful_h(condensed_parity, rng)
    # print(h)

    # tbt = h.two_body_tensor.real
    #
    # tbt_compressed = pyscf.ao2mo.restore(8, tbt, args.norb)
    #
    # tbt_uncompressed = pyscf.ao2mo.restore(1, tbt_compressed, args.norb)

    if args.no_U:
        U = np.eye(args.norb)
    elif args.U_close_to_id:
        x0 = rng.normal(size=comb(args.norb, 2), scale=1e-3)
        U = x_to_rotation(x0, args.norb)
    else:
        U = special_ortho_group.rvs(dim=args.norb, random_state=rng)
    # print(U)

    rotated_h = h.rotated(U)

    rot_moldata = ffsim.MolecularData(
        spin=0,
        nelec=(args.norb // 2, args.norb // 2),
        one_body_integrals=rotated_h.one_body_tensor.real,
        two_body_integrals=rotated_h.two_body_tensor.real,
        norb=args.norb,
        core_energy=0,
        mo_occ=np.array([2.] * (args.norb // 2) + [0.] * (args.norb - args.norb // 2))
    )

    h_op = ffsim.linear_operator(rotated_h, args.norb,
                                 rot_moldata.nelec)

    # w, v = spla.eigs(h_op, k=34, which="SR")
    # print(w)

    x = rotation_to_x(U.T)

    U_2 = x_to_rotation(x, args.norb)
    print("verifying sameness of unitary")
    print(np.linalg.norm(U_2 - U.T))

    f = commutator_cost(rot_moldata, "fci")
    f_2 = commutator_cost_2(rot_moldata, "fci")

    x_with_sen = np.concatenate([x, SENIORITY_ANGLES])
    print(f(x_with_sen))

    print("x")
    print(x)
    print(f_2(x))



    stamp = uuid4().hex[:6]
    # h_name = "test_h_" + args.h_type + "_"+ str(args.norb) + "_" + stamp
    h_name = "test_h"

    rot_moldata.to_fcidump(h_name + ".FCIDUMP")

    # scf = pyscf.tools.fcidump.to_scf(h_name + ".FCIDUMP")
    # scf.e_tot = -9999.999
    # scf.mo_occ = np.array([2.] * (args.norb // 2) + [0.] * (args.norb - args.norb // 2))
    # scf.mo_coeff = np.eye(args.norb)
    # moldata = ffsim.MolecularData.from_scf(scf)

    # fd = pyscf.tools.fcidump.read(h_name + ".FCIDUMP")
    #
    # h1e = fd['H1']  # One-body integrals, shape: (norb, norb)
    # h2e = fd['H2']
    #
    # moldata = ffsim.MolecularData(
    #     spin=0,
    #     nelec=scf.mol.nelec,
    #     one_body_integrals=h1e,
    #     two_body_integrals=h2e,
    #     norb=scf.mol.nao,
    #     core_energy=0
    # )

    moldata = ffsim.MolecularData.from_fcidump(h_name + ".FCIDUMP")

    f_3 = commutator_cost_2(moldata, "fci")
    print(f_3(x))
    np.savetxt(h_name + "_U.txt", U)

    np.savetxt(h_name + "_x0.txt", x)

    np.savetxt(h_name + "_x0_sen.txt", x_with_sen)