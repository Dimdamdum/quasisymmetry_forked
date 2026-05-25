"""Find approximate symmetries in the form of U * s_pq * U^*,
where s_pq is a parity of population of orbitals p, q"""

import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import sys
import networkx as nx
import matplotlib.pyplot as plt


from typing import Tuple, Callable
from itertools import combinations
from functools import cache
from matplotlib.colors import LogNorm

@cache
def make_quartets(norb: int, nelec):
    quartets = {}
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
    for i in range(norb):
        for j in range(i + 1, norb):
            quartets[(i, j)] = local_parities[i] @ local_parities[j]
    return quartets


def quartet_commutators(moldata: ffsim.MolecularData, state, U):
    """Return a weighted graph with edge weights being the commmutator norms"""
    G = nx.complete_graph(moldata.norb)
    quartets = make_quartets(moldata.norb, moldata.nelec)

    rotated_h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                      norb=moldata.norb, nelec=moldata.nelec)
    rotated_state = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)

    for i in range(moldata.norb):
        for j in range(i + 1, moldata.norb):
            commutator = rotated_h @ quartets[(i, j)] - quartets[(i, j)] @ rotated_h
            state_after_commutator = commutator @ rotated_state
            G[i][j]['weight'] = np.linalg.norm(state_after_commutator)**2

    return G



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    # parser.add_argument("initialguesses",
    #                     help="path to file with initial guesses (one line = one point)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")

    args = parser.parse_args()

    print("loading the hamiltonian")
    mol = pyscf.lib.chkfile.load_mol(args.molpath)
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(args.molpath)

    moldata = ffsim.MolecularData.from_scf(mf)
    print("done")

    xs_filename = (time.strftime("%Y%m%d_%H%M%S", time.localtime())
                   + "_quartet_opt.txt")
    # with open(xs_filename,
    #           "a", newline="") as fp:
    #     fp.write(parser.prog + "\n")
    #     fp.write(str(vars(args)) + "\n")

    print("creating h linop")
    h =  ffsim.linear_operator(moldata.hamiltonian,
                                      norb=moldata.norb, nelec=moldata.nelec)
    print("finding fci")
    fci_energy, fci_state = scipy.sparse.linalg.eigsh(h, k=1, which="SA")
    fci_energy = fci_energy[0]
    fci_state = fci_state[:, 0]
    print("finding commutators")
    G = quartet_commutators(moldata,
                            fci_state,
                            np.eye(moldata.norb))

    print(G)
    for e in G.edges(data=True):
        print(e[0], e[1], e[2]['weight'])

    plt.figure()
    plt.imshow(mf.mo_coeff, cmap="PuOr", vmin=-1, vmax=1)
    plt.yticks(range(mol.nao), mol.ao_labels())
    plt.title("Canonical orbitals \n" + args.molpath)
    plt.colorbar()
    plt.savefig("canonical_orbitals.png", dpi=600, bbox_inches="tight", format="png")
    # plt.show()

    adj = nx.to_numpy_array(G)
    plt.figure()
    plt.imshow(adj, norm=LogNorm(vmin=1e-4, vmax=1))
    plt.colorbar()
    plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|FCI\\rangle||^2$ \n" + args.molpath)
    plt.savefig("quartets.png", dpi=600, bbox_inches="tight", format="png")
    plt.show()



