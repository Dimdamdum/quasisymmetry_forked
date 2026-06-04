"""Find approximate symmetries in the form of U * s_pq * U^*,
where s_pq is a parity of population of orbitals p, q"""

import argparse
import json

import numpy as np
import time
import ffsim
import scipy
import pyscf
import networkx as nx
import matplotlib.pyplot as plt
import jax
import uuid

from functools import cache
from matplotlib.colors import LogNorm
from math import comb
from tqdm import tqdm
from pathlib import Path

from optimize import x_to_rotation
from xs_to_metrics import subspace_matrix

@cache
def make_quartets(norb: int, nelec):
    """pair parity operators s_i s_j. for i=j, just s_i"""
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
        for j in range(i, norb):
            if i == j:
                quartets[(i, j)] = local_parities[i]
            else:
                quartets[(i, j)] = local_parities[i] @ local_parities[j]
    return quartets


def all_quartet_commutators(moldata: ffsim.MolecularData, state, U):
    """Return a weighted graph with edge weights being the commmutator norms.
    Self-edges denote single-site parity commutators"""
    G = nx.complete_graph(moldata.norb)
    quartets = make_quartets(moldata.norb, moldata.nelec)

    rotated_h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                      norb=moldata.norb, nelec=moldata.nelec)
    rotated_state = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)

    for i in range(moldata.norb):
        for j in range(i, moldata.norb):
            nc_factor = quartet_noncommutation_factor(rotated_h, rotated_state, i, j)
            if i != j:
                G[i][j]['weight'] = nc_factor
            else:
                G.add_edge(i, i, weight=nc_factor)

    return G


def quartet_noncommutation_factor(h,
                       state: np.ndarray,
                       i: int,
                       j: int):
    quartets = make_quartets(moldata.norb, moldata.nelec)
    indices = (i, j) if i <= j else (j, i)
    commutator = h @ quartets[indices] - quartets[indices] @ h
    state_after_commutator = commutator @ state
    return np.linalg.norm(state_after_commutator)**2


def visualize_nc(moldata, state, U, mo=True, save=False):
    G = all_quartet_commutators(moldata,
                                state,
                                U)

    if mo:
        plt.figure()
        plt.imshow(scf.mo_coeff @ U, cmap="PuOr", vmin=-1, vmax=1)
        plt.yticks(range(mol.nao), mol.ao_labels())
        plt.xticks(range(mol.nao), range(mol.nao))
        plt.title("Optimized orbitals \n" + args.molpath)
        plt.colorbar()
        if save:
            plt.savefig("canonical_orbitals.png", dpi=600, bbox_inches="tight", format="png")

    adj = np.triu(nx.to_numpy_array(G), 0)
    plt.figure()
    plt.imshow(adj, norm=LogNorm(vmin=1e-4, vmax=1))
    plt.colorbar()
    plt.xticks(range(mol.nao), range(mol.nao))
    plt.yticks(range(mol.nao), range(mol.nao))
    if args.reference == "fci":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|FCI\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|FCI\\rangle||^2$")

        if save:
            plt.savefig("quartets.png", dpi=600, bbox_inches="tight", format="png")
    elif args.reference == "hf":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|HF\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|HF\\rangle||^2$")
        if save:
            plt.savefig("quartets_hf.png", dpi=600, bbox_inches="tight", format="png")
    plt.show()


def nc_cost(moldata, state, quartet_graph):
    def f(x):
        U = x_to_rotation(x, moldata.norb)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                      norb=moldata.norb, nelec=moldata.nelec)
        rotated_state = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)
        total_nc = 0
        for (i, j) in quartet_graph.edges():
            total_nc += quartet_noncommutation_factor(h, rotated_state, i, j)
        return total_nc
    return f


def quartet_sectors(quartets, norb, nelec):
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    bitstrings = ffsim.addresses_to_strings(range(dim), norb, nelec,
                                            bitstring_type=ffsim.BitstringType.INT, concatenate=False)

    quartet_bit_masks = []
    for q in quartets:
        quartet_bit_masks.append(2 ** q[0] + 2 ** q[1])

    sectors = {}
    for i in range(state.shape[0]):
        ab_parities = bitstrings[0][i] ^ bitstrings[1][i]
        sector_label = tuple(
            (int.bit_count(int(ab_parities & q)) % 2 for q in quartet_bit_masks)
        )
        sectors.setdefault(sector_label, []).append(i)

    return sectors


def sector_metrics(moldata, state, U,
                   sectors,
                   target_energy,
                   max_states_per_sector=500):
    print("sector dimensions:")
    print([len(v) for v in sectors.values()])


    rotated_h_linop = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                            norb=moldata.norb, nelec=moldata.nelec)
    rotated_state = ffsim.apply_orbital_rotation(state, U,
                                                 moldata.norb, moldata.nelec)

    sector_hamiltonians = {k: subspace_matrix(rotated_h_linop, v)
                           for k, v in sectors.items()}

    sector_eigen_decompositions = {}
    for sector_label, h_local in sector_hamiltonians.items():
        if h_local.shape[0] <= max_states_per_sector:
            sector_eigen_decompositions[sector_label] = np.linalg.eigh(h_local)
        else:
            sector_eigen_decompositions[sector_label] = scipy.sparse.linalg.eigsh(
                h_local, which="SA", k=max_states_per_sector)

    pooled_energies = {}
    for sector_label in sectors.keys():
        for i in range(len(sector_eigen_decompositions[sector_label][0])):
            pooled_energies[(sector_label, i)] = (
                sector_eigen_decompositions[sector_label][0][i]
            )

    vector_labels = list(pooled_energies.keys())
    energy_order = np.argsort(list(pooled_energies.values()))

    # pulling vectors by their local energies
    current_state_vectors = np.array(())
    for i in tqdm(range(len(energy_order))):
        sector_label = vector_labels[energy_order[i]][0]
        vector_id_in_sector = vector_labels[energy_order[i]][1]
        next_vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
        next_vector_big = np.zeros(rotated_h_linop.shape[0], dtype="complex")
        next_vector_big[sectors[sector_label]] = next_vector

        if i == 0:
            current_state_vectors = next_vector_big
        else:
            current_state_vectors = np.vstack([current_state_vectors, next_vector_big])

        h_subspace = current_state_vectors.conj() @ rotated_h_linop @ current_state_vectors.T

        if i == 0:
            subspace_energy = h_subspace.real
        elif i == 1:
            w, v = np.linalg.eigh(h_subspace)
            subspace_energy = w[0]
        else:
            subspace_energy = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")[0][0]


        if subspace_energy < target_energy:
            # print("K_en = {0:}".format(i + 1))
            K_en =  i + 1
            break
    else:
        K_en = np.nan
        print("Chemical accuracy not reached, try more vectors per sector")

    pooled_overlaps = {}
    for sector_label in sectors.keys():
        for i in range(len(sector_eigen_decompositions[sector_label][0])):
            vector_id_in_sector = vector_labels[energy_order[i]][1]
            vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
            projected_reference = rotated_state[sectors[sector_label]]
            pooled_overlaps[(sector_label, i)] = abs(vector.T.conj() @ projected_reference)**2

    overlap_order = np.argsort(list(pooled_overlaps.values()))[::-1]

    # pulling vectors by their local overlaps with FCI
    current_state_vectors = np.array(())
    for i in tqdm(range(len(overlap_order))):
        sector_label = vector_labels[overlap_order[i]][0]
        vector_id_in_sector = vector_labels[overlap_order[i]][1]
        next_vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
        next_vector_big = np.zeros(rotated_h_linop.shape[0], dtype="complex")
        next_vector_big[sectors[sector_label]] = next_vector

        if i == 0:
            current_state_vectors = next_vector_big
        else:
            current_state_vectors = np.vstack([current_state_vectors, next_vector_big])

        h_subspace = current_state_vectors.conj() @ rotated_h_linop @ current_state_vectors.T

        if i == 0:
            subspace_energy = h_subspace.real
        elif i == 1:
            w, v = np.linalg.eigh(h_subspace)
            subspace_energy = w[0]
        else:
            subspace_energy = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")[0][0]


        if subspace_energy < target_energy:
            # print("K_overlap = {0:}".format(i + 1))
            K_overlap =  i + 1
            break
    else:
        K_overlap = np.nan
        print("Chemical accuracy not reached, try more vectors per sector")

    energy_vectors_used = [vector_labels[i] for i in energy_order[:K_en]]
    energy_sectors_used = list(set([s[0] for s in energy_vectors_used]))

    overlap_vectors_used = [vector_labels[i] for i in overlap_order[:K_overlap]]
    overlap_sectors_used = list(set([s[0] for s in overlap_vectors_used]))


    # born_oppenheimer_vectors
    born_oppenheimer_vectors = []
    for sector_label, (w, v) in sector_eigen_decompositions.items():
        lowest_vector_id = np.argmin(w)
        lowest_vector = v[:, lowest_vector_id]
        next_vector_big = np.zeros(rotated_h_linop.shape[0], dtype="complex")
        next_vector_big[sectors[sector_label]] = lowest_vector

        born_oppenheimer_vectors.append(next_vector_big)

    bo_vectors_stacked = np.vstack(born_oppenheimer_vectors)
    h_subspace = bo_vectors_stacked.conj() @ rotated_h_linop @ bo_vectors_stacked.T

    es_bo, _ = scipy.sparse.linalg.eigsh(h_subspace, which="SA", k=1)

    sector_data = {"K_en": K_en,
                   "K_overlap": K_overlap,
                   "Energy sectors": energy_sectors_used,
                   "Overlap sectors": overlap_sectors_used,
                   "Decoupled_energy": min(list(pooled_energies.values())),
                   "BO energy": min(es_bo)
                   }

    return sector_data


def args_parser():
    parser = argparse.ArgumentParser(description="Optimize orbitals and/or graph structure "
                                                 "to find two-orbital quasisymmetries ('quartets')",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--optimization_mode",
                        help="Optimization regime. Available keys: \n"
                             "None: No optimization \n"
                             "OO: Orbital optimization for a fixed connection graph",
                        default=None,
                        )
    parser.add_argument("--visualize", action="store_true",
                        help="Draw the NC metrics after optimization")
    parser.add_argument("--quartet_graph", default="topm")
    parser.add_argument("--initial_guess", default=None)
    parser.add_argument("--initial_guess_scale",
                        default=-2, type=int)
    parser.add_argument("--dontsave", action="store_true")
    return parser


if __name__=="__main__":
    parser = args_parser()
    args = parser.parse_args()

    # print("loading the hamiltonian")
    # mol = pyscf.lib.chkfile.load_mol(args.molpath)
    # mf = pyscf.scf.RHF(mol)
    # mf.update_from_chk(args.molpath)
    # moldata = ffsim.MolecularData.from_scf(mf)

    p = Path(args.molpath)
    if p.suffix == ".chk":
        mol = pyscf.lib.chkfile.load_mol(args.molpath)
        scf = pyscf.scf.RHF(mol)
        scf.update_from_chk(args.molpath)

        moldata = ffsim.MolecularData.from_scf(scf)
    elif p.suffix == ".FCIDUMP":
        scf = pyscf.tools.fcidump.to_scf(args.molpath)
        moldata = ffsim.MolecularData.from_scf(scf)

    print("finding fci") # we need it for energy metrics anyway
    cisolver = pyscf.fci.FCI(scf)
    cisolver.kernel()

    if args.reference == "fci":
        state = np.array(cisolver.ci.reshape((-1,)), dtype="complex") # ffsim will complain without dtype='complex'
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("--reference can be 'hf' or 'fci'")

    mo_quartets_graph = all_quartet_commutators(moldata, state, np.eye(moldata.norb))
    mo_quartets_adj = np.triu(nx.adjacency_matrix(mo_quartets_graph).todense(), 1)
    iu = np.triu_indices(moldata.norb, 1)
    mo_quartets_sorted = np.sort(mo_quartets_adj[iu])

    best_mo_quartet_indices = np.argsort(mo_quartets_adj[iu])[:moldata.norb]

    sum_of_lowest_mo_quartets = np.sum(mo_quartets_sorted[:moldata.norb])
    print("Sum of lowest m quartets (canonical orbitals)", sum_of_lowest_mo_quartets)

    if args.quartet_graph == "ring":
        quartet_graph = nx.cycle_graph(moldata.norb)
    elif args.quartet_graph == "matching":
        edges = [(2 * i, 2 * i + 1) for i in range(moldata.norb // 2)]
        quartet_graph = nx.from_edgelist(edges)
    elif args.quartet_graph == "topm" or args.quartet_graph == "greedy":
        best_quartets = tuple([(int(iu[0][i]), int(iu[1][i]))
                               for i in best_mo_quartet_indices])
        quartet_graph = nx.from_edgelist(best_quartets)
    elif args.quartet_graph == "complete":
        quartet_graph = nx.complete_graph(moldata.norb)
    else:
        raise ValueError()
    print(list(quartet_graph.edges()))

    print(str(vars(args)))

    if args.optimization_mode is None or args.optimization_mode == "None":
        print("No orbital optimization")
        U = np.eye(moldata.norb)
        f = nc_cost(moldata, state, quartet_graph)
    elif args.optimization_mode == "OO":
        print("Optimizing orbitals keeping the quartet graph fixed")

        f = nc_cost(moldata, state, quartet_graph)

        if args.initial_guess == "random":
            rng = np.random.default_rng()
            x0 = rng.normal(scale=10**(args.initial_guess_scale),
                            size=comb(moldata.norb, 2))
            print("NC cost, random initial guess {0:2.4f}".format(f(x0)))
        else:
            x0 = np.zeros(comb(moldata.norb, 2))
            print("NC cost, canonical orbitals {0:2.4f}".format(f(x0)))

        res = scipy.optimize.minimize(f, x0, method="L-BFGS-B", options={"maxiter": 500})
        print(res)
        U = x_to_rotation(res.x, moldata.norb)
    else:
        raise ValueError()

    optimized_quartets_graph = all_quartet_commutators(moldata, state, U)
    opt_quartets_matrix = np.triu(nx.adjacency_matrix(optimized_quartets_graph).todense(), 1)
    opt_quartets_sorted = np.sort(opt_quartets_matrix[iu])
    sum_of_lowest_opt_quartets = np.sum(opt_quartets_sorted[:moldata.norb])
    print("Sum of lowest m quartets (optimized orbitals)", sum_of_lowest_opt_quartets)

    best_quartet_indices = np.argsort(opt_quartets_matrix[iu])[:moldata.norb]
    best_quartets = tuple([(int(iu[0][i]), int(iu[1][i]))
                     for i in best_quartet_indices])
    print("m lowest NC factor quartets")
    print(best_quartets)

    print("creating sectors")

    spanning_edges = list(nx.minimum_spanning_edges(optimized_quartets_graph, keys=False, data=False))

    print("Minimum spanning tree")
    print(spanning_edges)

    sectors = quartet_sectors(spanning_edges, moldata.norb, moldata.nelec)

    sector_data = sector_metrics(moldata, state, U, sectors,
                   target_energy=cisolver.e_tot + 0.0016)

    for k, v in sector_data.items():
        if k not in ("Energy sectors", "Overlap sectors"):
            print(k, v)

    if args.visualize:
        visualize_nc(moldata, state, U)

    output = {"vars": vars(args),
              "quartet_graph_for_opt": list(quartet_graph.edges()),
              "f(canonical)": f(np.zeros(comb(moldata.norb, 2))),
              "lowest_m_quartet_sum_mo": sum_of_lowest_mo_quartets,
              "f(optimized)": res.fun if 'res' in globals() else None,
              "lowest_nc_quartets": best_quartets,
              "lowest_m_quartet_sum_opt": sum_of_lowest_opt_quartets,
              "sector_data": sector_data,
              "Minimum spanning tree": spanning_edges,
              "FCI energy": cisolver.e_tot}
    stamp = uuid.uuid4().hex[:6]

    path_to_mol = Path(args.molpath)

    if not args.dontsave:
        title = "quartets_" + path_to_mol.name + "_" + stamp
        with open(title, "a") as fp:
            json.dump(output, fp)

        np.savetxt("quartet_U_" + path_to_mol.name + "_" + stamp + ".txt", U)
        np.savetxt("quartets_nc_" + path_to_mol.name + "_" + stamp + ".txt",
                   opt_quartets_matrix)





