import argparse
import pyscf
import ffsim
import numpy as np
import scipy

from itertools import product

from optimize import commutator_cost, variance_cost, x_to_rotation


def sector_partitioning_metrics(moldata: ffsim.MolecularData,
                                U: np.ndarray, a: float, b: float, c: float,
                                target_accuracy=0.0016):
    """Take the Hamiltonian, partition the Hilbert space into sectors
    according to quasisymmetries specified by U, a, b, c,
    and return the following numbers:

    e_dec: lowest single-sector energy
    e_bo: energy of a subspace constructed by taking the lowest eigenvector
          from each subspace ('Born-Oppenheimer energy')
    K_en: the number of sector eigenstates needed to reach the FCI energy up to target_accuracy,
        if the eignestates are ordered according to their energies
    K_overlap: same, but the states are ordered according to their overlap with FCI
    """

    rotated_h = moldata.hamiltonian.rotated(U)
    rotated_h_linop = ffsim.linear_operator(rotated_h,
                                            norb=moldata.norb,
                                            nelec=moldata.nelec)
    e_0, v_0 = scipy.sparse.linalg.eigsh(rotated_h_linop, which="SA", k=1)
    all_projector_sets = [generalized_seniority_projectors(i, a, b, c) for i in range(moldata.norb)]
    lens = [len(p) for p in all_projector_sets]
    iterators = [range(q) for q in lens]
    projected_eigenvectors = []
    relevant_sector_indices = []

    for i, w in enumerate(product(*iterators)):
        total_projector = ffsim.FermionOperator({(): 1})
        for orbital in range(moldata.norb):
            total_projector *= all_projector_sets[orbital][w[orbital]]
        total_projector_linop = ffsim.linear_operator(total_projector, moldata.norb, moldata.nelec)
        v = np.ones(total_projector_linop.shape[0])
        support_mask = (total_projector_linop @ v).real
        dimension = int(np.sum(support_mask))
        if dimension == 0:
            continue
        print(w)

        support = np.where(support_mask == 1)[0]

        h_sub = subspace_matrix(rotated_h_linop, support)
        sector_es, sector_vs_small = np.linalg.eigh(h_sub)
        sector_vs_big = np.zeros((rotated_h_linop.shape[0], dimension), dtype="complex")
        for j in range(dimension):
            sector_vs_big[support, j] = sector_vs_small[:, j]

        # projected_h = total_projector_linop @ rotated_h_linop @ total_projector_linop
        # sector_es, sector_vs = scipy.sparse.linalg.eigsh(projected_h, k=dimension, which="SA")
        # nonzero_eig_indices = np.where(sector_es < -1e-6)[0]
        projected_eigenvectors.append((sector_es, sector_vs_big))
        relevant_sector_indices.append(i)

    born_oppenheimer_vectors = [s[1][:, np.argmin(s[0])] for s in projected_eigenvectors]
    bo_vectors_together = np.vstack(born_oppenheimer_vectors).T
    h_bo = bo_vectors_together.T.conj() @ rotated_h_linop @ bo_vectors_together
    w_bo, v_bo = np.linalg.eigh(h_bo)
    e_bo = np.min(w_bo)

    lowest_eigenvectors_pooled = np.concatenate([s[1] for s in projected_eigenvectors], axis=1)
    energies_pooled = np.concatenate([s[0] for s in projected_eigenvectors])
    # origins = []
    # for i, s in enumerate(projected_eigenvectors):
    #     origins.extend([i] * s[0].shape[0])
    # origins = np.array(origins, dtype=int)
    overlaps_with_fci = (abs(lowest_eigenvectors_pooled.T.conj() @ v_0) ** 2).flatten()
    e_dec = np.min(energies_pooled)

    energy_order_of_vectors = np.argsort(energies_pooled)
    overlap_order_of_vectors = np.argsort(overlaps_with_fci)[::-1]

    h_vecs_pooled = np.zeros((len(energies_pooled), len(energies_pooled)), dtype="complex")
    for i, j in product(range(len(energies_pooled)), repeat=2):
        h_vecs_pooled[i, j] = lowest_eigenvectors_pooled[:, i].conj() @ rotated_h_linop @ lowest_eigenvectors_pooled[
            :, j]

    energies_by_energy_order = np.zeros_like(energies_pooled)
    energies_by_overlap_order = np.zeros_like(energies_pooled)

    for i in range(len(energies_pooled)):
        indices = overlap_order_of_vectors[:i + 1]
        h = h_vecs_pooled[:, indices][indices, :]
        energies_by_overlap_order[i] = np.linalg.eigvalsh(h)[0]
        if energies_by_overlap_order[i] - e_0 < target_accuracy:
            k_overlap = i + 1
            break
    else:
        raise AssertionError("This should never happen, go take a look")


    for i in range(len(energies_pooled)):
        indices = energy_order_of_vectors[:i + 1]
        h = h_vecs_pooled[:, indices][indices, :]
        energies_by_energy_order[i] = np.linalg.eigvalsh(h)[0]
        if energies_by_energy_order[i] - e_0 < target_accuracy:
            k_en = i + 1
            break
    else:
        raise AssertionError("This should never happen, go take a look")


    return e_dec, e_bo, k_en, k_overlap


def subspace_matrix(A, support):
    dim = support.shape[0]

    A_sub = np.zeros((dim, dim), dtype="complex")

    for i, big_index in enumerate(support):
        x = np.zeros(A.shape[0], dtype="complex")
        x[big_index] = 1
        y = A @ x
        A_sub[:, i] = y[support]

    return A_sub



def generalized_seniority_projectors(orbital_index, a, b, c):
    sectors = distinct_generalized_seniority_sectors(a, b, c)
    projectors = []
    for sector in sectors:
        operator = ffsim.FermionOperator({(): 0})
        for j in sector:
            if j == 0:
                operator += generalized_seniority_and_constant(orbital_index, -1, -1, 1, 1)
            elif j == 1:
                operator += generalized_seniority_and_constant(orbital_index, 1, 0, -1, 0)
            elif j == 2:
                operator += generalized_seniority_and_constant(orbital_index, 0, 1, -1, 0)
            elif j == 3:
                operator += generalized_seniority_and_constant(orbital_index, 0, 0, 1, 0)
            else:
                raise ValueError()
        projectors.append(operator)
    return projectors


def distinct_generalized_seniority_sectors(a, b, c, atol=1e-1):
    sector_eigenvalues = [0, a, b, a + b + c]

    # https://stackoverflow.com/a/38924644/
    partitions = [] # Found partitions
    for i, e in enumerate(sector_eigenvalues): # Loop over each element
        found = False # Note it is not yet part of a known partition
        for p in partitions:
            if np.isclose(e, sector_eigenvalues[p[0]], atol=atol): # Found a partition for it!
                p.append(i)
                found = True
                break
        if not found: # Make a new partition for it.
            partitions.append([i])
    return partitions


def trace_of_a_diagonal_projector(proj) -> int:
    """take a diagonal projector written as a sparse operator and find its trace"""
    v = np.ones(proj.shape[0])
    return int(np.sum(proj @ v).real)


def generalized_seniority_and_constant(i, a, b, c, d):
    return ffsim.FermionOperator(
                {
                    (ffsim.cre_a(i), ffsim.des_a(i)): a,
                    (ffsim.cre_b(i), ffsim.des_b(i)): b,
                    (ffsim.cre_a(i), ffsim.des_a(i), ffsim.cre_b(i), ffsim.des_b(i)): c,
                    (): d
                }
            )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("xs",
                        help="path to file with data points (one line = one point)")
    args = parser.parse_args()

    mol = pyscf.lib.chkfile.load_mol(args.molpath)
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(args.molpath)
    moldata = ffsim.MolecularData.from_scf(mf)

    commutator_fci = commutator_cost(moldata, "fci")
    commutator_hf = commutator_cost(moldata, "hf")
    variance_fci = variance_cost(moldata, "fci")
    variance_hf = variance_cost(moldata, "hf")

    xs = np.loadtxt(args.xs, skiprows=1)
    n_points = xs.shape[0]

    data_filename = args.xs + "_metrics.txt"
    fieldnames = ["V_fci", "V_hf", "C_fci", "C_hf", "b", "c", "E_dec", "E_bo", "K_en", "K_overlap"]
    print(" ".join(fieldnames) + "\n")

    with open(data_filename,
              "a", newline="") as fp:
        fp.write(" ".join(fieldnames) + "\n")

    for i in range(n_points):
        x = xs[i, :]
        phi1, phi2 = x[-2], x[-1]
        a_opt = np.sin(phi1) * np.cos(phi2)
        b_opt = np.sin(phi1) * np.sin(phi2)
        c_opt = np.cos(phi1)
        print(a_opt, b_opt, c_opt)

        U = x_to_rotation(x[:-2], moldata.norb)

        e_dec, e_bo, k_en, k_overlap = sector_partitioning_metrics(moldata, U, a_opt, b_opt, c_opt)

        costs_and_bc = np.array([variance_fci(x), variance_hf(x),
                                 commutator_fci(x), commutator_hf(x),
                                 b_opt / a_opt, c_opt / a_opt, e_dec, e_bo, k_en, k_overlap]).real
        print(costs_and_bc)

        with open(data_filename, "ab") as fp:
            np.savetxt(fp, costs_and_bc.reshape(1, costs_and_bc.shape[0]))