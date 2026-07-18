"""Tests for functions in src/cluster_number_operators.py on cluster numbers as quasisymmetries"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pytest
import scipy
import ffsim
from src.cluster_number_operators import build_one_orb_num_operators, build_two_orb_num_operators, number_matrix_to_operators, from_num_operator_to_expnum_operator, number_and_parity_symmetry_sectors, integers_to_phases_polynomial, get_cluster_indices
from scipy.sparse.linalg import LinearOperator
from src.sector_utils import symmetry_sectors
import tempfile
import shutil
from src.dmrg_solver import Block2DMRGSolver
from scipy.stats import unitary_group

def test_spin_orbital_occupations():
    """Testing to refresh scipy/ffsim basis ordering and operator usage"""
    norb = 3
    nelec = (1, 2) # alpha, beta
    # generate spin-orbital number operators
    alpha_number_operators = []
    beta_number_operators = []
    for i in range(norb):
        n_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): +1 # 1 * a_ialpha^dag a_ialpha
            }
        )
        n_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): +1 # 1 * a_ibeta^dag a_ibeta
            }
        )
        alpha_number_operators.append(ffsim.linear_operator(n_alpha, norb, nelec))
        beta_number_operators.append(ffsim.linear_operator(n_beta, norb, nelec))

    spin_orb_number_operators = alpha_number_operators + beta_number_operators

    # expected basis ordering: |alpha occupations -> 100; beta occupations -> 110>, |100; 101>, ...
    expected_occupations = [[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1], 
                            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], 
                            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]]
    # later, when using bitstrings, watch out for possible (intra-alpha/beta) flips.
    # get actual basis ordering
    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in spin_orb_number_operators]
        assert list(occupations) == expected_occupations[i]

def test_build_oneortwo_orb_num_operators():
    norb = 3
    nelec = (1, 2) # alpha, beta
    # get list of one-orbital alpha+beta occ number operators
    one_orbital_num_operators = build_one_orb_num_operators(norb, nelec)

    expected_occupations = [[2, 1, 0], [2, 0, 1], [1, 1, 1], 
                            [1, 2, 0], [1, 1, 1], [0, 2, 1], 
                            [1, 1, 1], [1, 0, 2], [0, 1, 2]]
    
    assert len(one_orbital_num_operators) == norb

    assert type(one_orbital_num_operators[0]) == scipy.sparse.linalg._interface._CustomLinearOperator

    assert np.allclose(one_orbital_num_operators[0] @ np.eye(9)[0] - 2 * np.eye(9)[0], np.zeros(9))

    two_orbital_num_operators = build_two_orb_num_operators(norb, nelec) # orbital pairs: 01, 02, 12

    assert len(two_orbital_num_operators) == (norb**2 - norb)/2

    expected_two_orb_occupations = [[3, 2, 1], [2, 3, 1], [2, 2, 2], 
                                [3, 1, 2], [2, 2, 2], [2, 1, 3], 
                                [2, 2, 2], [1, 3, 2], [1, 2, 3]]

    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in one_orbital_num_operators]
        two_orb_occupations = [np.vdot(basis_elem, op @ basis_elem) for op in two_orbital_num_operators]
        assert list(occupations) == expected_occupations[i]
        assert list(two_orb_occupations) == expected_two_orb_occupations[i]

def test_integers_to_phases_polynomial():
    def P(x, N):
        coeffs = integers_to_phases_polynomial(N)
        result = sum([coeffs[i] * x**i for i in range(N+1)])
        return(result)

    for N in range(10):
        for n in range(N + 1):
            omega = np.exp(1j * 2 * np.pi / (N+1))
            assert (omega ** n - P(n, N)).round(10) == 0

def test_from_num_operator_to_expnum_operator():
    integers = [0, 4, 6, 31, 19, 2]
    matrix = np.diag(integers)
    max_num_eval = max(integers)
    dim = len(integers)
    exp_integers = [np.exp(1.j * 2 * np.pi * n / (max_num_eval+1)) for n in integers]
    num_operator = LinearOperator((dim, dim), matvec=lambda v: matrix @ v)
    expnum_operator = from_num_operator_to_expnum_operator(num_operator, max_num_eval)
    for i in range(dim):
        basis_el = np.eye(dim)[i]
        assert np.allclose(expnum_operator @ basis_el, exp_integers[i] * basis_el, atol = 1e-15, rtol = 1e-10)

def test_number_matrix_to_operators():
    norb = 3
    nelec = (1, 2)
    cluster_matrix = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1]])
    # number operators
    cluster_num_operators = number_matrix_to_operators(cluster_matrix, norb, nelec)
    expected_cluster_num_operators = [build_one_orb_num_operators(norb, nelec)[0], build_two_orb_num_operators(norb, nelec)[1], LinearOperator((9, 9), matvec=lambda v: 3 * v)]
    # exponentiated versions
    cluster_expnum_operators = number_matrix_to_operators(cluster_matrix, norb, nelec, expnum=True)
    expected_cluster_expnum_operators = [from_num_operator_to_expnum_operator(expected_cluster_num_operators[i], 2 * sum(cluster_matrix[i])) for i in range(3)]
    for i in range(9):
        basis_elem = np.eye(9)[i]
        for j in range(3):
            assert np.allclose(cluster_num_operators[j] @ basis_elem, expected_cluster_num_operators[j] @ basis_elem)
            assert np.allclose(cluster_expnum_operators[j] @ basis_elem, expected_cluster_expnum_operators[j] @ basis_elem)

def test_number_and_parity_symmetry_sectors():
    # both parities and numbers
    norb = 3
    nelec = (1, 2)
    cluster_number_matrix = np.array([[1, 0, 0], [1, 0, 1]])
    cluster_parity_matrix = np.array([[0, 0, 1]])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    expected_sectors = {((2,2), (0,)): [0],
                        ((2,3), (1,)): [1],
                        ((1,2), (1,)): [2,4,6],
                        ((1,1), (0,)): [3],
                        ((0,1), (1,)): [5],
                        ((1,3), (0,)): [7],
                        ((0,2), (0,)): [8]}
    assert sectors == expected_sectors
    
    # only parities
    norb = 4
    nelec = (2, 3)
    cluster_number_matrix = np.array([])
    cluster_parity_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    aleksey_sectors = symmetry_sectors(cluster_parity_matrix, norb, nelec)
    expected_sectors = {((), k): v for k, v in aleksey_sectors.items()}
    assert sectors == expected_sectors

    # only numbers
    norb = 3
    nelec = (1, 1)
    cluster_number_matrix = np.array([[0, 1, 1]])
    cluster_parity_matrix = np.array([])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    expected_sectors = {((0,), ()): [0],
                        ((1,), ()): [1,2,3,6],
                        ((2,), ()): [4,5,7,8]}
    assert sectors == expected_sectors

def test_get_cluster_indices():
    cluster_matrices = [
        np.array([
            [1, 0]
        ]),
        np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ]),
        np.array([
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
    ]

    clusters_list = [get_cluster_indices(cluster_matrix) for cluster_matrix in cluster_matrices]

    expected_clusters_list = [
        [[0],[1]],
        [[0], [4,5], [3], [1,2,6]],
        [[3, 4], [1, 4], [1], [], [0,2]]
    ]

    for i in range(len(clusters_list)):
        clusters = clusters_list[i]
        expected_clusters = expected_clusters_list[i]
        assert len(clusters) == len(expected_clusters)
        for j in range(len(clusters)):
            assert np.all(clusters[j] == expected_clusters[j])

# the following is a good reminder on how orbital rotations are implemented
# both on fock space level, and on 1-, 2rdm level
def test_rdm_rotations():
    norb = 4
    nelec = (2, 1)  # (Na, Nb)
    dim = ffsim.dim(norb, nelec)

    tmp_dir = tempfile.mkdtemp(prefix='block2_test_')

    # dummy object to access, just to access solver.to_ci_vector
    solver = Block2DMRGSolver(
        h1e=np.zeros((4, 4)),
        g2e=np.zeros((4, 4, 4, 4)),
        ecore=0.0,
        n_elec=(2,1),
        spin=None,
        store_dir=tmp_dir,
        n_threads=1,
        save_integrals=False
    )

    # random mps
    mps = solver.driver.get_random_mps(tag='RAND', bond_dim=5, nroots=1)

    # get full state
    psi = solver.to_ci_vector(ket=mps)
    # print(f"Length of psi: {len(psi)}, compare with dim = {dim}")

    # Get RDMs
    rdm1_a, rdm1_b = solver.driver.get_1pdm(mps)
    rdm2_aa, rdm2_ab, rdm2_bb = solver.driver.get_2pdm(mps)
    # Declared formulas in block2 software:
    #
    # rdm1_sigma[p, q] = <mps|a^dag_{p sigma}a_{q sigma}|mps>
    #
    # rdm2_aa[p, q, r, s] = <mps| a^dag_{p,alpha} a^dag_{q,alpha} a_{r,alpha} a_{s,alpha} |mps>   (alpha-alpha)
    # rdm2_ab[p, q, r, s] = <mps| a^dag_{p,alpha} a^dag_{q,beta}  a_{r,beta}  a_{s,alpha} |mps>   (alpha-beta)
    # rdm2_bb[p, q, r, s] = <mps| a^dag_{p,beta}  a^dag_{q,beta}  a_{r,beta}  a_{s,beta}  |mps>   (beta-beta)

    # clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # orbital rotations: identity, and a random unitary rotation 
    Us = []
    Us.append(np.eye(4))
    for _ in range(2):
        U = unitary_group.rvs(dim=norb)
        Us.append(U)

    # single-double excitation operators on Na, Nb-sector
    cre_a, des_a = ffsim.cre_a, ffsim.des_a
    cre_b, des_b = ffsim.cre_b, ffsim.des_b

    op1_a = np.empty((norb, norb), dtype=object)
    op1_b = np.empty((norb, norb), dtype=object)
    for p in range(norb):
        for q in range(norb):
            op1_a[p, q] = ffsim.linear_operator(
                ffsim.FermionOperator({(cre_a(p), des_a(q)): 1}), norb, nelec)
            op1_b[p, q] = ffsim.linear_operator(
                ffsim.FermionOperator({(cre_b(p), des_b(q)): 1}), norb, nelec)

    op2_aa = np.empty((norb, norb, norb, norb), dtype=object)
    op2_ab = np.empty((norb, norb, norb, norb), dtype=object)
    op2_bb = np.empty((norb, norb, norb, norb), dtype=object)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    op2_aa[p, q, r, s] = ffsim.linear_operator(
                        ffsim.FermionOperator({(cre_a(p), cre_a(q), des_a(r), des_a(s)): 1}), norb, nelec)
                    op2_ab[p, q, r, s] = ffsim.linear_operator(
                        ffsim.FermionOperator({(cre_a(p), cre_b(q), des_b(r), des_a(s)): 1}), norb, nelec)
                    op2_bb[p, q, r, s] = ffsim.linear_operator(
                        ffsim.FermionOperator({(cre_b(p), cre_b(q), des_b(r), des_b(s)): 1}), norb, nelec)

    for U in Us:
        U_conj = np.conj(U)
        # PATH 1: efficient, orbital-space level way

        # rotate 1rdm
        rdm1_a_rotated = U_conj @ rdm1_a @ U.T
        rdm1_b_rotated = U_conj @ rdm1_b @ U.T

        # rotate 2rdm
        rdm2_aa_rotated = np.einsum('pi,qj,rk,sl,ijkl->pqrs', U_conj, U_conj, U, U, rdm2_aa)
        rdm2_ab_rotated = np.einsum('pi,qj,rk,sl,ijkl->pqrs', U_conj, U_conj, U, U, rdm2_ab)
        rdm2_bb_rotated = np.einsum('pi,qj,rk,sl,ijkl->pqrs', U_conj, U_conj, U, U, rdm2_bb)

        # PATH 2: inefficient, Fock-space level way

        # rotated psi as we do in cost functions!
        psi_rotated = ffsim.apply_orbital_rotation(psi, U, norb, nelec)

        # manually compute the 1rdm and 2rdm of psi_rotated, using the precomputed operators
        # use block2 conventions
        rdm1_a_manual = np.array([[psi_rotated.conj() @ (op1_a[p, q] @ psi_rotated)
                                    for q in range(norb)] for p in range(norb)])
        rdm1_b_manual = np.array([[psi_rotated.conj() @ (op1_b[p, q] @ psi_rotated)
                                    for q in range(norb)] for p in range(norb)])
        rdm2_aa_manual = np.array([[[[psi_rotated.conj() @ (op2_aa[p, q, r, s] @ psi_rotated)
                                    for s in range(norb)] for r in range(norb)]
                                    for q in range(norb)] for p in range(norb)])
        rdm2_ab_manual = np.array([[[[psi_rotated.conj() @ (op2_ab[p, q, r, s] @ psi_rotated)
                                    for s in range(norb)] for r in range(norb)]
                                    for q in range(norb)] for p in range(norb)])
        rdm2_bb_manual = np.array([[[[psi_rotated.conj() @ (op2_bb[p, q, r, s] @ psi_rotated)
                                    for s in range(norb)] for r in range(norb)]
                                    for q in range(norb)] for p in range(norb)])

        # COMPARE PATHS
        # compare all 5 spin-resolved matrix elements and check equality within some tolerance
        tol = 1e-14
        tag = "identity" if np.allclose(U, np.eye(norb)) else "random"
        checks = [
            (rdm1_a_rotated, rdm1_a_manual),
            (rdm1_b_rotated, rdm1_b_manual),
            (rdm2_aa_rotated, rdm2_aa_manual),
            (rdm2_ab_rotated, rdm2_ab_manual),
            (rdm2_bb_rotated, rdm2_bb_manual),
        ]
        for (a, b) in checks:
            diff = np.max(np.abs(a - b))
            assert diff < tol