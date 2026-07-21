"""
Functions for managing cluster number operators.
- build_one_orb_num_operators and build_two_orb_num_operators: adapts optimize_symmetries.parities to numbers
- number_matrix_to_operators: adapts optimize_symmetries.parity_matrix_to_quasisymmetries
- number_and_parity_symmetry_sectors: builds sectors for given cluster number operators and cluster parity operators
For usage examples, see notebooks cluster_numbers_and_parities.ipynb, cluster_numbers_search.ipynb, and test_cluster_number_operators.py
"""

import numpy as np
import ffsim
import scipy
from scipy.sparse.linalg import LinearOperator
from math import comb
from functools import cache
from scipy.special import factorial
from collections.abc import Callable
from src.orbital_rotation import params_to_U

@cache
def build_one_orb_num_operators(norb, nelec):
    """Returns list of single-orbital occupation number operators (alpha + beta)"""
    orb_number_operators = []
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
        n = n_alpha + n_beta
        orb_number_operators.append(ffsim.linear_operator(n, norb, nelec))
    return orb_number_operators

def build_two_orb_num_operators(norb, nelec):
    """Returns list of two-orbital occupation number operators (alpha + beta)."""
    one_orb_num_operators = build_one_orb_num_operators(norb, nelec)
    two_orb_number_operators = []
    for i in range(norb):
        for j in range(i+1, norb):
            two_orb_number_operators.append(one_orb_num_operators[i] + one_orb_num_operators[j])
    return two_orb_number_operators

def integers_to_phases_polynomial(N):
    """
    Computes the coefficients [a_0, a_1, ..., a_N] for the polynomial
    P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_N*x^N 
    that maps integers n=0..N to exp(i * n * 2 * pi / (N+1) ) (unit semicircle).
    """
    omega = np.exp(1j * 2 * np.pi / (N+1))
    final_poly = [0j] * (N + 1)
    
    # FIX: Initialize as 1.0 (or 1 + 0j), not the imaginary unit (1j)
    falling_fact = [1 + 0j] 
    
    for k in range(N + 1):
        scalar = ((omega - 1)**k) / factorial(k)
        for i, c in enumerate(falling_fact):
            final_poly[i] += c * scalar
        
        if k < N:
            next_fact = [0j] * (len(falling_fact) + 1)
            for i, c in enumerate(falling_fact):
                next_fact[i] -= c * k       
                next_fact[i+1] += c         
            falling_fact = next_fact
            
    return final_poly

def from_num_operator_to_expnum_operator(num_operator, max_num_eval):
    """Returns LinearOperator exp(i * pi * num_operator / (max_num_eval + 1)), built efficiently"""
    dim = num_operator.shape[0]
    zero_op = LinearOperator((dim, dim), matvec=lambda v: np.zeros_like(v))
    coeffs = integers_to_phases_polynomial(max_num_eval)
    summands = [coeffs[n] * (num_operator ** n) for n in range(max_num_eval+1)]
    return sum(summands, start=zero_op)

def number_matrix_to_operators(cluster_number_matrix: np.ndarray,
                                     norb,
                                     nelec,
                                     expnum=False):
    """Returns a list of cluster number operators. The orbitals of the i th operator correspond to the 1's in the ith row of the binary cluster_number_matrix."""
    if len(cluster_number_matrix) == 0:
        return([])
    # Probably this won't be a bottleneck, but may want to avoid multiple 
    # one_orb_num_operators calls (see build_two_orb_num_operators)
    if cluster_number_matrix.shape[1] != norb:
        raise ValueError("shape[1] of cluster_number_matrix must equal norb")
    
    # get one-orbital number operators
    one_orb_num_operators = build_one_orb_num_operators(norb, nelec)

    # add those to cluster number operators
    operators = [] # will contain the cluster number/quasisymmetry operators
    dim = scipy.special.comb(norb, nelec[0], exact=True) * scipy.special.comb(norb, nelec[1], exact=True)  # Hilbert space dim
    zero_op = LinearOperator((dim, dim), matvec=lambda v: np.zeros_like(v))
    for i in range(cluster_number_matrix.shape[0]):
        summands = [one_orb_num_operators[j] for j in range(norb) if cluster_number_matrix[i][j] == 1]
        num_operator = sum(summands, start=zero_op)
        if expnum == False:
            operators.append(num_operator)
        else:
            max_num_eval = 2 * sum(cluster_number_matrix[i]) # max cluster number eval = 2 * number of orbitals in the cluster
            operators.append(from_num_operator_to_expnum_operator(num_operator, int(max_num_eval)))
    return(operators)

def number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec):
    """Returns a dictionary sectors with key = symmetry label
    (couple of tuples of evals; one tuple for cluster numbers, one for cluster parities),
    value = list of integer indices of determinants spanning the sector with that symmetry label.
    Aleksey's convention: 0 for even parity (e.g. 0 particles), 1 for odd parity (e.g. 1 particle)."""
    # input shape checks
    if len(cluster_number_matrix) > 0:
        if cluster_number_matrix.shape[1] != norb:
            raise ValueError("cluster_number_matrix must have shape[1] = norb")
        cluster_number_matrix_to_int = []
        for cluster in cluster_number_matrix:
            # convert clusters from binary to integers; can't do directly due to order
            cluster_int = 0
            for i, bit in enumerate(cluster):
                if bit:  # If the bit is 1
                    cluster_int |= (1 << i)  # Set the i-th bit to 1
            cluster_number_matrix_to_int.append(cluster_int)
    if len(cluster_parity_matrix) > 0:
        if cluster_parity_matrix.shape[1] != norb:
            raise ValueError("cluster_parity_matrix must have shape[1] = norb")
        # same as above
        cluster_parity_matrix_to_int = []
        for cluster in cluster_parity_matrix:
            cluster_int = 0
            for i, bit in enumerate(cluster):
                if bit:  # If the bit is 1
                    cluster_int |= (1 << i)  # Set the i-th bit to 1
            cluster_parity_matrix_to_int.append(cluster_int)

    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    alpha_indices, beta_indices = ffsim.addresses_to_strings(
    range(dim), norb, nelec,
    bitstring_type=ffsim.BitstringType.INT,
    concatenate=False
    ) # integers corresponding to FLIPPED alpha/beta bitstrings of basis determinants
    sectors = {}  
    for i in range(dim):
        if len(cluster_number_matrix) == 0:
            sector_label_num = ()
        else:
            sector_label_num = tuple(bin(cluster_mask & alpha_indices[i]).count('1')  + bin(cluster_mask & beta_indices[i]).count('1')  for cluster_mask in cluster_number_matrix_to_int)
        if len(cluster_parity_matrix) == 0:
            sector_label_par = ()
        else:
            sector_label_par = tuple((bin(cluster_mask & alpha_indices[i]).count('1')  + bin(cluster_mask & beta_indices[i]).count('1') ) % 2 for cluster_mask in cluster_parity_matrix_to_int)
        sector_label = (sector_label_num, sector_label_par)
        sectors.setdefault(sector_label, []).append(i)

    return sectors

################################################################
# Function above are used in cluster_numbers_and_parities.ipynb,
# functions below in cluster_numbers_scalable_search.ipynb
################################################################


def get_cluster_indices(cluster_matrix, norb, with_ghost=True):
    """Convert the binary cluster_matrix into a list of orbital-index arrays,
    one per cluster, with default option to add back the "ghost" cluster of uncovered orbitals.
    Precompute this once (it only depends on cluster_matrix, not on U)."""
    if cluster_matrix.size == 0:
        if with_ghost:
            return [np.arange(norb)]
        else:
            return []
    clusters = [np.where(row)[0] for row in cluster_matrix]
    if with_ghost:
        covered = np.any(cluster_matrix, axis=0)
        ghost = np.where(~covered)[0]
        if ghost.size > 0:
            clusters.append(ghost)
    return clusters

def build_loc_number_evaluator(D, Gamma, cluster_matrix=np.array([])) -> Callable:
    """
    Constructs a local particle number expectation value evaluator for a given cluster configuration.
    Only uses 1- and 2-rdm -> scales O(norb^5), with norb = number of orbitals.

    Args:
        D (ndarray): The spin-summed 1-reduced density matrix (1-RDM) of an underlying state psi.
        Gamma (ndarray): The spin-summed 2-reduced density matrix (2-RDM) of psi.
        cluster_matrix (ndarray): A binary matrix/list defining the orbital 
            clusters. Defaults to `np.array([])`, which groups all orbitals together into a single cluster.

    Returns:
        Callable: A function `loc_number_evaluator(U)` that takes:
            - U (ndarray): A norb x norb orbital unitary matrix.
            
            And returns:
            - couple (ndarray, ndarray): The expectation values of $n_p$ and $n_p n_q$ (for orbitals $p, q$ 
            belonging to the same cluster) evaluated on the transformed state 
            U^{otimes N} @ psi.
    """
    norb = D.shape[0]

    # permute once: Gamma_tilde[k,n,l,m] = Gamma[k,l,m,n]
    Gamma_tilde = Gamma.transpose(0, 3, 1, 2).reshape(norb * norb, norb * norb)

    # Precomputed once: list of orbital-index arrays, one per cluster
    # (including the ghost cluster of orbitals not covered by cluster_matrix).
    clusters = get_cluster_indices(cluster_matrix, norb)

    # For each cluster, precompute the (p, q) index pairs with p <= q
    # (upper triangle, including diagonal), as absolute orbital indices.
    # This is done once, not per-call.
    cluster_pairs = []
    for idx in clusters:
        k = idx.size
        tp, tq = np.triu_indices(k)  # local indices within this cluster
        cluster_pairs.append((idx[tp], idx[tq]))  # absolute orbital indices
    # cluster_pairs is a list of tuples, one tuple per cluster. Each tuple contains two arrays, P and Q: the first array contains the absolute orbital indices p for the upper triangle of the cluster, and the second array contains the corresponding absolute orbital indices q.

    def loc_number_evaluator(U):
        Uc = U.conj()

        # n1(U)_p = sum_kl U*[p,k] U[p,l] D[k,l]   -- unrestricted, needed for all p
        n1 = np.einsum('pk,pl,kl->p', Uc, U, D, optimize=True).real # discard machine precision imaginary parts in case of U complex

        # Step 1: contract k -> new axis p  (matmul, O(norb^5)), all p needed
        #   T[p, n, l, m] = sum_k U*[p,k] Gamma_tilde[k,n,l,m]
        T = (Uc @ Gamma_tilde.reshape(norb, norb * norb * norb)).reshape(
            norb, norb, norb, norb
        )

        # Step 2: contract n, batched over p  (O(norb^4)), all p needed
        #   M[p, l, m] = sum_n T[p,n,l,m] U[p,n]
        M = np.einsum('pnlm,pn->plm', T, U, optimize=True)

        # Step 3 (restricted + symmetry-reduced): for each cluster, only
        # compute N[p,q] for p, q both in that cluster, and only for p <= q
        # (using N_pq = N_qp), then mirror. Zero outside clusters.
        N = np.zeros((norb, norb), dtype=M.dtype)
        for P, Q in cluster_pairs:
            if P.size == 0:
                continue
            # Gather only the rows/entries needed for these p<=q pairs.
            # M[P] reuses already-computed M (no recomputation), just indexing.
            M_pairs = M[P]     # (npairs, norb, norb): M[p,:,:] for each pair
            Uc_pairs = Uc[Q]   # (npairs, norb): U*[q,:] for each pair
            U_pairs = U[Q]     # (npairs, norb): U[q,:] for each pair

            # N_vals[n] = sum_lm M[p,l,m] * U*[q,l] * U[q,m]  for pair n=(p,q)
            N_vals = np.einsum('nlm,nl,nm->n', M_pairs, Uc_pairs, U_pairs,
                                optimize=True)

            N[P, Q] = N_vals
            N[Q, P] = N_vals
            # for p == q this just overwrites the same (real) entry harmlessly

        n2 = N.real + np.diag(n1) # discard machine precision imaginary parts in case of U complex
        return n1, n2

    return loc_number_evaluator

def number_variance_cost(D, Gamma, cluster_matrix) -> Callable:
    """Compare with optimize_symmetries.variance_cost_general.
    Cost function measuring summed variances of cluster number operators for orbital-rotated reference state.
    Only uses 1- and 2-rdm -> scales O(norb^5), with norb = number of orbitals.
        Args:
            D (ndarray): The spin-summed 1-reduced density matrix (1-RDM) of an underlying state psi.
            Gamma (ndarray): The spin-summed 2-reduced density matrix (2-RDM) of psi.
            cluster_matrix (ndarray): A binary matrix/list defining the orbital clusters.

        Returns:
            Callable: A function `f(x)` that takes:
                - x (ndarray): 1D array parameters of upper-triangle of norb x norb antisymmetric matrix.
                
                And returns:
                - Sum of variances of the number operators specified by cluster_matrix relative to the transformed state 
                U^{otimes N} @ psi.
    """
    norb = D.shape[0]
    loc_number_evaluator = build_loc_number_evaluator(D,Gamma,cluster_matrix=cluster_matrix)
    cluster_indices = get_cluster_indices(cluster_matrix, norb)
    def f(x: np.ndarray) -> float:
        U = params_to_U(x, norb)

        # efficiently get the one- and two-orbital number expectation values
        # of ffsim.apply_orbital_rotation(reference_state, U, ...)
        n1, n2 = loc_number_evaluator(U)
        total_var = 0
        for cluster in cluster_indices:
            expected_n = np.sum(n1[cluster])
            expected_n_squared = np.sum(n2[np.ix_(cluster, cluster)])
            var = expected_n_squared - (expected_n ** 2)
            total_var += var
        return total_var
    return f

def number_eval_eq_cost(D, Gamma, cluster_matrix, evals: list) -> Callable:
    """See optimize_symmetries.eval_eq_cost for the math idea. See number_variance_cost for usage.
    """
    if len(cluster_matrix) != len(evals):
        raise ValueError("len(cluster_matrix) must match len(evals)")
    norb = D.shape[0]
    loc_number_evaluator = build_loc_number_evaluator(D,Gamma,cluster_matrix=cluster_matrix)
    cluster_indices = get_cluster_indices(cluster_matrix, norb)
    # complete with the eigenvalue for the ghost cluster number
    num_elec = round(np.trace(D))
    ghost_eval = round(num_elec - sum(evals))
    evals_with_ghost = evals.copy()
    evals_with_ghost.append(ghost_eval)

    def f(x):
        U = params_to_U(x, norb)

        # efficiently get the one- and two-orbital number expectation values
        # of ffsim.apply_orbital_rotation(reference_state, U, ...)
        n1, n2 = loc_number_evaluator(U)
        total_score = 0
        for i in range(len(cluster_indices)):
            cluster = cluster_indices[i]
            eval = evals_with_ghost[i]
            expected_n = np.sum(n1[cluster])
            expected_n_squared = np.sum(n2[np.ix_(cluster, cluster)])
            term = expected_n_squared - 2 * eval * expected_n + eval ** 2
            total_score += term
        return total_score
    return f