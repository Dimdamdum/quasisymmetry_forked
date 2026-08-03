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
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Ensure JAX uses float64 for high-precision optimization
jax.config.update("jax_enable_x64", True)

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


def get_cluster_indices(cluster_matrix, norb, with_ghost=False):
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

def build_loc_number_evaluator(D, Gamma, cluster_matrix=np.array([]), with_ghost=False) -> Callable:
    """
    Constructs a local particle number expectation value evaluator for a given cluster configuration.
    Only uses 1- and 2-rdm -> scales O(norb^5), with norb = number of orbitals.

    Args:
        D (ndarray): The spin-summed 1-reduced density matrix (1-RDM) of an underlying state psi.
        Gamma (ndarray): The spin-summed 2-reduced density matrix (2-RDM) of psi.
        cluster_matrix (ndarray): A binary matrix/list defining the orbital 
            clusters. E.g., a row [0, 1, 0, 1] identifies the cluster of orbitals 1 and 3 (out of a total of 4).
            Defaults to `np.array([])`, which groups all orbitals together into a single cluster.
        with_ghost (bool): set to True to add a clusters with all orbitals corresponding to zero columns of cluster_matrix.
            Defaults to False.

    Returns:
        Callable: A function `loc_number_evaluator(U)` that takes:
            - U (ndarray): A norb x norb orbital unitary matrix.
            
            And returns:
            - couple (ndarray, ndarray): The expectation values of the operators $n_p$ and $n_p n_q$ (for orbitals $p, q$ 
            belonging to the same cluster) evaluated on the transformed state 
            U^{otimes N} @ psi.
    """
    norb = D.shape[0]

    # permute once: Gamma_tilde[k,n,l,m] = Gamma[k,l,m,n]
    Gamma_tilde = Gamma.transpose(0, 3, 1, 2).reshape(norb * norb, norb * norb)

    # Precomputed once: list of orbital-index arrays, one per cluster
    # (possibly including the ghost cluster of orbitals not covered by cluster_matrix).
    clusters = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)

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
        n1 = jnp.einsum('pk,pl,kl->p', Uc, U, D, optimize=True).real # discard machine precision imaginary parts in case of U complex

        # Step 1: contract k -> new axis p  (matmul, O(norb^5)), all p needed
        #   T[p, n, l, m] = sum_k U*[p,k] Gamma_tilde[k,n,l,m]
        T = (Uc @ Gamma_tilde.reshape(norb, norb * norb * norb)).reshape(
            norb, norb, norb, norb
        )

        # Step 2: contract n, batched over p  (O(norb^4)), all p needed
        #   M[p, l, m] = sum_n T[p,n,l,m] U[p,n]
        M = jnp.einsum('pnlm,pn->plm', T, U, optimize=True)

        # Step 3 (restricted + symmetry-reduced): for each cluster, only
        # compute N[p,q] for p, q both in that cluster, and only for p <= q
        # (using N_pq = N_qp), then mirror. Zero outside clusters.
        N = jnp.zeros((norb, norb), dtype=M.dtype)

        if not with_ghost:
            nonzero_cols = jnp.array((cluster_matrix != 0).any(axis=0).astype(int))

        for P, Q in cluster_pairs:
            if P.size == 0:
                continue
            # Gather only the rows/entries needed for these p<=q pairs.
            # M[P] reuses already-computed M (no recomputation), just indexing.
            M_pairs = M[P]     # (npairs, norb, norb): M[p,:,:] for each pair
            Uc_pairs = Uc[Q]   # (npairs, norb): U*[q,:] for each pair
            U_pairs = U[Q]     # (npairs, norb): U[q,:] for each pair

            # N_vals[n] = sum_lm M[p,l,m] * U*[q,l] * U[q,m]  for pair n=(p,q)
            N_vals = jnp.einsum('nlm,nl,nm->n', M_pairs, Uc_pairs, U_pairs,
                                optimize=True)

            N = N.at[P, Q].set(N_vals)
            N = N.at[Q, P].set(N_vals)
            # for p == q this just overwrites the same (real) entry harmlessly

        if with_ghost:
            n2 = N.real + jnp.diag(n1) # discard machine precision imaginary parts in case of U complex
        else:
            n2 = N.real + nonzero_cols * jnp.diag(n1)
        
        return n1, n2 # still returning full diagonal n1 even if not with_ghost - it is cheap

    return loc_number_evaluator



def params_to_U_jax(x, norb):
    """Convert parameters x to an orthogonal/unitary matrix U."""
    # Build upper triangle skew-symmetric matrix A
    A = jnp.zeros((norb, norb))
    triu_indices = jnp.triu_indices(norb, k=1)
    A = A.at[triu_indices].set(x)
    A = A - A.T  # Antisymmetric matrix
    
    # Exponentiate to get the unitary/orthogonal rotation matrix U
    return jax.scipy.linalg.expm(A)

def number_variance_cost(D, Gamma, cluster_matrix, with_ghost=False, var_exponent=1) -> Callable:
    """Compare with optimize_symmetries.variance_cost_general.
    Cost function measuring summed variances of cluster number operators for orbital-rotated reference state.
    Only uses 1- and 2-rdm -> scales O(norb^5), with norb = number of orbitals.
        Args:
            D (ndarray): The spin-summed 1-reduced density matrix (1-RDM) of an underlying state psi.
            Gamma (ndarray): The spin-summed 2-reduced density matrix (2-RDM) of psi.
            cluster_matrix (ndarray): A binary matrix/list defining the orbital clusters.
            with_ghost (bool): set to True to add a cluster with all orbitals that are not in cluster_matrix. Defaults to False.
            var_exponent (float): exponent to be put on the variances before summing them. Defaults to 1 (pure variances).
        Returns:
            Callable: A function `f(x)` that takes:
                - x (ndarray): 1D array parameters of upper-triangle of norb x norb antisymmetric matrix.
                
                And returns:
                - Sum of variances ** var_exponent of the number operators specified by cluster_matrix relative to the transformed state 
                U^{otimes N} @ psi.
    """
    norb = D.shape[0]

    # convert input to JAX
    D_jax = jnp.array(D)
    Gamma_jax = jnp.array(Gamma)

    loc_number_evaluator = build_loc_number_evaluator(D_jax, Gamma_jax,cluster_matrix=cluster_matrix, with_ghost=with_ghost)
    cluster_indices = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)
    def f(x: np.ndarray) -> float:
        U = params_to_U_jax(x, norb)

        # efficiently get the one- and two-orbital number expectation values
        # of ffsim.apply_orbital_rotation(reference_state, U, ...)
        n1, n2 = loc_number_evaluator(U)
        total_var = 0.
    
        for cluster in cluster_indices:
            expected_n = jnp.sum(n1[cluster])
            expected_n_squared = jnp.sum(n2[jnp.ix_(cluster, cluster)])
            var = expected_n_squared - (expected_n ** 2)
            if var_exponent != 1:
                total_var += var ** var_exponent
            else:
                total_var += var
        return total_var
    return f

def number_eval_eq_cost(D, Gamma, cluster_matrix, evals: list, with_ghost=False) -> Callable:
    """See optimize_symmetries.eval_eq_cost for the math idea. See number_variance_cost for usage.
    """
    if len(cluster_matrix) != len(evals):
        raise ValueError("len(cluster_matrix) must match len(evals)")
    norb = D.shape[0]
    
    # convert input to JAX
    D_jax = jnp.array(D)
    Gamma_jax = jnp.array(Gamma)

    loc_number_evaluator = build_loc_number_evaluator(D_jax, Gamma_jax,cluster_matrix=cluster_matrix, with_ghost=with_ghost)
    cluster_indices = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)
    evals_processed = evals.copy()

    if with_ghost:
        covered = np.any(cluster_matrix, axis=0)
        ghost = np.where(~covered)[0]
        if ghost.size > 0:
        # complete with the eigenvalue for the ghost cluster number
            num_elec = round(np.trace(D))
            ghost_eval = round(num_elec - sum(evals))
            evals_processed.append(ghost_eval)

    def f(x):
        U = params_to_U_jax(x, norb)

        # efficiently get the one- and two-orbital number expectation values
        # of ffsim.apply_orbital_rotation(reference_state, U, ...)
        n1, n2 = loc_number_evaluator(U)
        total_score = 0
        for i in range(len(cluster_indices)):
            cluster = cluster_indices[i]
            eval = evals_processed[i]
            expected_n = jnp.sum(n1[cluster])
            expected_n_squared = jnp.sum(n2[jnp.ix_(cluster, cluster)])
            term = expected_n_squared - 2 * eval * expected_n + eval ** 2
            total_score += term
        return total_score
    return f

def extremality_cost(D, cluster_matrix, with_ghost=False) -> Callable: # only needs 1rdm D
    norb = D.shape[0]

    # convert input to JAX
    D_jax = jnp.array(D)

    cluster_indices = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)

    def f(x: np.ndarray) -> float:

        U = params_to_U_jax(x, norb)
        Uc = U.conj()
    
        # n1(U)_p = sum_kl U*[p,k] U[p,l] D[k,l]   -- unrestricted, needed for all p
        n1 = jnp.einsum('pk,pl,kl->p', Uc, U, D, optimize=True).real # discard machine precision imaginary parts in case of U complex

        total_cost = 0.
        
        for cluster in cluster_indices:
            cluster_size = len(cluster)
            expected_n = jnp.sum(n1[cluster])
            total_cost += expected_n * (2 * cluster_size - expected_n)
        return total_cost
    return f

# function that returns <[H, N_C]^2> given 1elec orbitals h, 2elec orbitals g (chemist's notation), 1-, 2-, 3-, 4-rdms (spin-summed), cluster of orbitals C as a list.

"""
Key formula (copy-paste to markdown):

$$
\text{squared commutator exp value}(h,g_{chem},D^{(1)},D^{(2)},D^{(3)},D^{(4)},C) = A + B + B^* + C
$$

with $g_{pqrs} = g_{chem, prqs}$ and

$$
A=
\sum_{t,t'\in C}\sum_{p,q}\sum_{p',q'}
h_{pq}\,h_{p'q'}\,
(-\delta_{pt}+\delta_{qt})\,(-\delta_{p't'}+\delta_{q't'})\,
\Bigl(
D^{(2)}_{p\,p'\,q'\,q}
+\delta_{q p'}\,D^{(1)}_{p\,q'}
\Bigr)
$$

$$
B = \frac12
\sum_{t,t'\in C}\sum_{p,q}\sum_{p',q',r',s'}
h_{pq}\,g_{p'q'r's'}\,
(-\delta_{pt}+\delta_{qt})\,
(-\delta_{p't'}-\delta_{q't'}+\delta_{s't'}+\delta_{r't'})\,
\Bigl[
D^{(3)}_{p\,p'\,q'\,s'\,r'\,q}
+\delta_{q p'}\,D^{(2)}_{p\,q'\,s'\,r'}
+\delta_{q q'}\,D^{(2)}_{p\,p'\,r'\,s'}
\Bigr]
$$

$$
C =
\frac14
\sum_{t,t'\in C}\sum_{p,q,r,s}\sum_{p',q',r',s'}
g_{pqrs}\,g_{p'q'r's'}\,
(-\delta_{pt}-\delta_{qt}+\delta_{st}+\delta_{rt})\,
(-\delta_{p't'}-\delta_{q't'}+\delta_{s't'}+\delta_{r't'})\,
\Bigl[
D^{(4)}_{p\,q\,p'\,q'\,s'\,r'\,s\,r}


+\delta_{s p'}\,D^{(3)}_{p\,q\,q'\,s'\,r'\,r}
+\delta_{s q'}\,D^{(3)}_{p\,q\,p'\,r'\,s'\,r}
+\delta_{r p'}\,D^{(3)}_{p\,q\,q'\,s'\,s\,r'}
+\delta_{r q'}\,D^{(3)}_{p\,q\,p'\,r'\,s\,s'}

+\delta_{r p'}\delta_{s q'}\,D^{(2)}_{p\,q\,s'\,r'}
+\delta_{r q'}\delta_{s p'}\,D^{(2)}_{p\,q\,r'\,s'}
\Bigr]
$$

"""

def squared_commutator_exp_value(h, g_chem, D1, D2, D3, D4, C, return_terms=False):
    """
    Implements

        F = A + B + B* + C

    with the tensor contractions exactly as in the corrected LaTeX.

    Index conventions used literally as written in the formula:
      A:
        D2[p, pp, qp, q]
        D1[p, qp]

      B:
        D3[p, pp, qp, sp, rp, q]
        D2[p, qp, sp, rp]
        D2[p, pp, rp, sp]

      C (corrected version -- note the r<->s, r'<->s' swap on every D and
      every delta relative to the original/mistaken formula; the g tensors
      are NOT swapped):
        D4[p, q, pp, qp, sp, rp, s, r]
        D3[p, q, qp, sp, rp, r]
        D3[p, q, pp, rp, sp, r]
        D3[p, q, qp, sp, s, rp]
        D3[p, q, pp, rp, s, sp]
        D2[p, q, sp, rp]
        D2[p, q, rp, sp]

    Key simplification: since the (-delta+delta)-type factors depend on
    (p, q, t) resp. (p', q', r', s', t') separately from everything else,
    the sums over t, t' in C factor and can be pre-summed into simple
    "indicator" combinations:

        S1[p, q]       = sum_{t in C} (-delta(p,t) + delta(q,t))
                        = -I[p] + I[q]

        S2[p, q, r, s] = sum_{t in C} (-delta(p,t) - delta(q,t)
                                        + delta(s,t) + delta(r,t))
                        = -I[p] - I[q] + I[s] + I[r]

    where I is the indicator vector of the index set C. This turns the
    (t, t')-nested sums into ordinary tensor contractions (einsum).
    """
    h = jnp.asarray(h)
    g = jnp.asarray(g_chem).transpose(0, 2, 1, 3)
    D1 = jnp.asarray(D1)
    D2 = jnp.asarray(D2)
    D3 = jnp.asarray(D3)
    D4 = jnp.asarray(D4)

    norb = h.shape[0]
    
    # Determine safe working dtype (JAX handles complex differentiation cleanly)
    dtype = jnp.result_type(h, g, D1, D2, D3, D4, jnp.complex128)

    # JAX-compatible indicator vector (avoiding in-place mutation I[idx] = 1.0)
    # Using jn.zeros and jnp.at/scatter or index_update equivalents (at[].set())
    I = jnp.zeros(norb, dtype=dtype)
    if len(C) > 0:
        I = I.at[jnp.array(C, dtype=int)].set(1.0)

    # -----------------------
    # Aggregated delta-sums
    # -----------------------
    S1 = -I[:, None] + I[None, :]

    S2 = (-I[:, None, None, None] - I[None, :, None, None]
          + I[None, None, None, :] + I[None, None, :, None])

    M = h * S1
    Gt = g * S2

    # -----------------------
    # A term
    # -----------------------
    A1 = jnp.einsum('pq,rs,prsq->', M, M, D2, optimize=True)
    A2 = jnp.einsum('pq,qr,pr->', M, M, D1, optimize=True)
    A = A1 + A2

    # -----------------------
    # B term
    # -----------------------
    B1 = jnp.einsum('pq,ijkl,pijlkq->', M, Gt, D3, optimize=True)
    B2 = jnp.einsum('pq,qjkl,pjlk->', M, Gt, D2, optimize=True)
    B3 = jnp.einsum('pq,iqkl,pikl->', M, Gt, D2, optimize=True)
    B = 0.5 * (B1 + B2 + B3)

    # -----------------------
    # C term (corrected)
    # -----------------------
    C1 = jnp.einsum('pqrs,ijkl,pqijlksr->', Gt, Gt, D4, optimize=True)
    C2 = jnp.einsum('pqrs,sjkl,pqjlkr->', Gt, Gt, D3, optimize=True)
    C3 = jnp.einsum('pqrs,iskl,pqiklr->', Gt, Gt, D3, optimize=True)
    C4 = jnp.einsum('pqrs,rjkl,pqjlsk->', Gt, Gt, D3, optimize=True)
    C5 = jnp.einsum('pqrs,irkl,pqiksl->', Gt, Gt, D3, optimize=True)
    C6 = jnp.einsum('pqrs,rskl,pqlk->', Gt, Gt, D2, optimize=True)
    C7 = jnp.einsum('pqrs,srkl,pqkl->', Gt, Gt, D2, optimize=True)
    CT = 0.25 * (C1 + C2 + C3 + C4 + C5 + C6 + C7)

    F = A + B + jnp.conj(B) + CT

    if return_terms:
        return A, B, CT
    else:
        return F

# same thing as above, but manual - needed for checks
def squared_commutator_exp_value_expected(h1_linop, h2_linop, fci_state, cluster, norb, nelec, return_terms=False):

    # fermionic operators (alpha and beta)
    cre_a, des_a = ffsim.cre_a, ffsim.des_a
    cre_b, des_b = ffsim.cre_b, ffsim.des_b

    # Construct the spin-summed number operator for the specified orbitals:
    # N = sum_{i in orbitals} (a^\dagger_{i,α} a_{i,α} + a^\dagger_{i,β} a_{i,β})
    number_op_terms = {}
    for i in cluster:
        number_op_terms[(cre_a(i), des_a(i))] = 1.0
        number_op_terms[(cre_b(i), des_b(i))] = 1.0

    num_operator = ffsim.FermionOperator(number_op_terms)
    num_linop = ffsim.linear_operator(num_operator, norb=norb, nelec=nelec)

    commutator1 = h1_linop @  num_linop - num_linop @ h1_linop
    commutator2 = h2_linop @  num_linop - num_linop @ h2_linop

    A = np.vdot(fci_state, commutator1 @ commutator1 @ fci_state)
    B = np.vdot(fci_state, commutator1 @ commutator2 @ fci_state)
    C = np.vdot(fci_state, commutator2 @ commutator2 @ fci_state)

    F = A + B + np.conjugate(B) + C
    if return_terms:
        return A, B, C
    else:
        return F

# commutator-based cost function, specific to cluster numbers
def number_commutator_cost_v1(h1e, g2e_full, rdm1, rdm2, rdm3, rdm4, cluster_matrix, with_ghost=False) -> Callable:
    """Builds cost function returning a commutator-based score of an orbital rotation. 
    Because the commutator of a 1- and 2-body Hamiltonian with n_p is also a 1- and 2-body 
    operator, only up to the 4th rdms are needed.

    Args:
        h1e: 1-electron integrals in MO basis
        g2e_full: 2-electron integrals in MO basis; chemist's notation   
        rdm1 (ndarray): The spin-summed 1-reduced density matrix (1-RDM) of an underlying state psi.
        rdm2 (ndarray): The spin-summed 2-reduced density matrix (2-RDM) of psi.
        rdm3 (ndarray): The spin-summed 3-reduced density matrix (3-RDM) of psi.
        rdm4 (ndarray): The spin-summed 4-reduced density matrix (4-RDM) of psi.
        cluster_matrix (ndarray): A binary matrix/list defining the orbital clusters.
        with_ghost (bool): set to True to add a cluster with all orbitals that are not in cluster_matrix. Defaults to False.

    Returns:
        Callable: A function `f(x)` that returns the quantity - sum_C <psi(U)|[H(U), N_C]^2|psi(U)>.
    """
    norb = h1e.shape[0]
    
    # Get clusters as lists of orbitals
    clusters = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)


    def f(x: jnp.ndarray) -> float:
        # Create orbital unitary
        U = jnp.array(params_to_U_jax(x, norb))
        U_conj = jnp.conjugate(U)

        # rotate rdms
        rdm1_rotated = U_conj @ rdm1 @ U.T
        rdm2_rotated = jnp.einsum('pi,qj,rk,sl,ijkl->pqrs', U_conj, U_conj, U, U, rdm2, optimize=True)
        rdm3_rotated = jnp.einsum('pi,qj,rk,sl,tm,un,ijklmn->pqrstu', U_conj, U_conj, U_conj, U, U, U, rdm3, optimize=True)
        rdm4_rotated = jnp.einsum('pi,qj,rk,sl,tm,un,vo,wa,ijklmnoa->pqrstuvw', U_conj, U_conj, U_conj, U_conj, U, U, U, U, rdm4, optimize=True)

        # rotate electronic integrals
        h1e_rotated = jnp.einsum('pr,qs,rs->pq', U, U_conj, h1e, optimize=True)
        g2e_full_rotated = jnp.einsum('pr,tu,qs,vz,rsuz->pqtv', U, U, U_conj, U_conj, g2e_full, optimize=True)

        # get score
        total_score = 0.0
        for cluster in clusters:
            total_score += squared_commutator_exp_value(h1e_rotated, g2e_full_rotated, rdm1_rotated, rdm2_rotated, rdm3_rotated, rdm4_rotated, cluster)
        return -total_score.real
        
    return f

######## Start of second implementation of number_commutator_cost with helpers ########
######## AI written, human tested against the previous implementations. ########

"""
Reformulation of `number_commutator_cost_v1`.

THE KEY IDEA
------------
For a fixed cluster C, `squared_commutator_exp_value` builds

    S1[p, q]       = -I_C[p] + I_C[q]
    S2[p, q, r, s] = -I_C[p] - I_C[q] + I_C[r] + I_C[s]

(I_C is the 0/1 indicator vector of the cluster) and then contracts
M = h * S1  and  Gt = g * S2  with the D-tensors. Every one of A, B, C is
built from EXACTLY TWO copies of "h or g weighted by S1 or S2" contracted
against a D-tensor. Since S1 and S2 are *linear* in I_C, and every term
uses exactly two such factors, F(C) = A(C)+B(C)+B*(C)+C(C) is a
**quadratic form in I_C**:

        F(C) = I_C^T K I_C

for a single norb x norb matrix K that depends only on h, g, D1..D4 --
NOT on the cluster. K is built ONCE (per orbital rotation), and every
cluster's score is then a cheap O(norb^2) bilinear-form evaluation
instead of a fresh O(norb^8)-class tensor contraction.

Mechanically: expand each product S1*S1, S1*S2 or S2*S2 into its
2x2=4, 2x4=8 or 4x4=16 sign-weighted terms of the form
(+/-1) * I[x] * I[y]. If x and y are literally the same dummy label (this
happens whenever an index is shared between the two S-factors, e.g. B2's
and C2-C7's "hinge" index), then I[x]*I[y] = I[x] (indicator idempotency,
I in {0,1}), which is folded into the diagonal of K. Otherwise the term
contributes an off-diagonal (or symmetric) entry K[x, y].

This has been validated against the original einsum implementation to
machine precision (see accompanying test); see docstring at the bottom
for the complexity comparison.
"""
from typing import Callable, Sequence

# --------------------------------------------------------------------------
# Generic helper: expand one bilinear (S x S) product into kernel updates
# --------------------------------------------------------------------------
def _add_bilinear(K: jnp.ndarray, operands: Sequence[jnp.ndarray], base_subs: str,
                   sites1, sites2, prefactor: float, dtype) -> jnp.ndarray:
    """
    operands / base_subs: the BARE tensors (h, g, D...) and the einsum
        subscript string (no '->', no output) that would fully contract
        them to a scalar if S1/S2 were absent.
    sites1, sites2: lists of (letter, sign) giving the attachment points of
        the two S-factors (S1 has 2 sites, S2 has 4 sites), using the same
        einsum-letter names as `base_subs`.
    prefactor: the term's overall scalar prefactor (1, 0.5, or 0.25).

    For every (l1, s1) in sites1 x (l2, s2) in sites2, adds
        coeff = s1 * s2 * prefactor
    either to the diagonal of K (if l1 == l2, since I_x*I_x = I_x for a
    0/1 indicator) or to the (l1, l2) block of K (off-diagonal, general
    quadratic term).
    """
    for (l1, s1) in sites1:
        for (l2, s2) in sites2:
            coeff = s1 * s2 * prefactor
            if l1 == l2:
                out = jnp.einsum(f'{base_subs}->{l1}', *operands, optimize=True)
                K = K + coeff * jnp.diag(out.astype(dtype))
            else:
                out = jnp.einsum(f'{base_subs}->{l1}{l2}', *operands, optimize=True)
                K = K + coeff * out.astype(dtype)
    return K


def _s1_sites(a, b):
    """S1_{ab} = -delta_{a,t} + delta_{b,t}  ->  sites (a, -1), (b, +1)."""
    return [(a, -1.0), (b, +1.0)]


def _s2_sites(a, b, c, d):
    """S2_{abcd} = -I_a - I_b + I_c + I_d  ->  sites (a,-1),(b,-1),(c,+1),(d,+1)."""
    return [(a, -1.0), (b, -1.0), (c, +1.0), (d, +1.0)]


def build_variance_kernel(h, g_chem, D1, D2, D3, D4) -> jnp.ndarray:
    """
    Build the norb x norb kernel K such that, for ANY cluster indicator I_C,

        squared_commutator_exp_value(h, g_chem, D1..D4, C) == I_C @ K @ I_C

    K depends only on h, g_chem, D1..D4 (i.e. only on the orbital basis /
    rotation) -- NOT on the cluster -- so it is computed once per call to
    f(x) and then reused for every cluster.
    """
    h = jnp.asarray(h)
    g = jnp.asarray(g_chem).transpose(0, 2, 1, 3)      # g_pqrs = g_chem[p, r, q, s]
    D1 = jnp.asarray(D1)
    D2 = jnp.asarray(D2)
    D3 = jnp.asarray(D3)
    D4 = jnp.asarray(D4)
    norb = h.shape[0]
    dtype = jnp.result_type(h, g, D1, D2, D3, D4, jnp.complex128)

    K = jnp.zeros((norb, norb), dtype=dtype)

    # ---- A = A1 + A2  (h,h with S1,S1) ----
    K = _add_bilinear(K, [h, h, D2], 'pq,rs,prsq',
                       _s1_sites('p', 'q'), _s1_sites('r', 's'), 1.0, dtype)
    K = _add_bilinear(K, [h, h, D1], 'pq,qr,pr',
                       _s1_sites('p', 'q'), _s1_sites('q', 'r'), 1.0, dtype)

    # ---- B = 0.5*(B1+B2+B3), plus its conjugate B* ----
    # (S1 has 2 sites, S2 has 4 sites -> 2x4 = 8 combinations per sub-term)
    KB = jnp.zeros((norb, norb), dtype=dtype)
    KB = _add_bilinear(KB, [h, g, D3], 'pq,ijkl,pijlkq',
                        _s1_sites('p', 'q'), _s2_sites('i', 'j', 'k', 'l'), 0.5, dtype)
    KB = _add_bilinear(KB, [h, g, D2], 'pq,qjkl,pjlk',
                        _s1_sites('p', 'q'), _s2_sites('q', 'j', 'k', 'l'), 0.5, dtype)
    KB = _add_bilinear(KB, [h, g, D2], 'pq,iqkl,pikl',
                        _s1_sites('p', 'q'), _s2_sites('i', 'q', 'k', 'l'), 0.5, dtype)
    K = K + KB + jnp.conj(KB)

    # ---- C = 0.25*(C1+...+C7)  (S2 has 4 sites -> 4x4 = 16 combinations each) ----
    K = _add_bilinear(K, [g, g, D4], 'pqrs,ijkl,pqijlksr',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('i', 'j', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D3], 'pqrs,sjkl,pqjlkr',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('s', 'j', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D3], 'pqrs,iskl,pqiklr',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('i', 's', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D3], 'pqrs,rjkl,pqjlsk',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('r', 'j', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D3], 'pqrs,irkl,pqiksl',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('i', 'r', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D2], 'pqrs,rskl,pqlk',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('r', 's', 'k', 'l'), 0.25, dtype)
    K = _add_bilinear(K, [g, g, D2], 'pqrs,srkl,pqkl',
                       _s2_sites('p', 'q', 'r', 's'), _s2_sites('s', 'r', 'k', 'l'), 0.25, dtype)

    return K


def number_commutator_cost_v2(h1e, g2e_full, rdm1, rdm2, rdm3, rdm4,
                                    cluster_matrix, with_ghost=False) -> Callable:
    """
    Drop-in, numerically-identical replacement for `number_commutator_cost_v1`
    that computes the norb x norb kernel ONCE per call to f(x) and scores
    every cluster with a cheap bilinear form instead of re-running the full
    A/B/C tensor contractions per cluster.

    See module docstring / accompanying complexity note for why this is
    exact and for the big-O comparison.
    """
    norb = h1e.shape[0]
    clusters = get_cluster_indices(cluster_matrix, norb, with_ghost=with_ghost)

    # Build the (n_clusters, norb) indicator matrix ONCE -- independent of x.
    I_mat = np.zeros((len(clusters), norb))
    for c_idx, cluster in enumerate(clusters):
        if len(cluster) > 0:
            I_mat[c_idx, np.asarray(cluster, dtype=int)] = 1.0
    I_mat = jnp.asarray(I_mat)

    def f(x: jnp.ndarray) -> float:
        U = jnp.array(params_to_U_jax(x, norb))
        U_conj = jnp.conjugate(U)

        # rotate rdms (unchanged, and already done ONCE per f(x) call, not per cluster)
        rdm1_rotated = U_conj @ rdm1 @ U.T
        rdm2_rotated = jnp.einsum('pi,qj,rk,sl,ijkl->pqrs', U_conj, U_conj, U, U, rdm2, optimize=True)
        rdm3_rotated = jnp.einsum('pi,qj,rk,sl,tm,un,ijklmn->pqrstu',
                                   U_conj, U_conj, U_conj, U, U, U, rdm3, optimize=True)
        rdm4_rotated = jnp.einsum('pi,qj,rk,sl,tm,un,vo,wa,ijklmnoa->pqrstuvw',
                                   U_conj, U_conj, U_conj, U_conj, U, U, U, U, rdm4, optimize=True)

        # rotate electronic integrals
        h1e_rotated = jnp.einsum('pr,qs,rs->pq', U, U_conj, h1e, optimize=True)
        g2e_full_rotated = jnp.einsum('pr,tu,qs,vz,rsuz->pqtv', U, U, U_conj, U_conj, g2e_full, optimize=True)

        # ---- the actual optimization: build K ONCE, score all clusters cheaply ----
        K = build_variance_kernel(h1e_rotated, g2e_full_rotated,
                                   rdm1_rotated, rdm2_rotated, rdm3_rotated, rdm4_rotated)
        # scores[c] = I_mat[c] @ K @ I_mat[c]   for every cluster at once, O(n_clusters * norb^2)
        scores = jnp.einsum('cp,pq,cq->c', I_mat.astype(K.dtype), K, I_mat.astype(K.dtype), optimize=True)
        total_score = jnp.sum(scores)

        return -total_score.real

    return f

######## End of efficient new implementation of number_commutator_cost with helpers ########