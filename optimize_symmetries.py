import argparse
import numpy as np
import time
import ffsim
import scipy
import scipy.optimize
import scipy.sparse.linalg
import pyscf
import pyscf.fci
import openfermion as of
import openfermionpyscf

from typing import Callable, Any, Union
from math import comb
from functools import cache, reduce

from chemistry import load_moldata, fcidump_data

from src.state_utils import get_cisd_gs, get_fci_state_openfermion
from src.bs import beam
import fcidump_openfermion


def commutator_cost(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray) -> Callable:
    def f(x: np.ndarray) -> float:
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


def variance_cost(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray) -> Callable:
    def f(x: np.ndarray) -> float:
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        total_var = 0
        for s in symmetries:
            total_var += 1 - ((rotated_state.T.conj() @ s @ rotated_state)**2).real
        return total_var
    return f


@cache
def parities(norb: int, nelec: tuple[int, int]) -> list[scipy.sparse.linalg.LinearOperator]:
    local_parities = []
    for i in range(norb):
        s_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): -2,
                (): 1
            } # FermionOperator for alpha spin  1 - a_{i} a^\dagger_{i}, AKA, local parity
        )
        s_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): -2,
                (): 1
            } # FermionOperator for beta spin  1 - b_{i} b^\dagger_{i}, AKA, local parity
        )
        s = s_alpha * s_beta
        local_parities.append(ffsim.linear_operator(s, norb, nelec))
    return local_parities


def parity_matrix_to_quasisymmetries(parity_matrix: np.ndarray,
                                     norb: int,
                                     nelec: tuple[int, int]) -> list[scipy.sparse.linalg.LinearOperator]:
    local_parities = parities(norb, nelec)
    if parity_matrix.shape[1] == norb:
        operators = []
        for i in range(parity_matrix.shape[0]): # rows
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


def x_to_rotation(x: np.ndarray, norb: int) -> np.ndarray:
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)


def get_fci(dumpdata: dict, flatten: bool = True) -> tuple[float, np.ndarray]:
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 10000 
    cisolver.conv_tol = 1e-10
    e_fci, fcivec = cisolver.kernel(
        dumpdata["H1"],
        dumpdata["H2"],
        dumpdata["NORB"],
        dumpdata["NELEC"],
        ecore=dumpdata["ECORE"],
    )
    if not cisolver.converged:
        import warnings
        warnings.warn(
            f"FCI didn't converge (conv_tol={cisolver.conv_tol}); using best available result.",
            RuntimeWarning,
            stacklevel=2,
        )
    if flatten:
        return e_fci, np.array(fcivec.reshape((-1,)), dtype="complex")
    else:
        return e_fci, fcivec


def expand_state(mol: of.MolecularData, ci: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
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


def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
    print(time.strftime("%a, %d %b %Y %H:%M:%S",
                        time.localtime()), end=" ")
    print("{0:4.6f}".format(intermediate_result.fun))


def comm_sq_exp_fast(sym_ops: list[of.QubitOperator], H: Any, state: np.ndarray, n_qubits: int, verbose: bool = False) -> Union[float, complex]:
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



# ─────────────────────────────────────────────────────────────────────────────
# New: arbitrary operator support via of_to_ffsim
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT ALREADY WORKS with any LinearOperator:
#   commutator_cost      — accepts list[LinearOperator], no restrictions
#   x_to_rotation        — pure math, no operator knowledge
#   callback             — pure math
#
# WHAT IS QUBIT-ONLY AND INCOMPATIBLE WITH ffsim LinearOperators:
#   comm_sq_exp_fast     — calls of.get_sparse_operator; pass it a LinearOperator and it breaks
#   expand_state         — expands into 2^(2*norb) Fock space; only needed for the JW path
#
# To add a new symmetry type:
#   1. Express it as an of.FermionOperator  →  fermion_op_to_linop(op, norb, nelec)
#   2. Or build ffsim.FermionOperator directly  →  ffsim.linear_operator(op, norb, nelec)
#   3. Pass the resulting LinearOperator in the symmetries list to commutator_cost
#   That's it. commutator_cost needs no changes.
 
import pyscf.tools.fcidump
 
 
def of_to_ffsim(op: of.FermionOperator) -> ffsim.FermionOperator:
    """
    Remap of.FermionOperator to ffsim.FermionOperator.
    OF: mode 2p -> cre/des_a(p),  mode 2p+1 -> cre/des_b(p).
    Pure index remap — no Jordan-Wigner, no 2^(2*norb) matrix.
    """
    terms = {}
    for term, coeff in op.terms.items():
        ffsim_ops = []
        for mode, action in term:
            p, spin = divmod(mode, 2)
            if action == 1:
                ffsim_ops.append(ffsim.cre_a(p) if spin == 0 else ffsim.cre_b(p))
            else:
                ffsim_ops.append(ffsim.des_a(p) if spin == 0 else ffsim.des_b(p))
        terms[tuple(ffsim_ops)] = coeff
    return ffsim.FermionOperator(terms)
 
 
def fermion_op_to_linop(
    op: of.FermionOperator,
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """of.FermionOperator -> LinearOperator on the ffsim FCI subspace. No qubits."""
    return ffsim.linear_operator(of_to_ffsim(op), norb, nelec)

 
def fci_reference(moldata: ffsim.MolecularData) -> np.ndarray:
    h = ffsim.linear_operator(moldata.hamiltonian, norb=moldata.norb, nelec=moldata.nelec)
    _, v = scipy.sparse.linalg.eigsh(h, k=1, which="SA")
    return v[:, 0].astype(complex)
 
 
def hf_reference(moldata: ffsim.MolecularData) -> np.ndarray:
    return ffsim.hartree_fock_state(moldata.norb, moldata.nelec).astype(complex)
 
 
def optimize_fcidump(
    input_path: str,
    symmetry_op: "of.FermionOperator | list[of.FermionOperator] | list[scipy.sparse.linalg.LinearOperator]",
    reference_fn: "Callable[[ffsim.MolecularData], np.ndarray]",
    output_path: "str | None" = None,
    x0: "np.ndarray | None" = None,
    method: str = "L-BFGS-B",
    maxiter: int = 500,
    verbose: bool = False,
    cost: str = "NC",
) -> scipy.optimize.OptimizeResult:
    """
    Load FCIDUMP, minimise cost(H(U), S, |psi(U)>), write rotated FCIDUMP.
 
    symmetry_op  : of.FermionOperator, list[of.FermionOperator], or list[LinearOperator].
    reference_fn : Callable[[ffsim.MolecularData], np.ndarray]
                   Built-ins: fci_reference, hf_reference.
                   Custom:    lambda md: your_solver(md, your_initial_vector)
    cost         : "NC" (commutator, default) or "variance".
    output_path  : write rotated FCIDUMP here; None to skip.
    """
    print(f"optimize_fcidump: input_path={input_path}, output_path={output_path}, "
          f"symmetry_op={type(symmetry_op)}, reference_fn={reference_fn.__name__}, cost={cost}, method={method}, maxiter={maxiter}")
    
    moldata = load_moldata(input_path)
    norb, nelec = moldata.norb, moldata.nelec
 
    state = reference_fn(moldata)
 
    if isinstance(symmetry_op, of.FermionOperator):
        sym_linops = [fermion_op_to_linop(symmetry_op, norb, nelec)]
    elif symmetry_op and isinstance(symmetry_op[0], of.FermionOperator):
        sym_linops = [fermion_op_to_linop(op, norb, nelec) for op in symmetry_op]
    else:
        sym_linops = list(symmetry_op)  # already LinearOperators
 
    if cost == "NC":
        f = commutator_cost(moldata, sym_linops, state)
    elif cost == "variance":
        f = variance_cost(moldata, sym_linops, state)
    else:
        raise ValueError("cost must be 'NC' or 'variance'")
 
    x0 = np.zeros(comb(norb, 2)) if x0 is None else x0
 
    if verbose:
        print(f"NC before: {f(x0):.6e}")
 
    res = scipy.optimize.minimize(f, x0, method=method,
                                  options={"maxiter": maxiter},
                                  callback=callback if verbose else None)
    if verbose:
        print(f"NC after:  {res.fun:.6e}  ({res.message})")
 
    if output_path is not None:
        U_opt = x_to_rotation(res.x, norb)
        rh    = moldata.hamiltonian.rotated(U_opt)
        ffsim.MolecularData(
            atom=moldata.atom, basis=moldata.basis, spin=moldata.spin, nelec=nelec,
            hf_energy=moldata.hf_energy, norb=norb, core_energy=moldata.core_energy,
            one_body_integrals=rh.one_body_tensor,
            two_body_integrals=rh.two_body_tensor,
        ).to_fcidump(output_path)
 
    return res
 
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("molpath")
    parser.add_argument("parity", nargs="?", default=None,
                        help="path to the incidence matrix of symmetries")
    parser.add_argument("--seniority", action="store_true")
    parser.add_argument("--reference", default="fci")   # fci or hf
    parser.add_argument("--cost_function", default="NC")   # NC or variance
    parser.add_argument("--x0", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None)
    parser.add_argument("--output_fcidump", default=None,
                        help="write rotated FCIDUMP here")
    parser.add_argument("--orbene_npy", default=None,
                        help="save rotated orbital energies (h1e diagonal) to this .npy file")

    args = parser.parse_args()

    moldata  = load_moldata(args.molpath)
    dumpdata = fcidump_data(args.molpath)

    # ── symmetries ────────────────────────────────────────────────────────────
    if args.seniority:
        sym_of = of.FermionOperator()
        for p in range(moldata.norb):
            # local seniority operator: n_{p,alpha} + n_{p,beta} - 2 n_{p,alpha} n_{p,beta}
            a, b = 2 * p, 2 * p + 1
            sym_of += of.FermionOperator(f"{a}^ {a}", 1.0)
            sym_of += of.FermionOperator(f"{b}^ {b}", 1.0)
            sym_of += of.FermionOperator(((a, 1), (b, 1), (b, 0), (a, 0)), -2.0)
        symmetry_op = sym_of
        sym_linops  = [fermion_op_to_linop(sym_of, moldata.norb, moldata.nelec)]
    elif args.parity is not None:
        parity_matrix = np.loadtxt(args.parity, dtype=int)
        symmetry_op   = parity_matrix_to_quasisymmetries(parity_matrix,
                                                          moldata.norb,
                                                          moldata.nelec)
        sym_linops    = symmetry_op
    else:
        parser.error("supply a parity matrix file or --seniority")

    # ── reference state ───────────────────────────────────────────────────────
    if args.reference == "fci":
        _, state = get_fci(dumpdata)
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("reference must be fci or hf")
    # Wrap as callable; optimize_fcidump will call reference_fn(moldata) internally.
    # Using a lambda that returns the already-computed state avoids running the
    # solver a second time.
    reference_fn = lambda md: state

    # ── nc_before (needed for logging) ────────────────────────────────────────
    x0 = np.loadtxt(args.x0) if args.x0 else np.zeros(comb(moldata.norb, 2))
    if args.cost_function == "NC":
        f_before = commutator_cost(moldata, sym_linops, state)
    elif args.cost_function == "variance":
        f_before = variance_cost(moldata, sym_linops, state)
    else:
        raise ValueError("cost must be 'NC' or 'variance'")
    nc_before = f_before(x0)
    print("before optimization: {0:4.6f}".format(nc_before))

    # ── optimise ──────────────────────────────────────────────────────────────
    t_start = time.time()
    res = optimize_fcidump(
        input_path=args.molpath,
        symmetry_op=symmetry_op,
        reference_fn=reference_fn,
        output_path=args.output_fcidump,
        x0=x0,
        method="L-BFGS-B",
        maxiter=100,
        verbose=args.verbose,
        cost=args.cost_function,
    )
    elapsed = time.time() - t_start
    print(res.message)
    print("optimized: {0:4.6f}".format(res.fun))

    outname = args.outname if args.outname else time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".txt"
    with open(outname, "a", newline="") as fp:
        fp.write(str(vars(args)) + "\n")
        fp.write(f"# nc_before={nc_before:.6e}  nc_after={res.fun:.6e}  "
                 f"converged={res.success}  nit={res.nit}  nfev={res.nfev}  "
                 f"elapsed_s={elapsed:.1f}  message={res.message}\n")
        np.savetxt(fp, res.x)

    # ── orbene_npy (optional) ─────────────────────────────────────────────────
    if args.orbene_npy:
        # Generalized Fock matrix diagonal in the rotated orbital basis.
        # See comments in the original __main__ block for the full rationale.
        from pyscf.fci import rdm as fci_rdm
        norb, nelec = moldata.norb, moldata.nelec
        state_2d = np.asarray(state).real.reshape(
            comb(norb, nelec[0]), comb(norb, nelec[1])
        )
        U_opt = x_to_rotation(res.x, norb)
        rh    = moldata.hamiltonian.rotated(U_opt)
        rdm1a = fci_rdm.make_rdm1_spin1(
            fname='FCImake_rdm1a',
            cibra=state_2d, ciket=state_2d, norb=norb, nelec=nelec
        )
        rdm1b = fci_rdm.make_rdm1_spin1(
            fname='FCImake_rdm1b',
            cibra=state_2d, ciket=state_2d, norb=norb, nelec=nelec
        )
        rdm1_rot = U_opt.T @ (rdm1a + rdm1b) @ U_opt
        h2       = rh.two_body_tensor
        J        = np.einsum('pqrs,rs->pq', h2, rdm1_rot)
        K        = np.einsum('psrq,rs->pq', h2, rdm1_rot)
        fock_rot = rh.one_body_tensor + J - 0.5 * K
        np.save(args.orbene_npy, np.diag(fock_rot).real)
