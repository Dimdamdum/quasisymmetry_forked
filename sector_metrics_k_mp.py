"""
sector_metrics.py

Decompose an FCI ground state into "sectors" defined by a collection of
approximate Z2 symmetries of the form (-1)^(sum_i n_i), and report:

  1. n_eigenstates : number of sector eigenstates (globally ranked by
     overlap with the ground state, pooled across sectors) needed so
     that the truncated Rayleigh quotient reaches chemical accuracy.
  2. n_sectors      : number of sectors (ranked by their FCI weight)
     needed so that the state projected onto their union reaches
     chemical accuracy.
  3. dim_max_sector : dimension of the largest sector used in (2).
  4. dim_sum_sectors: sum of dimensions of all sectors used in (2).

Design notes / assumptions (confirm before trusting numbers):

  * "Sector eigenstates" (metric 1) = eigenstates of P_S H P_S, the
    Hamiltonian projected onto one sector S -- as you defined it.
    They are never explicitly assembled as a matrix; P_S H P_S is
    applied to a vector via pyscf's matrix-free sigma-vector routine
    (`contract_2e`) with a projection (zeroing) step before and after.
  * Metric 1's "reached chemical accuracy" check uses the *truncated,
    non-renormalized* Rayleigh quotient <psi_trunc|H|psi_trunc> /
    <psi_trunc|psi_trunc> of the running sum of chosen eigenstates,
    per your preference (truncating over renormalizing).
  * Metric 2 does not require any diagonalization: sectors are ranked
    purely by their FCI weight sum_{I in S} |c_I|^2, and the same
    truncated-Rayleigh-quotient criterion decides how many are needed.
    Metrics 3 and 4 report properties of that same chosen sector set.
  * The orbital rotation is applied directly to the CI vector via
    pyscf.fci.addons.transform_ci_for_orbital_rotation (Thouless-type
    single-excitation basis change) -- the Hamiltonian integrals are
    never rotated/rebuilt, and no CI-space matrix is ever built.
  * Sector membership is computed from the *integer bitmask* string
    representation pyscf already uses for determinants (each alpha-
    and beta-string is an int with one bit per occupied spatial
    orbital), via `popcount(string & mask) % 2` -- O(1) per
    determinant per symmetry, no explicit operators anywhere.
  * A symmetry matrix may be given as (k, norb) -- spin-symmetric,
    same mask applied to alpha and beta strings -- or (k, 2*norb),
    general, with columns [alpha_0..alpha_{norb-1}, beta_0..beta_{norb-1}].

Chemical accuracy default: 1 kcal/mol = 1.5936e-3 Hartree.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigsh

from pyscf import fci, ao2mo
from pyscf.tools import fcidump as fcidump_tools

import time

from mpi4py.futures import MPIPoolExecutor

CHEMICAL_ACCURACY = 1.5936e-3  # Hartree


# --------------------------------------------------------------------------
# Hamiltonian / FCI-state loading
# --------------------------------------------------------------------------

def load_hamiltonian_fcidump(path):
    """Read integrals from an FCIDUMP file."""
    data = fcidump_tools.read(path)
    h1e = data["H1"]
    eri = data["H2"]
    norb = data["NORB"]
    nelec = data["NELEC"]
    ecore = data.get("ECORE", 0.0)
    ms2 = data.get("MS2", 0)
    na = (nelec + ms2) // 2
    nb = (nelec - ms2) // 2
    eri = ao2mo.restore(1, eri, norb)  # -> dense (norb,norb,norb,norb)
    return h1e, eri, norb, (na, nb), ecore


def load_hamiltonian_chkfile(path):
    """Read mean-field orbitals from a pyscf checkfile and build MO integrals."""
    from pyscf import lib, scf
    mol = lib.chkfile.load_mol(path)
    mo_coeff = lib.chkfile.load(path, "scf/mo_coeff")
    mf = scf.RHF(mol)
    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    eri = ao2mo.full(mol, mo_coeff)
    norb = mo_coeff.shape[1]
    na = mol.nelec[0]
    nb = mol.nelec[1]
    eri = ao2mo.restore(1, eri, norb)
    return h1e, eri, norb, (na, nb), mol.energy_nuc()


def get_fci_state(h1e, eri, norb, nelec, civec=None):
    """Return (energy, civec). Computes the FCI ground state if civec is None."""
    solver = fci.direct_spin1.FCI()
    if civec is None:
        e, civec = solver.kernel(h1e, eri, norb, nelec, verbose=0)
        return e, civec
    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
    sigma = fci.direct_spin1.contract_2e(h2e, civec, norb, nelec)
    e = float(np.vdot(civec, sigma) / np.vdot(civec, civec))
    return e, civec


# --------------------------------------------------------------------------
# Orbital rotation
# --------------------------------------------------------------------------

def build_rotation_matrix(generator_utri, norb):
    """Antisymmetric kappa from upper-triangular entries, then U = expm(kappa)."""
    iu = np.triu_indices(norb, 1)
    if len(generator_utri) != len(iu[0]):
        raise ValueError(
            f"Expected {len(iu[0])} upper-triangular entries for norb={norb}, "
            f"got {len(generator_utri)}."
        )
    kappa = np.zeros((norb, norb))
    kappa[iu] = generator_utri
    kappa -= kappa.T
    return expm(kappa)


def rotate_civec(civec, norb, nelec, u):
    return fci.addons.transform_ci_for_orbital_rotation(civec, norb, nelec, u)


def rotate_integrals(h1e, eri, u):
    """Rotate one/two-electron integrals into the same basis civec was
    rotated into. Must match pyscf's convention CI_new = kernel(u^T h1 u, ...)
    -- i.e. new index p, q, ... contract against u's *column* p, q, ...
    (h1_new = u^T h1 u; eri_new[p,q,r,s] = sum_ijkl u[i,p]u[j,q]u[k,r]u[l,s] eri[i,j,k,l])."""
    h1e_rot = u.T @ h1e @ u
    eri_rot = np.einsum("ip,jq,kr,ls,ijkl->pqrs", u, u, u, u, eri, optimize=True)
    return h1e_rot, eri_rot


# --------------------------------------------------------------------------
# Sector assignment
# --------------------------------------------------------------------------



def _popcount_parity(strings, mask):
    """(-1) exponent parity of popcount(string & mask), vectorized."""
    masked = np.bitwise_and(strings.astype(np.int64), int(mask))
    # numpy has no builtin popcount for arbitrary width; norb <= ~20 here,
    # so unpackbits on a view is simple and fast enough.
    out = np.zeros(masked.shape, dtype=np.int64)
    m = masked.copy()
    while np.any(m):
        out ^= (m & 1)
        m >>= 1
    return out  # 0 or 1 per string


def assign_sectors(symmetries, norb, nelec):
    """
    symmetries: (k, norb) spin-symmetric or (k, 2*norb) general 0/1 matrix.
    Returns sector_id: (n_alpha_strings, n_beta_strings) int array whose
    value's binary digits are the k symmetry eigenvalues (0/1 each) for
    that determinant -- i.e. the sector signature packed into an int.
    """
    na, nb = nelec
    strings_a = fci.cistring.make_strings(range(norb), na)
    strings_b = fci.cistring.make_strings(range(norb), nb)
    symmetries = np.asarray(symmetries)
    k = symmetries.shape[0]

    sector_id = np.zeros((len(strings_a), len(strings_b)), dtype=np.int64)
    for row_idx in range(k):
        row = symmetries[row_idx]
        if symmetries.shape[1] == norb:
            mask_a = mask_b = int(np.sum(row * (1 << np.arange(norb))))
        elif symmetries.shape[1] == 2 * norb:
            mask_a = int(np.sum(row[:norb] * (1 << np.arange(norb))))
            mask_b = int(np.sum(row[norb:] * (1 << np.arange(norb))))
        else:
            raise ValueError("symmetries must have norb or 2*norb columns")
        par_a = _popcount_parity(strings_a, mask_a)
        par_b = _popcount_parity(strings_b, mask_b)
        bit = np.add.outer(par_a, par_b) % 2  # (na_str, nb_str)
        sector_id |= (bit.astype(np.int64) << row_idx)
    return sector_id


def sector_weights(civec, sector_id):
    """dict: sector signature (int) -> total FCI weight sum|c_I|^2."""
    flat_sec = sector_id.ravel()
    flat_w = (np.abs(civec.ravel()) ** 2)
    order = np.argsort(flat_sec)
    sec_sorted = flat_sec[order]
    w_sorted = flat_w[order]
    uniq, start = np.unique(sec_sorted, return_index=True)
    sums = np.add.reduceat(w_sorted, start)
    return dict(zip(uniq.tolist(), sums.tolist()))


def sector_dim(sector_id):
    """dict: sector signature (int) -> number of determinants in it."""
    uniq, counts = np.unique(sector_id, return_counts=True)
    return dict(zip(uniq.tolist(), counts.tolist()))


# --------------------------------------------------------------------------
# Matrix-free H, and P_S H P_S
# --------------------------------------------------------------------------

def make_h2e(h1e, eri, norb, nelec):
    return fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)


def apply_h(vec2d, h2e, norb, nelec):
    return fci.direct_spin1.contract_2e(h2e, vec2d, norb, nelec)


def rayleigh_quotient(vec2d, h2e, norb, nelec):
    sigma = apply_h(vec2d, h2e, norb, nelec)
    num = np.vdot(vec2d, sigma).real
    den = np.vdot(vec2d, vec2d).real
    return num / den


def sector_restricted_eigs(
    h2e,
    norb,
    nelec,
    sector_id,
    sig,
    max_eigs,
    shape,
):
    mask = (sector_id == sig)
    dim = int(mask.sum())

    if dim == 0:
        return np.array([]), np.empty((0, 0))

    def matvec(v_flat):
        # Expand the compressed sector vector into the full FCI space.
        v2d = np.zeros(shape, dtype=v_flat.dtype)
        v2d[mask] = v_flat
        # Apply H.
        sigma = apply_h(v2d, h2e, norb, nelec)
        # Compress back to the sector.
        return sigma[mask]

    if dim == 1:
        v = np.array([1.0], dtype=np.float64)
        # Since the sector is one-dimensional, we only need H|v>.
        e = matvec(v)[0]
        return np.array([e]), v.reshape(1, 1)

    if dim <= 200:
        # Construct the sector-restricted Hamiltonian.
        basis = np.eye(dim)
        cols = [matvec(basis[:, j]) for j in range(dim)]
        Hs = np.stack(cols, axis=1)
        # Numerical symmetrization protects against tiny
        # floating-point non-Hermiticity.
        Hs = 0.5 * (Hs + Hs.T.conj())

        evals, evecs = np.linalg.eigh(Hs)
        # Keep only the lowest max_eigs states.
        n_keep = min(max_eigs, dim)
        return evals[:n_keep], evecs[:, :n_keep]

    else:
        from scipy.sparse.linalg import LinearOperator, eigsh
        linop = LinearOperator(
            (dim, dim),
            matvec=matvec,
            dtype=np.float64,
        )
        n_keep = min(max_eigs, dim - 1)

        evals, evecs = eigsh(
            linop,
            k=n_keep,
            which="SA",
        )
        order = np.argsort(evals)
        return evals[order], evecs[:, order]




# --------------------------------------------------------------------------
# Metrics 2, 3, 4  (whole-sector greedy truncation, no diagonalization)
# --------------------------------------------------------------------------

def metrics_234(civec, sector_id, h1e, eri, norb, nelec, e_exact,
                 chem_acc=CHEMICAL_ACCURACY):
    weights = sector_weights(civec, sector_id)
    dims = sector_dim(sector_id)
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])

    h2e = make_h2e(h1e, eri, norb, nelec)
    chosen = []
    mask_union = np.zeros_like(civec, dtype=bool)
    for sig, w in ranked:
        chosen.append(sig)
        mask_union |= (sector_id == sig)
        trunc = np.where(mask_union, civec, 0.0)
        e_trunc = rayleigh_quotient(trunc, h2e, norb, nelec)
        if abs(e_trunc - e_exact) < chem_acc:
            break

    n_sectors = len(chosen)
    dims_used = [dims[s] for s in chosen]
    return {
        "n_sectors": n_sectors,
        "dim_max_sector": max(dims_used),
        "dim_sum_sectors": sum(dims_used),
        "chosen_sectors": chosen,
    }


# --------------------------------------------------------------------------
# Metric 1  (globally ranked sector eigenstates)
# --------------------------------------------------------------------------

_worker_state = {}


def _init_worker(
    h2e,
    norb,
    nelec,
    sector_id,
    shape,
    civec,
    max_eigs,
):
    """
    Initialization for the sector-diagonalization workers.
    """
    _worker_state.update(
        h2e=h2e,
        norb=norb,
        nelec=nelec,
        sector_id=sector_id,
        shape=shape,
        civec=civec,
        max_eigs=max_eigs,
    )


def process_sector(sig):
    """
    Diagonalize one sector.

    Returns compressed sector eigenvectors rather than full FCI vectors.
    """
    s = _worker_state

    evals, evecs = sector_restricted_eigs(
        s["h2e"],
        s["norb"],
        s["nelec"],
        s["sector_id"],
        sig,
        s["max_eigs"],
        s["shape"],
    )

    mask = (s["sector_id"] == sig)
    civec_sector = s["civec"][mask]

    out = []

    for e, v in zip(evals, evecs.T):
        overlap = np.vdot(v, civec_sector)

        out.append(
            (
                float(abs(overlap) ** 2),
                float(e),
                overlap,
                sig,
                v.copy(),
            )
        )

    return out


# --------------------------------------------------------------------------
# Second-stage workers
# --------------------------------------------------------------------------

_metric_worker_state = {}


def _init_metric_worker(
    h2e,
    norb,
    nelec,
    sector_id,
    shape,
):
    """
    Initialization for the workers used in the post-diagonalization stage.
    Only genuinely shared, worker-independent data goes here -- NOT the
    (potentially huge) list of pooled states, which belongs in .map()
    instead so each task's payload is sent once, not duplicated into
    every worker.
    """
    _metric_worker_state.update(
        h2e=h2e,
        norb=norb,
        nelec=nelec,
        sector_id=sector_id,
        shape=shape,
    )


def process_hamiltonian_state(state):
    """
    Calculate H | phi_j > for one pooled state.

    The result is returned as a full FCI vector.

    This is intentionally the expensive operation that we distribute
    over MPI workers. `state` carries exactly the data this one task
    needs -- nothing shared across all tasks is duplicated here.
    """
    s = _metric_worker_state

    overlap, energy, sig, v_sector = state

    mask = (s["sector_id"] == sig)

    v_full = np.zeros(s["shape"], dtype=v_sector.dtype)
    v_full[mask] = v_sector

    sigma = apply_h(
        v_full,
        s["h2e"],
        s["norb"],
        s["nelec"],
    )

    # return sigma
    nz_rows, nz_cols = np.nonzero(sigma)
    nz_vals = sigma[nz_rows, nz_cols]
    return nz_rows.astype(np.int32), nz_cols.astype(np.int32), nz_vals


# --------------------------------------------------------------------------
# Metric 1
# --------------------------------------------------------------------------

def metric_1(
    civec,
    sector_id,
    h1e,
    eri,
    norb,
    nelec,
    e_exact,
    chem_acc=CHEMICAL_ACCURACY,
    max_eigs_per_sector=20,
    max_sectors_considered=None,
    n_workers=None,
    max_eigs_total=1000
):
    # ================================================================
    # 1. Rank sectors by FCI weight
    # ================================================================

    weights = sector_weights(civec, sector_id)

    ranked = sorted(
        weights.items(),
        key=lambda kv: -kv[1],
    )

    if max_sectors_considered is not None:
        ranked = ranked[:max_sectors_considered]

    h2e = make_h2e(
        h1e,
        eri,
        norb,
        nelec,
    )

    sigs = [sig for sig, _ in ranked]

    # ================================================================
    # 2. Diagonalize sectors in parallel
    # ================================================================

    pool = []

    with MPIPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            h2e,
            norb,
            nelec,
            sector_id,
            civec.shape,
            civec,
            max_eigs_per_sector,
        ),
    ) as executor:

        for result in executor.map(
            process_sector,
            sigs,
        ):
            pool.extend(result)

    # ================================================================
    # 3. Globally rank the sector eigenstates
    # ================================================================

    pool.sort(
        key=lambda x: -x[0]
    )

    print(
        f"Number of candidate eigenstates: {len(pool)}"
    )

    #
    # Convert
    #
    #   (overlap2, energy, overlap, sig, v)
    #
    # into
    #
    #   (overlap, energy, sig, v)
    #
    states = [
        (
            overlap,
            energy,
            sig,
            v,
        )
        for overlap2, energy, overlap, sig, v
        in pool
    ]

    del pool

    # ================================================================
    # 4. Calculate H | phi_j > in parallel
    # ================================================================

    #
    # We keep sigma_j vectors in the manager only temporarily.
    #
    # More importantly, we calculate each expensive contract_2e
    # exactly once.
    #

    sigma_vectors = [None] * min(len(states), max_eigs_total)

    with MPIPoolExecutor(
        max_workers=n_workers,
        initializer=_init_metric_worker,
        initargs=(
            h2e,
            norb,
            nelec,
            sector_id,
            civec.shape,
        ),
    ) as executor:

        for j, sigma in enumerate(
            executor.map(process_hamiltonian_state, states[:min(len(states), max_eigs_total)])
        ):
            sigma_vectors[j] = sigma

    # ================================================================
    # 5. Calculate prefix Rayleigh quotients
    # ================================================================

    #
    # At this point there is NO expensive Hamiltonian operation left.
    #
    # We reconstruct the running vector and its H-image simultaneously:
    #
    #     psi_n       = sum_i c_i phi_i
    #
    #     H psi_n    = sum_i c_i H phi_i
    #
    # Therefore each subsequent energy calculation only requires
    # vector additions and dot products.
    #

    running = np.zeros_like(civec)
    sigma_running = np.zeros_like(civec)

    for n, (
        overlap,
        sector_energy,
        sig,
        v_sector,
    ) in enumerate(
        states[:min(len(states), max_eigs_total)],
        start=1,
    ):
        mask = (sector_id == sig)

        # Add c_n |phi_n>
        running[mask] += overlap * v_sector

        # Add c_n H|phi_n>
        # sigma_running += overlap * sigma_vectors[n - 1]

        nz_rows, nz_cols, nz_vals = sigma_vectors[n - 1]
        sigma_running[nz_rows, nz_cols] += overlap * nz_vals

        # Rayleigh quotient.
        numerator = np.vdot(
            running,
            sigma_running,
        ).real

        denominator = np.vdot(
            running,
            running,
        ).real

        e_trunc = numerator / denominator

        if n % 100 == 0 or n == 1:
            print(
                f"n = {n}, "
                f"E = {e_trunc:.12f}, "
                f"|E-E_exact| = "
                f"{abs(e_trunc - e_exact):.6e}"
            )

        if abs(e_trunc - e_exact) < chem_acc:
            return {
                "n_eigenstates": n,
                "n_pool": len(states),
            }

    return {
        "n_eigenstates": None,
        "n_pool": len(states),
        "warning": (
            "chemical accuracy not reached; "
            "increase max_eigs_per_sector."
        ),
    }



# --------------------------------------------------------------------------
# End-to-end driver
# --------------------------------------------------------------------------

def compute_all_metrics(h1e, eri, norb, nelec, symmetries,
                         rotation_generator=None, civec=None,
                         chem_acc=CHEMICAL_ACCURACY,
                         max_eigs_per_sector=20, max_sectors_considered=None,
                         ecore=0.0, max_eigs_total=1000):
    # e_exact (and every Rayleigh quotient computed downstream) is the
    # *electronic* energy only -- contract_2e/absorb_h1e never see ecore.
    # Comparisons against chem_acc are differences, so the missing constant
    # cancels and does not affect convergence; ecore is added back only in
    # the final reported total energy below.
    t = time.time()
    e_exact, civec = get_fci_state(h1e, eri, norb, nelec, civec)
    print("FCI:", time.time() - t)

    if rotation_generator is not None:
        u = build_rotation_matrix(rotation_generator, norb)
        civec = rotate_civec(civec, norb, nelec, u)
        # civec is now expressed in the rotated-orbital determinant basis;
        # every H-application below must use integrals in that same basis.
        h1e, eri = rotate_integrals(h1e, eri, u)

    sector_id = assign_sectors(symmetries, norb, nelec)
    t = time.time()
    m234 = metrics_234(civec, sector_id, h1e, eri, norb, nelec, e_exact, chem_acc)
    print("Metrics234:", time.time() - t)

    t = time.time()
    m1 = metric_1(civec, sector_id, h1e, eri, norb, nelec, e_exact, chem_acc,
                  max_eigs_per_sector, max_sectors_considered, max_eigs_total=max_eigs_total)
    print("Metric1:", time.time() - t)

    return {
        "E_exact": e_exact + ecore,
        "E_exact_electronic": e_exact,
        "ecore": ecore,
        "n_eigenstates": m1["n_eigenstates"],
        "n_sectors": m234["n_sectors"],
        "dim_max_sector": m234["dim_max_sector"],
        "dim_sum_sectors": m234["dim_sum_sectors"],
        "_details": {"metric1": m1, "metric234": m234},
    }


if __name__ == "__main__":
    print("Starting at " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcidump", help="path to FCIDUMP file")
    parser.add_argument("--chkfile", help="path to pyscf checkfile")
    parser.add_argument("--symmetries", required=True,
                         help=".npy file, (k,norb) or (k,2*norb) 0/1 array")
    parser.add_argument("--rotation", default=None,
                         help=".npy file of upper-triangular generator entries")
    parser.add_argument("--chem-acc", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--max-eigs-per-sector", type=int, default=20)
    parser.add_argument("--max-sectors-considered", type=int, default=None)
    parser.add_argument("--max-eigs-total", type=int, default=1000)
    args = parser.parse_args()

    if args.fcidump:
        h1e, eri, norb, nelec, ecore = load_hamiltonian_fcidump(args.fcidump)
    elif args.chkfile:
        h1e, eri, norb, nelec, ecore = load_hamiltonian_chkfile(args.chkfile)
    else:
        raise SystemExit("Provide --fcidump or --chkfile")

    def load_matrix(path):
        return np.load(path) if path.endswith(".npy") else np.loadtxt(path)

    symmetries = load_matrix(args.symmetries)
    rotation = load_matrix(args.rotation) if args.rotation else None

    result = compute_all_metrics(
        h1e, eri, norb, nelec, symmetries,
        rotation_generator=rotation,
        chem_acc=args.chem_acc,
        max_eigs_per_sector=args.max_eigs_per_sector,
        max_sectors_considered=args.max_sectors_considered,
        ecore=ecore,
        max_eigs_total=args.max_eigs_total
    )
    for k in ["E_exact", "n_eigenstates", "n_sectors", "dim_max_sector", "dim_sum_sectors"]:
        print(f"{k}: {result[k]}")
    if result["n_eigenstates"] is None:
        print("Couldn't reach chemical accuracy, try increasing --max-eigs-per-sector or --max-eigs-total") 
    print("Finished at " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
