"""System-agnostic approximate-symmetry diagnostics for iron-sulfur Hamiltonians.

Covers all three quasisymmetry families of the project:

  * ORBITAL SENIORITIES (local parities) s_p = (-1)^{n_p}, roadmap Eq. 1.
    A one-orbital ParityConstraint IS a local seniority; `all_odd` builds the
    joint all-singly-occupied sector.  GLOBAL seniority Omega is not a parity
    label (it is a sum, not a Z2 grading) and gets its own routines:
    `seniority_weights`, `seniority_values`, `seniority_mask`.
  * QUARTETS / block parities Q_pq = s_p s_q, roadmap Eq. 2 -> `pair_parities`,
    `block_parities`.
  * LOCAL SPINS S_A, roadmap Eq. 3 -> `local_spin_distribution`,
    `joint_local_spin_weight`.  These are NOT diagonal in the determinant
    basis and are handled with genuine operator projectors, not masks.

Non-commutativity (NC) scores ||[H, Q]|psi>||^2 for any diagonal label
operator Q are in `commutator_norm2` / `nc_matrix`.

Implements the six reporting requirements of Sec. 8 of the benchmark roadmap
(FeS_approximate_symmetry_benchmark_roadmap, 2026-07-13) for an arbitrary
active space, arbitrary orbital blocks, and arbitrary parity sector specs:

  8.1 state concentration   sector_weight, orbital_parity_weights,
                            local_spin_distribution
  8.2 compression           sector_dimensions  ->  C(P) = dim H / dim(P H)
  8.3 projected accuracy    projected_spectrum  ->  dE_k, overlaps
  8.4 leakage               leakage             ->  ||(1-P) H P |Psi>||^2
  8.5 orbital-basis robust. every routine takes (h1e, eri, ci) for one frame;
                            the driver loops frames.  Frame construction and
                            the orbital map between frames are NOT here.
  8.6 reference convergence not applicable to exact FCI; for DMRG the caller
                            must supply states at several bond dimensions.

Design notes (why this differs from the CAS(10e,10o) prototype)
--------------------------------------------------------------
1. Parity sectors are DIAGONAL in the determinant basis.  A group parity
   (-1)^{n_g} for any orbital set g is one bit per alpha string XOR one bit
   per beta string.  Projection is therefore an O(dim) boolean mask, not a
   sequence of sparse matvecs, and never a Python loop over determinants.
   This is what makes CAS(10e,20o) (dim 2.4e8) and CAS(22e,16o) (dim 1.9e7)
   tractable with the same code.

2. Local spin uses the spin-free form of Dobrautz et al. (JCTC 2021, 17,
   5684), Appendix C/D:
       S_i^2      = (3/4) (E_ii - e_ii,ii)
       S_i . S_j  = -(1/2) e_ij,ji + (1/4) e_ii,jj      (i != j)
   so (a) the full spin-correlation matrix C[p,q] comes from ONE spin-free
   2-RDM instead of norb^2 operator applications, and (b) S_A^2 acting on a
   CI vector is a single pyscf contract_2e call.

3. The local-spin distribution w_A(s) uses the product (spectral-filter)
   form of the projector, not the moment/Lagrange form.  Both are the same
   polynomial; the product form avoids forming high moments of S_A^2, whose
   Vandermonde solve degrades as the block grows.  moment_local_spin_
   distribution is kept only as an independent cross-check in the tests.

4. Nothing in this module chooses a sector, a block, or an orbital frame.
   Those are inputs.

Units: energies in Hartree throughout.  Leakage is Ha^2 before normalization;
see the `energy_scale` argument of `leakage` and the note there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Iterable, Sequence

import numpy as np
from pyscf.fci import cistring, direct_nosym, direct_spin1, spin_op

# --------------------------------------------------------------------------
# 1. Parity sector specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityConstraint:
    """Requires (-1)^{sum_{p in orbitals} n_p} == eigenvalue.

    orbitals   : spatial orbital indices forming the group g (0-based).
                 len == 1 -> orbital seniority s_p  (roadmap Eq. 1)
                 len == 2 -> quartet / pair parity Q_pq (roadmap Eq. 2)
                 len  > 2 -> block parity of a chemical fragment
    eigenvalue : +1 (even occupation) or -1 (odd occupation)
    """

    orbitals: tuple[int, ...]
    eigenvalue: int

    def __post_init__(self):
        if self.eigenvalue not in (1, -1):
            raise ValueError("parity eigenvalue must be +1 or -1")
        if len(set(self.orbitals)) != len(self.orbitals):
            raise ValueError("repeated orbital in a parity group")


@dataclass
class SectorSpec:
    """A joint parity sector: the product of its constraints' projectors."""

    constraints: list[ParityConstraint] = field(default_factory=list)
    label: str = ""

    def orbitals_touched(self) -> set[int]:
        return {p for c in self.constraints for p in c.orbitals}

    def n_generators(self) -> int:
        return len(self.constraints)


def all_odd(orbitals: Iterable[int], label: str = "") -> SectorSpec:
    """P = prod_p (1 - s_p)/2 over the given orbitals (roadmap Eq. 6).

    Orbitals NOT listed are unconstrained.  For the CAS(10e,10o) control,
    listing all ten orbitals reproduces the all-singly-occupied sector.
    """
    orbitals = tuple(orbitals)
    return SectorSpec(
        [ParityConstraint((p,), -1) for p in orbitals],
        label or f"all-odd({len(orbitals)} orb)",
    )


def pair_parities(pairs: Sequence[tuple[int, int]], eigenvalue: int = 1,
                  label: str = "") -> SectorSpec:
    """Joint sector of two-orbital parities Q_pq = s_p s_q."""
    return SectorSpec(
        [ParityConstraint(tuple(pq), eigenvalue) for pq in pairs],
        label or f"{len(pairs)} pair-parities = {eigenvalue:+d}",
    )


def block_parities(blocks: Sequence[Sequence[int]], eigenvalues: Sequence[int],
                   label: str = "") -> SectorSpec:
    if len(blocks) != len(eigenvalues):
        raise ValueError("one eigenvalue per block required")
    return SectorSpec(
        [ParityConstraint(tuple(b), e) for b, e in zip(blocks, eigenvalues)],
        label or f"{len(blocks)} block parities",
    )


# --------------------------------------------------------------------------
# 2. Sector masks, weights, projection  (state concentration, Sec. 8.1)
# --------------------------------------------------------------------------


def _group_parity_bits(strings: np.ndarray, orbitals: tuple[int, ...]) -> np.ndarray:
    """Per-string bit: 1 if an odd number of `orbitals` are occupied."""
    bitmask = 0
    for p in orbitals:
        bitmask |= 1 << int(p)
    masked = np.asarray(strings, dtype=np.int64) & bitmask
    # popcount parity via the standard XOR-fold (int64 -> 1 bit)
    x = masked.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        x ^= x >> shift
    return (x & 1).astype(np.uint8)


def sector_mask(spec: SectorSpec, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Boolean (n_alpha_str, n_beta_str) mask of determinants inside the sector."""
    na, nb = nelec
    strs_a = cistring.make_strings(range(norb), na)
    strs_b = cistring.make_strings(range(norb), nb)
    mask = np.ones((len(strs_a), len(strs_b)), dtype=bool)
    for c in spec.constraints:
        pa = _group_parity_bits(strs_a, c.orbitals)
        pb = _group_parity_bits(strs_b, c.orbitals)
        joint_odd = pa[:, None] ^ pb[None, :]           # 1 -> parity is -1
        want_odd = np.uint8(1 if c.eigenvalue == -1 else 0)
        mask &= joint_odd == want_odd
    return mask


def project(ci: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """P |Psi> (unnormalized), same shape as ci."""
    return np.asarray(ci).reshape(mask.shape) * mask


def sector_weight(ci: np.ndarray, mask) -> float:
    """W(P) = <Psi|P|Psi> = || P|Psi> ||^2 for a projector.  Roadmap Eq. 10.

    `mask` may be a determinant mask (parity/seniority sectors) or a callable
    projector (local-spin sectors, which are not diagonal).
    """
    if callable(mask):
        pv = mask(ci)
        return float(np.asarray(ci).ravel() @ np.asarray(pv).ravel())
    v = np.asarray(ci).reshape(mask.shape)
    return float(np.sum(np.abs(v[mask]) ** 2))


def orbital_parity_weights(ci: np.ndarray, norb: int,
                           nelec: tuple[int, int]) -> np.ndarray:
    """w_p^- for every spatial orbital (roadmap Eq. 11): P(orbital p is odd).

    Returns shape (norb,).  Note prod_p w_p^- is NOT the joint weight; the
    joint weight is sector_weight(all_odd(...)).  Reporting both quantifies
    how correlated the orbital parities are.
    """
    na, nb = nelec
    strs_a = cistring.make_strings(range(norb), na)
    strs_b = cistring.make_strings(range(norb), nb)
    v = np.asarray(ci).reshape(len(strs_a), len(strs_b))
    w2 = np.abs(v) ** 2
    out = np.empty(norb)
    for p in range(norb):
        pa = _group_parity_bits(strs_a, (p,))
        pb = _group_parity_bits(strs_b, (p,))
        odd = pa[:, None] ^ pb[None, :]
        out[p] = float(w2[odd.astype(bool)].sum())
    return out


def seniority_weights(ci: np.ndarray, norb: int,
                      nelec: tuple[int, int]) -> dict[int, float]:
    """Global seniority distribution W_Omega, vectorized (no determinant loop)."""
    na, nb = nelec
    strs_a = np.asarray(cistring.make_strings(range(norb), na), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), nb), dtype=np.int64)
    v = np.abs(np.asarray(ci).reshape(len(strs_a), len(strs_b))) ** 2
    out: dict[int, float] = {}
    # chunk over alpha strings to bound memory at large dim
    chunk = max(1, int(2e7 // max(1, len(strs_b))))
    for start in range(0, len(strs_a), chunk):
        a = strs_a[start:start + chunk]
        singly = a[:, None] ^ strs_b[None, :]
        omega = _popcount(singly)
        for om in np.unique(omega):
            out[int(om)] = out.get(int(om), 0.0) + float(
                v[start:start + chunk][omega == om].sum())
    return dict(sorted(out.items()))


def seniority_values(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Omega(D) for every determinant: the number of singly occupied orbitals."""
    na, nb = nelec
    sa = np.asarray(cistring.make_strings(range(norb), na), dtype=np.int64)
    sb = np.asarray(cistring.make_strings(range(norb), nb), dtype=np.int64)
    return _popcount(sa[:, None] ^ sb[None, :])


def seniority_mask(norb: int, nelec: tuple[int, int], omega: int) -> np.ndarray:
    """Projector onto a fixed GLOBAL seniority Omega, as a determinant mask."""
    return seniority_values(norb, nelec) == omega


def parity_values(norb: int, nelec: tuple[int, int],
                  orbitals: Iterable[int]) -> np.ndarray:
    """Eigenvalues +-1 of the group parity (-1)^{n_g} for every determinant.

    orbitals of length 1 -> a local seniority s_p;  length 2 -> a quartet Q_pq.
    """
    na, nb = nelec
    orbitals = tuple(int(p) for p in orbitals)
    sa = cistring.make_strings(range(norb), na)
    sb = cistring.make_strings(range(norb), nb)
    odd = _group_parity_bits(sa, orbitals)[:, None] ^ \
        _group_parity_bits(sb, orbitals)[None, :]
    return 1 - 2 * odd.astype(np.int8)


def commutator_norm2(ci, qvals, h1e, eri, norb, nelec) -> float:
    """NC score  C[Q] = || [H, Q] |psi> ||^2  for a DIAGONAL label operator Q.

    qvals is the array of eigenvalues q(D) over determinants, e.g. from
    `seniority_values` (global seniority), `parity_values((p,))` (orbital
    seniority s_p) or `parity_values((p, q))` (quartet).  Units: Ha^2.
    [AUTHOR: this is the raw, unnormalized score of Eq. (nc_score) in
    Fe2S2_characterization; see `leakage` for the normalization discussion.]
    """
    v = np.asarray(ci, dtype=float).reshape(qvals.shape)
    lhs = _h_apply(qvals * v, h1e, eri, norb, nelec)
    rhs = qvals * _h_apply(v, h1e, eri, norb, nelec)
    return float(np.sum((lhs - rhs) ** 2))


def nc_matrix(ci, h1e, eri, norb, nelec) -> np.ndarray:
    """Symmetric matrix: diagonal = c_p for orbital seniorities s_p,
    upper triangle = c_pq for quartets Q_pq = s_p s_q.  Reproduces the
    noncommutativity map figures of Fe2S2_characterization."""
    M = np.zeros((norb, norb))
    for p in range(norb):
        M[p, p] = commutator_norm2(ci, parity_values(norb, nelec, (p,)),
                                   h1e, eri, norb, nelec)
        for q in range(p + 1, norb):
            M[p, q] = M[q, p] = commutator_norm2(
                ci, parity_values(norb, nelec, (p, q)), h1e, eri, norb, nelec)
    return M


def ffsim_parity_projector(ci, orbitals, eigenvalue, norb, nelec):
    """Same projection via openfermion/ffsim operators, for cross-checking.

    Kept because the operator route is the intuitive one, and because a
    parity projector is diagonal in the determinant basis, so the mask is not
    an approximation of the operator -- it is the same operator evaluated in
    its eigenbasis.  Measured agreement: bit-for-bit (max|diff| = 0.0) for the
    joint all-odd projector at norb = 10, 12, 14, and 1e-17 here, the rounding
    of the 0.5*(v + eig * L v) form when L expands into several terms.
    Measured cost of the joint all-odd projector, operator vs mask:
    8.8x at norb=10, 82x at norb=12, 61x at norb=14, with the ratio growing
    because the operator route is O(norb) passes over the vector and the mask
    route is one.  Use this for tests, the mask for production.
    """
    import ffsim
    import openfermion as of

    op = of.FermionOperator("", 1.0)
    for p in orbitals:
        term = of.FermionOperator("", 1.0)
        for q in (2 * int(p), 2 * int(p) + 1):
            term = term * (of.FermionOperator("", 1.0)
                           - 2 * of.FermionOperator(f"{q}^ {q}"))
        op = op * term
    ffop = ffsim.FermionOperator({})
    for term, coeff in op.terms.items():
        ffop += ffsim.FermionOperator(
            {tuple((bool(d), bool(i % 2), i // 2) for i, d in term): coeff})
    lin = ffsim.linear_operator(ffop, norb=norb, nelec=nelec)
    v = np.asarray(ci, dtype=float).ravel()
    return (0.5 * (v + eigenvalue * (lin @ v))).reshape(np.asarray(ci).shape)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    count = np.zeros_like(x)
    for _ in range(64):
        count += x & 1
        x >>= 1
        if not x.any():
            break
    return count


# --------------------------------------------------------------------------
# 3. Compression  (Sec. 8.2)
# --------------------------------------------------------------------------


def n_csf(omega: int, two_s: int) -> int:
    """Number of spin-S CSFs from omega singly occupied orbitals, 2S = two_s.

    Weyl/Paldus: C(omega, omega/2 - S) - C(omega, omega/2 - S - 1).
    """
    if two_s < 0 or omega < two_s or (omega - two_s) % 2:
        return 0
    k = (omega - two_s) // 2
    return comb(omega, k) - comb(omega, k - 1) if k >= 1 else comb(omega, 0)


def _iterate_parity_patterns(spec: SectorSpec, norb: int):
    """Yield (omega, multiplicity) for parity patterns allowed by spec.

    Enumerates 2^norb bit patterns.  Fine for norb <= ~24 (16.8M);
    raises above that rather than silently taking hours.
    """
    if norb > 24:
        raise NotImplementedError(
            f"pattern enumeration is 2^{norb}; supply an analytic counter for "
            "this sector family before running norb > 24")
    patterns = np.arange(1 << norb, dtype=np.int64)
    keep = np.ones(len(patterns), dtype=bool)
    for c in spec.constraints:
        bits = 0
        for p in c.orbitals:
            bits |= 1 << int(p)
        par = _popcount(patterns & bits) & 1
        keep &= par == (1 if c.eigenvalue == -1 else 0)
    omegas = _popcount(patterns[keep])
    vals, counts = np.unique(omegas, return_counts=True)
    return list(zip(vals.tolist(), counts.tolist()))


def sector_dimensions(spec: SectorSpec | None, norb: int,
                      nelec: tuple[int, int], two_s: int | None = None) -> dict:
    """Dimensions of the sector in the same N / M_S / S sector as the reference.

    Returns dict with:
      det_dim      determinants in the (na, nb) space inside the sector
      csf_dim      CSFs of total spin S = two_s/2 inside the sector
                   (only if two_s is given)
    Compression is the ratio of the unconstrained value to these; see
    `compression_ratio`.
    """
    na, nb = nelec
    nelec_tot = na + nb
    if spec is None or not spec.constraints:
        patterns = [(om, comb(norb, om)) for om in range(norb + 1)]
    else:
        patterns = _iterate_parity_patterns(spec, norb)

    det_dim = 0
    csf_dim = 0
    for omega, mult in patterns:
        rest = nelec_tot - omega
        if rest < 0 or rest % 2:
            continue
        ndoc = rest // 2
        if ndoc > norb - omega:
            continue
        n_cfg = mult * comb(norb - omega, ndoc)
        n_alpha_open = na - ndoc
        if 0 <= n_alpha_open <= omega:
            det_dim += n_cfg * comb(omega, n_alpha_open)
        if two_s is not None:
            csf_dim += n_cfg * n_csf(omega, two_s)
    out = {"det_dim": det_dim}
    if two_s is not None:
        out["csf_dim"] = csf_dim
    return out


def compression_ratio(spec: SectorSpec, norb: int, nelec: tuple[int, int],
                      two_s: int | None = None) -> dict:
    """C(P) = dim H / dim(P H), roadmap Eq. 13, in both det and CSF counting."""
    full = sector_dimensions(None, norb, nelec, two_s)
    sec = sector_dimensions(spec, norb, nelec, two_s)
    out = {
        "det_dim_full": full["det_dim"],
        "det_dim_sector": sec["det_dim"],
        "compression_det": (full["det_dim"] / sec["det_dim"]
                            if sec["det_dim"] else np.inf),
        "n_generators": spec.n_generators(),
    }
    if two_s is not None:
        out.update({
            "csf_dim_full": full["csf_dim"],
            "csf_dim_sector": sec["csf_dim"],
            "compression_csf": (full["csf_dim"] / sec["csf_dim"]
                                if sec["csf_dim"] else np.inf),
        })
    return out


# --------------------------------------------------------------------------
# 4. Local spin  (Sec. 8.1, full distribution)
# --------------------------------------------------------------------------


def local_s2_integrals(block: Sequence[int], norb: int):
    """(h1, g) such that  S_A^2 = sum_i h1_ii E_ii + (1/2) sum g_pqrs e_pq,rs.

    From Dobrautz et al. Appendix C:
        S_i^2     = (3/4)(E_ii - e_ii,ii)
        S_i . S_j = -(1/2) e_ij,ji - (1/4) e_ii,jj     (i != j)
    g is built symmetric under (pq)<->(rs) and (pq)->(qp),(rs)->(sr).
    """
    block = list(block)
    h1 = np.zeros((norb, norb))
    g = np.zeros((norb,) * 4)
    for i in block:
        h1[i, i] += 0.75
        g[i, i, i, i] += -1.5                     # (1/2)(-3/2) = -3/4
    for i in block:
        for j in block:
            if i == j:
                continue
            # ordered pairs: sum_{i!=j} S_i.S_j counts {i,j} twice, so the
            # per-unordered-pair coefficients are doubled relative to Eq. 48.
            g[i, j, j, i] += -1.0                 # -> -(1/2) e_ij,ji per (i,j)
            g[i, i, j, j] += -0.5                 # -> -(1/4) e_ii,jj per (i,j)
    return h1, g


def apply_local_s2(ci: np.ndarray, block: Sequence[int], norb: int,
                   nelec: tuple[int, int]) -> np.ndarray:
    h1, g = local_s2_integrals(block, norb)
    h2 = direct_nosym.absorb_h1e(h1, g, norb, nelec, 0.5)
    shape = ci.shape
    out = direct_nosym.contract_2e(h2, np.asarray(ci, dtype=float), norb, nelec)
    return np.asarray(out).reshape(shape)


def local_s2_expectation(ci: np.ndarray, block: Sequence[int], norb: int,
                         nelec: tuple[int, int]) -> float:
    v = np.asarray(ci).ravel()
    return float(v @ apply_local_s2(ci, block, norb, nelec).ravel())


def allowed_local_spins(block_size: int) -> list[float]:
    """All S_A values with possibly nonzero weight for a block of m orbitals.

    A block of m spatial orbitals can hold 0..2m electrons, so S_A runs over
    0, 1/2, ..., m/2 -- i.e. m+1 values.  Restricting this set is only valid
    when the state is known to live in a subspace that forbids the others
    (e.g. inside an all-odd sector every block orbital is singly occupied).
    """
    return [0.5 * k for k in range(block_size + 1)]


def local_spin_projector_apply(ci: np.ndarray, block: Sequence[int],
                               s_target: float, norb: int,
                               nelec: tuple[int, int],
                               s_values: Sequence[float] | None = None
                               ) -> np.ndarray:
    """P_A(S_A = s_target) |Psi>, applied as a product of spectral filters."""
    if s_values is None:
        s_values = allowed_local_spins(len(list(block)))
    lam = {s: s * (s + 1) for s in s_values}
    if s_target not in lam:
        raise ValueError(f"s_target {s_target} not in s_values")
    out = np.asarray(ci, dtype=float).copy()
    lt = lam[s_target]
    for s in s_values:
        if s == s_target:
            continue
        out = (apply_local_s2(out, block, norb, nelec) - lam[s] * out) / (lt - lam[s])
    return out


def local_spin_distribution(ci: np.ndarray, block: Sequence[int], norb: int,
                            nelec: tuple[int, int],
                            s_values: Sequence[float] | None = None
                            ) -> dict[float, float]:
    """w_A(S_A) for every allowed S_A (roadmap Eq. 12).  Weights sum to 1."""
    block = list(block)
    if s_values is None:
        s_values = allowed_local_spins(len(block))
    v = np.asarray(ci, dtype=float).ravel()
    out = {}
    for s in s_values:
        pv = local_spin_projector_apply(ci, block, s, norb, nelec, s_values)
        out[s] = float(v @ pv.ravel())
    return out


def moment_local_spin_distribution(ci: np.ndarray, block: Sequence[int],
                                   norb: int, nelec: tuple[int, int],
                                   s_values: Sequence[float] | None = None
                                   ) -> dict[float, float]:
    """Same quantity via moments of S_A^2 + Lagrange interpolation.

    Kept ONLY as an independent numerical cross-check of
    local_spin_distribution.  It becomes ill-conditioned as the block grows;
    test_conditioning quantifies where.
    """
    block = list(block)
    if s_values is None:
        s_values = allowed_local_spins(len(block))
    eig = np.array([s * (s + 1) for s in s_values])
    v = np.asarray(ci, dtype=float).ravel()
    moments = np.empty(len(eig))
    power = np.asarray(ci, dtype=float).copy()
    moments[0] = float(v @ power.ravel())
    for k in range(1, len(eig)):
        power = apply_local_s2(power, block, norb, nelec)
        moments[k] = float(v @ power.ravel())
    out = {}
    for j, s in enumerate(s_values):
        poly = np.poly(np.delete(eig, j))
        poly = poly / np.polyval(poly, eig[j])
        out[s] = float(np.dot(poly[::-1], moments))
    return out


def joint_local_spin_weight(ci: np.ndarray, blocks: Sequence[Sequence[int]],
                            targets: Sequence[float], norb: int,
                            nelec: tuple[int, int]) -> float:
    """<Psi| prod_A P_A(S_A = target_A) |Psi>  for disjoint blocks.

    Roadmap Eq. 7 with an arbitrary number of centres (needed for the
    CAS(20e,20o) tetramer, where there are four).
    """
    seen: set[int] = set()
    for b in blocks:
        s = set(b)
        if s & seen:
            raise ValueError("blocks must be disjoint for a joint projector")
        seen |= s
    out = np.asarray(ci, dtype=float).copy()
    for block, t in zip(blocks, targets):
        out = local_spin_projector_apply(out, block, t, norb, nelec)
    return float(np.asarray(ci).ravel() @ out.ravel())


def spin_correlation_matrix(dm1: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """C[i,j] = <s_i . s_j> from the spin-free 1- and 2-RDM.

    Dobrautz et al. Eqs. 39, 48, with pyscf's dm2[p,q,r,s] = <e_pq,rs>:
        C[i,i] = (3/4)(dm1[i,i] - dm2[i,i,i,i])
        C[i,j] = -(1/2) dm2[i,j,j,i] - (1/4) dm2[i,i,j,j]      (i != j)
    Cost is one RDM build, not norb^2 operator applications.
    """
    norb = dm1.shape[0]
    C = np.empty((norb, norb))
    for i in range(norb):
        for j in range(norb):
            if i == j:
                C[i, i] = 0.75 * (dm1[i, i] - dm2[i, i, i, i])
            else:
                C[i, j] = -0.5 * dm2[i, j, j, i] - 0.25 * dm2[i, i, j, j]
    return C


def local_spin_sector_dimensions(blocks, spins, norb, nelec, two_s=None):
    """Exact dimension of  prod_A P_A(S_A = spins[A])  in the fixed N / M_S / S
    sector, for DISJOINT blocks covering all norb orbitals.

    A block of m orbitals holding n electrons supports
        N(s; m, n) = sum_Omega C(m, Omega) C(m - Omega, (n - Omega)/2)
                     * n_csf(Omega, 2s)
    states of local spin s.  Summing over the ways to distribute the electrons
    among the blocks and then coupling the local spins gives the sector
    dimension.  Exact and combinatorial -- no operator applications.

    Returns {"det_dim": ..., "csf_dim": ...} matching sector_dimensions.
    """
    from itertools import product as _product

    blocks = [list(b) for b in blocks]
    if sorted(p for b in blocks for p in b) != list(range(norb)):
        raise ValueError("blocks must be disjoint and cover all orbitals")
    na, nb = nelec
    ntot = na + nb

    def n_states(s, m, n):
        if n < 0 or n > 2 * m:
            return 0
        tot = 0
        for omega in range(0, m + 1):
            rest = n - omega
            if rest < 0 or rest % 2:
                continue
            ndoc = rest // 2
            if ndoc > m - omega:
                continue
            tot += comb(m, omega) * comb(m - omega, ndoc) * n_csf(omega, int(round(2 * s)))
        return tot

    sizes = [len(b) for b in blocks]
    det_dim = csf_dim = 0
    ranges = [range(0, 2 * m + 1) for m in sizes]
    for occ in _product(*ranges):
        if sum(occ) != ntot:
            continue
        mult = 1
        for s, m, n in zip(spins, sizes, occ):
            mult *= n_states(s, m, n)
        if not mult:
            continue
        # couple the local spins; count M_S = (na-nb)/2 product states and
        # total-spin-S states by successive angular-momentum coupling
        ms_target = (na - nb) / 2.0
        dist = {0.0: 1}
        for s in spins:
            new = {}
            for cur, c in dist.items():
                k = int(round(2 * s))
                for i in range(k + 1):
                    m_s = s - i
                    new[cur + m_s] = new.get(cur + m_s, 0) + c
            dist = new
        det_dim += mult * dist.get(ms_target, 0)
        if two_s is not None:
            coupled = {0.0: 1}
            for s in spins:
                new = {}
                for cur, c in coupled.items():
                    lo, hi = abs(cur - s), cur + s
                    x = lo
                    while x <= hi + 1e-9:
                        new[x] = new.get(x, 0) + c
                        x += 1.0
                coupled = new
            csf_dim += mult * coupled.get(two_s / 2.0, 0)
    out = {"det_dim": det_dim}
    if two_s is not None:
        out["csf_dim"] = csf_dim
    return out


def local_spin_projector(blocks, spins, norb, nelec):
    """Return a callable psi -> prod_A P_A(S_A) psi, usable as `projector`."""
    def apply(ci):
        out = np.asarray(ci, dtype=float)
        for b, s in zip(blocks, spins):
            out = local_spin_projector_apply(out, list(b), s, norb, nelec)
        return out
    return apply


def _range_basis(project_fn, shape, dim, seed=0, tol=1e-8):
    """Orthonormal basis of range(P) for a projector of known dimension.

    P applied to `dim` generic random vectors spans the range with probability
    one; extra vectors are drawn if the Gram-Schmidt loses rank.
    """
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(10 * dim + 20):
        if len(basis) == dim:
            break
        v = project_fn(rng.standard_normal(shape)).ravel()
        for b in basis:
            v = v - (b @ v) * b
        n = np.linalg.norm(v)
        if n > tol:
            basis.append(v / n)
    if len(basis) != dim:
        raise RuntimeError(f"found {len(basis)} of {dim} range vectors")
    return np.array(basis).T


# --------------------------------------------------------------------------
# 5. Leakage (Sec. 8.4) and projected accuracy (Sec. 8.3)
# --------------------------------------------------------------------------


def _h_apply(ci, h1e, eri, norb, nelec):
    h2 = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    shape = np.asarray(ci).shape
    return np.asarray(direct_spin1.contract_2e(
        h2, np.asarray(ci, dtype=float), norb, nelec)).reshape(shape)


def _as_projector(projector):
    """Accept either a determinant mask or a callable projector."""
    if callable(projector):
        return projector
    mask = projector
    return lambda ci: project(ci, mask)


def leakage(ci, mask, h1e, eri, norb, nelec, energy_scale: float | None = None):
    """l_k(P) = || (1 - P) H P |Psi_k> ||^2   (roadmap Eq. 16), in Ha^2.

    If `energy_scale` (Ha) is given, also returns the normalized value
    l / energy_scale^2, which is the comparable-across-Hamiltonians number
    the roadmap asks for.  The roadmap requires a normalization but does not
    fix which scale; the caller must choose one and state it.  Candidates
    used in the literature: || H|Psi> - E|Psi> || (the state's own energy
    spread), the S=0..S_max ladder width, or |E_corr|.
    """
    P = _as_projector(mask)
    pv = P(ci)
    hpv = _h_apply(pv, h1e, eri, norb, nelec)
    resid = hpv - P(hpv)
    val = float(np.sum(resid ** 2))
    out = {"leakage_ha2": val}
    if energy_scale is not None:
        out["energy_scale_ha"] = float(energy_scale)
        out["leakage_normalized"] = val / float(energy_scale) ** 2
    return out


def projected_spectrum_general(project_fn, sector_dim, shape, h1e, eri, norb,
                               nelec, nroots=1, ecore=0.0,
                               reference_states=None, target_two_s=None,
                               seed=0):
    """Diagonalize P H P for a projector given as a callable of known dimension.

    Builds an orthonormal basis of range(P) explicitly, so this is exact but
    costs `sector_dim` projector applications; it is meant for the small,
    strongly contracted sectors.

    `sector_dim` MUST be the dimension in the (na, nb) DETERMINANT space, not
    the total-spin CSF dimension.  A local-spin sector with S_A = S_B = 5/2 is
    one-dimensional for each total spin S, but six-dimensional at M_S = 0
    because it spans S = 0..5; passing the CSF dimension returns one random
    vector from that six-dimensional space, which the reported <S^2> exposes
    immediately (20.21 where 0 was expected, in testing).  Use
    `target_two_s` to select the root of the intended total spin.
    """
    B = _range_basis(project_fn, shape, sector_dim, seed=seed)
    HB = np.array([_h_apply(B[:, i].reshape(shape), h1e, eri, norb,
                            nelec).ravel() for i in range(B.shape[1])]).T
    A = B.T @ HB
    A = 0.5 * (A + A.T)
    w, U = np.linalg.eigh(A)
    V = B @ U
    s2 = [float(V[:, i] @ apply_local_s2(V[:, i].reshape(shape), range(norb),
                                         norb, nelec).ravel())
          for i in range(V.shape[1])]
    if target_two_s is not None:
        c = (target_two_s / 2.0) * (target_two_s / 2.0 + 1.0)
        keep = [i for i in range(len(w)) if abs(s2[i] - c) < 1e-4]
        if not keep:
            raise RuntimeError(f"no root of 2S={target_two_s} in the sector; "
                               f"<S^2> found: {s2}")
        w, V, s2 = w[keep], V[:, keep], [s2[i] for i in keep]
    k = min(nroots, len(w))
    w, V, s2 = w[:k], V[:, :k], s2[:k]
    info = {"sector_dim": sector_dim, "s2": s2}
    if reference_states is not None:
        info["overlaps"] = np.array(
            [[abs(np.asarray(r).ravel() @ V[:, i]) ** 2 for i in range(k)]
             for r in reference_states])
    return w + ecore, V, info


def projected_spectrum(mask, h1e, eri, norb, nelec, nroots=1, ecore=0.0,
                       reference_states=None, target_two_s=None,
                       spin_penalty=1.0, dense_limit=400, seed=0):
    """Lowest eigenpairs of H_P = P H P restricted to range(P).

    Returns (energies, vectors, info); info["s2"] is the computed <S^2> of each
    root and info["overlaps"] the |<Psi_ref|Psi_proj>|^2 required by Sec. 8.3.

    target_two_s
        2S of the state the reference calculation targets.  REQUIRED for a
        meaningful comparison: at fixed (na, nb) the sector contains states of
        every total spin from |na-nb|/2 upwards, so the unconstrained lowest
        root of P H P is generally NOT the projected image of the reference
        state.  (In the H6 smoke test the S=0, 1 and 2 rows all returned the
        S=3 energy before this argument existed.)  When given, the operator is
            P [ H + lambda (S^2 - S(S+1))^2 ] P,
        zero on the target-spin block and positive elsewhere; the reported
        <S^2> must still be checked against S(S+1) by the caller.

    The Krylov space started from a vector in range(P) stays in range(P)
    because P H P maps range(P) into itself, so no spurious zero eigenvalues
    from the complement appear.  Sectors of dimension <= dense_limit are
    diagonalized densely (ARPACK cannot handle k >= N).
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    dim = mask.size
    n_sec = int(mask.sum())
    if n_sec == 0:
        raise ValueError("sector is empty")
    h2 = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    c_target = (None if target_two_s is None
                else (target_two_s / 2.0) * (target_two_s / 2.0 + 1.0))

    def h_in_sector(v):
        hv = np.asarray(direct_spin1.contract_2e(h2, v, norb, nelec))
        return project(hv.reshape(mask.shape), mask)

    def matvec(x):
        v = project(x.reshape(mask.shape), mask)
        hv = h_in_sector(v)
        if c_target is not None:
            sv = apply_local_s2(v, range(norb), norb, nelec) - c_target * v
            sv = apply_local_s2(sv, range(norb), norb, nelec) - c_target * sv
            hv = hv + spin_penalty * project(sv, mask)
        return hv.ravel()

    idx = np.flatnonzero(mask.ravel())
    if n_sec <= dense_limit:
        A = np.empty((n_sec, n_sec))
        e = np.zeros(dim)
        for j, col in enumerate(idx):
            e[col] = 1.0
            A[:, j] = matvec(e)[idx]
            e[col] = 0.0
        A = 0.5 * (A + A.T)
        w_all, V_all = np.linalg.eigh(A)
        k = min(nroots, n_sec)
        V = np.zeros((dim, k))
        V[idx, :] = V_all[:, :k]
    else:
        op = LinearOperator((dim, dim), matvec=matvec, dtype=float)
        rng = np.random.default_rng(seed)
        v0 = project(rng.standard_normal(mask.shape), mask).ravel()
        v0 /= np.linalg.norm(v0)
        k = min(nroots, n_sec - 1)
        w, V = eigsh(op, k=k, which="SA", v0=v0)
        V = V[:, np.argsort(w)]

    info = {"sector_dim": n_sec, "s2": []}
    energies = np.empty(V.shape[1])
    for i in range(V.shape[1]):
        vec = V[:, i].reshape(mask.shape)
        info["s2"].append(float(vec.ravel() @ apply_local_s2(
            vec, range(norb), norb, nelec).ravel()))
        # report the true energy, never the penalized one
        energies[i] = float(vec.ravel() @ h_in_sector(vec).ravel()) + ecore
    order = np.argsort(energies)
    energies, V = energies[order], V[:, order]
    info["s2"] = [info["s2"][i] for i in order]
    if reference_states is not None:
        info["overlaps"] = np.array(
            [[abs(np.asarray(r).ravel() @ V[:, i]) ** 2
              for i in range(V.shape[1])] for r in reference_states])
    return energies, V, info



# --------------------------------------------------------------------------
# 6. Spin-ladder models (Sec. 8.3: same fit to full and projected spectra)
# --------------------------------------------------------------------------


def fit_spin_models(energies: Sequence[float], s_values: Sequence[float],
                    s_a: float, s_b: float) -> dict:
    """Fit bilinear and bilinear-biquadratic two-site Heisenberg models.

    Bilinear      H = J S_A.S_B          -> E(S) = (J/2) S(S+1) + const
    Biquadratic   H = J'S_A.S_B + K(S_A.S_B)^2, eigenvalues per Dobrautz Eq. 18.
    Also returns omega, the relative average error per state of Dobrautz
    Eq. 19, for each model.

    NOTE: this is a TWO-SITE model.  It must not be applied to the four-centre
    tetramer/cubane benchmarks without replacing it by the appropriate
    multi-J or two-layer model.  `n_sites` is therefore not a free parameter
    here on purpose.
    """
    E = np.asarray(energies, float)
    S = np.asarray(s_values, float)
    dE = E - E[0]
    x = S * (S + 1)

    J = float(2.0 * (x @ dE) / (x @ x))                      # E = (J/2) x

    def bq_energy(Jp, K):
        sasb = 0.5 * (x - s_a * (s_a + 1) - s_b * (s_b + 1))
        e = Jp * sasb + K * sasb ** 2
        return e - e[0]

    A = np.vstack([bq_energy(1.0, 0.0), bq_energy(0.0, 1.0)]).T
    Jp, K = np.linalg.lstsq(A, dE, rcond=None)[0]

    def omega(model):
        span = abs(dE[-1] - dE[0])
        return 100.0 * np.sum(np.abs(dE - model)) / (len(dE) * span) if span else np.nan

    return {
        "J_bilinear": J,
        "omega_bilinear": omega(0.5 * J * x),
        "J_biquadratic": float(Jp),
        "K_biquadratic": float(K),
        "omega_biquadratic": omega(A @ np.array([Jp, K])),
        "dE": dE,
    }
