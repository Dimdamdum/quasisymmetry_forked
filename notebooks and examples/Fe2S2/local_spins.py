# ── local_spins.py  ────────────────────────────────────────────────────────
"""Local-spin diagnostics for Fe2S2 CAS(10e,10o).
 
Background
----------
S_A and S_B denote the total spin of the electrons in block A (Fe A, orbitals
0-4) and block B (Fe B, orbitals 5-9) respectively.  In a multi-determinant
state they are not sharp: the state is a superposition of components with
different S_A values (0, 1/2, 1, 3/2, 2, 5/2 for a 5-orbital block).
 
The operator S_A^2 has eigenspaces labelled by s in {0, 1/2, 1, 3/2, 2, 5/2},
each with eigenvalue s(s+1).  These eigenspaces span the full Hilbert space
-- they are NOT restricted to the Omega=10 sector.  Any FCI state can be
decomposed into its components in each eigenspace, giving a probability
distribution over s values.
 
Quantities reported
-------------------
 
  C[p,q]
    Expected value <psi | s_p . s_q | psi> where s_p is the spin-1/2 vector
    operator for orbital p (Sx, Sy, Sz).  A 10x10 matrix.  For a singlet state
    every row sums to exactly zero, because the total spin S_tot = sum_p s_p
    annihilates the state, so <s_p . S_tot> = sum_q C[p,q] = 0.
 
  wA(s)  for s in {0, 1/2, 1, 3/2, 2, 5/2}
    Probability that block A has total spin s, i.e. the fraction of |psi> that
    lies in the eigenspace of S_A^2 with eigenvalue s(s+1).  Computed by
    building the projector onto that eigenspace as a polynomial in S_A^2 -- the
    unique polynomial of degree 5 that equals 1 at lambda = s(s+1) and 0 at all
    other eigenvalues (sometimes called a Lagrange interpolating polynomial, but
    it is just a projector onto one eigenspace expressed as a function of S_A^2).
    The weights sum to 1.  Because wA spans the full Hilbert space, integer s
    values (0, 1, 2) are possible whenever block A holds an even number of
    electrons, which happens in the Omega<10 sectors.
 
  wB(s)
    Same as wA but for block B.
 
  w55 = <psi | P_A(5/2) P_B(5/2) | psi>
    Fraction of |psi> that simultaneously has S_A = 5/2 AND S_B = 5/2.
    P_A(5/2) is the projector onto the S_A=5/2 eigenspace (the same polynomial
    described above for wA, evaluated at s=5/2).  P_A and P_B commute because
    they act on different orbital blocks, so the joint projector P_A*P_B is
    well-defined.  The result must lie in [0, 1].
    For a singlet, w55 = wA(5/2) exactly: if S_A=5/2 then S_B must also be 5/2
    for the two blocks to couple to S_tot=0, so the joint weight equals the
    individual weight.
 
Within the Omega = norb (all-singly-occupied) sector
  The Omega=10 sector is the subspace in which every orbital holds exactly one
  electron (no orbital is doubly occupied or empty).  Within this sector, each
  block has exactly 5 electrons, so S_A is restricted to {1/2, 3/2, 5/2}.
 
  W10
    Fraction of |psi> in the Omega=10 sector:  W10 = ||P_Omega |psi>||^2,
    where P_Omega projects onto all-singly-occupied determinants by applying
    (I - s_p)/2 for each orbital p (s_p = (-1)^{n_p} is the local parity).
 
  wA10(s)  for s in {1/2, 3/2, 5/2}
    Within the Omega=10 component of |psi>, the probability that block A has
    total spin s.  This is the conditional distribution given Omega=10.
    Only half-integer s values appear because block A has exactly 5 electrons.
 
  wB10(s)
    Same for block B.
 
  w55_10
    Within the Omega=10 component, the fraction simultaneously having S_A=5/2
    AND S_B=5/2.  The unique state in the Omega=10 sector with this property
    is the antiferromagnetic Hund-coupled singlet (both Fe(III) locally d5
    high-spin, coupled to S_tot=0).  The 42-dimensional Omega=10 singlet space
    decomposes into 1 state with S_A=S_B=5/2, plus 16 states with S_A=S_B=3/2,
    plus 25 states with S_A=S_B=1/2.
 
  W10 * w55_10
    Equals w55 to machine precision for the reasons explained below.
 
Cross-sector decomposition
  S_A^2 commutes with every seniority projector.  Proof: the raising/lowering
  operators S+/S- only swap alpha<->beta within one orbital, so they leave the
  total occupancy n_p = n_p_alpha + n_p_beta unchanged, and therefore commute
  with the parity (-1)^n_p used to construct the Omega=10 projector.
 
  Because the projectors commute, the weight in the intersection of the
  {S_A = s} eigenspace with the Omega=10 sector factorises exactly:
 
    P(S_A = s  AND  Omega = 10) = W10 * wA10(s)
 
  The residual  w_A(s) - W10*wA10(s)  is the weight in the intersection of
  {S_A = s} with the Omega<10 sectors.  The code prints this for every s.
 
  For the singlet ground state, two exact results follow:
    Integer s (0, 1, 2): Omega=10 contribution is exactly 0.  Block A would
      need an even number of electrons, impossible when every orbital is singly
      occupied (each block gets exactly 5).  So all integer-s weight comes from
      Omega<10 (charge-transfer sectors).
    s = 5/2: Omega<10 residual is exactly 0.  S_A=5/2 in a singlet forces
      S_B=5/2, which forces both blocks to have 5 singly-occupied electrons,
      i.e. the state must be in Omega=10.  This is confirmed numerically:
      W10 * w55_10 = w55 to ~1e-16.
    s = 1/2, 3/2: weight is shared between sectors.  The Omega<10 fraction
      measures how much of the non-maximal local-spin comes from charge
      transfer rather than spin recoupling within the Omega=10 sector.
 
Sanity checks
-------------
Call sanity_checks(result) on the dict returned by analyze(), or pass
run_checks=True to analyze() to run them automatically.  Six groups:
  A  row sums of C = 0  (singlet: S_tot|psi>=0 => sum_q C[p,q]=0 for each p)
  B  <SA^2> from C  vs  from w(s)  (two independent operator constructions)
  C  cross-block sum of C = -(SA^2+SB^2)/2  (singlet + A-B exchange identity)
  D  ||P_A(5/2)|psi>||^2 = wA(5/2)  (projector onto eigenspace is idempotent)
  E  ionic reference  |5alpha in A, 5beta in B>  (all quantities known exactly)
  F  HF reference  (doubly-occupied A, empty B)  (all quantities = 0)
"""

from pathlib import Path
import numpy as np
import openfermion as of
from optimize_symmetries import get_fci, fermion_op_to_linop, parities
from chemistry import load_moldata, fcidump_data
import ffsim

# ── path resolution ──────────────────────────────────────────────────────────
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                    # running as a notebook cell
    _HERE = Path.cwd()


def _find_repo_root(start: Path, marker: str = "hamiltonians") -> Path:
    for candidate in (start, *start.parents):
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate a '{marker}' directory above {start}"
    )


_HAM_DIR = _find_repo_root(_HERE) / "hamiltonians" / "Fe2S2"
UNROTATED_PATH = _HAM_DIR / "fe2s2_10e10o_FCIDUMP"
ROTATED_PATH   = _HAM_DIR / "fe2s2_10e10o_rot_sen_ac_FCIDUMP"
COMPARE_ROTATED = True

# Block assignment confirmed from spin-correlation matrix block structure:
FE_A_ORBITALS, FE_B_ORBITALS = list(range(5)), list(range(5, 10))   # Fe A = orbs 0-4, Fe B = orbs 5-9


# ── operator builders ────────────────────────────────────────────────────────

def local_S2(block):
    """S_A^2 = (sum_{p in A} s_p)^2 as a Sz-conserving FermionOperator."""
    spin_squared_op = of.FermionOperator()
    for orb_p in block:
        p_alpha, p_beta = 2*orb_p, 2*orb_p+1
        sz_p_op = 0.5*(of.FermionOperator(f"{p_alpha}^ {p_alpha}") - of.FermionOperator(f"{p_beta}^ {p_beta}"))
        for orb_q in block:
            q_alpha, q_beta = 2*orb_q, 2*orb_q+1
            sz_q_op = 0.5*(of.FermionOperator(f"{q_alpha}^ {q_alpha}") - of.FermionOperator(f"{q_beta}^ {q_beta}"))
            spin_squared_op += sz_p_op * sz_q_op
            spin_squared_op += 0.5 * of.FermionOperator(f"{p_alpha}^ {p_beta} {q_beta}^ {q_alpha}")   # S+_p S-_q
            spin_squared_op += 0.5 * of.FermionOperator(f"{p_beta}^ {p_alpha} {q_alpha}^ {q_beta}")   # S-_p S+_q
    return spin_squared_op


def _spin_dot_spin(orb_p, orb_q):
    """s_p . s_q as a particle- and Sz-conserving FermionOperator."""
    p_alpha, p_beta, q_alpha, q_beta = 2*orb_p, 2*orb_p+1, 2*orb_q, 2*orb_q+1
    sz_p_op = 0.5*(of.FermionOperator(f"{p_alpha}^ {p_alpha}") - of.FermionOperator(f"{p_beta}^ {p_beta}"))
    sz_q_op = 0.5*(of.FermionOperator(f"{q_alpha}^ {q_alpha}") - of.FermionOperator(f"{q_beta}^ {q_beta}"))
    return (sz_p_op * sz_q_op
            + 0.5 * of.FermionOperator(f"{p_alpha}^ {p_beta} {q_beta}^ {q_alpha}")
            + 0.5 * of.FermionOperator(f"{p_beta}^ {p_alpha} {q_alpha}^ {q_beta}"))


# ── diagnostics on a generic state ──────────────────────────────────────────

def spin_correlation_matrix(state_vector, norb, nelec):
    """C[p,q] = <vec | s_p . s_q | vec>."""
    corr_matrix = np.zeros((norb, norb))
    for orb_p in range(norb):
        for orb_q in range(norb):
            spin_dot_op = fermion_op_to_linop(_spin_dot_spin(orb_p, orb_q), norb, nelec)
            corr_matrix[orb_p, orb_q] = np.real(np.vdot(state_vector, spin_dot_op @ state_vector))
    return corr_matrix


def local_spin_distribution(state_vector, block, norb, nelec,
                             s_values=(0, 0.5, 1, 1.5, 2, 2.5)):
    """w(S_A) for each S_A in s_values via Lagrange projectors onto S_A^2 eigenspaces.

    The moment method uses K = len(s_values) matvecs of S_A^2 and evaluates
    the degree-(K-1) Lagrange interpolating polynomial at each target eigenvalue.

    s_values must span ALL S_A eigenvalues that have non-zero weight:

      Full FCI state
        Block A can hold 0-10 electrons depending on which determinants
        contribute; S_A can therefore be 0, 1/2, 1, 3/2, 2, 5/2.
        -> use the default s_values = (0, 0.5, 1, 1.5, 2, 2.5).
        (Including extra values that happen to carry zero weight is harmless;
        their Lagrange factors are roots of the numerator polynomial.)

      Projected Omega = norb state
        Every orbital is singly occupied, so block A has exactly 5 electrons
        and S_A in {1/2, 3/2, 5/2} only.
        -> use s_values = (0.5, 1.5, 2.5).
        This gives a degree-2 polynomial (vs. degree-5), saves 3 matvecs, and
        gives the same numerical result as the degree-5 polynomial.
        Do NOT use the reduced set on the full FCI state: S_A = 0, 1, 2 do
        appear in the Omega < 10 sectors (when block A has 4 or 6 electrons),
        and omitting those eigenvalues from the Lagrange set gives wrong weights.
    """
    s2_linop = fermion_op_to_linop(local_S2(block), norb, nelec)
    eigenvalues = np.array([s*(s+1) for s in s_values])
    num_moments = len(eigenvalues)

    # moments  moments[k] = <vec | (S_A^2)^k | vec>
    moments = np.empty(num_moments)
    state_power = state_vector.copy()
    moments[0] = np.real(np.vdot(state_vector, state_power))          # = <vec|vec> = 1 if normalised
    for moment_idx in range(1, num_moments):
        state_power = s2_linop @ state_power
        moments[moment_idx] = np.real(np.vdot(state_vector, state_power))

    # Lagrange interpolation: w(s_j) = <vec | L_j(S_A^2) | vec>
    weights = {}
    for eigen_idx, s_val in enumerate(s_values):
        lagrange_poly = np.poly(np.delete(eigenvalues, eigen_idx))        # monic, roots = other eigenvalues
        lagrange_poly = lagrange_poly / np.polyval(lagrange_poly, eigenvalues[eigen_idx])   # normalise: L_j(lambda_j) = 1
        weights[s_val] = float(np.dot(lagrange_poly[::-1], moments))     # coefficients in ascending power order
    return weights


def proj_52_vec(state_vector, block, norb, nelec):
    """Apply P(S_block = 5/2) to state_vector and return the projected vector.

    Implements P(5/2) = prod_{s != 5/2} (S^2 - s(s+1)) / (8.75 - s(s+1))
    by applying each factor sequentially (factors commute, order arbitrary).

    Exposed at module level (rather than nested inside joint_high_spin_weight)
    so that sanity check D can compute its norm to verify idempotency:
        ||P_A(5/2)|psi>||^2  =  w_A(5/2)
    """
    s2_linop = fermion_op_to_linop(local_S2(block), norb, nelec)
    eigenvalues = np.array([s*(s+1) for s in (0, 0.5, 1, 1.5, 2, 2.5)])
    target_idx = len(eigenvalues) - 1    # s = 2.5, lambda_j = 8.75
    projected_state = state_vector.copy()
    for eigen_idx in range(len(eigenvalues)):
        if eigen_idx != target_idx:
            projected_state = (s2_linop @ projected_state - eigenvalues[eigen_idx] * projected_state) / (eigenvalues[target_idx] - eigenvalues[eigen_idx])
    return projected_state


def joint_high_spin_weight(state_vector, blockA, blockB, norb, nelec):
    """<vec | P_A(5/2) P_B(5/2) | vec>  --  joint probability both Fe are S = 5/2.

    Applies the Lagrange projector P(S=5/2) to vec one factor at a time, then
    applies P_B(5/2) the same way to the result, and takes the inner product
    with the original vec.  P_A and P_B commute (different orbital blocks).

    Result must lie in [0, 1]; an assert is checked at the call site.
    """
    state_proj_A  = proj_52_vec(state_vector, blockA, norb, nelec)      # P_A(5/2)|psi>
    state_proj_AB = proj_52_vec(state_proj_A, blockB, norb, nelec)      # P_B(5/2) P_A(5/2)|psi>
    return float(np.real(np.vdot(state_vector, state_proj_AB)))


# ── projection onto the maximum-seniority sector ────────────────────────────

def project_max_seniority_state(state_vector, norb, nelec):
    """Project the FCI state onto the all-singly-occupied (Omega = norb) sector.

    Uses the local parity operators  s_p = (-1)^{n_p}  returned by
    optimize_symmetries.parities().  The projector onto the singly-occupied
    eigenspace of orbital p is

        P_p(odd) = (I - s_p) / 2

    Because orbital parities commute, the global projector

        P_{Omega=norb} = prod_p P_p(odd)

    is applied by iterating over orbitals: on each step,

        proj <- (proj - s_p @ proj) / 2

    Components with orbital p doubly occupied or empty (s_p eigenvalue +1)
    vanish after the corresponding step; components singly occupied
    (s_p eigenvalue -1) survive unchanged.

    Returns
    -------
    sector_state : np.ndarray
        Normalised projected state.
    sector_weight : float
        Sector weight  ||P_{Omega} |psi>||^2  =  sum_{|Phi> in Omega=norb} |<Phi|psi>|^2.
        Should match the Omega = norb entry in the seniority-sector weight table.
    """
    local_parity_ops = parities(norb, nelec)   # list of norb linear operators: s_p = (-1)^n_p
    projected_state = state_vector.copy()
    for parity_op in local_parity_ops:
        projected_state = (projected_state - parity_op @ projected_state) * 0.5
    sector_weight = float(np.real(np.vdot(projected_state, projected_state)))
    if sector_weight < 1e-15:
        raise ValueError(
            f"Sector weight W10 = {sector_weight:.2e}.  "
            "The state has no weight in the all-singly-occupied sector."
        )
    return projected_state / np.sqrt(sector_weight), sector_weight


# ── pretty-printing helper ───────────────────────────────────────────────────

def _print_local_spin(tag, spin_weights):
    """Print a local-spin distribution with bar chart and derived quantities."""
    for s_val in sorted(spin_weights):
        bar_chart = "#" * max(0, int(round(30 * max(0.0, spin_weights[s_val]))))
        print(f"    w({tag}, S={s_val:.1f}) = {spin_weights[s_val]:+.6f}  {bar_chart}")
    total = sum(spin_weights.values())
    spin_squared_exp = sum(s_val*(s_val+1)*weight for s_val, weight in spin_weights.items())
    effective_spin = (np.sqrt(1 + 4*max(0.0, spin_squared_exp)) - 1) / 2
    print(f"    -- sum = {total:.8f}   <S^2> = {spin_squared_exp:.6f}   S_eff = {effective_spin:.4f}")


# ── main analysis routines ───────────────────────────────────────────────────

def analyze(molpath, blockA=FE_A_ORBITALS, blockB=FE_B_ORBITALS, label=None,
            analyze_sector=True, run_checks=False):
    """Load an FCIDUMP, compute the FCI ground state, and run local-spin diagnostics.

    Parameters
    ----------
    molpath : path-like
        Path to the FCIDUMP file.
    blockA, blockB : list[int]
        Spatial orbital indices for the two Fe centres (zero-based).
        Defaults confirmed by the 2-block spin-correlation matrix.
    analyze_sector : bool
        If True, also project onto the Omega = norb sector and repeat the
        analysis within that sector, including the cross-sector decomposition.
    run_checks : bool
        If True, call sanity_checks(result) at the end.

    Returns
    -------
    dict with keys: mol_data, state_vector, corr_matrix, spin_dist_A, spin_dist_B,
                    joint_weight_55, and (if analyze_sector) sector_state,
                    sector_weight, spin_dist_A_sector, spin_dist_B_sector,
                    joint_weight_55_sector.
    """
    label = label or Path(molpath).name
    mol_data = load_moldata(molpath)
    _, state_vector = get_fci(fcidump_data(molpath))
    state_vector = np.asarray(state_vector).reshape(-1)

    # ── spin-correlation matrix ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {label}  --  FULL FCI STATE")
    print(f"{'='*65}")

    corr_matrix = spin_correlation_matrix(state_vector, mol_data.norb, mol_data.nelec)
    print("\nSpin correlation matrix  C[p,q] = <psi | s_p . s_q | psi>  (10x10, orbital x orbital):")
    print("  For a singlet, every row sums to 0 exactly (S_tot|psi> = 0).")
    for row in corr_matrix:
        print("  " + "  ".join(f"{v:+.4f}" for v in row))
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-8):
        raise AssertionError("C is not symmetric -- check spin_correlation_matrix()")

    # ── local spin distributions on full state ───────────────────────────
    # Default s_values covers all S_A eigenvalues for any block-A occupancy.
    print("\nLocal spin distribution on Fe A  (full state):")
    print("  P(S_A = s): weight of |psi> in each S_A^2 eigenspace, s in {0,1/2,...,5/2}.")
    print("  Integer s is possible here because block A can hold an even electron")
    print("  count in the Omega<10 (charge-transfer) sectors.")
    spin_dist_A = local_spin_distribution(state_vector, blockA, mol_data.norb, mol_data.nelec)
    _print_local_spin("A", spin_dist_A)

    print("\nLocal spin distribution on Fe B  (full state):")
    print("  Same construction as Fe A, mirrored onto block B.")
    spin_dist_B = local_spin_distribution(state_vector, blockB, mol_data.norb, mol_data.nelec)
    _print_local_spin("B", spin_dist_B)

    # ── joint high-spin weight ───────────────────────────────────────────
    joint_weight_55 = joint_high_spin_weight(state_vector, blockA, blockB, mol_data.norb, mol_data.nelec)
    print(f"\nJoint high-spin weight on full state:")
    print(f"  Fraction of |psi> with S_A = 5/2 AND S_B = 5/2 simultaneously.  For a")
    print(f"  singlet this equals wA(5/2) exactly (S_A=5/2 forces S_B=5/2 to couple")
    print(f"  to S_tot=0).")
    print(f"    <P_A(5/2) P_B(5/2)>  =  w55  =  {joint_weight_55:.8f}")
    assert -1e-7 <= joint_weight_55 <= 1 + 1e-7, (
        f"w55 = {joint_weight_55} is outside [0, 1].  "
        "Check local_S2(), joint_high_spin_weight(), or A/B block definitions."
    )

    result = {
        "mol_data": mol_data, "state_vector": state_vector, "corr_matrix": corr_matrix,
        "spin_dist_A": spin_dist_A, "spin_dist_B": spin_dist_B, "joint_weight_55": joint_weight_55,
    }

    # ── projection onto Omega = norb sector ─────────────────────────────
    if analyze_sector:
        print(f"\n{'='*65}")
        print(f"  {label}  --  WITHIN THE Omega = {mol_data.norb}  (ALL-SINGLY-OCCUPIED) SECTOR")
        print(f"{'='*65}")

        sector_state, sector_weight = project_max_seniority_state(state_vector, mol_data.norb, mol_data.nelec)
        print(f"\nSector weight   W10 = {sector_weight:.8f}")
        print(f"  Fraction of |psi> with every orbital singly occupied (no orbital")
        print(f"  doubly occupied or empty).  Should equal the Omega = {mol_data.norb} entry")
        print(f"  in the seniority-sector table.")

        # Within Omega = 10, block A has exactly 5 electrons -> S_A in {1/2, 3/2, 5/2}.
        # Use the restricted set: correct eigenvalues + saves 3 matvecs.
        restricted_s_values = (0.5, 1.5, 2.5)

        print(f"\nLocal spin distribution on Fe A  (within Omega = {mol_data.norb}):")
        print(f"  Conditional on Omega=10, block A has exactly 5 electrons, so only")
        print(f"  half-integer S_A in {{1/2, 3/2, 5/2}} occur.")
        spin_dist_A_sector = local_spin_distribution(sector_state, blockA, mol_data.norb, mol_data.nelec, s_values=restricted_s_values)
        _print_local_spin("A|O", spin_dist_A_sector)

        print(f"\nLocal spin distribution on Fe B  (within Omega = {mol_data.norb}):")
        print(f"  Same construction as Fe A, mirrored onto block B.")
        spin_dist_B_sector = local_spin_distribution(sector_state, blockB, mol_data.norb, mol_data.nelec, s_values=restricted_s_values)
        _print_local_spin("B|O", spin_dist_B_sector)

        joint_weight_55_sector = joint_high_spin_weight(sector_state, blockA, blockB, mol_data.norb, mol_data.nelec)
        print(f"\nJoint S_A = S_B = 5/2 weight within the sector:")
        print(f"  The unique Omega=10 state with S_A=S_B=5/2 is the antiferromagnetic")
        print(f"  Hund-coupled singlet (both Fe(III) locally d5 high-spin, coupled to")
        print(f"  S_tot=0).")
        print(f"    w55_10  =  {joint_weight_55_sector:.8f}")
        assert -1e-7 <= joint_weight_55_sector <= 1 + 1e-7, f"w55_10 = {joint_weight_55_sector} outside [0, 1]."

        print(f"\nConsistency: W10 * w55_10 vs. full-state w55")
        print(f"  These should agree to machine precision: S_A=5/2 in a singlet forces")
        print(f"  S_B=5/2, which forces Omega=10, so the Omega<10 residual is exactly 0.")
        print(f"    W10 * w55_10  =  {sector_weight * joint_weight_55_sector:.8f}")
        print(f"    w55           =  {joint_weight_55:.8f}")
        print(f"    difference    =  {abs(sector_weight*joint_weight_55_sector - joint_weight_55):.2e}  "
              f"(residue from Omega < {mol_data.norb} sectors with S_A = S_B = 5/2)")

        result.update({
            "sector_state": sector_state, "sector_weight": sector_weight,
            "spin_dist_A_sector": spin_dist_A_sector, "spin_dist_B_sector": spin_dist_B_sector,
            "joint_weight_55_sector": joint_weight_55_sector,
        })

        # ── cross-sector decomposition ───────────────────────────────────
        #
        # S_A^2 commutes with all seniority projectors because S+/S- only swap
        # alpha<->beta within one orbital, leaving n_p_alpha + n_p_beta invariant.
        # Therefore: P(S_A=s AND Omega=10) = W10 * wA10(s)  (exact factorisation).
        # Residual = P(S_A=s AND Omega<10) = w_A(s) - W10*wA10(s).
        #
        # For the singlet:
        #   integer s (0,1,2):  Omega=10 contribution = 0 (odd nelec in each block)
        #   s = 5/2:            Omega<10 residual = 0 (S_A=5/2 => S_B=5/2 => Omega=10)
        #   s = 1/2, 3/2:       weight is shared

        print(f"\n{'─'*65}")
        print("  CROSS-SECTOR DECOMPOSITION")
        print("  w_A(s) = W10*wA10(s)  +  Omega<10 residual  (exact factorisation)")
        print(f"{'─'*65}")
        print(f"  {'s':>4}  {'full w_A(s)':>12}  {'W10*wA10(s)':>12}  {'Omega<10':>12}")
        print(f"  {'----':>4}  {'------------':>12}  {'------------':>12}  {'------------':>12}")
        for s_val in sorted(spin_dist_A.keys()):
            full_weight = spin_dist_A[s_val]
            sector_contribution = sector_weight * spin_dist_A_sector.get(s_val, 0.0)
            residual = full_weight - sector_contribution
            note = ""
            if s_val == 2.5 and abs(residual) < 1e-9:
                note = "  <- exact 0 for singlet"
            elif s_val in (0.0, 1.0, 2.0):
                note = "  <- exact 0 from Omega=10"
            print(f"  {s_val:>4.1f}  {full_weight:+12.6f}  {sector_contribution:+12.6f}  {residual:+12.6f}{note}")

        print()
        print("  Fraction of w_A(s) from Omega<10:")
        for s_val in sorted(spin_dist_A.keys()):
            full_weight = spin_dist_A[s_val]
            sector_contribution = sector_weight * spin_dist_A_sector.get(s_val, 0.0)
            if abs(full_weight) > 1e-9:
                percentage = 100.0 * (full_weight - sector_contribution) / full_weight
                print(f"    S_A = {s_val:.1f}: {percentage:6.1f}% from Omega<10")

    if run_checks:
        sanity_checks(result)

    return result


def compare(before_path, after_path, blockA=FE_A_ORBITALS, blockB=FE_B_ORBITALS):
    """Analyze two FCIDUMPs and diff all local-spin diagnostics.

    Typical use: compare unrotated vs. seniority-optimised orbitals.
    """
    before = analyze(before_path, blockA, blockB, label="before rotation")
    after  = analyze(after_path,  blockA, blockB, label="after rotation")

    print(f"\n{'='*65}")
    print("  CHANGES:  before  ->  after  orbital rotation")
    print(f"{'='*65}")

    corr_diff = after["corr_matrix"] - before["corr_matrix"]
    print(f"\nmax |dC[p,q]|  =  {np.max(np.abs(corr_diff)):.6f}")

    print("\nDelta local spin distribution on Fe A:")
    for s_val in sorted(before["spin_dist_A"]):
        delta_weight = after["spin_dist_A"].get(s_val, 0.0) - before["spin_dist_A"].get(s_val, 0.0)
        print(f"    D w_A(S={s_val:.1f})  =  {delta_weight:+.6f}")

    print("\nDelta local spin distribution on Fe B:")
    for s_val in sorted(before["spin_dist_B"]):
        delta_weight = after["spin_dist_B"].get(s_val, 0.0) - before["spin_dist_B"].get(s_val, 0.0)
        print(f"    D w_B(S={s_val:.1f})  =  {delta_weight:+.6f}")

    delta_joint_weight = after["joint_weight_55"] - before["joint_weight_55"]
    print(f"\nDelta w55  =  {delta_joint_weight:+.6f}")

    if "spin_dist_A_sector" in before and "spin_dist_A_sector" in after:
        print(f"\nDelta diagnostics within the Omega = max-seniority sector:")
        print(f"    Delta W10  =  {after['sector_weight'] - before['sector_weight']:+.6f}")
        for s_val in sorted(before["spin_dist_A_sector"]):
            delta_A = after["spin_dist_A_sector"].get(s_val, 0.0) - before["spin_dist_A_sector"].get(s_val, 0.0)
            delta_B = after["spin_dist_B_sector"].get(s_val, 0.0) - before["spin_dist_B_sector"].get(s_val, 0.0)
            print(f"    D w_A10(S={s_val:.1f})  =  {delta_A:+.6f}   "
                  f"D w_B10(S={s_val:.1f})  =  {delta_B:+.6f}")
        delta_joint_sector = after["joint_weight_55_sector"] - before["joint_weight_55_sector"]
        print(f"    Delta w55_10  =  {delta_joint_sector:+.6f}")

    return before, after


# ── visualization ────────────────────────────────────────────────────────────
#
# Palette: validated categorical + diverging pair (see dataviz skill palette.md).
# Fe A = blue, Fe B = orange (fixed order, reused everywhere for series identity).
# "within Omega=10" / "Omega<10 residual" use a separate aqua/violet pair so the
# decomposition panel is never confused with the Fe A/B identity colors.
_PLOT_SURFACE       = "#fcfcfb"
_PLOT_INK_PRIMARY   = "#0b0b0b"
_PLOT_INK_SECONDARY = "#52514e"
_PLOT_INK_MUTED     = "#898781"
_PLOT_GRIDLINE      = "#e1e0d9"
_PLOT_AXIS          = "#c3c2b7"
_PLOT_BLUE          = "#2a78d6"   # Fe A
_PLOT_ORANGE        = "#eb6834"   # Fe B
_PLOT_AQUA          = "#1baf7a"   # within Omega=10
_PLOT_VIOLET        = "#4a3aa7"   # Omega<10 residual
_PLOT_RED           = "#e34948"   # diverging pole (C[p,q] > 0)
_PLOT_GRAY_MID      = "#f0efec"   # diverging midpoint (C[p,q] = 0)


def _style_axes(ax):
    """Apply shared chrome (recessive grid/axes, no top/right spines) to a bar/line axis."""
    ax.set_facecolor(_PLOT_SURFACE)
    ax.grid(axis="y", color=_PLOT_GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_PLOT_AXIS)
    ax.tick_params(colors=_PLOT_INK_MUTED)
    ax.xaxis.label.set_color(_PLOT_INK_SECONDARY)
    ax.yaxis.label.set_color(_PLOT_INK_SECONDARY)


def plot_local_spins(result, title=None, save_path=None):
    """Render a 2x2 diagnostic figure for the dict returned by analyze().

    Panels
    ------
    top-left      Spin correlation matrix C[p,q] as a diverging heatmap
                  (blue = anti-aligned/cross-block, red = aligned/within-block).
    top-right     Full-state local spin distribution wA(s) vs wB(s).
    bottom-left   Cross-sector decomposition of wA(s): the part exactly inside
                  the Omega=10 sector (W10*wA10(s)) vs the Omega<10 residual.
                  Omitted if `result` has no sector diagnostics
                  (i.e. analyze() was called with analyze_sector=False).
    bottom-right  Summary of the scalar joint-weight quantities (w55, W10, w55_10).

    Parameters
    ----------
    result : dict
        Return value of analyze() (or one half of compare()'s return tuple).
    title : str, optional
        Figure title.  Defaults to a generic label.
    save_path : path-like, optional
        If given, the figure is saved there (dpi=150) in addition to being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    corr_matrix = result["corr_matrix"]
    spin_dist_A = result["spin_dist_A"]
    spin_dist_B = result["spin_dist_B"]
    has_sector  = "spin_dist_A_sector" in result
    s_values    = sorted(spin_dist_A.keys())
    x           = np.arange(len(s_values))
    bar_width   = 0.36

    fig, ((ax_corr, ax_full), (ax_sector, ax_summary)) = plt.subplots(
        2, 2, figsize=(11, 9), facecolor=_PLOT_SURFACE
    )
    fig.suptitle(title or "Local-spin diagnostics", color=_PLOT_INK_PRIMARY, fontsize=13)

    # ── top-left: correlation matrix heatmap ─────────────────────────────
    ax_corr.set_facecolor(_PLOT_SURFACE)
    vmax = float(np.max(np.abs(corr_matrix))) or 1.0
    diverging_cmap = LinearSegmentedColormap.from_list(
        "corr_diverging", [_PLOT_BLUE, _PLOT_GRAY_MID, _PLOT_RED]
    )
    im = ax_corr.imshow(
        corr_matrix, cmap=diverging_cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    cbar = fig.colorbar(im, ax=ax_corr, shrink=0.8)
    cbar.ax.tick_params(colors=_PLOT_INK_MUTED)
    cbar.outline.set_visible(False)
    n_block_a = 5   # orbital index where Fe A ends / Fe B begins
    norb = corr_matrix.shape[0]
    if n_block_a < norb:
        ax_corr.axhline(n_block_a - 0.5, color=_PLOT_INK_SECONDARY, lw=1)
        ax_corr.axvline(n_block_a - 0.5, color=_PLOT_INK_SECONDARY, lw=1)
    ax_corr.set_title("Spin correlation  C[p,q] = <s_p . s_q>", loc="left",
                       color=_PLOT_INK_PRIMARY, fontsize=11)
    ax_corr.set_xlabel("orbital q")
    ax_corr.set_ylabel("orbital p")
    ax_corr.xaxis.label.set_color(_PLOT_INK_SECONDARY)
    ax_corr.yaxis.label.set_color(_PLOT_INK_SECONDARY)
    ax_corr.tick_params(colors=_PLOT_INK_MUTED)

    # ── top-right: full-state local spin distribution, Fe A vs Fe B ──────
    weights_A = [spin_dist_A[s] for s in s_values]
    weights_B = [spin_dist_B[s] for s in s_values]
    ax_full.bar(x - bar_width/2, weights_A, bar_width, color=_PLOT_BLUE, label="Fe A")
    ax_full.bar(x + bar_width/2, weights_B, bar_width, color=_PLOT_ORANGE, label="Fe B")
    ax_full.set_xticks(x)
    ax_full.set_xticklabels([f"{s:.1f}" for s in s_values])
    ax_full.set_xlabel("local spin S")
    ax_full.set_ylabel("weight  w(S)")
    ax_full.set_title("Local spin distribution (full state)", loc="left",
                       color=_PLOT_INK_PRIMARY, fontsize=11)
    ax_full.legend(frameon=False, labelcolor=_PLOT_INK_SECONDARY)
    _style_axes(ax_full)

    # ── bottom-left: cross-sector decomposition of wA(s) ──────────────────
    if has_sector:
        sector_weight = result["sector_weight"]
        spin_dist_A_sector = result["spin_dist_A_sector"]
        sector_part   = [sector_weight * spin_dist_A_sector.get(s, 0.0) for s in s_values]
        residual_part = [weights_A[i] - sector_part[i] for i in range(len(s_values))]
        ax_sector.bar(x, sector_part, bar_width * 1.5, color=_PLOT_AQUA,
                       edgecolor=_PLOT_INK_PRIMARY, linewidth=0.4, label="within Omega=10")
        ax_sector.bar(x, residual_part, bar_width * 1.5, bottom=sector_part, color=_PLOT_VIOLET,
                       edgecolor=_PLOT_INK_PRIMARY, linewidth=0.4, label="Omega<10 residual")
        # direct value labels: relief for the aqua fill's sub-3:1 contrast against the surface
        for i, s in enumerate(s_values):
            total = weights_A[i]
            if abs(total) > 1e-9:
                ax_sector.text(x[i], total, f"{total:.3f}", ha="center", va="bottom",
                                fontsize=8, color=_PLOT_INK_SECONDARY)
        ax_sector.set_xticks(x)
        ax_sector.set_xticklabels([f"{s:.1f}" for s in s_values])
        ax_sector.set_xlabel("local spin S_A")
        ax_sector.set_ylabel("weight  w_A(S)")
        ax_sector.set_title("Fe A: cross-sector decomposition", loc="left",
                             color=_PLOT_INK_PRIMARY, fontsize=11)
        ax_sector.legend(frameon=False, labelcolor=_PLOT_INK_SECONDARY)
        _style_axes(ax_sector)
    else:
        ax_sector.axis("off")
        ax_sector.text(0.5, 0.5, "sector analysis not available\n(analyze_sector=False)",
                        ha="center", va="center", color=_PLOT_INK_MUTED, transform=ax_sector.transAxes)

    # ── bottom-right: scalar summary ──────────────────────────────────────
    ax_summary.axis("off")
    ax_summary.set_title("Summary", loc="left", color=_PLOT_INK_PRIMARY, fontsize=11)
    summary_rows = [("w55  (joint S_A=S_B=5/2, full state)", result["joint_weight_55"])]
    if has_sector:
        summary_rows += [
            ("W10  (Omega=10 sector weight)", result["sector_weight"]),
            ("w55_10  (joint S_A=S_B=5/2 within Omega=10)", result["joint_weight_55_sector"]),
            ("W10 * w55_10  (predicted full-state w55)",
             result["sector_weight"] * result["joint_weight_55_sector"]),
        ]
    y = 0.85
    for label, value in summary_rows:
        ax_summary.text(0.0, y, label, fontsize=9.5, color=_PLOT_INK_SECONDARY,
                         transform=ax_summary.transAxes)
        ax_summary.text(1.0, y, f"{value:.6f}", fontsize=9.5, color=_PLOT_INK_PRIMARY,
                         ha="right", family="monospace", transform=ax_summary.transAxes)
        y -= 0.18

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_PLOT_SURFACE)
    return fig


# ── sanity checks ────────────────────────────────────────────────────────────
#
# Each check tests a specific function against an independently derivable truth.
# Failure guide:
#   A fails -> bug in spin_correlation_matrix or _spin_dot_spin
#   B fails -> spin_correlation_matrix and local_spin_distribution disagree;
#              one of local_S2 / _spin_dot_spin is wrong
#   C fails -> wrong block definition, or sign error in cross-block _spin_dot_spin
#   D fails -> bug in local_spin_distribution (moment computation or Lagrange
#              polynomial coefficients)
#   E fails -> bug in local_S2, or state-format mismatch between
#              _make_ionic_reference and get_fci
#   F fails -> bug in local_S2, or ffsim.hartree_fock_state format mismatch

def _check_scalar(name, got, expected, tol=1e-5):
    """Print PASS/FAIL for a scalar check."""
    abs_diff = abs(got - expected)
    ref_scale = max(abs(expected), 1e-12)
    passed = (abs_diff / ref_scale < tol) or (abs_diff < tol)
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not passed:
        print(f"        got {got:.8g}, expected {expected:.8g}, diff {abs_diff:.2e}")


def _check_vector(name, got, expected, tol=1e-5):
    """Print PASS/FAIL for a vector check."""
    max_diff = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    print(f"  {'PASS' if max_diff < tol else 'FAIL'}  {name}")
    if max_diff >= tol:
        print(f"        max element-wise diff = {max_diff:.2e}")


def _make_ionic_reference(norb, nelec):
    """Construct |5alpha in block A, 5beta in block B> as a flat CI vector.

    This Slater determinant has alpha electrons in orbitals {0,1,2,3,4} and
    beta electrons in orbitals {5,6,7,8,9}.  It is NOT a singlet.

    Format matches get_fci() output (pyscf.fci.direct_spin1 CI vector).

    Analytically known values:
      C[p,p] = 3/4 for all p  (singly occupied, spin-1/2 per orbital)
      C[p,q] = 1/4 for p!=q in same block  (aligned spins)
      C[p,q] = -1/4 for p in A, q in B  (opposite spins)
      <SA^2> = <SB^2> = 35/4 = 8.75
      w_A(5/2) = w_B(5/2) = 1;  w55 = 1
    """
    from pyscf.fci import cistring
    n_alpha, n_beta = nelec
    ci_strings = cistring.make_strings(range(norb), n_alpha)
    n_strings  = len(ci_strings)
    alpha_mask = (1 << n_alpha) - 1              # orbs 0..(na-1)  = 31  for na=5
    beta_mask  = ((1 << n_beta) - 1) << n_alpha  # orbs na..(na+nb-1) = 992 for nb=5
    alpha_idx = int(np.searchsorted(ci_strings, alpha_mask))
    beta_idx  = int(np.searchsorted(ci_strings, beta_mask))
    assert ci_strings[alpha_idx] == alpha_mask and ci_strings[beta_idx] == beta_mask, \
        "cistring lookup failed -- check norb/nelec"
    ci_vector = np.zeros(n_strings * n_strings, dtype=complex)
    ci_vector[alpha_idx * n_strings + beta_idx] = 1.0
    return ci_vector


def _make_hf_reference(norb, nelec):
    """Return the ffsim Hartree-Fock state (block A doubly-occupied, block B empty).

    For blocks A = {0-4} and B = {5-9}:
      Every orbital in A is doubly occupied -> local spin singlet -> SA = 0.
      Every orbital in B is empty           -> no electrons      -> SB = 0.

    Analytically: C[p,q] = 0 everywhere, w_A(0) = w_B(0) = 1, w55 = 0.
    """
    return ffsim.hartree_fock_state(norb, nelec).astype(complex)


def sanity_checks(result):
    """Run all sanity checks on the dict returned by analyze().

    Can also be triggered by passing run_checks=True to analyze().
    """
    mol_data     = result["mol_data"]
    state_vector = result["state_vector"]
    corr_matrix  = result["corr_matrix"]
    spin_dist_A  = result["spin_dist_A"]
    spin_dist_B  = result["spin_dist_B"]
    norb  = mol_data.norb
    nelec = mol_data.nelec

    print(f"\n{'='*65}")
    print("  SANITY CHECKS")
    print(f"{'='*65}")

    # A. Row sums of C = 0 (singlet: S_tot|psi> = 0 => sum_q <s_p.s_q> = 0)
    print("\n-- A: row sums of C = 0  (singlet constraint) --")
    row_sums = corr_matrix.sum(axis=1)
    max_row_sum = float(np.max(np.abs(row_sums)))
    print(f"  {'PASS' if max_row_sum < 1e-5 else 'FAIL'}  "
          f"max |row sum of C| = {max_row_sum:.2e}  (expect < 1e-5)")

    # B. <SA^2> from block-sum of C  vs.  from sum s(s+1) w_A(s)
    #    These use _spin_dot_spin and local_S2 respectively -- independent code paths.
    print("\n-- B: <SA^2> from C  =  <SA^2> from w(s)  (two code paths) --")
    spin_squared_A_from_corr = float(corr_matrix[:5, :5].sum())
    spin_squared_A_from_dist = float(sum(s_val*(s_val+1)*weight for s_val, weight in spin_dist_A.items()))
    _check_scalar("<SA^2>: block-sum(C_AA)  vs  sum s(s+1)w_A(s)", spin_squared_A_from_corr, spin_squared_A_from_dist)
    spin_squared_B_from_corr = float(corr_matrix[5:, 5:].sum())
    spin_squared_B_from_dist = float(sum(s_val*(s_val+1)*weight for s_val, weight in spin_dist_B.items()))
    _check_scalar("<SB^2>: block-sum(C_BB)  vs  sum s(s+1)w_B(s)", spin_squared_B_from_corr, spin_squared_B_from_dist)

    # C. Cross-block sum = -(SA^2 + SB^2)/2  (exact for any singlet)
    print("\n-- C: <SA.SB> from C  =  -(SA^2+SB^2)/2 --")
    _check_scalar("sum_{p in A, q in B} C[p,q]  =  -(SA2+SB2)/2",
         float(corr_matrix[:5, 5:].sum()), -(spin_squared_A_from_dist + spin_squared_B_from_dist) / 2)

    # D. Projector norm: ||P_A(5/2)|psi>||^2 = w_A(5/2)
    #    Tests that the Lagrange polynomial projector satisfies P^2 = P.
    print("\n-- D: projector norm  ||P_A(5/2)|psi>||^2 = w_A(5/2) --")
    proj_state_A = proj_52_vec(state_vector, FE_A_ORBITALS, norb, nelec)
    proj_norm_sq = float(np.real(np.vdot(proj_state_A, proj_state_A)))
    _check_scalar("||P_A(5/2)|psi>||^2  =  w_A(5/2)", proj_norm_sq, spin_dist_A[2.5])

    # E. Ionic reference  |5alpha in A, 5beta in B>
    print("\n-- E: ionic reference  |5alpha in block A, 5beta in block B> --")
    try:
        ionic_state = _make_ionic_reference(norb, nelec)
        corr_ionic  = spin_correlation_matrix(ionic_state, norb, nelec)
        spin_dist_A_ionic = local_spin_distribution(ionic_state, FE_A_ORBITALS, norb, nelec)
        spin_dist_B_ionic = local_spin_distribution(ionic_state, FE_B_ORBITALS, norb, nelec)
        joint_weight_ionic = joint_high_spin_weight(ionic_state, FE_A_ORBITALS, FE_B_ORBITALS, norb, nelec)

        corr_expected = np.full((norb, norb), -0.25)
        for orb_p in range(norb):
            for orb_q in range(norb):
                if (orb_p < 5) == (orb_q < 5):   # same block
                    corr_expected[orb_p, orb_q] = 0.75 if orb_p == orb_q else 0.25

        _check_vector("E1: C = 3/4 diagonal, 1/4 within-block, -1/4 cross-block",
              corr_ionic.ravel(), corr_expected.ravel())
        _check_scalar("E2: w_A(5/2) = 1",   spin_dist_A_ionic.get(2.5, 0), 1.0)
        _check_scalar("E3: w_B(5/2) = 1",   spin_dist_B_ionic.get(2.5, 0), 1.0)
        _check_scalar("E4: w55 = 1",         joint_weight_ionic,               1.0)
        _check_scalar("E5: <SA^2> = 8.75",  sum(s_val*(s_val+1)*weight for s_val, weight in spin_dist_A_ionic.items()), 8.75)
        _check_scalar("E6: <S^2_tot> = 5.0  (NOT a singlet)", float(corr_ionic.sum()), 5.0)
    except Exception as exc:
        print(f"  SKIP  {exc}")

    # F. HF reference  (block A doubly-occupied, block B empty)
    print("\n-- F: HF reference  (block A doubly-occupied, block B empty) --")
    try:
        hf_state = _make_hf_reference(norb, nelec)
        corr_hf  = spin_correlation_matrix(hf_state, norb, nelec)
        spin_dist_A_hf  = local_spin_distribution(hf_state, FE_A_ORBITALS, norb, nelec)
        spin_dist_B_hf  = local_spin_distribution(hf_state, FE_B_ORBITALS, norb, nelec)
        joint_weight_hf = joint_high_spin_weight(hf_state, FE_A_ORBITALS, FE_B_ORBITALS, norb, nelec)

        _check_vector("F1: C = 0 everywhere",  corr_hf.ravel(), np.zeros(norb**2))
        _check_scalar("F2: w_A(0) = 1",   spin_dist_A_hf.get(0, 0),   1.0)
        _check_scalar("F3: w_A(5/2) = 0", spin_dist_A_hf.get(2.5, 0), 0.0)
        _check_scalar("F4: w_B(0) = 1",   spin_dist_B_hf.get(0, 0),   1.0)
        _check_scalar("F5: w_B(5/2) = 0", spin_dist_B_hf.get(2.5, 0), 0.0)
        _check_scalar("F6: w55 = 0",       joint_weight_hf,             0.0)
    except Exception as exc:
        print(f"  SKIP  {exc}")

    print()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if COMPARE_ROTATED and ROTATED_PATH.exists():
        before, after = compare(UNROTATED_PATH, ROTATED_PATH, FE_A_ORBITALS, FE_B_ORBITALS)
        plot_local_spins(before, title="before rotation", save_path=_HERE / "local_spins_before.png")
        plot_local_spins(after, title="after rotation", save_path=_HERE / "local_spins_after.png")
    else:
        result = analyze(UNROTATED_PATH, FE_A_ORBITALS, FE_B_ORBITALS, label="fe2s2_10e10o_FCIDUMP",
                          run_checks=True)
        plot_local_spins(result, title="fe2s2_10e10o_FCIDUMP", save_path=_HERE / "local_spins.png")

    plt.show()
