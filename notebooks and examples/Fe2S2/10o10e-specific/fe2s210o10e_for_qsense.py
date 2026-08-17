"""
Generate the Q-SENSE 'phys_spatial' pickle for the Fe2S2 iron-sulfur cluster
in the 10-orbital / 10-electron active space, starting from the `fe2s2` FCIDUMP.

This mirrors the existing FCIDUMP generator (10e10o_active_space_rrf.py) but, in
addition to the FCIDUMP, it writes the pickle that the qsense driver
(CSF_UCSF_onidx.py) consumes for an "MCSCF" Hamiltonian:

    [Enuc, obt_spatial, tbt_spatial, orbene, nelec, nactmo, nactel]

where
  Enuc        : effective core energy ecore  (FCIDUMP constant + the 10 frozen
                doubly-occupied core orbitals folded in)        -- scalar
  obt_spatial : effective 1e integrals in the active MO basis   -- (nact, nact)
  tbt_spatial : 2e integrals, PHYSICIST ordering with the 1/2 factor, i.e.
                tbt[p,q,r,s] for  a_p^+ a_q^+ a_r a_s            -- (nact,)*4
  orbene      : active CASSCF orbital energies                  -- (nact,)
  nelec       : electrons in this (active) Hamiltonian = 10
  nactmo      : number of active spatial orbitals = 10
  nactel      : number of active electrons = 10

The 2e ordering is built to be byte-for-byte identical to what the molecular
example scripts (n2_S0*.py / h2o_sto3g.py) produce via
    g = 0.5 * einsum('psqr,pa,qb,rc,sd->abcd', g_ao_chem, O,O,O,O)
For an MO-basis chemist tensor eri[p,q,r,s] = (pq|rs) this is exactly
    tbt[a,b,c,d] = 0.5 * eri[a,d,b,c]
and the loader's tbt_phys_spatial_to_spin expands it with the
(alpha,alpha,alpha,alpha)/(beta,beta,beta,beta)/(alpha,beta,beta,alpha)/
(beta,alpha,alpha,beta) blocks.

Run from the qsense run directory:                 then run qsense with:
    python fe2s2_10o10e_qsense_phys.py             --act_start 0 --act_end 9 --S_x2 0

------------------------------------------------------------------------------
ORBITAL ORDERING / SYMMETRY SENSITIVITY  -- read before changing the run setup
------------------------------------------------------------------------------
The active orbitals are written in their NATIVE CASSCF order (the order produced
by sort_mo + canonicalization), NOT sorted by orbital energy. This is the
correct choice *because symmetry is off* in this workflow, i.e. the qsense driver
(CSF_UCSF_onidx.py) runs with `l_no_sym = True`. Concretely:

  * The driver's axial-symmetry detector (CSF_UCSF_onidx.py ~L187-192) flags a
    symmetry pair by spotting ADJACENT degenerate orbital energies
    (np.isclose(orbene[i], orbene[i+1])). That adjacency requirement is the only
    thing that would force an energy sort -- and with `l_no_sym = True` the
    detector's result is discarded (L199-201 set l_axial_sym = False). So with
    symmetry off, the ordering of the active orbitals is irrelevant to the
    physics, and native order keeps this pickle index-consistent with the
    FCIDUMP and the *_mo_energy.npy that are written here in the same order.

  * This system additionally has NO exact spatial symmetry (mol.symmetry = False
    on the FCIDUMP SCF). The Fe[3d] near-degeneracy (~0.02 Ha spread) is an
    electronic/strong-correlation effect, not a symmetry degeneracy, and stays
    well outside np.isclose's default tolerance, so it would not trip the
    detector even if it were active.

IF YOU LATER TURN SYMMETRY ON (l_no_sym = False) OR USE A PARTIAL ACTIVE WINDOW
(--act_start > 0): you must energy-sort the active orbitals, applying ONE single
permutation consistently to orbene, obt (both axes), tbt (all four axes), AND the
re-dumped FCIDUMP and *_mo_energy.npy together. A sort is just a relabeling and
changes no converged result, but (a) sorting orbene without permuting obt/tbt, or
(b) sorting this pickle while leaving the FCIDUMP/npy in native order, will
silently corrupt the Hamiltonian / the MP2 denominators. One canonical order
across every artifact, or none.
"""

import os
import math
import pickle
import numpy as np
from pyscf import mcscf, ao2mo, tools, fci
import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
fcidump_in   = 'fe2s2_fresh_download'          # input FCIDUMP (20 spatial orbitals, 30 electrons)
norb         = 20               # spatial orbitals in the FCIDUMP
nact         = 10               # active spatial orbitals
nelec_act    = (5, 5)           # (alpha, beta) active electrons -> 10 active e-
active_pick  = [2, 3, 4, 5, 6] + [13, 14, 15, 16, 17]   # global MO indices (base=0)
moltag       = 'fe2s2'
outdir       = 'hamiltonians'

# Ground-state character analysis (printed after the artifacts are written, so a
# slow/interrupted analysis never costs you the pickle). Set False for fast
# regeneration. STATE_SPECIFIC_LADDER re-optimizes orbitals per spin (≈6 small
# CASSCF, a few minutes) and is the variational answer to "which spin is lowest".
ANALYZE_GROUND_STATE   = True
STATE_SPECIFIC_LADDER  = True
LARGE_CI_TOL           = 0.03      # report determinants with |coeff| above this
max_seniority_probe    = 10         # the qsense --max_seniority you intend to use
VERBOSE_DUMPS          = True      # print Fock/integral matrices + .analyze() (Toby-style)
PRINT_LATEX            = True      # append a copy-pastable LaTeX summary at the end
SAVE_LADDER_PLOT        = True      # save a spin-ladder plot to the run directory
LADDER_PLOT_PATH = "Figures/"

date_str     = datetime.date.today().strftime('%Y%m%d')
fcidump_out  = f'{moltag}_10e10o_casscf_FCIDUMP_{date_str}'
pickle_out   = os.path.join(outdir, f'{moltag}_10o10e_phys_spatial_{date_str}')
orbene_npy   = f'{moltag}_10e10o_casscf_mo_energy_{date_str}.npy'
casscf_orbitals_npy = f'casscf_orbitals_{date_str}.npy'
ladder_plot_out = os.path.join(LADDER_PLOT_PATH, f'{moltag}_10o10e_spin_ladder_{date_str}.png')

os.makedirs(outdir, exist_ok=True)

# ----------------------- to check spin ladder ----------------------------------------
def heisenberg_analysis(E, HA2CM):
    E = np.asarray(E, float); S = np.arange(len(E)); x = S*(S+1)
    dE = (E - E[0]) * HA2CM
    J_gap = dE[1]/2                                      # singlet-triplet
    lo = S <= 2
    J_low = np.sum(x[lo]*dE[lo]) / np.sum(x[lo]**2)      # low-S fit through origin
    J_all = np.sum(x*dE) / np.sum(x**2)                  # whole-ladder fit through origin
    Jbq, Kbq = np.linalg.lstsq(np.vstack([x, x**2]).T, dE, rcond=None)[0]
    resid = dE - J_low*x                                 # per-rung residual of the low-S line
    Jeff = np.concatenate([[np.nan], np.diff(dE)/(2*S[1:])])
    print("\n% --- Heisenberg residual table ---")
    print(r"\begin{tabular}{rrrrr}\hline")
    print(r"$S$ & $\Delta E$ & $J_{\rm low}\,S(S+1)$ & residual & $J_{\rm eff}(S)$ \\")
    print(r"    & (cm$^{-1}$) & (cm$^{-1}$) & (cm$^{-1}$) & (cm$^{-1}$) \\\hline")
    for s in S:
        je = "--" if s == 0 else f"{Jeff[s]:.1f}"
        print(f"{s} & {dE[s]:.1f} & {J_low*x[s]:.1f} & {resid[s]:+.1f} & {je} \\\\")
    print(r"\hline\end{tabular}")
    print(f"% J_gap={J_gap:.1f}  J_low={J_low:.1f}  J_all={J_all:.1f}  "
          f"biquadratic K={Kbq:.2f} cm^-1")
    return dict(J_gap=J_gap, J_low=J_low, J_all=J_all, Kbq=Kbq, resid=resid, Jeff=Jeff)

# --- printing helpers in the style of the original utils_misc scripts --------
def print_matrix(matrix, n_per_group=6):
    matrix = np.asarray(matrix)
    ncol = matrix.shape[1]
    for imin in range(0, ncol, n_per_group):
        imax = min(imin + n_per_group, ncol)
        for row in matrix[:, imin:imax]:
            print(' '.join(f'{num:12.6f}' for num in row))
        print('')


def print_eigen_solution(eigen_values, eigen_vectors, n_per_group=5):
    eigen_vectors = np.asarray(eigen_vectors)
    nsol = len(eigen_values)
    for imin in range(0, nsol, n_per_group):
        imax = min(imin + n_per_group, nsol)
        print(' '.join(f'{v:12.6f}' for v in eigen_values[imin:imax]))
        print('')
        for row in eigen_vectors[:, imin:imax]:
            print(' '.join(f'{c:12.6f}' for c in row))
        print('')


def plot_spin_ladder(E, HA2CM, outpath, J_exp_cm=148.0, title_tag=""):
    """Two-panel spin-ladder figure from a length-N array of total energies E[S], S=0..N-1.

    Left  : Delta E vs S(S+1) with the low-S Heisenberg fit and a bilinear-biquadratic
            fit; every rung labelled with its excitation energy (cm^-1) and S.
    Right : per-rung effective coupling J_eff(S) = [E(S)-E(S-1)]/2S, against expt. |J|.

    Returns dict(J_low, J_gap, b). Fits use the same formulas as heisenberg_analysis(),
    so the figure and the residual table agree.
    """
    import matplotlib
    matplotlib.use("Agg")                       # headless-safe (cluster / CI)
    import matplotlib.pyplot as plt

    E = np.asarray(E, float)
    if E.size < 2 or not np.all(np.isfinite(E)):
        print("plot_spin_ladder: need >=2 finite energies, skipping"); return None

    S   = np.arange(E.size)
    x   = S * (S + 1)
    dE  = (E - E[0]) * HA2CM                     # cm^-1
    lo  = S <= 2
    J_gap = dE[1] / 2.0                                          # singlet-triplet
    J_low = float(np.sum(x[lo] * dE[lo]) / np.sum(x[lo] ** 2))   # low-S fit through origin
    a, b  = np.linalg.lstsq(np.vstack([x, x ** 2]).T, dE, rcond=None)[0]
    Jeff  = np.diff(dE) / (2 * S[1:])

    style = {"font.size": 11, "font.family": "serif",
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.1))

        xx = np.linspace(0, x.max(), 200)
        ax1.plot(xx, J_low * xx, "--", color="0.5",
                 label=fr"Heisenberg (low-$S$), $J={J_low:.0f}$ cm$^{{-1}}$")
        ax1.plot(xx, a * xx + b * xx ** 2, "-", color="steelblue", alpha=0.8,
                 label=fr"bilinear-biquadratic ($b={b:.2f}$)")
        ax1.scatter(x, dE, s=55, color="tomato", zorder=5, edgecolor="k", linewidth=0.5)
        for xi, yi, Si in zip(x, dE, S):
            ax1.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                         xytext=(7, -3), fontsize=8.5, color="0.15")
            ax1.annotate(f"$S={Si}$", (xi, yi), textcoords="offset points",
                         xytext=(2, -14), fontsize=8, color="0.45")
        ax1.set_xlabel(r"$S(S+1)$")
        ax1.set_ylabel(r"$E(S)-E(0)$  (cm$^{-1}$)")
        ax1.set_title("Spin ladder vs. Heisenberg interval rule")
        ax1.legend(fontsize=8.5, loc="upper left", frameon=False)
        ax1.set_xlim(-1, x.max() + 1)

        ax2.axhline(J_exp_cm, color="seagreen", ls=":", lw=1.6,
                    label=fr"expt. $|J|={J_exp_cm:.0f}$ cm$^{{-1}}$ (Gillum 1976)")
        ax2.plot(S[1:], Jeff, "o-", color="tomato", lw=1.6, markersize=7,
                 markeredgecolor="k", markeredgewidth=0.5,
                 label=r"$J_{\rm eff}(S)=[E(S)-E(S-1)]/2S$")
        for Si, Ji in zip(S[1:], Jeff):
            ax2.annotate(f"{Ji:.0f}", (Si, Ji), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8.5, color="0.15")
        ax2.set_xlabel(r"$S$  (upper level of the rung)")
        ax2.set_ylabel(r"$J_{\rm eff}$  (cm$^{-1}$)")
        ax2.set_title("High-spin softening of the exchange")
        ax2.set_xticks(S[1:])
        ax2.set_ylim(0, max(J_exp_cm, Jeff.max()) * 1.12)
        ax2.legend(fontsize=8.5, loc="lower left", frameon=False)

        if title_tag:
            fig.suptitle(title_tag, fontsize=10, color="0.35", y=1.02)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"spin-ladder figure -> {outpath}  "
          f"(J_gap={J_gap:.1f}, J_low={J_low:.1f}, b={b:.3f} cm^-1)")
    return dict(J_low=J_low, J_gap=J_gap, b=float(b))

# ---------------------------------------------------------------------------
# 1) SCF on the FCIDUMP, stabilized to an internally stable solution
# ---------------------------------------------------------------------------
mf_as = tools.fcidump.to_scf(fcidump_in)
ecore_fcidump_in = mf_as.mol.energy_nuc()   # ECORE constant from the original FCIDUMP
mf_as.mol.verbose = 4
mf_as.kernel()
for _ in range(10):
    moi, moe, si, se = mf_as.stability(return_status=True)
    mf_as.kernel(mf_as.make_rdm1(moi, mf_as.mo_occ))

# ---------------------------------------------------------------------------
# 2) CASCI then CASSCF in the 10o(5,5)e active space (kept faithful to the
#    original generator). CASSCF orbitals/integrals are what we pickle.
# ---------------------------------------------------------------------------
mf_as.mo_coeff = np.eye(norb)
mf_as.mol.symmetry = False
mycasci = mcscf.CASCI(mf_as, nact, nelec_act)
mycasci.sort_mo(active_pick, base=0)
mycasci.fix_spin_(ss=0)
mycasci.kernel(mycasci.sort_mo(active_pick, base=0))

mf_as.mo_coeff = np.eye(norb)
mf_as.mol.symmetry = False
mycas = mcscf.CASSCF(mf_as, nact, nelec_act)
mo = mycas.sort_mo(active_pick, base=0)
mycas.fix_spin_(ss=0)
mycas.kernel(mo)
np.save(casscf_orbitals_npy, mycas.mo_coeff)

ncore = mycas.ncore                 # 10 doubly-occupied inactive orbitals
ncas  = mycas.ncas                  # 10
nelecas = mycas.nelecas             # (5, 5)
n_active_elec = int(sum(nelecas))   # 10

# ---------------------------------------------------------------------------
# 2b) S=5 cross-check against Li & Chan, same unrelaxed orbitals
# ---------------------------------------------------------------------------
# S = 5 in a 10-active-electron space is fully polarized: (10 alpha, 0 beta).
# With no beta electrons there is only one possible determinant, so this is
# automatically a pure S=5 eigenstate -- do NOT apply fix_spin_(ss=0) here,
# that penalty targets S=0 and would be wrong for this run.
nelec_S5 = (10, 0)

mf_as.mo_coeff = np.eye(norb)          # same reset used before every CASCI/CASSCF call above
mf_as.mol.symmetry = False
mycasci_S5 = mcscf.CASCI(mf_as, nact, nelec_S5)
mo_S5 = mycasci_S5.sort_mo(active_pick, base=0)   # identical orbital pick as the S=0 run
mycasci_S5.kernel(mo_S5)

s2_0, _  = mycasci.fcisolver.spin_square(mycasci.ci, nact, nelec_act)
s2_5, _  = mycasci_S5.fcisolver.spin_square(mycasci_S5.ci, nact, nelec_S5)

print('\n=== CASCI cross-check: S=0 vs S=5, same UNRELAXED starting orbitals ===')
print(f'{"":>26} {"E_tot (Ha)":>16} {"E(CI) = E[act] (Ha)":>22} {"<S^2>":>8}')
print(f'{"S=0 (this run)":>26} {mycasci.e_tot:>16.8f} {mycasci.e_cas:>22.10f} {s2_0:>8.3f}')
print(f'{"S=0 (Li & Chan)":>26} {"--":>16} {-27.887643:>22.6f} {0.0:>8.3f}')
print(f'{"S=5 (this run)":>26} {mycasci_S5.e_tot:>16.8f} {mycasci_S5.e_cas:>22.10f} {s2_5:>8.3f}')
print(f'{"S=5 (Li & Chan)":>26} {"--":>16} {-27.890357:>22.6f} {30.0:>8.3f}')

print(f'\n|S=0 - Li&Chan| = {abs(mycasci.e_cas - (-27.887643)):.2e} Ha')
print(f'|S=5 - Li&Chan| = {abs(mycasci_S5.e_cas - (-27.890357)):.2e} Ha')
print('Ground state on THESE (unrelaxed) orbitals: '
      f'{"S=0" if mycasci.e_cas < mycasci_S5.e_cas else "S=5"}')

# ---------------------------------------------------------------------------
# 2c) Fe-3d population check: did CASSCF relaxation pull in S-3p character?
#
# WHY THIS EXISTS -- The S=0/S=5 ordering flips between the unrelaxed CASCI
# step above (matches Li & Chan's literature values, S=5 lower) and the
# state-specific CASSCF ladder further down this script (S=0 lower). The
# literature explanation for why a Fe-3d-only (10e,10o) space gets the
# ordering "wrong" is the absence of S 3p orbitals for superexchange -- but
# CASSCF optimizes over the FULL one-particle basis, not just within the
# fixed 10 reference columns, so it is free to rotate S-3p character INTO
# the 10 orbitals this script still labels "active" without ever being told
# to. If that happened, the CASSCF ladder isn't really testing a pure
# Fe-3d active space anymore, and the "relaxation alone fixes the ordering"
# reading would be wrong.
#
# HOW -- mf_as.mol here has no atoms/basis (it's an FCIDUMP-only mol), so
# real Mulliken/Loewdin population analysis isn't available. But get_ovlp
# is hard-set to the identity by fcidump.to_scf, so the ORIGINAL 20-orbital
# FCIDUMP basis is exactly orthonormal, and Li & Chan's own README labels
# active_pick = [2,3,4,5,6,13,14,15,16,17] as the Fe-3d subset of those 20.
# That makes summed squared coefficients against those 20 reference columns
# an EXACT population measure here (no Loewdin symmetrization needed),
# rather than an approximation.
#
# READING IT -- the "unrelaxed CASCI" column below is a built-in self-check
# and MUST print 1.0000 everywhere (CASCI only rotates within the already-
# selected active block, never mixes in the other 10 reference orbitals).
# The "CASSCF-relaxed" column is the real test: still ~1.0000 across the
# board => the ordering flip is genuine within-Fe3d relaxation physics;
# rows dropping noticeably below 1.0 => CASSCF has mixed in S-3p (or other)
# character, and the active space is no longer what the pickle claims.
# ---------------------------------------------------------------------------
fe3d_idx  = active_pick
other_idx = [i for i in range(norb) if i not in fe3d_idx]

def fe3d_weight(mo_coeff, ncore, ncas):
    C = mo_coeff[:, ncore:ncore + ncas]          # (20, 10)
    norm = np.sum(C**2, axis=0)
    assert np.allclose(norm, 1.0, atol=1e-6), f"columns not unit-norm: {norm}"
    return np.sum(C[fe3d_idx, :]**2, axis=0)

w_unrelaxed = fe3d_weight(mycasci.mo_coeff, mycasci.ncore, mycasci.ncas)
w_casscf    = fe3d_weight(mycas.mo_coeff,   mycas.ncore,   mycas.ncas)

print('\n=== Fe-3d population of the active orbitals: unrelaxed vs CASSCF-relaxed ===')
print(f"{'active idx':>10} {'global MO':>9} {'unrelaxed CASCI':>16} {'CASSCF-relaxed':>16}")
for i in range(mycas.ncas):
    print(f"{i:>10d} {mycas.ncore+i:>9d} {w_unrelaxed[i]:>16.4f} {w_casscf[i]:>16.4f}")
print(f"\nmean Fe3d-index weight: unrelaxed = {w_unrelaxed.mean():.4f}, "
      f"CASSCF-relaxed = {w_casscf.mean():.4f}")

# ---------------------------------------------------------------------------
# 2d) Causal control: FROZEN-rotation CASSCF (no Fe3d<->S3p mixing allowed)
#
# WHY -- the population check showed CASSCF lets ~15% S-3p character leak
# into the nominally-Fe3d active orbitals. That's a correlation (leakage
# co-occurs with the S=0/S=5 flip), not yet a demonstrated cause. In THIS
# problem ncore=10, ncas=10, nvir=0 (verified directly from PySCF's
# uniq_var_indices: with no frozen orbitals the only nonzero rotation block
# is active<->core, i.e. Fe3d<->S3p -- there is no active-active rotation
# (internal_rotation=False by default) and no virtual block (nvir=0) to
# provide any other channel). So Fe3d<->S3p mixing isn't A source of
# CASSCF's orbital freedom here -- it is the ONLY source. Locking it
# (frozen = the 10 core/S3p positions, in CASSCF's POST-sort_mo indexing,
# i.e. list(range(ncore)) -- NOT the original global FCIDUMP indices in
# active_pick/other_idx, which no longer apply after sort_mo reorders the
# columns) should collapse CASSCF's orbital gradient to exactly zero,
# leaving mo_coeff unchanged from the CASCI starting point. If so, its
# energy/ordering MUST reproduce the unrelaxed CASCI numbers (S=5 lower,
# matching Li & Chan) -- verified as a pure boolean-mask fact via
# CASSCF.uniq_var_indices before running any chemistry. If the frozen run
# instead differs from CASCI, something is leaking through an unaccounted
# channel and the "leakage causes the flip" story needs re-examining.
# ---------------------------------------------------------------------------
mf_as.mo_coeff = np.eye(norb)
mf_as.mol.symmetry = False
mycas_frozen = mcscf.CASSCF(mf_as, nact, nelec_act)
mo_frozen = mycas_frozen.sort_mo(active_pick, base=0)
mycas_frozen.fix_spin_(ss=0)
mycas_frozen.frozen = list(range(mycas_frozen.ncore))   # lock all active<->core rotation
mycas_frozen.kernel(mo_frozen)

w_frozen = fe3d_weight(mycas_frozen.mo_coeff, mycas_frozen.ncore, mycas_frozen.ncas)

print('\n=== Causal control: CASSCF with active<->core rotation frozen ===')
print(f'{"":>28} {"E_tot (Ha)":>16} {"mean Fe3d weight":>18}')
print(f'{"unrelaxed CASCI (S=0)":>28} {mycasci.e_tot:>16.8f} {w_unrelaxed.mean():>18.4f}')
print(f'{"frozen CASSCF (S=0)":>28} {mycas_frozen.e_tot:>16.8f} {w_frozen.mean():>18.4f}')
print(f'{"unrestricted CASSCF (S=0)":>28} {mycas.e_tot:>16.8f} {w_casscf.mean():>18.4f}')
print(f'\n|frozen CASSCF - unrelaxed CASCI| = {abs(mycas_frozen.e_tot - mycasci.e_tot):.2e} Ha '
      '(should be ~0 -- confirms no other rotation channel exists)')
print(f'energy gained ONLY by allowing Fe3d<->S3p mixing = '
      f'{mycas.e_tot - mycas_frozen.e_tot:.6f} Ha')

# ---------------------------------------------------------------------------
# 3) Active-space effective integrals
# ---------------------------------------------------------------------------
h1e_cas, ecore = mycas.get_h1eff()              # (nact, nact), scalar core energy
h2e_cas        = mycas.get_h2eff()              # packed (chemist) active ERIs
eri_chem       = ao2mo.restore(1, h2e_cas, ncas)  # (pq|rs) chemist, shape (nact,)*4

# Keep writing the FCIDUMP too (unchanged behavior).
tools.fcidump.from_integrals(fcidump_out, h1e_cas, h2e_cas, ncas, n_active_elec,
                             nuc=ecore, ms=0)

# Physicist-ordered tbt with the 1/2 factor, matching the molecular generators.
#   tbt[a,b,c,d] = 0.5 * eri_chem[a,d,b,c]
tbt_phys = 0.5 * np.einsum('adbc->abcd', eri_chem)

# Active CASSCF orbital energies (this is the sidecar .npy content). These feed
# the MP2 pair angles and the axial-symmetry/degeneracy detection in qsense.
#
# NOTE (symmetry sensitivity): kept in NATIVE CASSCF order, NOT energy-sorted.
# Valid only because symmetry is off (l_no_sym = True) -- see the ORBITAL
# ORDERING note in the module docstring. If you enable symmetry or use a partial
# active window, apply one consistent energy sort here to orbene AND to
# h1e_cas / eri_chem / the FCIDUMP write beSet conv_tol_gradlow, e.g.:
#     perm   = np.argsort(mycas.mo_energy[active])
#     orbene = orbene[perm]; h1e_cas = h1e_cas[np.ix_(perm, perm)]
#     eri_chem = eri_chem[np.ix_(perm, perm, perm, perm)]   # rebuild tbt_phys after
# (and re-dump the FCIDUMP from the permuted integrals so all artifacts agree).
active = slice(ncore, ncore + ncas)
orbene = np.asarray(mycas.mo_energy[active]).copy()
np.save(orbene_npy, orbene)

# ---------------------------------------------------------------------------
# 4) Self-check: reconstruct the CASSCF energy from exactly what we pickle.
#    eri_chem[p,q,r,s] = 2 * tbt_phys[p,r,s,q], and PySCF's 2-RDM satisfies
#    E2 = 0.5 * einsum('pqrs,pqrs', eri_chem, dm2).
# ---------------------------------------------------------------------------
dm1, dm2 = mycas.fcisolver.make_rdm12(mycas.ci, ncas, nelecas)
eri_from_tbt = 2.0 * np.einsum('prsq->pqrs', tbt_phys)
E_recon = (ecore
           + np.einsum('pq,pq->', h1e_cas, dm1)
           + 0.5 * np.einsum('pqrs,pqrs->', eri_from_tbt, dm2))
assert abs(E_recon - mycas.e_tot) < 1e-7, \
    f'Integral self-check failed: {E_recon} vs {mycas.e_tot}'

# ---------------------------------------------------------------------------
# 4b) DIAGNOSTIC: is mycas.mo_energy[active] a set of true (semicanonical)
#     orbital energies, or the raw diagonal of a non-diagonal generalized
#     Fock block? And how catastrophic are the resulting MP2 denominators?
#     These only PRINT; they change nothing that gets written.
# ---------------------------------------------------------------------------
print('\n=== DIAGNOSTIC: nature of the sidecar orbital energies ===')

# (A) Active-block off-diagonality of the generalized Fock matrix.
mo = np.asarray(mycas.mo_coeff)
fock_ao = np.asarray(mycas.get_fock())        # generalized Fock (AO basis)
fock_mo = mo.T @ fock_ao @ mo                 # -> MO basis
A = fock_mo[active, active]
offdiag = np.abs(A - np.diag(np.diag(A)))
print(f'active-block max |F_off-diag|      : {offdiag.max():.3e} Ha')
print(f'orbene == diag(F_active)?          : {np.allclose(orbene, np.diag(A), atol=1e-8)}')
if offdiag.max() < 1e-6:
    print('  -> active block IS (semi)canonical; orbene are genuine orbital energies.')
else:
    print('  -> active block is NOT diagonal; orbene are generalized-Fock diagonals,')
    print('     NOT eigenvalues. MP2 denominators are only heuristic in this frame.')

# (B) How small are the MP2 denominators these orbene produce?
gaps = np.abs(np.subtract.outer(orbene, orbene))
nz = gaps[gaps > 1e-8]
print(f'orbital-energy spread (Ha)         : {np.ptp(orbene):.4f}')
print(f'smallest nonzero |eps_s - eps_r|   : {nz.min():.3e} Ha')
print(f'number of pairs with gap < 1e-3 Ha : {int((nz < 1e-3).sum())} / {nz.size}')
print('=========================================================\n')

# ---------------------------------------------------------------------------
# 5) Write the qsense 'MCSCF' phys_spatial pickle
#    [Enuc, obt_spatial, tbt_spatial, orbene, nelec, nactmo, nactel]
#    For this effective active-space Hamiltonian the whole orbital window IS the
#    active space, so nelec == nactel == 10 and (at run time) --act_start 0.
# ---------------------------------------------------------------------------
Enuc        = float(ecore)
obt_spatial = np.asarray(h1e_cas).copy()
tbt_spatial = tbt_phys
nelec       = n_active_elec
nactmo      = ncas
nactel      = n_active_elec

with open(pickle_out, 'wb') as f:
    pickle.dump([Enuc, obt_spatial, tbt_spatial, orbene, nelec, nactmo, nactel], f)

# ---------------------------------------------------------------------------
# Report: artifacts (label + numbers, in the style of the original scripts)
# ---------------------------------------------------------------------------
ss_gs, mult_gs = mycas.fcisolver.spin_square(mycas.ci, ncas, nelecas)
occ_diag = np.diag(dm1)
print('\n=== Fe2S2 10o10e -> qsense phys pickle ===')
print(f'CASSCF e_tot         : {mycas.e_tot:.10f} Ha')
print(f'<S^2>, 2S+1          : {ss_gs:.6f}, {mult_gs:.4f}')
print(f'energy self-check    : {E_recon:.10f} Ha  (|diff| = {abs(E_recon-mycas.e_tot):.2e})')
print(f'ecore (orig FCIDUMP) : {ecore_fcidump_in:.10f} Ha')
print(f'ecore (10o10e AS)    : {Enuc:.10f} Ha')
print(f'ecore delta (fold)   : {Enuc - ecore_fcidump_in:.10f} Ha  (core-orbital folding contribution)')
print(f'obt_spatial shape    : {obt_spatial.shape}')
print(f'tbt_spatial shape    : {tbt_spatial.shape}')
print(f'nelec/nactmo/nactel  : {nelec} / {nactmo} / {nactel}')
print(f'wrote: {pickle_out}')
print(f'wrote: {fcidump_out}')
print(f'wrote: {orbene_npy}')
print(f'wrote: {casscf_orbitals_npy}')
print(f'qsense args          : -f {pickle_out} --act_start 0 --act_end {ncas-1} --S_x2 0')

# ---------------------------------------------------------------------------
# Ground-state character analysis (answers: ground-state spin? ground-state
# CSF / closed shell?). Active index 0..9 == global MO {ncore..ncore+ncas-1}.
# ---------------------------------------------------------------------------
if ANALYZE_GROUND_STATE:
    HA2CM = 219474.6313708     # Hartree -> cm^-1

    # verbose matrix/orbital dumps, like n2_S0.py / h2o_sto3g.py
    if VERBOSE_DUMPS:
        print('\n=== FCIDUMP-SCF Fock matrix (active block, MO basis) ===')
        print_matrix(np.asarray(mf_as.get_fock())[active, active])
        print('=== effective 1e integrals h1e_cas (obt_spatial) ===')
        print_matrix(obt_spatial)
        print('=== active 1-RDM (spatial) ===')
        print_matrix(dm1)
        print('=== CASSCF active orbital energies & active-block coefficients ===')
        print_eigen_solution(orbene, np.asarray(mycas.mo_coeff)[active, active])
        # mcscf.analyze() is not called: its AO-population step is undefined for
        # an FCIDUMP (no AO basis) and it rotates ci/mo in place. Its only
        # meaningful piece here, the natural occupations, is printed instead.
        print('=== CASSCF natural-orbital occupations (eig of active 1-RDM) ===')
        print('   ' + np.array2string(np.linalg.eigvalsh(dm1)[::-1],
                                       precision=5, floatmode='fixed'))

    # active orbital energies & occupations (mirrors the original Fe2S2 script)
    print('\n=== active orbital energies & occupations (feed the MP2 angles) ===')
    print(f"{'active':>6} {'global MO':>9} {'orb energy (Ha)':>16} {'diag occ':>10}")
    for i, (e, n) in enumerate(zip(orbene, occ_diag)):
        print(f'{i:>6d} {ncore + i:>9d} {e:>16.6f} {n:>10.4f}')
    noons = np.linalg.eigvalsh(dm1)[::-1]
    n_open = int(np.sum((noons > 0.1) & (noons < 1.9)))
    print(f'orb-energy spread (Ha)                 : {np.ptp(orbene):.4f}')
    print(f'NOONs                                  : '
          f'{np.array2string(noons, precision=4, floatmode="fixed")}')
    print(f'orbitals partially occupied (0.1<n<1.9): {n_open} / {ncas}')

    # CSF-level character: spatial configurations + seniority distribution.
    # A CSF is a spin-adapted combination of the determinants of one spatial
    # configuration; seniority = # singly-occupied orbitals also sets the right
    # qsense --max_seniority.
    civec = np.asarray(mycas.ci)
    na_e, nb_e = nelecas
    strs_a = fci.cistring.make_strings(range(ncas), na_e)
    strs_b = fci.cistring.make_strings(range(ncas), nb_e)
    civec = civec.reshape(len(strs_a), len(strs_b))
    sen_w, cfg_w = {}, {}
    for ia, A in enumerate(strs_a):
        row = civec[ia]
        for ib, B in enumerate(strs_b):
            w = row[ib] * row[ib]
            if w < 1e-12:
                continue
            singly = int(A) ^ int(B)
            doubly = int(A) & int(B)
            Om = bin(singly).count('1')
            sen_w[Om] = sen_w.get(Om, 0.0) + w
            cfg_w[(doubly, singly)] = cfg_w.get((doubly, singly), 0.0) + w

    def occ_string(doubly, singly, n):
        return ''.join('2' if (doubly >> p) & 1 else ('1' if (singly >> p) & 1 else '0')
                       for p in range(n))

    def n_csf(Omega, S):             # # spin-S CSFs in a seniority-Omega config
        if Omega < 2 * S:
            return 0
        return int(round((2 * S + 1) / (Omega / 2 + S + 1)
                         * math.comb(Omega, int(Omega / 2 - S))))

    print('\n=== seniority distribution (Omega = # singly-occupied) ===')
    for Om in sorted(sen_w):
        print(f'  seniority {Om:>2d} : {sen_w[Om]:>7.4f}  {"#" * int(round(60 * sen_w[Om]))}')
    w_le = sum(v for k, v in sen_w.items() if k <= max_seniority_probe)
    print(f'  cumulative weight seniority <= {max_seniority_probe} : {w_le:.4f}')

    print('\n=== dominant spatial configurations (2=doubly 1=singly 0=empty, orb 0..9) ===')
    top_cfg = sorted(cfg_w.items(), key=lambda kv: -kv[1])[:8]
    print(f"{'weight':>8} {'Omega':>6}  occupation")
    for (doubly, singly), w in top_cfg:
        print(f'{w:>8.4f} {bin(singly).count("1"):>6d}  {occ_string(doubly, singly, ncas)}')
    lead_doubly, lead_singly = top_cfg[0][0]
    lead_Om = bin(lead_singly).count('1')
    lead_ndoc = bin(lead_doubly).count('1')
    print(f'leading config: weight {top_cfg[0][1]:.4f}  seniority {lead_Om}  '
          f'n_doubly {lead_ndoc}  n_singlet_CSF {n_csf(lead_Om, 0)}')

    # leading Slater determinants (spin couplings of the dominant configuration)
    dets = sorted(mycas.fcisolver.large_ci(mycas.ci, ncas, nelecas,
                                           tol=LARGE_CI_TOL, return_strs=False),
                  key=lambda t: -abs(t[0]))
    print(f'\n=== leading Slater determinants (|coeff| > {LARGE_CI_TOL:g}) ===')
    print(f"{'weight':>8} {'coeff':>9}   alpha-occ / beta-occ (active idx)   #doubly")
    for c, oa, ob in dets[:6]:
        oa = [int(x) for x in oa]; ob = [int(x) for x in ob]
        print(f'{c*c:>8.4f} {c:>+9.4f}   a={oa} b={ob}   {len(set(oa) & set(ob))}')

    # spin ladder: lowest root of each Ms=S sector (na,nb)=(5+S,5-S), at the
    # CASSCF orbitals. For an AFM, lowest root of Ms=S == lowest total-spin-S.
    print('\n=== spin ladder S=0..5 (vertical CASCI @ CASSCF orbitals) ===')
    print(f"{'S':>3} {'(na,nb)':>9} {'E (Ha)':>16} {'dE (cm^-1)':>12} {'<S^2>':>8}")
    na0 = sum(nelecas) // 2
    E_vert = []
    for S in range(6):
        ne = (na0 + S, na0 - S)
        casci = mcscf.CASCI(mf_as, ncas, ne)
        casci.fcisolver = fci.direct_spin1.FCI()
        casci.verbose = 0
        casci.kernel(mycas.mo_coeff)
        E_vert.append(casci.e_tot)
    E_vert = np.array(E_vert)
    for S in range(6):
        print(f'{S:>3} {f"({na0+S},{na0-S})":>9} {E_vert[S]:>16.8f} '
              f'{(E_vert[S]-E_vert[0])*HA2CM:>12.1f} {S*(S+1):>8.3f}')
    print(f'lowest spin (vertical) : S = {int(np.argmin(E_vert))}')

    # state-specific (variational) ladder + Heisenberg J fit  E(S)=E0+J*S(S+1)
    if STATE_SPECIFIC_LADDER:
        print('\n=== state-specific ladder (CASSCF per spin) ===')
        print(f"{'S':>3} {'E (Ha)':>16} {'dE (cm^-1)':>12} {'<S^2>':>8}")
        E_ss = []
        for S in range(6):
            ne = (na0 + S, na0 - S)
            try:
                cs = mcscf.CASSCF(mf_as, ncas, ne)
                cs.verbose = 0
                cs.kernel(mycas.mo_coeff)
                s2, _ = cs.fcisolver.spin_square(cs.ci, ncas, ne)
                E_ss.append(cs.e_tot)
                dE = 0.0 if S == 0 else (cs.e_tot - E_ss[0]) * HA2CM
                print(f'{S:>3} {cs.e_tot:>16.8f} {dE:>12.1f} {s2:>8.3f}')
            except Exception as exc:
                E_ss.append(np.nan)
                print(f'{S:>3}  CASSCF failed: {exc}')
        E_ss = np.array(E_ss)
        if np.all(np.isfinite(E_ss)):
            hb = heisenberg_analysis(E_ss, HA2CM)     # residual table + J_gap/J_low/J_all/K
            print(f'lowest spin (state-specific)    : S = {int(np.argmin(E_ss))}')
            if SAVE_LADDER_PLOT:
                os.makedirs(LADDER_PLOT_PATH, exist_ok=True)
                plot_spin_ladder(E_ss, HA2CM, ladder_plot_out,
                                 title_tag=r"Fe$_2$S$_2$ (10e,10o) CASSCF")
    
    # ---------------------------------------------------------------------------
    # A. Per-orbital odd-parity weights, joint all-odd weight,
    #    and dominant parity bit-strings within Omega=8 and Omega=6
    # ---------------------------------------------------------------------------
    print('\n=== Per-orbital odd-parity probabilities w_p^(-) ===')
    # w_p^(-) = probability that orbital p is singly occupied
    w_odd = np.zeros(ncas)
    for ia, A in enumerate(strs_a):
        row = civec[ia]
        for ib, B in enumerate(strs_b):
            w = row[ib] ** 2
            if w < 1e-15:
                continue
            singly = int(A) ^ int(B)
            for p in range(ncas):
                if (singly >> p) & 1:
                    w_odd[p] += w

    print(f"{'orbital':>8} {'w_p^(-)':>10}")
    for p in range(ncas):
        print(f'{p:>8d} {w_odd[p]:>10.6f}')
    print(f'{"product":>8} {np.prod(w_odd):>10.6f}  (not the joint weight)')

    # Joint all-odd weight = W_{Omega=10} (cross-check)
    w_all_odd = sen_w.get(10, 0.0)
    print(f'\nJoint all-odd weight (= W_{{Omega=10}}): {w_all_odd:.6f}')
    print(f'Sum of w_p^(-): {np.sum(w_odd):.6f}  (should equal <Omega> = {sum(k*v for k,v in sen_w.items()):.6f})')

    # Dominant parity bit-string sectors within Omega=8 and Omega=6
    # A "parity bit-string" is the 10-bit pattern of singly(1) vs doubly/empty(0)
    print('\n=== Dominant local-seniority bit-string sectors within Omega=8 ===')
    bitstring_w = {}  # key = (Omega, singly_pattern_as_int) -> weight
    for ia, A in enumerate(strs_a):
        row = civec[ia]
        for ib, B in enumerate(strs_b):
            w = row[ib] ** 2
            if w < 1e-15:
                continue
            singly = int(A) ^ int(B)
            Om = bin(singly).count('1')
            bitstring_w[(Om, singly)] = bitstring_w.get((Om, singly), 0.0) + w

    for target_Om in [8, 6]:
        entries = [(pat, wt) for (Om, pat), wt in bitstring_w.items() if Om == target_Om]
        entries.sort(key=lambda x: -x[1])
        n_sectors = len(entries)
        # expected: C(10,8)=45 for Om=8, C(10,6)=210 for Om=6
        print(f'\nOmega={target_Om}: {n_sectors} populated sectors '
            f'(max possible: {math.comb(ncas, target_Om)}), '
            f'total weight = {sum(w for _, w in entries):.6f}')
        print(f"{'rank':>5} {'pattern':>12} {'weight':>10}  doubly/empty orbitals")
        for rank, (pat, wt) in enumerate(entries[:10], 1):
            bits = ''.join('1' if (pat >> p) & 1 else '0' for p in range(ncas))
            non_singly = [p for p in range(ncas) if not ((pat >> p) & 1)]
            print(f'{rank:>5d} {bits:>12s} {wt:>10.6f}  orbs {non_singly}')


    # ---------------------------------------------------------------------------
    # B. Number of local-seniority bit-string sectors per global seniority
    # ---------------------------------------------------------------------------
    print('\n=== Local-seniority sector count per global seniority ===')
    print(f"{'Omega':>6} {'C(10,Omega)':>12} {'W_Omega':>10}")
    for Om in range(0, ncas + 1, 2):
        n_patterns = math.comb(ncas, Om)
        print(f'{Om:>6d} {n_patterns:>12d} {sen_w.get(Om, 0.0):>10.4e}')

    
    # ---------------------------------------------------------------------------
    # C. Seniority (especially Omega=max) weight for each spin-ladder state
    # ---------------------------------------------------------------------------
    if STATE_SPECIFIC_LADDER:
        print('\n=== Seniority weights across the spin ladder ===')
        print(f"{'S':>3} {'E (Ha)':>16} {'W_Omega=max':>12} {'<Omega>':>8}")
        na0 = sum(nelecas) // 2
        for S in range(6):
            ne_s = (na0 + S, na0 - S)
            cas_s = mcscf.CASSCF(mf_as, ncas, ne_s)
            cas_s.verbose = 0
            cas_s.kernel(mycas.mo_coeff)  # start from S=0 CASSCF orbitals

            ci_s = np.asarray(cas_s.ci)
            strs_a_s = fci.cistring.make_strings(range(ncas), ne_s[0])
            strs_b_s = fci.cistring.make_strings(range(ncas), ne_s[1])
            ci_s = ci_s.reshape(len(strs_a_s), len(strs_b_s))

            max_om = min(sum(ne_s), 2 * ncas - sum(ne_s))  # max possible seniority for this (na, nb)
            sen_w_s = {}
            for ia, A in enumerate(strs_a_s):
                row = ci_s[ia]
                for ib, B in enumerate(strs_b_s):
                    w = row[ib] ** 2
                    if w < 1e-15:
                        continue
                    Om = bin(int(A) ^ int(B)).count('1')
                    sen_w_s[Om] = sen_w_s.get(Om, 0.0) + w

            w_max = sen_w_s.get(max_om, 0.0)
            avg_om = sum(k * v for k, v in sen_w_s.items())
            print(f'{S:>3d} {cas_s.e_tot:>16.8f} {w_max:>12.6f} {avg_om:>8.4f}')

            # Print full distribution for each spin
            for Om in sorted(sen_w_s):
                print(f'      Omega={Om:>2d}: {sen_w_s[Om]:.6f}')

    # ---------------------------------------------------------------------------
    # F. Fe3d weight for each spin-ladder state (state-specific CASSCF)
    # ---------------------------------------------------------------------------
    if STATE_SPECIFIC_LADDER:
        print('\n=== Fe3d weight across the state-specific spin ladder ===')
        print(f"{'S':>3} {'E (Ha)':>16} {'mean w_3d':>10} {'min w_3d':>10} {'max w_3d':>10}")
        na0 = sum(nelecas) // 2
        for S in range(6):
            ne_s = (na0 + S, na0 - S)
            cas_s = mcscf.CASSCF(mf_as, ncas, ne_s)
            cas_s.verbose = 0
            cas_s.kernel(mycas.mo_coeff)
            w_s = fe3d_weight(cas_s.mo_coeff, cas_s.ncore, cas_s.ncas)
            print(f'{S:>3d} {cas_s.e_tot:>16.8f} {w_s.mean():>10.4f} '
                f'{w_s.min():>10.4f} {w_s.max():>10.4f}')
            
    # ---------------------------------------------------------------------------
    # G. Characterization of the 10 inactive (nominally S3p) orbitals
    #    Fe3d weight of inactive MOs: how pure is the Fe3d/S3p separation?
    # ---------------------------------------------------------------------------
    print('\n=== Fe3d character of INACTIVE (core) orbitals ===')
    print('  (low values confirm these are predominantly S3p)')
    C_core = np.asarray(mycas.mo_coeff)[:, :ncore]   # (20, ncore)
    print(f"{'core idx':>9} {'global MO':>10} {'Fe3d weight':>12} {'S3p weight':>12}")
    for i in range(ncore):
        w3d = np.sum(C_core[fe3d_idx, i] ** 2)
        w3p = np.sum(C_core[other_idx, i] ** 2)
        print(f'{i:>9d} {i:>10d} {w3d:>12.4f} {w3p:>12.4f}')

    w3d_core_mean = np.mean([np.sum(C_core[fe3d_idx, i]**2) for i in range(ncore)])
    print(f'\nMean Fe3d weight of inactive orbitals: {w3d_core_mean:.4f}')
    print(f'Mean S3p  weight of inactive orbitals: {1 - w3d_core_mean:.4f}')

    # ---------------------------------------------------------------------------
# LaTeX summary (copy-pastable into a paper/SI)
# ---------------------------------------------------------------------------
if PRINT_LATEX:
    print('\n' + '=' * 70)
    print('=== LaTeX summary (copy-paste ready) ===')
    print('=' * 70)

    # --- ecore comparison table ---
    # --- CASCI S=0/S=5 cross-check vs Li & Chan ---
    print('\n% --- CASCI cross-check vs Li & Chan (unrelaxed orbitals) ---')
    print(r'\begin{tabular}{lrrr}')
    print(r'  \hline')
    print(r'  State & $E_{\mathrm{act}}$ (Ha), this work & $E_{\mathrm{act}}$ (Ha), Li \& Chan & $\langle S^2 \rangle$ \\')
    print(r'  \hline')
    print(f'  $S=0$ & ${mycasci.e_cas:.7f}$ & $-27.887643$ & ${s2_0:.4f}$ \\\\')
    print(f'  $S=5$ & ${mycasci_S5.e_cas:.7f}$ & $-27.890357$ & ${s2_5:.4f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')
    print(f'% |S=0 - Li\\&Chan| = {abs(mycasci.e_cas - (-27.887643)):.2e} Ha, '
          f'|S=5 - Li\\&Chan| = {abs(mycasci_S5.e_cas - (-27.890357)):.2e} Ha')

    # --- Fe3d population weight: unrelaxed CASCI vs CASSCF-relaxed ---
    print('\n% --- Fe3d population of active orbitals: unrelaxed vs CASSCF-relaxed ---')
    print(r'\begin{tabular}{rrrr}')
    print(r'  \hline')
    print(r'  Active idx & Global MO & Unrelaxed CASCI & CASSCF-relaxed \\')
    print(r'  \hline')
    for i in range(mycas.ncas):
        print(f'  {i} & {mycas.ncore+i} & ${w_unrelaxed[i]:.4f}$ & ${w_casscf[i]:.4f}$ \\\\')
    print(r'  \hline')
    print(f'  mean & & ${w_unrelaxed.mean():.4f}$ & ${w_casscf.mean():.4f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')

    # --- Causal control: frozen (active<->core locked) vs unrestricted CASSCF ---
    print('\n% --- Causal control: active<->core (Fe3d<->S3p) rotation frozen vs free ---')
    print(r'\begin{tabular}{lrr}')
    print(r'  \hline')
    print(r'  & $E_{\mathrm{tot}}$ (Ha) & mean Fe3d weight \\')
    print(r'  \hline')
    print(f'  Unrelaxed CASCI ($S=0$) & ${mycasci.e_tot:.8f}$ & ${w_unrelaxed.mean():.4f}$ \\\\')
    print(f'  Frozen CASSCF ($S=0$) & ${mycas_frozen.e_tot:.8f}$ & ${w_frozen.mean():.4f}$ \\\\')
    print(f'  Unrestricted CASSCF ($S=0$) & ${mycas.e_tot:.8f}$ & ${w_casscf.mean():.4f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')
    print(f'% |frozen CASSCF - unrelaxed CASCI| = {abs(mycas_frozen.e_tot - mycasci.e_tot):.2e} Ha')
    print(f'% Energy gained solely from Fe3d$\\leftrightarrow$S3p mixing = '
          f'{mycas.e_tot - mycas_frozen.e_tot:.6f} Ha')
    
    # --- CASSCF summary table ---
    print('\n% --- CASSCF energy summary ---')
    print(r'\begin{tabular}{ll}')
    print(r'  \hline')
    print(r'  Quantity & Value \\')
    print(r'  \hline')
    print(f'  CASSCF $E_{{\\mathrm{{tot}}}}$ & ${mycas.e_tot:.8f}$\\,Ha \\\\')
    print(f'  $\\langle S^2 \\rangle$ & ${ss_gs:.4f}$ \\\\')
    print(f'  $2S+1$ & ${mult_gs:.2f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')

    # --- orbital energies & NOONs ---
    noons_latex = np.linalg.eigvalsh(dm1)[::-1]
    print('\n% --- active orbital energies and NOONs ---')
    print(r'\begin{tabular}{rrrr}')
    print(r'  \hline')
    print(r'  Active idx & Global MO & $\epsilon$ (Ha) & NOON \\')
    print(r'  \hline')
    for i, (e, n) in enumerate(zip(orbene, noons_latex)):
        print(f'  {i} & {ncore + i} & ${e:.6f}$ & ${n:.4f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')

    # --- seniority distribution ---
    print('\n% --- seniority distribution ---')
    print(r'\begin{tabular}{rr}')
    print(r'  \hline')
    print(r'  $\Omega$ & Weight \\')
    print(r'  \hline')
    for Om in sorted(sen_w):
        print(f'  {Om} & ${sen_w[Om]:.4f}$ \\\\')
    print(r'  \hline')
    print(r'\end{tabular}')

    # --- spin ladder (state-specific if available, else vertical) ---
    if STATE_SPECIFIC_LADDER and 'E_ss' in dir() and np.all(np.isfinite(E_ss)):
        print('\n% --- state-specific spin ladder ---')
        print(r'\begin{tabular}{rrrr}')
        print(r'  \hline')
        print(r'  $S$ & $E$ (Ha) & $\Delta E$ (cm$^{-1}$) & $S(S+1)$ \\')
        print(r'  \hline')
        for S in range(len(E_ss)):
            dE = (E_ss[S] - E_ss[0]) * HA2CM
            print(f'  {S} & ${E_ss[S]:.8f}$ & ${dE:.1f}$ & ${S*(S+1):.0f}$ \\\\')
        print(r'  \hline')
        print(r'\end{tabular}')
    elif 'E_vert' in dir():
        print('\n% --- vertical spin ladder ---')
        print(r'\begin{tabular}{rrrr}')
        print(r'  \hline')
        print(r'  $S$ & $E$ (Ha) & $\Delta E$ (cm$^{-1}$) & $S(S+1)$ \\')
        print(r'  \hline')
        for S in range(len(E_vert)):
            dE = (E_vert[S] - E_vert[0]) * HA2CM
            print(f'  {S} & ${E_vert[S]:.8f}$ & ${dE:.1f}$ & ${S*(S+1):.0f}$ \\\\')
        print(r'  \hline')
        print(r'\end{tabular}')
