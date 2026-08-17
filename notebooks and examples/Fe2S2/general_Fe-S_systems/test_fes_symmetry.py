"""Tests for fes_symmetry.  Run:  python -m pytest test_fes_symmetry.py -q

Each test checks a specific claim against an independently derivable truth:
  T2  combinatorial sector dimensions vs brute-force determinant enumeration
  T2b sum_S csf_dim(S) == det_dim(M_S=0)   (telescoping Weyl identity)
  T2c the two dimension claims asserted in Fe2S2_characterization
  T5  block = whole space  =>  S_A^2 == pyscf spin_square
  T3  product-form vs moment-form local spin distribution
  T4  non-interacting blocks  =>  w_A(s) is a delta
  E/F ionic and closed-shell references with analytically known answers
  T7  sum_pq C[p,q] == <S^2>  (valid for ANY state, not only singlets)
  --  projector idempotency, sector weight consistency, PHP variational bound
"""

import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import cistring, direct_spin1, spin_op

import fes_symmetry as fs


# ---------------------------------------------------------------- fixtures

def random_ci(norb, nelec, seed=0):
    na, nb = nelec
    da = len(cistring.make_strings(range(norb), na))
    db = len(cistring.make_strings(range(norb), nb))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((da, db))
    return v / np.linalg.norm(v)


def block_diagonal_hamiltonian(norb, blocks, seed=0):
    """h1, eri with no excitation operator connecting different blocks."""
    rng = np.random.default_rng(seed)
    h1 = np.zeros((norb, norb))
    L = np.zeros((norb, norb, 3))
    for b in blocks:
        b = list(b)
        m = rng.standard_normal((len(b), len(b)))
        h1[np.ix_(b, b)] = m + m.T
        for k in range(3):
            m2 = rng.standard_normal((len(b), len(b)))
            L[np.ix_(b, b, [k])] = ((m2 + m2.T) / 2)[:, :, None]
    eri = np.einsum('pqk,rsk->pqrs', L, L)
    return h1, eri


def h4_system(r=1.4):
    mol = gto.M(atom=[('H', (0, 0, i * r)) for i in range(4)],
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run()
    norb = mol.nao
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e, ci = fci.direct_spin1.kernel(h1, eri, norb, (2, 2))
    return h1, eri, norb, (2, 2), e + mol.energy_nuc(), np.asarray(ci)


# ---------------------------------------------------------------- T2

@pytest.mark.parametrize("norb,nelec", [(6, (3, 3)), (8, (4, 4)), (8, (5, 3)),
                                        (10, (5, 5)), (6, (4, 2))])
def test_sector_dim_matches_brute_force_mask(norb, nelec):
    rng = np.random.default_rng(norb * 10 + nelec[0])
    specs = [
        fs.all_odd(range(norb)),
        fs.all_odd(range(0, norb, 2)),
        fs.pair_parities([(0, norb - 1), (1, norb - 2)], +1),
        fs.pair_parities([(0, 1)], -1),
        fs.block_parities([tuple(range(norb // 2))], [+1]),
        fs.SectorSpec([fs.ParityConstraint((int(p),), int(s))
                       for p, s in zip(rng.choice(norb, 3, replace=False),
                                       rng.choice([-1, 1], 3))]),
    ]
    for spec in specs:
        mask = fs.sector_mask(spec, norb, nelec)
        assert fs.sector_dimensions(spec, norb, nelec)["det_dim"] == int(mask.sum())


def test_full_space_dim():
    assert fs.sector_dimensions(None, 10, (5, 5))["det_dim"] == 63504


@pytest.mark.parametrize("norb,nelec", [(6, (3, 3)), (8, (4, 4)), (10, (5, 5))])
def test_csf_dims_sum_to_determinant_dim(norb, nelec):
    """sum over S of the spin-S CSF count == M_S=0 determinant count."""
    for spec in [None, fs.all_odd(range(norb)),
                 fs.pair_parities([(0, 1), (2, 3)], +1)]:
        det = fs.sector_dimensions(spec, norb, nelec)["det_dim"]
        tot = sum(fs.sector_dimensions(spec, norb, nelec, two_s=k)["csf_dim"]
                  for k in range(0, norb + 1))
        assert det == tot


def test_documented_fe2s2_dimensions():
    """The two counts asserted in Fe2S2_characterization, now checked."""
    norb, nelec = 10, (5, 5)
    assert fs.sector_dimensions(fs.all_odd(range(10)), norb, nelec)["det_dim"] == 252
    quartets = fs.pair_parities([(0, 9), (1, 5), (2, 6), (3, 7), (4, 8)], +1)
    assert fs.sector_dimensions(quartets, norb, nelec)["det_dim"] == 4304


def test_omega10_singlet_decomposition():
    """The 1 / 16 / 25 claim in the local_spins.py docstring."""
    norb, nelec = 10, (5, 5)
    spec = fs.all_odd(range(10))
    assert fs.sector_dimensions(spec, norb, nelec, two_s=0)["csf_dim"] == 42


# ---------------------------------------------------------------- T5

@pytest.mark.parametrize("norb,nelec", [(4, (2, 2)), (6, (3, 3)), (6, (4, 2))])
def test_full_block_s2_equals_pyscf_spin_square(norb, nelec):
    ci = random_ci(norb, nelec, seed=1)
    mine = fs.local_s2_expectation(ci, range(norb), norb, nelec)
    ref = spin_op.spin_square0(ci, norb, nelec)[0]
    assert abs(mine - ref) < 1e-10


def test_spin_correlation_matrix_from_rdms():
    h1, eri, norb, nelec, _, ci = h4_system()
    dm1, dm2 = direct_spin1.make_rdm12(ci, norb, nelec)
    C = fs.spin_correlation_matrix(dm1, dm2)
    assert np.allclose(C, C.T, atol=1e-10)
    # T7: total, valid for any state (the singlet row-sum rule is a special case)
    assert abs(C.sum() - spin_op.spin_square0(ci, norb, nelec)[0]) < 1e-9
    # block sums must agree with the operator route
    for block in [[0, 1], [2, 3], [0, 2]]:
        assert abs(C[np.ix_(block, block)].sum()
                   - fs.local_s2_expectation(ci, block, norb, nelec)) < 1e-9


# ---------------------------------------------------------------- T3

@pytest.mark.parametrize("norb,nelec,block", [(4, (2, 2), [0, 1]),
                                              (6, (3, 3), [0, 1, 2]),
                                              (6, (3, 3), [0, 2, 4])])
def test_distribution_normalization_and_moment(norb, nelec, block):
    ci = random_ci(norb, nelec, seed=2)
    w = fs.local_spin_distribution(ci, block, norb, nelec)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > -1e-9 for v in w.values())
    s2 = sum(s * (s + 1) * v for s, v in w.items())
    assert abs(s2 - fs.local_s2_expectation(ci, block, norb, nelec)) < 1e-8
    wm = fs.moment_local_spin_distribution(ci, block, norb, nelec)
    assert max(abs(w[s] - wm[s]) for s in w) < 1e-7


def test_projector_idempotency():
    norb, nelec, block = 6, (3, 3), [0, 1, 2]
    ci = random_ci(norb, nelec, seed=3)
    p = fs.local_spin_projector_apply(ci, block, 1.5, norb, nelec)
    pp = fs.local_spin_projector_apply(p, block, 1.5, norb, nelec)
    assert np.allclose(p, pp, atol=1e-9)
    w = fs.local_spin_distribution(ci, block, norb, nelec)
    assert abs(float(p.ravel() @ p.ravel()) - w[1.5]) < 1e-9


# ---------------------------------------------------------------- T4

def test_non_interacting_blocks_give_sharp_local_spin():
    norb, nelec = 6, (3, 3)
    blocks = [[0, 1, 2], [3, 4, 5]]
    h1, eri = block_diagonal_hamiltonian(norb, blocks, seed=7)
    _, ci = fci.direct_spin1.kernel(h1, eri, norb, nelec, nroots=1)
    ci = np.asarray(ci)
    for b in blocks:
        w = fs.local_spin_distribution(ci, b, norb, nelec)
        assert max(w.values()) > 1 - 1e-7, w


# ---------------------------------------------------------------- E / F

def _determinant_ci(norb, nelec, occ_a, occ_b):
    strs_a = cistring.make_strings(range(norb), nelec[0])
    strs_b = cistring.make_strings(range(norb), nelec[1])
    ia = int(np.searchsorted(strs_a, sum(1 << p for p in occ_a)))
    ib = int(np.searchsorted(strs_b, sum(1 << p for p in occ_b)))
    v = np.zeros((len(strs_a), len(strs_b)))
    v[ia, ib] = 1.0
    return v


def test_ionic_reference():
    """|3 alpha in A, 3 beta in B>: every quantity is known analytically."""
    norb, nelec = 6, (3, 3)
    A, B = [0, 1, 2], [3, 4, 5]
    ci = _determinant_ci(norb, nelec, A, B)
    dm1, dm2 = direct_spin1.make_rdm12(ci, norb, nelec)
    C = fs.spin_correlation_matrix(dm1, dm2)
    expect = np.full((norb, norb), -0.25)
    for p in range(norb):
        for q in range(norb):
            if (p < 3) == (q < 3):
                expect[p, q] = 0.75 if p == q else 0.25
    assert np.allclose(C, expect, atol=1e-9)
    assert abs(C.sum() - 3.0) < 1e-9          # <S^2> = M_S(M_S+1) + N_beta
    for blk in (A, B):
        w = fs.local_spin_distribution(ci, blk, norb, nelec)
        assert abs(w[1.5] - 1.0) < 1e-9
    assert abs(fs.joint_local_spin_weight(ci, [A, B], [1.5, 1.5], norb, nelec)
               - 1.0) < 1e-9


def test_closed_shell_reference():
    """Block A doubly occupied, block B empty: S_A = S_B = 0, C = 0."""
    norb, nelec = 6, (3, 3)
    A, B = [0, 1, 2], [3, 4, 5]
    ci = _determinant_ci(norb, nelec, A, A)
    dm1, dm2 = direct_spin1.make_rdm12(ci, norb, nelec)
    assert np.allclose(fs.spin_correlation_matrix(dm1, dm2), 0.0, atol=1e-9)
    for blk in (A, B):
        w = fs.local_spin_distribution(ci, blk, norb, nelec)
        assert abs(w[0.0] - 1.0) < 1e-9
    assert abs(fs.joint_local_spin_weight(ci, [A, B], [1.5, 1.5], norb, nelec)) < 1e-9


# ---------------------------------------------------------------- sectors on a real state

def test_sector_weight_and_parity_consistency():
    h1, eri, norb, nelec, _, ci = h4_system()
    w_p = fs.orbital_parity_weights(ci, norb, nelec)
    sen = fs.seniority_weights(ci, norb, nelec)
    assert abs(sum(sen.values()) - 1.0) < 1e-10
    assert abs(sum(w_p) - sum(om * wt for om, wt in sen.items())) < 1e-10
    joint = fs.sector_weight(ci, fs.sector_mask(fs.all_odd(range(norb)), norb, nelec))
    assert abs(joint - sen.get(norb, 0.0)) < 1e-12
    assert joint <= min(w_p) + 1e-12


def test_projected_spectrum_and_leakage():
    h1, eri, norb, nelec, e_fci, ci = h4_system()
    ecore = e_fci - float(np.asarray(ci).ravel() @ fs._h_apply(
        ci, h1, eri, norb, nelec).ravel())

    full_mask = np.ones_like(fs.sector_mask(fs.all_odd([]), norb, nelec))
    w, _, _ = fs.projected_spectrum(full_mask, h1, eri, norb, nelec, ecore=ecore,
                                    target_two_s=0)
    assert abs(w[0] - e_fci) < 1e-8
    assert fs.leakage(ci, full_mask, h1, eri, norb, nelec)["leakage_ha2"] < 1e-16

    spec = fs.all_odd(range(norb))
    mask = fs.sector_mask(spec, norb, nelec)
    w2, _, info = fs.projected_spectrum(mask, h1, eri, norb, nelec, ecore=ecore,
                                        target_two_s=0)
    assert abs(info['s2'][0]) < 1e-6          # penalty really enforced S=0
    assert w2[0] >= e_fci - 1e-10                       # variational bound
    lk = fs.leakage(ci, mask, h1, eri, norb, nelec, energy_scale=1.0)
    assert lk["leakage_ha2"] > 0


def test_compression_ratio_monotone_in_generators():
    norb, nelec = 10, (5, 5)
    prev = 1.0
    for k in range(1, 6):
        spec = fs.all_odd(range(k))
        c = fs.compression_ratio(spec, norb, nelec, two_s=0)["compression_det"]
        assert c >= prev - 1e-12
        prev = c


def test_spin_model_fit_recovers_exact_heisenberg():
    S = np.arange(6)
    J = 2.5e-3
    E = 0.5 * J * S * (S + 1)
    out = fs.fit_spin_models(E, S, 2.5, 2.5)
    assert abs(out["J_bilinear"] - J) < 1e-12
    assert out["omega_bilinear"] < 1e-8
    assert abs(out["K_biquadratic"]) < 1e-12


# ---------------------------------------------------------------- T9 regression
# Pins the published CAS(10e,10o) numbers so the generalization refactor can be
# shown not to move them.  Skipped if the FCIDUMP is not mounted.

FE2S2 = "/mnt/user-data/uploads/fe2s2_10e10o_FCIDUMP"


@pytest.mark.skipif(not __import__("os").path.exists(FE2S2),
                    reason="Fe2S2 FCIDUMP not available")
def test_fe2s2_regression():
    from pyscf import fci as _fci
    from pyscf.tools import fcidump
    d = fcidump.read(FE2S2, verbose=False)
    norb, nelec = d["NORB"], (5, 5)
    h1, eri, ec = d["H1"], ao2mo.restore(1, d["H2"], norb), d["ECORE"]
    plain = _fci.direct_spin1.FCI()
    plain.max_cycle, plain.conv_tol = 10000, 1e-10
    e0, ci0 = plain.kernel(h1, eri, norb, nelec)
    s = _fci.addons.fix_spin_(_fci.direct_spin1.FCI(), shift=0.2, ss=0)
    s.max_cycle, s.conv_tol = 10000, 1e-12
    e, ci = s.kernel(h1, eri, norb, nelec, ci0=np.asarray(ci0))
    ci = np.asarray(ci)

    assert abs(e + ec - (-116.517180)) < 1e-6            # reported E_FCI
    sen = fs.seniority_weights(ci, norb, nelec)
    assert abs(sen[10] - 0.90711) < 1e-4                 # reported W_Omega=10
    om = sum(k * v for k, v in sen.items())
    assert abs(om - 9.785) < 1e-3                        # reported <Omega>
    assert abs(sum(k * k * v for k, v in sen.items()) - om ** 2 - 0.502) < 1e-3

    om_vals = fs.seniority_values(norb, nelec).astype(float)
    assert abs(fs.commutator_norm2(ci, om_vals, h1, eri, norb, nelec)
               - 0.0733) < 5e-4                          # reported NC score

    M = fs.nc_matrix(ci, h1, eri, norb, nelec)
    assert abs(M[4, 5] - 4.68e-4) < 1e-5                 # the s_{4,5} outlier
    cross = M[:5, 5:]
    within = np.concatenate([M[:5, :5][np.triu_indices(5, 1)],
                             M[5:, 5:][np.triu_indices(5, 1)]])
    assert abs(cross.mean() - 0.0316) < 5e-4             # reported statistics
    assert abs(within.mean() - 0.0380) < 5e-4
    assert abs(cross.min() - 0.00047) < 5e-5
    assert abs(within.min() - 0.0144) < 5e-4

    # local spin, blocks read from the correlation matrix
    import blocks as bl
    dm1, dm2 = direct_spin1.make_rdm12(ci, norb, nelec)
    C = fs.spin_correlation_matrix(dm1, dm2)
    assert bl.correlation_bipartition(C) == ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    w = fs.local_spin_distribution(ci, [0, 1, 2, 3, 4], norb, nelec)
    assert abs(w[2.5] - 0.903431) < 1e-5
    # singlet identity: S_A = 5/2 forces S_B = 5/2, so the joint weight equals it
    assert abs(fs.joint_local_spin_weight(
        ci, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], [2.5, 2.5], norb, nelec)
        - w[2.5]) < 1e-9


@pytest.mark.skipif(not __import__("os").path.exists(FE2S2),
                    reason="Fe2S2 FCIDUMP not available")
def test_mask_equals_ffsim_operator_exactly():
    """The parity mask is the projector, up to floating-point rounding."""
    ffsim = pytest.importorskip("ffsim")
    pytest.importorskip("openfermion")
    norb, nelec = 6, (3, 3)
    ci = random_ci(norb, nelec, seed=5)
    for orbs, eig in [((0,), -1), ((1,), +1), ((0, 3), +1), ((2, 4), -1)]:
        mask = fs.sector_mask(fs.SectorSpec([fs.ParityConstraint(orbs, eig)]),
                              norb, nelec)
        a = fs.project(ci, mask)
        b = fs.ffsim_parity_projector(ci, orbs, eig, norb, nelec)
        assert np.max(np.abs(a - np.real(b))) < 1e-15


def test_local_spin_sector_dimensions():
    """Verifies the 1 / 16 / 25 decomposition asserted in local_spins.py."""
    norb, nelec = 10, (5, 5)
    A, B = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    # inside Omega = 10 each block has 5 singly occupied orbitals, so the count
    # of local-spin-s states is n_csf(5, 2s): 1, 4, 5 for s = 5/2, 3/2, 1/2.
    assert [fs.n_csf(5, k) for k in (5, 3, 1)] == [1, 4, 5]
    assert 1 * 1 + 4 * 4 + 5 * 5 == fs.sector_dimensions(
        fs.all_odd(range(10)), norb, nelec, two_s=0)["csf_dim"]
    # the S_A = S_B = 5/2 sector is one-dimensional per total spin, but
    # six-dimensional in the M_S = 0 determinant space (it spans S = 0..5)
    d = fs.local_spin_sector_dimensions([A, B], [2.5, 2.5], norb, nelec, two_s=0)
    assert d == {"det_dim": 6, "csf_dim": 1}
    for S in range(6):
        dd = fs.local_spin_sector_dimensions([A, B], [2.5, 2.5], norb,
                                             (5 + S, 5 - S), two_s=2 * S)
        assert dd == {"det_dim": 6 - S, "csf_dim": 1}


def test_rank_one_sector_overlap_equals_weight():
    """For a sector that is 1-D inside the target spin block, |<psi|Phi>|^2 = W(P)."""
    h1, eri, norb, nelec, e_fci, ci = h4_system()
    A, B = [0, 1], [2, 3]
    P = fs.local_spin_projector([A, B], [1.0, 1.0], norb, nelec)
    d = fs.local_spin_sector_dimensions([A, B], [1.0, 1.0], norb, nelec, two_s=0)
    if d["csf_dim"] != 1:
        pytest.skip("sector is not rank one for this system")
    w, V, info = fs.projected_spectrum_general(
        P, d["det_dim"], np.asarray(ci).shape, h1, eri, norb, nelec,
        reference_states=[ci], target_two_s=0)
    assert abs(info["overlaps"][0, 0] - fs.sector_weight(ci, P)) < 1e-9
