"""Driver: produce the Sec.-8 report of the benchmark roadmap for one
Hamiltonian, one orbital frame, and a user-supplied list of candidate
projectors.

    python run_benchmark.py FCIDUMP config.json

Everything system-specific lives in the config, not in the code:

{
  "label":        "Fe2S2 CAS(22e,16o), localized Fe/S frame",
  "nelec":        [11, 11],
  "spin_ladder":  [0, 1, 2, 3, 4, 5],
  "blocks":       {"FeA": [0,1,2,3,4], "FeB": [5,6,7,8,9]},
  "local_spin_targets": {"FeA": 2.5, "FeB": 2.5},
  "energy_scale_ha": null,
  "projectors": [
    {"label": "P_d,odd", "type": "all_odd", "orbitals": [0,1,2,3,4,5,6,7,8,9]},
    {"label": "Fe-S pairs Q=+1", "type": "pair", "pairs": [[0,10],[1,11]],
     "eigenvalue": 1}
  ]
}

[AUTHOR: `energy_scale_ha` is the normalization the roadmap Sec. 8.4 requires
 but does not fix.  Until it is set, leakage is reported in Ha^2 only and is
 NOT comparable between the (10e,20o) and (22e,16o) Hamiltonians.]

[AUTHOR: `blocks` is the Fe/S orbital assignment for THIS frame.  It must come
 from the orbital metadata shipped with the integrals (localization output,
 population analysis), not from index conventions carried over from another
 active space.  `infer_blocks_from_correlation` below is a diagnostic to check
 an assignment, not a substitute for one.]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from pyscf import ao2mo
from pyscf.fci import direct_spin1, spin_op
from pyscf.tools import fcidump

import fes_symmetry as fs


# ------------------------------------------------------------------ config

def build_spec(entry) -> fs.SectorSpec:
    t = entry["type"]
    if t == "all_odd":
        return fs.all_odd(entry["orbitals"], entry["label"])
    if t == "pair":
        return fs.pair_parities([tuple(p) for p in entry["pairs"]],
                                entry.get("eigenvalue", 1), entry["label"])
    if t == "block":
        return fs.block_parities(entry["blocks"], entry["eigenvalues"],
                                 entry["label"])
    if t == "explicit":
        return fs.SectorSpec(
            [fs.ParityConstraint(tuple(c["orbitals"]), c["eigenvalue"])
             for c in entry["constraints"]], entry["label"])
    raise ValueError(f"unknown projector type {t!r}")


def infer_blocks_from_correlation(C, threshold=0.0):
    """Diagnostic only: connected components of the |C[p,q]| graph.

    Returns candidate blocks.  Reported so an assumed Fe/S assignment can be
    checked against the state; it does not establish a chemical assignment.
    """
    norb = C.shape[0]
    adj = np.abs(C - np.diag(np.diag(C))) > threshold
    seen, blocks = set(), []
    for p in range(norb):
        if p in seen:
            continue
        stack, comp = [p], []
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            comp.append(q)
            stack += [r for r in range(norb) if adj[q, r] and r not in seen]
        blocks.append(sorted(comp))
    return blocks


# ------------------------------------------------------------------ report

def solve_state(h1, eri, norb, nelec, two_s=None, conv_tol=1e-12):
    """Tight, spin-fixed FCI.

    pyscf's default Davidson tolerance is not enough here: on the Fe2S2
    CAS(10e,10o) Hamiltonian the default solve stops at a state with
    <S^2> = 8e-3 that is 4.6e-6 Ha ABOVE the true singlet.  conv_tol=1e-10
    with max_cycle=10000 (the setting already used in optimize_symmetries.
    get_fci) reaches <S^2> = 1e-6; the spin penalty below takes it to 1e-12.
    The seeded two-step is used because starting the penalized solver cold
    converged to an <S^2> = 1.5 state in testing.
    """
    from pyscf import fci as _fci
    plain = _fci.direct_spin1.FCI()
    plain.max_cycle, plain.conv_tol = 10000, 1e-10
    e0, ci0 = plain.kernel(h1, eri, norb, nelec)
    if two_s is None:
        return e0, np.asarray(ci0)
    ss = (two_s / 2.0) * (two_s / 2.0 + 1.0)
    solver = _fci.addons.fix_spin_(_fci.direct_spin1.FCI(), shift=0.2, ss=ss)
    solver.max_cycle, solver.conv_tol = 10000, conv_tol
    e, ci = solver.kernel(h1, eri, norb, nelec, ci0=np.asarray(ci0))
    return e, np.asarray(ci)


def report(h1, eri, ecore, norb, cfg):
    label = cfg.get("label", "unlabelled")
    ladder = cfg.get("spin_ladder", [0])
    na0 = (sum(cfg["nelec"])) / 2.0
    scale = cfg.get("energy_scale_ha")
    specs = [build_spec(e) for e in cfg["projectors"]]
    blocks = {k: list(v) for k, v in cfg.get("blocks", {}).items()}
    targets = cfg.get("local_spin_targets", {})

    print("=" * 78)
    print(f"  {label}")
    print(f"  norb={norb}  nelec={tuple(cfg['nelec'])}  ladder S={ladder}")
    print("=" * 78)

    states, energies = {}, {}
    for S in ladder:
        ne = (int(na0 + S), int(na0 - S))
        e, ci = solve_state(h1, eri, norb, ne, two_s=int(2 * S))
        s2 = spin_op.spin_square0(ci, norb, ne)[0]
        states[S] = (ne, ci)
        energies[S] = e + ecore
        # the M_S=S lowest root is only the lowest total-spin-S state if this
        # value matches S(S+1); it is printed, never assumed.
        flag = "" if abs(s2 - S * (S + 1)) < 1e-3 else "   <-- NOT spin-pure"
        print(f"  S={S}  E={e + ecore:>18.10f} Ha   <S^2>={s2:8.4f}{flag}")

    print("\n-- 8.1 state concentration / 8.2 compression / 8.4 leakage --")
    hdr = (f"  {'projector':<26s}{'gen':>4s}{'W(P)':>12s}{'det dim':>12s}"
           f"{'C_det':>12s}{'leak Ha^2':>13s}")
    print(hdr + ("   leak/scale^2" if scale else ""))
    for S in ladder:
        ne, ci = states[S]
        print(f"  [S={S}]")
        for spec in specs:
            mask = fs.sector_mask(spec, norb, ne)
            W = fs.sector_weight(ci, mask)
            comp = fs.compression_ratio(spec, norb, ne, two_s=int(2 * S))
            lk = fs.leakage(ci, mask, h1, eri, norb, ne, energy_scale=scale)
            line = (f"  {spec.label:<26.26s}{spec.n_generators():>4d}{W:>12.6f}"
                    f"{comp['det_dim_sector']:>12,d}"
                    f"{comp['compression_det']:>12,.1f}"
                    f"{lk['leakage_ha2']:>13.3e}")
            if scale:
                line += f"{lk['leakage_normalized']:>15.3e}"
            print(line)

    print("\n-- 8.3 projected Hamiltonian accuracy --")
    print(f"  {'projector':<26s}{'S':>3s}{'E(PHP)':>20s}{'dE (Ha)':>14s}"
          f"{'dE (cm^-1)':>13s}{'overlap':>10s}")
    print("  [a near-zero overlap with a large compression means the sector is "
          "small AND wrong, not compact]")
    HA2CM = 219474.6313708
    projected = {spec.label: {} for spec in specs}
    for spec in specs:
        for S in ladder:
            ne, ci = states[S]
            mask = fs.sector_mask(spec, norb, ne)
            if not mask.any():
                print(f"  {spec.label:<26.26s}{S:>3d}   EMPTY SECTOR")
                continue
            w, V, info = fs.projected_spectrum(
                mask, h1, eri, norb, ne, ecore=ecore,
                reference_states=[ci], target_two_s=int(2 * S))
            dE = w[0] - energies[S]
            projected[spec.label][S] = w[0]
            bad = "" if abs(info["s2"][0] - S * (S + 1)) < 1e-3 else "  <-- wrong S"
            print(f"  {spec.label:<26.26s}{S:>3d}{w[0]:>20.10f}{dE:>14.6f}"
                  f"{dE * HA2CM:>13.1f}{info['overlaps'][0, 0]:>10.6f}{bad}")

    if len(blocks) == 2 and targets:
        print("\n-- 8.1 local spin (full distribution) --")
        for S in ladder:
            ne, ci = states[S]
            dm1, dm2 = direct_spin1.make_rdm12(ci, norb, ne)
            C = fs.spin_correlation_matrix(dm1, dm2)
            print(f"  [S={S}]  sum_pq C = {C.sum():.6f}  (= <S^2>)")
            for name, blk in blocks.items():
                w = fs.local_spin_distribution(ci, blk, norb, ne)
                s2 = sum(s * (s + 1) * v for s, v in w.items())
                top = " ".join(f"{s:.1f}:{v:.4f}" for s, v in sorted(w.items())
                               if abs(v) > 1e-4)
                print(f"    {name:<6s} <S^2>={s2:8.4f}   {top}")
            names = list(blocks)
            jw = fs.joint_local_spin_weight(
                ci, [blocks[n] for n in names],
                [targets[n] for n in names], norb, ne)
            print(f"    joint P(" +
                  ", ".join(f"S_{n}={targets[n]}" for n in names) +
                  f") = {jw:.6f}")
        print("  [AUTHOR: for >2 centres the joint weight above still works, "
              "but the two-site spin model below does not.]")

    if len(ladder) > 2 and len(blocks) == 2 and targets:
        print("\n-- 8.3 spin-model fit, full vs projected spectra --")
        names = list(blocks)
        sa, sb = (targets[names[0]], targets[names[1]])
        full = fs.fit_spin_models([energies[S] for S in ladder], ladder, sa, sb)
        print(f"  full      J={full['J_bilinear'] * HA2CM:>9.2f} cm^-1  "
              f"omega_bilin={full['omega_bilinear']:.2f}%  "
              f"J'={full['J_biquadratic'] * HA2CM:>9.2f}  "
              f"K={full['K_biquadratic'] * HA2CM:>8.2f}  "
              f"omega_biq={full['omega_biquadratic']:.2f}%")
        for spec in specs:
            got = projected[spec.label]
            if len(got) != len(ladder):
                continue
            p = fs.fit_spin_models([got[S] for S in ladder], ladder, sa, sb)
            print(f"  {spec.label:<26.26s}J={p['J_bilinear'] * HA2CM:>9.2f} cm^-1  "
                  f"omega_bilin={p['omega_bilinear']:.2f}%  "
                  f"J'={p['J_biquadratic'] * HA2CM:>9.2f}  "
                  f"K={p['K_biquadratic'] * HA2CM:>8.2f}  "
                  f"omega_biq={p['omega_biquadratic']:.2f}%")


def main(fcidump_path, config_path):
    d = fcidump.read(fcidump_path, verbose=False)
    norb = d["NORB"]
    h1 = d["H1"]
    eri = ao2mo.restore(1, d["H2"], norb)
    ecore = d["ECORE"]
    cfg = json.load(open(config_path))
    report(h1, eri, ecore, norb, cfg)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("usage: python run_benchmark.py FCIDUMP config.json")
