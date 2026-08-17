"""Choosing the orbital blocks for local-spin quantum numbers.

The CAS(10e,10o) blocks were read off the spin-correlation matrix by eye.  That
does not transfer: for CAS(10e,20o) the d' orbitals may or may not belong to
their Fe's block, for CAS(22e,16o) the bridging S 3p orbitals belong to neither
Fe, and for the tetramer there are four centres and three inequivalent ways to
pair them.

This module separates the three things that were conflated in "confirmed from
the spin-correlation matrix":

  (a) PROPOSE   candidate partitions, cheaply, from C[p,q] alone
                (`spectral_blocks`, `correlation_bipartition`, and exhaustive
                enumeration for small spaces);
  (b) SCREEN    them with <S_A^2>, which is a block sum of C and therefore free
                once the 2-RDM exists (`screen_blocks`);
  (c) EVALUATE  the survivors with the actual local-spin distribution w_A(s),
                which needs operator applications (`evaluate_block`).

Only (c) answers the question that matters -- is S_A sharp? -- so (a) and (b)
are filters, not verdicts.  A partition that maximizes sharpness for the state
it was selected on is still selected and evaluated on the same state; the
roadmap Sec. 8.5 cross-frame and cross-spin checks are the validation, and
`stability_report` runs them.

[AUTHOR: nothing here establishes a CHEMICAL assignment.  A block that gives a
 sharp S_A is a good local-spin block; whether it is "Fe_1" requires the
 orbital metadata (localization output, population analysis).]
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

import fes_symmetry as fs


# ---------------------------------------------------------------- propose

def correlation_bipartition(C: np.ndarray) -> tuple[list[int], list[int]]:
    """Split orbitals by the sign of the most antiferromagnetic mode of C.

    Uses the eigenvector of the algebraically LARGEST eigenvalue of the
    off-diagonal part of C.  For two Hund-coupled centres with within-block
    entries +a and cross-block entries -b, the staggered vector
    (+1..+1, -1..-1) has eigenvalue a(m-1) + bm, which is the largest, not the
    smallest.  (Taking the smallest returns the uniform mode, eigenvalue
    a(m-1) - bm, which puts every orbital in one block -- this is what the
    first version of this function did on the real Fe2S2 state.)
    """
    A = C - np.diag(np.diag(C))
    w, V = np.linalg.eigh(A)
    v = V[:, -1]
    left = [p for p in range(len(v)) if v[p] >= 0]
    right = [p for p in range(len(v)) if v[p] < 0]
    return left, right


def spectral_blocks(C: np.ndarray, n_blocks: int, seed: int = 0,
                    n_restarts: int = 20) -> list[list[int]]:
    """k-means on the leading eigenvectors of the off-diagonal part of C.

    For n_blocks = 2 this normally reproduces `correlation_bipartition`.  For
    the four-centre cubane the three inequivalent Fe-Fe pairings show up as
    distinct local minima across restarts; inspect all of them rather than
    taking the first.
    """
    A = C - np.diag(np.diag(C))
    w, V = np.linalg.eigh(A)
    X = V[:, -n_blocks:]
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_restarts):
        centres = X[rng.choice(len(X), n_blocks, replace=False)]
        for _ in range(100):
            d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = np.array([X[lab == k].mean(0) if np.any(lab == k)
                            else centres[k] for k in range(n_blocks)])
            if np.allclose(new, centres):
                break
            centres = new
        cost = d.min(1).sum()
        if cost < best_cost:
            best_cost, best = cost, lab.copy()
    return [sorted(np.flatnonzero(best == k).tolist()) for k in range(n_blocks)]


def equal_bipartitions(norb: int, orbitals=None):
    """All ways to split `orbitals` into two equal halves (up to swap)."""
    orbitals = list(range(norb)) if orbitals is None else list(orbitals)
    m = len(orbitals) // 2
    first = orbitals[0]
    for comb in combinations(orbitals, m):
        if first in comb:
            yield list(comb), [p for p in orbitals if p not in comb]


# ---------------------------------------------------------------- screen

def screen_blocks(C: np.ndarray, candidates) -> list[dict]:
    """Rank candidates by <S_A^2>, which is the block sum of C -- free.

    A block whose <S_A^2> is far below the maximum s_max(s_max+1) for its size
    cannot have a sharp high-spin local moment, so this rules candidates out
    cheaply.  It cannot rule them in: a large <S_A^2> is consistent with a
    broad distribution.
    """
    out = []
    for cand in candidates:
        blocks = [list(b) for b in cand]
        row = {"blocks": blocks, "s2": [], "s2_max": [], "cross": None}
        for b in blocks:
            s2 = float(C[np.ix_(b, b)].sum())
            smax = len(b) / 2.0
            row["s2"].append(s2)
            row["s2_max"].append(smax * (smax + 1))
        if len(blocks) == 2:
            row["cross"] = float(C[np.ix_(blocks[0], blocks[1])].sum())
        row["fraction_of_max"] = float(np.mean(
            [a / b for a, b in zip(row["s2"], row["s2_max"]) if b > 0]))
        out.append(row)
    return sorted(out, key=lambda r: -r["fraction_of_max"])


# ---------------------------------------------------------------- evaluate

def evaluate_block(ci, block, norb, nelec) -> dict:
    """Full local-spin distribution for one block, plus sharpness summaries."""
    w = fs.local_spin_distribution(ci, list(block), norb, nelec)
    s2 = sum(s * (s + 1) * v for s, v in w.items())
    s_star = max(w, key=lambda s: w[s])
    var = sum((s * (s + 1)) ** 2 * v for s, v in w.items()) - s2 ** 2
    return {"block": list(block), "w": w, "S2": s2, "s_star": s_star,
            "w_star": w[s_star], "var_S2": var,
            "entropy": float(-sum(v * np.log(v) for v in w.values() if v > 1e-15))}


def rank_blocks(ci, candidates, norb, nelec, top=None) -> list[dict]:
    """Evaluate candidate partitions by the sharpness of every block in them.

    Two scores are returned, and they do NOT agree:

      w_star     min over blocks of w_A(s*), the sharpness of the local moment;
      frac_max   min over blocks of <S_A^2> / s_max(s_max+1).

    On the real Fe2S2 CAS(10e,10o) singlet, sharpness barely discriminates
    (0.9034 for the true Fe/Fe split vs 0.8981 for the runner-up), because a
    mixed block can still have a sharp -- but lower -- local spin.  The
    fraction-of-maximum separates them decisively (0.964 vs 0.461).  Sort on
    `frac_max` and use `w_star` as the confirmation, not the other way round.
    """
    out = []
    for cand in (candidates[:top] if top else candidates):
        blocks = [list(b) for b in cand]
        evals = [evaluate_block(ci, b, norb, nelec) for b in blocks]
        joint = None
        if len({p for b in blocks for p in b}) == sum(len(b) for b in blocks):
            joint = fs.joint_local_spin_weight(
                ci, blocks, [e["s_star"] for e in evals], norb, nelec)
        frac = min(e["S2"] / ((len(b) / 2.0) * (len(b) / 2.0 + 1.0))
                   for b, e in zip(blocks, evals))
        out.append({"blocks": blocks, "evals": evals, "joint": joint,
                    "w_star": min(e["w_star"] for e in evals),
                    "frac_max": frac})
    return sorted(out, key=lambda r: -r["frac_max"])


def stability_report(states: dict, blocks, norb, nelec_by_state) -> dict:
    """Re-evaluate one partition across frames and/or spin states.

    `states` maps a label ("CASSCF S=0", "rotated S=0", "CASSCF S=2", ...) to a
    CI vector; `nelec_by_state` maps the same labels to (na, nb).  A partition
    selected on one state and one frame is only established if w_A(s*) stays
    high across these.
    """
    out = {}
    for label, ci in states.items():
        ne = nelec_by_state[label]
        out[label] = [evaluate_block(ci, b, norb, ne) for b in blocks]
    return out
