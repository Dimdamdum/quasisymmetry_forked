# Status: switching Fe2S2 seniority_oo from FCI to DMRG reference

Pin/handoff note for where this investigation currently stands. See also
`dmrg_nc_gradient_plan.md` (separate doc: sketch for an analytic NC gradient,
not implemented).

## Goal

Re-run the Fe2S2 `hamiltonians/Fe2S2/seniority_oo` orbital optimization
(`optimize_symmetries.py`, `parity_10_sens.txt`, `--cost_function NC`) with
`--reference dmrg` instead of `--reference fci`, to validate the DMRG-native
path before using it on systems too large for FCI.

## Findings, in the order they were discovered

1. **I/O bottleneck (fixed by workaround, not code change).** `/workspace` is
   a 9p mount (WSL2 bridge to the Windows host filesystem). block2 stores
   each DMRG wavefunction as tens of thousands of small per-site/per-block
   files; every file op over 9p is slow. A single store dir had 51,658 files
   at bond_dim 20. Moving `--wavefunction_dir` to a native path
   (`/root/dmrg_wavefunctions/...`) gave a ~12x speedup on cost evaluations.
   **Always point `--wavefunction_dir` outside `/workspace` for any DMRG run
   in this environment.**

2. **RNG-purity bug (fixed in code, committed).** `DMRGOrbitalCosts.commutator`
   / `.variance` (`src/dmrg_costs.py`) were not pure functions of `x`: every
   MPO-MPS multiply (`apply_mpo`, `src/dmrg_solver.py:808`) draws a fresh
   random MPS from block2's *global*, unseeded RNG as its compression
   starting guess. Same `x` evaluated twice could return different cost
   values (measured: `0.16308442` / `0.16313670` / `0.16308317` at identical
   `x=0`, bond_dim=20). This injected noise directly into every
   finite-difference gradient (no analytic gradient exists for this cost —
   see `dmrg_nc_gradient_plan.md`). Fix: reseed `block2.Random.rand_seed(...)`
   to a fixed constant at the top of both cost functions. Verified
   bit-identical results across repeated calls at the same `x` post-fix.
   This bug, not bond dimension per se, was the main cause of the optimizer
   either moving the wrong direction or reporting spurious convergence
   (`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`, which just means
   "stopped changing," not "found a minimum") at low bond dimension.

3. **Bond-dimension scan, `--multiply_sweeps 8`, `--optimizer_maxiter 3` each
   (finite differences, no analytic gradient):**

   | bond_dim | pre/post fix | cost trajectory | nit | nfev | evals/iter | elapsed | s/eval |
   |---|---|---|---|---|---|---|---|
   | 250 | (fix not yet needed to look healthy) | 0.177352→0.177166 (real decrease) | 3 | 230 | ~77 | 18279s | 79.5 |
   | 125 | pre-fix | 0.177194→0.177192 (barely moving) | 3 | 460 | ~153 | 20377s | 44.3 |
   | 63  | pre-fix | 0.172410→0.172410 (flat/increase) | 2 | 782 | ~391 | 22309s | 28.5 | **broken** |
   | 63  | post-fix | 0.172395→0.172379 (real decrease) | 3 | 506 | ~169 | 15057s | 29.8 | **fixed** |
   | 20  | pre-fix | 0.163084→0.163107 (increase, ABNORMAL at ms=2) | varies | — | — | — | — | **broken** |
   | 20  | post-fix | 0.163147→0.163049 (real decrease) | 3 | 1472 | ~491 | 32131s | 21.8 | **fixed but slowest overall** |

   Post-fix, evals/iteration still climbs sharply as bond_dim drops
   (77→169→491) even though the qualitative breakage is gone — likely because
   L-BFGS-B's finite-difference step (`eps=1e-8`, fixed absolute, not
   noise-adaptive) is small enough that residual bond-dim-dependent
   compression imprecision still makes the line search backtrack repeatedly.
   Net effect: **`bond_dim=20` was the slowest in wall-clock time of the
   three post-fix points (32131s), despite the cheapest per-eval cost
   (21.8s)** — `bond_dim=63` was fastest overall (15057s) and looked
   healthiest. **`bond_dim=63` (or nearby, e.g. 75-100) is the best candidate
   so far for a production bond dimension**, not 250 and not 20.
   `bond_dim=250` was not re-tested post-fix (it already looked healthy
   pre-fix, since the true signal apparently dominates the RNG noise there).

4. **`--orbene_npy` implemented for `--reference dmrg`** (previously hard
   blocked). Builds the generalized-Fock diagonal from the fixed DMRG
   reference's 1-RDM (`solver.driver.get_1pdm`), rotated via
   `U_opt.T @ rdm1 @ U_opt` (convention validated empirically against the
   FCI-path's own orbenes output for a shared test rotation — matched to
   ~0.0009, attributable to bond_dim=20 approximation, not a convention bug).

## Committed (this pin)

- `src/dmrg_costs.py` — RNG reseed fix.
- `optimize_symmetries.py` — `--orbene_npy` support for `--reference dmrg`.
- `run_oo_chained.py` (new) — attempt-chaining wrapper reproducing the
  original FCI run's pattern (repeated `--optimizer_maxiter`-capped attempts,
  `x_opt` fed forward, per-attempt logs + `driver.log` + `summary.json`) for
  any `optimize_symmetries.py` reference/cost combination.
- `dmrg_nc_gradient_plan.md` — analytic-gradient design sketch (not
  implemented).
- `fe2s2_dmrg_oo_status.md` — this file.

Full test suite (`pytest tests/`, 59 tests) passes with these changes.

## Not committed (scratch, gitignored or intentionally left local)

- All actual run output under `hamiltonians/Fe2S2/seniority_oo/dmrg_*`
  (smoke tests, bond-dim scan attempts) — these are data, not code, matching
  the existing convention that the original FCI run's output in the same
  directory was also never committed.
- DMRG wavefunction stores under `/root/dmrg_wavefunctions/` — local scratch,
  not under `/workspace` at all.

## Open questions / next steps

- Pin down the practical bond-dimension sweet spot more precisely (something
  between 63 and 250 untested; 75-100 is a reasonable next guess).
- Consider whether tuning `finite_diff_rel_step`/`eps` in the
  `scipy.optimize.minimize` call (currently SciPy's fixed default) reduces
  evals/iteration independent of bond dimension — untested hypothesis from
  the line-search analysis above.
- The analytic-gradient project (`dmrg_nc_gradient_plan.md`) would sidestep
  the finite-difference cost entirely if pursued; currently unimplemented,
  multi-day estimate, main open derivation is the quadratic (seniority
  density-density) term's transition-2RDM contraction.
- Once a bond dimension is settled, re-run the full attempt-chain (via
  `run_oo_chained.py`) to actual convergence, not just the 3-iteration
  timing/behavior probes documented above.
