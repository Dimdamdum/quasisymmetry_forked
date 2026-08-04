# Plan: analytic gradient for the DMRG-native `NC` cost

Status: **design sketch only, not implemented.** Written to scope the effort and
risk before committing engineering time. Every convention below is grounded in
the actual code (cited by file:line) or in a numerical check run against it —
not re-derived from first principles — to avoid the "confidently wrong tensor
convention" failure mode.

## 1. What we're differentiating

`NC(x) = sum_k f_k(x)`, one term per parity-matrix row `k`, evaluated in
`DMRGOrbitalCosts.commutator` (`src/dmrg_costs.py:167-182`):

```
f_k(x) = || [H, S~_k(x)] |psi> ||^2
       = <psi| S~_k H^2 S~_k |psi> + ... (expanded via chi, xi in the code)
```
concretely, using the code's own variables:

```
phi_k = S~_k(x) |psi>              (apply_rotated_parity)
xi_k  = S~_k(x) H|psi>             (apply_rotated_parity on the cached eta = H|psi>)
chi_k = H phi_k                    (apply_mpo)
f_k(x) = <chi_k|chi_k> + <xi_k|xi_k> - 2 Re<chi_k|xi_k>
       = || Delta_k(x) ||^2,   Delta_k(x) := chi_k - xi_k = [H, S~_k(x)]|psi>
```

`S~_k(x) = U(x)^dagger S_k U(x)`, `U(x) = expm(A(x))`, `A(x)` antisymmetric with
free entries only on the allowed `(i,j)` planes (`src/orbital_rotation.py:46-66`,
`params_to_U`). `x` has `N_free = 45` entries for this system (`norb=10`, full
`SO(10)` packing).

**Important, easy to get wrong: `S_k` here is not a bare one-body operator.**
For `parity_10_sens.txt` (identity matrix, one `1` per row, `norb` columns),
`rotated_parity_factor_mpos` (`src/dmrg_solver.py:722-764`) builds each row's
operator via `_spin_parity_factor_mpo(density, alpha=True, beta=True)`
(`src/dmrg_solver.py:701-720`), i.e. the **full seniority operator**:

```
S~_k = 1 - 2 n~_alpha - 2 n~_beta + 4 n~_alpha n~_beta
```

with `n~_alpha`, `n~_beta` sharing one classical one-body coefficient matrix

```
D_k(x) = U(x)^T @ (e_k e_k^T) @ U(x)      [ D_k[q,r] = U[k,q] * U[k,r] ]
```

— this exact formula is read directly off `_rotated_occupation_density`
(`src/dmrg_solver.py:694-699`), not guessed. The `4 n~_alpha n~_beta` term is
**quadratic in `D_k`**, i.e. `S~_k` is not linear in the classical rotation
matrix; it has a one-body piece and a density-density ("two-body-like") piece
(`add_sum_term("cCDd", 4.0 * einsum("qr,st->qstr", D_k, D_k))`,
`src/dmrg_solver.py:716-719`). Any gradient derivation must carry both terms.

**Convention warning, validated empirically (see the `orbene_npy` work):**
`src/dmrg_solver.rotate_integrals` (Hamiltonian integrals, matches ffsim)
transforms as `X_rot = U @ X @ U^T`. `D_k(x)` above and the 1-RDM both
transform as `U^T @ (.) @ U` — the *opposite* direction. These are genuinely
different, coexisting conventions in this codebase; mixing them up is the
most likely way to introduce a silent sign/transpose bug. Any new code must
match `_rotated_occupation_density`'s convention for `D_k`, not
`rotate_integrals`'s.

## 2. Two-layer chain rule

**Layer 1 (classical, cheap, already well-understood math):**
`dU(x)/dx_i`, hence `dD_k(x)/dx_i = (dU/dx_i)^T @ (e_k e_k^T) @ U + U^T @ (e_k e_k^T) @ (dU/dx_i)`.

`A(x)` is real antisymmetric (purely imaginary eigenvalues, conjugate pairs),
so `dU/dx_i` has a closed form via the Fréchet derivative of `expm`
(`scipy.linalg.expm_frechet`, or the eigen-decomposition/Loewner-matrix
formula). At `norb=10` this is trivial — microseconds, no MPS involved. Could
even get away with plain central differences on the 10x10 `U(x)` here, since
this layer is not the bottleneck; no need to be clever.

**Layer 2 (the actual problem): `d f_k / d(one classical matrix D_k)`, propagated
through block2's compression.** This is the layer with no library-level
derivative today (see prior conversation: block2 exposes no `grad`/`deriv`
named method). The adjoint trick below avoids needing one, by not
differentiating the compression step at all.

## 3. The adjoint trick (core idea)

We do **not** need `d(compressed MPS)/dx_i` for each of the 45 directions.
`S~_k(x)` depends on `x` **only** through the classical matrix `D_k(x)`
(a `norb x norb` object, cheap per Layer 1). If we can express
`d f_k/dx_i` as a **fixed** (x-independent-to-recompute) bra contracted
against `dD_k(x)/dx_i` (which Layer 1 gives for all 45 `i` at once, for free),
then the expensive part — MPS-level work — is paid **once per row k**, not
once per parameter.

For the linear (one-body) piece of `S~_k` alone, this is exact and clean:

```
d f_k/dx_i = 2 Re[ <H Delta_k| (d S~_k/dx_i) |psi> - <Delta_k| (d S~_k/dx_i) |H psi> ]
```

(chain rule through `Delta_k = [H, S~_k]|psi>`, using `H` Hermitian to move it
onto the bra). Since `d S~_k/dx_i` is a one-body operator with classical
coefficient matrix `dD_k/dx_i`:

```
<H Delta_k| (one-body, matrix M) |psi>  =  sum_pq M[p,q] * T_k[p,q],
  T_k[p,q] := <H Delta_k| a_p^dagger a_q |psi>
```

`T_k` is a **transition 1-RDM between two fixed MPS** (`H Delta_k` and
`psi`) — computable **once per row**, independent of `i`, via block2's
`DMRGDriver.get_trans_1pdm(bra, ket)` (confirmed to exist and do exactly
this: "Transition 1-Particle Density Matrix between the given bra and ket
MPSs"). Likewise need `U_k[p,q] := <Delta_k| a_p^dagger a_q |H psi>`
(another `get_trans_1pdm` call). Then:

```
d f_k/dx_i (one-body part) = 2 Re Tr[ (dD_k/dx_i)^T @ (T_k - U_k) ]
```

— all 45 components from one classical contraction each, given `T_k - U_k`
computed **once**.

**The quadratic (`4 n~_alpha n~_beta`) piece needs the same trick one order up:**
its contribution to `d f_k/dx_i` will involve `dD_k/dx_i` contracted against a
**transition 2-RDM**-like object between the same fixed bra/ket pairs
(`get_trans_2pdm(bra, ket)`, confirmed to exist in the same API). Not yet
worked out in detail here — this is the main remaining derivation task, not
just bookkeeping. It is very likely tractable (same structural trick, one
index rank up) but must be derived carefully, since this is exactly the kind
of step where a sign or index-transposition error would silently corrupt
results rather than crash.

**Net cost per row k, if this all works out:** the quantities `Delta_k`,
`H Delta_k` are already needed for (or trivially derived from) the existing
forward cost evaluation — `chi_k`, `xi_k` are already computed; `H Delta_k`
needs one more `apply_mpo` (H applied to `chi_k - xi_k`). Then 2-4
`get_trans_1pdm`/`get_trans_2pdm` calls per row. That's roughly a small
constant multiple of one forward-eval's cost to get the **entire 45-component
gradient** — replacing the current ~46 forward evaluations per gradient.
That's the whole point: turning an O(N_free) cost into O(N_sym) (here,
O(10)).

## 4. Known risks / open questions (in rough order of how likely each is to bite)

1. **The quadratic-term transition-2RDM derivation is not done.** Section 3's
   clean result is only for a bare one-body `S_k`; Fe2S2's actual `S_k` is the
   full seniority operator. This is the main remaining math work.
2. **Transition NPDM phase ambiguity.** Block2's own docs warn: "there can be
   an overall phase uncertainty for transition NPDMs." `T_k`, `U_k` must
   combine in a way that's manifestly phase-invariant (or the phase must be
   pinned down explicitly), or the assembled gradient will be silently wrong
   in a way that may not show up as an obvious crash.
3. **Convention consistency**: must use `_rotated_occupation_density`'s
   `U^T (.) U` convention throughout this derivation, not
   `rotate_integrals`'s `U (.) U^T` — see the warning in Section 1. Easy to
   transpose by accident.
4. **Only covers the single-factor-per-row case** (Fe2S2's actual
   `parity_10_sens.txt`, one nonzero per row). General multi-orbital-support
   rows (`--seniority` on other systems, or a denser parity matrix) apply
   several `_spin_parity_factor_mpo` factors sequentially
   (`apply_rotated_parity`, `src/dmrg_solver.py:824-849`); differentiating a
   *product* of several x-dependent operators needs a product-rule sum over
   which factor is perturbed, with the other factors' MPOs still applied
   around it. Not addressed here; would need its own derivation pass if ever
   needed beyond this system's identity-matrix case.
5. **`variance` cost is unaffected by any of this** — it already has a much
   simpler, fully classical gradient path (1-RDM contraction, no MPS ops after
   one initial RDM build) discussed separately; this document is NC-only.

## 5. Validation plan (required before trusting this for anything)

1. Implement the one-body-only piece first (Section 3) in isolation, on a toy
   `S_k` that really is one-body (e.g. a plain occupation-number symmetry, not
   full seniority) so the quadratic-term derivation isn't a confounder.
2. Cross-check the resulting analytic gradient against the current
   finite-difference gradient, component-by-component, on a small case (few
   parameters, low `bond_dim`, fast to iterate).
3. Only then add the quadratic (`cCDd`) term's contribution and re-validate
   the same way against finite differences, on Fe2S2's actual seniority
   operator.
4. Confirm the assembled gradient reproduces `scipy.optimize.minimize`'s
   *existing* finite-difference-driven trajectory closely enough (same
   descent direction, comparable step-to-step cost decrease) on a short run
   before trusting it for a real production optimization.

## 6. Effort estimate

Layer 1 + the one-body piece of Layer 2 (Section 3, minus the quadratic term):
plausibly a day or two, including validation against finite differences.
Adding the quadratic-term transition-2RDM piece correctly, plus the phase and
convention pitfalls in Section 4: bring the total to several days, with
correctness risk concentrated in items 1-2 of Section 4. This is a real,
concrete, buildable plan — not "impossible" — but it is genuinely more work
than the `variance`-cost analytic gradient, which needs none of this.
