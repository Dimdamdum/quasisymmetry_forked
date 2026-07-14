#### Install

```bash
git clone --recurse-submodules https://github.com/aleksey-uvarov/quasisymmetry.git
cd quasisymmetry

python -m pip install -U pip
python -m pip install numpy scipy matplotlib tqdm mpi4py
python -m pip install pyscf ffsim openfermion openfermionpyscf
python -m pip install block2
```

`block2` is required for any path that uses `dmrg`. `pyscf` / `ffsim` are required for FCI / statevector / Davidson. On Windows, prefer a conda/binary build of `block2` and `pyscf` if pip tries to compile from source.

#### CLI cheat sheet

**`optimize_symmetries.py`** — only `--reference` (picks wavefunction and cost engine)

| Flag | What happens |
|------|--------------|
| `--reference fci` (default) | PySCF FCI + ffsim CI costs |
| `--reference hf` | Hartree–Fock + ffsim CI costs |
| `--reference dmrg` | Block2 MPS + MPS-native NC / variance |

Sector energy costs (`decoupled` / `fixed_sector` / `switching_sector`) need `--reference fci` or `hf`. Shared DMRG flags with `--reference dmrg`: `--bond_dim`, `--wavefunction_dir`, `--n_threads`.

**`metrics.py`** — `--reference` plus `--backend` (sector eigensolver)

| Combination | What happens |
|-------------|--------------|
| `--reference fci --backend fci` | eigsh / dense eigh per sector (default) |
| `--reference fci --backend davidson` | PySCF Davidson on the same sector blocks |
| `--reference dmrg --backend dmrg` | Block2 sector-targeted DMRG for E_dec / K |

Omitting `--reference` in metrics picks the natural default (`dmrg` with `--backend dmrg`, else `fci`). `--solver` is a deprecated alias of `--backend`.

See `src/workflow_cli.py`. Each script prints a `[workflow]` banner at startup. Run `python <script> --help` for examples.

# Approximate symmetry finder (small systems)

This is a summary of the module that I am making by stitching together Praveen’s code with Linjun’s and mine.

To use this module, clone as follows:

`git clone --recurse-submodules https://github.com/aleksey-uvarov/quasisymmetry.git`

#### make\_pyscf\_hamiltonian.py

Can be used to produce a bunch of example Hamiltonians from a hard-coded list.

#### show\_symmetries.py

Given an electronic structure Hamiltonian, calculates the NC scores for quartets and shows them as a heatmap plot.

#### find\_pauli\_symmetries.py

Finds symmetries via beam search  
input:

1. Hamiltonian  
2. the reference cost function (“fci”, “hf”, “cisd”)  
3. cost function  
4. keyword --senquart that constrains the symmetries to seniorities and quartets.

Output:

1. List of Pauli symmetries. If they are all Zs, also spits out a parity matrix.

#### optimize\_symmetries.py

Input arguments and keywords:

1. Hamiltonian. Checkfile or FCIDUMP. In the future I might add support for of.QubitOperator.  
2. Parity matrix of the symmetries (or \--seniority)  
3. Reference / cost path: \--reference fci\|hf\|dmrg (defaults to fci). ``fci``/``hf`` use ffsim CI costs; ``dmrg`` uses Block2 MPS-native NC/variance. Shared DMRG flags: \--bond\_dim / \--wavefunction\_dir / \--n\_threads.  
4. Cost function: NC, variance, decoupled, fixed\_sector, switching\_sector (decoupled / sector modes require \--reference fci or hf)  
5. x0: optional initial guess for orbital rotations

Supported cost functions:

1. `NC`: non-commutator cost against the chosen reference state.
2. `variance`: symmetry variance against the chosen reference state.
3. `decoupled`: expensive objective that rescans all sectors at every orbital-optimization step and minimizes the best single-sector energy.
4. `fixed_sector`: cheaper objective that optimizes one sector. If `--fixed_sector` is not supplied, the initial lowest-energy sector is selected automatically.
5. `switching_sector`: compromise objective. It selects the initial lowest-energy sector, optimizes it, rescans all sectors, switches if another sector is lower, and repeats until the selected sector stops changing or `--sector_switch_maxiter` is reached.

Returns:

1. Optimized rotation parameters in a JSON file (plus optional rotated FCIDUMP / orbene).

#### optimize\_dmrg.py

FCIDUMP-friendly entry point for the same MPS-native optimizer (no pyscf/ffsim required). Arguments mirror ``optimize_symmetries.py --reference dmrg``.

#### solve\_dmrg.py

Runs block2 DMRG (fermionic SZ mode, straight from the integrals — no Jordan–Wigner) and stores the wavefunction locally so later stages can reload it without re-solving.

Inputs:

1. Hamiltonian (.chk needs pyscf; FCIDUMP works standalone, including on native Windows)  
2. \--U: optional orbital-rotation parameters x (same file format as metrics.py); integrals are rotated with the same convention as ffsim’s `hamiltonian.rotated(U)`  
3. \--parity\_matrix: optional; reports the symmetry expectations ⟨S\_k⟩ of the ground state  
4. \--decoupled: sector-resolved DMRG on the exactly decoupled Hamiltonian (cross-sector couplings removed), reporting per-sector energies, E\_decoupled and the K = 1 check  
5. \--k\_coupled: PT-screened coupled-energy K from DMRG sector excited states (implies \--decoupled); \--states\_per\_sector sets roots per sector  
6. \--entanglement / \--entropies: bipartite cut entropies, 1-orbital entropies and mutual information  
7. \--reorder fiedler|gaopt: optional orbital reordering before DMRG (parity matrix is remapped automatically)  
8. \--bond\_dim, \--n\_sweeps, \--n\_threads, \--penalty, \--max\_sectors, \--no\_reuse

Outputs:

1. `wavefunctions/<name>/` — the block2 MPS files, integrals and a metadata.json (energies, sweep settings, sector labels). Reused automatically on the next run.  
2. A result.txt with energies, sector/K data and (optionally) entanglement diagnostics.

Core library: `src/dmrg_solver.py` (`Block2DMRGSolver`). MPS-native costs: `src/dmrg_costs.py`. Diagnostics (E\_decoupled, K, entropies): `src/dmrg_diagnostics.py`. Validated in `tests/test_dmrg_*.py`. The older `src/tn.py` helpers are JW/Pauli SGB only — use `dmrg_solver` for the fermionic pipeline.

#### metrics.py

Inputs:

1. JSON from ``optimize_symmetries.py`` (molpath, parity, rotation, …)  
2. \--backend fci\|davidson\|dmrg — sector eigensolver (alias: \--solver)  
3. \--reference fci\|dmrg — comparison energy/state for dE and reference-K (must match backend; default follows backend)  
4. Shared DMRG flags when using dmrg: \--bond\_dim, \--wavefunction\_dir, \--n\_threads; also \--penalty, \--max\_sectors, \--reorder, \--entanglement  
5. Davidson flags (only with ``--backend davidson``): \--davidson\_tol, \--davidson\_max\_cycle, \--davidson\_max\_space

Compare sector solvers on the same OO JSON:

```bash
python metrics.py oo.json --backend fci
python metrics.py oo.json --backend davidson --davidson_tol 1e-10
python metrics.py oo.json --reference dmrg --backend dmrg --bond_dim 250 --penalty 30
```

Outputs:

1. Decoupled energy  
2. K: number of sector eigenstates needed to reach chemical accuracy  
3. Which sectors do these eigenstates come from  
4. ``backend`` / ``reference`` / ``solve_time_s`` (and Davidson per-sector meta when applicable)

All of this is saved in a JSON file.

If you want to run the FCI path in parallel, call:

`mpiexec -n 5 python -m mpi4py.futures metrics.py [arguments]`

on a desktop and

`srun -n 5 python -m mpi4py.futures metrics.py [arguments]`
