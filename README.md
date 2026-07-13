# Approximate symmetry finder (small systems)

This is a summary of the module that I am making by stitching together Praveen’s code with Linjun’s and mine.

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
2. Parity matrix of the symmetries  
3. Reference state: \--reference, “fci”, “hf”, “dmrg” (defaults to fci). With “dmrg”, \--bond\_dim sets the bond dimension and \--wavefunction\_dir points at a local MPS store to reuse/create.  
4. Cost backend: \--backend statevector|dmrg (default statevector). The dmrg backend evaluates NC/variance with MPS-native MPO–MPS multiplies on a fixed DMRG reference (no CI vector), using the identity ``||[H(U), S] U|ψ⟩|| = ||[H, U†SU]|ψ⟩||``. Same ``x`` output format.  
5. Cost function: variance to ref, NC to ref (defaults to NC)  
6. x0: optional initial guess for orbital rotations

Returns:

1. Optimized parameters for an orbital rotation saved as a text file.

#### optimize\_dmrg.py

FCIDUMP-friendly entry point for the same MPS-native optimizer (no pyscf/ffsim required). Arguments mirror ``optimize_symmetries.py --backend dmrg``.

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

1. Hamiltonian  
2. Parity matrix  
3. rotation (optional)  
4. \--solver fci|dmrg — with `dmrg`, runs the MPS-native path (`run_dmrg_metrics`: sector DMRG for E\_decoupled, PT-screened K, no dense `subspace_matrix`); \--bond\_dim, \--wavefunction\_dir, \--penalty, \--max\_sectors, \--reorder, \--entanglement as in solve\_dmrg.py

Outputs:

1. Decoupled energy  
2. K: number of sector eigenstates needed to reach chemical accuracy  
3. Which sectors do these eigenstates come from

All of this is saved in a text file.

