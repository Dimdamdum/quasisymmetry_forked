Optimizing quasisymmetries for quantum subspace expansion

Alexey Uvarov, Linjun Wang

Suggested usage:

1. generate a Hamiltonian using `make_pyscf_hamiltonian.py` or bring your own (must be a pyscf .chk or .FCIDUMP)
2. Find relevant quasisymmetries using `show_symmetries.py`
3. Write down the ones you like as a binary matrix and optimize orbitals using `optimize_orbitals.py`
4. Evaluate all sorts of things using `metrics.py`

Note: for `metrics.py`, use the `--direct_K` key.