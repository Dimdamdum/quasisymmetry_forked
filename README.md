Optimizing quasisymmetries for quantum subspace expansion

Linjun Wang, Alexey Uvarov

New workflow (using ffsim):

1. make_pyscf_hamiltonian
2. generate_guesses
3. optimize
4. xs_to_costs


Previous workflow:

1. make_hamiltonian
2. generate_init_guesses
3. optimize_for_commutator
4. xs_to_cost_functions_fixed_abc


TODO: make the two `optimize_for_...` files into one, with optimization target being chosen as an input argument
