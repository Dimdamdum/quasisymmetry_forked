import numpy as np
import openfermion as of
from typing import Union, Callable
from itertools import combinations
import scipy

from chemistry import build_U_from_thetas
from optimization_different_abc import rotated_seniority_orbital_fermion

def commutator_cost_thermal(H: of.FermionOperator,
                            beta: float,
                            orbital_energies: np.ndarray) -> Callable:
    n_orbitals = len(orbital_energies)
    pairs = list(combinations(range(n_orbitals), 2))

    spin_orbital_energies = duplicate_each_element(orbital_energies)
    thermal_pops = np.array(1 / (1 + np.exp(beta * spin_orbital_energies)))

    def f(x):
        U = build_U_from_thetas(n_orbitals, x, pairs)
        a = np.sin(x[-2]) * np.cos(x[-1])
        b = np.sin(x[-2]) * np.sin(x[-1])
        c = np.cos(x[-2])
        total_commutator_norm = 0
        for i in range(n_orbitals):
            Si_ferm = of.normal_ordered(
                rotated_seniority_orbital_fermion(
                    U, i, n_orbitals, a, b, c
                )
            )
            comm = of.normal_ordered(of.commutator(H, Si_ferm))
            total_commutator_norm += trace_with_thermal_state(comm, thermal_pops)
        return total_commutator_norm

    return f


def trace_with_thermal_state(op: of.FermionOperator, populations: np.ndarray) -> float:
    """assumes op is normal ordered"""
    total_trace = 0
    for coeff, monomial in op.terms.items():
        pass
    raise NotImplementedError()
    return 0


def duplicate_each_element(x: np.ndarray) -> np.ndarray:
    if len(x.shape) != 1:
        raise ValueError("Only 1D arrays accepted")
    y = np.zeros((x.shape[0], 2))
    y[:, 0] = x
    y[:, 1] = x
    return y.flatten()

def commutator_cost_number_pres(H, psi, num_electrons, n_orbitals):
    """THIS COULD STILL BE FASTER"""
    pairs = list(combinations(range(n_orbitals), 2))
    m = len(pairs)

    def f(x):
        a = np.sin(x[m]) * np.cos(x[m + 1])
        b = np.sin(x[m]) * np.sin(x[m + 1])
        c = np.cos(x[m])
        U = build_U_from_thetas(n_orbitals, x[:m], pairs)
        symmetries = [(of.FermionOperator(((2 * i, 1), (2 * i, 0)), a)
                       + of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), b)
                       + (of.FermionOperator(((2 * i, 1), (2 * i, 0)), c)
                       * of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), 1)))
                      for i in range(n_orbitals)]

        symmetry_inter_ops = [of.get_interaction_operator(s, n_qubits=2 * n_orbitals)
                              for s in symmetries]

        total = 0
        for op in symmetry_inter_ops:
            op.rotate_basis(U.T.conj())
            op = of.get_fermion_operator(op)
            op_mat = of.get_number_preserving_sparse_operator(
                op, num_qubits=2 * n_orbitals,
                num_electrons=num_electrons)
            # print(op_mat.shape)
            # print(H.shape)
            comm = H @ op_mat - op_mat @ H
            total += np.linalg.norm(comm @ psi)**2
        return total

    return f, m


def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)


def rotation_to_x(U):
    norb = U.shape[0]
    iu = np.triu_indices(norb, k=1)
    rotation_generator = scipy.linalg.logm(U)
    x = rotation_generator[iu]
    return x