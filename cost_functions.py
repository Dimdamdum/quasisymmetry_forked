import numpy as np
import openfermion as of
from typing import Union, Callable
from itertools import combinations

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