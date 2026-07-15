""" decoupled-sector energy objectives.

For a fixed orbital rotation U, each symmetry sector s gives a block

    H_s(U) = P_s H(U) P_s.

The decoupled energy is the lowest ground-state energy among these blocks:

    E_dec(U) = min_s E0(H_s(U)).

"""

import ffsim
import numpy as np
import scipy.optimize

from src.sector_utils import subspace_matrix
from src.clifford_sectors import (
    molecular_hamiltonian_to_jw,
    prepare_clifford_context,
    solve_tapered_sector,
    transform_hamiltonian_in_context,
)


def params_to_rotation(x, norb, pairs=None):
    """Convert orbital-rotation parameters into an orthogonal matrix."""
    from src.orbital_rotation import params_to_U

    return params_to_U(x, norb, pairs)


def sector_ground_energy(h_linop, sector_indices):
    """Build one sector block and return its lowest eigenvalue."""
    h_block = subspace_matrix(h_linop, sector_indices)
    h_block = 0.5 * (h_block + h_block.conj().T)
    return float(np.linalg.eigvalsh(h_block)[0].real)


def rotated_hamiltonian_linop(moldata, x, pairs=None):
    """Build the rotated Hamiltonian LinearOperator for orbital parameters x."""
    U = params_to_rotation(x, moldata.norb, pairs)
    rotated_h = moldata.hamiltonian.rotated(U)
    return ffsim.linear_operator(rotated_h, norb=moldata.norb, nelec=moldata.nelec)


def scan_sector_energies(moldata, sectors, x, pairs=None):
    """Return a list of (energy, sector_label, dimension) for all sectors."""
    h_linop = rotated_hamiltonian_linop(moldata, x, pairs)
    results = []
    for label, indices in sectors.items():
        energy = sector_ground_energy(h_linop, indices)
        results.append((energy, label, len(indices)))
    results.sort(key=lambda item: item[0])
    return results


def best_sector(moldata, sectors, x, pairs=None):
    """Return the lowest-energy sector as (energy, label, dimension)."""
    return scan_sector_energies(moldata, sectors, x, pairs)[0]


def decoupled_energy(moldata, sectors, x, pairs=None):
    """Return E_dec(U), scanning all sectors."""
    energy, _, _ = best_sector(moldata, sectors, x, pairs)
    return energy


def fixed_sector_energy(moldata, sector_indices, x, pairs=None):
    """Return E0(H_s(U)) for one fixed sector s."""
    h_linop = rotated_hamiltonian_linop(moldata, x, pairs)
    return sector_ground_energy(h_linop, sector_indices)


def make_decoupled_energy_cost(moldata, sectors, pairs=None):
    """Make an optimizer objective that rescans all sectors every time."""
    return lambda x: decoupled_energy(moldata, sectors, x, pairs)


def make_fixed_sector_energy_cost(moldata, sector_indices, pairs=None):
    """Make an optimizer objective for one fixed sector."""
    return lambda x: fixed_sector_energy(moldata, sector_indices, x, pairs)


def scan_clifford_sector_energies(moldata, context, x, pairs=None):
    """Return one-root energies for all physical tapered sectors."""
    rotation = params_to_rotation(x, moldata.norb, pairs)
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    jw_hamiltonian = molecular_hamiltonian_to_jw(rotated_hamiltonian, moldata.nelec)
    frame = transform_hamiltonian_in_context(jw_hamiltonian, context)
    results = []
    for label, indices in context["physical_sectors"].items():
        solved = solve_tapered_sector(frame, label, indices, 1)
        results.append((float(solved["energies"][0]), label, solved["dimension"]))
    results.sort(key=lambda item: item[0])
    return results


def clifford_fixed_sector_energy(moldata, context, label, x, pairs=None):
    """Return one tapered sector energy at orbital parameters x."""
    rotation = params_to_rotation(x, moldata.norb, pairs)
    rotated_hamiltonian = moldata.hamiltonian.rotated(rotation)
    jw_hamiltonian = molecular_hamiltonian_to_jw(rotated_hamiltonian, moldata.nelec)
    frame = transform_hamiltonian_in_context(jw_hamiltonian, context)
    solved = solve_tapered_sector(
        frame,
        label,
        context["physical_sectors"][label],
        1,
    )
    return float(solved["energies"][0])


def make_clifford_decoupled_energy_cost(moldata, symmetries, pairs=None):
    """Make a cost that scans all tapered sectors at every evaluation."""
    context = prepare_clifford_context(symmetries, moldata.norb, moldata.nelec)
    cost = lambda x: scan_clifford_sector_energies(moldata, context, x, pairs)[0][0]
    return cost, context


def make_clifford_fixed_sector_energy_cost(moldata, context, label, pairs=None):
    """Make a cost for one fixed tapered sector."""
    if label not in context["physical_sectors"]:
        raise ValueError(f"sector {label} is not present in the physical space")
    return lambda x: clifford_fixed_sector_energy(moldata, context, label, x, pairs)


def optimize_with_sector_switching(
    moldata,
    sectors,
    x0,
    max_switches=5,
    maxiter=100,
    callback=None,
    pairs=None,
):
    """Optimize one sector, rescan, switch sectors if needed, and repeat."""
    x = np.array(x0, dtype=float)
    current_energy, current_label, _ = best_sector(moldata, sectors, x, pairs)
    history = []

    for _ in range(max_switches + 1):
        cost = make_fixed_sector_energy_cost(
            moldata, sectors[current_label], pairs=pairs
        )
        result = scipy.optimize.minimize(
            cost,
            x,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            callback=callback,
        )

        x = result.x
        new_energy, new_label, _ = best_sector(moldata, sectors, x, pairs)
        history.append(
            {
                "start_sector": current_label,
                "start_energy": current_energy,
                "optimized_energy": float(result.fun),
                "best_sector_after_rescan": new_label,
                "best_energy_after_rescan": new_energy,
            }
        )

        if new_label == current_label:
            return result, history

        current_label = new_label
        current_energy = new_energy

    return result, history


def optimize_with_clifford_sector_switching(
    moldata,
    symmetries,
    x0,
    max_switches=5,
    maxiter=100,
    callback=None,
    pairs=None,
):
    """Optimize one tapered sector, rescan, and switch until stable."""
    context = prepare_clifford_context(symmetries, moldata.norb, moldata.nelec)
    x = np.asarray(x0, dtype=float)
    current_energy, current_label, _ = scan_clifford_sector_energies(
        moldata, context, x, pairs
    )[0]
    history = []

    for _ in range(max_switches + 1):
        cost = make_clifford_fixed_sector_energy_cost(
            moldata, context, current_label, pairs=pairs
        )
        result = scipy.optimize.minimize(
            cost,
            x,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            callback=callback,
        )
        x = result.x
        new_energy, new_label, _ = scan_clifford_sector_energies(
            moldata, context, x, pairs
        )[0]
        history.append(
            {
                "start_sector": current_label,
                "start_energy": current_energy,
                "optimized_energy": float(result.fun),
                "best_sector_after_rescan": new_label,
                "best_energy_after_rescan": new_energy,
            }
        )
        if new_label == current_label:
            return result, history
        current_label = new_label
        current_energy = new_energy
    return result, history
