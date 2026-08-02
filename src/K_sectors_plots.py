import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from chemistry import CHEMICAL_PRECISION
from math import inf
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict
import json
import glob

# This file is piecewise AI-written and human-verified

# =============================================================================
# Core functions for computing metrics
# =============================================================================

def projected_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors):
    """Compute energy using only the sectors with indices in retained_sectors"""
    # check definitions

    # Create compressed coefficient vector
    compressed_coeffs = np.zeros_like(psi, dtype='complex')
    for sector_index in retained_sectors:
        sector_label, (projection, norm_squared, energy) = ordered_state_projections_in_sectors[sector_index]
        compressed_coeffs += projection

    # Calculate norm and safeguard against division by zero
    norm = np.linalg.norm(compressed_coeffs)
    if norm < 1e-15:
        return np.nan

    # Normalize
    compressed_coeffs /= norm

    # Compute energy
    e_proj = compressed_coeffs.T.conj() @ h_linop @ compressed_coeffs
    return e_proj.real

"""
#BUG need to pass the whole sectors. easy to fix, if needed.
def lowest_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors):
    # Return the exact lowest eigenvalue of H restricted to the selected sectors.
    selected_indices = []

    for sector_index in retained_sectors:
        _, (projection, norm_squared, energy) = ordered_state_projections_in_sectors[sector_index]
        selected_indices.extend(np.flatnonzero(projection))

    if not selected_indices:
        raise ValueError("retained_sectors must contain at least one sector")

    selected_indices = np.unique(selected_indices)
    subspace_dim = len(selected_indices)

    H_sub = np.zeros((subspace_dim, subspace_dim), dtype=complex)

    for col, basis_index in enumerate(selected_indices):
        basis_vector = np.zeros_like(psi, dtype=complex)
        basis_vector[basis_index] = 1.0
        H_sub[:, col] = (h_linop @ basis_vector)[selected_indices]

    # Enforce Hermiticity numerically before diagonalization.
    H_sub = 0.5 * (H_sub + H_sub.conj().T)

    return np.linalg.eigvalsh(H_sub)[0].real
"""

def energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors, projected_or_lowest):
    # if lowest needed, should fix lowest_energy_few_sectors (easy)
    # if projected_or_lowest == 'projected':
    #    return projected_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors)
    # if projected_or_lowest == 'lowest':
    #        return lowest_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors)
    assert projected_or_lowest == 'projected', "For now only projected_or_lowest='projected' is supported"
    if projected_or_lowest == 'projected':
        return projected_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors)

def get_K_sectors_values_energies(psi, h_linop, ref_energy, sectors, max_elec_transfers, projected_or_lowest, max_K_sectors=inf, verbose=0):
    assert np.isclose(ref_energy, np.vdot(psi, h_linop @ psi).real, atol=1e-10)
    assert np.isclose(0, np.vdot(psi, h_linop @ psi).imag, atol=1e-10)
    state_projections_in_sectors = {} # key = sector label (as in sectors), value = (projection of psi into sectors, norm squared)
    for sector_label, sector_indices in sectors.items():
        projection = np.zeros(psi.shape, dtype='complex')
        projection[sector_indices] = psi[sector_indices]
        norm_squared = np.linalg.norm(projection)**2
        energy = np.real(projection.T.conj() @ h_linop @ projection / norm_squared) if norm_squared > 0 else np.nan
        state_projections_in_sectors[sector_label] = (projection, norm_squared, energy)

    # order state_projections_in_sectors projections by norm_squared
    ordered_state_projections_in_sectors = sorted(state_projections_in_sectors.items(), key=lambda x: x[1][1], reverse=True)
    # print(f"\nState projections in sectors (ordered by norm squared):")
    #for sector_label, (projection, norm_squared, energy) in ordered_state_projections_in_sectors:
    #    print(f"Projection of psi into sector {sector_label}: norm squared = {norm_squared:.6f}, energy of projection = {energy:.6f}")

    # Convergence in the number of sectors retained - preparing output objects
    K_sectors_values = []
    K_sectors_energies = []
    K_sectors_retained_dimensions = []
    chem_accuracy_reached = False

    # Get label of main sector
    main_sector_label = ordered_state_projections_in_sectors[0][0][0] # here we do discard the parity dummy label
    if verbose > 0:
        print(f"Label of main sector: {main_sector_label}\n")

    # temp objects
    K_sectors = 0 # number of retained sectors
    sector_index = -1
    error = inf
    retained_sectors = []
    retained_dim = 0

    while error > CHEMICAL_PRECISION and sector_index < len(sectors) - 1 and K_sectors < max_K_sectors:
        K_sectors += 1
        sector_index += 1
        sector_label = ordered_state_projections_in_sectors[sector_index][0][0]
        num_particles_moved = round(sum(np.abs(np.array(sector_label) - np.array(main_sector_label))) / 2)
        if num_particles_moved > max_elec_transfers:
            K_sectors -= 1 # failed attempt
            continue
        retained_sectors.append(sector_index)
        retained_dim += len(sectors[ordered_state_projections_in_sectors[sector_index][0]])
        if verbose > 0:
            print(f"{sector_label} sector retained; {num_particles_moved} particles moved")
        K_sectors_retained_dimensions.append(retained_dim)
        K_sectors_values.append(K_sectors)
        e_K = energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors, projected_or_lowest)
        K_sectors_energies.append(e_K)
        error = e_K - ref_energy
        #print(f"K_sector={K_sectors:2d}: E={e_K:.8f}, Error={error:.8f} Ha = {error*27.2114:.4f} eV")
        if error < CHEMICAL_PRECISION:
            if verbose > 0:
                print(f"--> Chemical accuracy achieved at K_sectors={K_sectors}!")
            chem_accuracy_reached = True
        else:
            if verbose > 0:
                print(f"--> Chemical accuracy not reached.")
    return K_sectors_values, K_sectors_energies, K_sectors_retained_dimensions, chem_accuracy_reached

def projected_energy_few_states(retained_states_indices, psi, h_linop, ordered_decoupled_states):
    """Compute energy using only the states with indices in retained_states_indices"""
    
    # Initialize an empty state in the full space
    projected_state = np.zeros_like(psi, dtype=complex)
    
    # Reconstruct the state using only the retained decoupled eigenstates
    for idx in retained_states_indices:
        phi_i, sector_label, coeff = ordered_decoupled_states[idx]
        
        # Add the projection of psi onto this specific basis state
        projected_state += coeff * phi_i

    # Calculate norm and safeguard against division by zero
    norm = np.linalg.norm(projected_state)
    if norm < 1e-15:
        return np.nan

    # Normalize the reconstructed state
    projected_state /= norm

    # Compute energy (using np.vdot to automatically handle complex conjugation)
    e_proj = np.vdot(projected_state, h_linop @ projected_state)
    
    return e_proj.real

def energy_few_states(retained_states_indices, psi, h_linop, ordered_decoupled_states, projected_or_lowest):
    assert projected_or_lowest == 'projected', "For now only projected_or_lowest='projected' is supported"
    if projected_or_lowest == 'projected':
        return projected_energy_few_states(retained_states_indices, psi, h_linop, ordered_decoupled_states)


def get_K_states_values_energies(psi, h_linop, ref_energy, ordered_decoupled_states, max_elec_transfers, projected_or_lowest, max_K_states=inf, verbose=0):

    assert projected_or_lowest == 'projected', "For now only projected_or_lowest='projected' is supported"

    K_states_values = []
    K_states_energies = []
    K_states_num_retained_state_sectors = []
    chem_accuracy_reached = False
    retained_sector_labels = set() # will contain labels (with discarded parity dummy entry) of the used sectors

    main_sector_label = ordered_decoupled_states[0][1][0]  # here we do discard the parity dummy entry
    if verbose > 0:
        print(f"Label of main sector: {main_sector_label}\n")

    K_states = 0 # number of retained sectors
    state_index = -1
    error = inf
    retained_states_indices = [] # will contain indices of retained states
    num_retained_state_sectors = 0
    total_dim = len(psi)

    while error > CHEMICAL_PRECISION and state_index < total_dim - 1 and K_states < max_K_states:
        K_states += 1
        state_index += 1
        # state = ordered_decoupled_states[state_index][0] # not needed
        sector_label = ordered_decoupled_states[state_index][1][0] # here we do discard the parity dummy entry
        num_particles_moved = round(sum(np.abs(np.array(sector_label) - np.array(main_sector_label))) / 2)
        if num_particles_moved > max_elec_transfers:
            K_states -= 1 # failed attempt
            continue
        retained_states_indices.append(state_index)
        retained_sector_labels.add(sector_label)
        num_retained_state_sectors = len(retained_sector_labels)
        if verbose > 0:
            print(f"One decoupled state from {sector_label} sector retained; {num_particles_moved} particles moved")
        e_K = energy_few_states(retained_states_indices, psi, h_linop, ordered_decoupled_states, projected_or_lowest)
        K_states_values.append(K_states)
        K_states_energies.append(e_K)
        K_states_num_retained_state_sectors.append(num_retained_state_sectors)
        error = e_K - ref_energy
        if error < CHEMICAL_PRECISION:
            if verbose > 0:
                print(f"--> Chemical accuracy achieved at K_states={K_states}!")
            chem_accuracy_reached = True
        else:
            if verbose > 0:
                print(f"--> Chemical accuracy not reached.")

    return K_states_values, K_states_energies, K_states_num_retained_state_sectors, chem_accuracy_reached, retained_sector_labels

# =============================================================================
# Plotting functions - see below for version where data is read from .json
# =============================================================================

def plot_energy_vs_K(
    data_label_list,
    K_lists,
    energies_lists,
    retained_dimensions_or_num_sectors_list, # should be retained_dimensions if sectors_or_states == sectors, num_retained_state_sectors if sectors_or_states == states
    ref_energy,
    molecule='...',
    basis_set='...',
    norb='...',
    cluster_sizes='...',
    max_elec_transfers='...',
    cost='...',
    sectors_or_states='...'
):
    """
    Double plot: energy and retained dimension/retained num. of sectors vs list of integers K_list (either K_sectors_values_list or K_states_values_list).
    
    Args:
        data_label_list: List of data labels
        K_lists: List of lists of K values
        energies_lists: List of lists of energies
        retained_dimensions_or_num_sectors_list: List of lists of retained dimensions/numbers of retained sectors
        ref_energy: Reference energy
        molecule: Molecule name (for title)
        basis: Basis set of molecule object (for title)
        norb: Number of orbitals (for title)
        cluster_sizes: Cluster sizes (for title)
        max_elec_transfers: Maximum electron transfers (for title)
        cost: Cost type (for title)
        sectors_or_states: 'sectors' or 'states'
    """
    assert sectors_or_states in ['sectors', 'states'], "sectors_or_states must be 'sectors' or 'states'"

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

    num_curves = len(data_label_list)
    assert num_curves == len(K_lists)
    assert num_curves == len(energies_lists)

    # --- First subplot (Top) ---
    ax1 = axes[0]
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(num_curves):
        ax1.plot(
            K_lists[i], 
            [e - ref_energy for e in energies_lists[i]], 
            'o-', 
            label=data_label_list[i]
        )
    ax1.axhline(CHEMICAL_PRECISION, color='r', linestyle='--', label='Chemical accuracy')
    ax1.set_ylabel('$E - E_{ref}$ (Ha)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.margins(y=0.1)

    # --- Second subplot (Bottom) ---
    ax2 = axes[1]
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(num_curves):
        ax2.plot(
            K_lists[i], 
            retained_dimensions_or_num_sectors_list[i], 
            'o-', 
            label=data_label_list[i]
        )
    ax2.set_xlabel('Number of retained ' + sectors_or_states)
    if sectors_or_states == 'sectors':
        y_label = 'cumulative retained dimensions'
    if sectors_or_states == 'states':
        y_label = 'num. retained sectors'
    ax2.set_ylabel(y_label)
    ax2.grid(True, alpha=0.3)
    # ax2.set_yscale('log')
    # ax2.legend(loc='upper right')
    ax2.margins(y=0.1)

    # --- Figure-level settings ---
    fig.suptitle(
        'Energy against number of ' + sectors_or_states + 
        f' for {molecule} in {basis_set} basis \n Num. orbitals = {norb}, ' +
        f'cluster sizes = {cluster_sizes} + ghost, max $e^-$ transfers = {max_elec_transfers}, cost = {cost}',
        y=0.98
    )

    return fig


def plot_dual_bar_chart(
    x_data,
    y1_data,
    y2_data,
    label1="...",
    label2="...",
    title="...",
    colors=None,
    alpha=(0.8, 0.4)
):
    """
    Plots a dual-axis bar chart for two sets of y-data sharing the same x-axis labels.

    Args:
        x_data: List of x-axis labels
        y1_data: First set of y-data
        y2_data: Second set of y-data
        label1: Label for first y-axis
        label2: Label for second y-axis
        title: Plot title
        colors: List of colors (one per x tick)
        alpha: Tuple of (alpha_for_y1, alpha_for_y2)
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(x_data))
    width = 0.35

    if colors is None:
        # Default: single colors for all bars
        color1, color2 = 'tab:blue', 'tab:orange'
        ax1.bar(x - width/2, y1_data, width, label=label1, color=color1)
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, y2_data, width, label=label2, color=color2)
    else:
        # Per-bar-pair colors with fading
        ax2 = ax1.twinx()
        # Create proxy artists for legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], alpha=alpha[0], label=label1),
            Patch(facecolor=colors[0], alpha=alpha[1], label=label2)
        ]
        ax1.legend(handles=legend_elements)

        for i in range(len(x_data)):
            base_color = colors[i % len(colors)]  # Cycles if needed
            ax1.bar(x[i] - width/2, y1_data[i], width, color=base_color, alpha=alpha[0])
            ax2.bar(x[i] + width/2, y2_data[i], width, color=base_color, alpha=alpha[1])

    ax1.set_ylabel(label1)
    ax1.tick_params(axis='y')
    ax2.set_ylabel(label2)
    ax2.tick_params(axis='y')

    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_data)
    ax1.set_xlabel('Basis')

    plt.title(title)

    return fig

# =============================================================================
# Data Loading Functions
# =============================================================================


def load_metrics_file(filepath: str | Path) -> dict[str, Any]:
    """
    Load a single JSON metrics file.
    
    Args:
        filepath: Path to the JSON metrics file
        
    Returns:
        Dictionary containing the metrics data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return data


def load_aggregate_metrics_files(directory: str | Path, pattern: str = "results_*.json") -> list[dict]:
    """
    Load many JSON metrics files.
    
    Args:
        directory: Directory containing metrics files
        pattern: Glob pattern to match metrics files (default: "results_*.json")
        
    Returns:
        List of metrics dictionaries
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    metrics_files = list(directory.glob(pattern))
    if not metrics_files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {directory}")
    
    metrics_list = []
    for filepath in metrics_files:
        try:
            metrics_list.append(load_metrics_file(filepath))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return metrics_list

# AI implemented
def generate_plots_across_bond_lengths(input_dir, output_dir, angle_in_title=False):
    """
    Walk input_dir (bond_*/angle_*/max_transfers_*/*.json), and for every
    (bond_angle, max_elec_transfers, molecule, basis_set, cost, cluster_matrix)
    group, produce 4 PNG plots (one per y-metric) with bond_length on the
    x-axis and one curve per data_label.

    Output tree: output_dir/<angle_dir>/<max_transfers_dir>/*.png
    (bond_length is dropped as a directory level since it becomes the x-axis
    of the plots themselves -- see notes below the code).
    """
    import os, glob, json, hashlib, re
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if isinstance(angle_in_title, str):
        angle_in_title = angle_in_title.strip().lower() in ("true", "1", "yes")

    # (json key, y-axis label, boolean convergence-flag key)
    Y_METRICS = [
        ("num_retained_sectors", "K sectors mode: num. retained sectors", "chem_accuracy_reached_sectors"),
        ("retained_dim", "K sectors mode: cumulative retained dim.", "chem_accuracy_reached_sectors"),
        ("num_retained_states", "K states mode: num. retained states", "chem_accuracy_reached_states"),
        ("num_retained_state_sectors", "K states mode: num. retained sectors", "chem_accuracy_reached_states"),
    ]

    def sanitize(s):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

    def matrix_key(matrix):
        return tuple(tuple(row) for row in matrix)

    def matrix_hash(matrix):
        return hashlib.md5(json.dumps(matrix, sort_keys=False).encode()).hexdigest()[:8]

    def cluster_sizes(matrix):
        return tuple(sum(row) for row in matrix)

    json_paths = sorted(glob.glob(
        os.path.join(input_dir, "bond_*", "angle_*", "max_transfers_*", "*.json")
    ))
    if not json_paths:
        print(f"No JSON files found under {input_dir} "
              f"matching bond_*/angle_*/max_transfers_*/*.json")
        return

    # groups[(angle_dir, mt_dir)][(molecule, basis_set, cost, cluster_key, max_elec_transfers)]
    #   = {"bond_angle":..., "cluster_matrix":..., "curves": {data_label: [record, ...]}}
    groups = {}

    for jp in json_paths:
        mt_dir = os.path.basename(os.path.dirname(jp))
        angle_dir = os.path.basename(os.path.dirname(os.path.dirname(jp)))

        try:
            with open(jp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping unreadable/invalid JSON {jp}: {e}")
            continue

        meta = data.get("metadata", {})
        molecule = meta.get("molecule")
        basis_set = meta.get("basis_set")
        bond_length = meta.get("bond_length")
        bond_angle = meta.get("bond_angle")
        cost = meta.get("cost")
        cluster_matrix = meta.get("cluster_matrix")

        if any(v is None for v in (molecule, basis_set, bond_length, bond_angle, cost, cluster_matrix)):
            print(f"Skipping {jp}: missing required metadata field(s).")
            continue

        ck = matrix_key(cluster_matrix)
        tree_key = (angle_dir, mt_dir)
        groups.setdefault(tree_key, {})

        for basis_result in data.get("basis_results", []):
            data_label = basis_result.get("data_label", "unknown")
            for tr in basis_result.get("transfer_results", []):
                max_elec_transfers = tr.get("max_elec_transfers")
                sub_key = (molecule, basis_set, cost, ck, max_elec_transfers)
                bucket = groups[tree_key].setdefault(sub_key, {
                    "bond_angle": bond_angle,
                    "cluster_matrix": cluster_matrix,
                    "curves": {},
                })
                record = {
                    "bond_length": bond_length,
                    "num_retained_sectors": tr.get("num_retained_sectors"),
                    "retained_dim": tr.get("retained_dim"),
                    "chem_accuracy_reached_sectors": tr.get("chem_accuracy_reached_sectors", False),
                    "num_retained_states": tr.get("num_retained_states"),
                    "num_retained_state_sectors": tr.get("num_retained_state_sectors"),
                    "chem_accuracy_reached_states": tr.get("chem_accuracy_reached_states", False),
                }
                bucket["curves"].setdefault(data_label, []).append(record)

    n_plots = 0
    for (angle_dir, mt_dir), subgroups in groups.items():
        out_subdir = os.path.join(output_dir, angle_dir, mt_dir)
        os.makedirs(out_subdir, exist_ok=True)

        for (molecule, basis_set, cost, ck, max_elec_transfers), info in subgroups.items():
            curves = info["curves"]
            bond_angle = info["bond_angle"]
            cmatrix = info["cluster_matrix"]
            h = matrix_hash(cmatrix)
            sizes = cluster_sizes(cmatrix)

            for metric_key, metric_label, converge_key in Y_METRICS:
                fig, ax = plt.subplots(figsize=(8, 6))

                colors = {}
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                color_cycle = list(color_cycle)  # make it mutable

                label_order = list(curves.keys())  # preserves first-seen (JSON) order
                for i, data_label in enumerate(label_order):
                    colors[data_label] = color_cycle[i % len(color_cycle)]

                any_data = False
                nonconv_by_label = {}

                for data_label in label_order:
                    points = curves[data_label]
                    conv_pts = sorted(
                        (p["bond_length"], p[metric_key])
                        for p in points
                        if p.get(converge_key) and p.get(metric_key) is not None
                    )
                    nonconv_x = sorted(set(
                        p["bond_length"] for p in points if not p.get(converge_key)
                    ))
                    if conv_pts:
                        xs, ys = zip(*conv_pts)
                        ax.plot(xs, ys, marker="o", label=data_label, color=colors[data_label])
                        any_data = True
                    if nonconv_x:
                        nonconv_by_label[data_label] = nonconv_x

                if not any_data and not nonconv_by_label:
                    plt.close(fig)
                    continue

                ylim = ax.get_ylim() if any_data else (0, 1)
                if not any_data:
                    ax.set_ylim(*ylim)

                any_nonconverged = False
                for data_label, xs in nonconv_by_label.items():
                    any_nonconverged = True
                    color = colors.get(data_label, "gray")
                    any_nonconverged = False
                    n_labels = len(nonconv_by_label)
                    y_range = ylim[1] - ylim[0]
                    offset_step = 0.02 * y_range  # % of axis height per label
                    for i, (data_label, xs) in enumerate(sorted(nonconv_by_label.items())):
                        any_nonconverged = True
                        color = colors.get(data_label, "gray")
                        y_offset = ylim[0] + (i + 1) * offset_step
                        ax.scatter(xs, [y_offset] * len(xs), marker="x", color=color, zorder=5, s=15)
                if any_data:
                    ax.set_ylim(*ylim)  # keep scale fixed after adding bottom markers

                ax.set_xlabel("Bond length")
                ax.set_ylabel(metric_label)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if angle_in_title == True:
                    ax.set_title(
                        f"{molecule} / {basis_set} / cost={cost}\n"
                        f"bond angle = {bond_angle}, max. elec. transfers = {max_elec_transfers}\n"
                        f"cluster sizes = {sizes} (cluster_matrix #{h})",
                        fontsize=12.5,
                    )
                else:
                    ax.set_title(
                        f"{molecule} / {basis_set} / cost={cost}\n"
                        f"max. elec. transfers = {max_elec_transfers}\n"
                        f"cluster sizes = {sizes} (cluster_matrix #{h})",
                        fontsize=12.5,
                    )

                handles, labels = ax.get_legend_handles_labels()
                if any_nonconverged:
                    proxy = Line2D([0], [0], marker="x", color="gray", linestyle="None")
                    handles.append(proxy)
                    labels.append("not converged (excluded)")
                ax.legend(handles, labels, loc="upper right", fontsize=8)

                fig.tight_layout()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = (
                    f"{sanitize(molecule)}_{sanitize(basis_set)}_{sanitize(cost)}_"
                    f"cm{h}_mt{sanitize(max_elec_transfers)}_{metric_key}_{timestamp}.png"
                )
                fig.savefig(os.path.join(out_subdir, fname), dpi=150)
                plt.close(fig)
                n_plots += 1

    print(f"Generated {n_plots} plot(s) under {output_dir}")