import numpy as np
import matplotlib.pyplot as plt
from chemistry import CHEMICAL_PRECISION
from math import inf
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict
import json
import glob

def projected_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors):
    """Compute energy using only the sectors with indices in retained_sectors"""
    # check definitions

    # Create compressed coefficient vector
    compressed_coeffs = np.zeros_like(psi, dtype='complex')
    for sector_index in retained_sectors:
        sector_label, (projection, norm_squared, energy) = ordered_state_projections_in_sectors[sector_index]
        compressed_coeffs += projection

    # Normalize
    compressed_coeffs /= np.linalg.norm(compressed_coeffs)

    # Compute energy
    e_proj = compressed_coeffs.T.conj() @ h_linop @ compressed_coeffs
    return e_proj.real

def lowest_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors):
    """Return the exact lowest eigenvalue of H restricted to the selected sectors."""
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

def energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors, projected_or_lowest):
    if projected_or_lowest == 'projected':
        return projected_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors)
    if projected_or_lowest == 'lowest':
            return lowest_energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors)

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

    # Convergence in the number of sector states retained
    K_sectors_values = []
    K_sectors_energies = []

    # Get label of main sector
    main_sector_label = ordered_state_projections_in_sectors[0][0][0]
    if verbose > 0:
        print(f"Label of main sector: {main_sector_label}\n")

    K_sectors = 0 # number of retained sectors
    sector_index = -1
    error = inf
    retained_sectors = []
    retained_dim = 0
    chem_accuracy_reached = False

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
        K_sectors_values.append(K_sectors)
        e_K = energy_few_sectors(retained_sectors, psi, h_linop, ordered_state_projections_in_sectors, projected_or_lowest)
        K_sectors_energies.append(e_K)
        error = e_K - ref_energy
        #print(f"K_sector={K_sectors:2d}: E={e_K:.8f}, Error={error:.8f} Ha = {error*27.2114:.4f} eV")
    if verbose > 0:
        if error < CHEMICAL_PRECISION:
            print(f"--> Chemical accuracy achieved at K_sector={K_sectors}!")
            chem_accuracy_reached = True
        else:
            print(f"--> Chemical accuracy not reached.")
    return K_sectors_values, K_sectors_energies, retained_dim, chem_accuracy_reached

def plot_energy_vs_K_sectors(data_label_list, K_sectors_values_list, K_sectors_energies_list, ref_energy, molecule='...', basis='...', norb='...', cluster_sizes='...', max_elec_transfers='...', var_exponent='...'):
    plt.figure(figsize=(8, 5))
    num_curves = len(data_label_list)
    assert num_curves == len(K_sectors_values_list)
    assert num_curves == len(K_sectors_energies_list)
    for i in range(num_curves):
        plt.plot(K_sectors_values_list[i], [e - ref_energy for e in K_sectors_energies_list[i]], 'o-', label=data_label_list[i])
    plt.axhline(CHEMICAL_PRECISION, color='r', linestyle='--', label='Chemical accuracy')
    plt.xlabel('Number of retained sectors')
    plt.ylabel('$E - E_{ref}$ (Ha)')
    plt.title(f'Energy against number of sectors for {molecule} in {basis} basis \n Num. orbitals = {norb}, cluster sizes = {cluster_sizes} + ghost, max $e^-$ transfers = {max_elec_transfers}, var. exp. = {var_exponent}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

def plot_dual_bar_chart(x_data, y1_data, y2_data,
                        label1="...", label2="...",
                        title="...", colors=None, alpha=(0.8, 0.4)):
    """
    Plots a dual-axis bar chart for two sets of y-data sharing the same x-axis labels.

    Args:
        colors: List of colors (one per x tick). Each color is used for both bars
                at that position, with fading controlled by alpha.
        alpha: Tuple of (alpha_for_y1, alpha_for_y2). Default: (0.8, 0.4).
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
    fig.tight_layout()
    plt.show()


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_metrics_file(filepath: str | Path) -> dict[str, Any]:
    """
    Load a JSON metrics file.
    
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


def aggregate_metrics(directory: str | Path, pattern: str = "results_*.json") -> list[dict]:
    """
    Aggregate multiple metrics files for comparison.
    
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


# =============================================================================
# Enhanced Plotting Functions with File Support
# =============================================================================


def plot_energy_vs_K_sectors_from_file(
    filepath: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
    save: bool = True,
    **kwargs
) -> None:
    """
    Load data from a metrics file and create energy vs K sectors plot.
    
    Args:
        filepath: Path to the metrics JSON file
        output_path: Path to save the plot (default: None, auto-generated)
        show: Whether to display the plot (default: True)
        save: Whether to save the plot to file (default: True)
        **kwargs: Additional arguments to pass to plot_energy_vs_K_sectors
    """
    data = load_metrics_file(filepath)
    
    metadata = data.get("metadata", {})
    basis_results = data.get("basis_results", [])
    
    if not basis_results:
        raise ValueError(f"No basis_results found in {filepath}")
    
    # Extract data
    data_label_list = [r.get("data_label", f"Basis {i}") for i, r in enumerate(basis_results)]
    K_sectors_values_list = [r.get("K_sectors_values", []) for r in basis_results]
    K_sectors_energies_list = [r.get("K_sectors_energies", []) for r in basis_results]
    ref_energy = metadata.get("dmrg_energy", 0.0)
    
    # Set default kwargs from metadata
    defaults = {
        "molecule": metadata.get("molecule", "..."),
        "basis": metadata.get("basis_set", "..."),
        "norb": metadata.get("norb", "..."),
        "cluster_sizes": metadata.get("cluster_sizes", "..."),
        "max_elec_transfers": metadata.get("max_elec_transfers", "..."),
        "var_exponent": metadata.get("var_exponent", "..."),
    }
    defaults.update(kwargs)
    
    # Create plot
    plot_energy_vs_K_sectors(
        data_label_list, K_sectors_values_list, K_sectors_energies_list,
        ref_energy, **defaults
    )
    
    # Save if requested
    if save:
        if output_path is None:
            timestamp = metadata.get("timestamp", "unknown")
            git_hash = metadata.get("git_hash", "unknown")
            output_path = Path("plots") / f"energy_vs_sectors_{timestamp}_{git_hash}.png"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_path}")
    
    if not show:
        plt.close()


def plot_dual_bar_chart_from_file(
    filepath: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
    save: bool = True,
    **kwargs
) -> None:
    """
    Load data from a metrics file and create dual bar chart.
    
    Args:
        filepath: Path to the metrics JSON file
        output_path: Path to save the plot (default: None, auto-generated)
        show: Whether to display the plot (default: True)
        save: Whether to save the plot to file (default: True)
        **kwargs: Additional arguments to pass to plot_dual_bar_chart
    """
    data = load_metrics_file(filepath)
    
    metadata = data.get("metadata", {})
    basis_results = data.get("basis_results", [])
    
    if not basis_results:
        raise ValueError(f"No basis_results found in {filepath}")
    
    # Extract data
    x_data = [r.get("data_label", f"Basis {i}") for i, r in enumerate(basis_results)]
    y1_data = [r.get("num_retained_sectors", 0) for r in basis_results]
    y2_data = [r.get("retained_dim", 0) for r in basis_results]
    
    # Set default kwargs from metadata
    defaults = {
        "label1": "Number of retained sectors",
        "label2": "Retained dimension",
        "title": f"Sector analysis for {metadata.get('molecule', '...')} in {metadata.get('basis_set', '...')} basis",
    }
    defaults.update(kwargs)
    
    # Create plot
    plot_dual_bar_chart(x_data, y1_data, y2_data, **defaults)
    
    # Save if requested
    if save:
        if output_path is None:
            timestamp = metadata.get("timestamp", "unknown")
            git_hash = metadata.get("git_hash", "unknown")
            output_path = Path("plots") / f"retained_sectors_{timestamp}_{git_hash}.png"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_path}")
    
    if not show:
        plt.close()


def plot_all_metrics_in_directory(
    directory: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "results_*.json",
    show: bool = False,
    save: bool = True,
) -> None:
    """
    Create all plots for all metrics files in a directory.
    
    Args:
        directory: Directory containing metrics files
        output_dir: Directory to save plots (default: same as directory/plots)
        pattern: Glob pattern to match metrics files (default: "results_*.json")
        show: Whether to display plots (default: False)
        save: Whether to save plots to files (default: True)
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Set output directory
    if output_dir is None:
        output_dir = directory / "plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all metrics files
    metrics_files = list(directory.glob(pattern))
    if not metrics_files:
        print(f"No files matching pattern '{pattern}' found in {directory}")
        return
    
    print(f"Found {len(metrics_files)} metrics files to process")
    
    # Process each file
    for filepath in metrics_files:
        try:
            print(f"Processing {filepath.name}...")
            
            # Extract timestamp and git hash from filename if possible
            name_parts = filepath.stem.split("_")
            timestamp = None
            git_hash = None
            if len(name_parts) >= 3 and name_parts[0] == "results":
                timestamp = name_parts[1]
                git_hash = name_parts[2]
            
            # Plot energy vs sectors
            if save:
                output_path = output_dir / f"energy_vs_sectors_{filepath.stem}.png"
                plot_energy_vs_K_sectors_from_file(
                    filepath, 
                    output_path=output_path, 
                    show=show, 
                    save=True
                )
            
            # Plot dual bar chart
            if save:
                output_path = output_dir / f"retained_sectors_{filepath.stem}.png"
                plot_dual_bar_chart_from_file(
                    filepath,
                    output_path=output_path,
                    show=show,
                    save=True
                )
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print("Finished processing all metrics files")


# =============================================================================
# Modified Plotting Functions with Optional Display
# =============================================================================


def plot_energy_vs_K_sectors(
    data_label_list,
    K_sectors_values_list,
    K_sectors_energies_list,
    ref_energy,
    molecule='...',
    basis='...',
    norb='...',
    cluster_sizes='...',
    max_elec_transfers='...',
    var_exponent='...',
    show: bool = True
):
    """
    Plot energy vs number of retained sectors.
    
    Args:
        data_label_list: List of data labels
        K_sectors_values_list: List of K sectors values for each basis
        K_sectors_energies_list: List of K sectors energies for each basis
        ref_energy: Reference energy
        molecule: Molecule name (for title)
        basis: Basis set (for title)
        norb: Number of orbitals (for title)
        cluster_sizes: Cluster sizes (for title)
        max_elec_transfers: Maximum electron transfers (for title)
        var_exponent: Variance exponent (for title)
        show: Whether to display the plot (default: True)
    """
    plt.figure(figsize=(8, 5))
    num_curves = len(data_label_list)
    assert num_curves == len(K_sectors_values_list)
    assert num_curves == len(K_sectors_energies_list)
    for i in range(num_curves):
        plt.plot(K_sectors_values_list[i], [e - ref_energy for e in K_sectors_energies_list[i]], 'o-', label=data_label_list[i])
    plt.axhline(CHEMICAL_PRECISION, color='r', linestyle='--', label='Chemical accuracy')
    plt.xlabel('Number of retained sectors')
    plt.ylabel('$E - E_{ref}$ (Ha)')
    plt.title(f'Energy against number of sectors for {molecule} in {basis} basis \n Num. orbitals = {norb}, cluster sizes = {cluster_sizes} + ghost, max $e^-$ transfers = {max_elec_transfers}, var. exp. = {var_exponent}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')
    
    if show:
        plt.show()


def plot_dual_bar_chart(
    x_data,
    y1_data,
    y2_data,
    label1="...",
    label2="...",
    title="...",
    colors=None,
    alpha=(0.8, 0.4),
    show: bool = True
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
        show: Whether to display the plot (default: True)
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
    fig.tight_layout()
    
    if show:
        plt.show()