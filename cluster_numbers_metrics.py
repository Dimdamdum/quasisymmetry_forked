from __future__ import annotations

# This file is piecewise AI-written and human-verified

"""
Cluster Numbers Metrics Script

This script performs DMRG calculations, extracts RDMs, optimizes orbital bases,
and analyzes symmetry sectors to evaluate the quality of cluster decompositions.
See notebook cluster_numbers_search (same implementation, clearer structure).

Usage:
    python cluster_numbers_metrics.py h4_square 6-31g 1.0 --max-transfers 2
    python cluster_numbers_metrics.py h2o 6-31g 1.0 --bond-angle 104.5 --max-transfers 2
"""

""""
Detail usage guide

# Run a standard analysis
python cluster_numbers_metrics.py h4_square 6-31g 1.0

# With custom parameters
python cluster_numbers_metrics.py h2o 6-31g 1.0 --bond-angle 104.5 --max-transfers 2

# python cluster_numbers_metrics.py h4_square 6-31g 1.0 --cluster-matrix '[[1,0,0,0,0,0,0,1],[0,0,1,0,1,0,0,0],[0,1,0,1,0,0,0,0]]'   

# Skip orbital optimization for faster runs
python cluster_numbers_metrics.py h4_square 6-31g 1.0 --no-optimization --bond-dim 250 --n-sweeps 20

Complete list of CLI arguments
Required:
molecule: Molecule name (h2, h2o, n2, lih, h4_linear, h4_square, h4_rectangle)
basis_set: Basis set (e.g., sto-3g, 6-31g) 
bond_length: Bond length in Angstrom
Optional: 
--bond-angle: Bond angle in degrees (for H2O, default: None)
--max-transfers: Maximum electron transfers (default: 2) 
--cluster-matrix: Custom cluster matrix as JSON array or path to file
--bond-dim: DMRG bond dimension (default: 500) 
--n-sweeps: Number of DMRG sweeps (default: 50)
--cost-function: Cost function type (variance, eval_eq, mixed, default: variance)
--var-exponent: Variance exponent (default: 1) 
--maxiter: Maximum optimization iterations (default: 1000)
--no-optimization: Skip orbital optimization 
--bases: Which orbital bases to analyze (default: all 5 - MOs, optimized from MOs, NatOs, optimized from NatOs, random)
--output-dir: Custom output directory 
--plots-dir: Custom plots directory
--no-plots: Disable plot generation 
--show-plots: Display plots interactively
--n-threads: Number of threads (default: 1)
--no-reuse: Don't reuse existing wavefunction
--verbose: Enable verbose logging

Output Files
Results are saved in a structured directory in the git-ignored outputs_:
outputs_/cluster_number/
  └── {molecule}/
      └── {basis_set}/
          └── bond_{length}/
              [ └── angle_{angle}/ ]  # if bond_angle specified
              └── max_transfers_{N}/
                  └── results_{timestamp}_{git_hash}.json
JSON output contains:
metadata: molecule, basis_set, bond length/angle, max electron transfers, timestamp, git hash, norb, dmrg_energy, computation time
basis_results: Array of results for each orbital basis with K_sectors values, energies, retained sectors count, dimension, and chemical accuracy flag
"""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import comb, inf
from pathlib import Path
from typing import Any
from itertools import compress

import numpy as np
import pyscf

# Configure JAX for float64 precision
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from chemistry import CHEMICAL_PRECISION, get_geometry_and_description
from src.cluster_number_operators import (
    build_loc_number_evaluator,
    get_cluster_indices,
    number_and_parity_symmetry_sectors,
    number_eval_eq_cost,
    number_variance_cost,
    extremality_cost,
    params_to_U_jax,
)
from src.dmrg_solver import Block2DMRGSolver, DMRGConfig, solve_or_load_ground_state
from src.K_sectors_plots import (
    get_K_sectors_values_energies,
    plot_dual_bar_chart,
    plot_energy_vs_K_sectors,
)
from src.orbital_rotation import params_to_U

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BasisResult:
    """Results for a single orbital basis (e.g., MOs, NatOs, optimized from MOs) analysis."""

    data_label: str
    K_sectors_values: list[int]
    K_sectors_energies: list[float]
    num_retained_sectors: int
    retained_dim: int
    chem_accuracy_reached: bool


@dataclass
class MetricsOutput:
    """Complete output containing metadata and results for all bases. One .json output file produced by this script containes one instance of this class."""

    metadata: dict[str, Any]
    basis_results: list[BasisResult] = field(default_factory=list)


@dataclass
class MetricsConfig:
    """Configuration for cluster numbers metrics computation."""

    # Input parameters
    molecule: str
    basis_set: str
    bond_length: float
    bond_angle: float | None = None
    cluster_matrix: np.ndarray | None = None
    max_elec_transfers: int = 2

    # DMRG parameters
    bond_dim: int = 500
    n_sweeps: int = 50

    # Orbital optimization parameters
    type_cost_function: str = "variance"
    var_exponent: int = 1
    maxiter: int = 1000

    # Computation options
    run_dmrg: bool = True
    run_orbital_optimization: bool = True
    analyze_bases: list[str] = field(default_factory=lambda: [
        "MOs", "Opt. from MOs", "NatOs", "Opt. from NatOs", "Random"
    ])

    # Output options
    output_dir: Path | None = None
    plots_dir: Path | None = None
    save_plots: bool = True
    show_plots: bool = False

    # HPC options
    n_threads: int = 1
    reuse_wavefunction: bool = True

    # Validation
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.molecule.lower() not in [
            "h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"
        ]:
            raise ValueError(
                f"Unsupported molecule: {self.molecule}. "
                "Supported: h2, h2o, n2, lih, h4_linear, h4_square, h4_rectangle"
            )
        if self.type_cost_function not in ["variance", "eval_eq", "extremality", "mixed"]:
            raise ValueError(
                f"Unsupported cost function: {self.type_cost_function}. "
                "Supported: variance, eval_eq, extremality, mixed"
            )
        if self.max_elec_transfers < 0:
            raise ValueError("max_elec_transfers must be >= 0")


# =============================================================================
# Helper functions
# =============================================================================


def get_cluster_matrix_from_config(config: MetricsConfig) -> np.ndarray:
    """Get cluster matrix, either from config or use default for molecule."""
    if config.cluster_matrix is not None:
        return config.cluster_matrix
    
    # Default cluster matrices for common molecules
    molecule = config.molecule.lower()
    basis_set = config.basis_set.lower()
    
    # For H2 in minimal basis sets
    if molecule == "h2" and basis_set == "sto-3g":
        # H2 in sto-3g has 2 orbitals
        return np.array([
            [1, 0]
        ])
    
    if molecule == "h2" and basis_set == "6-31g":
        # H2 in 6-31g has 4 orbitals
        return np.array([
            [1, 1, 0, 0, 0]
        ])
    
    if molecule == "h4_square" and basis_set == "6-31g":
        # Default cluster matrix for H4 square in 6-31g (8 orbitals)
        return np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0]
        ])
    
    if molecule == "h4_square" and basis_set == "sto-3g":
        # H4 square in sto-3g has 4 orbitals (one per H atom)
        return np.array([
            [1, 0, 1, 0]
        ])
    
    if molecule == "h2o" and basis_set == "sto-3g":
        # H2O in sto-3g has 7 orbitals
        return np.array([
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0]
        ])
    
    if molecule == "h2o" and basis_set == "6-31g":
        # H2O in 6-31g has 13 orbitals
        return np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        ])
    
    if molecule == "n2" and basis_set == "sto-3g":
        # N2 in sto-3g has 10 orbitals
        return np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ])
    
    if molecule == "lih" and basis_set == "sto-3g":
        # LiH in sto-3g has 6 orbitals
        return np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0]
        ])
    
    raise ValueError(
        f"No default cluster matrix for {config.molecule} with basis set {config.basis_set}. "
        "Please provide a cluster_matrix using --cluster-matrix argument."
    )


def validate_cluster_matrix(cluster_matrix: np.ndarray, norb: int) -> None:
    """Validate the cluster matrix."""
    if cluster_matrix.shape[1] != norb:
        raise ValueError(
            f"Number of columns of cluster_matrix ({cluster_matrix.shape[1]}) "
            f"does not match number of orbitals ({norb})"
        )
    
    # Calculate the sum of each column (one column per orbital)
    column_sums = np.sum(cluster_matrix, axis=0)
    if not np.all(np.isin(column_sums, [0, 1])):
        invalid_columns = np.where((column_sums != 0) & (column_sums != 1))[0]
        raise ValueError(
            f"Error: Orbitals {invalid_columns} should appear in at most one cluster."
        )
    if any([all(bit == 0 for bit in cluster) for cluster in cluster_matrix]):
        raise ValueError("Error: There is one or more empty clusters.")


def create_output_dirs(config: MetricsConfig) -> tuple[Path, Path]:
    """Create output and plots directories based on configuration."""
    molecule = config.molecule.lower()
    basis_set = config.basis_set.lower()
    bond_length_str = f"{config.bond_length:.4f}".replace(".", "_")
    
    # Build output directory path
    if config.output_dir is not None:
        output_dir = config.output_dir
    else:
        output_dir = Path("outputs_") / "cluster_number" / molecule / basis_set / f"bond_{bond_length_str}"
        if config.bond_angle is not None:
            angle_str = f"{config.bond_angle:.4f}".replace(".", "_")
            output_dir = output_dir / f"angle_{angle_str}"
        output_dir = output_dir / f"max_transfers_{config.max_elec_transfers}"
    
    # Build plots directory path
    if config.plots_dir is not None:
        plots_dir = config.plots_dir
    else:
        plots_dir = Path("plots") / "cluster_number" / molecule / basis_set / f"bond_{bond_length_str}"
        if config.bond_angle is not None:
            angle_str = f"{config.bond_angle:.4f}".replace(".", "_")
            plots_dir = plots_dir / f"angle_{angle_str}"
        plots_dir = plots_dir / f"max_transfers_{config.max_elec_transfers}"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, plots_dir


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_git_hash() -> str:
    """Get current git hash for reproducibility."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_checkpoint_path(output_dir: Path) -> Path:
    """Get checkpoint file path."""
    return output_dir / ".checkpoint.json"


def save_checkpoint(checkpoint: dict, checkpoint_path: Path) -> None:
    """Save checkpoint state to file."""
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load checkpoint state from file."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


# =============================================================================
# Main Computation Functions
# =============================================================================


def compute_dmrg(config: MetricsConfig) -> tuple[Block2DMRGSolver, float, np.ndarray, np.ndarray, float, tuple[int, int], Any]:
    """
    Run DMRG to get ground state MPS and integrals.
    
    Returns:
        solver: Block2DMRGSolver instance
        dmrg_energy: Ground state energy
        h1e: One-electron integrals
        g2e: Two-electron integrals
        ecore: Nuclear energy
        nelec: Number of electrons
        result: DMRGResult object with mps_tag
    """
    logger.info("Building molecule and running HF...")
    
    geometry, _ = get_geometry_and_description(
        config.molecule, 
        config.bond_length, 
        hoh_angle_deg=config.bond_angle
    )
    mol = pyscf.M(atom=geometry, basis=config.basis_set)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    
    norb, nelec = mol.nao, mol.nelec
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = pyscf.ao2mo.full(mol, mf.mo_coeff)
    ecore = mol.energy_nuc()
    
    logger.info(f"Number of orbitals: {norb}")
    logger.info(f"Number of electrons: {nelec}")
    
    # Create store directory for wavefunction
    geometry_key = hashlib.sha1(
        json.dumps(geometry, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    store_dir = Path("wavefunctions") / (
        f"dmrg_{config.molecule}_{config.basis_set}_geom{geometry_key}_bd{config.bond_dim}_sw{config.n_sweeps}"
    )
    
    solver = Block2DMRGSolver(
        h1e=h1e, 
        g2e=g2e, 
        ecore=ecore,
        n_elec=nelec, 
        spin=mol.spin,
        store_dir=store_dir, 
        n_threads=config.n_threads,
    )
    
    logger.info("Running DMRG...")
    result = solve_or_load_ground_state(
        solver,
        config=DMRGConfig(max_bond_dim=config.bond_dim, n_sweeps=config.n_sweeps),
        reuse=config.reuse_wavefunction
    )
    dmrg_energy = result.energy
    logger.info(f"DMRG Energy: {dmrg_energy:.10f} Ha")
    
    return solver, dmrg_energy, h1e, g2e, ecore, nelec, result


def extract_rdms(solver: Block2DMRGSolver, mps_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 1- and 2-RDMs from MPS.
    
    Returns:
        rdm1_a, rdm1_b: 1-RDMs for alpha and beta
        rdm2_aa, rdm2_ab, rdm2_bb: 2-RDMs for alpha-alpha, alpha-beta, beta-beta
    """
    logger.info("Extracting RDMs from MPS...")
    
    available_tags = solver.stored_tags()
    if mps_tag not in available_tags:
        raise RuntimeError(
            f"Expected MPS tag '{mps_tag}' not found in store "
            f"{solver.store_dir}; available tags: {available_tags}"
        )
    mps = solver.get_mps(tag=mps_tag)
    
    rdm1_a, rdm1_b = solver.driver.get_1pdm(mps)
    rdm2_aa, rdm2_ab, rdm2_bb = solver.driver.get_2pdm(mps)
    
    logger.info(f"rdm1_a shape: {rdm1_a.shape}, trace: {np.trace(rdm1_a).round(8)}")
    logger.info(f"rdm2_aa shape: {rdm2_aa.shape}")
    
    return rdm1_a, rdm1_b, rdm2_aa, rdm2_ab, rdm2_bb


def optimize_orbital_basis(
    config: MetricsConfig,
    norb: int,
    nelec: tuple[int, int],
    rdm1_a: np.ndarray,
    rdm1_b: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    cluster_matrix: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Optimize orbital basis starting from MOs and NatOs.
    
    Returns:
        U_opt_list: List of optimized unitary transformations
        cost_data: List of (initial_cost, optimized_cost) tuples
    """
    logger.info("Starting orbital optimization...")
    
    # Prepare density matrices
    D_MOs = rdm1_a + rdm1_b
    Gamma_MOs = rdm2_aa + rdm2_bb + rdm2_ab + rdm2_ab.transpose(1, 0, 3, 2)
    
    # Get Natural Orbitals
    spec, evecs = np.linalg.eigh(D_MOs)
    U_NatOs = evecs.T
    U_NatOs_conj = np.conj(U_NatOs)
    D_NatOs = U_NatOs_conj @ D_MOs @ U_NatOs.T
    Gamma_NatOs = np.einsum(
        "pi,qj,rk,sl,ijkl->pqrs", 
        U_NatOs_conj, U_NatOs_conj, U_NatOs, U_NatOs, Gamma_MOs, 
        optimize=True
    )
    
    U_opt_list = []
    cost_data = []
    
    for D, Gamma in [(D_MOs, Gamma_MOs), (D_NatOs, Gamma_NatOs)]:
        loc_number_evaluator = build_loc_number_evaluator(
            D, Gamma, cluster_matrix=cluster_matrix
        )
        
        # Build cost function based on type
        if config.type_cost_function == "variance":
            f = number_variance_cost(
                D, Gamma, cluster_matrix, 
                with_ghost=False, 
                var_exponent=config.var_exponent
            )
        elif config.type_cost_function == "eval_eq":
            clusters = get_cluster_indices(cluster_matrix, norb, with_ghost=False)
            evals = []
            for cluster in clusters:
                cluster_num_average = D[cluster, cluster].sum()
                evals.append(round(cluster_num_average))
            f = number_eval_eq_cost(D, Gamma, cluster_matrix, evals, with_ghost=False)
        elif config.type_cost_function == "extremality":
            raise NotImplementedError("extremality cost function not yet implemented")
        elif config.type_cost_function == "mixed":
            f1 = number_variance_cost(
                D, Gamma, cluster_matrix, 
                with_ghost=False, 
                var_exponent=config.var_exponent
            )
            f2 = extremality_cost(D, cluster_matrix, with_ghost=False)
            c1 = 1.
            c2 = .5
            def f(x):
                return c1 * f1(x) + c2 * f2(x) # set some combination of f1, f2, f3; can also define manually f
        else:
            raise ValueError(f"Unsupported cost function: {config.type_cost_function}")
        
        # Get with JAX function and gradient
        f_val_and_grad = jax.jit(jax.value_and_grad(f))
        
        # Wrap for SciPy
        def scipy_f(x):
            val, grad = f_val_and_grad(x)
            return float(val), jnp.asarray(grad, dtype=jnp.float64)
        
        # Initial guess (identity rotation)
        x0 = np.zeros(comb(norb, 2))
        initial_cost = f(x0)
        logger.info(f"Initial cost (identity rotation): {initial_cost:.6e}")
        
        # Run optimization
        import scipy.optimize
        
        class IterationTracker:
            def __init__(self):
                self.iteration = 0
            
            def __call__(self, intermediate_result):
                self.iteration += 1
                cost = intermediate_result.fun
                if self.iteration % 10 == 0:
                    logger.info(f"Iteration {self.iteration:3d} | Cost: {cost:.6f}")
        
        tracker = IterationTracker()
        
        result = scipy.optimize.minimize(
            fun=scipy_f,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            callback=tracker,
            options={"maxiter": config.maxiter, "disp": False}
        )
        optimized_cost = result.fun
        logger.info(f"Optimized cost: {optimized_cost:.6e}")
        logger.info(f"Cost change: {(-initial_cost + optimized_cost)/initial_cost * 100:.2f}%")
        
        # Extract optimized orbital rotation
        x_opt = result.x
        U_opt = params_to_U_jax(x_opt, norb)
        
        U_opt_list.append(U_opt)
        cost_data.append((initial_cost, optimized_cost))
    
    return U_opt_list, cost_data


def compute_sector_analysis(
    config: MetricsConfig,
    solver: Block2DMRGSolver,
    dmrg_energy: float,
    h1e: np.ndarray,
    g2e: np.ndarray,
    ecore: float,
    nelec: tuple[int, int],
    norb: int,
    cluster_matrix: np.ndarray,
    U_list: list[np.ndarray],
    data_label_list: list[str],
) -> list[BasisResult]:
    """
    Perform sector analysis for each orbital basis.
    
    Returns:
        List of BasisResult objects for each basis
    """
    logger.info("Starting sector analysis...")
    
    import ffsim
    import pyscf.ao2mo
    
    basis_results = []
    
    # Get full state vector and Hamiltonian in MO basis
    mps = solver.get_mps()
    psi_MOs = solver.to_ci_vector(ket=mps)
    
    # Restore compressed g2e to a 4D array
    g2e_full = pyscf.ao2mo.restore(1, g2e, norb)
    
    # Construct the ffsim Hamiltonian
    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor=h1e,
        two_body_tensor=g2e_full,
        constant=ecore
    )
    
    # Generate the linear operator
    h_linop_MOs = ffsim.linear_operator(hamiltonian, norb, nelec)
    
    # Get rotated full state vectors and rotated Hamiltonians
    psi_rotated_list = []
    h_linop_rotated_list = []
    
    for U in U_list:
        psi_rotated = ffsim.apply_orbital_rotation(
            psi_MOs, np.array(U), norb, nelec
        )
        psi_rotated_list.append(psi_rotated)
        h_rotated = hamiltonian.rotated(U)
        h_linop_rotated = ffsim.linear_operator(h_rotated, norb, nelec)
        h_linop_rotated_list.append(h_linop_rotated)
    
    # Identify symmetry sectors
    # We want labels to also include the ghost cluster
    if not np.all(cluster_matrix.any(axis=0)):
        ghost_row = np.array([1 - cluster_matrix.sum(axis=0)])
        cluster_matrix_with_ghost = np.append(cluster_matrix, ghost_row, axis=0)
    else:
        cluster_matrix_with_ghost = cluster_matrix
    
    sectors = number_and_parity_symmetry_sectors(
        cluster_matrix_with_ghost, [], norb, nelec
    )
    
    logger.info(f"Number of symmetry sectors identified: {len(sectors)}")
    logger.info(f"Max electron transfers across clusters: {config.max_elec_transfers}")
    
    # Compute K_sectors for each basis
    for i in range(len(U_list)):
        data_label = data_label_list[i]
        psi = psi_rotated_list[i]
        h_linop = h_linop_rotated_list[i]
        
        logger.info(f"Analyzing basis: {data_label}")
        
        K_sectors_values, K_sectors_energies, retained_dim, chem_accuracy_reached = (
            get_K_sectors_values_energies(
                psi, 
                h_linop, 
                dmrg_energy, 
                sectors, 
                config.max_elec_transfers, 
                "projected",
                max_K_sectors=inf,
                verbose=0
            )
        )
        
        basis_result = BasisResult(
            data_label=data_label,
            K_sectors_values=K_sectors_values,
            K_sectors_energies=K_sectors_energies,
            num_retained_sectors=len(K_sectors_values),
            retained_dim=retained_dim,
            chem_accuracy_reached=chem_accuracy_reached
        )
        basis_results.append(basis_result)
        
        logger.info(
            f"  {data_label}: {len(K_sectors_values)} sectors, "
            f"dim={retained_dim}, chem_accuracy={chem_accuracy_reached}"
        )
    
    return basis_results


def compute_cluster_number_metrics(config: MetricsConfig) -> MetricsOutput:
    """
    Main function to compute cluster metrics.
    
    This function:
    1. Runs DMRG to get ground state
    2. Extracts RDMs
    3. Optimizes orbital basis (if enabled)
    4. Performs sector analysis for each basis
    
    Returns:
        MetricsOutput with all results
    """
    start_time = time.time()
    
    # Get cluster matrix
    cluster_matrix = get_cluster_matrix_from_config(config)
    
    # Create output directories
    output_dir, plots_dir = create_output_dirs(config)
    
    # Build metadata
    metadata = {
        "molecule": config.molecule,
        "basis_set": config.basis_set,
        "bond_length": config.bond_length,
        "bond_angle": config.bond_angle,
        "max_elec_transfers": config.max_elec_transfers,
        "timestamp": get_timestamp(),
        "git_hash": get_git_hash(),
        "norb": None,
        "dmrg_energy": None,
    }
    
    # Step 1: Run DMRG
    if config.run_dmrg:
        solver, dmrg_energy, h1e, g2e, ecore, nelec, result = compute_dmrg(config)
        metadata["dmrg_energy"] = float(dmrg_energy)
        metadata["norb"] = int(solver.n_sites)
        metadata["nelec"] = [int(x) for x in nelec]
    else:
        raise NotImplementedError("Skipping DMRG not yet implemented")
    
    norb = metadata["norb"]
    
    # Validate cluster matrix
    validate_cluster_matrix(cluster_matrix, norb)
    metadata["cluster_matrix"] = cluster_matrix.tolist()
    metadata["cluster_sizes"] = [int(round(sum(row))) for row in cluster_matrix]
    
    # Step 2: Extract RDMs
    rdm1_a, rdm1_b, rdm2_aa, rdm2_ab, rdm2_bb = extract_rdms(solver, result.mps_tag)
    
    # Step 3: Orbital optimization (if enabled)
    U_opt_list = []
    cost_data = []
    
    if config.run_orbital_optimization:
        U_opt_list, cost_data = optimize_orbital_basis(
            config, norb, nelec, rdm1_a, rdm1_b, rdm2_aa, rdm2_ab, rdm2_bb, cluster_matrix
        )
    
    # Step 4: Sector analysis for each basis
    # Prepare U_list and data_label_list based on which analyses to run
    U_list = []
    data_label_list = []
    
    # Always add MOs
    U_list.append(np.eye(norb))
    data_label_list.append("MOs")
    
    # Add optimized from MOs if available
    if "Opt. from MOs" in config.analyze_bases and config.run_orbital_optimization and U_opt_list:
        U_list.append(U_opt_list[0])
        data_label_list.append("Opt. from MOs")
    
    # Add NatOs
    if "NatOs" in config.analyze_bases:
        D_MOs = rdm1_a + rdm1_b
        spec, evecs = np.linalg.eigh(D_MOs)
        U_NatOs = evecs.T
        U_list.append(U_NatOs)
        data_label_list.append("NatOs")
    
    # Add optimized from NatOs if available
    if "Opt. from NatOs" in config.analyze_bases and config.run_orbital_optimization and len(U_opt_list) > 1:
        D_MOs = rdm1_a + rdm1_b
        spec, evecs = np.linalg.eigh(D_MOs)
        U_NatOs = evecs.T
        U_list.append(U_opt_list[1] @ U_NatOs)
        data_label_list.append("Opt. from NatOs")
    
    # Add Random
    if "Random" in config.analyze_bases:
        U_random = params_to_U(10 * np.random.rand(comb(norb, 2)), norb)
        U_list.append(U_random)
        data_label_list.append("Random")
    
    basis_results = compute_sector_analysis(
        config, solver, dmrg_energy, h1e, g2e, ecore, nelec, norb,
        cluster_matrix, U_list, data_label_list
    )
    
    # Compute total time
    computation_time = time.time() - start_time
    metadata["computation_time_seconds"] = computation_time
    metadata["var_exponent"] = config.var_exponent
    
    # Create output
    output = MetricsOutput(
        metadata=metadata,
        basis_results=basis_results
    )
    
    return output


def save_metrics(output: MetricsOutput, output_dir: Path) -> Path:
    """
    Save metrics to JSON file.
    
    Returns:
        Path to the saved file
    """
    timestamp = output.metadata.get("timestamp", get_timestamp())
    git_hash = output.metadata.get("git_hash", "unknown")
    
    filename = f"results_{timestamp}_{git_hash}.json"
    filepath = output_dir / filename
    
    # Convert to dict for serialization
    output_dict = {
        "metadata": output.metadata,
        "basis_results": [
            asdict(result) for result in output.basis_results
        ]
    }
    
    with open(filepath, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def load_metrics(filepath: Path) -> MetricsOutput:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    basis_results = [
        BasisResult(**result_dict) for result_dict in data["basis_results"]
    ]
    
    return MetricsOutput(
        metadata=data["metadata"],
        basis_results=basis_results
    )


def generate_plots(
    output: MetricsOutput,
    plots_dir: Path,
    show: bool = False,
    save: bool = True
) -> None:
    """Generate and save plots from metrics output."""
    import matplotlib
    if not save:
        matplotlib.use("Agg")  # Use non-interactive backend if not saving
    import matplotlib.pyplot as plt
    
    metadata = output.metadata
    basis_results = output.basis_results
    
    # Prepare data for plotting
    data_label_list = [r.data_label for r in basis_results]
    K_sectors_values_list = [r.K_sectors_values for r in basis_results]
    K_sectors_energies_list = [r.K_sectors_energies for r in basis_results]
    num_retained_sectors_list = [r.num_retained_sectors for r in basis_results]
    retained_dim_list = [r.retained_dim for r in basis_results]
    
    # Generate timestamp for filename
    timestamp = metadata.get("timestamp", get_timestamp())
    
    # Plot 1: Energy vs K_sectors
    if save:
        plot_energy_vs_K_sectors(
            data_label_list,
            K_sectors_values_list,
            K_sectors_energies_list,
            metadata["dmrg_energy"],
            molecule=metadata["molecule"],
            basis_set=metadata["basis_set"],
            norb=metadata.get("norb", "?"),
            cluster_sizes=metadata.get("cluster_sizes", "?"),
            max_elec_transfers=metadata["max_elec_transfers"],
            var_exponent=metadata.get("var_exponent", 1)
        )
        
        filename = f"energy_vs_sectors_{timestamp}.png"
        filepath = plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Energy vs sectors plot saved to {filepath}")
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot 2: Dual bar chart
    chem_accuracy_reached_list = [result.chem_accuracy_reached for result in output.basis_results]
    if not any(chem_accuracy_reached_list):
        print("No basis reached chemical accuracy. Skipping 2nd plot (dual bar chart).")
    else:
        # Prepare data for dual bar chart
        x_data = data_label_list
        y1_data = num_retained_sectors_list
        y2_data = retained_dim_list
        colors = [f'C{i}' for i in range(len(output.basis_results))]
        colors_cp = list(compress(colors, chem_accuracy_reached_list))
        x_data_cp = list(compress(x_data, chem_accuracy_reached_list))
        y1_data_cp = list(compress(y1_data, chem_accuracy_reached_list))
        y2_data_cp = list(compress(y2_data, chem_accuracy_reached_list))

        if save:
            plt.figure(figsize=(8, 5))
            
            title = (
                f"Sector analysis for {metadata['molecule']} in {metadata['basis_set']} basis set \n "
                f"Num. orbitals = {metadata.get('norb', '?')}, cluster sizes = {metadata.get('cluster_sizes', '?')}, "
                f"max $e^-$ transfers = {metadata['max_elec_transfers']}"
            )

            plot_dual_bar_chart(
                x_data=x_data_cp,
                y1_data=y1_data_cp,
                y2_data=y2_data_cp,
                label1="Number of retained sectors",
                label2="Retained dimension",
                title=title,
                colors=colors_cp,
                alpha=(0.8, 0.4)
            )
            
            filename = f"retained_sectors_bar_chart_{timestamp}.png"
            filepath = plots_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Retained sectors bar chart saved to {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()

# =============================================================================
# CLI Interface
# =============================================================================


def parse_cluster_matrix(matrix_str: str) -> np.ndarray:
    """Parse cluster matrix from string representation."""
    import re
    
    # Try to parse as JSON array
    try:
        import json
        return np.array(json.loads(matrix_str))
    except json.JSONDecodeError:
        pass
    
    # Try to parse as Python literal
    try:
        import ast
        return np.array(ast.literal_eval(matrix_str))
    except (ValueError, SyntaxError):
        pass
    
    # Try to parse from file
    matrix_path = Path(matrix_str)
    if matrix_path.exists():
        if matrix_path.suffix == ".npy":
            return np.load(matrix_path)
        elif matrix_path.suffix == ".txt":
            return np.loadtxt(matrix_path)
    
    raise ValueError(
        f"Cannot parse cluster matrix from: {matrix_str}. "
        "Expected JSON array, Python literal, or path to .npy/.txt file."
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Compute cluster numbers metrics for quasisymmetry analysis"
    )
    
    # Required arguments
    parser.add_argument(
        "molecule",
        type=str,
        choices=["h2", "h2o", "n2", "lih", "h4_linear", "h4_square", "h4_rectangle"],
        help="Molecule to analyze"
    )
    parser.add_argument(
        "basis_set",
        type=str,
        help="Basis set (e.g., sto-3g, 6-31g)"
    )
    parser.add_argument(
        "bond_length",
        type=float,
        help="Bond length in Angstrom"
    )
    
    # Optional arguments
    parser.add_argument(
        "--bond-angle",
        type=float,
        default=None,
        help="Bond angle in degrees (for H2O)"
    )
    parser.add_argument(
        "--max-transfers",
        type=int,
        default=2,
        help="Maximum electron transfers (default: 2)"
    )
    parser.add_argument(
        "--cluster-matrix",
        type=str,
        default=None,
        help="Cluster matrix as JSON array, Python literal, or path to file"
    )
    
    # DMRG parameters
    parser.add_argument(
        "--bond-dim",
        type=int,
        default=500,
        help="DMRG bond dimension (default: 500)"
    )
    parser.add_argument(
        "--n-sweeps",
        type=int,
        default=50,
        help="Number of DMRG sweeps (default: 50)"
    )
    
    # Orbital optimization parameters
    parser.add_argument(
        "--cost-function",
        type=str,
        default="variance",
        choices=["variance", "eval_eq", "mixed"],
        help="Cost function type (default: variance)"
    )
    parser.add_argument(
        "--var-exponent",
        type=int,
        default=1,
        help="Variance exponent (default: 1)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum optimization iterations (default: 1000)"
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Skip orbital optimization"
    )
    
    # Basis selection
    parser.add_argument(
        "--bases",
        type=str,
        nargs="+",
        default=None,
        help="Which bases to analyze (default: all)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Plots directory"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively"
    )
    
    # HPC options
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Number of threads (default: 1)"
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Don't reuse existing wavefunction"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Parse cluster matrix if provided
    cluster_matrix = None
    if args.cluster_matrix is not None:
        cluster_matrix = parse_cluster_matrix(args.cluster_matrix)
    
    # Build configuration
    config = MetricsConfig(
        molecule=args.molecule,
        basis_set=args.basis_set,
        bond_length=args.bond_length,
        bond_angle=args.bond_angle,
        cluster_matrix=cluster_matrix,
        max_elec_transfers=args.max_transfers,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        type_cost_function=args.cost_function,
        var_exponent=args.var_exponent,
        maxiter=args.maxiter,
        run_orbital_optimization=not args.no_optimization,
        n_threads=args.n_threads,
        reuse_wavefunction=not args.no_reuse,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
        save_plots=not args.no_plots,
        show_plots=args.show_plots,
    )
    
    # Override bases if specified
    if args.bases is not None:
        config.analyze_bases = args.bases
    
    logger.info(f"Starting computation for {args.molecule} in {args.basis_set} basis set")
    logger.info(f"Configuration: molecule={config.molecule}, basis set={config.basis_set}, "
                f"bond_length={config.bond_length}, max_transfers={config.max_elec_transfers}")
    
    # Run computation
    try:
        output = compute_cluster_number_metrics(config)
        
        # Save results
        output_dir, plots_dir = create_output_dirs(config)
        filepath = save_metrics(output, output_dir)
        
        # Generate plots if enabled
        if config.save_plots:
            generate_plots(output, plots_dir, show=config.show_plots, save=True)
        
        logger.info("Computation completed successfully!")
        logger.info(f"Results saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Computation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
