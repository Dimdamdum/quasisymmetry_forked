from __future__ import annotations

# This file is piecewise AI-written and human-verified

"""
Cluster Numbers Metrics Script

This script performs DMRG calculations, extracts RDMs, optimizes orbital bases,
and analyzes symmetry sectors to evaluate the quality of cluster decompositions.
See notebook cluster_numbers_search (same implementation, clearer structure).

Usage:
    python cluster_numbers_metrics.py h4_square 6-31g 1.0 --max-transfers 1 2
    python cluster_numbers_metrics.py h2o 6-31g 1.0 --bond-angle 104.5 --max-transfers 1 2 3 --bases MOs "Opt. from MOs" "NatOs" "Opt. from NatOs"
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
cost_function: Cost function type (variance, eval_eq, mixed, extremality, commutator)
Optional: 
--bond-angle: Bond angle in degrees (for H2O, default: None)
--max-transfers: Maximum electron transfers (default: [1 2 3]) 
--cluster-matrix: Custom cluster matrix as JSON array or path to file
--bond-dim: DMRG bond dimension (default: 500) 
--n-sweeps: Number of DMRG sweeps (default: 50)
--var-exponent: Variance exponent (default: 1) 
--maxiter: Maximum optimization iterations (default: 500)
--no-optimization: Skip orbital optimization 
--bases: Which orbital bases to analyze (default: all 5 - MOs, optimized from MOs, NatOs, optimized from NatOs, random)
--skip-K-sectors: Skip sector-based analysis
--skip-K-states: Skip decoupled state-based analysis
--output-dir: Custom output directory 
--plots-dir: Custom plots directory
--wavefunction-dir: Wavefunction directory for input and output
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
metadata: molecule, basis_set, bond length/angle, max electron transfers, timestamp, git hash, norb, dmrg_energy, computation time, cost
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
    number_commutator_cost_v1
)
from src.dmrg_solver import Block2DMRGSolver, DMRGConfig, solve_or_load_ground_state
from src.K_sectors_plots import (
    get_K_sectors_values_energies,
    get_K_states_values_energies,
    plot_dual_bar_chart,
    plot_energy_vs_K,
)
from src.orbital_rotation import params_to_U

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SingleMaxelectransferResult:
    """Results for a specific max_elec_transfers value."""
    max_elec_transfers: int

    # sector-based analysis
    K_sectors_values: list[int]
    K_sectors_energies: list[float]
    K_sectors_retained_dimensions: list[int]
    num_retained_sectors: int
    retained_dim: int
    chem_accuracy_reached_sectors: bool

    # decoupled states-based analysis
    K_states_values: list[int]
    K_states_energies: list[float]
    K_states_num_retained_state_sectors: list[int]
    num_retained_states: int
    num_retained_state_sectors: int # watch out, different from num_retained_sectors!
    chem_accuracy_reached_states: bool


@dataclass
class BasisResult:
    """Results for a single orbital basis analysis; one SingleMaxelectransferResult multiple max_elec_transfers values."""

    data_label: str
    transfer_results: list[SingleMaxelectransferResult]


@dataclass
class MetricsOutput:
    """Complete output containing metadata and results for all bases. One .json output file produced by this script containes one instance of this class."""

    metadata: dict[str, Any]
    basis_results: list[BasisResult] = field(default_factory=list) # one BasisResult per basis


@dataclass
class MetricsConfig:
    """Configuration for cluster numbers metrics computation."""

    # Input parameters
    molecule: str
    basis_set: str
    bond_length: float
    type_cost_function: str
    bond_angle: float | None = None
    cluster_matrix: np.ndarray | None = None
    max_elec_transfers: list[int] = field(default_factory=lambda: [1, 2, 3])

    # DMRG parameters
    bond_dim: int = 500
    n_sweeps: int = 50

    # Orbital optimization parameters
    var_exponent: int = 1
    maxiter: int = 500

    # Computation options
    run_dmrg: bool = True
    run_orbital_optimization: bool = True
    analyze_bases: list[str] = field(default_factory=lambda: [
        "MOs", "Opt. from MOs", "NatOs", "Opt. from NatOs", "Random"
    ])

    # Analysis options
    skip_K_sectors: bool = False
    skip_K_states: bool = False

    # Output options
    output_dir: Path | None = None
    plots_dir: Path | None = None
    wavefunction_dir: str = "wavefunctions"
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
        if self.type_cost_function not in ["variance", "eval_eq", "extremality", "mixed", "commutator"]:
            raise ValueError(
                f"Unsupported cost function: {self.type_cost_function}. "
                "Supported: variance, eval_eq, extremality, mixed, commutator"
            )


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


def create_output_dirs(config: MetricsConfig) -> dict[int, tuple[Path, Path]]:
    """Create individual output and plots directories for each max electron transfer value."""
    molecule = config.molecule.lower()
    basis_set = config.basis_set.lower()
    bond_length_str = f"{config.bond_length:.4f}".replace(".", "_")
    
    directories = {}
    
    for max_elec in config.max_elec_transfers:
        # Build output directory path
        if config.output_dir is not None:
            output_dir = config.output_dir / f"max_transfers_{max_elec}"
        else:
            output_dir = Path("outputs_") / "cluster_number" / molecule / basis_set / f"bond_{bond_length_str}"
            if config.bond_angle is not None:
                angle_str = f"{config.bond_angle:.4f}".replace(".", "_")
                output_dir = output_dir / f"angle_{angle_str}"
            output_dir = output_dir / f"max_transfers_{max_elec}"
        
        # Build plots directory path
        if config.plots_dir is not None:
            plots_dir = config.plots_dir / f"max_transfers_{max_elec}"
        else:
            plots_dir = Path("plots") / "cluster_number" / molecule / basis_set / f"bond_{bond_length_str}"
            if config.bond_angle is not None:
                angle_str = f"{config.bond_angle:.4f}".replace(".", "_")
                plots_dir = plots_dir / f"angle_{angle_str}"
            plots_dir = plots_dir / f"max_transfers_{max_elec}"
        
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        directories[max_elec] = (output_dir, plots_dir)
        
    return directories


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
    store_dir = Path(config.wavefunction_dir) / ( 
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


def extract_rdms(solver: Block2DMRGSolver, mps_tag: str, with_34_rdms=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract spin-summed 1- and 2-RDMs from MPS.
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
    D = rdm1_a + rdm1_b
    logger.info(f"D shape: {D.shape}, trace: {np.trace(D).round(8)}")

    rdm2_aa, rdm2_ab, rdm2_bb = solver.driver.get_2pdm(mps)
    Gamma = rdm2_aa + rdm2_bb + rdm2_ab + rdm2_ab.transpose(1, 0, 3, 2)
    logger.info(f"Gamma shape: {Gamma.shape}")

    if not with_34_rdms:
        return D, Gamma
    
    else:
        # get 3-rdm and 4-rdm in MO basis
        rdm3_aaa, rdm3_aab, rdm3_abb, rdm3_bbb = solver.driver.get_3pdm(mps)
        rdm3 = (
            rdm3_aaa 
            + rdm3_aab + rdm3_aab.transpose(0, 2, 1, 4, 3, 5) + rdm3_aab.transpose(2, 1, 0, 5, 4, 3) 
            + rdm3_abb + rdm3_abb.transpose(1, 0, 2, 3, 5, 4) + rdm3_abb.transpose(2, 1, 0, 5, 4, 3) 
            + rdm3_bbb
        )
        logger.info(f"rdm3 shape: {rdm3.shape}")

        rdm4_aaaa, rdm4_aaab, rdm4_aabb, rdm4_abbb, rdm4_bbbb = solver.driver.get_4pdm(mps)
        rdm4 = (
            # 0 betas (k=0)
            rdm4_aaaa
            
            # 1 beta (k=1)
            # Target beta positions: (3,), (2,), (1,), (0,)
            + rdm4_aaab
            + rdm4_aaab.transpose(0, 1, 3, 2,     5, 4, 6, 7) # (0, 1, 3, 2, 5, 4, 6, 7) 
            + rdm4_aaab.transpose(0, 3, 2, 1,     6, 5, 4, 7) # (0, 3, 1, 2, 5, 6, 4, 7) 
            + rdm4_aaab.transpose(3, 1, 2, 0,     7, 5, 6, 4) # (3, 0, 1, 2, 5, 6, 7, 4) 
            
            # 2 betas (k=2)
            # Target beta positions: (2,3), (1,3), (1,2), (0,3), (0,2), (0,1)
            + rdm4_aabb
            + rdm4_aabb.transpose(0, 2, 1, 3,     4, 6, 5, 7) # (0, 2, 1, 3, 4, 6, 5, 7) 
            + rdm4_aabb.transpose(0, 3, 2, 1,     6, 5, 4, 7) # (0, 2, 3, 1, 6, 4, 5, 7) 
            + rdm4_aabb.transpose(2, 1, 0, 3,     4, 7, 6, 5) # (2, 0, 1, 3, 4, 6, 7, 5) 
            + rdm4_aabb.transpose(3, 1, 2, 0,     7, 5, 6, 4) # (2, 0, 3, 1, 6, 4, 7, 5) 
            + rdm4_aabb.transpose(2, 3, 0, 1,     6, 7, 4, 5) # (2, 3, 0, 1, 6, 7, 4, 5)
            
            # 3 betas (k=3)
            # Target beta positions: (1,2,3), (0,2,3), (0,1,3), (0,1,2)
            + rdm4_abbb
            + rdm4_abbb.transpose(1, 0, 2, 3,     4, 5, 7, 6) # (1, 0, 2, 3, 4, 5, 7, 6) 
            + rdm4_abbb.transpose(2, 1, 0, 3,     4, 7, 6, 5) # (1, 2, 0, 3, 4, 7, 5, 6) 
            + rdm4_abbb.transpose(3, 1, 2, 0,     7, 5, 6, 4) # (1, 2, 3, 0, 7, 4, 5, 6) 
            
            # 4 betas (k=4)
            + rdm4_bbbb
        )
        logger.info(f"rdm4 shape: {rdm4.shape}")
        return D, Gamma, rdm3, rdm4

def optimize_orbital_basis(
    config: MetricsConfig,
    norb: int,
    nelec: tuple[int, int],
    rdms: tuple, # spin-summed 1- and 2-rdm, or spin-summed 1-, 2-, 3-, 4-rdms in MO basis
    cluster_matrix: np.ndarray,
    h1e: np.ndarray,
    g2e: np.ndarray
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Optimize orbital basis starting from MOs and NatOs.
    
    Returns:
        U_opt_list: List of optimized unitary transformations
        cost_data: List of (initial_cost, optimized_cost) tuples
    """
    logger.info("Starting orbital optimization...")

    if len(rdms) == 2:
        D_MOs, Gamma_MOs = rdms
    if len(rdms) == 4:
        D_MOs, Gamma_MOs, rdm3_MOs, rdm4_MOs = rdms
        assert config.type_cost_function == 'commutator', "Cost function type is not commutator so rdm3 and rdm4 are useless, but were anyways computed"
    
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
            f = extremality_cost(D, cluster_matrix, with_ghost=False)
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
        elif config.type_cost_function == "commutator":
            assert len(rdms) == 4, "For commutator cost, rdms needs to contain up to 4-rdm."
            g2e_full = pyscf.ao2mo.restore(1, g2e, norb)
            if np.allclose(D, D_MOs, atol=1e-12): # we are using MOs
                f = number_commutator_cost_v1(h1e, g2e_full, D, Gamma, rdm3_MOs, rdm4_MOs, cluster_matrix, with_ghost=False)
            if np.allclose(D, D_NatOs, atol=1e-12): # we are using natural orbitals
                rdm3_rotated = np.einsum('pi,qj,rk,sl,tm,un,ijklmn->pqrstu', U_NatOs_conj, U_NatOs_conj, U_NatOs_conj, U_NatOs, U_NatOs, U_NatOs, rdm3_MOs, optimize=True)
                rdm4_rotated = np.einsum('pi,qj,rk,sl,tm,un,vo,wa,ijklmnoa->pqrstuvw', U_NatOs_conj, U_NatOs_conj, U_NatOs_conj, U_NatOs_conj, U_NatOs, U_NatOs, U_NatOs, U_NatOs, rdm4_MOs, optimize=True)
                h1e_rotated = np.einsum('pr,qs,rs->pq', U_NatOs, U_NatOs_conj, h1e, optimize=True)
                g2e_full_rotated = np.einsum('pr,tu,qs,vz,rsuz->pqtv', U_NatOs, U_NatOs, U_NatOs_conj, U_NatOs_conj, g2e_full, optimize=True)
                f = number_commutator_cost_v1(h1e_rotated, g2e_full_rotated, D, Gamma, rdm3_rotated, rdm4_rotated, cluster_matrix, with_ghost=False)
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
    logger.info(f"Max electron transfers across clusters (list): {config.max_elec_transfers}")
    
    for i in range(len(U_list)):
        data_label = data_label_list[i]
        psi = psi_rotated_list[i]
        h_linop = h_linop_rotated_list[i]
        
        logger.info(f"Analyzing basis: {data_label}")
        
        transfer_results = []
        
        # Loop through each max_elec limit specified in the configuration
        for max_elec in config.max_elec_transfers:
            # Initialize result object with empty/default values
            transfer_result = SingleMaxelectransferResult(
                max_elec_transfers=max_elec,
                K_sectors_values=[],
                K_sectors_energies=[],
                K_sectors_retained_dimensions=[],
                num_retained_sectors=0,
                retained_dim=None,
                chem_accuracy_reached_sectors=False,

                K_states_values=[],
                K_states_energies=[],
                K_states_num_retained_state_sectors=[],
                num_retained_states=0,
                num_retained_state_sectors=0,
                chem_accuracy_reached_states=False
            )
            if not config.skip_K_sectors:
                # --- A) SECTOR-BASED APPROACH - Collect data ---
                K_sectors_values, K_sectors_energies, K_sectors_retained_dimensions, chem_accuracy_reached_sectors = (
                    get_K_sectors_values_energies(
                        psi, 
                        h_linop, 
                        dmrg_energy, 
                        sectors, 
                        max_elec,  # Use the specific integer here
                        "projected",
                        max_K_sectors=inf,
                        verbose=0
                    )
                )
                
                # Store data
                transfer_result.K_sectors_values = K_sectors_values
                transfer_result.K_sectors_energies = K_sectors_energies
                transfer_result.K_sectors_retained_dimensions = K_sectors_retained_dimensions
                transfer_result.num_retained_sectors = len(K_sectors_values)
                transfer_result.retained_dim = K_sectors_retained_dimensions[-1]
                transfer_result.chem_accuracy_reached_sectors = chem_accuracy_reached_sectors
                
                logger.info(
                    f"  A) Sector-based approach - max_elec={max_elec}: {len(K_sectors_values)} sectors, "
                    f"dim={K_sectors_retained_dimensions[-1]}, chem_accuracy={chem_accuracy_reached_sectors}"
                )

            if not config.skip_K_states:
                # --- B) DECOUPLED STATE-BASED APPROACH - Collect data ---

                K_states_values, K_states_energies, K_states_num_retained_state_sectors, chem_accuracy_reached_states, retained_sector_labels = (
                                    get_K_states_values_energies(
                                        psi, 
                                        h_linop, 
                                        dmrg_energy, 
                                        sectors, 
                                        max_elec,  # Use the specific integer here
                                        "projected",
                                        max_K_states=inf,
                                        verbose=0
                                    )
                                )

                transfer_result.K_states_values = K_states_values
                transfer_result.K_states_energies = K_states_energies
                transfer_result.K_states_num_retained_state_sectors = K_states_num_retained_state_sectors
                transfer_result.num_retained_states = len(K_states_values)
                transfer_result.num_retained_state_sectors = K_states_num_retained_state_sectors[-1]
                transfer_result.chem_accuracy_reached_states = chem_accuracy_reached_states

                logger.info(
                    f"  B) State-based approach - max_elec={max_elec}: {len(K_states_values)} decoupled states, "
                    f"num_sectors={K_states_num_retained_state_sectors[-1]}, chem_accuracy={chem_accuracy_reached_states}"
                )
                
            # Store data from both sector- and state-based approaches
            transfer_results.append(transfer_result)
        
        # Package all transfer limit results into the final BasisResult
        basis_result = BasisResult(
            data_label=data_label,
            transfer_results=transfer_results
        )
        basis_results.append(basis_result)
    
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
    directories = create_output_dirs(config) # couples (output_dir, plots_dir) - one per max_elec_transfers value
    
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
        "cost": config.type_cost_function,
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
    if config.type_cost_function == 'commutator':
        rdms_MOs = extract_rdms(solver, result.mps_tag, with_34_rdms=True) # with 3rdm and 4rdm
    else:
        rdms_MOs = extract_rdms(solver, result.mps_tag, with_34_rdms=False) # without 3rdm and 4rdm

    # Step 3: Orbital optimization (if enabled)
    U_opt_list = []
    cost_data = []
    
    if config.run_orbital_optimization:
        U_opt_list, cost_data = optimize_orbital_basis(
            config, norb, nelec, rdms_MOs, cluster_matrix, h1e, g2e
        )
    
    # Step 4: Sector analysis for each basis
    # Prepare U_list and data_label_list based on which analyses to run
    U_list = []
    data_label_list = []
    D_MOs = rdms_MOs[0]
    
    # Always add MOs
    U_list.append(np.eye(norb))
    data_label_list.append("MOs")
    
    # Add optimized from MOs if available
    if "Opt. from MOs" in config.analyze_bases and config.run_orbital_optimization and U_opt_list:
        U_list.append(U_opt_list[0])
        data_label_list.append("Opt. from MOs")
    
    # Add NatOs
    if "NatOs" in config.analyze_bases:
        spec, evecs = np.linalg.eigh(D_MOs)
        U_NatOs = evecs.T
        U_list.append(U_NatOs)
        data_label_list.append("NatOs")
    
    # Add optimized from NatOs if available
    if "Opt. from NatOs" in config.analyze_bases and config.run_orbital_optimization and len(U_opt_list) > 1:
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


def save_metrics(output: MetricsOutput, output_dir: Path, max_elec: int) -> Path:
    """
    Save metrics to JSON file for a specific max electron transfer limit.
    
    Returns:
        Path to the saved file
    """
    timestamp = output.metadata.get("timestamp", get_timestamp())
    git_hash = output.metadata.get("git_hash", "unknown")
    
    filename = f"results_{output.metadata['cost']}_{timestamp}_{git_hash}.json"
    filepath = output_dir / filename
    
    # Filter basis_results to only include the specified max_elec
    filtered_basis_results = []
    for result in output.basis_results:
        # Extract the specific transfer result from the list
        transfer_result = next(
            (tr for tr in result.transfer_results if tr.max_elec_transfers == max_elec), 
            None
        )
        
        if transfer_result:
            filtered_basis_results.append({
                "data_label": result.data_label,
                "transfer_results": [asdict(transfer_result)]
            })
        else:
            # Fallback if no matching result is found for this basis
            filtered_basis_results.append({
                "data_label": result.data_label,
                "transfer_results": []
            })
    
    # Update the max_elec_transfers in metadata to reflect this specific file's limit
    metadata_copy = output.metadata.copy()
    metadata_copy["max_elec_transfers"] = max_elec

    # Convert to dict for serialization
    output_dict = {
        "metadata": metadata_copy,
        "basis_results": filtered_basis_results
    }
    
    with open(filepath, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def load_metrics(filepath: Path) -> MetricsOutput:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    basis_results = []
    for result_dict in data["basis_results"]:
        # Reconstruct the SingleMaxelectransferResult objects
        transfer_results = [
            SingleMaxelectransferResult(**tr) 
            for tr in result_dict.get("transfer_results", [])
        ]
        
        basis_results.append(BasisResult(
            data_label=result_dict["data_label"],
            transfer_results=transfer_results
        ))
    
    return MetricsOutput(
        metadata=data["metadata"],
        basis_results=basis_results
    )


def generate_plots(
    output: MetricsOutput,
    plots_dir: Path,
    max_elec: int,
    show: bool = False,
    save: bool = True,
    skip_K_sectors: bool = False,
    skip_K_states: bool = False
) -> None:
    """Generate and save two pairs of plots (sector-based + decoupled states-based pairs) from metrics output for a specific max electron transfer limit."""
    import matplotlib
    if not save:
        matplotlib.use("Agg")  # Use non-interactive backend if not saving
    import matplotlib.pyplot as plt
    
    metadata = output.metadata
    basis_results = output.basis_results
    
    # Prepare data for plotting by filtering for max_elec
    data_label_list = []

    K_sectors_values_list = []
    K_sectors_energies_list = []
    K_sectors_retained_dimensions_list = []
    num_retained_sectors_list = []
    retained_dim_list = []
    chem_accuracy_reached_sectors_list = []

    K_states_values_list = []
    K_states_energies_list = []
    K_states_num_retained_state_sectors_list = []
    num_retained_states_list = []
    num_retained_state_sectors_list = []
    chem_accuracy_reached_states_list = []
    
    for r in basis_results:
        tr = next((t for t in r.transfer_results if t.max_elec_transfers == max_elec), None)
        if tr is None:
            continue
            
        data_label_list.append(r.data_label)

        K_sectors_values_list.append(tr.K_sectors_values)
        K_sectors_energies_list.append(tr.K_sectors_energies)
        K_sectors_retained_dimensions_list.append(tr.K_sectors_retained_dimensions)
        num_retained_sectors_list.append(tr.num_retained_sectors)
        retained_dim_list.append(tr.retained_dim)
        chem_accuracy_reached_sectors_list.append(tr.chem_accuracy_reached_sectors)

        K_states_values_list.append(tr.K_states_values)
        K_states_energies_list.append(tr.K_states_energies)
        K_states_num_retained_state_sectors_list.append(tr.K_states_num_retained_state_sectors)
        num_retained_states_list.append(tr.num_retained_states)
        num_retained_state_sectors_list.append(tr.num_retained_state_sectors)
        chem_accuracy_reached_states_list.append(tr.chem_accuracy_reached_states)
    
    # Guard clause in case there's no data matching the max_elec
    if not data_label_list:
        logger.warning(f"No data found for max_elec={max_elec}. Skipping plots.")
        return

    # Generate timestamp for filename
    timestamp = metadata.get("timestamp", get_timestamp())

    # --- A) Sector-based metrics ---
    if not skip_K_sectors:
        # Plot 1: Energy vs K_sectors
        fig = plot_energy_vs_K(
            data_label_list,
            K_sectors_values_list,
            K_sectors_energies_list,
            K_sectors_retained_dimensions_list,
            metadata["dmrg_energy"],
            molecule=metadata["molecule"],
            basis_set=metadata["basis_set"],
            norb=metadata.get("norb", "?"),
            cluster_sizes=metadata.get("cluster_sizes", "?"),
            max_elec_transfers=max_elec,  # pass the specific integer here
            cost=metadata["cost"],
            sectors_or_states='sectors'
        )
        if save:
            filename = f"energy_vs_sectors_{metadata['cost']}_{timestamp}.png"
            filepath = plots_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Energy vs sectors plot saved to {filepath}")
        if show:
            plt.show()
        plt.close(fig)
        
        # Plot 2: Dual bar chart
        if not any(chem_accuracy_reached_sectors_list):
            print(f"No basis reached chemical accuracy for max_elec={max_elec}. Skipping 2nd plot (dual bar chart).")
        else:
            # Prepare data for dual bar chart
            colors = [f'C{i}' for i in range(len(data_label_list))]
            colors_cp = list(compress(colors, chem_accuracy_reached_sectors_list))
            x_data_cp = list(compress(data_label_list, chem_accuracy_reached_sectors_list))
            y1_data_cp = list(compress(num_retained_sectors_list, chem_accuracy_reached_sectors_list))
            y2_data_cp = list(compress(retained_dim_list, chem_accuracy_reached_sectors_list))
                
            title = (
                f"Sector analysis for {metadata['molecule']} in {metadata['basis_set']} basis set \n "
                f"Num. orbitals = {metadata.get('norb', '?')}, cluster sizes = {metadata.get('cluster_sizes', '?')}, "
                f"max $e^-$ transfers = {max_elec}, cost = {metadata['cost']}"
            )

            fig = plot_dual_bar_chart(
                x_data=x_data_cp,
                y1_data=y1_data_cp,
                y2_data=y2_data_cp,
                label1="Number of retained sectors",
                label2="Retained dimension",
                title=title,
                colors=colors_cp,
                alpha=(0.8, 0.4)
            )

            if save:
                filename = f"retained_sectors_bar_chart_{metadata['cost']}_{timestamp}.png"
                filepath = plots_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Retained sectors bar chart saved to {filepath}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)

    # --- B) Decoupled states-based metrics ---
        if not skip_K_states:
            # Plot 1: Energy vs K_states
            fig = plot_energy_vs_K(
                data_label_list,
                K_states_values_list,
                K_states_energies_list,
                K_states_num_retained_state_sectors_list,
                metadata["dmrg_energy"],
                molecule=metadata["molecule"],
                basis_set=metadata["basis_set"],
                norb=metadata.get("norb", "?"),
                cluster_sizes=metadata.get("cluster_sizes", "?"),
                max_elec_transfers=max_elec,  # pass the specific integer here
                cost=metadata["cost"],
                sectors_or_states='states'
            )
            if save:
                filename = f"energy_vs_states_{metadata['cost']}_{timestamp}.png"
                filepath = plots_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Energy vs states plot saved to {filepath}")
            if show:
                plt.show()
            plt.close(fig)

            # Plot 2: Dual bar chart
            if not any(chem_accuracy_reached_states_list):
                print(f"No basis reached chemical accuracy for max_elec={max_elec}. Skipping 2nd plot (dual bar chart).")
            else:
                # Prepare data for dual bar chart
                colors = [f'C{i}' for i in range(len(data_label_list))]
                colors_cp = list(compress(colors, chem_accuracy_reached_states_list))
                x_data_cp = list(compress(data_label_list, chem_accuracy_reached_states_list))
                y1_data_cp = list(compress(num_retained_states_list, chem_accuracy_reached_states_list))
                y2_data_cp = list(compress(num_retained_state_sectors_list, chem_accuracy_reached_states_list))

                title = (
                    f"State analysis for {metadata['molecule']} in {metadata['basis_set']} basis set \n "
                    f"Num. orbitals = {metadata.get('norb', '?')}, cluster sizes = {metadata.get('cluster_sizes', '?')}, "
                    f"max $e^-$ transfers = {max_elec}, cost = {metadata['cost']}"
                )

                fig = plot_dual_bar_chart(
                    x_data=x_data_cp,
                    y1_data=y1_data_cp,
                    y2_data=y2_data_cp,
                    label1="Number of retained states",
                    label2="Number of retained state sectors",
                    title=title,
                    colors=colors_cp,
                    alpha=(0.8, 0.4)
                )

                if save:
                    filename = f"retained_states_bar_chart_{metadata['cost']}_{timestamp}.png"
                    filepath = plots_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches="tight")
                    logger.info(f"Retained states bar chart saved to {filepath}")

                if show:
                    plt.show()
                else:
                    plt.close(fig)



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
    parser.add_argument(
        "cost_function",
        type=str,
        help="Cost function type (variance, eval_eq, extremality, mixed, commutator)"
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
    nargs="+",
    default=[1, 2, 3],
    help="Maximum electron transfers (default: 1 2 3)",
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
        "--var-exponent",
        type=int,
        default=1,
        help="Variance exponent (default: 1)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Maximum optimization iterations (default: 500)"
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

    # Analysis options
    parser.add_argument(
        "--skip-K-sectors",
        action="store_true",
        help="Disable sector-based analysis"
    )
    parser.add_argument(
        "--skip-K-states",
        action="store_true",
        help="Disable decoupled states-based analysis"
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
        '--wavefunction-dir',
        type=str,
        default='wavefunctions',
        help='MPS wavefunction directory (for input and output)'
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
        type_cost_function=args.cost_function,
        bond_angle=args.bond_angle,
        cluster_matrix=cluster_matrix,
        max_elec_transfers=args.max_transfers,
        bond_dim=args.bond_dim,
        n_sweeps=args.n_sweeps,
        var_exponent=args.var_exponent,
        maxiter=args.maxiter,
        run_orbital_optimization=not args.no_optimization,
        n_threads=args.n_threads,
        reuse_wavefunction=not args.no_reuse,
        skip_K_sectors=args.skip_K_sectors,
        skip_K_states=args.skip_K_states,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
        wavefunction_dir=args.wavefunction_dir,
        save_plots=not args.no_plots,
        show_plots=args.show_plots,
    )
    
    # Override bases if specified
    if args.bases is not None:
        config.analyze_bases = args.bases
    
    logger.info(f"Starting computation for {args.molecule} in {args.basis_set} basis set")
    logger.info(f"Configuration: molecule={config.molecule}, basis set={config.basis_set}, "
                f"bond_length={config.bond_length}, max_transfers={config.max_elec_transfers}, cost function type={config.type_cost_function}")
    
    # Run computation
    try:
        output = compute_cluster_number_metrics(config)
        
        # create_output_dirs now returns a dictionary mapping max_elec -> (output_dir, plots_dir)
        directories_by_transfer = create_output_dirs(config)
        
        for max_elec, (output_dir, plots_dir) in directories_by_transfer.items():
            # Pass max_elec to save_metrics so it can extract the correct SingleMaxelectransferResult
            filepath = save_metrics(output, output_dir, max_elec=max_elec)
            
            # Generate plots if enabled
            if config.save_plots:
                generate_plots(
                    output, 
                    plots_dir, 
                    max_elec=max_elec, 
                    show=config.show_plots, 
                    save=True,
                    skip_K_sectors=config.skip_K_sectors,
                    skip_K_states=config.skip_K_states
                )

        logger.info("Computation completed successfully!")
        logger.info(f"Results saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Computation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
