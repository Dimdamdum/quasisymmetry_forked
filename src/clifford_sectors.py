"""Clifford-frame construction and solution of tapered LAS sectors."""

import itertools
import json
from pathlib import Path

import numpy as np
import openfermion as of
import pyscf.fci.cistring
import scipy.sparse.linalg

from external_imports import Clifford, taper_hamiltonian
from src.coupled_energy_core import (
    coupled_dimension_from_order,
    one_shot_from_hamiltonian,
    reference_candidate_order as _reference_candidate_order,
)


def qubit_operator_to_data(operator):
    """Convert a QubitOperator to JSON-compatible data."""
    terms = []
    for pauli_term, coefficient in operator.terms.items():
        value = complex(coefficient)
        terms.append(
            {
                "pauli": [[int(index), pauli] for index, pauli in pauli_term],
                "coefficient": [float(value.real), float(value.imag)],
            }
        )
    return {"terms": terms}


def qubit_operator_from_data(data):
    """Build a QubitOperator from JSON-compatible data."""
    operator = of.QubitOperator()
    for item in data["terms"]:
        pauli_term = tuple((int(index), str(pauli)) for index, pauli in item["pauli"])
        real, imag = item["coefficient"]
        operator += of.QubitOperator(pauli_term, complex(real, imag))
    operator.compress(abs_tol=1e-12)
    return operator


def save_symmetry_manifest(path, symmetries, parity_matrix, metadata=None):
    """Save ordered signed symmetries and their parity matrix."""
    output = {
        "schema": "quasisymmetry.symmetry_manifest",
        "version": 1,
        "n_qubits": int(parity_matrix.shape[1]),
        "symmetries": [qubit_operator_to_data(symmetry) for symmetry in symmetries],
        "parity_matrix": np.asarray(parity_matrix, dtype=int).tolist(),
    }
    if metadata:
        output["metadata"] = metadata
    with Path(path).open("w") as file:
        json.dump(output, file, indent=2)


def load_symmetry_manifest(path):
    """Load a symmetry manifest and reconstruct its QubitOperators."""
    with Path(path).open() as file:
        data = json.load(file)
    if data.get("schema") != "quasisymmetry.symmetry_manifest":
        raise ValueError("not a quasisymmetry symmetry manifest")
    data["symmetries"] = [
        qubit_operator_from_data(item) for item in data["symmetries"]
    ]
    data["parity_matrix"] = np.asarray(data["parity_matrix"], dtype=int)
    return data


def z_symmetries_from_parity_matrix(parity_matrix, norb):
    """Convert a spatial- or spin-orbital parity matrix to Z products."""
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if parity_matrix.shape[1] == norb:
        expanded = np.zeros((parity_matrix.shape[0], 2 * norb), dtype=int)
        for orbital in range(norb):
            expanded[:, 2 * orbital] = parity_matrix[:, orbital]
            expanded[:, 2 * orbital + 1] = parity_matrix[:, orbital]
    elif parity_matrix.shape[1] == 2 * norb:
        expanded = parity_matrix
    else:
        raise ValueError("parity matrix must have norb or 2*norb columns")

    symmetries = []
    for row in expanded:
        support = np.flatnonzero(row % 2)
        if support.size == 0:
            raise ValueError("identity rows are not independent LAS generators")
        term = tuple((int(index), "Z") for index in support)
        symmetries.append(of.QubitOperator(term, 1.0))
    return symmetries


def validate_z_symmetries(symmetries, n_qubits):
    """Validate the commuting single-term Z products supported by this backend."""
    if not symmetries:
        raise ValueError("at least one symmetry is required")
    for symmetry in symmetries:
        if len(symmetry.terms) != 1:
            raise ValueError("each symmetry must contain exactly one Pauli term")
        term, coefficient = next(iter(symmetry.terms.items()))
        if not term or any(pauli != "Z" for _, pauli in term):
            raise ValueError("the first Clifford backend supports Z products only")
        if any(index < 0 or index >= n_qubits for index, _ in term):
            raise ValueError("symmetry index lies outside the qubit register")
        coefficient = complex(coefficient)
        if not np.isclose(coefficient.imag, 0.0, atol=1e-12):
            raise ValueError("symmetry coefficients must be real")
        if not np.isclose(abs(coefficient.real), 1.0, atol=1e-12):
            raise ValueError("symmetry coefficients must be +1 or -1")


def molecular_hamiltonian_to_jw(molecular_hamiltonian, nelec):
    """Convert an ffsim MolecularHamiltonian to a Jordan-Wigner Pauli LCU."""
    norb = molecular_hamiltonian.one_body_tensor.shape[0]
    multiplicity = abs(int(nelec[0]) - int(nelec[1])) + 1
    molecule = of.MolecularData(
        geometry="FCIDUMP",
        basis="unknown",
        multiplicity=multiplicity,
        charge=0,
    )
    molecule.n_orbitals = norb
    molecule.n_qubits = 2 * norb
    molecule.n_electrons = int(sum(nelec))
    molecule.nuclear_repulsion = float(np.real(molecular_hamiltonian.constant))
    molecule.one_body_integrals = np.asarray(
        molecular_hamiltonian.one_body_tensor, dtype=float
    )
    # ffsim stores (p,s,q,r); OpenFermion MolecularData stores (p,q,r,s).
    molecule.two_body_integrals = np.transpose(
        np.asarray(molecular_hamiltonian.two_body_tensor, dtype=float),
        (0, 2, 3, 1),
    )
    fermion_hamiltonian = of.get_fermion_operator(molecule.get_molecular_hamiltonian())
    jw_hamiltonian = of.jordan_wigner(fermion_hamiltonian)
    jw_hamiltonian.compress(abs_tol=1e-12)
    return jw_hamiltonian


def build_clifford_frame(jw_hamiltonian, symmetries, n_qubits):
    """Map ordered Z-product symmetries to leading positive-Z qubits."""
    validate_z_symmetries(symmetries, n_qubits)
    clifford = Clifford.from_symmetries(
        symmetries,
        n_qubits=n_qubits,
        symmetry_qubits_first=True,
        synthesis_basis="Z",
        generator_mapping="positive_z",
    )
    transformed_symmetries = list(clifford.transformed_symmetries)
    expected = [of.QubitOperator(((index, "Z"),), 1.0) for index in range(len(symmetries))]
    for actual, target in zip(transformed_symmetries, expected):
        if actual != target:
            raise ValueError("Clifford did not map an input symmetry to the expected +Z")
    transformed_hamiltonian = clifford.transform(jw_hamiltonian)
    transformed_hamiltonian.compress(abs_tol=1e-12)
    return {
        "clifford": clifford,
        "hamiltonian": transformed_hamiltonian,
        "n_qubits": int(n_qubits),
        "n_symmetries": len(symmetries),
        "n_residual_qubits": int(n_qubits - len(symmetries)),
    }


def prepare_clifford_context(symmetries, norb, nelec):
    """Prepare rotation-independent Clifford and physical-sector information."""
    n_qubits = 2 * int(norb)
    frame = build_clifford_frame(of.QubitOperator(), symmetries, n_qubits)
    physical_sectors = physical_sector_indices(
        int(norb),
        nelec,
        frame["clifford"],
        frame["n_symmetries"],
    )
    return {
        "clifford": frame["clifford"],
        "n_qubits": frame["n_qubits"],
        "n_symmetries": frame["n_symmetries"],
        "n_residual_qubits": frame["n_residual_qubits"],
        "physical_sectors": physical_sectors,
    }


def transform_hamiltonian_in_context(jw_hamiltonian, context):
    """Transform a JW Hamiltonian using an existing Clifford context."""
    transformed = context["clifford"].transform(jw_hamiltonian)
    transformed.compress(abs_tol=1e-12)
    return {
        "clifford": context["clifford"],
        "hamiltonian": transformed,
        "n_qubits": context["n_qubits"],
        "n_symmetries": context["n_symmetries"],
        "n_residual_qubits": context["n_residual_qubits"],
    }


def occupation_bits(alpha_occupied, beta_occupied, norb):
    """Return an interleaved Jordan-Wigner occupation bitstring."""
    bits = [0] * (2 * norb)
    for orbital in alpha_occupied:
        bits[2 * orbital] = 1
    for orbital in beta_occupied:
        bits[2 * orbital + 1] = 1
    return bits


def ci_vector_to_jw_state(ci_vector, norb, nelec, threshold=1e-14):
    """Embed a PySCF/ffsim fixed-spin CI vector in the full JW register."""
    alpha_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), int(nelec[0])), dtype=np.int64
    )
    beta_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), int(nelec[1])), dtype=np.int64
    )
    ci_matrix = np.asarray(ci_vector).reshape(len(alpha_strings), len(beta_strings))
    state = np.zeros(1 << (2 * norb), dtype=complex)

    for alpha_address, alpha_string in enumerate(alpha_strings):
        alpha = [orbital for orbital in range(norb) if (int(alpha_string) >> orbital) & 1]
        for beta_address, beta_string in enumerate(beta_strings):
            coefficient = ci_matrix[alpha_address, beta_address]
            if abs(coefficient) <= threshold:
                continue
            beta = [orbital for orbital in range(norb) if (int(beta_string) >> orbital) & 1]
            inversions = sum(beta_orbital < alpha_orbital for alpha_orbital in alpha for beta_orbital in beta)
            phase = -1.0 if inversions % 2 else 1.0
            state[bits_to_index(occupation_bits(alpha, beta, norb))] = phase * coefficient
    return state


def bits_to_index(bits):
    """Convert qubit-order bits to an OpenFermion matrix index."""
    n_qubits = len(bits)
    return sum(int(bit) << (n_qubits - 1 - qubit) for qubit, bit in enumerate(bits))


def apply_clifford_to_basis_bits(bits, clifford):
    """Apply a Z-native Clifford to one computational-basis bitstring."""
    transformed = list(int(bit) for bit in bits)
    for gate in clifford.parsed_gates:
        name = gate[0]
        if name == "X":
            transformed[gate[1]] ^= 1
        elif name == "CNOT":
            transformed[gate[2]] ^= transformed[gate[1]]
        elif name in ("S", "Sdg"):
            continue
        else:
            raise ValueError(
                "fixed-particle mapping requires a Z-native Clifford without Hadamards"
            )

    permuted = [0] * len(transformed)
    for old_qubit, new_qubit in enumerate(clifford.permutation):
        permuted[new_qubit] = transformed[old_qubit]
    return permuted


def physical_sector_indices(norb, nelec, clifford, n_symmetries):
    """Map fixed-(Nalpha,Nbeta) determinants to tapered residual indices."""
    return physical_clifford_basis(norb, nelec, clifford, n_symmetries)[
        "residual_indices"
    ]


def physical_clifford_basis(norb, nelec, clifford, n_symmetries):
    """Map physical determinants to their Clifford-frame sector coordinates."""
    sector_entries = {}
    full_indices = []
    for alpha in itertools.combinations(range(norb), int(nelec[0])):
        for beta in itertools.combinations(range(norb), int(nelec[1])):
            bits = occupation_bits(alpha, beta, norb)
            transformed = apply_clifford_to_basis_bits(bits, clifford)
            label = tuple(transformed[:n_symmetries])
            residual_index = bits_to_index(transformed[n_symmetries:])
            full_index = bits_to_index(transformed)
            physical_position = len(full_indices)
            full_indices.append(full_index)
            sector_entries.setdefault(label, []).append(
                (residual_index, physical_position)
            )

    if len(set(full_indices)) != len(full_indices):
        raise ValueError("Clifford mapping is not one-to-one on physical determinants")

    residual_indices = {}
    physical_positions = {}
    for label, entries in sector_entries.items():
        ordered = sorted(entries)
        indices = [item[0] for item in ordered]
        positions = [item[1] for item in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Clifford mapping is not one-to-one in sector {label}")
        residual_indices[label] = indices
        physical_positions[label] = positions
    return {
        "full_indices": full_indices,
        "residual_indices": residual_indices,
        "physical_positions": physical_positions,
    }


def parse_sector_labels(text, n_symmetries):
    """Parse labels such as '000,011,101'; return None for all sectors."""
    if text is None or not text.strip():
        return None
    labels = []
    for item in text.split(","):
        item = item.strip()
        if len(item) != n_symmetries or any(bit not in "01" for bit in item):
            raise ValueError(f"invalid {n_symmetries}-bit sector label: {item}")
        labels.append(tuple(int(bit) for bit in item))
    return labels


def tapered_operator(frame, bra_label, ket_label):
    """Construct <bra|Hc|ket> on the residual qubits."""
    if len(bra_label) != frame["n_symmetries"] or len(ket_label) != frame["n_symmetries"]:
        raise ValueError("sector label length does not match the symmetry count")
    return taper_hamiltonian(
        frame["hamiltonian"],
        list(bra_label),
        list(ket_label),
        shift_to_zero=True,
    )


def restricted_operator_matrix(operator, n_qubits, row_indices, column_indices=None):
    """Return an operator matrix restricted to selected residual indices."""
    if column_indices is None:
        column_indices = row_indices
    matrix = of.get_sparse_operator(operator, n_qubits=n_qubits).tocsr()
    return matrix[np.asarray(row_indices), :][:, np.asarray(column_indices)]


def pauli_term_action_masks(pauli_term, n_qubits):
    """Return bit masks and a phase for one Pauli word acting on a ket."""
    flip_mask = 0
    sign_mask = 0
    y_count = 0
    for qubit, pauli in pauli_term:
        mask = 1 << (n_qubits - 1 - qubit)
        if pauli == "X":
            flip_mask |= mask
        elif pauli == "Y":
            flip_mask |= mask
            sign_mask |= mask
            y_count += 1
        elif pauli == "Z":
            sign_mask |= mask
        else:
            raise ValueError(f"unsupported Pauli factor: {pauli}")
    return flip_mask, sign_mask, 1j**y_count


def physical_clifford_matrix(frame, full_indices):
    """Build Hc only on the physical Clifford-frame determinant support.

    This applies every Pauli word in the transformed Hamiltonian directly to
    the physical computational-basis states.  It avoids making a full
    2**n_qubits sparse matrix or separately tapering every sector pair.
    """
    dimension = len(full_indices)
    index_to_position = {
        int(full_index): position for position, full_index in enumerate(full_indices)
    }
    rows = []
    columns = []
    values = []

    for pauli_term, coefficient in frame["hamiltonian"].terms.items():
        flip_mask, sign_mask, y_phase = pauli_term_action_masks(
            pauli_term, frame["n_qubits"]
        )
        for column, source_index in enumerate(full_indices):
            target_index = int(source_index) ^ flip_mask
            row = index_to_position.get(target_index)
            if row is None:
                continue
            sign = -1.0 if (int(source_index) & sign_mask).bit_count() % 2 else 1.0
            rows.append(row)
            columns.append(column)
            values.append(complex(coefficient) * y_phase * sign)

    matrix = scipy.sparse.coo_matrix(
        (values, (rows, columns)), shape=(dimension, dimension), dtype=complex
    ).tocsr()
    return 0.5 * (matrix + matrix.getH())


def lowest_sector_eigenpairs(matrix, n_roots):
    """Return the requested lowest eigenpairs of one Hermitian sector block."""
    dimension = matrix.shape[0]
    if dimension == 0:
        raise ValueError("sector has no physical determinants")

    root_count = min(max(1, int(n_roots)), dimension)
    if dimension <= 2 or root_count >= dimension - 1:
        energies, vectors = np.linalg.eigh(matrix.toarray())
        return energies[:root_count], vectors[:, :root_count], "dense"

    energies, vectors = scipy.sparse.linalg.eigsh(
        matrix,
        k=root_count,
        which="SA",
        tol=1e-12,
    )
    order = np.argsort(energies)
    return energies[order], vectors[:, order], "eigsh"


def solve_tapered_sector(frame, label, physical_indices, n_roots):
    """Solve low roots of one physical tapered sector."""
    operator = tapered_operator(frame, label, label)
    matrix = restricted_operator_matrix(
        operator,
        frame["n_residual_qubits"],
        physical_indices,
    )
    matrix = 0.5 * (matrix + matrix.getH())
    energies, vectors, solver = lowest_sector_eigenpairs(matrix, n_roots)

    return {
        "label": tuple(label),
        "operator": operator,
        # Keep the restricted diagonal block so coupled-space construction does
        # not taper and restrict the same sector a second time.
        "matrix": matrix,
        "physical_indices": list(physical_indices),
        "energies": np.real_if_close(energies),
        "vectors": vectors,
        "dimension": int(matrix.shape[0]),
        "solver": solver,
        "pauli_count": len(operator.terms),
        "lcu_one_norm": float(sum(abs(complex(value)) for value in operator.terms.values())),
    }


def solve_physical_clifford_sector(
    physical_matrix,
    label,
    residual_indices,
    physical_positions,
    n_roots,
):
    """Solve one sector by slicing a prebuilt physical Clifford-frame matrix."""
    positions = np.asarray(physical_positions, dtype=int)
    matrix = physical_matrix[positions, :][:, positions]
    matrix = 0.5 * (matrix + matrix.getH())
    energies, vectors, solver = lowest_sector_eigenpairs(matrix, n_roots)
    return {
        "label": tuple(label),
        "matrix": matrix,
        "physical_indices": list(residual_indices),
        "physical_positions": list(physical_positions),
        "energies": np.real_if_close(energies),
        "vectors": vectors,
        "dimension": int(matrix.shape[0]),
        "solver": solver,
    }


def pauli_lcu_is_hermitian(operator, n_qubits, atol=1e-10):
    """Check Hermiticity without materializing the full Pauli matrix.

    Each Pauli word is Hermitian, so a QubitOperator is Hermitian exactly when
    every collected Pauli coefficient is real.  The qubit count is retained in
    the interface for compatibility with existing callers.
    """
    del n_qubits
    return all(abs(complex(coefficient).imag) <= atol for coefficient in operator.terms.values())


def sector_reference_amplitudes(transformed_state, label, residual_indices, n_residual_qubits):
    """Extract reference amplitudes for one label and physical residual support."""
    label_index = bits_to_index(label)
    offset = label_index << n_residual_qubits
    state = np.asarray(transformed_state).reshape(-1)
    return np.asarray([state[offset + index] for index in residual_indices], dtype=complex)


def sector_state_candidates(sector_results):
    """Flatten solved sector roots into simple candidate dictionaries."""
    candidates = []
    for label in sorted(sector_results):
        result = sector_results[label]
        for root, energy in enumerate(result["energies"]):
            candidates.append(
                {
                    "label": tuple(label),
                    "root": int(root),
                    "energy": float(np.real(energy)),
                    "vector": np.asarray(result["vectors"][:, root], dtype=complex),
                    "physical_indices": list(result["physical_indices"]),
                }
            )
    return candidates


def candidate_hamiltonian(frame, candidates, block_cache=None):
    """Build the coupled Hamiltonian from cached tapered sector-pair blocks."""
    dimension = len(candidates)
    h_coupled = np.zeros((dimension, dimension), dtype=complex)
    if block_cache is None:
        block_cache = {}

    for i, bra in enumerate(candidates):
        for j in range(i, dimension):
            ket = candidates[j]
            key = (bra["label"], ket["label"])
            if key not in block_cache:
                operator = tapered_operator(frame, bra["label"], ket["label"])
                block_cache[key] = restricted_operator_matrix(
                    operator,
                    frame["n_residual_qubits"],
                    bra["physical_indices"],
                    ket["physical_indices"],
                )
            block = block_cache[key]
            value = np.vdot(bra["vector"], block @ ket["vector"])
            h_coupled[i, j] = value
            h_coupled[j, i] = value.conjugate()

    h_coupled = 0.5 * (h_coupled + h_coupled.conj().T)
    return h_coupled, block_cache


def candidate_reference_weights(frame, candidates, transformed_reference):
    """Return squared overlaps of sector eigenstates with a reference state."""
    weights = []
    for candidate in candidates:
        amplitudes = sector_reference_amplitudes(
            transformed_reference,
            candidate["label"],
            candidate["physical_indices"],
            frame["n_residual_qubits"],
        )
        overlap = np.vdot(candidate["vector"], amplitudes)
        weights.append(float(abs(overlap) ** 2))
    return np.asarray(weights)


def reference_candidate_order(weights):
    """Order candidates from largest to smallest reference weight."""
    return _reference_candidate_order(weights)


def perturbative_coupled_energy_curve(
    h_coupled,
    exact_energy=None,
    tolerance=0.0016,
    denominator_floor=1e-8,
    tau_pt=1e-12,
    block_size=1,
):
    """One-shot PT ordering + nested variational curve (Clifford entry point)."""
    result = one_shot_from_hamiltonian(
        np.asarray(h_coupled),
        e_exact=exact_energy,
        tol=tolerance,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=denominator_floor,
    )
    return result.as_curve()


def coupled_energy_curve(h_coupled, order, exact_energy=None, tolerance=0.0016):
    """Nested variational energy curve along a fixed candidate order."""
    result = coupled_dimension_from_order(
        np.asarray(h_coupled),
        order,
        e_exact=exact_energy,
        tol=tolerance,
        k_start=1,
    )
    return result.as_curve()
