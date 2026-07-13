"""Clifford-frame construction and solution of tapered LAS sectors."""

import itertools
import json
from pathlib import Path

import numpy as np
import openfermion as of
import pyscf.fci.cistring
import scipy.sparse.linalg

from external_imports import Clifford, taper_hamiltonian


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
    sectors = {}
    for alpha in itertools.combinations(range(norb), int(nelec[0])):
        for beta in itertools.combinations(range(norb), int(nelec[1])):
            bits = occupation_bits(alpha, beta, norb)
            transformed = apply_clifford_to_basis_bits(bits, clifford)
            label = tuple(transformed[:n_symmetries])
            residual_index = bits_to_index(transformed[n_symmetries:])
            sectors.setdefault(label, []).append(residual_index)

    for label, indices in sectors.items():
        unique = sorted(set(indices))
        if len(unique) != len(indices):
            raise ValueError(f"Clifford mapping is not one-to-one in sector {label}")
        sectors[label] = unique
    return sectors


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


def solve_tapered_sector(frame, label, physical_indices, n_roots):
    """Solve low roots of one physical tapered sector."""
    operator = tapered_operator(frame, label, label)
    matrix = restricted_operator_matrix(
        operator,
        frame["n_residual_qubits"],
        physical_indices,
    )
    matrix = 0.5 * (matrix + matrix.getH())
    dimension = matrix.shape[0]
    if dimension == 0:
        raise ValueError(f"sector {label} has no physical determinants")

    root_count = min(max(1, int(n_roots)), dimension)
    if dimension <= 2 or root_count >= dimension - 1:
        energies, vectors = np.linalg.eigh(matrix.toarray())
        energies = energies[:root_count]
        vectors = vectors[:, :root_count]
        solver = "dense"
    else:
        energies, vectors = scipy.sparse.linalg.eigsh(
            matrix,
            k=root_count,
            which="SA",
            tol=1e-12,
        )
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
        solver = "eigsh"

    return {
        "label": tuple(label),
        "operator": operator,
        "physical_indices": list(physical_indices),
        "energies": np.real_if_close(energies),
        "vectors": vectors,
        "dimension": int(dimension),
        "solver": solver,
        "pauli_count": len(operator.terms),
        "lcu_one_norm": float(sum(abs(complex(value)) for value in operator.terms.values())),
    }


def pauli_lcu_is_hermitian(operator, n_qubits, atol=1e-10):
    """Check Hermiticity using the sparse matrix representation."""
    matrix = of.get_sparse_operator(operator, n_qubits=n_qubits).tocsr()
    difference = matrix - matrix.getH()
    if difference.nnz == 0:
        return True
    return bool(np.max(np.abs(difference.data)) <= atol)


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


def candidate_hamiltonian(frame, candidates):
    """Build the coupled Hamiltonian in the tapered sector-state basis."""
    dimension = len(candidates)
    h_coupled = np.zeros((dimension, dimension), dtype=complex)
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
    return list(np.argsort(np.asarray(weights))[::-1])


def perturbative_candidate_order(h_coupled, denominator_floor=1e-8):
    """Greedily order states by a coupling-squared-over-gap estimate."""
    if h_coupled.shape[0] == 0:
        return []
    chosen = [int(np.argmin(np.real(np.diag(h_coupled))))]
    remaining = set(range(h_coupled.shape[0])) - set(chosen)

    while remaining:
        submatrix = h_coupled[np.ix_(chosen, chosen)]
        energies, vectors = np.linalg.eigh(submatrix)
        ground_energy = float(np.real(energies[0]))
        ground_vector = vectors[:, 0]

        best_index = None
        best_score = -1.0
        for index in remaining:
            couplings = h_coupled[np.ix_(chosen, [index])][:, 0]
            effective_coupling = np.vdot(ground_vector, couplings)
            denominator = max(
                abs(float(np.real(h_coupled[index, index])) - ground_energy),
                denominator_floor,
            )
            score = float(abs(effective_coupling) ** 2 / denominator)
            if score > best_score:
                best_score = score
                best_index = index

        chosen.append(int(best_index))
        remaining.remove(best_index)
    return chosen


def perturbative_coupled_energy_curve(
    h_coupled,
    exact_energy=None,
    tolerance=0.0016,
    denominator_floor=1e-8,
):
    """Select perturbative candidates and stop when the target is reached."""
    if h_coupled.shape[0] == 0:
        return {"order": [], "energies": [], "K": None, "converged": False}

    chosen = [int(np.argmin(np.real(np.diag(h_coupled))))]
    remaining = set(range(h_coupled.shape[0])) - set(chosen)
    energies = []
    k_epsilon = None

    while chosen:
        submatrix = h_coupled[np.ix_(chosen, chosen)]
        eigenvalues, eigenvectors = np.linalg.eigh(submatrix)
        ground_energy = float(np.real(eigenvalues[0]))
        ground_vector = eigenvectors[:, 0]
        energies.append(ground_energy)

        if exact_energy is not None and abs(ground_energy - exact_energy) <= tolerance:
            k_epsilon = len(chosen)
            break
        if not remaining:
            break

        best_index = None
        best_score = -1.0
        for index in remaining:
            couplings = h_coupled[np.ix_(chosen, [index])][:, 0]
            effective_coupling = np.vdot(ground_vector, couplings)
            denominator = max(
                abs(float(np.real(h_coupled[index, index])) - ground_energy),
                denominator_floor,
            )
            score = float(abs(effective_coupling) ** 2 / denominator)
            if score > best_score:
                best_score = score
                best_index = index

        chosen.append(int(best_index))
        remaining.remove(best_index)

    return {
        "order": chosen,
        "energies": energies,
        "K": k_epsilon,
        "converged": k_epsilon is not None,
    }


def coupled_energy_curve(h_coupled, order, exact_energy=None, tolerance=0.0016):
    """Return the variational energy curve and first K reaching a tolerance."""
    energies = []
    k_epsilon = None
    for count in range(1, len(order) + 1):
        indices = order[:count]
        energy = float(np.linalg.eigvalsh(h_coupled[np.ix_(indices, indices)])[0])
        energies.append(energy)
        if exact_energy is not None and k_epsilon is None:
            if abs(energy - exact_energy) <= tolerance:
                k_epsilon = count
                break
    return {
        "order": [int(index) for index in order],
        "energies": energies,
        "K": k_epsilon,
        "converged": k_epsilon is not None,
    }
