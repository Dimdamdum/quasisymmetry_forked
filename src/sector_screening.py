"""Objective-neutral MPS screening of approximate-symmetry sectors."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SectorScreeningResult:
    """Screened labels and coefficient-threshold diagnostics."""

    labels: tuple[tuple[int, ...], ...]
    weights: dict[tuple[int, ...], float]
    recovered_norm: float
    selected_weight: float
    cutoff: float


def add_neighbour_labels(labels, n_bits, minimum):
    """Add deterministic Hamming-one neighbours until ``minimum`` is met."""
    ordered = []
    seen = set()
    for label in labels:
        label = tuple(int(bit) for bit in label)
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    for label in tuple(ordered):
        for bit in range(int(n_bits)):
            candidate = list(label)
            candidate[bit] ^= 1
            candidate = tuple(candidate)
            if candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
            if len(ordered) >= int(minimum):
                return ordered
    return ordered


def screen_sector_labels_with_diagnostics(
    solver,
    parity_matrix,
    reference_tag="GS",
    minimum=8,
    maximum=16,
    cutoff=1.0e-6,
):
    """Rank MPS sectors and return recovered-norm diagnostics.

    Determinants extracted above ``cutoff`` are used only to estimate sector
    weights. They do not define the Hilbert space of later sector solves.
    """
    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    ket = solver.get_mps(reference_tag)
    weighted = solver.dominant_sector_labels(
        parity_matrix,
        ket=ket,
        cutoff=float(cutoff),
        max_sectors=None,
    )
    weights = {
        tuple(int(bit) for bit in label): float(weight)
        for label, weight in weighted
    }
    labels = add_neighbour_labels(
        [label for label, _weight in weighted],
        parity_matrix.shape[0],
        minimum,
    )[: int(maximum)]
    recovered_norm = float(sum(weights.values()))
    selected_weight = float(sum(weights.get(label, 0.0) for label in labels))
    return SectorScreeningResult(
        labels=tuple(labels),
        weights=weights,
        recovered_norm=recovered_norm,
        selected_weight=selected_weight,
        cutoff=float(cutoff),
    )


def screen_sector_labels(
    solver,
    parity_matrix,
    reference_tag="GS",
    minimum=8,
    maximum=16,
    cutoff=1.0e-6,
):
    """Compatibility wrapper returning ``(labels, weight_map)``."""
    result = screen_sector_labels_with_diagnostics(
        solver,
        parity_matrix,
        reference_tag=reference_tag,
        minimum=minimum,
        maximum=maximum,
        cutoff=cutoff,
    )
    return list(result.labels), dict(result.weights)
