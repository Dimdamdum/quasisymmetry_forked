"""Tests for objective-neutral MPS sector screening."""

from unittest.mock import Mock

import numpy as np

from src.sector_screening import (
    screen_sector_labels,
    screen_sector_labels_with_diagnostics,
)


def test_screening_uses_weights_and_neighbours_without_enumeration():
    solver = Mock()
    solver.get_mps.return_value = object()
    solver.dominant_sector_labels.return_value = [
        ((0, 0, 0), 0.7),
        ((1, 0, 0), 0.2),
    ]
    labels, weights = screen_sector_labels(
        solver,
        np.eye(3, dtype=int),
        minimum=4,
        maximum=6,
    )
    assert len(labels) == 4
    assert len(set(labels)) == 4
    assert weights[(0, 0, 0)] == 0.7
    solver.dominant_sector_labels.assert_called_once()


def test_screening_reports_recovered_and_selected_weight():
    solver = Mock()
    solver.get_mps.return_value = object()
    solver.dominant_sector_labels.return_value = [
        ((0, 0), 0.60),
        ((1, 0), 0.25),
        ((1, 1), 0.10),
    ]
    result = screen_sector_labels_with_diagnostics(
        solver,
        np.eye(2, dtype=int),
        minimum=1,
        maximum=2,
        cutoff=1.0e-5,
    )
    assert result.labels == ((0, 0), (1, 0))
    assert np.isclose(result.recovered_norm, 0.95)
    assert np.isclose(result.selected_weight, 0.85)
    assert result.cutoff == 1.0e-5
