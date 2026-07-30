"""Tests for physical selected-sector support generation."""

import numpy as np

from src.sector_utils import symmetry_sectors
from src.selected_sector_lanczos import selected_sector_supports


def test_selected_supports_match_complete_small_system_enumeration():
    parity = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=int)
    labels = [(0, 0), (1, 0), (1, 1)]
    expected = symmetry_sectors(parity, norb=3, nelec=(1, 1))
    selected = selected_sector_supports(
        parity,
        labels,
        norb=3,
        nelec=(1, 1),
        print_progress=False,
    )
    for label in labels:
        assert selected[label]["full_addresses"].tolist() == expected[label]
        assert selected[label]["dimension"] == len(expected[label])


def test_selected_supports_validate_label_width():
    parity = np.asarray([[1, 0], [0, 1]], dtype=int)
    try:
        selected_sector_supports(
            parity,
            [(0,)],
            norb=2,
            nelec=(1, 1),
            print_progress=False,
        )
    except ValueError as error:
        assert "one bit per parity row" in str(error)
    else:
        raise AssertionError("invalid label width was accepted")
