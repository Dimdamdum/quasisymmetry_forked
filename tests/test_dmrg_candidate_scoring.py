"""Fast tests for objective-neutral per-candidate MPS scoring helpers."""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.dmrg_costs import (
    _copy_mps_store_for_tag,
    candidate_index_chunks,
    commutator_scores_by_row,
    parity_expectations_by_row,
)


class FakeCosts:
    """Small vector analogue of the MPS operations used by the scorer."""

    def __init__(self):
        self.parity_matrix = np.asarray([[1, 0], [0, 1]], dtype=int)
        self.pairs = []
        self.solver = SimpleNamespace(
            n_sites=2,
            mps_norm2=lambda vector: float(np.vdot(vector, vector).real),
            mps_overlap=lambda left, right: np.vdot(left, right),
        )
        self.ket = np.asarray([0.8, 0.6])
        self._h_mpo = np.asarray([[0.0, 0.4], [0.4, 1.0]])
        self._eta = None
        self._eval_count = 0

    def _ensure_eta(self):
        self._eta = self._h_mpo @ self.ket

    def _apply(self, matrix, ket, _prefix):
        return matrix @ ket

    def _apply_symmetry(self, row, _rotation, ket, _prefix):
        sign = np.asarray([1.0, -1.0]) if row[0] else np.asarray([-1.0, 1.0])
        return sign * ket


def test_per_candidate_scores_match_explicit_commutators():
    costs = FakeCosts()
    scores = commutator_scores_by_row(costs, np.zeros(0))
    expected = []
    for row in costs.parity_matrix:
        sign = np.asarray([1.0, -1.0]) if row[0] else np.asarray([-1.0, 1.0])
        residual = costs._h_mpo @ (sign * costs.ket) - sign * (
            costs._h_mpo @ costs.ket
        )
        expected.append(np.vdot(residual, residual).real)
    np.testing.assert_allclose(scores, expected)


def test_selected_parity_expectations_use_the_same_actions():
    costs = FakeCosts()
    values = parity_expectations_by_row(
        costs, np.zeros(0), costs.parity_matrix
    )
    expected = []
    for row in costs.parity_matrix:
        sign = np.asarray([1.0, -1.0]) if row[0] else np.asarray([-1.0, 1.0])
        expected.append(np.vdot(costs.ket, sign * costs.ket).real)
    np.testing.assert_allclose(values, expected)


def test_candidate_chunks_are_balanced_and_complete():
    chunks = candidate_index_chunks(91, 4)
    assert [len(chunk) for chunk in chunks] == [23, 23, 23, 22]
    assert [item for chunk in chunks for item in chunk] == list(range(91))
    assert candidate_index_chunks(3, 8) == [[0], [1], [2]]
    with pytest.raises(ValueError, match="at least one candidate"):
        candidate_index_chunks(0, 4)


def test_worker_store_copy_excludes_old_intermediates():
    root = Path(tempfile.mkdtemp(prefix="dmrg_store_copy_"))
    try:
        source = root / "source"
        target = root / "target"
        source.mkdir()
        keep = [
            "integrals.npz",
            "metadata.json",
            "GS-mps_info.bin",
            "F.MPS.GS.0",
            "F.MPS.INFO.GS.LEFT.0",
        ]
        for name in keep:
            (source / name).write_text(name, encoding="utf-8")
        (source / "F.MPS.SCORE_PHI_1.0").write_text("old", encoding="utf-8")

        copied = _copy_mps_store_for_tag(source, target, "GS")

        assert copied == len(keep)
        assert sorted(path.name for path in target.iterdir()) == sorted(keep)
    finally:
        shutil.rmtree(root, ignore_errors=True)
