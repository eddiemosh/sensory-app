"""Ranking of model output (FR4)."""

import numpy as np
import pytest

from sensoryaware.predictions import FramePredictions

NAMES = ("A", "B", "C")


def test_top_k_ranks_by_mean_score_across_frames():
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.7, 0.1]], dtype=np.float32)
    top = FramePredictions(NAMES, scores).top_k(2)
    assert [item.label for item in top] == ["B", "A"]
    assert top[0].score == pytest.approx(0.6)


def test_top_k_is_limited_by_number_of_classes():
    assert len(FramePredictions(NAMES, np.zeros((1, 3))).top_k(5)) == 3


def test_top_k_with_no_frames_is_empty():
    assert FramePredictions(NAMES, np.zeros((0, 3))).top_k(5) == []


def test_mismatched_scores_are_rejected():
    with pytest.raises(ValueError):
        FramePredictions(NAMES, np.zeros((2, 4)))
