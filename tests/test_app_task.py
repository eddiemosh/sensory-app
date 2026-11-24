"""Helpers for evaluating the app's own task on ESC-50 (ML investigation).

The app's task is detection: decide whether a clip contains a recurring
distracting sound, and which, among everything else a room might hold.
"""

import numpy as np
import pytest

from sensoryaware.decision import EngineConfig, Intervention, SoundCategory, decide
from sensoryaware.evaluation import (
    APP_TARGETS,
    ESC50_CATEGORIES,
    category_frame_scores,
    detection_summary,
    frame_training_labels,
    triggered_categories,
)
from sensoryaware.predictions import FramePredictions

NAMES = ("Clicking", "Tick", "Typing", "Speech")
CATS = (
    SoundCategory("clicking", "Clicking", ("Clicking", "Tick"), "g"),
    SoundCategory("typing", "Typing", ("Typing", "Missing label"), "g"),
)


def test_app_targets_are_esc50_classes_mapped_to_app_categories():
    assert set(APP_TARGETS) <= set(ESC50_CATEGORIES)
    assert set(APP_TARGETS.values()) == {"clicking", "typing", "banging", "alarms"}


def test_category_score_is_best_known_label_per_frame():
    scores = np.array([[0.1, 0.7, 0.2, 0.9], [0.3, 0.0, 0.6, 0.0]])
    result = category_frame_scores(scores, NAMES, CATS)
    assert result.tolist() == [[0.7, 0.2], [0.3, 0.6]]


def test_triggered_needs_threshold_in_min_frames():
    frame_scores = np.array([[0.5, 0.1], [0.5, 0.3], [0.1, 0.3], [0.5, 0.3]])
    assert triggered_categories(frame_scores, 0.3, 3).tolist() == [True, True]
    assert triggered_categories(frame_scores, 0.4, 3).tolist() == [True, False]
    assert triggered_categories(frame_scores, 0.3, 4).tolist() == [False, False]


def test_vectorised_rule_matches_the_app_rule_engine():
    rng = np.random.default_rng(1)
    config = EngineConfig(CATS, (Intervention("g", "G", "g"),), "g", 0.4, 3)
    for _ in range(50):
        scores = rng.random((8, len(NAMES))) * rng.random()
        app = decide(FramePredictions(NAMES, scores), config)
        mine = triggered_categories(category_frame_scores(scores, NAMES, CATS), 0.4, 3)
        assert [e.triggered for e in app.evidence] == mine.tolist()


def test_detection_summary_counts_hits_misses_and_false_alarms():
    # Columns: clicking, typing. Expected: 0 = clicking, 1 = typing, -1 = no target.
    triggered = np.array(
        [
            [True, False],   # clicking clip, correct
            [False, False],  # clicking clip, missed
            [False, True],   # typing clip, correct
            [True, True],    # typing clip, correct but clicking also fired
            [False, False],  # other clip, quiet: correct
            [False, True],   # other clip, false alarm
        ]
    )
    expected = np.array([0, 0, 1, 1, -1, -1])
    s = detection_summary(triggered, expected, ["clicking", "typing"])
    assert s["detection_rate"] == {"clicking": 0.5, "typing": 1.0}
    assert s["mean_detection_rate"] == pytest.approx(0.75)
    assert s["false_alarm_rate"] == pytest.approx(0.5)
    assert s["false_alarms_by_category"] == {"clicking": 0, "typing": 1}
    assert s["wrong_category_rate"] == pytest.approx(0.25)
    assert s["target_clips"] == 4 and s["other_clips"] == 2


def test_training_labels_use_loud_frames_of_target_clips_only():
    levels = np.array([-10.0, -12.0, -45.0, -100.0])
    # Target clip: frames within 20 dB of its loudest get the category; the rest are dropped.
    assert frame_training_labels(levels, 2, other=4).tolist() == [2, 2, -1, -1]
    # Non-target clip: every frame is "other".
    assert frame_training_labels(levels, None, other=4).tolist() == [4, 4, 4, 4]
