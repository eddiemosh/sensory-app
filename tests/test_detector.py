"""The trained frame detector (FR13): export, loading, and use by the rule engine."""

from pathlib import Path

import numpy as np
import pytest

from sensoryaware.config import DEFAULT_CONFIG
from sensoryaware.decision import decide
from sensoryaware.detector import (
    DETECTOR_CATEGORIES,
    DETECTOR_PATH,
    OTHER,
    FrameDetector,
    add_detector_scores,
    detector_label,
    fit_detector,
    training_frames,
    use_trained_detector,
)
from sensoryaware.predictions import FramePredictions

CLASSES = ("clicking", "typing", "banging", "alarms", OTHER)


def fitted(seed=0, classes=CLASSES, dims=8):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(classes)), 40)
    features = rng.normal(size=(len(labels), dims)) + labels[:, None] * 2.0
    return fit_detector(features, labels), features


def test_exported_detector_matches_scikit_learn():
    pipeline, features = fitted()
    detector = FrameDetector.from_pipeline(pipeline, CLASSES, {"trained_on": "test"})
    assert detector.probabilities(features) == pytest.approx(
        pipeline.predict_proba(features), abs=1e-6
    )


def test_save_and_load_round_trip(tmp_path):
    pipeline, features = fitted()
    detector = FrameDetector.from_pipeline(pipeline, CLASSES, {"trained_on": "test"})
    path = tmp_path / "detector.npz"
    detector.save(path)
    loaded = FrameDetector.load(path)
    assert loaded.classes == CLASSES
    assert loaded.metadata == {"trained_on": "test"}
    assert loaded.probabilities(features) == pytest.approx(detector.probabilities(features))


def test_detector_scores_are_added_as_named_columns_in_category_order():
    shuffled = ("alarms", OTHER, "typing", "clicking", "banging")
    pipeline, features = fitted(classes=shuffled)
    detector = FrameDetector.from_pipeline(pipeline, shuffled, {})
    base = FramePredictions(("Speech", "Silence"), np.zeros((5, 2)), np.full(5, -30.0))
    combined = add_detector_scores(base, detector, features[:5])
    assert combined.class_names == ("Speech", "Silence") + tuple(
        detector_label(c) for c in DETECTOR_CATEGORIES
    )
    probs = detector.probabilities(features[:5])
    for i, category in enumerate(DETECTOR_CATEGORIES):
        assert combined.scores[:, 2 + i] == pytest.approx(probs[:, shuffled.index(category)])
    assert combined.levels_db is base.levels_db


def test_trained_mode_swaps_only_the_four_detector_categories():
    config = use_trained_detector(DEFAULT_CONFIG, 0.5)
    by_id = {c.id: c for c in config.categories}
    for category in DETECTOR_CATEGORIES:
        assert by_id[category].labels == (detector_label(category),)
        assert by_id[category].min_score == 0.5
    original = {c.id: c for c in DEFAULT_CONFIG.categories}
    assert by_id["chewing"] == original["chewing"]
    assert by_id["speech"] == original["speech"]


def test_rule_engine_triggers_on_confident_detector_frames():
    names = ("Speech",) + tuple(detector_label(c) for c in DETECTOR_CATEGORIES)
    scores = np.zeros((10, len(names)))
    scores[[0, 2, 4, 6], names.index(detector_label("clicking"))] = 0.8
    scores[[1, 3, 5, 7], names.index(detector_label("typing"))] = 0.4  # below 0.5
    decision = decide(
        FramePredictions(names, scores, np.full(10, -30.0)),
        use_trained_detector(DEFAULT_CONFIG, 0.5),
    )
    assert decision.triggered_categories == ("Clicking and ticking",)


def test_training_frames_drop_quiet_target_frames_and_skip_other_when_closed():
    embeddings = np.arange(12, dtype=float).reshape(6, 2)
    levels = np.array([-10.0, -40.0, -10.0, -12.0, -10.0, -11.0])
    clip_index = np.array([0, 0, 1, 1, 2, 2])
    categories = ["typing", None, "alarms"]
    x, y = training_frames(embeddings, levels, clip_index, categories, [0, 1, 2], open_set=True)
    assert y.tolist() == [1, 4, 4, 3, 3]  # clip 0's quiet frame dropped; clip 1 is "other"
    x, y = training_frames(embeddings, levels, clip_index, categories, [0, 1, 2], open_set=False)
    assert y.tolist() == [1, 3, 3]
    assert x.shape == (3, 2)


def test_shipped_detector_file_is_valid():
    detector = FrameDetector.load(DETECTOR_PATH)
    assert detector.classes == DETECTOR_CATEGORIES + (OTHER,)
    assert detector.coef.shape == (len(detector.classes), 1024)
    assert "ESC-50" in detector.metadata["trained_on"]
    assert "CC BY-NC" in detector.metadata["licence"]
    probs = detector.probabilities(np.zeros((3, 1024)))
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])


def test_detector_path_is_inside_the_repository():
    assert Path(DETECTOR_PATH).parent.name == "models"
