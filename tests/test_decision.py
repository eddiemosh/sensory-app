"""Decision engine rules (FR6-FR11). Uses hand-written scores, so no model is needed."""

import numpy as np
import pytest

from sensoryaware.decision import EngineConfig, Intervention, SoundCategory, decide
from sensoryaware.predictions import FramePredictions

CLASS_NAMES = ("Clicking", "Tick", "Bang", "Speech", "Silence")

GENTLE = Intervention("gentle", "Gentle instrumental", "Soft music")
MASKING = Intervention("masking", "Stronger masking", "Broadband masking")
FOCUS = Intervention("focus", "General focus", "Focus audio")

CLICKING = SoundCategory("clicking", "Clicking", ("Clicking", "Tick"), "gentle")
BANGING = SoundCategory("banging", "Banging", ("Bang",), "masking")


def config(**overrides):
    settings = dict(
        categories=(CLICKING, BANGING),
        interventions=(GENTLE, MASKING, FOCUS),
        multi_category_intervention="focus",
        min_score=0.5,
        min_frames=3,
    )
    settings.update(overrides)
    return EngineConfig(**settings)


def predictions(num_frames=10, **frames_by_label):
    """Build scores where each label scores 0.9 in the listed frame indices."""
    scores = np.zeros((num_frames, len(CLASS_NAMES)), dtype=np.float32)
    for label, frames in frames_by_label.items():
        scores[list(frames), CLASS_NAMES.index(label)] = 0.9
    return FramePredictions(CLASS_NAMES, scores)


def test_repeated_category_triggers_its_intervention():
    decision = decide(predictions(Clicking=[1, 3, 5]), config())
    assert decision.intervention == GENTLE
    assert decision.triggered_categories == ("Clicking",)


def test_different_category_gets_different_intervention():
    decision = decide(predictions(Bang=[0, 4, 8, 9]), config())
    assert decision.intervention == MASKING


def test_isolated_detection_below_min_frames_is_ignored():
    decision = decide(predictions(Clicking=[2, 3]), config())
    assert decision.intervention is None
    assert decision.triggered_categories == ()


def test_score_exactly_at_threshold_counts():
    scores = np.zeros((3, len(CLASS_NAMES)), dtype=np.float32)
    scores[:, CLASS_NAMES.index("Clicking")] = 0.5
    decision = decide(FramePredictions(CLASS_NAMES, scores), config())
    assert decision.intervention == GENTLE


def test_score_below_threshold_does_not_trigger():
    scores = np.zeros((10, len(CLASS_NAMES)), dtype=np.float32)
    scores[:, CLASS_NAMES.index("Clicking")] = 0.49
    decision = decide(FramePredictions(CLASS_NAMES, scores), config())
    assert decision.intervention is None


def test_frames_are_pooled_across_labels_in_a_category_without_double_counting():
    decision = decide(predictions(Clicking=[1, 2], Tick=[2, 3]), config())
    assert decision.intervention == GENTLE
    assert decision.evidence[0].frames_detected == 3


def test_several_triggered_categories_recommend_general_focus():
    decision = decide(predictions(Clicking=[1, 2, 3], Bang=[5, 6, 7]), config())
    assert decision.intervention == FOCUS
    assert decision.triggered_categories == ("Clicking", "Banging")


def test_sound_outside_configured_categories_does_not_trigger():
    decision = decide(predictions(Speech=list(range(10))), config())
    assert decision.intervention is None


def test_unknown_label_is_reported_and_ignored():
    category = SoundCategory("odd", "Odd", ("Clicking", "Not a YAMNet label"), "gentle")
    decision = decide(predictions(Clicking=[1, 2, 3]), config(categories=(category,)))
    assert decision.intervention == GENTLE
    assert any("Not a YAMNet label" in warning for warning in decision.warnings)


def test_category_with_only_unknown_labels_does_not_trigger():
    category = SoundCategory("odd", "Odd", ("Not a YAMNet label",), "gentle")
    decision = decide(predictions(Clicking=[1, 2, 3]), config(categories=(category,)))
    assert decision.intervention is None
    assert decision.evidence[0].frames_detected == 0


def test_no_configured_categories_is_safe():
    decision = decide(predictions(Clicking=[1, 2, 3]), config(categories=()))
    assert decision.intervention is None
    assert "No sound categories" in decision.explanation


def test_clip_too_short_for_min_frames_is_explained():
    decision = decide(predictions(num_frames=2, Clicking=[0, 1]), config())
    assert decision.intervention is None
    assert "too short" in decision.explanation


def test_explanation_gives_evidence_for_a_trigger():
    explanation = decide(predictions(Clicking=[1, 3, 5]), config()).explanation
    assert "Clicking" in explanation
    assert "3 of 10" in explanation
    assert "Gentle instrumental" in explanation


def test_explanation_stays_within_the_clip():
    """NFR5: recurrence is only claimed within the analysed clip."""
    assert "in this clip" in decide(predictions(Clicking=[1, 3, 5]), config()).explanation


def test_evidence_covers_every_category_with_peak_score():
    evidence = decide(predictions(Clicking=[1]), config()).evidence
    assert [item.category for item in evidence] == ["Clicking", "Banging"]
    assert evidence[0].total_frames == 10
    assert evidence[0].peak_label == "Clicking"
    assert evidence[0].peak_score == pytest.approx(0.9)
    assert evidence[0].triggered is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_score": 0.0},
        {"min_score": 1.1},
        {"min_frames": 0},
        {"multi_category_intervention": "missing"},
        {"categories": (SoundCategory("x", "X", ("Bang",), "missing"),)},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


# --- Per-category thresholds and the loudness gate (FR12) --------------------

SPEECH = SoundCategory(
    "speech", "Loud speech", ("Speech",), "masking", min_score=0.95, min_level_db=-20.0
)


def with_levels(preds, levels):
    return FramePredictions(preds.class_names, preds.scores, np.array(levels, dtype=float))


def test_category_threshold_overrides_the_global_one():
    strict = SoundCategory("clicking", "Clicking", ("Clicking",), "gentle", min_score=0.95)
    decision = decide(predictions(Clicking=[1, 3, 5]), config(categories=(strict,)))
    assert decision.intervention is None
    assert decision.evidence[0].frames_detected == 0


def test_loud_speech_triggers_its_intervention():
    loud_speech = SoundCategory("speech", "Loud speech", ("Speech",), "masking", min_level_db=-20.0)
    preds = with_levels(predictions(Speech=[1, 2, 3, 4]), [-10.0] * 10)
    decision = decide(preds, config(categories=(loud_speech,)))
    assert decision.intervention == MASKING
    assert decision.evidence[0].frames_detected == 4


def test_quiet_speech_is_ignored():
    loud_speech = SoundCategory("speech", "Loud speech", ("Speech",), "masking", min_level_db=-20.0)
    preds = with_levels(predictions(Speech=[1, 2, 3, 4]), [-35.0] * 10)
    decision = decide(preds, config(categories=(loud_speech,)))
    assert decision.intervention is None
    assert decision.evidence[0].frames_detected == 0


def test_only_frames_that_are_both_confident_and_loud_count():
    loud_speech = SoundCategory("speech", "Loud speech", ("Speech",), "masking", min_level_db=-20.0)
    levels = [-10.0, -10.0, -40.0, -40.0, -10.0, -10.0, -40.0, -40.0, -40.0, -40.0]
    preds = with_levels(predictions(Speech=[0, 1, 2, 3]), levels)
    assert decide(preds, config(categories=(loud_speech,))).evidence[0].frames_detected == 2


def test_gated_category_without_levels_is_skipped_with_a_warning():
    loud_speech = SoundCategory("speech", "Loud speech", ("Speech",), "masking", min_level_db=-20.0)
    decision = decide(predictions(Speech=[1, 2, 3, 4]), config(categories=(loud_speech,)))
    assert decision.intervention is None
    assert any("loudness" in warning for warning in decision.warnings)


def test_explanation_states_the_category_rule_including_loudness():
    loud_speech = SoundCategory(
        "speech", "Loud speech", ("Speech",), "masking", min_score=0.6, min_level_db=-20.0
    )
    preds = with_levels(predictions(Speech=[1, 2, 3, 4]), [-10.0] * 10)
    explanation = decide(preds, config(categories=(loud_speech,))).explanation
    assert "score 0.60 or above" in explanation
    assert "-20 dBFS" in explanation


@pytest.mark.parametrize(
    "category",
    [
        SoundCategory("x", "X", ("Bang",), "masking", min_score=0.0),
        SoundCategory("x", "X", ("Bang",), "masking", min_score=1.5),
        SoundCategory("x", "X", ("Bang",), "masking", min_level_db=3.0),
    ],
)
def test_invalid_category_thresholds_are_rejected(category):
    with pytest.raises(ValueError):
        config(categories=(category,))
