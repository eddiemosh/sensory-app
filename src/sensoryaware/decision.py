"""Transparent rules that turn frame predictions into a recommendation.

A category triggers when any of its YAMNet labels reaches `min_score` in at
least `min_frames` frames of the clip. YAMNet frames overlap (0.96 s long, every
0.48 s), so one short sound usually appears in two or three frames; `min_frames`
has to be above that for "repeated" to mean anything.
"""

from dataclasses import dataclass

import numpy as np

from sensoryaware.predictions import FramePredictions


@dataclass(frozen=True)
class Intervention:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class SoundCategory:
    id: str
    name: str
    labels: tuple[str, ...]
    intervention_id: str


@dataclass(frozen=True)
class EngineConfig:
    categories: tuple[SoundCategory, ...]
    interventions: tuple[Intervention, ...]
    multi_category_intervention: str
    min_score: float
    min_frames: int

    def __post_init__(self):
        if not 0 < self.min_score <= 1:
            raise ValueError("min_score must be above 0 and at most 1")
        if self.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        known = {intervention.id for intervention in self.interventions}
        used = {category.intervention_id for category in self.categories}
        used.add(self.multi_category_intervention)
        if used - known:
            raise ValueError(f"Unknown interventions: {sorted(used - known)}")

    def intervention(self, intervention_id: str) -> Intervention:
        return next(i for i in self.interventions if i.id == intervention_id)


@dataclass(frozen=True)
class CategoryEvidence:
    category: str
    frames_detected: int
    total_frames: int
    peak_score: float
    peak_label: str | None
    triggered: bool


@dataclass(frozen=True)
class Decision:
    intervention: Intervention | None
    triggered_categories: tuple[str, ...]
    evidence: tuple[CategoryEvidence, ...]
    explanation: str
    warnings: tuple[str, ...]


def _evaluate_category(
    category: SoundCategory,
    predictions: FramePredictions,
    config: EngineConfig,
) -> CategoryEvidence:
    indices = {name: i for i, name in enumerate(predictions.class_names)}
    known_labels = [label for label in category.labels if label in indices]
    if not known_labels or predictions.num_frames == 0:
        return CategoryEvidence(
            category.name, 0, predictions.num_frames, 0.0, None, False
        )

    scores = predictions.scores[:, [indices[label] for label in known_labels]]
    frames_detected = int((scores.max(axis=1) >= config.min_score).sum())
    _, peak_column = np.unravel_index(scores.argmax(), scores.shape)
    return CategoryEvidence(
        category=category.name,
        frames_detected=frames_detected,
        total_frames=predictions.num_frames,
        peak_score=float(scores.max()),
        peak_label=known_labels[peak_column],
        triggered=frames_detected >= config.min_frames,
    )


def _rule_text(config: EngineConfig) -> str:
    return f"needs at least {config.min_frames} at score {config.min_score:.2f} or above"


def _frames_text(item: CategoryEvidence) -> str:
    return f"{item.frames_detected} of {item.total_frames} analysis frames"


def _explain_no_trigger(
    evidence: tuple[CategoryEvidence, ...],
    predictions: FramePredictions,
    config: EngineConfig,
) -> str:
    if predictions.num_frames < config.min_frames:
        return (
            f"This clip is too short to apply the rule: it has "
            f"{predictions.num_frames} analysis frames and a sound "
            f"{_rule_text(config)}. No intervention recommended."
        )
    closest = max(evidence, key=lambda item: item.frames_detected)
    if closest.frames_detected == 0:
        return (
            "No configured distracting category was detected in this clip. "
            "No intervention recommended."
        )
    return (
        f"No configured category met the rule in this clip. The closest was "
        f"{closest.category}, in {_frames_text(closest)} ({_rule_text(config)}). "
        f"No intervention recommended."
    )


def decide(predictions: FramePredictions, config: EngineConfig) -> Decision:
    """Apply the configured rules to one clip's predictions."""
    known = set(predictions.class_names)
    warnings = tuple(
        f"'{label}' in category {category.name} is not a YAMNet label and was ignored."
        for category in config.categories
        for label in category.labels
        if label not in known
    )

    if not config.categories:
        return Decision(
            None,
            (),
            (),
            "No sound categories are configured as distracting, so no "
            "intervention was considered.",
            warnings,
        )

    evidence = tuple(
        _evaluate_category(category, predictions, config)
        for category in config.categories
    )
    triggered = [
        (category, item)
        for category, item in zip(config.categories, evidence)
        if item.triggered
    ]
    names = tuple(item.category for _, item in triggered)

    if not triggered:
        explanation = _explain_no_trigger(evidence, predictions, config)
        return Decision(None, (), evidence, explanation, warnings)

    if len(triggered) == 1:
        category, item = triggered[0]
        intervention = config.intervention(category.intervention_id)
        explanation = (
            f"{item.category} was detected in {_frames_text(item)} in this clip "
            f"({_rule_text(config)}; peak {item.peak_score:.2f} for "
            f"'{item.peak_label}'). Recommended: {intervention.name}."
        )
    else:
        intervention = config.intervention(config.multi_category_intervention)
        details = ", ".join(
            f"{item.category} ({_frames_text(item)})" for _, item in triggered
        )
        explanation = (
            f"{len(triggered)} configured categories met the rule in this clip: "
            f"{details}. Recommended: {intervention.name}."
        )

    return Decision(intervention, names, evidence, explanation, warnings)
