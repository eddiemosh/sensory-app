"""Checks against the real YAMNet model (FR3, NFR3). Excluded from CI by its marker."""

import numpy as np
import pytest

from sensoryaware.config import DEFAULT_CONFIG
from sensoryaware.model import YamnetClassifier

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def classifier():
    return YamnetClassifier()


def test_model_has_521_classes(classifier):
    assert len(classifier.class_names) == 521


def test_silence_is_recognised(classifier):
    predictions = classifier.classify(np.zeros(16000 * 5, dtype=np.float32))
    assert predictions.top_k(1)[0].label == "Silence"


def test_results_are_reproducible(classifier):
    noise = np.random.default_rng(0).normal(0, 0.1, 16000 * 3).astype(np.float32)
    first = classifier.classify(noise).scores
    second = classifier.classify(noise).scores
    assert np.array_equal(first, second)


def test_every_default_label_exists_in_yamnet(classifier):
    labels = {label for c in DEFAULT_CONFIG.categories for label in c.labels}
    assert labels <= set(classifier.class_names)
