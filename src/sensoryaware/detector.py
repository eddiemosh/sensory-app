"""A frame detector trained on YAMNet embeddings (FR13, decision D3).

It scores each frame for four app categories plus "other", the class that
stops it firing on sounds it was not trained to detect. The shipped detector is
trained by scripts/train_frame_detector.py and evaluated, with this same code
and rule, by scripts/evaluate_app_task.py.

The app uses it through the existing rule engine: its probabilities are added
to the frame scores under their own label names, and the four categories it
covers switch to those labels with their own threshold. Chewing and loud speech
have no training data, so they keep YAMNet's labels.

At run time this is plain NumPy; scikit-learn is only needed to train it. The
file is a NumPy archive without pickled objects, so loading it runs no code.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensoryaware.decision import EngineConfig
from sensoryaware.predictions import FramePredictions

DETECTOR_CATEGORIES = ("clicking", "typing", "banging", "alarms")
OTHER = "other"
DETECTOR_PATH = Path(__file__).resolve().parents[2] / "models" / "frame_detector.npz"
DETECTION_THRESHOLD = 0.5
TRAINING_QUIET_DB = 20.0  # target-clip frames this far below the loudest are dropped


def detector_label(category_id: str) -> str:
    return f"Trained detector: {category_id}"


def frame_training_labels(levels, category, other: int, quiet_db=TRAINING_QUIET_DB):
    """Frame labels for training from clip labels.

    A target clip's sound is in its loud frames; its quiet frames (often padding)
    are dropped (-1) rather than mislabelled. Every frame of a non-target clip is
    "other".
    """
    if category is None:
        return np.full(len(levels), other)
    loud = (levels >= levels.max() - quiet_db) & (levels > -100.0)
    return np.where(loud, category, -1)


def training_frames(embeddings, levels, clip_index, clip_categories, clips, open_set=True):
    """Features and labels for the given clips. Labels index DETECTOR_CATEGORIES + other."""
    other = len(DETECTOR_CATEGORIES)
    xs, ys = [], []
    for clip in clips:
        frames = clip_index == clip
        category = clip_categories[clip]
        if category is None and not open_set:
            continue
        index = None if category is None else DETECTOR_CATEGORIES.index(category)
        labels = frame_training_labels(levels[frames], index, other)
        keep = labels >= 0
        xs.append(embeddings[frames][keep])
        ys.append(labels[keep])
    return np.concatenate(xs), np.concatenate(ys)


def fit_detector(features, labels):
    """Standardise, then multinomial logistic regression. Settings fixed in advance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
    )
    return model.fit(features, labels)


@dataclass(frozen=True)
class FrameDetector:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    metadata: dict

    @classmethod
    def from_pipeline(cls, pipeline, class_names, metadata):
        scaler, model = pipeline[0], pipeline[-1]
        if len(model.classes_) != len(class_names):
            raise ValueError("Every class must appear in the training labels")
        return cls(
            tuple(class_names[int(i)] for i in model.classes_),
            scaler.mean_.copy(),
            scaler.scale_.copy(),
            model.coef_.copy(),
            model.intercept_.copy(),
            dict(metadata),
        )

    def probabilities(self, embeddings) -> np.ndarray:
        """Softmax over classes for each frame (frames x classes)."""
        logits = ((embeddings - self.mean) / self.scale) @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            classes=np.array(self.classes),
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
            metadata=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path=DETECTOR_PATH):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                tuple(str(c) for c in data["classes"]),
                data["mean"],
                data["scale"],
                data["coef"],
                data["intercept"],
                json.loads(str(data["metadata"])),
            )


def add_detector_scores(predictions, detector, embeddings) -> FramePredictions:
    """Append the detector's per-frame probabilities as extra named columns."""
    probs = detector.probabilities(embeddings)
    extra = np.stack([probs[:, detector.classes.index(c)] for c in DETECTOR_CATEGORIES], axis=1)
    return FramePredictions(
        predictions.class_names + tuple(detector_label(c) for c in DETECTOR_CATEGORIES),
        np.concatenate([predictions.scores, extra], axis=1),
        predictions.levels_db,
    )


def use_trained_detector(config: EngineConfig, threshold=DETECTION_THRESHOLD) -> EngineConfig:
    """Switch the four covered categories to the detector's labels and threshold."""
    categories = tuple(
        dataclasses.replace(c, labels=(detector_label(c.id),), min_score=threshold)
        if c.id in DETECTOR_CATEGORIES
        else c
        for c in config.categories
    )
    return dataclasses.replace(config, categories=categories)
