"""Model output in a form that does not depend on TensorFlow."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedLabel:
    label: str
    score: float


@dataclass(frozen=True)
class FramePredictions:
    """Scores for every class in every analysis frame of a clip.

    `scores` has shape (frames, classes). The values are model confidence-like
    scores, not calibrated probabilities that a sound is present.
    """

    class_names: tuple[str, ...]
    scores: np.ndarray
    levels_db: np.ndarray | None = None  # loudness of each frame, if measured

    def __post_init__(self):
        if self.scores.ndim != 2 or self.scores.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected scores of shape (frames, {len(self.class_names)}), "
                f"got {self.scores.shape}"
            )
        if self.levels_db is not None and self.levels_db.shape != (self.num_frames,):
            raise ValueError(
                f"Expected one loudness level per frame ({self.num_frames}), "
                f"got {self.levels_db.shape}"
            )

    @property
    def num_frames(self) -> int:
        return self.scores.shape[0]

    def top_k(self, k: int = 5) -> list[RankedLabel]:
        """Return the k classes with the highest mean score across frames."""
        if self.num_frames == 0:
            return []
        mean_scores = self.scores.mean(axis=0)
        ranked = np.argsort(mean_scores)[::-1][:k]
        return [RankedLabel(self.class_names[i], float(mean_scores[i])) for i in ranked]
