"""Pretrained YAMNet inference.

TensorFlow is imported inside the class so that the rest of the package, and
its tests, can run without loading it.
"""

import csv
import os

import numpy as np

from sensoryaware.predictions import FramePredictions

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


class YamnetClassifier:
    """Downloads YAMNet on first use; TensorFlow Hub caches it afterwards."""

    def __init__(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        import tensorflow_hub as hub

        self._model = hub.load(YAMNET_URL)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as class_map:
            self.class_names = tuple(
                row["display_name"] for row in csv.DictReader(class_map)
            )

    def classify(self, waveform: np.ndarray) -> FramePredictions:
        """Score 16 kHz mono audio: one frame per 0.48 s, each covering 0.96 s."""
        scores, _embeddings, _spectrogram = self._model(waveform)
        return FramePredictions(self.class_names, scores.numpy())
