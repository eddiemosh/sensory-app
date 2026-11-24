"""Train the frame detector the app ships, on all of ESC-50, and save it.

Usage: python scripts/train_frame_detector.py

Uses the per-frame features cached by scripts/evaluate_app_task.py (run that
first). The same code and settings were evaluated there with five-fold
cross-validation; this run trains once on all 2,000 clips for deployment.
Writes models/frame_detector.npz.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sensoryaware.detector import (  # noqa: E402
    DETECTOR_CATEGORIES,
    DETECTOR_PATH,
    OTHER,
    FrameDetector,
    fit_detector,
    training_frames,
)
from sensoryaware.evaluation import APP_TARGETS, ESC50_CATEGORIES  # noqa: E402

CACHE = ROOT / "data" / "esc50_frames.npz"


def main():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} not found: run scripts/evaluate_app_task.py first")
    data = dict(np.load(CACHE))
    clip_categories = [APP_TARGETS.get(ESC50_CATEGORIES[int(t)]) for t in data["targets"]]
    x, y = training_frames(
        data["embeddings"], data["levels"], data["clip_index"], clip_categories,
        range(len(clip_categories)), open_set=True,
    )
    pipeline = fit_detector(x, y)
    metadata = {
        "trained_on": "ESC-50, all 2,000 clips (Piczak, 2015)",
        "licence": "Derived from ESC-50, CC BY-NC 3.0: non-commercial use only",
        "targets": APP_TARGETS,
        "features": "YAMNet 1,024-value frame embeddings",
        "model": "StandardScaler + LogisticRegression(C=0.1, class_weight=balanced)",
        "frames": {name: int((y == i).sum()) for i, name in enumerate(DETECTOR_CATEGORIES + (OTHER,))},
        "trained": date.today().isoformat(),
    }
    detector = FrameDetector.from_pipeline(pipeline, DETECTOR_CATEGORIES + (OTHER,), metadata)

    # The exported NumPy detector must give scikit-learn's probabilities.
    sample = x[:: max(1, len(x) // 2000)]
    if not np.allclose(detector.probabilities(sample), pipeline.predict_proba(sample), atol=1e-6):
        raise SystemExit("Exported detector does not match the trained model")

    detector.save(DETECTOR_PATH)
    print(f"Trained on {len(y)} frames: {metadata['frames']}")
    print(f"Wrote {DETECTOR_PATH} ({DETECTOR_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
