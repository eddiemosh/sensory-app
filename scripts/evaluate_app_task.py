"""Evaluate the app's own task on ESC-50: detection among everything else.

Usage: python scripts/evaluate_app_task.py [path/to/ESC-50]

The 5-class comparison in evaluate_esc50.py measures classification of whole
clips from a closed set. The app does something else: it scores single frames,
counts frames that pass a rule, and must stay quiet for sounds it does not
care about. This script measures that task. Five ESC-50 classes are targets
(APP_TARGETS); the other 45 should trigger nothing.

Arms, all scored with the app's rule (a category triggers when it passes in at
least 4 frames) and ESC-50's five folds:
  A          YAMNet labels, the app's default thresholds (score 0.2, 4 frames)
  A-tuned    YAMNet labels, thresholds chosen on four folds and tested on the fifth
  B-closed   a frame classifier trained on the target sounds only, so every frame
             is forced into one of them (tests the closed-set concern in D3)
  B-open     the same with an extra "other" class trained on the 45 other classes;
             this is the detector the app ships (sensoryaware.detector)

B arms count a frame when the detector's probability for a category is at least
0.5, the shipped rule, and use the shipped training code.

Loud speech is left out: ESC-50 has no speech class, and its recording levels
say nothing about a user's microphone. Per-frame features are cached in
data/esc50_frames.npz; delete it to extract them again.
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sensoryaware.audio import frame_levels_db, load_audio  # noqa: E402
from sensoryaware.config import DEFAULT_CONFIG  # noqa: E402
from sensoryaware.decision import decide  # noqa: E402
from sensoryaware.detector import (  # noqa: E402
    DETECTION_THRESHOLD,
    DETECTOR_CATEGORIES,
    OTHER,
    FrameDetector,
    fit_detector,
    training_frames,
)
from sensoryaware.evaluation import (  # noqa: E402
    APP_TARGETS,
    ESC50_CATEGORIES,
    category_frame_scores,
    detection_summary,
    load_metadata,
    triggered_categories,
)
from sensoryaware.predictions import FramePredictions  # noqa: E402

CACHE = ROOT / "data" / "esc50_frames.npz"
RESULTS = ROOT / "docs" / "evaluation" / "app-task-results.json"
MIN_FRAMES = DEFAULT_CONFIG.min_frames
SCORE_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
FRAME_GRID = tuple(range(1, 9))

# Categories scored: the app's, minus loud speech (see the docstring).
CONFIG = dataclasses.replace(
    DEFAULT_CONFIG,
    categories=tuple(c for c in DEFAULT_CONFIG.categories if c.min_level_db is None),
)
CATEGORY_IDS = [c.id for c in CONFIG.categories]


def extract(dataset):
    from sensoryaware.model import YamnetClassifier

    clips = load_metadata(dataset / "meta" / "esc50.csv")
    classifier = YamnetClassifier()
    scores, embeddings, levels, clip_index = [], [], [], []
    for n, clip in enumerate(clips):
        waveform = load_audio(dataset / "audio" / clip.filename)
        predictions, frame_embeddings = classifier.analyse(waveform)
        scores.append(predictions.scores)
        embeddings.append(frame_embeddings)
        levels.append(frame_levels_db(waveform, predictions.num_frames))
        clip_index.append(np.full(predictions.num_frames, n))
        if (n + 1) % 400 == 0:
            print(f"  {n + 1}/{len(clips)} clips")
    data = {
        "class_names": np.array(classifier.class_names),
        "scores": np.concatenate(scores).astype(np.float32),
        "embeddings": np.concatenate(embeddings).astype(np.float32),
        "levels": np.concatenate(levels).astype(np.float32),
        "clip_index": np.concatenate(clip_index),
        "targets": np.array([c.target for c in clips]),
        "folds": np.array([c.fold for c in clips]),
    }
    np.savez_compressed(CACHE, **data)
    return data


def expected_categories(targets):
    """Per clip: index into CATEGORY_IDS of the category it should trigger, or -1."""
    lookup = {ESC50_CATEGORIES.index(k): CATEGORY_IDS.index(v) for k, v in APP_TARGETS.items()}
    return np.array([lookup.get(int(t), -1) for t in targets])


def summarise_by_fold(triggered, expected, folds):
    overall = detection_summary(triggered, expected, CATEGORY_IDS)
    per_fold = [
        detection_summary(triggered[folds == f], expected[folds == f], CATEGORY_IDS)
        for f in np.unique(folds)
    ]
    for key in ("mean_detection_rate", "false_alarm_rate", "wrong_category_rate"):
        values = [s[key] for s in per_fold]
        overall[f"{key}_fold_std"] = float(np.std(values))
    return overall


def youden(summary):
    """Detection minus false alarms: rewards finding targets and staying quiet."""
    return summary["mean_detection_rate"] - summary["false_alarm_rate"]


def arm_labels(per_clip_scores, expected, folds):
    default = np.array(
        [triggered_categories(s, CONFIG.min_score, MIN_FRAMES) for s in per_clip_scores]
    )
    tuned = np.zeros_like(default)
    chosen = {}
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        best = None
        for t in SCORE_GRID:
            for k in FRAME_GRID:
                trig = np.array(
                    [triggered_categories(s, t, k) for s, keep in zip(per_clip_scores, train) if keep]
                )
                score = youden(detection_summary(trig, expected[train], CATEGORY_IDS))
                if best is None or score > best[0]:
                    best = (score, t, k)
        _, t, k = best
        chosen[int(f)] = {"min_score": t, "min_frames": k}
        for i in np.flatnonzero(test):
            tuned[i] = triggered_categories(per_clip_scores[i], t, k)
    return default, tuned, chosen


def arm_trained(data, expected, folds, open_set):
    """Train on four folds, test on the fifth, with the shipped detector's code and rule."""
    clip_categories = [CATEGORY_IDS[e] if e >= 0 else None for e in expected]
    class_names = DETECTOR_CATEGORIES + ((OTHER,) if open_set else ())
    columns = [CATEGORY_IDS.index(c) for c in DETECTOR_CATEGORIES]
    clip_of_frame = data["clip_index"]
    triggered = np.zeros((len(expected), len(CATEGORY_IDS)), dtype=bool)
    for f in np.unique(folds):
        x, y = training_frames(
            data["embeddings"], data["levels"], clip_of_frame, clip_categories,
            np.flatnonzero(folds != f), open_set=open_set,
        )
        detector = FrameDetector.from_pipeline(fit_detector(x, y), class_names, {})
        for c in np.flatnonzero(folds == f):
            probs = detector.probabilities(data["embeddings"][clip_of_frame == c])
            scores = np.zeros((len(probs), len(CATEGORY_IDS)))
            for category, col in zip(DETECTOR_CATEGORIES, columns):
                scores[:, col] = probs[:, detector.classes.index(category)]
            triggered[c] = triggered_categories(scores, DETECTION_THRESHOLD, MIN_FRAMES)
    return triggered


def main():
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "ESC-50"
    if CACHE.exists():
        print(f"Using cached frames from {CACHE}")
        data = dict(np.load(CACHE))
    else:
        print("Extracting per-frame YAMNet features (2,000 clips)...")
        data = extract(dataset)

    class_names = tuple(str(n) for n in data["class_names"])
    expected = expected_categories(data["targets"])
    folds = data["folds"]
    per_clip = [
        category_frame_scores(data["scores"][data["clip_index"] == c], class_names, CONFIG.categories)
        for c in range(len(expected))
    ]

    # Check: the vectorised rule gives exactly the app's decisions on every clip.
    for c in range(len(expected)):
        frames = data["clip_index"] == c
        app = decide(FramePredictions(class_names, data["scores"][frames]), CONFIG)
        mine = triggered_categories(per_clip[c], CONFIG.min_score, MIN_FRAMES)
        assert [e.triggered for e in app.evidence] == mine.tolist(), f"rule mismatch on clip {c}"
    print("Rule check: matches the app's decide() on all 2,000 clips.")

    start = time.perf_counter()
    default, tuned, chosen = arm_labels(per_clip, expected, folds)
    print(f"Label arms done in {time.perf_counter() - start:.0f} s")
    start = time.perf_counter()
    closed = arm_trained(data, expected, folds, open_set=False)
    opened = arm_trained(data, expected, folds, open_set=True)
    print(f"Trained arms done in {time.perf_counter() - start:.0f} s")

    arms = {
        "A labels, default thresholds": summarise_by_fold(default, expected, folds),
        "A labels, tuned thresholds": summarise_by_fold(tuned, expected, folds),
        "B-closed trained, targets only": summarise_by_fold(closed, expected, folds),
        "B-open trained, targets + other": summarise_by_fold(opened, expected, folds),
    }
    arms["A labels, tuned thresholds"]["chosen_per_test_fold"] = chosen

    results = {
        "task": "detect 5 ESC-50 target classes (mapped to app categories) among all 50; "
        "a category triggers when it passes in at least the minimum frames",
        "targets": APP_TARGETS,
        "categories_scored": CATEGORY_IDS,
        "frames": int(len(data["clip_index"])),
        "training_quiet_db": 20.0,
        "arms": arms,
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'Arm':34} {'Detect':>7} {'FalseAl':>8} {'WrongCat':>9}")
    for name, s in arms.items():
        print(
            f"{name:34} {s['mean_detection_rate']:7.3f} {s['false_alarm_rate']:8.3f} "
            f"{s['wrong_category_rate']:9.3f}"
        )
    print("\nDetection rate by category:")
    for name, s in arms.items():
        print(f"  {name:34} " + "  ".join(f"{k} {v:.2f}" for k, v in s["detection_rate"].items()))
    print("\nFalse alarms (of 1,800 non-target clips) by category:")
    for name, s in arms.items():
        print(f"  {name:34} " + "  ".join(f"{k} {v}" for k, v in s["false_alarms_by_category"].items()))
    print(f"\nTuned thresholds per test fold: {chosen}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
