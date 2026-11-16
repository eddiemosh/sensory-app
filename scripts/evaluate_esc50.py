"""Evaluate both ML approaches on ESC-50 and write docs/evaluation/results.json.

Usage: python scripts/evaluate_esc50.py [path/to/ESC-50]

The ESC-50 folder must contain meta/esc50.csv and audio/*.wav. YAMNet features
are extracted once and cached in data/esc50_features.npz; delete that file to
extract them again.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sensoryaware.audio import load_audio  # noqa: E402
from sensoryaware.evaluation import (  # noqa: E402
    APP_RELEVANT,
    ESC50_CATEGORIES,
    ESC50_TO_YAMNET,
    cross_validate,
    load_metadata,
    predict_mapped,
    summarise,
)

CACHE = ROOT / "data" / "esc50_features.npz"
RESULTS = ROOT / "docs" / "evaluation" / "results.json"

CLASSIFIERS = {
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)
    ),
    "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=10)),
    "knn_5": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
}


def extract_features(dataset: Path) -> dict:
    from sensoryaware.model import YamnetClassifier

    clips = load_metadata(dataset / "meta" / "esc50.csv")
    for clip in clips:
        if ESC50_CATEGORIES[clip.target] != clip.category:
            raise SystemExit(f"Category order mismatch at {clip.filename}")

    classifier = YamnetClassifier()
    scores, embeddings, seconds = [], [], []
    for number, clip in enumerate(clips, 1):
        waveform = load_audio(dataset / "audio" / clip.filename)
        start = time.perf_counter()
        predictions, frame_embeddings = classifier.analyse(waveform)
        seconds.append(time.perf_counter() - start)
        scores.append(predictions.scores.mean(axis=0))
        embeddings.append(frame_embeddings.mean(axis=0))
        if number % 200 == 0:
            print(f"  {number}/{len(clips)} clips")

    features = {
        "class_names": np.array(classifier.class_names),
        "scores": np.stack(scores),
        "embeddings": np.stack(embeddings),
        "targets": np.array([clip.target for clip in clips]),
        "folds": np.array([clip.fold for clip in clips]),
        "inference_seconds": np.array(seconds),
    }
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **features)
    return features


def top_confusions(targets, predicted, limit=8):
    pairs = Counter(
        (ESC50_CATEGORIES[t], ESC50_CATEGORIES[p])
        for t, p in zip(targets, predicted)
        if t != p
    )
    return [
        {"true": true, "predicted": guess, "clips": count}
        for (true, guess), count in pairs.most_common(limit)
    ]


def main():
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "ESC-50"
    if CACHE.exists():
        print(f"Using cached features from {CACHE}")
        features = dict(np.load(CACHE))
    else:
        print("Extracting YAMNet features (one pass over 2,000 clips)...")
        features = extract_features(dataset)

    targets, folds = features["targets"], features["folds"]
    class_names = tuple(str(name) for name in features["class_names"])
    arms = {}

    predicted = predict_mapped(
        features["scores"], class_names, ESC50_TO_YAMNET, ESC50_CATEGORIES
    )
    arms["A_mapped_labels"] = summarise(targets, predicted, folds, ESC50_CATEGORIES)
    arms["A_mapped_labels"]["top_confusions"] = top_confusions(targets, predicted)

    for name, make_model in CLASSIFIERS.items():
        start = time.perf_counter()
        result = cross_validate(features["embeddings"], targets, folds, make_model)
        arm = summarise(targets, result.predictions, folds, ESC50_CATEGORIES)
        arm["train_and_predict_seconds"] = time.perf_counter() - start
        arm["top_confusions"] = top_confusions(targets, result.predictions)
        arms[f"B_embeddings_{name}"] = arm

    seconds = features["inference_seconds"]
    results = {
        "dataset": "ESC-50, 2,000 clips, 50 classes, 5 predefined folds",
        "clip_features": "mean over YAMNet frames (scores: 521, embeddings: 1,024)",
        "yamnet_seconds_per_clip": {
            "mean": float(seconds.mean()),
            "p95": float(np.percentile(seconds, 95)),
        },
        "app_relevant_classes": list(APP_RELEVANT),
        "arms": arms,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'Arm':42} {'Accuracy':>9} {'Fold std':>9} {'Macro F1':>9}")
    for name, arm in arms.items():
        print(
            f"{name:42} {arm['accuracy']:9.3f} {arm['fold_std']:9.3f} {arm['macro_f1']:9.3f}"
        )
    print("\nF1 on app-relevant classes:")
    print(f"{'Class':18}" + "".join(f"{name[:14]:>16}" for name in arms))
    for category in APP_RELEVANT:
        print(
            f"{category:18}"
            + "".join(f"{arm['per_class_f1'][category]:16.3f}" for arm in arms.values())
        )
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
