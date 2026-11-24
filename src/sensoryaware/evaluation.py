"""ESC-50 evaluation of the two approaches compared in the ML investigation.

Arm A uses YAMNet's own labels, mapped by hand to ESC-50's 50 classes.
Arm B uses YAMNet's 1,024-value embeddings as features for a classifier trained
on ESC-50, with the dataset's five predefined folds as cross-validation splits.

Nothing here imports TensorFlow: features are extracted once by
scripts/evaluate_esc50.py and passed in as arrays.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from sensoryaware.detector import frame_training_labels  # noqa: F401  (shared with the app)

# Hand-written mapping from each ESC-50 class to the YAMNet labels that
# describe it. No label is used for two classes, so a clip's prediction is never
# a tie decided by list order. Weak or indirect mappings are marked; they are a
# known limitation of Arm A and are discussed in docs/evaluation.md.
ESC50_TO_YAMNET: dict[str, tuple[str, ...]] = {
    "dog": ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling"),
    "rooster": ("Crowing, cock-a-doodle-doo",),
    "pig": ("Pig", "Oink"),
    "cow": ("Cattle, bovinae", "Moo"),
    "frog": ("Frog", "Croak"),
    "cat": ("Cat", "Meow", "Purr", "Caterwaul"),
    "hen": ("Chicken, rooster", "Cluck", "Fowl"),
    "insects": ("Insect", "Mosquito", "Fly, housefly", "Buzz", "Bee, wasp, etc."),
    "sheep": ("Sheep", "Bleat"),
    "crow": ("Crow", "Caw"),
    "rain": ("Rain", "Raindrop", "Rain on surface"),
    "sea_waves": ("Ocean", "Waves, surf"),
    "crackling_fire": ("Fire", "Crackle"),
    "crickets": ("Cricket",),
    "chirping_birds": (
        "Bird",
        "Bird vocalization, bird call, bird song",
        "Chirp, tweet",
    ),
    "water_drops": ("Drip", "Plop"),
    "wind": ("Wind", "Wind noise (microphone)", "Rustling leaves"),
    "pouring_water": (
        "Pour",
        "Trickle, dribble",
        "Fill (with liquid)",
        "Water tap, faucet",
        "Gush",
    ),
    "toilet_flush": ("Toilet flush",),
    "thunderstorm": ("Thunderstorm", "Thunder"),
    "crying_baby": ("Baby cry, infant cry", "Crying, sobbing"),
    "sneezing": ("Sneeze",),
    "clapping": ("Clapping", "Applause"),
    "breathing": ("Breathing", "Pant", "Gasp", "Wheeze"),
    "coughing": ("Cough", "Throat clearing"),
    "footsteps": ("Walk, footsteps", "Shuffle", "Run"),
    "laughing": (
        "Laughter",
        "Giggle",
        "Snicker",
        "Belly laugh",
        "Chuckle, chortle",
        "Baby laughter",
    ),
    "brushing_teeth": ("Toothbrush", "Electric toothbrush"),
    "snoring": ("Snoring", "Snort"),
    "drinking_sipping": ("Liquid", "Slosh"),  # weak: YAMNet has no drinking label
    "door_wood_knock": ("Knock", "Thump, thud"),
    "mouse_click": ("Clicking",),
    "keyboard_typing": ("Typing", "Computer keyboard", "Typewriter"),
    "door_wood_creaks": ("Creak", "Squeak"),
    "can_opening": ("Burst, pop",),  # weak: no can-opening label
    "washing_machine": ("Whir", "Vibration"),  # weak: no washing-machine label
    "vacuum_cleaner": ("Vacuum cleaner",),
    "clock_alarm": ("Alarm clock", "Alarm", "Beep, bleep"),
    "clock_tick": ("Tick", "Tick-tock", "Clock"),
    "glass_breaking": ("Shatter", "Glass", "Chink, clink", "Breaking"),
    "helicopter": ("Helicopter",),
    "chainsaw": ("Chainsaw",),
    "siren": (
        "Siren",
        "Civil defense siren",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Emergency vehicle",
    ),
    "car_horn": ("Vehicle horn, car horn, honking", "Toot", "Air horn, truck horn"),
    "engine": (
        "Engine",
        "Idling",
        "Accelerating, revving, vroom",
        "Light engine (high frequency)",
        "Medium engine (mid frequency)",
        "Heavy engine (low frequency)",
        "Engine starting",
    ),
    "train": (
        "Train",
        "Rail transport",
        "Railroad car, train wagon",
        "Train wheels squealing",
        "Subway, metro, underground",
        "Clickety-clack",
        "Train horn",
        "Train whistle",
    ),
    "church_bells": ("Church bell", "Bell", "Change ringing (campanology)"),
    "airplane": (
        "Fixed-wing aircraft, airplane",
        "Aircraft",
        "Aircraft engine",
        "Jet engine",
        "Propeller, airscrew",
    ),
    "fireworks": ("Fireworks", "Firecracker", "Explosion"),
    "hand_saw": ("Sawing", "Filing (rasp)"),
}

# ESC-50 numbers its classes 0-49 in this order (the dataset's "target" column).
ESC50_CATEGORIES: tuple[str, ...] = tuple(ESC50_TO_YAMNET)

# Classes closest to the sounds the app is built to notice.
APP_RELEVANT = (
    "mouse_click",
    "keyboard_typing",
    "clock_tick",
    "door_wood_knock",
    "clock_alarm",
)


# The app's detection task on ESC-50: these classes should trigger the named app
# category; the other 45 classes should trigger nothing. Chewing and speech have
# no ESC-50 class, so they cannot be scored for detection here.
APP_TARGETS = {
    "mouse_click": "clicking",
    "clock_tick": "clicking",
    "keyboard_typing": "typing",
    "door_wood_knock": "banging",
    "clock_alarm": "alarms",
}


def category_frame_scores(scores, class_names, categories) -> np.ndarray:
    """Per frame, each category's score = its best known label's score (frames x categories)."""
    index = {name: i for i, name in enumerate(class_names)}
    columns = []
    for category in categories:
        known = [index[label] for label in category.labels if label in index]
        columns.append(scores[:, known].max(axis=1) if known else np.zeros(len(scores)))
    return np.stack(columns, axis=1)


def triggered_categories(frame_scores, threshold: float, min_frames: int) -> np.ndarray:
    """The app's rule: a category triggers when it reaches the threshold in enough frames."""
    return (frame_scores >= threshold).sum(axis=0) >= min_frames


def detection_summary(triggered, expected, category_names) -> dict:
    """Score clip-level detection. `expected` = category index per clip, -1 for none."""
    target = expected >= 0
    other = ~target
    rates = {}
    for i, name in enumerate(category_names):
        clips = expected == i
        if clips.any():
            rates[name] = float(triggered[clips, i].mean())
    hit = triggered[target, expected[target]]
    others_fired = triggered[target].copy()
    others_fired[np.arange(target.sum()), expected[target]] = False
    return {
        "detection_rate": rates,
        "mean_detection_rate": float(np.mean(list(rates.values()))),
        "overall_detection_rate": float(hit.mean()),
        "false_alarm_rate": float(triggered[other].any(axis=1).mean()),
        "false_alarms_by_category": {
            name: int(triggered[other, i].sum()) for i, name in enumerate(category_names)
        },
        "wrong_category_rate": float(others_fired.any(axis=1).mean()),
        "target_clips": int(target.sum()),
        "other_clips": int(other.sum()),
    }


@dataclass(frozen=True)
class Clip:
    filename: str
    fold: int
    target: int
    category: str


def load_metadata(csv_path: Path) -> list[Clip]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return [
            Clip(row["filename"], int(row["fold"]), int(row["target"]), row["category"])
            for row in csv.DictReader(handle)
        ]


def missing_labels(
    mapping: Mapping[str, Sequence[str]], class_names: Sequence[str]
) -> list[str]:
    known = set(class_names)
    return [label for labels in mapping.values() for label in labels if label not in known]


def predict_mapped(
    clip_scores: np.ndarray,
    class_names: Sequence[str],
    mapping: Mapping[str, Sequence[str]],
    categories: Sequence[str],
) -> np.ndarray:
    """Arm A: predict the category whose best label scores highest, per clip."""
    index = {name: i for i, name in enumerate(class_names)}
    per_category = np.stack(
        [
            clip_scores[:, [index[label] for label in mapping[c]]].max(axis=1)
            for c in categories
        ],
        axis=1,
    )
    return per_category.argmax(axis=1)


@dataclass(frozen=True)
class CrossValidationResult:
    predictions: np.ndarray
    fold_accuracies: list[float]


def cross_validate(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    make_model: Callable[[], object],
) -> CrossValidationResult:
    """Arm B: train on four folds and predict the fifth, for each fold in turn."""
    predictions = np.empty_like(targets)
    accuracies = []
    for fold in np.unique(folds):
        test = folds == fold
        model = make_model()
        model.fit(features[~test], targets[~test])
        predictions[test] = model.predict(features[test])
        accuracies.append(float(np.mean(predictions[test] == targets[test])))
    return CrossValidationResult(predictions, accuracies)


def summarise(
    targets: np.ndarray,
    predicted: np.ndarray,
    folds: np.ndarray,
    categories: Sequence[str],
) -> dict:
    labels = list(range(len(categories)))
    per_class = f1_score(targets, predicted, labels=labels, average=None, zero_division=0)
    fold_accuracies = [
        float(accuracy_score(targets[folds == f], predicted[folds == f]))
        for f in np.unique(folds)
    ]
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "fold_accuracies": fold_accuracies,
        "fold_std": float(np.std(fold_accuracies)),
        "macro_f1": float(
            f1_score(targets, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": {c: float(s) for c, s in zip(categories, per_class)},
    }
