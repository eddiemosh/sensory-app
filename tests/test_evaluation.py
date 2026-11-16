import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from sensoryaware.evaluation import (
    ESC50_CATEGORIES,
    ESC50_TO_YAMNET,
    cross_validate,
    load_metadata,
    missing_labels,
    predict_mapped,
    summarise,
)


def test_load_metadata_reads_clips(tmp_path):
    csv_path = tmp_path / "esc50.csv"
    csv_path.write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "1-100032-A-0.wav,1,0,dog,True,100032,A\n"
        "2-100038-A-14.wav,2,14,chirping_birds,False,100038,A\n"
    )
    clips = load_metadata(csv_path)
    assert [clip.filename for clip in clips] == ["1-100032-A-0.wav", "2-100038-A-14.wav"]
    assert [clip.fold for clip in clips] == [1, 2]
    assert [clip.target for clip in clips] == [0, 14]
    assert clips[1].category == "chirping_birds"


def test_mapping_covers_every_esc50_category():
    assert len(ESC50_CATEGORIES) == 50
    assert set(ESC50_TO_YAMNET) == set(ESC50_CATEGORIES)
    assert all(ESC50_TO_YAMNET[category] for category in ESC50_CATEGORIES)


def test_no_yamnet_label_is_mapped_to_two_categories():
    labels = [label for labels in ESC50_TO_YAMNET.values() for label in labels]
    assert len(labels) == len(set(labels))


def test_missing_labels_reports_labels_yamnet_lacks():
    mapping = {"dog": ("Dog", "Bark"), "robot": ("Beep-boop",)}
    assert missing_labels(mapping, ("Dog", "Bark", "Cat")) == ["Beep-boop"]


def test_predict_mapped_scores_a_category_by_its_best_label():
    class_names = ("Dog", "Bark", "Cat", "Meow")
    mapping = {"dog": ("Dog", "Bark"), "cat": ("Cat", "Meow")}
    # Cat's labels sum higher, but Bark alone beats either of them.
    clip_scores = np.array([[0.0, 0.5, 0.4, 0.4]])
    assert predict_mapped(clip_scores, class_names, mapping, ["dog", "cat"]).tolist() == [0]


def test_predict_mapped_picks_highest_category_per_clip():
    class_names = ("Dog", "Bark", "Cat", "Meow")
    mapping = {"dog": ("Dog", "Bark"), "cat": ("Cat", "Meow")}
    clip_scores = np.array([[0.1, 0.6, 0.3, 0.2], [0.1, 0.1, 0.2, 0.7]])
    predicted = predict_mapped(clip_scores, class_names, mapping, ["dog", "cat"])
    assert predicted.tolist() == [0, 1]


class RecordingModel:
    """Stands in for a classifier and remembers which rows it was trained on."""

    trained_on = []

    def fit(self, features, targets):
        RecordingModel.trained_on.append(set(features[:, 0].tolist()))
        self._majority = np.bincount(targets).argmax()
        return self

    def predict(self, features):
        return np.full(len(features), self._majority)


def test_cross_validate_never_trains_on_the_test_fold():
    RecordingModel.trained_on = []
    row_ids = np.arange(10, dtype=float).reshape(-1, 1)
    folds = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    targets = np.zeros(10, dtype=int)
    result = cross_validate(row_ids, targets, folds, RecordingModel)
    assert len(RecordingModel.trained_on) == 5
    for fold, trained in zip(range(1, 6), RecordingModel.trained_on):
        test_rows = set(row_ids[folds == fold, 0].tolist())
        assert not trained & test_rows
    assert result.predictions.shape == (10,)


def test_cross_validate_learns_separable_data():
    rng = np.random.default_rng(0)
    targets = np.repeat([0, 1], 50)
    features = rng.normal(size=(100, 2)) + targets[:, None] * 10
    folds = np.tile([1, 2, 3, 4, 5], 20)
    result = cross_validate(features, targets, folds, lambda: LogisticRegression())
    assert result.fold_accuracies == pytest.approx([1.0] * 5)


def test_summarise_reports_accuracy_and_macro_f1():
    targets = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])
    folds = np.array([1, 1, 2, 2])
    summary = summarise(targets, predicted, folds, ["a", "b"])
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["fold_accuracies"] == pytest.approx([0.5, 1.0])
    assert summary["per_class_f1"]["b"] == pytest.approx(0.8)
    assert 0 < summary["macro_f1"] < 1
