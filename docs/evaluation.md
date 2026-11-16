# ESC-50 evaluation (first run, 2026-11-11)

Feeds report section 3 (AI/ML investigation) and section 6 (testing and
evaluation). Raw numbers: [`evaluation/results.json`](evaluation/results.json).
Reproduce with `python scripts/evaluate_esc50.py` after putting ESC-50 in
`data/ESC-50/`.

## What was compared

| Arm | Approach | Training on ESC-50 |
| --- | --- | --- |
| A | Pretrained YAMNet labels, hand-mapped to the 50 ESC-50 classes (`ESC50_TO_YAMNET` in `src/sensoryaware/evaluation.py`). A clip is assigned the class whose best mapped label has the highest mean score. | None (zero-shot) |
| B | YAMNet's 1,024-value embeddings, averaged over the clip, fed to a scikit-learn classifier. Three classifiers were tried: logistic regression, an RBF SVM and 5-nearest-neighbours, each after standardisation. | Yes, 5-fold cross-validation |

Both arms use ESC-50's five predefined folds. The folds keep clips cut from the
same original recording together, so a random split would leak near-duplicates
between training and test data and inflate Arm B.

## Results

| Arm | Accuracy | Std across folds | Macro F1 |
| --- | --- | --- | --- |
| A: mapped labels | 0.654 | 0.016 | 0.615 |
| B: logistic regression | **0.853** | 0.020 | **0.853** |
| B: SVM (RBF) | 0.787 | 0.021 | 0.790 |
| B: 5-NN | 0.684 | 0.014 | 0.687 |

Chance is 0.02. For context, Piczak (2015) reports human listeners at about
81% on ESC-50 (check this citation before using it).

### Classes closest to the app's purpose (F1)

| Class | A: mapped | B: logistic | B: SVM | B: 5-NN |
| --- | --- | --- | --- | --- |
| mouse_click | 0.087 | 0.579 | 0.478 | 0.286 |
| keyboard_typing | 0.714 | 0.895 | 0.880 | 0.865 |
| clock_tick | 0.578 | 0.847 | 0.765 | 0.615 |
| door_wood_knock | 0.783 | 0.828 | 0.725 | 0.535 |
| clock_alarm | 0.692 | 0.947 | 0.933 | 0.889 |

### Where each arm goes wrong

**Arm A**, most frequent confusions (true → predicted, clips out of 40):
rooster → hen 38, snoring → breathing 31, chainsaw → engine 20,
pouring_water → water_drops 20, mouse_click → keyboard_typing 18.
Classes with F1 of zero: rooster, door_wood_creaks, can_opening.

**Arm B (logistic regression)**: airplane ↔ helicopter 15 in total,
water_drops → clock_tick 6, keyboard_typing → mouse_click 6. Weakest classes:
airplane 0.56, washing_machine 0.56, mouse_click 0.58.

### Cost

YAMNet takes 0.033 s per 5-second clip on average (95th percentile 0.054 s) on
a laptop CPU, so analysis is far faster than real time. Training and testing
all five folds took 88 s for logistic regression, 11 s for the SVM and 1 s for
5-NN.

## Interpretation

- **Training a small head on embeddings beats label mapping by 20 points.**
  Arm B learns ESC-50's own class boundaries; Arm A depends on whether
  AudioSet's label set happens to line up with them.
- **Arm A's errors are mostly mapping errors, not hearing errors.** YAMNet has
  one label, "Chicken, rooster", for both hens and roosters; it was mapped to
  hen, and roosters were then nearly always called hens. Snoring and breathing,
  chainsaw and engine, and pouring and dripping are all cases where the model
  hears roughly the right thing but the mapping forces a choice it has no
  basis for. Classes with no matching label (can opening, washing machine,
  drinking) score near zero.
- **The mapping was written before the results were seen and was not changed
  afterwards.** Editing it to fix rooster now would tune Arm A on its own test
  data. This is noted as a limitation instead.
- **Clicking is the app's weakest category.** Mapped labels identify only 9% of
  mouse clicks (F1 0.087), mostly confusing them with typing. Even Arm B only
  reaches 0.58. This matches the earlier observation that synthetic clicks
  scored low on YAMNet's `Tick` label.
- **Logistic regression beat the SVM and kNN.** With 1,600 training clips and
  1,024 features, a strongly regularised linear model generalises best; kNN
  suffers most from the high dimensionality.

## Implications for the app

The decision engine currently uses YAMNet's labels directly, the same approach
as Arm A. These results suggest a trained head (Arm B) would detect clicking
and ticking far more reliably. Options, to decide before the final build:

1. Keep labels in the app (simple, transparent, no training data needed) and
   report the weakness honestly.
2. Train the logistic-regression head on the ESC-50 classes the app cares about
   and feed its probabilities into the same rule engine.

## Limitations

- Arm B is trained and tested on ESC-50. Real rooms have overlapping sounds,
  background noise and different microphones, so its accuracy on users'
  recordings is likely lower. Arm A has no such training advantage.
- Classifier settings (C = 0.1 for logistic regression, C = 10 for the SVM,
  k = 5) were fixed in advance, not tuned. Tuning them would need a nested
  split to avoid tuning on the test folds.
- Clip features are a plain mean over frames, which includes any silent
  padding in shorter ESC-50 clips. Max pooling or mean plus standard deviation
  were not tried.
- ESC-50 clips each contain one labelled sound; the app's rule (a sound
  recurring over several frames) is not evaluated here.
- ESC-50 is believed to be licensed for non-commercial use only (see README).
