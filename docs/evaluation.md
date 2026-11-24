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

Chance is 0.02. For context, human listeners scored 81.3% on ESC-50 in a
crowdsourcing experiment (Piczak, K. J. "ESC: Dataset for Environmental Sound
Classification", Proceedings of the 23rd ACM International Conference on
Multimedia, 2015, doi:10.1145/2733373.2806390; figure confirmed on the
dataset's GitHub page on 2026-11-16).

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

The decision engine uses YAMNet's labels directly, the same approach as Arm A.
These results suggest a trained head (Arm B) would detect clicking and ticking
far more reliably on ESC-50.

**Decided (2026-11-16): keep labels in the app.** Arm B's result was measured
on a closed set of whole clips, which the app does not have: real recordings
are mostly sounds outside ESC-50's 50 classes, and the rule engine scores
single frames. The full reasoning is decision D3 in
[`decisions.md`](decisions.md). Clicking detection is therefore known to be
weak, and a trained head is future work.

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
- Label mapping is a judgement call. `Clickety-clack` is defined in AudioSet as
  rhythmic machine clicking, "such as keystrokes on a typewriter or steel
  wheels running on tracks". This evaluation maps it to *train*, while the app
  puts it under *clicking*. Both are defensible; the mapping here was fixed
  before the run and left unchanged.

---

# The app's own task: detection among everything else (2026-11-24)

The comparison above measures classification of whole clips from a closed set
of 50. The app's task is different: it scores single frames, applies the rule
(a category triggers when it passes in at least 4 frames), and must stay quiet
for everything it does not care about. This second evaluation measures that
task. Raw numbers: [`evaluation/app-task-results.json`](evaluation/app-task-results.json).
Reproduce with `python scripts/evaluate_app_task.py`.

## Set-up

- **Targets:** five ESC-50 classes mapped to app categories: mouse_click and
  clock_tick → clicking, keyboard_typing → typing, door_wood_knock → banging,
  clock_alarm → alarms (200 clips). The other 45 classes (1,800 clips) should
  trigger nothing.
- **Scores:** *detection rate* = share of target clips where the right
  category triggered (averaged over the four categories); *false alarm rate* =
  share of the 1,800 other clips where any category triggered.
- **Folds:** ESC-50's five folds; each arm is trained or tuned on four and
  tested on the fifth.
- **Check:** the evaluation's vectorised rule was compared with the app's
  `decide()` on all 2,000 clips and agreed on every one.
- **Not covered:** chewing and loud speech have no ESC-50 class. Chewing's
  false alarms are still counted.

| Arm | What it is |
| --- | --- |
| A, default | YAMNet labels with the app's thresholds (score 0.2, 4 frames): the app as shipped |
| A, tuned | YAMNet labels with thresholds chosen on four folds by detection minus false alarms, tested on the fifth |
| B-closed | Frame-level logistic regression on YAMNet embeddings, trained on target sounds only, so every frame is forced into a target category |
| B-open | The same with a fifth "other" class trained on frames from the 45 other classes |

For B, each training frame takes its clip's label, except quiet frames of a
target clip (more than 20 dB below its loudest frame), which are left out so
silent padding is not taught as the sound. Classifier settings (C = 0.1,
balanced class weights) were fixed in advance. For B, a frame counts when the
detector's probability for a category is at least 0.5, and a category
triggers in at least 4 frames, the same rule as A, untuned. B uses the exact
code and rule the app ships (`sensoryaware/detector.py`); the shipped model is
the same set-up trained once on all 2,000 clips
(`scripts/train_frame_detector.py`).

## Results

| Arm | Detection rate (± fold std) | False alarm rate (± fold std) |
| --- | --- | --- |
| A, default (as shipped) | 0.56 ± 0.06 | 0.047 ± 0.006 |
| A, tuned | 0.83 ± 0.04 | 0.135 ± 0.024 |
| B-closed | 0.95 ± 0.03 | **0.981** ± 0.009 |
| B-open (shipped as the app's trained detector) | **0.80** ± 0.06 | **0.030** ± 0.008 |

Detection by category:

| Arm | Clicking | Typing | Banging | Alarms |
| --- | --- | --- | --- | --- |
| A, default | 0.20 | 0.95 | 0.23 | 0.88 |
| A, tuned | 0.64 | 0.97 | 0.78 | 0.95 |
| B-closed | 0.89 | 0.97 | 0.97 | 0.97 |
| B-open | 0.88 | 0.93 | 0.45 | 0.95 |

## What this shows

- **The app as shipped is cautious.** It rarely raises a false alarm (4.7%)
  but misses most clicks (20% found) and knocks (23%).
- **The closed-set concern in D3 was right.** A detector that only knows the
  target sounds fires on 99% of everything else.
- **An "other" class fixes it.** B-open finds 80% of targets, against 56% for
  the shipped app, with a slightly lower false-alarm rate (3.0% vs 4.7%).
  Clicking rises from 20% to 88%. Its weak spot is knocking (45%).
- **Tuning the label thresholds is not a free win.** Every fold chose score
  0.1 and just 2 frames, which raises detection to 83% but triples false
  alarms (13.5%). Two frames is also about one short sound, which breaks the
  rule that a one-off sound is ignored (US3): the tuning objective rewards
  catching ESC-50's short clips, not recurrence. The defaults were kept.

## Limits of this evaluation

- B-open's "other" class has heard only ESC-50's 45 other classes. It has
  never heard ordinary talking or music, the most common sounds in a real
  room, so its false-alarm rate there is unknown. YAMNet's labels do include
  speech and music.
- ESC-50 clips are 5 seconds and often hold one or two events, which favours
  low frame counts; real recordings are longer.
- Chewing and loud speech are not scored for detection.
