# Development log

A dated record of what was built, why, the decisions taken and the problems
hit. It is written as the work happens and feeds the report's implementation,
testing and AI declaration sections. Each entry notes where AI assistance
(Claude, via Claude Code) was used.

Entries are newest last.

---

## 2026-11-09: Repository set-up

**Done**
- Created the repository layout (`src/sensoryaware/`, `tests/`, `docs/`,
  `data/` ignored), pinned dependencies, `pytest.ini`, and a GitHub Actions
  workflow that runs pytest on every push and pull request (commit `8b3d974`).
- Pushed to GitHub (`eddiemosh/sensory-app`).

**Why**
- The plan flags repository evidence as something that cannot be added at the
  end: CI and small, frequent commits from day one.

**AI assistance**: Claude generated the layout, config files and workflow from
the coursework plan.

---

## 2026-11-10: Requirements, audio loading, YAMNet, decision engine, interface

**Done**
- Committed the first draft of requirements and user stories
  (`docs/requirements.md`, `4be7598`) *before* the interface and decision-engine
  code, so the design visibly precedes the build.
- Audio loading (`audio.py`): any supported file to 16 kHz mono, with clear
  errors for corrupt, empty, non-finite or over-long (more than 5 minutes) audio
  (FR1, FR2).
- YAMNet wrapper (`model.py`) and a script printing the top five classes for one
  file (`scripts/yamnet_top5.py`) (FR3, FR4) (`9cbc5f1`).
- Rule-based decision engine (`decision.py`, `config.py`), written test-first
  (`4cab486`). Rule: a category triggers when any of its labels scores at least
  0.2 in at least 4 frames; two or more triggered categories lead to the general
  focus profile; every decision carries an explanation and per-category
  evidence (FR5 to FR11).
- Streamlit interface (`app.py`, `4b2d618`) with file upload and microphone
  recording, sidebar settings, a privacy notice and a consent checkbox.

**Decisions**
- *Recurrence within a clip.* YAMNet scores a frame every 0.48 s, so "detected
  in N of M frames" is an honest, testable measure of recurrence. The interface
  and requirements say it is recurrence within the clip only (NFR5).
- *Thresholds are provisional.* 0.2 and 4 frames were chosen by reasoning, not
  measurement; tuning is left for the evaluation.
- *Microphone input* was added because no test recordings were available.
- *Privacy.* Locally, audio never leaves the computer: a check during model
  loading and analysis showed no network requests apart from the one-time
  YAMNet download. Audio is never written to disk, audio file types are
  git-ignored, and Streamlit's usage statistics are turned off. NFR1 and NFR6
  were rewritten when hosting was added (below).

**Testing**
- 40 tests: 36 unit tests that run in CI, and 4 marked `model` that need the
  real YAMNet and are run locally.
- Manual check with synthetic audio: repeated beeps trigger "Stronger masking
  audio"; a single beep, silence and low noise trigger nothing. Synthetic clicks
  scored low on YAMNet's `Tick` label (mean 0.075), so real clicks and typing
  still need testing.

**AI assistance**: Claude drafted the requirements, wrote the code and tests,
and ran the checks above. I chose the features and reviewed the output.

---

## 2026-11-10: Hosting on Streamlit Community Cloud

**Done**
- Deployed the full app, with a rewritten in-app notice saying that audio is
  sent to the server running the app (a third-party service when hosted), is
  analysed there and is not saved or passed on.
- Merged a dev container configuration added on GitHub (`390a31c`, `7a9a6dc`).

**Problems**
- *App stuck loading.* The committed config bound the server to `localhost`,
  which the host's proxy cannot reach. Fixed by binding to `0.0.0.0` in
  `.streamlit/config.toml` and running locally with
  `--server.address localhost` (`e08bbe1`). The hosted fix is not yet confirmed.
- Community Cloud cannot change an app's branch or entry point in place; the
  app has to be deleted and redeployed.

**Decisions**
- The hosted app should stay restricted to invited viewers.

**AI assistance**: Claude diagnosed the binding problem and wrote the notice.

---

## 2026-11-11: Name

**Done**
- Renamed the app (not the package or the report) to **Earitation** (`68a2775`).

**AI assistance**: Claude suggested names; I chose this one.

---

## 2026-11-11: Development log started

**Done**
- Started this log at my request to document the project as it goes, and
  backfilled the entries above from the commit history.

---

## 2026-11-11: ESC-50 evaluation, first run

**Done**
- Downloaded ESC-50 from its GitHub repository into the git-ignored `data/`
  folder: 2,000 five-second clips, 50 classes, 5 predefined folds. Only the
  audio and the metadata CSV were extracted.
- Added `YamnetClassifier.analyse()`, returning per-frame embeddings as well as
  scores.
- Wrote `src/sensoryaware/evaluation.py` test-first (`tests/test_evaluation.py`
  failed before the module existed): metadata loading, the ESC-50 to YAMNet
  label mapping, Arm A prediction, fold-based cross-validation and summary
  metrics. A test checks that cross-validation never trains on the fold it
  tests. Two model tests check that every mapped label exists in YAMNet and
  that embeddings have 1,024 values per frame. Test count: 51.
- Wrote `scripts/evaluate_esc50.py`, which extracts features once (cached in
  `data/`), runs both arms and writes `docs/evaluation/results.json`.
- Wrote up the results in `docs/evaluation.md`.

**Results**
- Arm A (mapped labels): accuracy 0.654. Arm B (embeddings + logistic
  regression): 0.853; SVM 0.787; 5-NN 0.684.
- Clicking is the weakest app-relevant class: F1 0.087 with mapped labels,
  0.58 with the best trained head.

**Decisions**
- Used ESC-50's predefined folds, not a random split, to avoid leakage between
  clips cut from the same recording.
- Wrote the label mapping before running the evaluation and did not change it
  afterwards, even where it clearly failed (rooster mapped to hen), so Arm A
  is not tuned on its own test data.
- No label is mapped to two classes, so a prediction is never decided by tie
  order.
- Classifier settings were fixed in advance rather than tuned.

**Open question**
- Whether the app should keep using YAMNet's labels or switch to a trained
  head for the categories it cares about. See "Implications for the app" in
  `docs/evaluation.md`.

**AI assistance**: Claude wrote the label mapping, the evaluation code and
tests, ran the evaluation and drafted the write-up. The interpretation needs my
own review before it goes into the report.

---

## 2026-11-16: Decision on labels versus a trained classifier

**Done**
- Decided to keep YAMNet's labels in the app rather than switch to the trained
  classifier from the evaluation. Recorded as D3 in the new
  `docs/decisions.md`, along with two earlier decisions written up from this
  log (D1 recurrence within a clip, D2 audio handling).
- Considered and rejected tuning the clicking category's labels on ESC-50: it
  would be fitting by hand and could not fix the missing mouse-click label.
- Checked `Clickety-clack` against Google's published AudioSet ontology
  (`ontology.json` on GitHub). I had assumed it meant train wheels only; its
  definition also covers typewriter keystrokes, so it stays in the clicking
  category. Noted the overlap with typing.

**Why**
- The classifier's 0.853 accuracy was measured on whole clips from a closed
  set of 50 classes; the app scores single frames of recordings that are
  mostly speech, silence and other sounds outside that set. Details in D3.

**AI assistance**: Claude explained the trade-offs and drafted the decision
record. I made the decision after questioning whether tuning labels was
worthwhile.
