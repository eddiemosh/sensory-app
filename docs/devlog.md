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

---

## 2026-11-16: Architecture diagram and wireframes

**Done**
- `docs/design/architecture.svg`: modules, data passed between them, the
  boundary of the server that processes audio, the one-time model download,
  and the offline evaluation path. Requirement numbers are marked on the
  components that implement them.
- `docs/design/wireframes.svg`: low-fidelity layouts of the three screen
  states (before consent, choosing audio, results), annotated with the
  requirements each shows.
- `docs/design/README.md`: module responsibilities and dependencies, design
  principles, the data passed between stages, the decision rule, and the two
  deployment modes.

**Decisions**
- Drew the diagrams as SVG so they render on GitHub and can be inserted into
  the report without losing quality.
- The wireframes are labelled as drawn after the first build. The plan wanted
  design to visibly precede the build; the requirements did (commit
  `4be7598`), but the wireframes did not, and the report should not imply
  otherwise.

**Checks**
- Rendered both SVGs in a browser and fixed overlapping labels.
- Checked the design notes against the code, which corrected two claims: the
  rule engine depends on NumPy (not only the standard library) and has 20
  tests, not 18.

**AI assistance**: Claude drew the diagrams and drafted the design notes.

---

## 2026-11-16: System tests

**Done**
- `tests/test_system.py` (14 tests, run in CI): drives the real `app.py`
  through Streamlit's `AppTest` harness: ticking consent, choosing upload or
  microphone, uploading a file, recording a clip, changing settings, and
  uploading a corrupt file. YAMNet is replaced by a stand-in whose scores each
  test sets, so the tests check the whole path from input to displayed
  recommendation without the 2 GB TensorFlow dependency in CI.
- `tests/test_system_model.py` (4 tests, `model` marker): the same journeys
  with the real YAMNet on synthetic audio. Repeated beeps get stronger
  masking, silence and a single beep get nothing, and analysis makes no
  outside network connections and writes no files.
- Total tests: 69 (59 in CI).

**Checking the tests can fail (mutation checks)**
Each was a temporary edit to `app.py`, reverted afterwards:

| Deliberate fault | Caught by |
| --- | --- |
| Consent gate removed | `test_no_audio_input_is_offered_before_consent` |
| Upload saved to `leak.wav` | `test_analysis_writes_no_files` |
| Sidebar settings ignored | both settings tests |
| Connection opened to an outside address | `test_analysis_makes_no_network_requests_or_file_writes` |

**Problem**
- The no-network test first failed on a connection to `127.0.0.1`. That is
  loopback traffic inside the computer: on Windows, Python's asyncio event
  loop, which Streamlit uses, signals itself through a loopback socket. The
  test now fails on any connection that is not loopback. This matches NFR1,
  which is about audio reaching other services.

**Limits**
- The stand-in model means CI tests do not cover YAMNet itself; the
  model-marked tests do, but only locally.
- The file-write check sees writes made through Python. A write made inside
  TensorFlow's C++ code would not be seen; none is expected during analysis.
- Synthetic beeps show the pipeline works; they say nothing about accuracy on
  real recordings.

**AI assistance**: Claude wrote the tests and ran the mutation checks.

---

## 2026-11-16: Traceability table, and the gaps it found

**Done**
- `docs/traceability.md` maps all 11 functional and 6 non-functional
  requirements to the unit, integration and system tests that check them,
  with notes on partial coverage.
- Building it showed three requirements without a direct test. Added:
  - `tests/test_architecture.py` (NFR2): imports the rule engine and default
    rules in a fresh interpreter and checks that neither Streamlit nor
    TensorFlow is loaded; also checks `model.py` defers loading TensorFlow.
  - `test_raising_the_score_threshold_changes_the_decision` (FR5).
  - `test_label_missing_from_the_model_is_shown_as_a_warning` (FR11).
- Mutation checks on the new tests: adding `import streamlit` to the rule
  engine, and hiding warnings in the interface, were both caught.
- Checked by script that every test named in the table exists.
- Total tests: 74 (64 in CI).

**Remaining partial coverage**
- Unsupported file types are filtered by Streamlit's upload widget, which the
  test harness does not enforce.
- NFR4 (no medical claims) needs a manual review of the interface wording.
- Hosted processing is covered by the in-app notice, not a test.

**AI assistance**: Claude built the table, wrote the gap-filling tests and
ran the checks.

---

## 2026-11-16: Business research notes

**Done**
- `docs/business-research.md`: sourced notes on who has the problem, possible
  stakeholders, existing products, law and regulation, costs and funding. Each
  finding is marked as a primary or secondary source, and a list of what to
  check before writing the analysis is included.
- Confirmed from the ESC-50 repository that human listeners scored 81.3%
  (Piczak, 2015), and that the dataset is CC BY-NC 3.0 (ESC-10 is CC BY).
  Updated `evaluation.md` with the full citation.
- Confirmed Streamlit Community Cloud's limits from its documentation: up to
  2.7 GB of memory, 2 cores, and sleep after 12 hours without traffic.

**Decisions**
- These are notes, not the analysis. The plan requires the report's analysis
  and conclusions to be my own, so the notes state findings neutrally and
  list open questions rather than arguing a position.

**Problems**
- Checking the first draft against the search results found two errors,
  both fixed before committing: an office-noise citation that did not come
  from any result, and a prevalence figure attributed to the wrong paper. Every figure in the
  notes should still be checked against its source before it is used.
- The full text of the misophonia prevalence paper could not be read (the
  site blocked automated access), so its figures come from the abstract and
  the university's press release.

**Points raised for the report**
- Speech is the biggest distractor in office research, but it is not one of
  the app's categories.
- No product found both listens for distracting sounds and responds with
  audio, but the search was not exhaustive.

**AI assistance**: Claude ran the web searches, read the primary pages it
could reach, and compiled the notes.

---

## 2026-11-16: Loud speech category, and Spotify research

**Done**
- Added a **Loud speech** category at my request (decision D4). A frame
  counts only when a speech label scores at least 0.6 and the frame is at
  least -20 dBFS loud; it recommends stronger masking audio.
- Generalised the rule engine: any category can set its own score threshold
  and minimum loudness (new requirement FR12, user story US6). The other five
  categories are unchanged.
- `audio.py` measures each frame's loudness (RMS in dBFS over the same
  0.96 s window YAMNet scores); `app.py` attaches it to the predictions.
- Sidebar: a loudness slider for loud speech, and help text on the global
  threshold saying which categories use their own.
- Written test-first: 14 new unit tests failed before the code existed, then
  4 new system tests. Total tests: 92 (82 in CI).
- Mutation check: disabling the loudness gate made 4 tests fail.
- Updated requirements, design notes, traceability, the wireframe's category
  list and the business research notes.
- Researched Spotify playback for future development (decision D5).

**Problem**
- Two system tests failed at first: "Speech" became a configured label, so
  the stand-in model's label list contained it twice and the test scored the
  wrong copy. Fixed in the test helper.

**Decisions and open points**
- Loudness is relative to the microphone (dBFS), not a measured room level,
  so the user can adjust it.
- The office research says intelligibility, not loudness, predicts
  distraction; counting only loud speech is a trade-off to discuss in the
  report.
- Neither default has been checked on real speech. A real recording test is
  needed.
- Spotify: automatic playback needs Premium and is limited to 5 users in
  development mode, and since May 2025 wider access needs an organisation with
  250k monthly users. Opening playlist links needs none of that.

**AI assistance**: Claude implemented the change from my specification
("speech with a high threshold, only loud speech"), wrote the tests and docs,
and researched Spotify's API terms.

**Update (same day).** I accepted Spotify's limits for automatic playback
(Premium, 5 allow-listed users, no wider release without meeting the
extended-quota criteria). D5 now records automatic playback as the chosen
future route, with playlist links as the fallback. Nothing is built yet.

---

## 2026-11-16: Pull requests merged and branches tidied

**Done**
- Raised the five stacked feature branches as pull requests #1 to #5 and
  merged them into `main` in order, each with a merge commit after its CI run
  passed.
- Deleted the merged branches and `repo-setup`, and switched the GitHub
  default branch from `repo-setup` to `main`.
- Kept `feature/poc`: it holds one dev-container commit not in `main`, and the
  hosted app may still deploy from it.

**AI assistance**: Claude raised and merged the pull requests in the browser
and cleaned up the branches at my request.

---

## 2026-11-23: Full report draft written by AI

**Done**
- At my request, Claude wrote a complete first draft of the report, all eight
  sections plus a draft AI declaration, as a Claude Docs document:
  https://claude.ai/artifact/KDqYbenrfGD1kd4DAUiNog
- The draft uses figures from this repository (evaluation results, test
  counts, decisions D1 to D5) and the sources in `business-research.md`.
- About 3,100 to 3,200 words across sections 1 to 7 by estimate; section 2 is
  about 70 words over its budget.

**Decisions**
- My plan said the report's analysis and conclusions should be my own work,
  with AI limited to planning, structure and code. I chose to have AI draft
  every section anyway, as a starting point to rewrite. This must be stated in
  the AI declaration, and the draft rewritten in my own words before
  submission.

**Still to do in the draft**
- Items marked [verify]: several reference details (Venetjoki et al., Gong et
  al., Poulsen et al., Roelofsen, ICO guidance status, Access to Work).
- Items marked [add]: real-recording results, two figures, a results
  screenshot, the transcript reference, the final commit count.

**AI assistance**: Claude wrote the whole draft. My own rewrite and checks are
to be recorded here when done.
