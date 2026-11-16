# Design decisions

Each entry records a decision, the options considered and why one was chosen,
so the report can show the reasoning behind the design. Newest last. The
development log (`devlog.md`) says when each was made.

---

## D1: Measure recurrence within the analysed clip (2026-11-10)

**Context.** The plan's rule is that a one-off sound is ignored and a recurring
one triggers an intervention (US3, US4). The app analyses one recording at a
time and keeps nothing between recordings.

**Options.** (a) Track sounds over real time across recordings; (b) count how
many of YAMNet's frames (one every 0.48 s) in the clip contain the sound.

**Decision.** (b). A category triggers when its score reaches the threshold in
at least a minimum number of frames (FR6). The interface describes this as
recurrence within the clip only (NFR5).

**Why.** (a) needs continuous listening and stored history, which conflicts
with the privacy requirements (NFR1) and is beyond a proof of concept. (b) is
honest about what is measured and makes the rule unit-testable.

---

## D2: Process audio on the server running the app and keep nothing (2026-11-10)

**Context.** Recordings can capture other people's voices.

**Decision.** Audio is analysed in memory by the server running the app, never
written to disk or sent to another service (NFR1). When hosted, that server is
a third-party platform, which the in-app notice says before any audio is
accepted, behind a consent checkbox (NFR6). Usage statistics are off.

**Why.** Running locally gives the strongest privacy; hosting was wanted for
access, so the notice was rewritten to be accurate for both cases rather than
promising local-only processing.

---

## D3: Detect sounds with YAMNet's own labels, not a trained classifier (2026-11-16)

**Context.** The ESC-50 evaluation (`evaluation.md`) found that a logistic
regression trained on YAMNet embeddings (Arm B, accuracy 0.853) beats YAMNet's
labels mapped to ESC-50 classes (Arm A, 0.654), including on every class
closest to the app's purpose. Clicking is the weakest: F1 0.087 with labels,
0.579 with the trained classifier.

**Options.**
1. Keep YAMNet's labels: each app category is a list of label names, and a
   category's score in a frame is its best label's score.
2. Replace them with a classifier trained on ESC-50 embeddings, feeding its
   probabilities into the same rule engine.
3. A mix: the classifier where ESC-50 has a matching class, labels elsewhere.

**Decision.** Option 1. The comparison is reported as part of the ML
investigation, the clicking weakness is reported in the evaluation, and
option 2 is described as future development.

**Why the higher-scoring option was not adopted.** The 0.853 was measured in
conditions the app does not have:

- *Closed set.* Every ESC-50 clip belongs to one of its 50 classes, so the
  classifier always picks one. Real recordings are mostly speech, music,
  silence and room noise, which it would force into a class such as typing.
  YAMNet's 521 labels include those sounds, so they have somewhere harmless to
  go. Fixing this needs an "other" class and training data ESC-50 lacks.
- *Clip versus frame.* The classifier was trained on whole-clip averages, but
  the rule engine needs a score for every frame to count recurrence (D1).
  Single frames differ from averages, and training on frames would label
  ESC-50's silent padding as the clip's sound. The 0.853 would no longer apply.
- *Coverage.* ESC-50 has no chewing class, so option 2 becomes option 3, with
  two detection methods to explain and test.
- *Recording conditions.* ESC-50's clips are mostly clean single sounds.
  YAMNet was trained on about 2 million varied YouTube clips and is likely to
  degrade less on laptop recordings of shared rooms.
- *Explanations.* Labels give explanations such as "`Typing` scored at least
  0.2 in 7 of 12 frames" (FR10, US5). A classifier's score is harder to
  explain.
- *Thresholds.* The score threshold and frame count would have to be re-set
  for the classifier's probabilities.
- *Licence.* Shipping a model trained on ESC-50 brings its (believed)
  non-commercial licence into the product; with labels, ESC-50 is only used
  for testing.
- *Time.* An estimated 2 to 4 days of the 11 left before the target date.

**Rejected follow-up.** Tuning the clicking category's label list against
ESC-50's click clips was considered and rejected. It would fit the app to
data by hand, contaminate the evaluation if the same clips were reused, and
could not fix the underlying gap: YAMNet has no mouse-click label.

**Consequences.**
- The app keeps its current, transparent detection and explanations.
- Clicking detection is known to be weak and is reported as such.
- The label `Clickety-clack` was checked against the published AudioSet
  ontology: it means "fast and rhythmic clicking of machinery, such as
  keystrokes on a typewriter or steel wheels running on tracks" and sits under
  `Clicking`. It stays in the clicking category. Because it includes
  typewriter keystrokes, fast typing may trigger both clicking and typing, and
  so the general focus profile.
