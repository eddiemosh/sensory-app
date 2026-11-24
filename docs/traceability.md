# Requirement-to-test traceability

Every requirement in [`requirements.md`](requirements.md), with the tests that
check it. Test files are in `tests/`. The table was built on 2026-11-16; doing
so showed three requirements with no direct test (NFR2, the FR5 threshold
slider, and FR11's warning display), and tests were added for them the same
day.

**Test levels.**
- **Unit:** one module, with hand-made inputs.
- **Integration:** a module with the real YAMNet model.
- **System:** the whole Streamlit app driven as a user would, through
  Streamlit's `AppTest`.
- Tests marked † need the real model and run locally only. All others run in
  CI on every push.

## Functional requirements

| ID | Requirement (short) | Unit / integration tests | System tests | Status |
| --- | --- | --- | --- | --- |
| FR1 | Load an uploaded or recorded clip; convert to 16 kHz mono | `test_audio`: `test_16khz_mono_is_unchanged`, `test_higher_sample_rate_is_resampled_with_same_duration`, `test_resampling_preserves_frequency`, `test_stereo_is_averaged_to_mono`, `test_loads_from_file_like_object`, `test_samples_are_limited_to_model_range` | `test_consent_reveals_upload_by_default`, `test_microphone_option_reveals_recorder`, `test_uploaded_recurring_typing_gets_gentle_instrumental`, `test_recorded_clip_is_analysed_like_an_upload` | Covered |
| FR2 | Useful error for corrupt, unsupported, empty or over-long audio | `test_audio`: `test_corrupt_file_gives_useful_error`, `test_missing_file_gives_useful_error`, `test_empty_audio_gives_useful_error`, `test_overlong_audio_is_rejected`, `test_non_finite_samples_are_rejected` | `test_corrupt_upload_shows_an_error_and_no_results` | Covered; see note 1 |
| FR3 | Run YAMNet; per-frame scores for 521 classes | `test_model_integration`: `test_model_has_521_classes`†, `test_silence_is_recognised`†, `test_analyse_returns_one_embedding_per_frame`† | `test_system_model`: all four† | Covered (locally) |
| FR4 | Show the five highest-scoring classes | `test_predictions`: `test_top_k_ranks_by_mean_score_across_frames`, `test_top_k_is_limited_by_number_of_classes`, `test_top_k_with_no_frames_is_empty`, `test_mismatched_scores_are_rejected` | `test_uploaded_recurring_typing_gets_gentle_instrumental` (five rows), `test_silence_gets_no_intervention`† | Covered |
| FR5 | User configures categories, threshold, minimum frames and speech loudness | `test_decision`: `test_invalid_config_is_rejected` (parametrised), `test_invalid_category_thresholds_are_rejected` (parametrised) | `test_deselecting_a_category_stops_it_triggering`, `test_raising_the_score_threshold_changes_the_decision`, `test_raising_minimum_frames_changes_the_decision`, `test_raising_the_speech_loudness_setting_changes_the_decision` | Covered |
| FR6 | Trigger only at threshold in at least the minimum frames | `test_decision`: `test_isolated_detection_below_min_frames_is_ignored`, `test_score_exactly_at_threshold_counts`, `test_score_below_threshold_does_not_trigger`, `test_frames_are_pooled_across_labels_in_a_category_without_double_counting`, `test_clip_too_short_for_min_frames_is_explained` | `test_one_off_sound_is_ignored`, `test_single_beep_is_ignored`† | Covered |
| FR7 | A triggered category leads to its intervention | `test_decision`: `test_repeated_category_triggers_its_intervention`, `test_different_category_gets_different_intervention` | `test_uploaded_recurring_typing_gets_gentle_instrumental`, `test_recorded_clip_is_analysed_like_an_upload`, `test_repeated_beeps_get_stronger_masking`† | Covered |
| FR8 | Two or more categories → general focus | `test_decision`: `test_several_triggered_categories_recommend_general_focus` | `test_two_recurring_categories_get_general_focus` | Covered |
| FR9 | No category triggered → no intervention | `test_decision`: `test_sound_outside_configured_categories_does_not_trigger`, `test_no_configured_categories_is_safe` | `test_one_off_sound_is_ignored`, `test_silence_gets_no_intervention`† | Covered |
| FR10 | Plain-language explanation and per-category evidence | `test_decision`: `test_explanation_gives_evidence_for_a_trigger`, `test_evidence_covers_every_category_with_peak_score` | `test_explanation_is_shown_with_the_decision`, `test_uploaded_recurring_typing_gets_gentle_instrumental` (evidence table) | Covered |
| FR11 | Unknown labels reported and ignored, not a failure | `test_decision`: `test_unknown_label_is_reported_and_ignored`, `test_category_with_only_unknown_labels_does_not_trigger`; `test_model_integration`: `test_every_default_label_exists_in_yamnet`† | `test_label_missing_from_the_model_is_shown_as_a_warning` | Covered |
| FR12 | Per-category score threshold and minimum loudness; loud speech only | `test_decision`: `test_category_threshold_overrides_the_global_one`, `test_loud_speech_triggers_its_intervention`, `test_quiet_speech_is_ignored`, `test_only_frames_that_are_both_confident_and_loud_count`, `test_gated_category_without_levels_is_skipped_with_a_warning`, `test_explanation_states_the_category_rule_including_loudness`; `test_audio`: `test_full_scale_sine_is_about_minus_3_dbfs`, `test_silence_has_the_floor_level`, `test_levels_follow_yamnet_frame_timing`, `test_frames_past_the_end_get_the_floor_level`; `test_predictions`: `test_levels_must_match_the_number_of_frames` | `test_loud_speech_gets_stronger_masking`, `test_quiet_speech_is_ignored`, `test_background_speech_below_the_speech_threshold_is_ignored` | Covered with a stand-in model; see note 4 |
| FR13 | Optional trained detector for four categories; labels stay default | `test_detector`: `test_exported_detector_matches_scikit_learn`, `test_save_and_load_round_trip`, `test_detector_scores_are_added_as_named_columns_in_category_order`, `test_trained_mode_swaps_only_the_four_detector_categories`, `test_rule_engine_triggers_on_confident_detector_frames`, `test_training_frames_drop_quiet_target_frames_and_skip_other_when_closed`, `test_shipped_detector_file_is_valid`; `test_app_task`: all six | `test_labels_are_the_default_detection_method`, `test_trained_detector_drives_its_categories`, `test_trained_mode_ignores_yamnet_labels_for_its_categories`, `test_trained_mode_keeps_yamnet_labels_for_chewing`, `test_top_sounds_table_lists_only_yamnet_labels`, `test_trained_detector_mode_runs_with_the_real_models`† | Covered; accuracy in `evaluation.md`; see note 5 |

## Non-functional requirements

| ID | Requirement (short) | Tests | Other evidence | Status |
| --- | --- | --- | --- | --- |
| NFR1 | Audio processed only by the app's server; not saved or sent elsewhere | `test_analysis_writes_no_files`; `test_analysis_makes_no_network_requests_or_file_writes`† | Code inspection: audio is read from memory only. `.gitignore` excludes audio files. | Covered; see note 2 |
| NFR2 | Decision engine independent of interface and ML framework | `test_architecture`: `test_decision_engine_loads_neither_the_interface_nor_the_ml_framework`, `test_default_rules_load_neither_the_interface_nor_the_ml_framework`, `test_model_module_defers_loading_tensorflow_until_used` | All 20 `test_decision` tests run without Streamlit or TensorFlow | Covered |
| NFR3 | Same model and input give the same result | `test_results_are_reproducible`† | | Covered (locally) |
| NFR4 | Recommendations, not diagnosis or treatment | `test_notice_and_disclaimer_are_shown_before_consent` (checks "not medical advice") | Review of all interface wording | Partly; see note 3 |
| NFR5 | Recurrence described as within the clip | `test_decision`: `test_explanation_stays_within_the_clip`; `test_notice_and_disclaimer_are_shown_before_consent` (checks "within the clip") | | Covered |
| NFR6 | Explain audio handling and get agreement before accepting audio | `test_notice_and_disclaimer_are_shown_before_consent`, `test_no_audio_input_is_offered_before_consent`, `test_consent_reveals_upload_by_default` | | Covered |

## Notes on partial coverage

1. **FR2, unsupported formats.** The uploader only accepts WAV, FLAC, OGG and
   MP3; Streamlit's browser widget enforces this, and the test harness does
   not. A file with a supported extension but unreadable content is tested.
2. **NFR1, hosted mode.** The tests show the app itself writes nothing and
   connects to nothing outside the computer during analysis. When hosted, the
   platform receives the audio by design; that is covered by the in-app notice
   (NFR6), not a test. The write check sees writes made through Python, not
   inside TensorFlow's C++ code.
3. **NFR4.** A test can check that the disclaimer is present but not that all
   wording avoids medical claims. That needs a manual review of the interface
   text before submission.
4. **FR12 with real speech.** No test uses a real speech recording, because
   none is available and synthetic audio does not sound like speech to
   YAMNet. The 0.6 score and -20 dBFS defaults are unmeasured. Recording a
   normal and a loud conversation, then checking the app's evidence table,
   would test both.
5. **FR13 accuracy.** Tests check that the detector is exported, loaded and
   used correctly. Its accuracy is measured separately by the app-task
   evaluation (80% detection, 3.0% false alarms on ESC-50). It has not been
   tested on talking or music.

## Supporting tests not tied to one requirement

- `test_smoke.py`: the package imports and has a version.
- `test_evaluation.py` and `test_every_esc50_mapped_label_exists_in_yamnet`†:
  the ESC-50 evaluation code (ML investigation), including a check that
  cross-validation never trains on its test fold.

## Showing that the tests can fail

Six deliberate faults were put into the code one at a time and reverted. Each
was caught: removing the consent gate, saving the upload to disk, ignoring the
sidebar settings, opening an outside network connection, importing Streamlit
into the rule engine, and hiding warnings. Details are in the development log
(2026-11-16).
