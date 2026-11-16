"""System tests: drive the real Streamlit app (app.py) as a user would.

Streamlit's AppTest runs app.py in-process and lets a test tick boxes, choose
options, upload files and record clips. YAMNet is replaced by a stand-in whose
scores each test chooses, so these tests run in CI and check the whole path
from input to displayed recommendation. test_system_model.py repeats the key
journeys with the real model.
"""

import builtins
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest

import sensoryaware.model
from sensoryaware.config import DEFAULT_CONFIG
from sensoryaware.predictions import FramePredictions

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CLASS_NAMES = tuple(
    sorted({label for c in DEFAULT_CONFIG.categories for label in c.labels} | {"Speech", "Silence"})
)
NUM_FRAMES = 10


def wav_bytes(seconds=5.0, rate=16000, amplitude=0.1):
    """A tone. The stand-in model ignores its content, but its loudness is
    measured for real: amplitude 0.1 is about -23 dBFS, 0.5 about -9 dBFS."""
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, amplitude * np.sin(2 * np.pi * 440 * t), rate, format="WAV")
    return buffer.getvalue()


class FakeClassifier:
    """Stands in for YamnetClassifier. Each label scores 0.9 in the given frames."""

    frames_by_label: dict[str, list[int]] = {}
    calls = 0

    def __init__(self):
        self.class_names = CLASS_NAMES

    def classify(self, waveform):
        FakeClassifier.calls += 1
        scores = np.zeros((NUM_FRAMES, len(CLASS_NAMES)), dtype=np.float32)
        scores[:, CLASS_NAMES.index("Speech")] = 0.3
        for label, frames in FakeClassifier.frames_by_label.items():
            scores[frames, CLASS_NAMES.index(label)] = 0.9
        return FramePredictions(CLASS_NAMES, scores)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(sensoryaware.model, "YamnetClassifier", FakeClassifier)
    FakeClassifier.frames_by_label = {}
    FakeClassifier.calls = 0
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    yield at
    st.cache_resource.clear()


def consent(at):
    at.checkbox[0].check().run()
    return at


def upload(at, data=None, name="clip.wav"):
    at.file_uploader[0].upload(name, data or wav_bytes(), "audio/wav").run()
    return at


def recommendation(at):
    texts = [s.value for s in at.success] + [i.value for i in at.info]
    return " ".join(texts)


# --- Before any audio (NFR4, NFR5, NFR6) -------------------------------------


def test_app_starts_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "👂 Earitation"


def test_notice_and_disclaimer_are_shown_before_consent(app):
    assert "How your audio is handled" in app.info[0].value
    caption = " ".join(c.value for c in app.caption)
    assert "not medical advice" in caption
    assert "within the clip" in caption


def test_no_audio_input_is_offered_before_consent(app):
    assert not app.checkbox[0].value
    assert len(app.file_uploader) == 0
    assert len(app.audio_input) == 0
    assert len(app.radio) == 0


def test_consent_reveals_upload_by_default(app):
    consent(app)
    assert app.radio[0].value == "Upload a file"
    assert len(app.file_uploader) == 1


def test_microphone_option_reveals_recorder(app):
    consent(app)
    app.radio[0].set_value("Record with microphone").run()
    assert len(app.audio_input) == 1
    assert len(app.file_uploader) == 0


# --- Full journeys (FR1, FR4, FR7-FR10) --------------------------------------


def test_uploaded_recurring_typing_gets_gentle_instrumental(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    assert not app.exception
    assert FakeClassifier.calls >= 1
    assert "Gentle instrumental audio" in recommendation(app)
    top_sounds = app.table[0].value
    assert len(top_sounds) == 5
    evidence = app.table[1].value
    typing = evidence[evidence["Category"] == "Typing"].iloc[0]
    assert typing["Met the rule"] == "Yes"
    assert typing["Frames detected"] == f"5 of {NUM_FRAMES}"


def test_recorded_clip_is_analysed_like_an_upload(app):
    FakeClassifier.frames_by_label = {"Alarm": [1, 2, 3, 4, 5, 6]}
    consent(app)
    app.radio[0].set_value("Record with microphone").run()
    app.audio_input[0].upload("recording.wav", wav_bytes(), "audio/wav").run()
    assert not app.exception
    assert "Stronger masking audio" in recommendation(app)


def test_one_off_sound_is_ignored(app):
    FakeClassifier.frames_by_label = {"Bang": [3, 4]}
    upload(consent(app))
    assert "No intervention recommended" in recommendation(app)


def test_two_recurring_categories_get_general_focus(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 1, 2, 3], "Bang": [5, 6, 7, 8]}
    upload(consent(app))
    assert "General focus audio" in recommendation(app)


def test_explanation_is_shown_with_the_decision(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    text = " ".join(m.value for m in app.markdown)
    assert f"Typing was detected in 5 of {NUM_FRAMES} analysis frames in this clip" in text
    assert "peak 0.90 for 'Typing'" in text


# --- Settings (FR5) ----------------------------------------------------------


def test_raising_minimum_frames_changes_the_decision(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    assert "Gentle instrumental audio" in recommendation(app)
    frames_slider = next(s for s in app.slider if s.label == "Minimum frames detected")
    frames_slider.set_value(6).run()
    assert "No intervention recommended" in recommendation(app)


def test_raising_the_score_threshold_changes_the_decision(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    threshold = next(s for s in app.slider if s.label == "Score threshold")
    threshold.set_value(0.95).run()
    assert "No intervention recommended" in recommendation(app)


def test_deselecting_a_category_stops_it_triggering(app):
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    categories = app.multiselect[0]
    categories.set_value([c for c in categories.value if c != "Typing"]).run()
    assert "No intervention recommended" in recommendation(app)


# --- Bad input (FR2) ---------------------------------------------------------


def test_corrupt_upload_shows_an_error_and_no_results(app):
    upload(consent(app), data=b"this is not audio", name="broken.wav")
    assert not app.exception
    assert len(app.error) == 1
    assert FakeClassifier.calls == 0
    assert all(h.value != "Recommendation" for h in app.subheader)


# --- Loud speech (FR12) ------------------------------------------------------


def test_loud_speech_gets_stronger_masking(app):
    FakeClassifier.frames_by_label = {"Speech": [0, 1, 2, 3, 4, 5]}
    upload(consent(app), data=wav_bytes(amplitude=0.5))
    assert "Stronger masking audio" in recommendation(app)
    text = " ".join(m.value for m in app.markdown)
    assert "-20 dBFS" in text


def test_quiet_speech_is_ignored(app):
    FakeClassifier.frames_by_label = {"Speech": [0, 1, 2, 3, 4, 5]}
    upload(consent(app), data=wav_bytes(amplitude=0.01))
    assert "No intervention recommended" in recommendation(app)


def test_background_speech_below_the_speech_threshold_is_ignored(app):
    # The stand-in scores Speech 0.3 in every frame: above the global 0.2
    # threshold but below speech's own 0.6, so even loud audio does not trigger.
    upload(consent(app), data=wav_bytes(amplitude=0.5))
    assert "No intervention recommended" in recommendation(app)


def test_raising_the_speech_loudness_setting_changes_the_decision(app):
    FakeClassifier.frames_by_label = {"Speech": [0, 1, 2, 3, 4, 5]}
    upload(consent(app), data=wav_bytes(amplitude=0.5))
    assert "Stronger masking audio" in recommendation(app)
    loudness = next(s for s in app.slider if "minimum loudness" in s.label)
    loudness.set_value(-3).run()
    assert "No intervention recommended" in recommendation(app)


# --- Labels the model lacks (FR11) -------------------------------------------


def test_label_missing_from_the_model_is_shown_as_a_warning(app, monkeypatch):
    reduced = tuple(name for name in CLASS_NAMES if name != "Ringtone")
    monkeypatch.setitem(globals(), "CLASS_NAMES", reduced)
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(consent(app))
    assert not app.exception
    assert "Gentle instrumental audio" in recommendation(app)
    warnings = " ".join(w.value for w in app.warning)
    assert "'Ringtone'" in warnings and "not a YAMNet label" in warnings


# --- Privacy (NFR1) ----------------------------------------------------------


def test_analysis_writes_no_files(app, monkeypatch):
    written = []
    real_open, real_os_open = builtins.open, os.open

    def recording_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            written.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def recording_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            written.append(str(path))
        return real_os_open(path, flags, *args, **kwargs)

    consent(app)
    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(os, "open", recording_os_open)
    FakeClassifier.frames_by_label = {"Typing": [0, 2, 4, 6, 8]}
    upload(app)
    monkeypatch.undo()
    assert "Gentle instrumental audio" in recommendation(app)
    assert written == []
