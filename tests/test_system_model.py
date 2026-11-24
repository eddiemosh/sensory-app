"""System tests with the real YAMNet model. Excluded from CI by the model marker.

They use synthetic audio, so they show the whole system works end to end, not
how well it recognises real-world sounds.
"""

import builtins
import io
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.model

APP = str(Path(__file__).resolve().parent.parent / "app.py")
RATE = 16000


def wav_bytes(samples):
    buffer = io.BytesIO()
    sf.write(buffer, samples.astype(np.float32), RATE, format="WAV")
    return buffer.getvalue()


def repeated_beeps(seconds=10, beep_seconds=0.3, every=1.0, freq=1000):
    t = np.arange(int(seconds * RATE)) / RATE
    on = (t % every) < beep_seconds
    return 0.5 * np.sin(2 * np.pi * freq * t) * on


@pytest.fixture(scope="module", autouse=True)
def fresh_cache():
    st.cache_resource.clear()  # drop any stand-in model cached by other tests
    yield
    st.cache_resource.clear()


def analyse(samples, name="clip.wav"):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.checkbox[0].check().run()
    at.file_uploader[0].upload(name, wav_bytes(samples), "audio/wav").run()
    assert not at.exception
    return at


def recommendation(at):
    return " ".join([s.value for s in at.success] + [i.value for i in at.info])


def test_repeated_beeps_get_stronger_masking():
    at = analyse(repeated_beeps())
    assert "Stronger masking audio" in recommendation(at)
    evidence = at.table[1].value
    alarms = evidence[evidence["Category"] == "Alarms and beeps"].iloc[0]
    assert alarms["Met the rule"] == "Yes"


def test_silence_gets_no_intervention():
    at = analyse(np.zeros(5 * RATE))
    assert "No intervention recommended" in recommendation(at)
    assert at.table[0].value.iloc[0]["Sound"] == "Silence"


def test_single_beep_is_ignored():
    at = analyse(repeated_beeps(seconds=5, every=10.0))
    assert "No intervention recommended" in recommendation(at)


def test_analysis_makes_no_network_requests_or_file_writes(monkeypatch):
    analyse(np.zeros(RATE))  # make sure the model is downloaded and loaded first

    connections, written = [], []
    real_connect, real_open, real_os_open = socket.socket.connect, builtins.open, os.open

    def recording_connect(self, address):
        connections.append(address)
        return real_connect(self, address)

    def recording_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            written.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def recording_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            written.append(str(path))
        return real_os_open(path, flags, *args, **kwargs)

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.checkbox[0].check().run()
    monkeypatch.setattr(socket.socket, "connect", recording_connect)
    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(os, "open", recording_os_open)
    at.file_uploader[0].upload("beeps.wav", wav_bytes(repeated_beeps()), "audio/wav").run()
    monkeypatch.undo()

    assert "Stronger masking audio" in recommendation(at)
    # Loopback connections never leave the computer: on Windows, Python's
    # asyncio event loop (used by Streamlit) signals itself through one.
    external = [a for a in connections if not is_loopback(a)]
    assert external == []
    assert written == []


def is_loopback(address):
    host = address[0] if isinstance(address, tuple) else str(address)
    return host in ("127.0.0.1", "::1", "localhost")


def test_trained_detector_mode_runs_with_the_real_models():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    method = next(r for r in at.radio if r.label == "Detection method")
    method.set_value("Trained detector (experimental)").run()
    at.checkbox[0].check().run()
    at.file_uploader[0].upload("silence.wav", wav_bytes(np.zeros(5 * RATE)), "audio/wav").run()
    assert not at.exception
    assert "No intervention recommended" in recommendation(at)
    labels = set(at.table[1].value["Peak label"])
    assert labels <= {"-"} | {f"Trained detector: {c}" for c in ("clicking", "typing", "banging", "alarms")} | {
        "Chewing, mastication", "Speech", "Child speech, kid speaking", "Conversation",
        "Shout", "Yell", "Children shouting", "Hubbub, speech noise, speech babble",
    }
