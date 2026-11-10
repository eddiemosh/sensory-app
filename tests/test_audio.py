"""Audio loading (FR1, FR2). Clips are generated in temporary folders."""

import io

import numpy as np
import pytest
import soundfile as sf

from sensoryaware.audio import TARGET_SAMPLE_RATE, AudioLoadError, load_audio


def write_wav(path, samples, sample_rate):
    sf.write(path, samples, sample_rate, format="WAV")
    return path


def test_16khz_mono_is_unchanged(tmp_path):
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    waveform = load_audio(write_wav(tmp_path / "a.wav", samples, 16000))
    assert waveform.dtype == np.float32
    assert waveform == pytest.approx(samples, abs=1e-3)


def test_higher_sample_rate_is_resampled_with_same_duration(tmp_path):
    samples = np.zeros(44100 * 2, dtype=np.float32)
    waveform = load_audio(write_wav(tmp_path / "a.wav", samples, 44100))
    assert len(waveform) == TARGET_SAMPLE_RATE * 2


def test_resampling_preserves_frequency(tmp_path):
    times = np.arange(44100) / 44100
    samples = (0.5 * np.sin(2 * np.pi * 1000 * times)).astype(np.float32)
    waveform = load_audio(write_wav(tmp_path / "a.wav", samples, 44100))
    spectrum = np.abs(np.fft.rfft(waveform))
    assert np.argmax(spectrum) == pytest.approx(1000, abs=2)


def test_stereo_is_averaged_to_mono(tmp_path):
    left = np.full(16000, 0.5, dtype=np.float32)
    right = np.zeros(16000, dtype=np.float32)
    path = write_wav(tmp_path / "a.wav", np.column_stack([left, right]), 16000)
    waveform = load_audio(path)
    assert waveform.ndim == 1
    assert waveform == pytest.approx(np.full(16000, 0.25), abs=1e-3)


def test_loads_from_file_like_object(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    assert len(load_audio(io.BytesIO(path.read_bytes()))) == 16000


def test_samples_are_limited_to_model_range(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.full(100, 2.0, dtype=np.float32), 16000, subtype="FLOAT")
    assert load_audio(path).max() <= 1.0


def test_corrupt_file_gives_useful_error(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"this is not audio")
    with pytest.raises(AudioLoadError, match="could not be read"):
        load_audio(path)


def test_missing_file_gives_useful_error(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "missing.wav")


def test_empty_audio_gives_useful_error(tmp_path):
    path = write_wav(tmp_path / "a.wav", np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(AudioLoadError, match="no audio"):
        load_audio(path)


def test_overlong_audio_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("sensoryaware.audio.MAX_DURATION_SECONDS", 1)
    path = write_wav(tmp_path / "a.wav", np.zeros(32000, dtype=np.float32), 16000)
    with pytest.raises(AudioLoadError, match="too long"):
        load_audio(path)


def test_non_finite_samples_are_rejected(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.full(100, np.nan, dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(AudioLoadError, match="invalid"):
        load_audio(path)
