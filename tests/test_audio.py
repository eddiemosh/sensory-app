"""Audio loading (FR1, FR2). Clips are generated in temporary folders."""

import io

import numpy as np
import pytest
import soundfile as sf

from sensoryaware.audio import (
    LEVEL_FLOOR_DB,
    TARGET_SAMPLE_RATE,
    AudioLoadError,
    frame_levels_db,
    load_audio,
)


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


# --- Frame loudness (FR12) ---------------------------------------------------


def test_full_scale_sine_is_about_minus_3_dbfs():
    t = np.arange(TARGET_SAMPLE_RATE * 3) / TARGET_SAMPLE_RATE
    levels = frame_levels_db(np.sin(2 * np.pi * 440 * t).astype(np.float32), 5)
    assert levels == pytest.approx([-3.01] * 5, abs=0.05)


def test_silence_has_the_floor_level():
    assert frame_levels_db(np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32), 3).tolist() == [
        LEVEL_FLOOR_DB
    ] * 3


def test_levels_follow_yamnet_frame_timing():
    # Quiet for the first 2 s, loud after: frame i covers i*0.48 s to i*0.48+0.96 s.
    rate = TARGET_SAMPLE_RATE
    waveform = np.concatenate([np.full(2 * rate, 0.001), np.full(2 * rate, 0.5)]).astype(np.float32)
    levels = frame_levels_db(waveform, 7)
    assert levels[0] < -50          # 0.00-0.96 s, all quiet
    assert levels[5] == pytest.approx(20 * np.log10(0.5), abs=0.01)  # 2.40-3.36 s, all loud
    assert levels[0] < levels[3] < levels[5]  # 1.44-2.40 s straddles the change


def test_frames_past_the_end_get_the_floor_level():
    levels = frame_levels_db(np.full(TARGET_SAMPLE_RATE, 0.5, dtype=np.float32), 4)
    assert levels[-1] == LEVEL_FLOOR_DB
