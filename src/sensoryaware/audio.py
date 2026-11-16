"""Load audio and convert it to the format YAMNet expects."""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 300


class AudioLoadError(ValueError):
    """The audio could not be used; the message is suitable to show to a user."""


def load_audio(source) -> np.ndarray:
    """Read a path or file-like object into 16 kHz mono float32 samples in [-1, 1]."""
    try:
        with sf.SoundFile(source) as audio_file:
            sample_rate = audio_file.samplerate
            if audio_file.frames > MAX_DURATION_SECONDS * sample_rate:
                raise AudioLoadError(
                    f"The audio is too long; the limit is "
                    f"{MAX_DURATION_SECONDS // 60} minutes."
                )
            samples = audio_file.read(dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as error:
        raise AudioLoadError(
            "The audio could not be read. It may be corrupt or in an unsupported "
            "format; try a WAV, FLAC, OGG or MP3 file."
        ) from error

    if samples.size == 0:
        raise AudioLoadError("The file contains no audio.")
    if not np.isfinite(samples).all():
        raise AudioLoadError("The file contains invalid audio samples.")

    waveform = samples.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resample_poly(
            waveform, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor
        )
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


# YAMNet scores one frame every 0.48 s, each covering 0.96 s of audio.
FRAME_HOP = int(0.48 * TARGET_SAMPLE_RATE)
FRAME_LENGTH = int(0.96 * TARGET_SAMPLE_RATE)
LEVEL_FLOOR_DB = -100.0


def frame_levels_db(waveform: np.ndarray, num_frames: int) -> np.ndarray:
    """RMS loudness of each YAMNet frame, in dB relative to full scale (dBFS).

    0 dBFS is the loudest a recording can hold, so values are negative. They
    depend on the microphone's gain: they compare sounds within one setup
    rather than measuring sound pressure in the room.
    """
    levels = np.full(num_frames, LEVEL_FLOOR_DB)
    for i in range(num_frames):
        window = waveform[i * FRAME_HOP : i * FRAME_HOP + FRAME_LENGTH]
        if window.size:
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            if rms > 0:
                levels[i] = max(20 * np.log10(rms), LEVEL_FLOOR_DB)
    return levels
