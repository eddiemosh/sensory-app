"""Run the app's full pipeline on a folder of your own recordings.

Usage: python scripts/analyse_recordings.py [folder]   (default: data/recordings)

Each file is analysed exactly as the app does it (default settings), and the
results are written to docs/evaluation/real-recordings.md for the report. The
recordings themselves stay in data/, which git ignores. Name each file after
what it contains, for example typing.wav or loud-speech.wav.
"""

import dataclasses
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sensoryaware.audio import AudioLoadError, frame_levels_db, load_audio  # noqa: E402
from sensoryaware.config import DEFAULT_CONFIG  # noqa: E402
from sensoryaware.decision import decide  # noqa: E402
from sensoryaware.model import YamnetClassifier  # noqa: E402

RESULTS = ROOT / "docs" / "evaluation" / "real-recordings.md"
AUDIO_TYPES = {".wav", ".flac", ".ogg", ".mp3"}


def speech_levels(predictions, threshold):
    """Loudness of the frames where any speech label reaches the speech threshold."""
    speech = next(c for c in DEFAULT_CONFIG.categories if c.id == "speech")
    columns = [predictions.class_names.index(label) for label in speech.labels]
    confident = predictions.scores[:, columns].max(axis=1) >= threshold
    return predictions.levels_db[confident]


def analyse(path, classifier):
    waveform = load_audio(path)
    predictions = classifier.classify(waveform)
    predictions = dataclasses.replace(
        predictions, levels_db=frame_levels_db(waveform, predictions.num_frames)
    )
    decision = decide(predictions, DEFAULT_CONFIG)
    speech = next(c for c in DEFAULT_CONFIG.categories if c.id == "speech")
    levels = speech_levels(predictions, DEFAULT_CONFIG.score_threshold(speech))
    return {
        "file": path.name,
        "seconds": len(waveform) / 16000,
        "frames": predictions.num_frames,
        "top3": ", ".join(f"{r.label} {r.score:.2f}" for r in predictions.top_k(3)),
        "evidence": decision.evidence,
        "recommendation": decision.intervention.name if decision.intervention else "None",
        "loudness": (float(np.median(predictions.levels_db)), float(predictions.levels_db.max())),
        "speech_frames": len(levels),
        "speech_level": f"{np.median(levels):.0f}" if len(levels) else "-",
    }


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "recordings"
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_TYPES)
    if not files:
        raise SystemExit(f"No WAV, FLAC, OGG or MP3 files in {folder}")

    classifier = YamnetClassifier()
    rows = []
    for path in files:
        try:
            rows.append(analyse(path, classifier))
        except AudioLoadError as error:
            print(f"Skipped {path.name}: {error}")

    names = [c.name for c in DEFAULT_CONFIG.categories]
    lines = [
        "# Real-recording test",
        "",
        f"Run on {date.today().isoformat()} with the default settings: score "
        f"threshold {DEFAULT_CONFIG.min_score}, {DEFAULT_CONFIG.min_frames} frames; "
        "loud speech at score 0.6 and -20 dBFS. Produced by "
        "`scripts/analyse_recordings.py`; the recordings are not committed.",
        "",
        "## Recommendation per recording",
        "",
        "| Recording | Length (s) | Recommendation | Top 3 YAMNet labels (mean score) |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['seconds']:.1f} | {r['recommendation']} | {r['top3']} |"
        )
    lines += [
        "",
        "## Frames detected per category",
        "",
        "Frames that met the category's rule, out of the total. A category "
        f"triggers at {DEFAULT_CONFIG.min_frames} or more. * = triggered.",
        "",
        "| Recording | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for r in rows:
        cells = [
            f"{e.frames_detected}/{e.total_frames}{' *' if e.triggered else ''}"
            for e in r["evidence"]
        ]
        lines.append(f"| {r['file']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Loudness (dBFS)",
        "",
        "Median and loudest frame of each recording, and the median loudness of "
        "frames where speech scored 0.6 or more. Use these to judge whether "
        "-20 dBFS separates normal from loud speech on this microphone.",
        "",
        "| Recording | Median frame | Loudest frame | Confident speech frames | Their median |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        median, loudest = r["loudness"]
        lines.append(
            f"| {r['file']} | {median:.0f} | {loudest:.0f} | "
            f"{r['speech_frames']} | {r['speech_level']} |"
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
