"""Print YAMNet's five highest-scoring classes for one audio file.

Usage: python scripts/yamnet_top5.py path/to/audio.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sensoryaware.audio import AudioLoadError, load_audio  # noqa: E402
from sensoryaware.model import YamnetClassifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path)
    args = parser.parse_args()

    try:
        waveform = load_audio(args.audio_file)
    except AudioLoadError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    predictions = YamnetClassifier().classify(waveform)
    print(f"{args.audio_file.name}: {predictions.num_frames} frames")
    for rank, item in enumerate(predictions.top_k(5), start=1):
        print(f"{rank}. {item.label:<40} {item.score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
