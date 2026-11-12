"""Create or replay an audio comparison using two independently supplied originals."""

from __future__ import annotations

from pathlib import Path

from app.evidence.audio_reference import compare
from scripts.compare_video_references import main, replay


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False) -> dict:
    return replay(compare, earlier, submitted, output, verify=verify)


if __name__ == "__main__":
    main(description=__doc__, runner=run)
