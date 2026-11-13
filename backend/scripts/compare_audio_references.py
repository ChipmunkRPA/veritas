"""Create or replay an audio comparison using two independently supplied originals."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from app.evidence.audio_reference import (
    SCHEMA,
    SCHEMA_WITH_LEVEL_ADJUSTMENT,
    SCHEMA_WITH_TRIM_ALIGNMENT,
    compare,
)
from scripts.compare_video_references import main, replay


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False) -> dict:
    compare_source = compare
    if verify:
        exported = json.loads((output if output.is_file() else output / "comparison.json").read_bytes())
        record = exported.get("comparison", exported) if isinstance(exported, dict) else None
        if isinstance(record, dict) and record.get("schema") in {SCHEMA, SCHEMA_WITH_LEVEL_ADJUSTMENT, SCHEMA_WITH_TRIM_ALIGNMENT}:
            compare_source = partial(compare, include_level_adjustment=record["schema"] != SCHEMA,
                                     include_trim_alignment=record["schema"] == SCHEMA_WITH_TRIM_ALIGNMENT,
                                     include_trim_level_adjustment=False)
    return replay(compare_source, earlier, submitted, output, verify=verify)


if __name__ == "__main__":
    main(description=__doc__, runner=run)
