"""Create or replay an audio comparison using two independently supplied originals."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from app.evidence.audio_reference import (
    SCHEMA,
    SCHEMA_WITH_CHANNEL_TIMING,
    SCHEMA_WITH_CLEAN_SPEECH,
    SCHEMA_WITH_LEVEL_ADJUSTMENT,
    SCHEMA_WITH_LOCAL_TIMING,
    SCHEMA_WITH_TRIM_ALIGNMENT,
    SCHEMA_WITH_TRIM_AND_LEVEL,
    compare,
)
from scripts.compare_video_references import main, replay


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False, clean_speech: bool = False) -> dict:
    compare_source = partial(compare, include_clean_speech=True) if clean_speech else compare
    if verify:
        exported = json.loads((output if output.is_file() else output / "comparison.json").read_bytes())
        record = exported.get("comparison", exported) if isinstance(exported, dict) else None
        if isinstance(record, dict) and record.get("schema") in {SCHEMA, SCHEMA_WITH_LEVEL_ADJUSTMENT, SCHEMA_WITH_TRIM_ALIGNMENT}:
            compare_source = partial(compare, include_level_adjustment=record["schema"] != SCHEMA,
                                     include_trim_alignment=record["schema"] == SCHEMA_WITH_TRIM_ALIGNMENT,
                                     include_trim_level_adjustment=False, include_stereo=False)
        elif isinstance(record, dict) and record.get("schema") == SCHEMA_WITH_TRIM_AND_LEVEL:
            compare_source = partial(compare, include_stereo=False)
        elif isinstance(record, dict) and record.get("schema") == SCHEMA_WITH_LOCAL_TIMING:
            compare_source = partial(compare, include_local_timing=True)
        elif isinstance(record, dict) and record.get("schema") == SCHEMA_WITH_CHANNEL_TIMING:
            compare_source = partial(compare, include_local_timing=True, include_channel_timing=True)
        elif isinstance(record, dict) and record.get("schema") == SCHEMA_WITH_CLEAN_SPEECH:
            compare_source = partial(compare, include_clean_speech=True, include_local_timing="local_timing" in record,
                                     include_channel_timing="channel_timing" in record)
    return replay(compare_source, earlier, submitted, output, verify=verify)


if __name__ == "__main__":
    main(description=__doc__, runner=run, clean_audio_option=True)
