"""Create or replay an audio comparison using two independently supplied originals."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from app.evidence.audio_reference import (
    SCHEMA,
    SCHEMA_WITH_AAC_CLEAN_SPEECH,
    SCHEMA_WITH_AAC_TIMING,
    SCHEMA_WITH_ALIGNED_SPEECH,
    SCHEMA_WITH_CHANNEL_TIMING,
    SCHEMA_WITH_CLEAN_SPEECH,
    SCHEMA_WITH_CLEAN_SPEECH_EQ_GUARD,
    SCHEMA_WITH_CLEAN_SPEECH_SUPPORT,
    SCHEMA_WITH_CLOSE_STEREO_AAC,
    SCHEMA_WITH_DENSE_CLEAN_SPEECH,
    SCHEMA_WITH_DENSER_STEREO_AAC,
    SCHEMA_WITH_LEVEL_ADJUSTMENT,
    SCHEMA_WITH_LOCAL_TIMING,
    SCHEMA_WITH_LONG_AAC_CLEAN_SPEECH,
    SCHEMA_WITH_LONG_CLEAN_SPEECH,
    SCHEMA_WITH_MATCHING_WAVEFORM,
    SCHEMA_WITH_TRIM_ALIGNMENT,
    SCHEMA_WITH_TRIM_AND_LEVEL,
    STEREO_AAC_SPEECH_SCHEMAS,
    compare,
)
from scripts.compare_video_references import main, replay


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False, clean_speech: bool = False) -> dict:
    compare_source = partial(compare, include_clean_speech=True) if clean_speech else compare
    eq_schemas = {SCHEMA_WITH_CLEAN_SPEECH_EQ_GUARD, SCHEMA_WITH_CLEAN_SPEECH_SUPPORT}
    if verify:
        exported = json.loads((output if output.is_file() else output / "comparison.json").read_bytes())
        record = exported.get("comparison", exported) if isinstance(exported, dict) else None
        if not isinstance(record, dict):
            raise ValueError("The comparison record must be a JSON object.")
        schema = record.get("schema")
        waveform = schema == SCHEMA_WITH_MATCHING_WAVEFORM
        aligned = schema == SCHEMA_WITH_ALIGNED_SPEECH or waveform and "aligned_speech" in record
        if aligned or waveform:
            schema = record.get("comparison_schema")
        if schema in {SCHEMA, SCHEMA_WITH_LEVEL_ADJUSTMENT, SCHEMA_WITH_TRIM_ALIGNMENT}:
            compare_source = partial(compare, include_level_adjustment=schema != SCHEMA,
                                     include_trim_alignment=schema == SCHEMA_WITH_TRIM_ALIGNMENT,
                                     include_trim_level_adjustment=False, include_stereo=False)
        elif schema == SCHEMA_WITH_TRIM_AND_LEVEL:
            compare_source = partial(compare, include_stereo=False)
        elif schema == SCHEMA_WITH_LOCAL_TIMING:
            compare_source = partial(compare, include_local_timing=True)
        elif schema == SCHEMA_WITH_CHANNEL_TIMING:
            compare_source = partial(compare, include_local_timing=True, include_channel_timing=True)
        elif schema == SCHEMA_WITH_AAC_TIMING:
            compare_source = partial(compare, include_local_timing=True, include_aac_timing=True,
                                     include_clean_speech="clean_speech" in record,
                                     allow_long_clean_speech=record.get("local_timing", {}).get("twelve_second_speech_option", True))
        elif schema in {SCHEMA_WITH_AAC_CLEAN_SPEECH, SCHEMA_WITH_LONG_AAC_CLEAN_SPEECH, *STEREO_AAC_SPEECH_SCHEMAS}:
            compare_source = partial(compare, include_clean_speech=True, include_clean_speech_aac=True,
                                     allow_long_clean_speech=schema != SCHEMA_WITH_AAC_CLEAN_SPEECH,
                                     include_clean_speech_stereo_aac=schema in STEREO_AAC_SPEECH_SCHEMAS,
                                     allow_denser_stereo_windows=schema == SCHEMA_WITH_DENSER_STEREO_AAC,
                                     allow_close_stereo_consensus=schema == SCHEMA_WITH_CLOSE_STEREO_AAC,
                                     include_local_timing="local_timing" in record,
                                     include_aac_timing="aac_presentation" in record.get("local_timing", {}),
                                     include_channel_timing="channel_timing" in record)
        elif schema in {
            SCHEMA_WITH_CLEAN_SPEECH, SCHEMA_WITH_LONG_CLEAN_SPEECH, SCHEMA_WITH_DENSE_CLEAN_SPEECH,
            *eq_schemas,
        }:
            compare_source = partial(compare, include_clean_speech=True, include_local_timing="local_timing" in record,
                                     include_channel_timing="channel_timing" in record,
                                     allow_long_clean_speech=schema == SCHEMA_WITH_LONG_CLEAN_SPEECH,
                                     use_dense_clean_speech=schema in {SCHEMA_WITH_DENSE_CLEAN_SPEECH, *eq_schemas},
                                     use_clean_speech_eq_guard=schema in eq_schemas,
                                     include_clean_speech_match_support=schema == SCHEMA_WITH_CLEAN_SPEECH_SUPPORT)
        if aligned or waveform:
            compare_source = partial(compare_source, include_aligned_speech=aligned, include_matching_waveform=waveform)
    return replay(compare_source, earlier, submitted, output, verify=verify)


if __name__ == "__main__":
    main(description=__doc__, runner=run, clean_audio_option=True)
