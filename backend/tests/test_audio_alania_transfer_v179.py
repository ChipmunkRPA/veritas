import shutil
import subprocess
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.engine.boundary import Deadline
from scripts import audio_alania_transfer_v179 as study


def test_parquet_extraction_admits_only_frozen_clean_slots(tmp_path):
    metadata = [
        {
            "id": f"synthetic-{i}",
            "duration": 2.85 + 0.1 * i,
            "voice_id": f"voice-{i}",
            "voice_type": "designed-identity",
            "render_mode": "reference",
            "text_source": "domain",
            "text_doc_id": f"document-{i}",
            "generator": study.roster.GENERATOR,
            "postfilter": "resclip",
            "license": "CC-BY-4.0",
        }
        for i in range(4)
    ]
    file = tmp_path / "authored.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    **row,
                    "audio": {"bytes": f"clean-{i}".encode()},
                    "audio_channel": {"bytes": f"alternate-{i}".encode()},
                }
                for i, row in enumerate(metadata)
            ]
        ),
        file,
        row_group_size=2,
    )
    selected = [{"slot": "positive-00", **metadata[3], "row_group": 1, "row_offset": 1}]
    with pq.ParquetFile(file) as table:
        reads = []

        def read_row_group(group, *, columns, use_threads):
            reads.append((group, columns))
            return table.read_row_group(group, columns=columns, use_threads=use_threads)

        proxy = types.SimpleNamespace(
            metadata=table.metadata, num_row_groups=table.num_row_groups, read_row_group=read_row_group
        )
        assert list(study.selected_payloads(proxy, selected)) == [(selected[0], {"bytes": b"clean-3"})]
        assert reads == [(1, list(study.roster.COLUMNS)), (1, ["audio"])]
        reads.clear()
        with pytest.raises(ValueError, match="Frozen selected Parquet metadata differs"):
            list(study.selected_payloads(proxy, [{**selected[0], "row_offset": 0}]))
        assert reads == [(1, list(study.roster.COLUMNS))]


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Production decoder tools absent"
)
def test_prepare_keeps_full_variable_native_clock_and_rejects_outside_bounds(tmp_path, monkeypatch):
    output = tmp_path / "prepared"
    output.mkdir()
    monkeypatch.setattr(study, "prepared_dir", lambda: output)
    scratch = tmp_path / "scratch.flac"
    wav = output / "positive-00.wav"
    selected = {"slot": "positive-00", "render_mode": "reference", "duration": 2.713}

    def fixture(duration):
        return subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=330:sample_rate=48000:duration={duration}",
                "-c:a",
                "flac",
                "-f",
                "flac",
                str(tmp_path / f"input-{duration}.flac"),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )

    fixture(2.713)
    payload = (tmp_path / "input-2.713.flac").read_bytes()
    row = study.prepare_case(selected, {"bytes": payload}, scratch, wav)
    assert row["status"] == "qualified"
    assert row["samples"] == 43408
    assert row["source_range"] == study.source_range(43408)
    assert len(study.read_pcm(row)) == 43408
    fixture(4.03)
    too_long = (tmp_path / "input-4.03.flac").read_bytes()
    unavailable = study.prepare_case(
        {**selected, "duration": 4.0}, {"bytes": too_long}, scratch, output / "long.wav"
    )
    assert unavailable["status"] == "unavailable"
    assert not (output / "long.wav").exists()


@pytest.mark.parametrize(
    ("samples", "raw", "direction"), [(40000, 0.99999, "ai"), (64000, 0.5, "inconclusive")]
)
def test_serving_route_uses_full_variable_window(samples, raw, direction):
    pcm = np.random.default_rng(179).normal(0, 0.05, samples).astype(np.float32)
    calls = []

    def predict(chunks):
        calls.extend(chunks)
        return [raw]

    model = types.SimpleNamespace(predict=predict)
    activity = types.SimpleNamespace(spans=lambda _: [(0, samples / study.SAMPLE_RATE)])
    result = study.score_voice(pcm, model, activity, Deadline(10))
    assert result["decision"] == direction
    assert result["windows"] == [[0, samples]]
    assert len(calls) == 1 and np.array_equal(calls[0], pcm)


def test_preparation_unavailable_and_vad_rejections_stay_in_positive_denominators():
    prepared = [
        {"slot": "a", "render_mode": "reference", "status": "qualified", "samples": 48010},
        {"slot": "b", "render_mode": "reference", "status": "unavailable"},
        {"slot": "c", "render_mode": "reference+instruction", "status": "qualified", "samples": 50000},
    ]
    rows = [
        {
            **prepared[0],
            "route_status": "ok",
            "windows": [[0, 48010]],
            "decision": "ai",
            "pella_forwards": 1,
            "stability_probe": "not_required_for_decision",
        },
        {
            **prepared[1],
            "route_status": "preparation_unavailable",
            "windows": [],
            "decision": "unavailable",
            "pella_forwards": 0,
            "stability_probe": "not_run",
        },
        {
            **prepared[2],
            "route_status": "not_applicable",
            "windows": [],
            "decision": "unavailable",
            "pella_forwards": 0,
            "stability_probe": "not_run",
        },
    ]
    result = study.score_summary(rows, prepared)
    assert result["all_positive_cases"]["missed_ai"] == 2
    assert result["all_positive_cases"]["scored_coverage_over_planned"] == 1 / 3
    assert result["by_render_mode"]["reference"]["missed_ai_rate_over_planned"] == 0.5
    assert result["by_render_mode"]["reference+instruction"]["unavailable_rate_over_planned"] == 1
