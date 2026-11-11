import io
import tarfile

import pytest

from scripts import audio_pacc_sample_v176 as panel


def test_matched_params_require_equal_native_clock_and_measured_level():
    source = {"file_id": "sample", "pool": "AMI", "source_sample_rate": "16000"}
    params = {
        "file_id": "sample",
        "pool": "AMI",
        "condition": "babble_snr05",
        "sr_in": "16000",
        "sr_out": "16000",
        "n_in": "49000",
        "n_out": "49000",
        "target_snr_db": "5",
        "achieved_snr_db": "5.0",
        "written_clipped_samples": "0",
        "output": "babble_snr05/AMI/sample.flac",
    }
    assert panel.aligned_params(params, "babble_snr05", source, 5) == "pacc-p/babble_snr05/AMI/sample.flac"
    for key, changed in (
        ("sr_out", "48000"),
        ("n_out", "48999"),
        ("achieved_snr_db", "4"),
        ("written_clipped_samples", "1"),
    ):
        with pytest.raises(ValueError, match="sample clock"):
            panel.aligned_params({**params, key: changed}, "babble_snr05", source, 5)
    with pytest.raises(ValueError, match="publisher path"):
        panel.aligned_params({**params, "output": "other/AMI/sample.flac"}, "babble_snr05", source, 5)


def test_reads_regular_members_without_following_archive_links():
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("normal")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"a"))
        link = tarfile.TarInfo("linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "normal"
        archive.addfile(link)
    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r") as archive:
        assert panel.member(archive, "normal") == b"a"
        with pytest.raises(ValueError, match="regular archive member"):
            panel.member(archive, "linked")
        with pytest.raises(ValueError, match="Unsafe archive path"):
            panel.member(archive, "../normal")


def test_aggregate_keeps_unavailable_rows_and_counts_shared_context_once():
    plan = [
        {"slot": slot, "family": "VCTK", "source_context": "same", "known_historical_voice": True}
        for slot in ("left", "right")
    ]
    rows = [
        {
            "slot": row["slot"],
            "family": "VCTK",
            "condition": condition,
            "status": "unavailable" if row["slot"] == "right" else "qualified",
        }
        for row in plan
        for condition in panel.CONDITIONS
    ]
    result = panel.aggregate(plan, rows)
    assert result["declared_source_contexts"] == {"VCTK": 1}
    assert result["inputs_with_historical_voice_match"] == 2
    assert result["outcomes_by_source_family_and_condition"]["VCTK"]["clean"] == {
        "qualified": 1,
        "unavailable": 1,
    }
    with pytest.raises(ValueError, match="one ordered receipt"):
        panel.aggregate(plan, rows[:-1])
