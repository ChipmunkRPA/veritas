import io
import re
import zipfile

import pytest

from scripts import grid_clip_plan_v136 as planning


def remote_bytes(monkeypatch, payload, failure=None):
    calls = []

    class Response(io.BytesIO):
        status = 206 if failure != "ignored_range" else 200

    def open_request(request, **kwargs):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        calls.append((start, end))
        response = Response(payload[start:end] if failure == "short" else payload[start : end + 1])
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        if failure == "wrong_range":
            response.headers["Content-Range"] = "bytes 0-1/2"
        return response

    monkeypatch.setattr(planning, "urlopen", open_request)
    return calls


def test_real_zip_directory_is_read_with_bounded_ranges_without_opening_media_entries(monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED) as archive:
        for i in range(1000):
            archive.writestr(f"source/utterance{i:04d}.mpg", b"authored" * 200)
            archive.writestr(f"__MACOSX/source/._utterance{i:04d}.mpg", b"resource fork")
    payload = data.getvalue()
    calls = remote_bytes(monkeypatch, payload)
    media, count, transferred = planning.directory(
        {"url": "https://example.invalid/authored", "bytes": len(payload)}
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected = [
            {"name": r.filename, "bytes": r.file_size, "crc32": r.CRC}
            for r in archive.infolist()
            if not r.filename.startswith("__MACOSX/")
        ]
    assert media == expected and count == len(calls) <= 8
    assert transferred == sum(b - a + 1 for a, b in calls) < len(payload) / 4
    assert min(a for a, b in calls) > len(payload) / 2


@pytest.mark.parametrize("failure", ["ignored_range", "wrong_range", "short"])
def test_directory_reader_rejects_unbound_or_incomplete_responses_without_retry(monkeypatch, failure):
    calls = remote_bytes(monkeypatch, b"abcdefgh", failure)
    reader = planning.DirectoryReader({"url": "https://example.invalid", "bytes": 8})
    with pytest.raises(ValueError):
        reader.read(4)
    assert len(calls) == 1 and reader.tell() == 0


def test_reader_caches_exact_ranges_and_does_not_allow_unbounded_full_asset_read(monkeypatch):
    calls = remote_bytes(monkeypatch, b"abcdefgh")
    reader = planning.DirectoryReader({"url": "https://example.invalid", "bytes": 8})
    assert reader.read(6) == b"abcdef"
    reader.seek(2)
    assert reader.read(3) == b"cde" and len(calls) == 1
    reader.seek(-2, 2)
    assert reader.read() == b"gh"
    reader.seek(0)
    reader.row["bytes"] = 3 * 1024**2
    with pytest.raises(ValueError, match="Bounded"):
        reader.read()
    assert len(calls) == 2


def test_full_clip_plan_preserves_all_sources_and_prevents_repeated_utterances():
    rows = [
        {"source": {"ordinal": i}, "media": [{"name": f"source{i}/utterance{n:03d}.mpg"} for n in range(100)]}
        for i in range(16)
    ]
    selected = planning.clip_plan(rows)
    assert len(selected) == 32 and {r["source_ordinal"] for r in selected} == set(range(16))
    assert len({r["member"]["name"].split("/")[-1] for r in selected}) == 32
    assert all(r["study_role"] == "external_transfer" and not r["media_qualified"] for r in selected)
    assert planning.clip_plan([{**r, "media": list(reversed(r["media"]))} for r in rows]) == selected
    with pytest.raises(ValueError):
        planning.clip_plan(list(reversed(rows)))


def test_neural_commands_cannot_be_dispatched():
    with pytest.raises(ValueError):
        planning.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored"])
