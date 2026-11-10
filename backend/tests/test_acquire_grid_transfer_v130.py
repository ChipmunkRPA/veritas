import hashlib
import io
import zipfile

import pytest

from scripts import acquire_grid_transfer_v130 as acquisition


def publisher_record():
    files = [
        {
            "key": f"s{i}.zip",
            "size": 10000,
            "checksum": "md5:" + "a" * 32,
            "links": {"self": f"https://zenodo.org/api/files/example/s{i}.zip"},
        }
        for i in range(1, 35)
        if i != 21
    ]
    files.append(
        {
            "key": "jasagrid.pdf",
            "size": 20000,
            "checksum": "md5:" + "b" * 32,
            "links": {"self": "https://zenodo.org/api/files/example/jasagrid.pdf"},
        }
    )
    return {
        "id": 3625687,
        "metadata": {"license": {"id": "cc-by-4.0"}, "access_right": "open"},
        "files": files,
    }


def test_source_selection_is_fixed_before_any_media_content_or_response():
    record = publisher_record()
    plan = acquisition.select(record)
    record["files"].reverse()
    assert acquisition.select(record) == plan
    assert len(plan["selected"]) == 16 and len({r["publisher_name"] for r in plan["selected"]}) == 16
    assert [r["ordinal"] for r in plan["selected"]] == list(range(16))


@pytest.mark.parametrize(
    "problem", ["noncommercial", "unknown_license", "closed", "unbounded", "foreign_link"]
)
def test_unlicensed_or_unbound_sources_are_rejected(problem):
    record = publisher_record()
    if problem == "noncommercial":
        record["metadata"]["license"]["id"] = "cc-by-nc-nd-4.0"
    elif problem == "unknown_license":
        record["metadata"]["license"]["id"] = "unknown"
    elif problem == "closed":
        record["metadata"]["access_right"] = "restricted"
    elif problem == "unbounded":
        record["files"][0]["size"] = 700 * 1024**2
    else:
        record["files"][0]["links"]["self"] = "https://example.com/api/files/foreign.zip"
    with pytest.raises(ValueError):
        acquisition.select(record)


def test_archive_inventory_ignores_macos_resource_forks_without_extracting_them(tmp_path):
    file = tmp_path / "authored.zip"
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i in range(1000):
            archive.writestr(f"source/{i:04d}.mpg", b"x" * 1000)
            archive.writestr(f"__MACOSX/source/._{i:04d}.mpg", b"not a video")
    media = acquisition.inspect_archive(file)
    assert len(media) == 1000 and all("__MACOSX" not in r["name"] for r in media)
    assert not (tmp_path / "source").exists()
    with zipfile.ZipFile(file, "a") as archive:
        archive.writestr("../escape.mpg", b"x" * 1000)
    with pytest.raises(ValueError, match="Unsafe"):
        acquisition.inspect_archive(file)


def test_checksum_verified_download_retains_failed_or_partial_attempts(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200

    row = {"url": "https://example.invalid/authored", "bytes": 4, "md5": hashlib.md5(b"test").hexdigest()}
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: Response(b"test"))
    file = tmp_path / "good.bin"
    result = acquisition.download(row, file)
    assert result["sha256"] == hashlib.sha256(b"test").hexdigest() and file.read_bytes() == b"test"
    wrong = {**row, "md5": "0" * 32}
    failed = tmp_path / "failed.bin"
    with pytest.raises(ValueError, match="checksum"):
        acquisition.download(wrong, failed)
    assert failed.read_bytes() == b"test"
    with pytest.raises(FileExistsError):
        acquisition.download(row, failed)


def test_oversized_response_is_stopped_and_not_silently_truncated(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: Response(b"too long"))
    with pytest.raises(ValueError, match="size"):
        acquisition.download(
            {"url": "https://example.invalid", "bytes": 4, "md5": "0" * 32}, tmp_path / "oversized.bin"
        )


def test_invalid_or_neural_stage_is_not_dispatched():
    with pytest.raises(ValueError):
        acquisition.run(True)
    with pytest.raises(ValueError):
        acquisition.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
