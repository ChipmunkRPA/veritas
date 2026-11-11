import hashlib
import io

import pytest

from scripts import grid_acquisition_continuation_v135 as acquisition


def test_new_archive_assembly_verifies_whole_file_and_preserves_prefix_and_tail(tmp_path):
    original = tmp_path / "original-partial"
    original.write_bytes(b"preserved-prefix")
    prefix = {
        "file": str(original),
        "bytes": original.stat().st_size,
        "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    tail = tmp_path / "new-tail"
    tail.write_bytes(b"new-tail")
    whole = original.read_bytes() + tail.read_bytes()
    expected = {"bytes": len(whole), "md5": hashlib.md5(whole).hexdigest()}
    result = acquisition.assemble(prefix, tail, tmp_path / "new-archive", expected)
    assert result["sha256"] == hashlib.sha256(whole).hexdigest()
    assert original.read_bytes() == b"preserved-prefix" and tail.read_bytes() == b"new-tail"
    with pytest.raises(FileExistsError):
        acquisition.assemble(prefix, tail, tmp_path / "new-archive", expected)
    with pytest.raises(ValueError, match="checksum"):
        acquisition.assemble(prefix, tail, tmp_path / "failed-archive", {**expected, "md5": "0" * 32})
    assert (tmp_path / "failed-archive").read_bytes() == whole
    original.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="prefix"):
        acquisition.assemble(prefix, tail, tmp_path / "never-created", expected)
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("failure", [None, "full_response", "wrong_range", "short", "oversized", "encoded"])
def test_exact_missing_range_required_without_fallback_or_retry(tmp_path, monkeypatch, failure):
    requests = []

    class Response(io.BytesIO):
        status = 200 if failure == "full_response" else 206
        headers = {"Content-Range": "bytes 5-7/8" if failure == "wrong_range" else "bytes 4-7/8"}
        if failure == "encoded":
            headers["Content-Encoding"] = "gzip"

    def open_request(request, **kwargs):
        requests.append(request)
        return Response(b"xy" if failure == "short" else b"oversized" if failure == "oversized" else b"tail")

    monkeypatch.setattr(acquisition, "urlopen", open_request)
    target = tmp_path / "tail"
    row = {"url": "https://example.invalid/pinned", "bytes": 8}
    if failure is None:
        assert acquisition.download_tail(row, 4, target) == 4
        assert target.read_bytes() == b"tail"
    else:
        with pytest.raises(ValueError):
            acquisition.download_tail(row, 4, target)
        assert target.exists()
    assert len(requests) == 1 and requests[0].get_header("Range") == "bytes=4-7"
    with pytest.raises(FileExistsError):
        acquisition.download_tail(row, 4, target)
    assert len(requests) == 1


def test_exact_data_dispatch_and_original_source_ordinals_only(monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    output = acquisition.TRIAL / "source-01/bind-00/output"
    argv = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"]
    acquisition.routed_popen(1, argv)
    assert calls[0][3:] == ["scripts.grid_acquisition_continuation_v135", "acquire", str(output), "1"]
    for ordinal in (True, 0, 16):
        with pytest.raises(ValueError):
            acquisition.run(ordinal)
    with pytest.raises(ValueError):
        acquisition.routed_popen(1, [*argv[:4], "authored", *argv[5:]])
