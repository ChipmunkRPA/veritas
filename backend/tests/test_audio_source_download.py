import hashlib

import pytest

from scripts import fetch_audio_confirmation_sources_v7 as downloader


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, _):
        data, self.data = self.data, b""
        return data


def asset(data):
    return {
        "name": "source.tar",
        "url": "https://huggingface.co/example/source.tar",
        "bytes": len(data),
        "max_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_only_verified_complete_archive_is_published(tmp_path, monkeypatch):
    data = b"public source bytes"
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_a, **_k: Response(data))
    result = downloader.download(asset(data), tmp_path)
    assert result["status"] == "verified"
    assert (tmp_path / "source.tar").read_bytes() == data
    assert not (tmp_path / "source.tar.partial").exists()


def test_bad_checksum_preserves_partial_and_refuses_final_file(tmp_path, monkeypatch):
    data = b"expected source"
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_a, **_k: Response(b"changed source!"))
    with pytest.raises(ValueError):
        downloader.download(asset(data), tmp_path)
    assert not (tmp_path / "source.tar").exists()


def test_full_partial_is_checked_without_requesting_invalid_resume(tmp_path, monkeypatch):
    data = b"complete"
    (tmp_path / "source.tar.partial").write_bytes(data)

    def no_request(*_a, **_k):
        raise AssertionError("Completed source should not restart")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", no_request)
    downloader.download(asset(data), tmp_path)
    assert (tmp_path / "source.tar").read_bytes() == data


def test_wrong_resume_response_does_not_append_or_restart(tmp_path, monkeypatch):
    data = b"complete"
    (tmp_path / "source.tar.partial").write_bytes(data[:3])
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *_a, **_k: Response(data, status=200))
    with pytest.raises(ValueError, match="refused exact"):
        downloader.download(asset(data), tmp_path)
    assert (tmp_path / "source.tar.partial").read_bytes() == data[:3]


def test_unsafe_source_filename_is_refused(tmp_path):
    a = asset(b"abc")
    a["name"] = "../outside.tar"
    with pytest.raises(ValueError, match="Unsafe"):
        downloader.download(a, tmp_path)
