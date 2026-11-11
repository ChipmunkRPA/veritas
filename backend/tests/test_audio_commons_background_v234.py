"""Checks for the frozen Commons intake, with no live source requests."""

from __future__ import annotations

import hashlib
import io
import shutil
import wave
from dataclasses import asdict, replace
from email.message import Message
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import audio_commons_background_v234 as study

TEST_SOURCES = tuple(
    study.Source(
        f"Test recording {slot}.oga",
        slot,
        100 + slot,
        "2020-01-01T12:00:00Z",
        f"Test recordist {slot}",
        environment,
        license_name,
        "Authored test fixture",
        1234 + slot,
        hashlib.sha1(str(slot).encode()).hexdigest(),
        "application/ogg",
    )
    for slot, (environment, license_name) in enumerate(
        zip(
            ("traffic", "kitchen", "kitchen", "kitchen", "kitchen", "traffic", "traffic"),
            (
                "CC-BY-SA-4.0",
                "Public domain",
                "CC-BY-SA-4.0",
                "CC0-1.0",
                "Public domain",
                "CC-BY-SA-4.0",
                "CC0-1.0",
            ),
            strict=True,
        ),
        1,
    )
)


class PublisherResponse(io.BytesIO):
    status = 200

    def __init__(self, source, payload, url=None):
        super().__init__(payload)
        self.url = url or source.url
        self.headers = Message()
        self.headers["Content-Type"] = source.content_type
        self.headers["Content-Length"] = str(source.size)

    def geturl(self):
        return self.url


@pytest.fixture
def private_roster(tmp_path, monkeypatch):
    path = tmp_path / "private-roster.json"
    data = {"schema": study.PRIVATE_ROSTER_SCHEMA, "sources": [asdict(source) for source in TEST_SOURCES]}
    study.save(path, data)
    path.chmod(0o600)
    monkeypatch.setattr(study, "PRIVATE_ROSTER", path)
    monkeypatch.setattr(study, "PRIVATE_ROSTER_SHA256", study.digest(path))
    return path


def test_private_roster_has_distinct_recording_groups_and_exact_original_urls(private_roster):
    sources = study.load_sources()
    assert sources == TEST_SOURCES
    assert sum(source.environment == "kitchen" for source in sources) == 4
    assert len({source.author for source in sources}) == 7
    assert sources[0].url == "https://upload.wikimedia.org/wikipedia/commons/c/c3/Test_recording_1.oga"
    assert sources[1].revision_url == "https://commons.wikimedia.org/w/index.php?oldid=102"
    assert study.NoRedirect().redirect_request(None, None, 302, "", {}, sources[0].url) is None
    summary = study.identity(sources)
    assert "sources" not in summary and sources[0].title not in str(summary)
    assert summary["private_roster_sha256"] == study.digest(private_roster)


def test_private_roster_permissions_schema_and_bytes_are_required(private_roster, monkeypatch):
    private_roster.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        study.load_sources()
    private_roster.chmod(0o600)
    original = private_roster.read_text()
    private_roster.write_text(original.replace(study.PRIVATE_ROSTER_SCHEMA, "wrong-schema"))
    with pytest.raises(ValueError, match="bytes differ"):
        study.load_sources()
    monkeypatch.setattr(study, "PRIVATE_ROSTER_SHA256", study.digest(private_roster))
    with pytest.raises(ValueError, match="schema required"):
        study.load_sources()


def test_bounded_publisher_transfer_binds_two_digests(tmp_path):
    payload = b"deterministic test bytes"
    source = replace(TEST_SOURCES[0], size=len(payload), publisher_sha1=hashlib.sha1(payload).hexdigest())
    target = tmp_path / "original.oga"

    def open_exact(req, timeout):
        assert req.full_url == source.url
        assert req.get_header("Accept-encoding") == "identity"
        assert timeout == study.SOCKET_TIMEOUT
        return PublisherResponse(source, payload)

    receipt = study.download(source, target, SimpleNamespace(open=open_exact), float("inf"))
    assert target.read_bytes() == payload
    assert receipt == {
        "bytes": len(payload),
        "publisher_sha1": source.publisher_sha1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    study.verify_original(source, target, receipt)
    target.write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="Saved original bytes differ"):
        study.verify_original(source, target, receipt)


@pytest.mark.parametrize(
    ("body", "wrong_url", "reason"),
    [
        (b"bad", False, "unexpected_body_length"),
        (b"more!", False, "unexpected_body_length"),
        (b"same", False, "publisher_checksum_mismatch"),
        (b"good", True, "unexpected_publisher_response"),
    ],
)
def test_bad_source_bytes_or_route_never_create_an_original(tmp_path, body, wrong_url, reason):
    source = replace(TEST_SOURCES[0], size=4, publisher_sha1=hashlib.sha1(b"good").hexdigest())
    target = tmp_path / "original.oga"
    response = PublisherResponse(source, body, url="https://example.test/redirect" if wrong_url else None)
    opener = SimpleNamespace(open=lambda req, timeout: response)
    with pytest.raises(study.Rejected, match=reason):
        study.download(source, target, opener, float("inf"))
    assert not target.exists()


@pytest.fixture
def empty_study(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    cache = tmp_path / "archive"
    trial.mkdir()
    cache.mkdir()
    monkeypatch.setattr(study, "TRIAL", trial)
    monkeypatch.setattr(study, "CACHE", cache)
    monkeypatch.setattr(study, "headroom", lambda: None)
    monkeypatch.setattr(study, "load_sources", lambda: TEST_SOURCES)
    monkeypatch.setattr(study, "verify", lambda: ("x" * 64, study.load_sources()))
    return trial


def test_rate_limit_ends_serial_run_and_keeps_other_slots_unattempted(empty_study, monkeypatch):
    requests = []

    def rate_limit(source, _target, _opener, _deadline):
        requests.append(source.url)
        raise HTTPError(source.url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(study, "download", rate_limit)
    result = study.acquire()
    assert requests == [TEST_SOURCES[0].url]
    assert result["acquisition_rejection_counts"] == {"publisher_access_or_rate_limit": 1}
    assert result["counts_by_environment"]["traffic"]["acquisition_unattempted"] == 2
    assert result["counts_by_environment"]["kitchen"]["acquisition_unattempted"] == 4
    assert result["terminal_stages"] == {"acquisition": True, "qualification": False}
    with pytest.raises(FileExistsError):
        study.acquire()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="local decoder tools required"
)
def test_exact_local_pcm_remains_pending_auditory_review(empty_study, monkeypatch):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x07\x02\xfa\xfd" * 24000)
    payload = wav.getvalue()
    source = replace(
        TEST_SOURCES[0],
        title="test.wav",
        size=len(payload),
        publisher_sha1=hashlib.sha1(payload).hexdigest(),
    )
    monkeypatch.setattr(study, "load_sources", lambda: (source,))
    target, _ = study.asset_paths(1, source)
    target.write_bytes(payload)
    lock, _ = study.verify()
    study.save(
        empty_study / "acquisition-01.json",
        {
            "ordinal": 1,
            "freeze_sha256": lock,
            "status": "verified_original",
            "bytes": len(payload),
            "publisher_sha1": source.publisher_sha1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    study.save(empty_study / "acquisition-end.json", {"freeze_sha256": lock})
    result = study.qualify()
    assert result["counts_by_environment"]["traffic"]["verified_pcm"] == 1
    assert (
        result["auditory_reviewed"] == result["admitted_for_model_evaluation"] == result["model_calls"] == 0
    )
    assert study.read(empty_study / "qualification-01.json")["samples"] == 48000
    assert "test.wav" not in str(result)
    assert source.publisher_sha1 not in str(result)
