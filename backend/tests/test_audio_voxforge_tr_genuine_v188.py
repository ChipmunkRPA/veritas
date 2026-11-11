import gzip
import io
import shutil
import struct
import subprocess
import tarfile
import wave

import pytest

from scripts import audio_voxforge_tr_genuine_v188 as study

NOTICE = (
    "These files are free software: you can redistribute them and/or modify "
    "them under the terms of the GNU General Public License as published by "
    "the Free Software Foundation, either version 3 of the License, or "
    "(at your option) any later version."
)


def test_frozen_index_plan_groups_casefolded_names_and_excludes_anonymous(monkeypatch):
    monkeypatch.setattr(study, "GROUPS", 2)
    html = "".join(
        f'<a href="{name}">session</a>'
        for name in (
            "Contributor-20160101-abc.tgz",
            "contributor-20160102-def.tgz",
            "Other-20170101-ghi.tgz",
            "excluded-20200101-jkl.tgz",
            "TR-anonymous-20200101-mno.tgz",
            "../../escape-20200101-pqr.tgz",
        )
    ).encode()
    plan, count = study.source_plan(html, {"voxforge:excluded"})
    assert count == 2
    assert {row["contributor"]: len(row["sessions"]) for row in plan} == {"contributor": 2, "other": 1}
    assert study.source_plan(html, {"voxforge:excluded"}) == (plan, count)


def test_permission_is_explicit_and_session_language_matches():
    assert study.license_name(NOTICE) == "GPL-3.0-or-later"
    assert (
        study.license_name("This archive is licensed under GNU General Public License version 2.")
        == "GPL-2.0-only"
    )
    assert study.license_name("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007") is None
    assert study.license_name(NOTICE + " Research only.") is None
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:") as archive:
        files = {
            "session/etc/README": b"User Name: Example\nLanguage: TR\n",
            "session/etc/LICENSE.txt": NOTICE.encode(),
            "session/etc/PROMPTS": "tr-0001 Bugün hava güzel.\n".encode(),
        }
        for path, payload in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r:") as archive:
        files = study.regular_members(archive)
        prompts, evidence = study.metadata(archive, files, "example")
        assert set(prompts) == {"tr-0001"}
        assert evidence["license"] == "GPL-3.0-or-later"
        with pytest.raises(ValueError, match="matching named contributor"):
            study.metadata(archive, files, "another")


def test_archive_rejects_links_and_path_traversal():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:") as archive:
        member = tarfile.TarInfo("../outside.wav")
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        archive.addfile(member)
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r:") as archive:
        with pytest.raises(ValueError, match="safe regular session members"):
            study.regular_members(archive)


def test_bounded_acquisition_checks_exact_publisher_response_without_network():
    class Response(io.BytesIO):
        status = 200

        def geturl(self):
            return study.ORIGIN + "Example-20160101-abc.tgz"

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == Response(b"").geturl()
            response = Response(gzip.compress(b"test"))
            response.headers = {"Content-Length": str(len(response.getvalue())), "ETag": '"abc"'}
            return response

    payload, headers = study.download_archive(Opener(), "Example-20160101-abc.tgz")
    assert gzip.decompress(payload) == b"test" and headers["etag"] == '"abc"'
    with pytest.raises(ValueError, match="frozen original Turkish session"):
        study.download_archive(Opener(), "../outside.tgz")


def wav_bytes(frames):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, study.NATIVE_RATE, frames, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<hh", 1500, -1500) * (frames // 2))
    return buffer.getvalue()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Production decoder tools absent"
)
def test_production_preserves_full_native_clock_without_padding(tmp_path):
    expected_native = 130_224
    raw = wav_bytes(expected_native)
    assert study.native_samples(raw, ".wav") == expected_native
    wav, result = study.production_pcm(raw, ".wav", expected_native, tmp_path)
    assert len(study.read_wav(wav)) == 43_408
    assert result["source_range"] == study.source_range(43_408)
    with pytest.raises(ValueError, match="Expected native"):
        study.native_samples(wav_bytes(study.NATIVE_RATE * 4 + 2), ".wav")
    flac = tmp_path / "fixture.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(tmp_path / "original.wav"), "-c:a", "flac", str(flac)],
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert study.native_samples(flac.read_bytes(), ".flac") == expected_native
