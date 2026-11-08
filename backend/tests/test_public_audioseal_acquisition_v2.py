"""Pinned acquisition fails on changed bytes and never executes package installers."""

import hashlib
import json
import zipfile

import pytest

from scripts import acquire_public_audioseal_v2 as acquire


def row(data=b"checked source"):
    return {
        "path": "nested/source.py",
        "member": "root/source.py",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_selected_archive_members_only_and_byte_identity(tmp_path):
    archive = tmp_path / "code.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("root/source.py", b"checked source")
        z.writestr("root/setup.py", b"never executed")
    acquire.extract(archive, [row()], tmp_path / "output")
    assert (tmp_path / "output/nested/source.py").read_bytes() == b"checked source"
    assert not (tmp_path / "output/setup.py").exists()


@pytest.mark.parametrize("damage", ["hash", "duplicate", "symlink", "path"])
def test_changed_or_ambiguous_member_cannot_be_extracted(tmp_path, damage):
    archive = tmp_path / "code.zip"
    selected = row()
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("root/source.py")
        if damage == "symlink":
            info.create_system = 3
            info.external_attr = 0o120777 << 16
        z.writestr(info, b"checked source")
        if damage == "duplicate":
            with pytest.warns(UserWarning):
                z.writestr("root/source.py", b"checked source")
    if damage == "hash":
        selected["sha256"] = "0" * 64
    if damage == "path":
        selected["path"] = "../source.py"
    with pytest.raises(ValueError):
        acquire.extract(archive, [selected], tmp_path / "output")
    assert not (tmp_path / "source.py").exists()


def test_corrupt_cache_is_rejected_without_network_fallback(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "asset.bin").write_bytes(b"bad")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": "chipmunk.public-audioseal-assets/v2",
                "assets": [{**row(b"good"), "path": "asset.bin", "source": "https://huggingface.co/unused"}],
                "source_files": [],
                "dependency_files": [],
            }
        )
    )
    monkeypatch.setattr(acquire, "LOCK", lock)

    def forbidden(*_):
        raise AssertionError("No fallback download")

    monkeypatch.setattr(acquire, "download", forbidden)
    with pytest.raises(ValueError, match="identity mismatch"):
        acquire.prepare(tmp_path / "new", [cache], True)


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/a",
        "https://huggingface.co.evil.invalid/a",
        "https://user:secret@huggingface.co/a",
    ],
)
def test_public_download_route_does_not_accept_unknown_hosts_or_credentials(url):
    with pytest.raises(ValueError):
        acquire.url_allowed(url)
