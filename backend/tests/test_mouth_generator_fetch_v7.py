import hashlib

import pytest

from scripts.fetch_mouth_generator_v7 import acquire, fetch, verify


def test_manifest_digest_is_checked_before_output_or_network(tmp_path):
    manifest = tmp_path / "changed.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        fetch(manifest, "0" * 64, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_an_existing_partial_is_preserved_without_network_access(tmp_path, monkeypatch):
    partial = tmp_path / "musetalk/unet.pth.part"
    partial.parent.mkdir()
    partial.write_bytes(b"keep this prior attempt")
    monkeypatch.setattr(
        "scripts.fetch_mouth_generator_v7.urllib.request.urlopen",
        lambda *a, **kw: pytest.fail("must not start another download"),
    )
    with pytest.raises(FileExistsError):
        acquire(tmp_path, {"path": "musetalk/unet.pth", "url": "https://example.invalid"}, False)
    assert partial.read_bytes() == b"keep this prior attempt"


def test_cached_weights_require_publisher_size_and_digest(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"example weights")
    row = {"path": "weights", "bytes": 15, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="LFS digest"):
        verify(path, row)
    assert path.read_bytes() == b"example weights"


def test_small_configs_require_the_publisher_git_blob(tmp_path):
    body = b'{"channels":8}'
    path = tmp_path / "config.json"
    path.write_bytes(body)
    expected = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    row = {"path": "config.json", "bytes": len(body), "sha256": None, "git_blob": expected}
    assert verify(path, row)["sha256"] == hashlib.sha256(body).hexdigest()
