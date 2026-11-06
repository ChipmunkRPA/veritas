import io
import json
from pathlib import Path

import pytest

from scripts import fetch_mouth_benchmark_assets as assets

CATALOGUE = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/mouth-v4-renderer-assets.json"


def test_modified_asset_catalogue_is_refused_before_network_or_writes(tmp_path):
    modified = tmp_path / "catalogue.json"
    modified.write_text("{}")
    with pytest.raises(ValueError, match="reviewed public record"):
        assets.fetch(modified, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_partial_download_is_never_deleted_by_another_attempt(tmp_path, monkeypatch):
    first = json.loads(CATALOGUE.read_text())["code"][0]
    path = tmp_path / "liveportrait-source" / first["path"]
    partial = path.with_suffix(path.suffix + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"previous attempt must be preserved")
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b""))
    with pytest.raises(FileExistsError):
        assets.fetch(CATALOGUE, tmp_path)
    assert partial.read_bytes() == b"previous attempt must be preserved"


def test_wrong_cached_source_is_refused_without_overwrite(tmp_path):
    first = json.loads(CATALOGUE.read_text())["code"][0]
    path = tmp_path / "liveportrait-source" / first["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unexpected cached source")
    with pytest.raises(ValueError, match="Cached benchmark asset changed"):
        assets.fetch(CATALOGUE, tmp_path, verify_only=True)
    assert path.read_bytes() == b"unexpected cached source"
