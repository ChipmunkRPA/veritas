from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from scripts import prepare_tgif_localization as prepare


def setup_pair(tmp_path, monkeypatch, *, missing_image=False):
    pixels = np.full((64, 80, 3), 80, np.uint8)
    changed = pixels.copy()
    changed[12:25, 20:35] = 200
    assets = {}
    for role, rgb in (("original", pixels), ("image", changed)):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        data = buf.getvalue()
        name = prepare.sha(data) + ".png"
        (tmp_path / name).write_bytes(data)
        assets[role] = {"file": name, "sha256": prepare.sha(data), "bytes": len(data)}
    pair = {"source_id": "preserved-source", "family": "firefly", "image": ["image", "edited.png"],
            "original": ["original", "original.png"], "mask": ["mask", "missing.png"]}
    (tmp_path / "tgif-plan.json").write_text(json.dumps({"metadata_revision": prepare.REVISION,
                                                       "groups": {"firefly": [pair]}}))
    calls = []

    def fetch(url, members, root, *, allow_missing=False):
        calls.append((url, allow_missing))
        if url == "mask" or (url == "image" and missing_image):
            return {}
        role = "original" if url == "original" else "image"
        return {pair[role][1]: assets[role]}
    monkeypatch.setattr(prepare, "fetch_members", fetch)
    return pixels, changed, calls


def test_missing_secondary_mask_keeps_source_and_exact_changed_pixel_reference(tmp_path, monkeypatch):
    original, edited, calls = setup_pair(tmp_path, monkeypatch)
    result = prepare.acquire(tmp_path, allow_missing_author_masks=True)
    assert len(result["items"]) == 2
    assert {r["source_id"] for r in result["items"]} == {"preserved-source"}
    row = next(r for r in result["items"] if r["label"])
    assert row["author_mask"] is None and row["author_mask_status"] == "unavailable"
    assert row["changed_within_author_mask"] is None
    with Image.open(tmp_path / row["mask"]) as mask:
        np.testing.assert_array_equal(np.asarray(mask) > 0, np.any(original != edited, axis=2))
    assert ("mask", True) in calls and ("image", False) in calls and ("original", False) in calls


def test_default_still_requires_the_secondary_author_mask(tmp_path, monkeypatch):
    setup_pair(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="required image or reference"):
        prepare.acquire(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_secondary_mask_exception_never_waives_missing_primary_input(tmp_path, monkeypatch):
    setup_pair(tmp_path, monkeypatch, missing_image=True)
    with pytest.raises(ValueError, match="required image or reference"):
        prepare.acquire(tmp_path, allow_missing_author_masks=True)
    assert not (tmp_path / "manifest.json").exists()
