import io

import numpy as np
import pytest
from PIL import Image

from scripts import image_bg_real_backup_intake_v262 as intake


def png(pixels):
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    return out.getvalue()


def test_selects_only_the_exact_previously_reserved_ordinals():
    historical = [{"ordinal": index, "tier": "primary" if index < 24 else "backup"} for index in range(36)]
    selected = intake.backup_cases(historical)
    assert [row["ordinal"] for row in selected] == list(range(12))
    assert [row["v218_ordinal"] for row in selected] == list(range(24, 36))
    assert {row["tier"] for row in selected} == {"primary"}
    assert historical[24] == {"ordinal": 24, "tier": "backup"}
    historical[25]["tier"] = "primary"
    with pytest.raises(ValueError, match="Exact unopened v218 tail required"):
        intake.backup_cases(historical)


def test_scoped_intake_seals_twelve_and_preserves_rejections(monkeypatch, tmp_path):
    original = intake.original
    trial, public, cache = (tmp_path / name for name in ("trial", "freeze.json", "cache"))
    plan = {"roster_dir": str(tmp_path), "private_cache": str(cache)}
    body_by_ordinal, cases = {}, []
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[16:24, 16:24] = 255
    for index in range(12):
        recipient = f"oi_{index:016x}"
        pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        edited = pixels.copy()
        edited[16:24, 16:24] ^= 255
        body = {role: png(image) for role, image in zip(original.ASSETS, (pixels, edited, mask), strict=True)}
        case = {
            "ordinal": index,
            "tier": "primary",
            "v218_ordinal": index + 24,
            "recipient": recipient,
            "flickr_photo_id": str(index),
            "flickr_owner": f"fixture-{index}",
            "publisher_split": "train",
            "size": [64, 64],
            "semantic_edit_area": 64 / 4096,
            "original": {
                "path": f"v2/images/0/{recipient}_A0.png",
                "sha256": original.digest(body["original"]),
            },
            "edited": {"path": f"v2/images/0/{recipient}_F5.png", "sha256": original.digest(body["edited"])},
            "semantic_mask": {"path": f"v2/masks/edit/0/{recipient}_F5_edit.png", "sha256": None},
        }
        body_by_ordinal[index], cases = body, [*cases, case]

    def fetch(_session, cache, case, _deadline):
        index = case["ordinal"]
        folder = cache / f"{index:02d}"
        folder.mkdir()
        rows = {}
        for role, body in body_by_ordinal[index].items():
            file = folder / f"{role}.png"
            file.write_bytes(body)
            digest = original.digest(body)
            rows[role] = {
                "status": "saved",
                "repo_path": case[role]["path"],
                "file": str(file.relative_to(cache)),
                "resolver_revision": original.roster.REVISION,
                "final_host": original.HUB,
                "bytes": len(body),
                "sha256": digest,
                "publisher_asset_commitment": None if index == 1 and role == "semantic_mask" else digest,
            }
        return rows

    prior = ([], {cases[0]["original"]["sha256"]}, set())
    monkeypatch.setattr(intake, "TRIAL", trial)
    monkeypatch.setattr(intake, "PUBLIC", public)
    monkeypatch.setattr(intake, "preflight", lambda: plan)
    monkeypatch.setattr(intake, "bindings", lambda *_: (plan, cases, set(), prior))
    monkeypatch.setattr(original, "fetch_case", fetch)
    intake.freeze()
    expected = intake.read(public)["private_plan_sha256"]
    assert set(intake.read(public)) == {"private_plan_sha256"}
    intake.acquire(expected)
    intake.seal(expected)
    result = intake.readout(expected)
    assert result["attempted_pairs"] == 12 and result["qualified_pairs"] == 10
    assert result["statuses"] == {
        "prior_exposure_component": 1,
        "semantic_mask_publisher_commitment_missing": 1,
        "qualified": 10,
    }
    assert result["saved_asset_counts"] == {"saved": 36}
    assert result["qualified_publisher_bound_masks"] == 10
    assert len(list(cache.glob("*/exact_truth.png"))) == 10
    truth = cache / "02/exact_truth.png"
    truth.write_bytes(png(np.zeros((64, 64), dtype=np.uint8)))
    with pytest.raises(ValueError, match="Saved exact changed-pixel mask differs"):
        intake.readout(expected)
