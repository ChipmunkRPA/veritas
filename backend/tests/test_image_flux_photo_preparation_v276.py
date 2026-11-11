import json

import numpy as np
import pytest
from PIL import Image

from scripts import image_flux_photo_preparation_v276 as study


def test_exposure_graph_propagates_creator_and_rgb_links():
    known = [
        {
            "key": "old/1",
            "source_id": 1,
            "author": "a",
            "size": [400, 400],
            "rgb_sha256": "old",
            "sketch": "0",
        },
        {
            "key": "fresh-jpeg/2",
            "source_id": 2,
            "author": "b",
            "size": [400, 400],
            "rgb_sha256": "new",
            "sketch": "0",
        },
        {
            "key": "fresh-jpeg/3",
            "source_id": 3,
            "author": "c",
            "size": [400, 400],
            "rgb_sha256": "free",
            "sketch": "0",
        },
    ]
    later = [
        {
            "key": "later/4",
            "source_id": 4,
            "author": "a",
            "size": [400, 400],
            "rgb_sha256": "new",
            "sketch": "0",
        }
    ]
    assert study.cross_exposure(known, later) == [["fresh-jpeg/3"]]


def test_group_roster_is_stable_and_uses_cc_by_only():
    rows = [
        {"key": str(i), "source_jpeg_sha256": str(i), "license_id": licence, "size": [400, 400]}
        for i, licence in enumerate((4, 4, 4, 5))
    ]
    groups = [["0", "1"], ["2"], ["3"]]
    result, count = study.select(rows, groups)
    assert count == 2 and {tuple(group) for _, group in result} == {("0", "1"), ("2",)}
    assert study.select(list(reversed(rows)), list(reversed(groups))) == (result, count)


@pytest.mark.parametrize("slot,bounds,prompt", study.CASES)
def test_controls_have_exact_same_boundary(slot, bounds, prompt):
    source, _ = study.flux.fixture(slot)
    arrays, changed = study.make_controls(source, bounds)
    support = arrays["mask"] > 0
    assert int(support.sum()) == 4096 and bool(support[:, 0].any()) == (slot == "edge")
    assert all(0 < value <= 4096 for value in changed.values())
    for key in changed:
        exact = np.any(arrays[key] != source, axis=2)
        assert np.array_equal(exact, arrays[key + "_changed"] == 255)
        assert not exact[~support].any()


def test_center_crop_preserves_decoded_native_samples(tmp_path):
    y, x = np.indices((401, 430))
    image = np.stack((x, y, x + 2 * y), axis=2).astype(np.uint8)
    path = tmp_path / "photo.jpg"
    Image.fromarray(image).save(path, format="JPEG")
    crop, bounds = study.load_crop(path)
    assert bounds == (87, 72, 343, 328)
    with Image.open(path) as decoded:
        assert np.array_equal(crop, np.asarray(decoded.convert("RGB"))[72:328, 87:343])


def test_prepared_readback_rejects_png_and_manifest_changes(tmp_path):
    root = tmp_path.resolve()
    rows = []
    for slot, bounds, prompt in study.CASES:
        folder = root / slot
        folder.mkdir()
        source, _ = study.flux.fixture(slot)
        arrays, _ = study.make_controls(source, bounds)
        rows.append(
            {
                "slot": slot,
                "seed": study.SEED,
                "mask_ltrb_exclusive": list(bounds),
                "prompt": prompt,
                "rgb_or_mask_sha256": {key: study.flux.digest(value) for key, value in arrays.items()},
                "png_sha256": {
                    key: study.flux.save_png(folder / f"{key}.png", value) for key, value in arrays.items()
                },
            }
        )
    manifest = root / "private-inputs.json"
    private = {"schema": "chipmunk.image-flux-photo-prepared-private/276", "rows": rows, "audit": {}}
    manifest.write_text(json.dumps(private))
    assert study.verify_prepared(root) == private
    plan = study.freeze_plan(root)
    assert json.loads(json.dumps(plan)) == plan
    artifact = root / "edge/source.png"
    with artifact.open("ab") as stream:
        stream.write(b"change")
    with pytest.raises(ValueError, match="Prepared PNG changed"):
        study.verify_prepared(root)
    rows[0]["prompt"] = "replaced"
    manifest.write_text(json.dumps(private))
    with pytest.raises(ValueError, match="Exact case"):
        study.verify_prepared(root)
