import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from scripts import image_copy_transfer_v298 as transfer


def source_row(index, author=None):
    author = author or f"creator-{index}"
    return {
        "key": f"fresh-jpeg/{index}",
        "source_id": index,
        "author": author,
        "author_profile": f"https://www.flickr.com/photos/{author}",
        "photo_page": f"https://www.flickr.com/photos/{author}/{index}",
        "license_id": 4,
        "license_url": transfer.pilot.CC_BY,
        "size": [641, 480],
        "source_jpeg_sha256": hashlib.sha256(f"encoded/{index}".encode()).hexdigest(),
        "rgb_sha256": hashlib.sha256(f"rgb/{index}".encode()).hexdigest(),
    }


def test_roster_excludes_entire_consumed_component_and_balances_frozen_geometry():
    sources = [source_row(i) for i in range(11)]
    sources[-1] = source_row(10, author=sources[0]["author"])
    graph = {
        "sources": sources,
        "candidate_only_unlinked_components": [[sources[0]["key"], sources[-1]["key"]]]
        + [[row["key"]] for row in sources[1:-1]],
    }
    consumed = [
        {
            "group": graph["candidate_only_unlinked_components"][0],
            "source_jpeg_sha256": sources[0]["source_jpeg_sha256"],
        }
    ]
    available = transfer.eligible_sources(graph, consumed)
    attribution = {
        row["source_id"]: {
            "author_url": row["author_profile"],
            "photo_page": row["photo_page"],
            "license_url": row["license_url"],
        }
        for row in sources
    }
    planned = transfer.assign(available, attribution)

    assert len(available) == 9
    assert len({row["source"]["author"] for row in planned}) == 8
    assert all(row["source"]["author"] != sources[0]["author"] for row in planned)
    assert all(row["crop_ltrb_exclusive"] == [192, 112, 448, 368] for row in planned)
    for placement in transfer.TARGETS:
        for phase in transfer.DONORS:
            assert sum((row["placement"], row["jpeg_phase"]) == (placement, phase) for row in planned) == 2


@pytest.mark.parametrize("placement,phase", [("interior", "aligned"), ("edge", "shifted")])
def test_same_mask_paired_controls_keep_exact_native_truth(placement, phase):
    transfer.cv2.setNumThreads(1)
    source = transfer.cue.fixture_texture(298, (256, 256))
    target = transfer.TARGETS[placement]
    donor = transfer.DONORS[phase]
    proposal, variants = transfer.controls(source, target, donor)
    row = {
        "slot": 0,
        "placement": placement,
        "jpeg_phase": phase,
        "target_box_xywh": list(target),
        "donor_box_xywh": list(donor),
    }
    results = {(result["variant"], result["format"]): result for result in transfer.score_group(row, source)}

    assert len(results) == 10
    assert len({result["proposal_mask_sha256"] for result in results.values()}) == 1
    assert all(result["detector_status"] == transfer.DETECTOR_STATUS for result in results.values())
    for kind, native in variants.items():
        changed = np.any(source != native, axis=2)
        assert not (changed & ~proposal).any()
        png, jpeg = (results[kind, encoding] for encoding in transfer.FORMATS)
        assert (
            png["native_change_mask_sha256"] == jpeg["native_change_mask_sha256"] == transfer.digest(changed)
        )
        assert png["native_changed_pixels"] == jpeg["native_changed_pixels"] == int(changed.sum())
        assert png["encoded_changes_outside_native_truth"] == 0
        if kind in transfer.COPY_TRANSFORMS:
            assert png["exact_ordinary_donor_recovered"]
            assert jpeg["exact_ordinary_donor_recovered"]
            assert jpeg["donor"]["box_xywh"] == list(donor)


def test_native_crop_uses_one_exact_source_and_rejects_byte_drift(tmp_path, monkeypatch):
    cache = tmp_path.resolve()
    monkeypatch.setattr(transfer.pilot, "CACHE", cache)
    native = transfer.cue.fixture_texture(421, (400, 512))
    buffer = io.BytesIO()
    Image.fromarray(native).save(buffer, "JPEG", quality=90)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as image:
        native_rgb = np.asarray(image.convert("RGB"))
    sha = hashlib.sha256(payload).hexdigest()
    path = cache / f"{sha}.bin"
    path.write_bytes(payload)
    row = {
        "source": {
            "source_jpeg_sha256": sha,
            "source_jpeg_file": str(path),
            "source_jpeg_bytes": len(payload),
            "size": [512, 400],
            "rgb_sha256": transfer.digest(native_rgb),
        },
        "crop_ltrb_exclusive": [128, 72, 384, 328],
    }

    assert np.array_equal(transfer.load_crop(row), native_rgb[72:328, 128:384])
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="Encoded source binding"):
        transfer.load_crop(row)


def test_freeze_and_complete_denominator_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "TRIAL", tmp_path.resolve())
    opened = []
    monkeypatch.setattr(transfer, "readout", lambda *_: opened.append(True))
    digest = "a" * 64
    with pytest.raises(FileNotFoundError):
        transfer.run(digest)
    (tmp_path / transfer.PLAN).write_text("{}")
    with pytest.raises(ValueError, match="local record differs"):
        transfer.run(digest)
    assert opened == []

    plan = {"planned_views": 80, "rows": [{"slot": index} for index in range(8)]}
    errors = [{"slot": index} for index in range(8)]
    incomplete = transfer.aggregate(plan, [], errors)
    assert incomplete["detector_unassessed_views"] == 80
    assert incomplete["scored_views"] == 0
    assert incomplete["by_format"]["jpeg90"]["reflected_copy"]["source_unavailable"] == 8
    with pytest.raises(ValueError, match="Missing or duplicate"):
        transfer.aggregate(plan, [], errors[:-1])
