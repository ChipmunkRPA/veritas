from __future__ import annotations

import io
from copy import deepcopy
from hashlib import sha256

import pytest
from PIL import Image

from app.engine.build_info import sampling
from app.engine.customer_assessment import customer_assessment
from app.engine.exhibits import ExhibitWriter
from app.engine.fusion import abstain
from app.evidence import image_animation as check
from app.evidence import image_animation_binding as binding
from app.storage.files import validate_kind
from app.storage.objects import ObjectStore


def _apng(*, separate=True, same=False, blend=0, orientation=1):
    fallback = Image.new("RGBA", (80, 56), "white")
    opening = fallback.copy()
    if not same:
        opening.paste((255, 0, 0, 255), (16, 14, 30, 35))
    else:
        opening.paste((218, 17, 40, 0), (16, 14, 30, 35))
    tail = Image.new("RGBA", fallback.size, "blue")
    exif = Image.Exif()
    exif[274] = orientation
    output = io.BytesIO()
    fallback.save(output, format="PNG", save_all=True, default_image=separate, append_images=[opening, tail],
                  duration=400, blend=blend, exif=exif)
    return output.getvalue()


def test_comparison_uses_display_pixels_and_source_orientation():
    assert validate_kind("image", _apng(), "animation.png")[0] == "image/apng"
    assert check.measure(_apng(orientation=6))[0] == {
        "frame_count": 3, "changed_pixels": 294, "bounds": [21, 16, 21, 14],
    }
    assert check.measure(_apng(same=True))[0] == {"frame_count": 3, "changed_pixels": 0, "bounds": None}
    assert check.measure(_apng(separate=False)) is None
    assert check.measure(_apng(blend=1)) is None
    assert check.measure(_apng()[:65]) is None


def test_previews_bind_to_the_reported_source_and_playback_views(tmp_path):
    data = _apng()
    digest = sha256(data).hexdigest()
    store = ObjectStore(tmp_path / "objects")
    writer = ExhibitWriter("animated-test", store).bind(binding.KEY)
    ids = check.add_exhibits(data, writer)
    result = {"input": {"sha256": digest}, "exhibits": [e.model_dump() for e in writer.exhibits if e.id in ids]}
    review = check.from_result(result)
    source = {"sha256": digest, "width": 80, "height": 56, "coordinate_space": "exif-oriented-original"}
    record = binding.projection(review)
    assert review["bounds"] == [16, 14, 14, 21]
    assert len(binding.assets(record, source)) == 2
    check.verify(data, record, store.read_bytes, source_size=(80, 56))
    with pytest.raises(ValueError, match="measurements differ"):
        check.verify(data, record, store.read_bytes, source_size=(79, 56))
    malformed = {**record, "bounds": [79, 14, 14, 21]}
    with pytest.raises(ValueError, match="outside"):
        binding.assets(malformed, source)
    swapped = deepcopy(record)
    swapped["previews"]["opening"], swapped["previews"]["fallback"] = (
        record["previews"]["fallback"], record["previews"]["opening"],
    )
    with pytest.raises(ValueError, match="differs from the original display view"):
        check.verify(data, swapped, store.read_bytes, source_size=(80, 56))
    with pytest.raises(ValueError, match="measurements differ"):
        check.verify(_apng(same=True), record, store.read_bytes, source_size=(80, 56))


def test_multiple_images_do_not_inherit_a_single_image_confidence():
    image_sampling = sampling("image", {"probe": {"n_frames": 3}})
    assert abstain.outside_reason("image", {}, image_sampling, None) == abstain.MULTIFRAME_IMAGE_SCOPE
    result = {"kind": "image", "status": "done", "manifest": {"sampling": image_sampling},
              "condition_profile": {"within_validated_envelope": True},
              "decision": {"code": "no_ai_indicators", "fusion_mode": "artefact", "probability_ai_calibrated": 0.01}}
    assessment = customer_assessment(result)
    assert assessment["direction"] == "inconclusive" and assessment["confidence_pct"] is None
    assert assessment["scope_label"] == "First stored image"
    result["decision"]["provenance_record"] = {"kind": "trusted_capture"}
    assert customer_assessment(result)["direction"] == "record"
