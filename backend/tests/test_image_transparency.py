from __future__ import annotations

import io
from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.engine.exhibits import ExhibitWriter
from app.engine.stages import image as image_stage
from app.evidence import image_transparency as check
from app.evidence import image_transparency_binding as binding
from app.reports import engine, figures
from app.reports.specimens import specimen
from app.storage.objects import ObjectStore


def encode(pixels, *, orientation=1, animated=False):
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    image = Image.fromarray(pixels)
    image.save(output, format="PNG", exif=exif, save_all=animated,
               append_images=[Image.new("RGBA", image.size, "blue")] if animated else [])
    return output.getvalue()


def sample():
    pixels = np.full((200, 240, 4), (20, 70, 160, 255), dtype=np.uint8)
    pixels[12:120, 16:98, 3] = 0
    return pixels


def test_transparency_measurement_distinguishes_color_from_flat_fill_and_edge_bleed():
    plain = sample()
    assert check.measure(encode(plain)) == ((240, 200, 108 * 82, 0), None)
    bleed = plain.copy()
    bleed[12:15, 16:98:2, 0] = 230
    assert check.measure(encode(bleed)) == ((240, 200, 108 * 82, 0), None)
    half_visible = plain.copy()
    half_visible[12:120, 16:98, 3] = 128
    assert check.measure(encode(half_visible)) == ((240, 200, 0, 0), None)
    assert check.measure(encode(plain[:, :, :3])) is None
    assert check.measure(encode(plain, animated=True)) is None
    assert check.measure(encode(plain)[:68]) is None
    assert check.measure(encode(np.zeros((1025, 4096, 4), dtype=np.uint8))) is None


@pytest.mark.parametrize("profile", ["summary", "standard"])
def test_render_stage_exports_registered_transparency_crops_and_signed_report_view(tmp_path, monkeypatch, profile):
    source = sample()
    source[32:84, 32:76:2, 0] = 240
    data = encode(source, orientation=6)
    store = ObjectStore(tmp_path / "objects")
    sha = store.put_bytes(data).sha256
    context = SimpleNamespace(
        analysis_id="transparency-test", store=store,
        exhibits=ExhibitWriter("transparency-test", store),
        state={"media": SimpleNamespace(original_bytes=lambda: memoryview(data))},
        outputs={"ingest": {"mime": "image/png", "sha256": sha}, "probe": {"mode": "RGBA", "n_frames": 1}},
        sessions=lambda: nullcontext(SimpleNamespace(commit=lambda: None)),
        check=lambda _: None,
    )
    monkeypatch.setattr(image_stage, "_legacy_artifacts", lambda _: {})
    monkeypatch.setattr(image_stage.settings, "image_whole_review_python", None)
    monkeypatch.setattr(image_stage.settings, "image_png_exif_preview_review", False)
    monkeypatch.setattr(image_stage.settings, "image_native_interpolation_review", False)
    indexed = []
    monkeypatch.setattr(image_stage.common, "index_artefacts", lambda *_args, extra: indexed.extend(extra))
    rendered = image_stage.render(context)
    assert len(indexed) == rendered["artefacts_indexed"] == 2
    result = {"input": {"sha256": sha}, "exhibits": rendered["exhibits"]}
    review = check.from_result(result)
    assert review["source_sha256"] == sha and review["authorship_confidence_contribution"] == 0
    assert (review["width"], review["height"], review["qualifying_tiles"]) == (200, 240, 2)
    location = review["location"]
    assert location["bounds"] == {"x": 128, "y": 0, "w": 72, "h": 128}
    view, reveal = (np.asarray(Image.open(io.BytesIO(store.read_bytes(c["sha256"])))) for c in check.crops(review))
    oriented = np.rot90(source, -1)
    expected = np.asarray(Image.alpha_composite(Image.new("RGBA", (72, 128), "white"),
                                               Image.fromarray(oriented[:128, 128:])).convert("RGB"))
    np.testing.assert_array_equal(view, expected)
    assert set(map(tuple, reveal.reshape(-1, 3))) == {(210, 210, 210), (20, 70, 160), (240, 70, 160)}
    record = binding.projection(review)
    source_ref = {"sha256": sha, "width": 200, "height": 240, "coordinate_space": "exif-oriented-original"}
    assert len(binding.assets(record, source_ref)) == 2
    assert check.verify(data, record, store.read_bytes)
    with pytest.raises(ValueError, match="crop differs"):
        check.verify(data, record, lambda digest: store.read_bytes(location["visible_crop"]["sha256"]))
    invalid = {**record, "location": {**record["location"], "bounds": {"x": 129, "y": 0, "w": 72, "h": 128}}}
    with pytest.raises(ValueError, match="outside"):
        binding.assets(invalid, source_ref)
    changed = encode(sample(), orientation=6)
    with pytest.raises(ValueError, match="coverage differs"):
        check.verify(changed, {**record, "source_sha256": sha256(changed).hexdigest()}, store.read_bytes)

    example = specimen("image", profile=profile)
    snapshot = example.model_copy(update={
        "result": {**example.result, **result}, "media": example.media.model_copy(update={"sha256": sha}),
        "exhibit_images": {c["exhibit_id"]: c["sha256"] for c in check.crops(review)},
    })
    monkeypatch.setattr(figures, "default_store", lambda: store)
    pdf = engine.render_pdf(snapshot)
    text = " ".join(engine.pdf_text(pdf.pdf).split())
    assert check.TITLE in text
    assert "Source x 128, y 0, 72 × 128 px." in text
    assert "no weight in the AI assessment" in text
    assert check.HIDDEN in text
