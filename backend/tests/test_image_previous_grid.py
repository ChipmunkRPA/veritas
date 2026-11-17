# SPDX-License-Identifier: Apache-2.0
"""Check that the previous JPEG grid and its displayed crop use the same pixels."""

from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image, ImageOps

from app.core.canonical import canonical_hash
from app.detectors.image import jpeg_previous_grid
from app.evidence import image_previous_grid_binding as binding
from app.evidence import image_previous_grid_source as source

SIZE = (369, 335)
STORED_REGION = (37, 19, 85, 51)


def jpeg(orientation, *, progressive):
    image = Image.linear_gradient("L").resize(SIZE).convert("RGB")
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    image.save(output, "JPEG", quality=90, subsampling=2, exif=exif, progressive=progressive)
    return output.getvalue()


def positive_measurement(monkeypatch):
    region = jpeg_previous_grid.Region(
        (37, 19, 48, 32), 0.75, (0.18, 0.19), (0.52, 0.54), (2, 5), None,
    )
    measured = jpeg_previous_grid.Measurement((5, 3), 55, 0.6, 0.25, {"surrounded_scan": region})
    monkeypatch.setattr(jpeg_previous_grid, "measure", lambda *args, **kwargs: measured)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_previous_grid_uses_stored_measurement_and_oriented_crop(monkeypatch, orientation):
    positive_measurement(monkeypatch)
    data = jpeg(orientation, progressive=orientation % 2 == 1)
    record = source.record(sha256(data).hexdigest(), data)
    assert record["status"] == "matched"
    assert record["measurement"]["grid_offset"] == [5, 3]

    mask = Image.new("L", SIZE)
    mask.paste(255, STORED_REGION)
    mask.getexif()[274] = orientation
    assert record["region"] == list(ImageOps.exif_transpose(mask).getbbox())

    with Image.open(BytesIO(data)) as jpeg_image:
        display = ImageOps.exif_transpose(jpeg_image)
    assets, blobs = {}, {}
    for role, image in source.plates(display, record["region"]).items():
        output = BytesIO()
        image.save(output, "PNG")
        payload = output.getvalue()
        digest = sha256(payload).hexdigest()
        assets[role] = {"width": image.width, "height": image.height, "media_type": "image/png", "sha256": digest}
        blobs[digest] = payload
    commitment = canonical_hash(record)
    review = {"record": record, "artifacts": assets}
    metadata = {"sha256": record["source_sha256"], "width": display.width, "height": display.height,
                "exif_orientation": orientation, binding.COMMITMENT: commitment}
    document = {"schema": "chipmunk.image-evidence/1", "source": metadata, binding.KEY: review}
    binding.validate(document, commitment)
    assert source.verify(data, review, blobs.__getitem__)

    left, top, right, bottom = record["region"]
    record["region"] = [left + 1, top, right + 1, bottom]
    commitment = canonical_hash(record)
    metadata[binding.COMMITMENT] = commitment
    with pytest.raises(ValueError, match="measured pixel grid"):
        binding.validate(document, commitment)


def test_existing_reports_replay_with_their_original_profile(monkeypatch):
    positive_measurement(monkeypatch)
    data = jpeg(6, progressive=True)
    file_hash = sha256(data).hexdigest()
    old = source.record(file_hash, data, schema=binding.LEGACY_SCHEMA)
    assert source.record(file_hash, data)["status"] == "matched"
    assert old["status"] == "outside_profile"
    assert source.verify(data, {"record": old, "artifacts": {}}, {}.__getitem__)

    incomplete = data[:-2]
    assert source.record(sha256(incomplete).hexdigest(), incomplete)["status"] == "outside_profile"
