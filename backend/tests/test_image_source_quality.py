import hashlib
import io
import json
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from app.detectors.image import jpeg_bitstream, localiser
from app.evidence.image_eligibility import SOURCE_QUALITY, validate_source_quality
from app.reports import public_verify, verify_image_source
from app.reports.image_eligibility import inspection_check_view


@pytest.mark.parametrize("extension", ["png", "jpeg"])
def test_source_quality_is_recomputed_even_if_no_regions_were_selected(tmp_path, extension):
    pixels = np.random.default_rng(92).integers(10, 210, (96, 128, 3), dtype=np.uint8)
    pixels[:30] = (255, 0, 0)
    pixels[40:60] = (96, 96, 96)
    source = tmp_path / f"source.{extension}"
    Image.fromarray(pixels).save(source, **({"quality": 30} if extension == "jpeg" else {}))
    data = source.read_bytes()
    source_sha = hashlib.sha256(data).hexdigest()
    structure = jpeg_bitstream.read(data)
    with Image.open(source) as decoded:
        unreliable, shares = localiser.conditions_mask(np.asarray(decoded.convert("RGB")), structure.quality_luma if structure else None)
    grid = (64, 48)
    reliable = localiser.reliable_grid(unreliable, grid)
    assert shares["clipped"] > 0 and shares["flat"] > 0 and not reliable.all()
    quality = {"schema": SOURCE_QUALITY, "native_pixel_shares": shares}
    check = {"map_grid": list(grid), "source_quality": quality,
             "excluded_area_fraction": round(1 - float(reliable.mean()), 6)}
    evidence = {"source": {"width": 128, "height": 96, "exif_orientation": 1, "coordinate_space": "exif-oriented-original"},
                "coverage": {"inspection_checks": [check]}, "packets": []}
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "image"}, "source_commitment": {"sha256": source_sha}}

    def replay(grid_mask):
        image, archive = io.BytesIO(), io.BytesIO()
        Image.fromarray(grid_mask.astype(np.uint8) * 255).save(image, "PNG")
        raw = image.getvalue()
        check["reliability_mask_sha256"] = hashlib.sha256(raw).hexdigest()
        with ZipFile(archive, "w") as out:
            out.writestr("evidence.json", json.dumps(evidence))
            out.writestr(f"assets/{check['reliability_mask_sha256']}.png", raw)
        with ZipFile(archive) as reader:
            return verify_image_source.check(reader, bundle, source)

    assert "Recomputed source quality masks for 1 image check." in replay(reliable)
    validate_source_quality(check)
    reasons = inspection_check_view(check)["quality_reasons"]
    assert reasons[0] == {"label": "Clipped highlights or very dark pixels", "percent": round(shares["clipped"] * 100, 2)}
    with pytest.raises(ValueError, match="quality mask or condition shares"):
        replay(~reliable)
    quality["native_pixel_shares"] = {**shares, "clipped": 0}
    with pytest.raises(ValueError, match="quality mask or condition shares"):
        replay(reliable)
    quality["native_pixel_shares"]["unreliable"] = 0
    with pytest.raises(ValueError, match="contradict their union"):
        validate_source_quality(check)
