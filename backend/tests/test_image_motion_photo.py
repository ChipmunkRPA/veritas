"""Exercise the Motion Photo byte boundaries and portable time and crop bindings."""

import hashlib
import io

import pytest
from PIL import Image

from app.detectors.image import motion_photo
from app.evidence import image_motion_photo_binding as binding
from app.evidence.image_motion_photo import inspect


def _jpeg():
    output = io.BytesIO()
    Image.new("RGB", (320, 320), "white").save(output, format="JPEG")
    return output.getvalue()


def _box(kind, payload):
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _source(*, time=501000, gain=False):
    still = _jpeg()
    movie = _box(b"ftyp", b"isom" * 2) + _box(b"mdat", b"0" * 96) + _box(b"moov", b"")
    gain_map = _jpeg() if gain else b""
    middle = (f'<rdf:li><C:Item I:Semantic="GainMap" I:Mime="image/jpeg" I:Length="{len(gain_map)}"/></rdf:li>'
              if gain else "")
    xml = f'''<rdf:RDF xmlns:rdf="{motion_photo.RDF[1:-1]}" xmlns:G="{motion_photo.CAMERA[1:-1]}"
       xmlns:C="{motion_photo.CONTAINER[1:-1]}" xmlns:I="{motion_photo.ITEM[1:-1]}">
       <rdf:Description G:MotionPhoto="1" G:MotionPhotoVersion="1" G:MotionPhotoPresentationTimestampUs="{time}">
       <C:Directory><rdf:Seq><rdf:li><C:Item I:Semantic="Primary" I:Mime="image/jpeg"/></rdf:li>
       {middle}<rdf:li><C:Item I:Semantic="MotionPhoto" I:Mime="video/mp4" I:Length="{len(movie)}"/></rdf:li>
       </rdf:Seq></C:Directory></rdf:Description></rdf:RDF>'''
    packet = motion_photo.XMP + xml.encode()
    segment = b"\xff\xe1" + (len(packet) + 2).to_bytes(2, "big") + packet
    return still[:2] + segment + still[2:] + gain_map + movie, movie


@pytest.mark.parametrize("gain", [False, True])
def test_declared_video_requires_complete_attachment_and_gain_map(gain):
    data, movie = _source(gain=gain)
    assert motion_photo.carrier(_jpeg()) is None
    result = motion_photo.carrier(memoryview(data))
    assert result.timestamp_us == 501000
    assert data[result.video_start:] == movie
    assert data[result.image_end - 2:result.image_end] == b"\xff\xd9"
    if gain:
        assert data[result.image_end:result.video_start] == _jpeg()
    missing = data[:-len(movie)]
    with pytest.raises(ValueError):
        motion_photo.carrier(missing)
    recorded, pictures = inspect(missing, hashlib.sha256(missing).hexdigest())
    assert recorded["status"] == "unassessed" and not pictures


def test_declared_capture_needs_an_explicit_time():
    with pytest.raises(ValueError, match="explicit Motion Photo timestamp"):
        motion_photo.carrier(_source(time=-1)[0])


def test_export_validates_motion_photo_time_and_ai_weight():
    digest = "a" * 64
    record = {"schema": binding.SCHEMA, "source_sha256": digest, "status": "no_region",
              "authorship_confidence_contribution": 0,
              "clock": {"declared_timestamp_us": 501000, "time_base": "1/1000", "frame_index": 15,
                        "frame_pts": 501, "supporting_pts": [468, 501, 534], "video_sha256": "b" * 64},
              "comparison": {"still_grid": [320, 320], "video_grid": [320, 320], "compared_pixels": 90000},
              "region": None, "previews": {}}
    source = {"width": 320, "height": 320, "sha256": digest, "coordinate_space": "exif-oriented-original"}
    assert binding.assets(record, source) == {}
    with pytest.raises(ValueError, match="Motion Photo comparison"):
        binding.assets({**record, "authorship_confidence_contribution": 1}, source)
    with pytest.raises(ValueError, match="declared clock"):
        binding.assets({**record, "clock": {**record["clock"], "frame_pts": 534}}, source)
