# SPDX-License-Identifier: Apache-2.0
"""Bind Motion Photo crops and measured times to the uploaded JPEG."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import motion_photo
from app.engine.boundary import DetectorTimeout
from app.engine.exhibits import ExhibitBudgetExceeded
from app.evidence import image_motion_photo_binding as binding


def _bounds(x, y, right, bottom, size):
    left = max(0, int(np.floor(x)))
    top = max(0, int(np.floor(y)))
    width = min(size[0], int(np.ceil(right))) - left
    height = min(size[1], int(np.ceil(bottom))) - top
    if width < 1 or height < 1:
        raise ValueError("Motion Photo region is outside its frame")
    return [left, top, width, height]


def _rectangle(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def _regions(measured):
    still_h, still_w = measured.still.shape[:2]
    frame_h, frame_w = measured.frame.shape[:2]
    x, y, right, bottom = _rectangle(measured.region)
    native_w, native_h = measured.original_size
    original = _bounds(x * native_w / still_w, y * native_h / still_h,
                       right * native_w / still_w, bottom * native_h / still_h, measured.original_size)
    corners = np.array([[x - .5, y - .5], [right - .5, y - .5],
                        [x - .5, bottom - .5], [right - .5, bottom - .5]])
    matrix = measured.frame_transform
    projected = corners @ matrix[:, :2].T + matrix[:, 2] + .5
    frame = _bounds(*projected.min(axis=0), *projected.max(axis=0), (frame_w, frame_h))
    return {"still": original, "video": frame, "mask": list(measured.region), "marked_pixels": int(measured.mask.sum())}


def _resize(image, *, limit=384):
    return image.resize(binding.preview_size(*image.size, limit=limit), Image.Resampling.LANCZOS)


def _pictures(data, attachment, measured, region):
    context = Image.fromarray(measured.still)
    mask_area = Image.fromarray(measured.mask.astype(np.uint8) * 140)
    context.paste("#ffbf47", _rectangle(region["mask"]), mask_area)
    mask = Image.fromarray(measured.mask.astype(np.uint8) * 255).convert("RGB")
    video = Image.fromarray(measured.frame).crop(_rectangle(region["video"]))
    with Image.open(BytesIO(data[:attachment.image_end]), formats=["JPEG"]) as original:
        with ImageOps.exif_transpose(original) as oriented:
            still = oriented.crop(_rectangle(region["still"])).convert("RGB")
    return {"context": _resize(context, limit=512), "still": _resize(still), "video": _resize(video), "mask": mask}


def _unassessed(digest):
    return {"schema": binding.SCHEMA, "source_sha256": digest, "status": "unassessed",
            "authorship_confidence_contribution": 0, "clock": None, "comparison": None, "region": None, "previews": {}}


def inspect(data, digest, *, deadline=None):
    """Return a portable record and the exact display pixels for any local evidence."""
    result = _unassessed(digest)
    try:
        attachment = motion_photo.carrier(data)
        if attachment is None:
            return None, {}
        measured = motion_photo.measure(data, attachment, deadline=deadline)
        comparison = {"still_grid": [measured.still.shape[1], measured.still.shape[0]],
                      "video_grid": [measured.frame.shape[1], measured.frame.shape[0]], "compared_pixels": measured.compared_pixels}
        clock = {**measured.clock, "video_sha256": sha256(data[attachment.video_start:]).hexdigest()}
        result.update(status="no_region", clock=clock, comparison=comparison)
        if measured.region is None:
            return result, {}
        region = _regions(measured)
        pictures = _pictures(data, attachment, measured, region)
        result.update(status="different", region=region)
        return result, pictures
    except (ValueError, OSError, SyntaxError, EOFError, cv2.error, Image.DecompressionBombError, DetectorTimeout):
        return _unassessed(digest), {}


def capture(data, digest, writer, *, deadline):
    record, pictures = inspect(data, digest, deadline=deadline)
    if record is None:
        return None, set()
    ids = set()
    try:
        for role, image in pictures.items():
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                payload = buffer.getvalue()
            if len(payload) > binding.MAX_ASSET_BYTES:
                raise ExhibitBudgetExceeded("Motion Photo comparison picture exceeds its size limit")
            exhibit_id = writer.image(payload, role="comparison", media_type="image/png",
                                      title=binding.ROLES[role], media_ref="original")
            ids.add(exhibit_id)
            record["previews"][role] = {"exhibit_id": exhibit_id, "sha256": sha256(payload).hexdigest()}
    except ExhibitBudgetExceeded:
        record, ids = _unassessed(digest), set()
    caption = f"{binding.describe(record)} {binding.SCOPE} {binding.LIMITATION}"
    ids.add(writer.table(["Status"], [[record["status"]]], title=binding.TITLE, caption=caption))
    return record, ids


def from_result(result):
    stored = (result.get("manifest", {}).get("sampling") or {}).get(binding.KEY)
    if not stored or stored["source_sha256"] != (result.get("input") or {}).get("sha256"):
        return None
    return {**stored, "summary": binding.describe(stored), "scope": binding.SCOPE, "limitation": binding.LIMITATION,
            "previews": {role: {**asset, "title": binding.ROLES[role], "url": f"/api/v2/exhibits/{asset['exhibit_id']}/payload"}
                         for role, asset in stored["previews"].items()}}


def verify(data, record, read_asset, *, source_size):
    digest = sha256(data).hexdigest()
    binding.assets(record, {"sha256": digest, "width": source_size[0], "height": source_size[1],
                            "coordinate_space": "exif-oriented-original"})
    if record["status"] == "unassessed":
        return False
    observed, pictures = inspect(data, digest)
    if observed is None or any(observed[key] != record[key] for key in binding.FIELDS):
        raise ValueError("Motion Photo measurements differ from the submitted JPEG")
    for role, picture in pictures.items():
        asset = record["previews"][role]
        data = read_asset(asset["sha256"])
        if len(data) > binding.MAX_ASSET_BYTES or sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError("A Motion Photo image differs from its signed bytes")
        with Image.open(BytesIO(data), formats=["PNG"]) as saved:
            if saved.mode != "RGB" or saved.size != picture.size or saved.tobytes() != picture.tobytes():
                raise ValueError("A Motion Photo image differs from the measured pixels")
    return True
