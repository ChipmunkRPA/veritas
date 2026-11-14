"""Compare the separate fallback view and opening animation frame of an APNG."""
from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageChops, ImageOps

from app.engine.exhibits import ExhibitBudgetExceeded
from app.engine.media import refuse_truncated_images
from app.evidence import image_animation_binding as binding

COLUMNS = ["Stored images", "Changed pixels", "X", "Y", "Width", "Height"]


def _display(source):
    with ImageOps.exif_transpose(source) as oriented:
        rgba = oriented.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def measure(data):
    if (len(data) > binding.MAX_INPUT_BYTES or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR"
            or bytes(data[24:26]) not in {b"\x08\x02", b"\x08\x06"}):
        return None
    refuse_truncated_images()
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as source:
            if (not source.info.get("default_image") or source.n_frames < 2
                    or max(source.size) > binding.MAX_EDGE or source.width * source.height > binding.MAX_PIXELS):
                return None
            count = source.n_frames
            fallback = _display(source)
            source.seek(1)
            # With a separate fallback the animation begins on a transparent
            # canvas. Pillow can blend or retain parts of the fallback for
            # other layouts, so use only a full replacement of that canvas.
            if source.info.get("blend") != 0 or source.info.get("bbox") != (0, 0, *source.size):
                return None
            opening = _display(source)
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError):
        return None

    red, green, blue = ImageChops.difference(fallback, opening).split()
    difference = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    box = difference.getbbox()
    changed = difference.width * difference.height - difference.histogram()[0]
    bounds = [box[0], box[1], box[2] - box[0], box[3] - box[1]] if box else None
    return {"frame_count": count, "changed_pixels": changed, "bounds": bounds}, (fallback, opening)


def _preview(image):
    return image.resize(binding.preview_size(*image.size), Image.Resampling.BOX)


def add_exhibits(data, writer):
    measured = measure(data)
    if measured is None:
        return set()
    record, pictures = measured
    changed, bounds = record["changed_pixels"], record["bounds"]
    selected = set()
    if bounds:
        summary = (f"The static fallback differs from the first animation frame at {changed:,} displayed pixels. "
                   f"Changed area starts at source x {bounds[0]}, y {bounds[1]} and spans {bounds[2]} × {bounds[3]} px.")
        try:
            for title, picture in zip(binding.ROLES.values(), pictures, strict=True):
                output = io.BytesIO()
                _preview(picture).save(output, format="PNG")
                selected.add(writer.image(output.getvalue(), role="comparison", media_type="image/png",
                                          title=title, media_ref="original"))
        except ExhibitBudgetExceeded:
            selected.clear()
            summary += " The preview pair could not be saved."
    else:
        summary = "The static fallback and the first animation frame have identical displayed pixels."
    frames = record["frame_count"] - 1
    summary += f" The header declares {frames} animation {'frame' if frames == 1 else 'frames'} plus the static fallback."
    selected.add(writer.table(COLUMNS, [[record["frame_count"], changed, *(bounds or [None] * 4)]],
                              title=binding.TITLE, caption=f"{summary} {binding.SCOPE} {binding.LIMITATION}"))
    return selected


def from_result(result):
    exhibits = {e["title"]: e for e in result.get("exhibits", ()) if e.get("created_by") == binding.KEY}
    table = exhibits.get(binding.TITLE)
    digest = (result.get("input") or {}).get("sha256")
    if not digest or not table or table.get("columns") != COLUMNS:
        return None
    (frames, count, *bounds), = table["rows"]
    previews = {role: {"exhibit_id": e["id"], "sha256": e["sha256"], "title": title,
                       "url": f"/api/v2/exhibits/{e['id']}/payload"}
                for role, title in binding.ROLES.items() if (e := exhibits.get(title))}
    return {"schema": binding.SCHEMA, "source_sha256": digest, "frame_count": int(frames), "changed_pixels": int(count),
            "bounds": list(map(int, bounds)) if count else None, "previews": previews, "summary": table["caption"],
            "authorship_confidence_contribution": 0}


def verify(data, record, read_asset, *, source_size):
    measured = measure(data)
    if (measured is None or measured[1][0].size != source_size
            or any(measured[0][key] != record[key] for key in binding.FIELDS)):
        raise ValueError("Animated PNG measurements differ from the original")
    pictures = dict(zip(binding.ROLES, measured[1], strict=True))
    for role, asset in record["previews"].items():
        raw = read_asset(asset["sha256"])
        if len(raw) > binding.MAX_ASSET_BYTES or hashlib.sha256(raw).hexdigest() != asset["sha256"]:
            raise ValueError("Animated PNG preview bytes differ from the report")
        preview = _preview(pictures[role])
        with Image.open(io.BytesIO(raw), formats=["PNG"]) as image:
            if image.mode != "RGB" or image.size != preview.size or image.tobytes() != preview.tobytes():
                raise ValueError("Animated PNG preview differs from the original display view")
