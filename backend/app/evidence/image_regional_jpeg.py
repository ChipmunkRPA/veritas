"""Source scope and repeat-view encoding for regional JPEG evidence."""

from __future__ import annotations

import io

from app.detectors.image import jpeg_bitstream


def evaluated_jpeg(data: bytes, *, quality: int = 75) -> bool:
    structure = jpeg_bitstream.read(data)
    frames = {"baseline", "progressive"} if quality in {75, 90} else {"baseline"}
    return bool(structure and structure.frame in frames and structure.precision == 8
                and max(structure.width, structure.height) == 768 and min(structure.width, structure.height) >= 512
                and [(c.id, c.h, c.v) for c in structure.components] == [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
                and all(structure.table_bits.get(c.table_id) == 8 for c in structure.components)
                and structure.ijg_quality == quality and structure.table(2) == structure.table(1)
                and (structure.frame == "baseline" or jpeg_bitstream.progressive_frame(data) is not None))


def reencode(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75, subsampling="4:2:0", optimize=False, progressive=False)
    return buffer.getvalue()
