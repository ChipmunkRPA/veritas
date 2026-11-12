"""Source-bound whole-image inspection evidence outside the calibrated estimate."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import psutil
from PIL import Image

from app.core.config import BACKEND_DIR, settings
from app.core.logging import get_logger
from app.engine.exhibits import ExhibitBudgetExceeded
from app.engine.models import MemoryBudgetExceeded, get_model_manager
from app.storage.objects import ObjectStoreError

CREATOR = "render"
TABLE = "Supplementary whole-image check"
CROP = "Inspected image view"
COLUMNS = ["Status", "Inspected view", "Source x, y, width, height"]
BOX_KEYS = ("x", "y", "w", "h")
STATUSES = {"flagged", "no_flag", "skipped"}
TIMEOUT_S = 45
MEMORY_RESERVE = 5 * 1024 ** 3
MIN_SIDE, MAX_SIDE = 1024, 4920
MAX_CROP_BYTES = 900_000
LOCK = threading.Lock()
log = get_logger(__name__)

FLAGGED = ("The supplementary whole-image check found a pattern associated with fully generated images. "
           "The views and bounds show which source pixels were inspected. "
           "This result carries zero weight in the overall assessment.")
NO_FLAG = ("The supplementary whole-image check measured three image views and gave no elevated response. "
           "It carries zero weight in the overall assessment.")
UNAVAILABLE = "The supplementary whole-image check was unavailable for this analysis."
SCOPE = ("The supplementary check covers upright RGB JPEGs with a short side of at least 1,024 pixels "
         "and a long side of at most 4,920 pixels.")


def _coordinates(bounds):
    return ", ".join(map(str, bounds))


def _summary(writer, status, text):
    try:
        return [writer.table(COLUMNS, [[status, "", ""]], title=TABLE, caption=text)]
    except ExhibitBudgetExceeded:
        return []


def _output(directory, source_sha256, size):
    path = directory / "result.json"
    if path.stat().st_size > 4096:
        raise ValueError("Image review record exceeds its size limit")
    result = json.loads(path.read_bytes())
    flagged, regions = result["flagged"], result["regions"]
    if (result["source_sha256"] != source_sha256 or type(flagged) is not bool or result["checked_views"] != 3
            or not isinstance(regions, list) or len(regions) != (3 if flagged else 0)):
        raise ValueError("Image review record is not bound to the submitted file")
    crops = []
    for number, region in enumerate(regions, 1):
        x, y, w, h = region["bounds"]
        if (region["region"] != number or any(type(n) is not int for n in (x, y, w, h))
                or not (0 <= x < x + w <= size[0] and 0 <= y < y + h <= size[1])):
            raise ValueError("Image review source bounds are invalid")
        with (directory / f"crop-{number}.png").open("rb") as stream:
            data = stream.read(MAX_CROP_BYTES + 1)
        if len(data) > MAX_CROP_BYTES or hashlib.sha256(data).hexdigest() != region["sha256"]:
            raise ValueError("Image review crop differs from its commitment")
        with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
            if image.mode != "RGB" or image.size != (528, 528):
                raise ValueError("Image review crop geometry is invalid")
            image.verify()
        crops.append(((x, y, w, h), data))
    return flagged, crops


def add_exhibits(source: Path, source_sha256: str, writer, *, remaining: float | None):
    """A missing, busy or slow optional check leaves primary analysis intact."""
    python = settings.image_whole_review_python
    try:
        with Image.open(source, formats=["JPEG"]) as image:
            size = image.size
            if image.mode != "RGB" or image.getexif().get(274, 1) != 1 or min(size) < MIN_SIDE or max(size) > MAX_SIDE:
                return _summary(writer, "skipped", SCOPE)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return _summary(writer, "skipped", SCOPE)
    if python is None or not python.is_file() or settings.models_dir is None:
        log.info("image_review.skipped", reason="runtime_not_configured")
        return _summary(writer, "skipped", UNAVAILABLE)
    try:
        memory, disk = psutil.virtual_memory().available, shutil.disk_usage(settings.data_dir).free
    except (OSError, psutil.Error):
        log.info("image_review.skipped", reason="resource_check_unavailable")
        return _summary(writer, "skipped", UNAVAILABLE)
    if (disk < settings.min_free_bytes or (remaining is not None and remaining < TIMEOUT_S + 5)
            or not LOCK.acquire(blocking=False)):
        log.info("image_review.skipped", reason="resource_or_deadline", available_memory=memory, free_disk=disk,
                 remaining_seconds=remaining)
        return _summary(writer, "skipped", UNAVAILABLE)
    try:
        reclaim = memory < MEMORY_RESERVE
        lease = (get_model_manager().lease([], prefer="cpu", activation_bytes=MEMORY_RESERVE)
                 if reclaim else nullcontext())
        with lease:
            if reclaim and (memory := psutil.virtual_memory().available) < MEMORY_RESERVE:
                log.info("image_review.skipped", reason="insufficient_memory", available_memory=memory)
                return _summary(writer, "skipped", UNAVAILABLE)
            with tempfile.TemporaryDirectory(prefix="chipmunk-image-review-") as tmp:
                directory = Path(tmp)
                env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
                env.update(PYTHONPATH=str(BACKEND_DIR), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="4")
                model = settings.models_dir / "research/image-v7-smaug"
                subprocess.run([str(python.absolute()), "-B", "-m", "scripts.image_whole_evidence_worker", str(source),
                                str(model), str(directory)], env=env, check=True, timeout=TIMEOUT_S,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                flagged, crops = _output(directory, source_sha256, size)
        if not flagged:
            return _summary(writer, "no_flag", NO_FLAG)
        ids, rows = [], []
        for number, (bounds, data) in enumerate(crops, 1):
            x, y, w, h = bounds
            ids.append(writer.image(data, role="comparison", media_type="image/png", media_ref="original", title=f"{CROP} {number}",
                                    caption=f"x {x}, y {y}, {w} × {h} source pixels. Inspected input 528 × 528 pixels."))
            rows.append(["flagged", str(number), _coordinates(bounds)])
        ids.append(writer.table(COLUMNS, rows, title=TABLE, caption=FLAGGED))
        return ids
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError, ExhibitBudgetExceeded, ObjectStoreError,
            MemoryBudgetExceeded, psutil.Error) as exc:
        log.info("image_review.skipped", reason=type(exc).__name__, exit_code=getattr(exc, "returncode", None))
        return _summary(writer, "skipped", UNAVAILABLE)
    finally:
        LOCK.release()


def is_review_exhibit(item):
    return (isinstance(item, Mapping) and item.get("created_by") == CREATOR
            and item.get("title") in (TABLE, f"{CROP} 1", f"{CROP} 2", f"{CROP} 3"))


def from_result(result):
    source_sha256 = (result.get("input") or {}).get("sha256")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        return None
    exhibits = [item for item in result.get("exhibits") or [] if is_review_exhibit(item)]
    table = next((item for item in exhibits if item.get("kind") == "table" and item.get("title") == TABLE
                  and item.get("columns") == COLUMNS), None)
    if table is None:
        return None
    try:
        rows = table["rows"]
        status = rows[0][0]
        if status not in STATUSES or (len(rows) != 3 if status == "flagged" else rows != [[status, "", ""]]):
            return None
        crops = []
        if status == "flagged":
            images = {item["title"]: item for item in exhibits if item.get("kind") == "image"}
            if len(images) != 3 or len(exhibits) != 4:
                return None
            for number, (flag, region, bounds) in enumerate(rows, 1):
                if flag != status or region != str(number):
                    return None
                crop = images[f"{CROP} {number}"]
                x, y, w, h = map(int, bounds.split(","))
                if min(x, y) < 0 or min(w, h) <= 0 or re.fullmatch(r"[0-9a-f]{64}", crop["sha256"]) is None:
                    return None
                crops.append({"region": number, "bounds": dict(zip(BOX_KEYS, (x, y, w, h), strict=True)),
                              "exhibit_id": crop["id"], "sha256": crop["sha256"],
                              "url": f"/api/v2/exhibits/{crop['id']}/payload"})
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return {"source_sha256": source_sha256, "status": status,
            "interpretation": table.get("caption") or UNAVAILABLE, "authorship_confidence_contribution": 0, "crops": crops}
