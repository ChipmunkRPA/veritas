"""Scope measurement for the optional whole-image PNG review."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.core.config import BACKEND_DIR
from app.engine.fusion.content_kinds import flat_block_share

TIMEOUT_S = 8
TESSDATA = BACKEND_DIR.parent / "models/weights/ocr/tessdata"
ENGLISH_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
FLAT_LIMIT = .5346
MIN_EDGE_PIXELS = 80
MAX_TSV_BYTES = 256 * 1024
MAX_WORDS = 400
MAX_WORD_PIXELS = 750_000


def has_interface_layout(rgb: np.ndarray, source: Path) -> bool:
    """Treat exact screen-like geometry as a scope limit for this check."""
    flat_share = flat_block_share(rgb)
    if flat_share is None:
        raise ValueError("PNG too small for layout measurement")
    if flat_share > FLAT_LIMIT:
        return True
    return _has_long_edge(rgb) or _same_ink_words(rgb, _recognized_lines(source)) >= 3


def _has_long_edge(rgb):
    height, width = rgb.shape[:2]
    return any(edge >= max(MIN_EDGE_PIXELS, .10 * length)
               for edge, length in zip(_constant_edges(rgb), (width, height), strict=True))


def _recognized_lines(source):
    binary = shutil.which("tesseract")
    if not binary:
        raise OSError("Local OCR executable is unavailable")
    with (TESSDATA / "eng.traineddata").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ENGLISH_SHA256:
            raise ValueError("English OCR language data differs")
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [binary, str(source), "stdout", "--tessdata-dir", str(TESSDATA), "-l", "eng", "--psm", "11",
             "-c", "tessedit_create_tsv=1"],
            stdout=output, stderr=subprocess.DEVNULL, check=True, timeout=TIMEOUT_S,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
        output.seek(0)
        return _word_boxes(output.read(MAX_TSV_BYTES + 1))


def _word_boxes(tsv):
    if len(tsv) > MAX_TSV_BYTES:
        raise ValueError("Local OCR output exceeds its size limit")
    lines = defaultdict(list)
    count = area = 0
    for row in csv.DictReader(io.StringIO(tsv.decode("utf-8")), delimiter="\t"):
        if row["level"] != "5" or float(row["conf"]) < 55 or sum(char.isalnum() for char in row["text"]) < 3:
            continue
        x, y, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
        if 6 <= height <= 96 and width >= 9:
            count += 1
            area += width * height
            if count > MAX_WORDS or area > MAX_WORD_PIXELS:
                raise ValueError("Local OCR word measurement exceeds its size limit")
            lines[tuple(row[key] for key in ("block_num", "par_num", "line_num"))].append((x, y, width, height))
    return lines.values()


def _same_ink_words(rgb, lines):
    height, width = rgb.shape[:2]
    largest = 0
    for line in lines:
        tones = Counter()
        for x, y, w, h in line:
            if not (0 <= x < x+w <= width and 0 <= y < y+h <= height):
                raise ValueError("OCR word lies outside the source")
            pixels = rgb[y:y+h, x:x+w]
            around = rgb[max(0, y-4):min(height, y+h+4), max(0, x-4):min(width, x+w+4)]
            ring = np.concatenate((around[0], around[-1], around[:, 0], around[:, -1]))
            background = np.median(ring, axis=0)
            colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
            clear_ink = (np.abs(colors-background).sum(axis=1) >= 96) & (counts >= w*h*.04)
            tones.update(map(tuple, colors[clear_ink].tolist()))
        largest = max(largest, max(tones.values(), default=0))
    return largest


def _constant_edges(rgb):
    for plane in (rgb, rgb.transpose(1, 0, 2)):
        best = 0
        for start in range(0, len(plane)-1, 128):
            block = plane[start:start+129].astype("int16")
            steps = block[1:] - block[:-1]
            aligned = ((np.abs(steps[:, 1:]).sum(axis=2) >= 36)
                       & (steps[:, 1:] == steps[:, :-1]).all(axis=2))
            for row in aligned[aligned.sum(axis=1) >= MIN_EDGE_PIXELS-1]:
                cuts = np.diff(np.pad(row.astype("int8"), 1))
                runs = np.flatnonzero(cuts == -1) - np.flatnonzero(cuts == 1)
                best = max(best, int(runs.max(initial=0)) + 1)
        yield best
