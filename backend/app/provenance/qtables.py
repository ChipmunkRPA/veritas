"""Quantisation-table fingerprints and the make/model vs encoder rule (I-4; IF §4.1, the quantisation-table half of M4).

- **Fingerprint.** `jpeg_bitstream.JpegStructure.fingerprint`: sha256 over the component tables and sampling factors.
- **ExifTool JPEGDigest** (`-JPEGDigest` with print conversion, on stdin) names the encoder of known tables: IJG
  qualities, Photoshop levels, Lightroom and other applications, and some camera firmware tables ("Sony DSLR-A700,
  Fine"). A name that mentions the claimed make is firmware shared across that make's models, not a software signature.
  Unknown tables print as `Unknown (<md5>:<subsampling>)`. Without ExifTool the digest is unavailable and only the
  standard libjpeg check names a software signature.
- **In-house table DB** (`qtable_db.json`, built by `scripts/v2/build_qtable_db.py` from vb R1/R2 camera originals):
  per normalised make and model, the number of reference files and the fingerprints with their counts, plus the luma
  and chroma tables of every fingerprint for the table-diff exhibit.
- **Rule** (severities as in `metadata_rules`):
  - no Make or Model: info (the encoder is reported, there is no claim to test);
  - the claimed model has ≥ `MIN_REFERENCE_FILES` reference files and the fingerprint is among them: info;
  - reference files exist, the fingerprint is not among them and the tables are a software signature: anomaly;
  - reference files exist, the fingerprint is not among them, no software signature: notable (firmware variants);
  - no reference for the model: a software signature is notable; otherwise info ("no reference for this model").
- A match proves the encoder, not the content; editing is not deception.
"""
from __future__ import annotations

import functools
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.detectors.image.jpeg_bitstream import JpegStructure
from app.provenance import exiftool

DB_PATH = Path(__file__).with_name("qtable_db.json")
DIGEST_ARGS = ("-config", "", "-j", "-JPEGDigest", "-")
MIN_REFERENCE_FILES = 2
UNKNOWN_RE = re.compile(r"^Unknown \(([0-9a-f]{32})(?::[0-9]+)?\)$")
SEVERITY_ORDER = {"info": 0, "notable": 1, "anomaly": 2}


@dataclass(frozen=True)
class Digest:
    name: str | None  # the encoder ExifTool names, e.g. "Adobe Photoshop, Quality 3"; None for unknown tables
    md5: str | None  # present for unknown tables
    available: bool  # False without a usable ExifTool or when it fails


@dataclass(frozen=True)
class ModelEntry:
    make: str
    model: str
    files: int
    fingerprints: Mapping[str, int]

    @property
    def most_common(self) -> str:
        return max(sorted(self.fingerprints), key=lambda fp: self.fingerprints[fp])


@dataclass(frozen=True)
class TableDB:
    models: Mapping[str, ModelEntry]
    tables: Mapping[str, list[list[int]]]  # fingerprint → [luma, chroma] in natural order (chroma may be empty)
    built_from: Mapping[str, Any] = field(default_factory=dict)

    def models_with(self, fingerprint: str) -> list[str]:
        return sorted(f"{e.make} {e.model}" for e in self.models.values() if fingerprint in e.fingerprints)

    def without(self, key: str, fingerprint: str) -> TableDB:
        """The DB with one reference file of `key` using `fingerprint` removed (leave-one-out for the FPR record)."""
        entry = self.models.get(key)
        if entry is None or fingerprint not in entry.fingerprints:
            return self
        counts = {fp: c - (fp == fingerprint) for fp, c in entry.fingerprints.items()}
        reduced = replace(entry, files=entry.files - 1, fingerprints={fp: c for fp, c in counts.items() if c > 0})
        return replace(self, models={**self.models, key: reduced})


@dataclass(frozen=True)
class Assessment:
    severity: str  # info | notable | anomaly
    code: str
    observed: str
    note: str
    claim: str | None
    reference_files: int
    signature: str | None
    reference_fingerprint: str | None


def normalise(make: str | None, model: str | None) -> tuple[str, str] | None:
    """(make, model) in lower case with collapsed spaces and the make prefix removed from the model; None without both."""
    mk = " ".join(str(make or "").split()).casefold()
    md = " ".join(str(model or "").split()).casefold()
    if md.startswith(mk + " "):
        md = md[len(mk) + 1:]
    return (mk, md) if mk and md else None


def key(make: str | None, model: str | None) -> str | None:
    pair = normalise(make, model)
    return None if pair is None else f"{pair[0]}|{pair[1]}"


def jpeg_digest(data: bytes | memoryview) -> Digest:
    found = exiftool.binary()
    if found is None:
        return Digest(None, None, False)
    try:
        out = subprocess.run([found[0], *DIGEST_ARGS], input=bytes(data), capture_output=True, timeout=exiftool.TIMEOUT_S,
                             check=False, env=exiftool._env())
        records = json.loads(out.stdout or b"[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return Digest(None, None, False)
    value = str(records[0].get("JPEGDigest") or "") if records and isinstance(records[0], dict) else ""
    if not value:
        return Digest(None, None, True)
    unknown = UNKNOWN_RE.match(value)
    return Digest(None, unknown.group(1), True) if unknown else Digest(value, None, True)


@functools.cache
def load_db(path: Path = DB_PATH) -> TableDB:
    raw = json.loads(path.read_text())
    models = {k: ModelEntry(make=v["make"], model=v["model"], files=int(v["files"]), fingerprints=dict(v["fingerprints"]))
              for k, v in raw["models"].items()}
    return TableDB(models=models, tables=raw["tables"], built_from=raw.get("built_from", {}))


def names_make(name: str, make: str | None) -> bool:
    """True when a JPEGDigest name mentions the first word of the claimed make ("Sony DSLR-A700, Fine" for SONY)."""
    words = str(make or "").casefold().split()
    return bool(words) and re.search(rf"\b{re.escape(words[0])}\b", name.casefold()) is not None


def signature(structure: JpegStructure, digest: Digest, make: str | None = None) -> str | None:
    """A software signature: the encoder ExifTool names, unless it names a camera of the claimed make (firmware tables
    shared across models), or standard libjpeg tables."""
    if digest.name and not names_make(digest.name, make):
        return digest.name
    if structure.ijg_quality is not None:
        return f"standard libjpeg tables, quality {structure.ijg_quality}"
    return None


def assess(structure: JpegStructure, make: str | None, model: str | None, digest: Digest, db: TableDB) -> Assessment:
    fingerprint, sig = structure.fingerprint, signature(structure, digest, make)
    pair = normalise(make, model)
    encoder = sig or "tables not in ExifTool's list" + ("" if digest.available else " (ExifTool not available)")
    if pair is None:
        return Assessment("info", "M4.no_claim", f"encoder: {encoder}", "No Make and Model to compare the tables with",
                          None, 0, sig, None)
    claim = f"{make} {model}".strip()
    entry = db.models.get(f"{pair[0]}|{pair[1]}")
    files = entry.files if entry is not None else 0
    if entry is None or files < MIN_REFERENCE_FILES:
        if sig is not None:
            return Assessment("notable", "M4.no_reference_software", f"claimed camera {claim}; encoder signature: {sig}",
                              "No reference files for this model; the tables are an application's, so the file was probably "
                              "re-saved after capture", claim, files, sig, None)
        return Assessment("info", "M4.no_reference", f"claimed camera {claim}; encoder: {encoder}",
                          "No reference files for this model: uninformative", claim, files, sig, None)
    if fingerprint in entry.fingerprints:
        return Assessment("info", "M4.tables_match", f"claimed camera {claim}; tables equal those of "
                          f"{entry.fingerprints[fingerprint]} of {files} reference files from this model",
                          "Consistent with the claimed camera's encoder; this identifies the encoder, not the content", claim, files,
                          sig, fingerprint)
    if sig is not None:
        return Assessment("anomaly", "M4.tables_mismatch_software", f"claimed camera {claim}; encoder signature: {sig}; none of "
                          f"{files} reference files from this model use these tables",
                          "The tables contradict the claimed camera: the file was re-saved by other software after capture, "
                          "or the Make and Model were written by another program", claim, files, sig, entry.most_common)
    return Assessment("notable", "M4.tables_variant", f"claimed camera {claim}; tables differ from the {files} reference files "
                      "from this model", "Firmware versions and quality settings change the tables; not a software signature",
                      claim, files, sig, entry.most_common)


def diff_rows(observed: list[int] | tuple[int, ...], reference: list[int] | None) -> list[list[str]]:
    """8 × 8 rows of `observed` (natural order) with the reference step and the ratio when a reference exists."""
    rows = []
    for r in range(8):
        cells = []
        for c in range(8):
            value = int(observed[8 * r + c])
            if reference is None:
                cells.append(str(value))
            else:
                ref = int(reference[8 * r + c])
                cells.append(f"{value} / {ref} (×{value / ref:.2f})" if ref else f"{value} / {ref}")
        rows.append([str(r), *cells])
    return rows
