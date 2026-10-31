"""Agent visual label audit (D-06; ML §2.1 P6): per-source contact sheets and the audit record.

- A contact sheet shows up to 50 items of one source, drawn by a seeded RNG and grouped by label and generator or
  stratum; each tile carries its index, label and generator (or real stratum).
- Images are drawn from their native bytes, videos as one frame at 40 % of the duration, audio as a log-magnitude
  spectrogram of the first 8 s. Viewing cannot verify an audio label: audio sheets check format and content type only.
- Sheets stay under `data/vbench/audit/` (gitignored; P5, nothing is redistributed). `docs/v2/bench/audit.yaml` is the
  record: one entry per source with the sheet's sha256, the sampled item ids' sha256, the counts and the finding.
- Criterion: a source passes when mislabelled + unusable items are at most 10 % of the items viewed; the record's
  `result` must agree with its counts.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.bench.media import MediaError, av_input, open_image, to_rgb8

AUDIT_ITEMS = 50
MAX_FAIL_RATE = 0.10
TILE = 200
CAPTION = 30
HEADER = 40
COLUMNS = 10
FRAME_AT = 0.4
AUDIO_SECONDS = 8
_HEX64 = r"^[0-9a-f]{64}$"


class AuditError(ValueError):
    """The audit record cannot be read, is inconsistent, or does not cover the manifest."""


# --- sampling and sheets -----------------------------------------------------------------------------------

def sample_items(rows: list[dict], source_id: str, *, n: int = AUDIT_ITEMS, seed: int = 7) -> list[dict]:
    """Up to `n` items of the source by a seeded RNG over the item-id order, displayed grouped by label and kind."""
    pool = sorted((r for r in rows if r["source_id"] == source_id), key=lambda r: r["item_id"])
    picked = random.Random(f"{seed}:audit:{source_id}").sample(pool, min(n, len(pool)))
    return sorted(picked, key=lambda r: (r["label"], _kind(r), r["item_id"]))


def sample_sha256(items: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(r["item_id"] for r in items)).encode()).hexdigest()


def _kind(row: dict) -> str:
    return str(row.get("generator") or row.get("real_stratum") or row.get("family") or "")


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MediaError("ffmpeg is not installed")
    return path


def _run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False, stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not proc.stdout:
        raise MediaError(proc.stderr.decode(errors="replace").strip()[-200:] or "nothing decoded")
    return proc.stdout


def image_tile(path: Path) -> Image.Image:
    im = to_rgb8(open_image(path.read_bytes()))
    im.thumbnail((TILE, TILE))
    return im


def video_tile(path: Path, duration_s: float | None) -> Image.Image:
    at = max(0.0, float(duration_s or 0.0) * FRAME_AT)
    data = _run([_ffmpeg(), "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-ss", f"{at:.3f}", "-i", av_input(path), "-frames:v", "1",
                 "-vf", f"scale={TILE}:{TILE}:force_original_aspect_ratio=decrease", "-f", "image2pipe", "-vcodec", "png", "-"])
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def audio_tile(path: Path) -> Image.Image:
    data = _run([_ffmpeg(), "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", av_input(path), "-t", str(AUDIO_SECONDS), "-ac", "1", "-ar", "16000",
                 "-f", "s16le", "-"])
    x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    frame, hop = 512, 128
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    starts = hop * np.arange(1 + (x.size - frame) // hop)
    spectrum = np.abs(np.fft.rfft(x[starts[:, None] + np.arange(frame)[None, :]] * np.hanning(frame), axis=1))
    db = 20.0 * np.log10(spectrum + 1e-6)
    scaled = np.clip((db - (db.max() - 80.0)) / 80.0, 0.0, 1.0)
    return Image.fromarray((255 * scaled.T[::-1]).astype(np.uint8), "L").resize((TILE, TILE)).convert("RGB")


def render_tile(row: dict, vbench_dir: Path) -> Image.Image:
    path = vbench_dir / row["native_path"]
    if row["modality"] == "image":
        return image_tile(path)
    if row["modality"] == "video":
        return video_tile(path, row.get("duration_s"))
    return audio_tile(path)


@dataclass
class Sheet:
    source_id: str
    path: Path
    sha256: str
    pool: int
    items: list[dict]
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def sample_sha256(self) -> str:
        return sample_sha256(self.items)


def build_sheet(rows: list[dict], vbench_dir: Path, source_id: str, out_dir: Path, *, n: int = AUDIT_ITEMS,
                seed: int = 7) -> Sheet:
    items = sample_items(rows, source_id, n=n, seed=seed)
    if not items:
        raise AuditError(f"no manifest items for source {source_id!r}")
    pool = sum(1 for r in rows if r["source_id"] == source_id)
    out = out_dir / f"{source_id}.png"
    sha256, failures = render_sheet(items, vbench_dir, out, title=f"{source_id} · {len(items)} of {pool} items · seed {seed}",
                                    caption=lambda index, row: (f"{index + 1} {row['label']}", _kind(row)[:30]))
    return Sheet(source_id, out, sha256, pool, items, failures)


def render_sheet(items: list[dict], vbench_dir: Path, out: Path, *, title: str,
                 caption: Callable[[int, dict], tuple[str, str]]) -> tuple[str, dict[str, str]]:
    """A contact sheet of `items` in the given order with a two-line caption per tile → (PNG sha256, undecodable items)."""
    if not items:
        raise AuditError("a contact sheet needs at least one item")
    columns = min(COLUMNS, len(items))
    lines = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * TILE, HEADER + lines * (TILE + CAPTION)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    small, large = ImageFont.load_default(size=13), ImageFont.load_default(size=20)
    draw.text((8, 10), title, fill=(0, 0, 0), font=large)
    failures: dict[str, str] = {}
    for index, row in enumerate(items):
        x, y = (index % columns) * TILE, HEADER + (index // columns) * (TILE + CAPTION)
        try:
            tile = render_tile(row, vbench_dir)
        except (MediaError, OSError, ValueError, subprocess.TimeoutExpired) as e:
            failures[row["item_id"]] = str(e) or type(e).__name__
            tile = Image.new("RGB", (TILE, TILE), (200, 200, 200))
            ImageDraw.Draw(tile).text((8, TILE // 2 - 8), "not decodable", fill=(0, 0, 0), font=small)
        sheet.paste(tile, (x + (TILE - tile.width) // 2, y + (TILE - tile.height) // 2))
        first, second = caption(index, row)
        draw.text((x + 4, y + TILE + 1), first, fill=(0, 0, 0), font=small)
        draw.text((x + 4, y + TILE + 15), second, fill=(60, 60, 60), font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    out.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest(), failures


# --- record ------------------------------------------------------------------------------------------------

class SourceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    result: Literal["pass", "fail"]
    audit_kind: Literal["agent", "human"] = "agent"
    viewed: int = Field(ge=1)
    mislabelled: int = Field(ge=0)
    unusable: int = Field(ge=0)
    sheet_sha256: str = Field(pattern=_HEX64)
    sample_sha256: str = Field(pattern=_HEX64)
    date: str
    finding: str

    @model_validator(mode="after")
    def _criterion(self) -> SourceAudit:
        if self.mislabelled + self.unusable > self.viewed:
            raise ValueError(f"{self.source_id}: more flagged items than items viewed")
        expected = "pass" if (self.mislabelled + self.unusable) <= MAX_FAIL_RATE * self.viewed else "fail"
        if self.result != expected:
            raise ValueError(f"{self.source_id}: result {self.result!r} contradicts the counts (criterion gives {expected!r})")
        return self


class AuditFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    bench: str
    criterion: str
    sources: list[SourceAudit]

    @model_validator(mode="after")
    def _unique(self) -> AuditFile:
        ids = [s.source_id for s in self.sources]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate audit entries: {', '.join(duplicates)}")
        return self


def load_audit(path: Path) -> AuditFile:
    try:
        return AuditFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as e:
        raise AuditError(f"{path}: {e}") from None
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
        raise AuditError(f"{path}: {problems}") from None


def verify_samples(audit: AuditFile, rows: list[dict], *, seed: int = 7) -> list[str]:
    """Entries whose recorded sample no longer matches the manifest (the source changed after it was audited)."""
    stale = []
    for entry in audit.sources:
        items = sample_items(rows, entry.source_id, seed=seed)
        if not items or sample_sha256(items) != entry.sample_sha256 or len(items) != entry.viewed:
            stale.append(entry.source_id)
    return stale


def render_markdown(audit: AuditFile, *, generated_from: str) -> str:
    passed = sum(1 for s in audit.sources if s.result == "pass")
    lines = [
        f"# {audit.bench} label audit", "",
        f"Generated from `{generated_from}` by `backend/scripts/vbench_audit.py --render`. Audit kind per D-06: the loop's "
        "agent viewed each contact sheet itself; the human audit of ML §2.1 P6 is a post-loop task.", "",
        f"- **Criterion:** {audit.criterion}",
        f"- **Sources:** {len(audit.sources)} audited, {passed} passed, {len(audit.sources) - passed} failed (failed sources "
        "are excluded from every split).", "",
        "| Source | Result | Kind | Viewed | Mislabelled | Unusable | Date | Finding |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in audit.sources:
        finding = " ".join(s.finding.split()).replace("|", "\\|")
        lines.append(f"| `{s.source_id}` | {s.result} | {s.audit_kind} | {s.viewed} | {s.mislabelled} | {s.unusable} | "
                     f"{s.date} | {finding} |")
    lines += ["", "Sheet and sample digests (sha256) are in `audit.yaml`; the sheets themselves stay under "
              "`backend/data/vbench/audit/` and are rebuilt by `make vbench-audit`."]
    return "\n".join(lines) + "\n"
