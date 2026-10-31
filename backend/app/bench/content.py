"""The content measurements of the bytes the bench scored (D-65 "Bench"; BUILD_SPEC N-18).

The runtime caps an image of a kind the validation set does not hold: a share of flat 8 × 8 blocks above the bench's limit,
a file that matches the screenshot rule, or a greyscale image (`abstain.content_conditions`). The condition stage measures
a served image; the score cache's recorded contexts were written before that stage measured and hold neither number. So
the report measures the files themselves, with the function the condition stage calls (`content_kinds.measure`), and
keeps what it measured:

- **Which file.** Variant `norm` is the manifest's `norm_path`, `native` its `native_path`; a transform is the `path` of
  `perturbed/manifest.parquet` for (item, transform). Paths are relative to the bench directory.
- **What is measured.** `flat_block_share`, `greyscale`, and for the screenshot rule the file's format, size and whether
  it has an EXIF block (`abstain.screenshot_likelihood`, on the file's own pixel grid: the rule accepts either
  orientation).
- **The table** (`data/vbench/features/content_kinds-scored-<bench>.parquet` by default, not in version control): one row
  per (item, variant, transform) with the file's relative path, byte count and sha256. A row is used again only while the
  file has that path and byte count. A file that is missing or does not decode has a row with its error and no
  measurement: the report counts it as not measured and applies no content rule to it.

Nothing here reads a label, a score or a split: which items are measured is decided by the report that asks.
"""
from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from app.engine.fusion import abstain, content_kinds

COLUMNS = ("item_id", "variant", "transform", "path", "bytes", "sha256", "format", "width", "height", "has_exif",
           "flat_block_share", "greyscale", "screenshot_likelihood", "error")
Key = tuple[str, str, str]


def measure_file(path: Path) -> dict[str, Any]:
    """The measurements of one file; `error` is set, and the measurements are None, when it is missing or does not decode."""
    blank = dict.fromkeys(("bytes", "sha256", "format", "width", "height", "has_exif", "flat_block_share", "greyscale",
                           "screenshot_likelihood", "error"))
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return {**blank, "error": f"{type(e).__name__}: {e}"[:200]}
    found = {**blank, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, has_exif = im.format, len(im.getexif()) > 0
        measured = content_kinds.measure(data)
    except Exception as e:  # noqa: BLE001  a file that does not decode is counted by the report, not skipped in silence
        return {**found, "error": f"{type(e).__name__}: {e}"[:200]}
    width, height = measured["width"], measured["height"]
    return {**found, "format": fmt, "width": width, "height": height, "has_exif": has_exif,
            "flat_block_share": measured["flat_block_share"], "greyscale": measured["greyscale"],
            "screenshot_likelihood": abstain.screenshot_likelihood(fmt, width, height, has_exif)}


class ScoredContent:
    """(item_id, variant, transform) → the content measurements of the scored file, measured on first use."""

    def __init__(self, bench_dir: Path, rows: Iterable[Mapping[str, Any]], *, table: Path | None = None):
        self.dir, self.table = Path(bench_dir), None if table is None else Path(table)
        self._paths: dict[str, dict[str, str | None]] = {
            r["item_id"]: {"norm": r.get("norm_path"), "native": r.get("native_path")} for r in rows}
        self._perturbed: dict[tuple[str, str], str] | None = None
        self._rows: dict[Key, dict[str, Any]] = {}
        self.measured_now = 0
        if self.table is not None and self.table.is_file():
            import pyarrow.parquet as pq

            for row in pq.read_table(self.table).to_pylist():
                self._rows[(row["item_id"], row["variant"], row["transform"])] = row

    def path_of(self, item_id: str, variant: str, transform: str) -> str | None:
        """The relative path of the scored file, None when the manifests name none."""
        if transform == "none":
            return (self._paths.get(item_id) or {}).get(variant)
        if self._perturbed is None:
            import pyarrow.parquet as pq

            manifest = self.dir / "perturbed" / "manifest.parquet"
            self._perturbed = ({(r["item_id"], r["transform"]): r["path"]
                                for r in pq.read_table(manifest, columns=["item_id", "transform", "path"]).to_pylist()}
                               if manifest.is_file() else {})
        return self._perturbed.get((item_id, transform))

    def get(self, item_id: str, variant: str, transform: str = "none") -> dict[str, Any] | None:
        """The measurements, or None when the file is not named, is missing or does not decode."""
        relative = self.path_of(item_id, variant, transform)
        if relative is None:
            return None
        key, path = (item_id, variant, transform), self.dir / relative
        row = self._rows.get(key)
        size = path.stat().st_size if path.is_file() else None
        if row is None or row["path"] != relative or row["bytes"] != size:
            row = {"item_id": item_id, "variant": variant, "transform": transform, "path": relative, **measure_file(path)}
            self._rows[key] = row
            self.measured_now += 1
        return None if row["error"] is not None else row

    def save(self) -> Path | None:
        """Write the table when something was measured in this run (atomically: a report that stops keeps the old one)."""
        if self.table is None or not self.measured_now:
            return self.table
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([("item_id", pa.string()), ("variant", pa.string()), ("transform", pa.string()), ("path", pa.string()),
                            ("bytes", pa.int64()), ("sha256", pa.string()), ("format", pa.string()), ("width", pa.int64()),
                            ("height", pa.int64()), ("has_exif", pa.bool_()), ("flat_block_share", pa.float64()),
                            ("greyscale", pa.bool_()), ("screenshot_likelihood", pa.float64()), ("error", pa.string())])
        rows = [{c: row.get(c) for c in COLUMNS} for _, row in sorted(self._rows.items())]
        self.table.parent.mkdir(parents=True, exist_ok=True)
        partial = self.table.with_suffix(".parquet.partial")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), partial)
        partial.replace(self.table)
        self.measured_now = 0
        return self.table
