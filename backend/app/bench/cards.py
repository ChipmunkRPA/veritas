"""Model-card `eval` rows from the score cache (ARCHITECTURE §2.3; BUILD_SPEC I-14, Q-1). Written by `vbench_report.py
--update-cards`, never by hand.

- **Rows.** One row per detector a fuser artefact was fitted on, per modality: slice `<modality>/<split>` over the usable items
  of that modality and split (no failed audit, no C2PA manifest) that the detector scored, without a transform (images: `norm`
  variant; video and audio: `native`). A detector fitted in two artefacts (the voice detector in the video and the audio fuser)
  gets one row per modality. τ_0.01 comes from the detector's own `cal` reals by the report's rule. `auc` and `ece` are None when
  a class is missing, `tpr_at_fpr01` when τ_0.01 is not established or there is no fake. `n` counts the scored items; values are
  rounded to 4 dp.
- **Writing.** The YAML is edited as text so the hand-written layout and comments survive: the detector block's `eval:`
  line (or block list) is replaced, or inserted after `legacy_weight` / `default_enabled`, with one JSON flow list. The
  whole manifest is then re-validated and every written row read back; on any error the file is left unchanged.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from app.bench import metrics as mt
from app.bench import report as rp

_DETECTOR = re.compile(r"^  - id: (?P<id>\S+)\s*$")
_ANCHORS = ("    legacy_weight:", "    default_enabled:")


class CardError(ValueError):
    """The model cards cannot be updated."""


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def eval_rows(manifest: Path, store: rp.ScoreStore, keys: Mapping[str, str], *, bench: str, date: str,
              split: str = "test", modality: str = "image") -> dict[str, list[dict[str, Any]]]:
    """{detector id: [one EvalRow document]} for each `<id>: <score key>` over the `modality` items."""
    variant = "norm" if modality == "image" else "native"
    items = [rp.Item.from_row(r) for r in rp.read_manifest(manifest)
             if r["modality"] == modality and r["audit_ok"] is not False and not r["has_c2pa"]]
    out: dict[str, list[dict[str, Any]]] = {}
    for detector_id, key in sorted(keys.items()):
        system = rp.ScoreSystem(store, key)
        ev, _ = rp.evaluate(system, (i for i in items if i.split == split), variant=variant)
        cal_reals, _ = rp.evaluate(system, (i for i in items if i.split == "cal" and i.y == 0), variant=variant)
        tau = rp.thresholds(cal_reals)["0.01"]["tau"] if cal_reals.items else None
        counts = ev.counts() if ev.items else {"n_real": 0, "n_fake": 0}
        both = counts["n_real"] > 0 and counts["n_fake"] > 0
        ece = mt.ece(ev.s, ev.y) if both else None
        out[detector_id] = [{"bench": bench, "slice": f"{modality}/{split}",
                             "auc": _round(mt.auc(ev.s, ev.y)) if both else None,
                             "tpr_at_fpr01": _round(mt.tpr_at(ev.s, ev.y, tau).value) if tau is not None and counts["n_fake"] else None,
                             "ece": _round(ece["ece"]) if ece else None,
                             "n": len(ev.items), "date": date}]
    return out


def merge_rows(*parts: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """One row list per detector over several `eval_rows` results, in the order given."""
    out: dict[str, list[dict[str, Any]]] = {}
    for part in parts:
        for detector_id, entries in part.items():
            out.setdefault(detector_id, []).extend(entries)
    return dict(sorted(out.items()))


def _with_eval(lines: list[str], detector_id: str, entries: list[dict[str, Any]]) -> list[str]:
    start = next((k for k, line in enumerate(lines) if (m := _DETECTOR.match(line)) and m["id"] == detector_id), None)
    if start is None:
        raise CardError(f"no detector card {detector_id} in the manifest")
    end = next((k for k in range(start + 1, len(lines))
                if lines[k].startswith("  - ") or (lines[k].strip() and not lines[k].startswith(" "))), len(lines))
    kept: list[str] = []
    k = start
    while k < end:
        if lines[k].startswith("    eval:"):
            k += 1
            while k < end and lines[k].startswith("      "):  # rows of a block-style list
                k += 1
            continue
        kept.append(lines[k])
        k += 1
    anchor = next((k for prefix in _ANCHORS for k, line in enumerate(kept) if line.startswith(prefix)), None)
    if anchor is None:
        raise CardError(f"detector card {detector_id} has no legacy_weight or default_enabled line to place eval after")
    kept.insert(anchor + 1, f"    eval: {json.dumps(entries, ensure_ascii=False)}\n")
    return lines[:start] + kept + lines[end:]


def write_eval_rows(path: Path, rows: Mapping[str, list[dict[str, Any]]]) -> None:
    from app.engine.manifest import load_manifest

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for detector_id, entries in rows.items():
        lines = _with_eval(lines, detector_id, entries)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    try:
        manifest = load_manifest(tmp)
        for detector_id, entries in rows.items():
            card = manifest.detector(detector_id)
            if card is None or [r.model_dump() for r in card.eval] != entries:
                raise CardError(f"the eval rows of {detector_id} do not read back as written")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        tmp.unlink(missing_ok=True)
        raise CardError(f"the updated manifest does not validate: {exc}") from exc
    os.replace(tmp, path)
