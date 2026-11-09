"""Qualify role-bound original audio and prepare matched synthesis prompts; no models."""
from __future__ import annotations

import hashlib
import json
import tarfile
import time
from collections import Counter, OrderedDict
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq

from app.detectors.audio.decode import decode_audio
from scripts.audio_calibration_roles_v71 import ROLES, assert_voice_roles
from scripts.audio_expressive_inventory_v66 import sha, write
from scripts.audio_genuine_sources_v67 import DESTINATION as GENUINE_ROOT

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-source-qualification-v72"
DESTINATION = GENUINE_ROOT.parent / "qualified-v72"
RECIPE = BASE / "audio-calibration-roles-v71/plan.json"
RECIPE_SHA = "3c45cdbb89b617fdda938bee6b0b7ca63aa516d1b42b73ffc7c85031428bd977"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def prefix_hash(pcm, count):
    return hashlib.sha256(np.asarray(pcm[:count], "<f4").tobytes()).hexdigest()


def eligibility(pcm, minimum, full_sha, prefix_sha, raw_sha, history, used_bytes, used_pcm, *, allow_history=False):
    if len(pcm) < minimum:
        return "shorter_than_frozen_minimum"
    if not np.isfinite(pcm).all():
        raise ValueError("Nonfinite production source PCM")
    if float(np.sqrt(np.mean(pcm[:minimum].astype(np.float64) ** 2))) < 1e-4:
        return "silent_fixed_prefix"
    if raw_sha in used_bytes or full_sha in used_pcm or prefix_sha in used_pcm:
        return "duplicate_selected_bytes_or_pcm"
    if not allow_history and ({"bytes:" + raw_sha, "pcm:" + full_sha, "pcm:" + prefix_sha} & history):
        return "recorded_historical_bytes_or_pcm"
    return None


class Sources:
    def __init__(self):
        self.readers, self.groups, self.archives = {}, OrderedDict(), {}

    def close(self):
        for archive in self.archives.values():
            archive.close()

    def original(self, kind, row):
        if kind in {"genuine", "speech"}:
            path = GENUINE_ROOT / row["parquet"]
            if path.name not in self.readers:
                require(sha(path) == row["parquet_sha256"], "Complete genuine corpus checksum changed")
                self.readers[path.name] = pq.ParquetFile(path)
            key = (path.name, row["parquet_row_group"])
            if key not in self.groups:
                self.groups[key] = self.readers[path.name].read_row_group(key[1], columns=["id", "speaker_id", "text", "audio"])
                while len(self.groups) > 2:
                    self.groups.popitem(last=False)
            self.groups.move_to_end(key)
            value = self.groups[key].slice(row["parquet_group_row"], 1).to_pylist()[0]
            require("libri:" + value["id"] == row["id"] and value["text"] == row["transcript"]
                    and "librispeech:" + str(value["speaker_id"]) == row["canonical_speaker"],
                    "Original corpus row identity/text/parent differs")
            raw = value["audio"]["bytes"]
        elif kind in {"clone", "expressive"}:
            base = BASE / ("audio-v7/fresh-sources/archives" if kind == "clone" else "audio-v8/reserved-sources/archives")
            path = base / row["archive"]
            if str(path) not in self.archives:
                expected = row.get("archive_sha256")
                if expected is None:
                    receipt = json.loads((base / "receipt.json").read_text())
                    expected = next(item["sha256"] for item in receipt["archives"] if item["name"] == row["archive"])
                require(sha(path) == expected, "Original synthesis archive checksum changed")
                archive = tarfile.open(path, "r:")
                archive.getmembers()
                self.archives[str(path)] = archive
            archive = self.archives[str(path)]
            name = PurePosixPath(row["member"])
            require(not name.is_absolute() and ".." not in name.parts, "Safe archived original path required")
            member = archive.getmember(str(name))
            require(member.isfile() and 0 < member.size <= 16_000_000, "Bounded original audio member required")
            with archive.extractfile(member) as stream:
                raw = stream.read(16_000_001)
            require(len(raw) == member.size, "Complete original audio payload required")
        else:
            path = Path(row["source_path"])
            require(sha(path) == row["sha256"], "Original background bytes changed")
            raw = path.read_bytes()
        require(isinstance(raw, bytes) and 0 < len(raw) <= 16_000_000, "Bounded complete original audio required")
        if row.get("sha256"):
            require(hashlib.sha256(raw).hexdigest() == row["sha256"], "Selected original source hash differs")
        return raw


def qualify():
    require(sha(RECIPE) == RECIPE_SHA, "Prospective role/candidate recipe changed")
    plan = json.loads(RECIPE.read_text())
    require(all(sha(ROOT / path) == value for path, value in plan["input_bindings"].items()),
            "Frozen source input metadata changed")
    require(not TRIAL.exists() and not DESTINATION.exists(), "Never qualified destination required")
    TRIAL.mkdir()
    (DESTINATION / "sources").mkdir(parents=True)
    history = set(json.loads((BASE / "audio-history-v58/catalogue.json").read_text())["keys"])
    sources, selected, rejected = Sources(), [], []
    used_bytes, used_pcm, used_text = set(), set(), set()
    pending = [group for group in plan["source_groups"] if group["kind"] == "stock_generation"]
    explicit = [group for group in plan["source_groups"] if group["kind"] in {"clone", "expressive"}]
    # Reserve all exact selected synthesis scripts before choosing from ordered
    # genuine pools; no later source may take an explicitly committed script.
    reserved_scripts = {key for group in explicit for row in group["selected_candidates"] for key in row["text_identity"]}
    started = time.monotonic()
    with (TRIAL / "qualification-journal.jsonl").open("x") as journal:
        try:
            ordered = [*explicit, *[g for g in plan["source_groups"] if g["kind"] == "genuine"], *plan["backgrounds"]]
            for group_index, group in enumerate(ordered):
                kind = group["kind"]
                minimum = 64000 if kind in {"genuine", "clone", "expressive"} else 128000
                candidates = group.get("selected_candidates") or group.get("ordered_candidates") or [group["selected_candidate"]]
                chosen = []
                for row in candidates:
                    record = {"kind": kind, "role": group["role"], "source_id": row.get("id", group.get("source_id"))}
                    keys = set(row.get("text_identity", []))
                    reason = "selected_declared_script_or_prefix_reuse" if keys & used_text else None
                    if kind == "genuine" and keys & reserved_scripts:
                        reason = "script_reserved_for_fixed_synthesis_source"
                    if reason is None:
                        raw = sources.original(kind, row)
                        raw_sha = hashlib.sha256(raw).hexdigest()
                        path = DESTINATION / "sources" / (raw_sha + ".audio")
                        if not path.exists():
                            with path.open("xb") as stream:
                                stream.write(raw)
                        else:
                            require(sha(path) == raw_sha, "Staged original bytes changed")
                        audio = decode_audio(path, source_sha256=raw_sha)
                        full_sha = audio.mapping["pcm_sha256"]
                        fixed_sha = prefix_hash(audio.pcm, minimum)
                        first4_sha = prefix_hash(audio.pcm, 64000)
                        reason = eligibility(audio.pcm, minimum, full_sha, fixed_sha, raw_sha, history, used_bytes, used_pcm,
                                             allow_history=kind == "music")
                        if (reason is None and minimum > 64000 and kind != "music"
                                and (first4_sha in used_pcm or "pcm:" + first4_sha in history)):
                            reason = "selected_or_historical_four_second_prefix"
                        record.update(raw_sha256=raw_sha, samples=len(audio.pcm), pcm_sha256=full_sha,
                                      prefix_pcm_sha256=fixed_sha, first4_pcm_sha256=first4_sha)
                    if reason is not None:
                        record.update(status="ineligible", reason=reason)
                        rejected.append(record)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        require(kind in {"genuine", "speech"}, "Explicitly committed source failed eligibility; preserve and stop")
                        continue
                    value = {**row, **record, "status": "ready", "label": 0 if kind in {"genuine", "speech", "noise", "music"} else 1,
                             "canonical_speaker": group.get("canonical_speaker"),
                             "source_path": str(path), "source_bytes": len(raw),
                             "production_input": audio.mapping, "minimum_samples": minimum,
                             "study_group": group.get("canonical_speaker", group.get("source_id")),
                             "foreground": kind in {"genuine", "clone", "expressive"}}
                    chosen.append(value)
                    selected.append(value)
                    used_bytes.add(raw_sha)
                    used_pcm.update({full_sha, fixed_sha, first4_sha})
                    used_text.update(keys)
                    journal.write(json.dumps({**record, "status": "ready"}, allow_nan=False) + "\n")
                    journal.flush()
                    if len(chosen) == group["required_sources"]:
                        break
                require(len(chosen) == group["required_sources"], "Required fixed voice source count unavailable")
                if (group_index + 1) % 10 == 0:
                    print(json.dumps({"qualified_groups": group_index + 1, "planned_groups": len(ordered),
                                      "qualified_originals": len(selected), "rejections": len(rejected)}), flush=True)
        except BaseException as error:
            write(TRIAL / "failure.json", {"type": type(error).__name__, "detail": str(error)[:300],
                                         "selected_originals": len(selected), "rejected_candidates": len(rejected),
                                         "new_model_calls": 0, "code_sha256": sha(Path(__file__))})
            raise
        finally:
            sources.close()
    foregrounds = [row for row in selected if row["foreground"]]
    backgrounds = [row for row in selected if not row["foreground"]]
    require(len(foregrounds) == 400 and len(backgrounds) == 48, "Complete raw source panel required")
    prompts = []
    for group in pending:
        parents = sorted([row for row in foregrounds if row["canonical_speaker"] == group["matched_genuine_voice"]], key=lambda r: r["id"])
        require(len(parents) == 4 and {row["role"] for row in parents} == {group["role"]}, "Four exact within-role genuine prompt parents required")
        for parent in parents:
            prompts.append({"role": group["role"], "canonical_speaker": group["canonical_speaker"],
                            "voice_asset": group["asset"], "voice_asset_sha256": group["sha256"],
                            "language": group["language"], "transcript": parent["transcript"],
                            "matched_genuine_source_id": parent["id"], "study_group": parent["study_group"],
                            "matched_genuine_pcm_sha256": parent["pcm_sha256"], "label": 1, "generator": "Kokoro-82M"})
    require(len(prompts) == 64, "Every prespecified matched synthesis prompt required")
    assert_voice_roles([*foregrounds, *[row for row in backgrounds if row["kind"] == "speech"],
                        *[{**row, "matched_genuine_voice": row["study_group"]} for row in prompts]])
    manifest = {"schema": "chipmunk.audio-confidence-qualified-originals/72", "source_recipe_sha256": RECIPE_SHA,
                "foregrounds": foregrounds, "backgrounds": backgrounds, "pending_matched_generation": prompts,
                "code_sha256": sha(Path(__file__)), "new_model_calls": 0}
    write(TRIAL / "manifest.json", manifest)
    summary = {"schema": "chipmunk.audio-confidence-source-qualification-summary/72",
               "qualified_original_foregrounds": len(foregrounds), "qualified_backgrounds": len(backgrounds),
               "pending_matched_synthetic_sources": len(prompts), "roles": {},
               "rejected_candidates": dict(Counter(row["reason"] for row in rejected)),
               "source_recipe_sha256": RECIPE_SHA, "manifest_sha256": sha(TRIAL / "manifest.json"),
               "qualification_journal_sha256": sha(TRIAL / "qualification-journal.jsonl"),
               "code_sha256": sha(Path(__file__)), "duration_s": time.monotonic() - started,
               "new_model_calls": 0, "calibration_fitted": False,
               "scope": "Complete original bytes and production PCM for role-bound sources, not detector quality.64 matched synthesis sources remain pending a separately published generator freeze. Source IDs/text/PCM hashes remain private. Four-second foreground scope; shorter natural utterances remain unqualified."}
    for role in ROLES:
        summary["roles"][role] = {"foregrounds": dict(Counter(row["kind"] for row in foregrounds if row["role"] == role)),
                                  "backgrounds": dict(Counter(row["kind"] for row in backgrounds if row["role"] == role)),
                                  "pending_stock_sources": sum(row["role"] == role for row in prompts)}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    qualify()
