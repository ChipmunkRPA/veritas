"""Use complete verified original corpus files after dataset-server rate limits.

The original partial acquisition remains unchanged. Reserved speaker identities
and deterministic eligibility rules are preserved; no neural model is imported.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts.plan_audio_multisource_v8 import consumed_sources, digest
from scripts.prepare_audio_confirmation_inputs_v7 import store
from scripts.prepare_audio_confirmation_v7 import text_keys
from scripts.prepare_audio_guard_v7 import write


class PinnedRows:
    def __init__(self, paths, index):
        import pyarrow.parquet as pq

        self.files = {name: pq.ParquetFile(path) for name, path in paths.items()}
        self.offsets = {name: min(r["row_index"] for r in index if r["parquet"] == name) for name in paths}
        self.ends = {
            name: np.cumsum(
                [reader.metadata.row_group(i).num_rows for i in range(reader.num_row_groups)]
            ).tolist()
            for name, reader in self.files.items()
        }
        self.cache = OrderedDict()

    def get(self, expected):
        name = expected["parquet"]
        local = expected["row_index"] - self.offsets[name]
        ends = self.ends[name]
        if not 0 <= local < ends[-1]:
            raise ValueError("Pinned corpus row offset outside original file")
        group = bisect.bisect_right(ends, local)
        offset = local - (ends[group - 1] if group else 0)
        key = (name, group)
        if key not in self.cache:
            self.cache[key] = self.files[name].read_row_group(
                group, columns=["id", "speaker_id", "text", "audio"]
            )
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        row = self.cache[key].slice(offset, 1).to_pylist()[0]
        if (
            row["id"] != expected["id"]
            or str(row["speaker_id"]) != str(expected["speaker_id"])
            or row["text"] != expected["text"]
        ):
            raise ValueError("Original pinned row identity/speaker/text differs")
        raw = row["audio"]["bytes"]
        if not isinstance(raw, bytes) or len(raw) > 16_000_000:
            raise ValueError("Bounded embedded original audio required")
        return raw, {
            "parquet": name,
            "parquet_row_group": group,
            "parquet_group_row": offset,
            "parquet_local_row": local,
        }


def acquire(root, output):
    plan_path = root / "source-planning/data-plan.json"
    plan = json.loads(plan_path.read_text())
    ai_path = root / "source-planning/reserved-ai-sources.json"
    ai = json.loads(ai_path.read_text())
    if ai["data_plan_sha256"] != digest(plan_path):
        raise ValueError("Fixed reservations differ")
    download_path = root / "source-planning/libri-acquisition-plan.json"
    download = json.loads(download_path.read_text())
    archive_root = root / "reserved-sources/libri-parquet"
    receipt_path = archive_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["plan_sha256"] != digest(download_path) or len(receipt["archives"]) != 14:
        raise ValueError("All fourteen pinned corpus files required")
    specs = {r["name"]: r for r in download["assets"]}
    paths = {}
    for row in receipt["archives"]:
        spec = specs[row["name"]]
        path = archive_root / row["name"]
        if (
            row["status"] != "verified"
            or not row["publisher_sha256_verified"]
            or digest(path) != spec["sha256"]
        ):
            raise ValueError("Original corpus publisher checksum failed")
        paths[spec["source_path"]] = path
    data = root.parent
    index = json.loads((data / "audio-v3/metadata/libri-train100-index.json").read_text())
    reader = PinnedRows(paths, index)
    prior_path = root / "source-planning/genuine-acquisition.jsonl"
    prior_rows = (
        [json.loads(line) for line in prior_path.read_text().splitlines()][1:] if prior_path.exists() else []
    )
    prior = {r["source"]["speaker_id"]: r["source"] for r in prior_rows if r["status"] == "selected"}
    previous_sources = consumed_sources(data)
    for version in ("audio", "audio-v2"):
        previous_sources.extend(json.loads((data / version / "manifest.json").read_text())["base_files"])
    previous_sources.extend(
        json.loads((data / "audio-v7/confirmation/prepared-sources.json").read_text())["sources"]
    )
    seen_hashes = {r["sha256"] for r in previous_sources + ai["sources"]}
    seen_pcm = {
        r[k]
        for r in previous_sources + ai["sources"]
        for k in (
            "production_pcm_sha256",
            "source_prefix_pcm_sha256",
            "prefix_pcm_sha256",
            "decoded_pcm_sha256",
        )
        if r.get(k)
    }
    seen_text = set().union(*(text_keys(r.get("transcript") or "") for r in previous_sources + ai["sources"]))
    sources_root = root / "reserved-sources"
    journal_path = root / "source-planning/genuine-parquet-acquisition.jsonl"
    header = {
        "schema": "chipmunk.audio-v8-genuine-original-acquisition/1",
        "data_plan_sha256": digest(plan_path),
        "ai_sources_sha256": digest(ai_path),
        "download_receipt_sha256": digest(receipt_path),
        "previous_partial_journal_sha256": digest(prior_path) if prior_path.exists() else None,
        "code_sha256": digest(Path(__file__)),
        "planned_speakers": 78,
    }
    selected, rejections, parity = [], [], []
    with journal_path.open("x") as journal:
        journal.write(json.dumps(header) + "\n")
        journal.flush()
        for voice in plan["new_genuine_speaker_reservations"]:
            existing = prior.get(voice["speaker_id"])
            candidates = (
                [r for r in voice["ordered_candidates"] if "libri:" + r["id"] == existing["id"]]
                if existing
                else voice["ordered_candidates"]
            )
            for expected in candidates:
                keys = text_keys(expected["text"])
                reason = "known text/prefix connection" if keys & seen_text else None
                source = None
                if reason is None:
                    raw, origin = reader.get(expected)
                    ref = store(sources_root, raw)
                    decoded = decode_audio(sources_root / ref["path"], source_sha256=ref["sha256"])
                    prefix = decoded.pcm[:64000]
                    prefix_sha = hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest()
                    if len(decoded.pcm) < voice["minimum_seconds"] * 16000:
                        reason = "shorter than role's fixed source duration"
                    elif float(np.sqrt(np.mean(prefix.astype(float) ** 2))) < 1e-4:
                        reason = "silent fixed prefix"
                    elif ref["sha256"] in seen_hashes or prefix_sha in seen_pcm:
                        reason = "source or prefix already used"
                    else:
                        source = {
                            **ref,
                            **origin,
                            "id": "libri:" + expected["id"],
                            "speaker_id": voice["speaker_id"],
                            "role": voice["role"],
                            "label": 0,
                            "generator": "human",
                            "dataset": "openslr/librispeech_asr",
                            "revision": download["revision"],
                            "voice_key": "libri-speaker:" + voice["speaker_id"],
                            "source_group": "libri:" + expected["id"],
                            "transcript": expected["text"],
                            "row_index": expected["row_index"],
                            "url": "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/"
                            + download["revision"]
                            + "/"
                            + origin["parquet"],
                            "license": "CC-BY-4.0",
                            "license_url": "https://www.openslr.org/12",
                            "production_pcm_sha256": decoded.mapping["pcm_sha256"],
                            "prefix_pcm_sha256": prefix_sha,
                            "full_samples": len(decoded.pcm),
                            "source_production_input": decoded.mapping,
                            "publisher_whole_parquet_sha256_verified": True,
                        }
                        if existing:
                            if (
                                source["production_pcm_sha256"] != existing["production_pcm_sha256"]
                                or prefix_sha != existing["prefix_pcm_sha256"]
                            ):
                                raise ValueError(
                                    "Original/source-server PCM differ; preserve selections and inspect before fitting"
                                )
                            source["previous_server_source_sha256"] = existing["sha256"]
                            source["previous_server_pcm_parity"] = True
                            parity.append(voice["speaker_id"])
                if reason:
                    record = {
                        "status": "ineligible",
                        "speaker_id": voice["speaker_id"],
                        "candidate_id": expected["id"],
                        "reason": reason,
                    }
                    rejections.append(record)
                else:
                    record = {"status": "selected", "source": source}
                    selected.append(source)
                    seen_hashes.add(source["sha256"])
                    seen_pcm.add(source["prefix_pcm_sha256"])
                    seen_text.update(keys)
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                if source is not None:
                    if len(selected) % 5 == 0 or len(selected) == 78:
                        print(
                            json.dumps(
                                {
                                    "selected_speakers": len(selected),
                                    "planned": 78,
                                    "ineligible_recordings": len(rejections),
                                    "neural_inference_run": False,
                                }
                            ),
                            flush=True,
                        )
                    break
            else:
                raise ValueError("Reserved speaker has no fixed eligible source; do not replace identities")
    if len(selected) != 78 or len(parity) != len(prior):
        raise ValueError("Every reserved identity and previous source parity required")
    result = {
        **header,
        "schema": "chipmunk.audio-v8-genuine-materialization/2",
        "sources": selected,
        "acquisition_journal_sha256": digest(journal_path),
        "original_server_pcm_parities": len(parity),
        "ineligible_recordings": len(rejections),
        "neural_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
    }
    write(output, result)
    print(
        json.dumps(
            {
                "sources": len(selected),
                "original_server_pcm_parities": len(parity),
                "materialization_sha256": digest(output),
                "neural_inference_run": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.root, args.output)
