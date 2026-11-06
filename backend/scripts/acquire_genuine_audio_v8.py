"""Acquire the78 reserved genuine identities with deterministic header eligibility.

Only pinned dataset-server sources, original bytes and production decoding.
No classifier, VAD, deepfake response, customer upload or fitted parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts.plan_audio_multisource_v8 import consumed_sources, digest
from scripts.prepare_audio_confirmation_inputs_v7 import store
from scripts.prepare_audio_confirmation_v7 import text_keys
from scripts.prepare_audio_guard_v7 import write


def request(url, maximum):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Chipmunk-Verify-public-evaluation/1"})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read(maximum + 1)
            if len(raw) > maximum:
                raise ValueError("Public source exceeds fixed byte budget")
            return raw
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Public source HTTP status {error.code}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RuntimeError(
                    "Public source transport unavailable; retain same source selection"
                ) from None
        time.sleep(2**attempt)
    raise RuntimeError("Public source retrieval incomplete")


def source_asset(expected, revision):
    params = {
        "dataset": "openslr/librispeech_asr",
        "config": "clean",
        "split": "train.100",
        "offset": expected["row_index"],
        "length": 1,
    }
    response = json.loads(
        request("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params), 2_000_000)
    )
    if len(response["rows"]) != 1:
        raise ValueError("Exactly one reserved metadata row required")
    row = response["rows"][0]["row"]
    if (
        row["id"] != expected["id"]
        or str(row["speaker_id"]) != str(expected["speaker_id"])
        or row["text"] != expected["text"]
    ):
        raise ValueError("Dataset row identity/speaker/text changed")
    url = row["audio"][0]["src"]
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname == "huggingface.co" or parsed.hostname.endswith(".huggingface.co"))
        or revision not in url
    ):
        raise ValueError("Source asset must identify the exact pinned public corpus revision")
    return request(url, 16_000_000), url.split("?", 1)[0]


def acquire(root, output):
    plan_path = root / "source-planning/data-plan.json"
    plan = json.loads(plan_path.read_text())
    reserved = json.loads((root / "source-planning/reserved-ai-sources.json").read_text())
    if reserved["data_plan_sha256"] != digest(plan_path):
        raise ValueError("AI and genuine identity reservations differ")
    data = root.parent
    revision = json.loads((data / "audio-v3/research/openslr_librispeech_asr-publisher.json").read_text())[
        "sha"
    ]
    source_root = root / "reserved-sources"
    journal = root / "source-planning/genuine-acquisition.jsonl"
    header = {
        "schema": "chipmunk.audio-v8-genuine-acquisition/1",
        "data_plan_sha256": digest(plan_path),
        "reserved_ai_sha256": digest(root / "source-planning/reserved-ai-sources.json"),
        "code_sha256": digest(Path(__file__)),
        "source_revision": revision,
        "planned_speakers": 78,
    }
    records = []
    if journal.exists():
        previous, *records = [json.loads(line) for line in journal.read_text().splitlines()]
        if previous != header:
            raise ValueError("Cannot resume altered source selection or acquisition implementation")
    else:
        with journal.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    chosen = [r["source"] for r in records if r["status"] == "selected"]
    expected_voices = plan["new_genuine_speaker_reservations"]
    if [r["speaker_id"] for r in chosen] != [r["speaker_id"] for r in expected_voices[: len(chosen)]]:
        raise ValueError("Acquisition receipt must be an exact selected-identity prefix")
    previous_sources = consumed_sources(data)
    for version in ("audio", "audio-v2"):
        previous_sources.extend(json.loads((data / version / "manifest.json").read_text())["base_files"])
    previous_sources.extend(
        json.loads((data / "audio-v7/confirmation/prepared-sources.json").read_text())["sources"]
    )
    seen_hashes = {r["sha256"] for r in previous_sources + reserved["sources"]}
    seen_pcm = {
        r[k]
        for r in previous_sources + reserved["sources"]
        for k in (
            "production_pcm_sha256",
            "source_prefix_pcm_sha256",
            "prefix_pcm_sha256",
            "decoded_pcm_sha256",
        )
        if r.get(k)
    }
    seen_text = set().union(
        *(text_keys(r.get("transcript") or "") for r in previous_sources + reserved["sources"])
    )
    for row in chosen:
        if digest(source_root / row["path"]) != row["sha256"]:
            raise ValueError("Previously acquired genuine source changed")
        seen_hashes.add(row["sha256"])
        seen_pcm.add(row["prefix_pcm_sha256"])
        seen_text.update(text_keys(row["transcript"]))
    rejected = {r["candidate_id"] for r in records if r["status"] == "ineligible"}
    with journal.open("a") as stream:
        for voice in expected_voices[len(chosen) :]:
            for candidate in voice["ordered_candidates"]:
                if candidate["id"] in rejected:
                    continue
                keys = text_keys(candidate["text"])
                reason = "known text/prefix connection" if keys & seen_text else None
                source = None
                if reason is None:
                    raw, url = source_asset(candidate, revision)
                    ref = store(source_root, raw)
                    decoded = decode_audio(source_root / ref["path"], source_sha256=ref["sha256"])
                    prefix = decoded.pcm[:64000]
                    pcm_sha = hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest()
                    if len(decoded.pcm) < voice["minimum_seconds"] * 16000:
                        reason = "shorter than role's fixed source duration"
                    elif float(np.sqrt(np.mean(prefix.astype(float) ** 2))) < 1e-4:
                        reason = "silent fixed source prefix"
                    elif ref["sha256"] in seen_hashes or pcm_sha in seen_pcm:
                        reason = "source/prefix already consumed or selected"
                    else:
                        source = {
                            **ref,
                            "id": "libri:" + candidate["id"],
                            "speaker_id": voice["speaker_id"],
                            "role": voice["role"],
                            "label": 0,
                            "generator": "human",
                            "dataset": "openslr/librispeech_asr",
                            "revision": revision,
                            "voice_key": "libri-speaker:" + voice["speaker_id"],
                            "source_group": "libri:" + candidate["id"],
                            "url": url,
                            "config": "clean",
                            "upstream_split": "train.100",
                            "row_index": candidate["row_index"],
                            "source_parquet": candidate["parquet"],
                            "transcript": candidate["text"],
                            "license": "CC-BY-4.0",
                            "license_url": "https://www.openslr.org/12",
                            "production_pcm_sha256": decoded.mapping["pcm_sha256"],
                            "prefix_pcm_sha256": pcm_sha,
                            "full_samples": len(decoded.pcm),
                            "source_production_input": decoded.mapping,
                            "publisher_whole_parquet_sha256_verified": False,
                            "integrity_scope": "Pinned revision/row/speaker/text checked through public dataset-server asset; original received bytes/PCM hashed. Not a full parquet/official archive checksum verification.",
                        }
                if reason:
                    entry = {
                        "status": "ineligible",
                        "speaker_id": voice["speaker_id"],
                        "role": voice["role"],
                        "candidate_id": candidate["id"],
                        "reason": reason,
                    }
                    rejected.add(candidate["id"])
                else:
                    entry = {"status": "selected", "candidate_id": candidate["id"], "source": source}
                    chosen.append(source)
                    seen_hashes.add(source["sha256"])
                    seen_pcm.add(source["prefix_pcm_sha256"])
                    seen_text.update(keys)
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                if source is not None:
                    if len(chosen) % 5 == 0 or len(chosen) == 78:
                        print(
                            json.dumps(
                                {
                                    "selected_speakers": len(chosen),
                                    "planned": 78,
                                    "ineligible_recordings": len(rejected),
                                    "neural_inference_run": False,
                                }
                            ),
                            flush=True,
                        )
                    break
            else:
                raise ValueError(
                    "A reserved identity exhausted fixed eligible candidates; preserve journal and stop"
                )
    if len(chosen) != 78:
        raise ValueError("Every reserved genuine identity is required")
    result = {
        **header,
        "schema": "chipmunk.audio-v8-genuine-materialization/1",
        "sources": chosen,
        "acquisition_journal_sha256": digest(journal),
        "ineligible_recordings": len(rejected),
        "neural_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
    }
    write(output, result)
    print(
        json.dumps(
            {
                "selected_speakers": len(chosen),
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
