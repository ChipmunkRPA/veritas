"""Keep valid frozen waveforms and rebuild only changed-role source recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from scripts.audio_source_identity_v9 import digest, role_violations
from scripts.plan_audio_multisource_v8 import stable
from scripts.prepare_audio_confirmation_variants_v7 import mixtures
from scripts.prepare_audio_multisource_variants_v8 import background_pools


def pcm_hash(pcm):
    return hashlib.sha256(np.asarray(pcm, "<f4").tobytes()).hexdigest()


def changed_role_background_index(source_id, kind, count):
    return int(hashlib.sha256(("v9-background/" + source_id + "/" + kind).encode()).hexdigest(), 16) % count


def prepare(root):
    plan_path = root / "source-plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["planning_code_sha256"] != digest(Path(__file__).with_name("plan_audio_identity_v9.py")) or plan[
        "identity_code_sha256"
    ] != digest(Path(__file__).with_name("audio_source_identity_v9.py")):
        raise ValueError("Frozen identity/role planning code changed")
    if role_violations(plan["sources"] + plan["backgrounds"]):
        raise ValueError("Known speaker connection crosses a repaired role/group")
    old_root = Path(plan["original_source_plan_root"])
    old, old_pins = {}, {}
    for name in ("development", "confirmation"):
        path = old_root / name / "variants.json"
        old_pins[name] = digest(path)
        for row in json.loads(path.read_text())["variants"]:
            old[row["base_id"], row["condition"]] = row
    for name in ("development", "confirmation"):
        if (root / name / "variants.json").exists():
            raise ValueError("Keep existing repaired input manifests")
    pools, background_pcm = background_pools(plan["backgrounds"]), {}
    for row in plan["backgrounds"]:
        path = Path(row["source_path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Fixed background bytes changed")
        audio = decode_audio(path, source_sha256=row["sha256"])
        if audio.mapping["pcm_sha256"] != row["production_pcm_sha256"]:
            raise ValueError("Fixed background production decode changed")
        background_pcm[row["sha256"]] = audio.pcm
    (root / "inputs").mkdir(exist_ok=True)
    rows, counts = [], Counter()
    conditions = (
        "clean",
        "noise_0",
        "noise_5",
        "noise_15",
        "music_5",
        "music_15",
        "speech_0",
        "speech_5",
        "speech_15",
    )
    for index, source in enumerate(plan["sources"], 1):
        previous = [old[source["id"], condition] for condition in conditions]
        if any(r["source_sha256"] != source["sha256"] or r["role"] != source["prior_role"] for r in previous):
            raise ValueError("Prior exact input source/role binding differs")
        retained = source["recipe_policy"] == "retain-exact-v8"
        if retained:
            if source["prior_role"] != source["role"]:
                raise ValueError("A moved role cannot retain another role's backgrounds")
            for row in previous:
                path = old_root / row["path"]
                if (
                    digest(path) != row["sha256"]
                    or pcm_hash(read_wav(path.read_bytes())) != row["pcm_sha256"]
                ):
                    raise ValueError("Prior exact WAV/PCM changed")
                target = root / "inputs" / (row["sha256"] + ".wav")
                if not target.exists():
                    os.link(path, target)
                if digest(target) != row["sha256"]:
                    raise ValueError("Retained input bytes changed")
                rows.append(
                    {
                        **row,
                        "role": source["role"],
                        "group": source["group"],
                        "path": str(target.relative_to(root)),
                        "retained_exact_v8_input": True,
                        "prior_input_sha256": row["sha256"],
                    }
                )
                counts["retained"] += 1
        else:
            path = Path(source["source_path"])
            if digest(path) != source["sha256"]:
                raise ValueError("Changed-role original bytes differ")
            audio = decode_audio(path, source_sha256=source["sha256"])
            if (
                audio.mapping["pcm_sha256"] != source["production_pcm_sha256"]
                or pcm_hash(audio.pcm[:64000]) != source["prefix_pcm_sha256"]
            ):
                raise ValueError("Changed-role original production prefix differs")
            selected, views = {}, {}
            for kind in ("noise", "music", "speech"):
                pool = pools[source["role"], kind]
                bg = pool[changed_role_background_index(source["id"], kind, len(pool))]
                full = background_pcm[bg["sha256"]]
                start = int(stable(source["id"] + kind + ":offset")[:8], 16) % (len(full) - 64000 + 1)
                views[kind] = full[start : start + 64000]
                selected[kind] = (bg, start)
            values, gain = mixtures(audio.pcm[:64000], views)
            clean_decoded = None
            for previous_row in previous:
                condition = previous_row["condition"]
                payload = values[condition]
                raw = wav_bytes(payload)
                sha = hashlib.sha256(raw).hexdigest()
                target = root / "inputs" / (sha + ".wav")
                if not target.exists():
                    with target.open("xb") as stream:
                        stream.write(raw)
                elif target.read_bytes() != raw:
                    raise ValueError("Existing rebuilt waveform differs")
                decoded = decode_audio(target, source_sha256=sha)
                if len(decoded.pcm) != 64000:
                    raise ValueError("Rebuilt input sample clock changed")
                if condition == "clean":
                    clean_decoded = decoded.pcm
                kind = condition.split("_")[0]
                bg, start = selected[kind] if kind != "clean" else (None, None)
                measured = (
                    None
                    if bg is None
                    else float(
                        20
                        * np.log10(
                            np.sqrt(np.mean(clean_decoded.astype(float) ** 2))
                            / np.sqrt(np.mean((decoded.pcm.astype(float) - clean_decoded) ** 2))
                        )
                    )
                )
                rows.append(
                    {
                        **previous_row,
                        "role": source["role"],
                        "group": source["group"],
                        "path": str(target.relative_to(root)),
                        "sha256": sha,
                        "payload_pcm_sha256": pcm_hash(payload),
                        "pcm_sha256": decoded.mapping["pcm_sha256"],
                        "production_input": decoded.mapping,
                        "common_peak_gain": gain,
                        "background_id": bg["source_id"] if bg else None,
                        "background_role": source["role"] if bg else None,
                        "background_kind": kind if bg else None,
                        "background_sha256": bg["sha256"] if bg else None,
                        "background_start_sample": start,
                        "background_interval_pcm_sha256": pcm_hash(views[kind]) if bg else None,
                        "measured_snr_db": measured,
                        "retained_exact_v8_input": False,
                        "prior_input_sha256": previous_row["sha256"],
                    }
                )
                counts["regenerated"] += 1
        if index % 100 == 0 or index == len(plan["sources"]):
            print(
                json.dumps({"foregrounds": index, "planned": len(plan["sources"]), "counts": dict(counts)}),
                flush=True,
            )
    if (
        len(rows) != 5328
        or len({r["id"] for r in rows}) != 5328
        or counts != {"retained": 5274, "regenerated": 54}
    ):
        raise ValueError("Every repaired source/condition must remain accounted for")
    for name in ("development", "confirmation"):
        selected = [r for r in rows if (r["role"] == "confirmation") == (name == "confirmation")]
        path = root / name / "variants.json"
        path.parent.mkdir(exist_ok=True)
        body = {
            "schema": "chipmunk.audio-identity-exact-variants/9",
            "panel": name,
            "source_plan_sha256": digest(plan_path),
            "preparation_code_sha256": digest(Path(__file__)),
            "old_variants_sha256": old_pins,
            "mixture_code_sha256": digest(
                Path(__file__).with_name("prepare_audio_confirmation_variants_v7.py")
            ),
            "variants": selected,
            "source_root": str(root.resolve()),
            "new_neural_inference_run": False,
            "deployment_eligible": False,
        }
        with path.open("x") as stream:
            json.dump(body, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "panel": name,
                    "inputs": len(selected),
                    "role_counts": dict(Counter(r["role"] for r in selected)),
                    "variants_sha256": digest(path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.root)
