"""Assemble separate noisy short-edit controls from retained licensed audio."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import audio_communication_replication as previous

FRESH, QUALIFIED, BASE = previous.FRESH, previous.QUALIFIED, previous.BASE
RATE, SAMPLES = previous.RATE, previous.SAMPLES
read, write, sha, digest, require = previous.read, previous.write, previous.sha, previous.digest, previous.require
ROOMS = ("ES2012b", "IS1001b")
LAYOUTS = (
    (("clone", 10240, "early"), ("expressive", 12800, "interior"), ("stock", 15360, "interior")),
    (("stock", 10240, "late"), ("clone", 12800, "interior"), ("expressive", 15360, "interior")),
)


def histories(meta):
    _, calibration, ids, contexts, seen, used, rooms = previous.source_roles(meta)
    prior = meta.source("earlier_communication_confirmation", "audio-communication-replication-2026-11-12/roster.json")
    require(not previous.participants(previous.ANNOTATIONS, ROOMS) & previous.participants(previous.ANNOTATIONS, FRESH.SESSIONS),
            "Held room participants overlap the calibration rooms")
    for row in prior:
        for role in ("host", "donor"):
            if witness := row.get(role):
                used.add(witness["canonical_speaker"])
                for key, kind in (("source_sha256", "bytes"), ("decoder_input_sha256", "bytes"), ("pcm_sha256", "pcm")):
                    if key in witness:
                        seen.add(f"{kind}:{witness[key]}")
        room = row["background"]["room"]
        if room in ROOMS:
            start = row["background"]["capture_start_sample"]
            rooms[room][3].append((start, start + SAMPLES))
    forbidden_profiles = {witness["canonical_speaker"] for row in calibration for role in ("host", "donor")
                          if (witness := row.get(role))}
    return ids, contexts, seen, used, rooms, forbidden_profiles, calibration, prior


def anchors(host, width, key, role):
    positions = FRESH.anchors(host, width, key)
    if role == "interior":
        # Place near a disjoint-grid boundary when admitted source context permits.
        positions.sort(key=lambda start: not any(start < boundary < start + width for boundary in (2 * RATE, 4 * RATE)))
    return positions


def prepare(folder):
    meta = QUALIFIED.sources.Metadata()
    ids, contexts, seen, used, rooms, forbidden_profiles, calibration, prior = histories(meta)
    genuine = FRESH.genuine_rows(meta, ids, seen, used)
    donors = {
        "clone": FRESH.clone_rows(meta, ids, seen),
        "expressive": FRESH.source.donors(meta, ids, seen),
        "stock": FRESH.cue.recent.stock_rows(meta, seen, ids),
    }
    prior_profiles = used & {row["voice"] for family in ("expressive", "stock") for row in donors[family]}
    allowed_reuse = prior_profiles - forbidden_profiles
    used.difference_update(allowed_reuse)
    output, texts, fragments = [], set(), set()
    with ExitStack() as stack, TemporaryDirectory(prefix="chipmunk-audio-edge-inputs-") as temporary:
        reader = QUALIFIED.source_reader.SourceReader(stack, meta.bindings)
        source = Path(temporary) / "native"
        for i, (room, design) in enumerate(zip(ROOMS, LAYOUTS, strict=True)):
            for j, (family, width, location) in enumerate(design):
                key = f"edge-grid-{i}-{j}-{location}"
                host, ordinary, ordinary_start, host_pin = FRESH.genuine(genuine, reader, source, used, seen, texts, width, key, host=True)
                _, other, other_start, other_pin = FRESH.genuine(genuine, reader, source, used, seen, texts, width, key + "-other")
                synthetic, synth_start, synth_pin = FRESH.generated(donors[family], reader, source, used, seen, texts, contexts, width, key)
                synth_pin["profile_exposed_in_other_studies"] = synth_pin["canonical_speaker"] in allowed_reuse
                start, noise, witness = FRESH.capture(rooms[room], room, anchors(host, width, key, location), width)
                noise *= np.linalg.norm(host * 32768) * 10 ** .25 / np.linalg.norm(noise)
                mixtures = [("untouched", host * 32768, None)]
                for arm, signal, offset, pin in (
                    ("ordinary_same_host", ordinary, ordinary_start, host_pin),
                    ("ordinary_other_voice", other, other_start, other_pin),
                    ("synthetic", synthetic, synth_start, synth_pin),
                ):
                    fragment = QUALIFIED.pcm_digest(signal[offset:offset + width].astype(np.float32))
                    require(fragment not in fragments and f"pcm:{fragment}" not in seen, "Prior donor fragment overlaps held input")
                    fragments.add(fragment)
                    foreground = previous.trial.OLD.inputs.splice(host, signal, start, offset, width) * 32768
                    mixtures.append((arm, foreground, pin | {"pcm_sha256": fragment}))
                gain = min(1., 31_120 / max(float(np.max(abs(pcm) + abs(noise))) for _, pcm, _ in mixtures))
                for arm, pcm, pin in mixtures:
                    FRESH.source.emit(output, f"{key}-{arm}", key, arm, [start, start + width], witness, host_pin, pin,
                                      np.rint(gain * pcm), np.rint(gain * noise))
                    output[-1][0].update(family=family, coverage=location)
            _, noise, witness = FRESH.capture(rooms[room], room, [32000], 12800)
            key = f"edge-grid-room-{i}"
            noise = np.rint(noise * min(1., 31_120 / np.max(abs(noise))))
            FRESH.source.emit(output, key, key, "background_only", None, witness, None, None, np.zeros(SAMPLES), noise)
            output[-1][0].update(family=None, coverage="room")

    records = [row for row, *_ in output]
    require(len(records) == 26 and len(fragments) == 18
            and Counter(row["family"] for row in records if row["truth"]) == dict.fromkeys(FRESH.FAMILIES, 2),
            "Held source denominator or generator balance changed")
    require(not {row["pcm_sha256"] for row in records} & {row["pcm_sha256"] for row in calibration + prior},
            "The held mixtures overlap prior original PCM")
    folder.mkdir(mode=0o700)
    (folder / "pcm").mkdir()
    for row, raw, _, _ in output:
        (folder / "pcm" / f"{row['id']}.pcm").write_bytes(raw)
    write(folder / "roster.json", records)
    write(folder / "sources.json", {
        "implementation_sha256": sha(Path(__file__)),
        "roster_sha256": sha(folder / "roster.json"),
        "annotation_sha256": sha(previous.ANNOTATIONS),
        "source_bindings": meta.bindings,
        "layout": LAYOUTS,
    })
    print(f"Sealed {len(records)} held noisy audio inputs in {len(ROOMS)} separate rooms.", flush=True)


def held(folder):
    source = read(folder / "sources.json")
    require(source["implementation_sha256"] == sha(Path(__file__))
            and source["roster_sha256"] == sha(folder / "roster.json")
            and source["annotation_sha256"] == sha(previous.ANNOTATIONS), "Frozen held sources changed")
    for row in read(folder / "roster.json"):
        raw = (folder / "pcm" / f"{row['id']}.pcm").read_bytes()
        require(len(raw) == 2 * SAMPLES and row["pcm_sha256"] == digest(raw), "Frozen held input PCM changed")
        yield row, raw
