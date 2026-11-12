"""Render disjoint speech and room inputs for the grouped short-window replication."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import audio_edge_window_sources as earlier

BASE, RATE, SAMPLES = earlier.BASE, earlier.RATE, earlier.SAMPLES
FRESH, QUALIFIED = earlier.FRESH, earlier.QUALIFIED
read, write, sha, digest, require = earlier.read, earlier.write, earlier.sha, earlier.digest, earlier.require
ROOMS = ("ES2007c", "IS1004c")
LAYOUTS = (
    (("clone", 10240, "early", -5), ("expressive", 12800, "interior", -8), ("stock", 15360, "interior", -5),
     ("expressive", 10240, "late", -5), ("stock", 12800, "interior", -8), ("clone", 15360, "interior", -8)),
    (("stock", 10240, "early", -8), ("clone", 12800, "interior", -5), ("expressive", 15360, "interior", -8),
     ("clone", 10240, "late", -8), ("expressive", 12800, "interior", -5), ("stock", 15360, "interior", -5)),
)
HELD = BASE / "audio-edge-window-grouped-held-2026-11-12"


def candidates(meta):
    ids, contexts, seen, used, rooms, forbidden, calibration, preceding = earlier.histories(meta)
    exploratory = meta.source("short_window_exploratory_held", "audio-edge-window-held-2026-11-12/roster.json")
    earlier.previous.participants(earlier.previous.ANNOTATIONS, (*ROOMS, *earlier.ROOMS, *FRESH.SESSIONS))
    for row in exploratory:
        for role in ("host", "donor"):
            if witness := row.get(role):
                forbidden.add(witness["canonical_speaker"])
                for key, kind in (("source_sha256", "bytes"), ("decoder_input_sha256", "bytes"), ("pcm_sha256", "pcm")):
                    if key in witness:
                        seen.add(f"{kind}:{witness[key]}")
    used.update(forbidden)
    genuine = FRESH.genuine_rows(meta, ids, seen, used)
    donors = {
        "clone": FRESH.clone_rows(meta, ids, seen),
        "expressive": FRESH.source.donors(meta, ids, seen),
        "stock": FRESH.cue.recent.stock_rows(meta, seen, ids),
    }
    prior_profiles = used & {row["voice"] for family in ("expressive", "stock") for row in donors[family]}
    reused = prior_profiles - forbidden
    used.difference_update(reused)
    return genuine, donors, contexts, seen, used, rooms, reused, calibration + preceding + exploratory


def prepare(folder):
    meta = QUALIFIED.sources.Metadata()
    genuine, donors, contexts, seen, used, rooms, reused, history = candidates(meta)
    output, fragments, texts = [], set(), set()
    with ExitStack() as stack, TemporaryDirectory(prefix="chipmunk-noisy-speech-inputs-") as temporary:
        reader = QUALIFIED.source_reader.SourceReader(stack, meta.bindings)
        path = Path(temporary) / "native"
        for i, (room, design) in enumerate(zip(ROOMS, LAYOUTS, strict=True)):
            for j, (family, width, coverage, snr) in enumerate(design):
                key = f"grouped-grid-{i}-{j}-{coverage}"
                host, same, same_at, host_proof = FRESH.genuine(genuine, reader, path, used, seen, texts, width, key, host=True)
                _, other, other_at, other_proof = FRESH.genuine(genuine, reader, path, used, seen, texts, width, key + "-other")
                synthetic, ai_at, ai_proof = FRESH.generated(donors[family], reader, path, used, seen, texts, contexts, width, key)
                ai_proof["profile_exposed_in_other_studies"] = ai_proof["canonical_speaker"] in reused
                start, noise, witness = FRESH.capture(rooms[room], room, earlier.anchors(host, width, key, coverage), width)
                noise *= np.linalg.norm(host * 32768) * 10 ** (-snr / 20) / np.linalg.norm(noise)
                arms = [("untouched", host * 32768, None)]
                for arm, signal, offset, proof in (
                    ("ordinary_same_host", same, same_at, host_proof),
                    ("ordinary_other_voice", other, other_at, other_proof),
                    ("synthetic", synthetic, ai_at, ai_proof),
                ):
                    fragment = QUALIFIED.pcm_digest(signal[offset:offset + width].astype(np.float32))
                    require(fragment not in fragments and f"pcm:{fragment}" not in seen, "Held fragment overlaps earlier exact audio")
                    fragments.add(fragment)
                    pcm = earlier.previous.trial.OLD.inputs.splice(host, signal, start, offset, width) * 32768
                    arms.append((arm, pcm, proof | {"pcm_sha256": fragment}))
                gain = min(1., 31_120 / max(float(np.max(abs(pcm) + abs(noise))) for _, pcm, _ in arms))
                for arm, pcm, proof in arms:
                    FRESH.source.emit(output, f"{key}-{arm}", key, arm, [start, start + width], witness, host_proof, proof,
                                      np.rint(gain * pcm), np.rint(gain * noise))
                    output[-1][0].update(family=family, coverage=coverage, snr_db=snr, room_group=i + 1)
            _, noise, witness = FRESH.capture(rooms[room], room, [32000], 12800)
            key = f"grouped-grid-room-{i}"
            noise = np.rint(noise * min(1., 31_120 / np.max(abs(noise))))
            FRESH.source.emit(output, key, key, "background_only", None, witness, None, None, np.zeros(SAMPLES), noise)
            output[-1][0].update(family=None, coverage="room", snr_db=None, room_group=i + 1)
    records = [row for row, *_ in output]
    require(len(records) == 50 and len(fragments) == 36
            and Counter((row["family"], row["snr_db"]) for row in records if row["truth"])
            == dict.fromkeys(((family, snr) for family in FRESH.FAMILIES for snr in (-5, -8)), 2),
            "Held family or interference balance changed")
    require(not {row["pcm_sha256"] for row in records} & {row["pcm_sha256"] for row in history},
            "Held PCM appeared in a prior source roster")
    folder.mkdir(mode=0o700)
    (folder / "pcm").mkdir()
    for row, raw, _, _ in output:
        (folder / "pcm" / f"{row['id']}.pcm").write_bytes(raw)
    write(folder / "roster.json", records)
    write(folder / "sources.json", {
        "implementation_sha256": sha(Path(__file__)),
        "roster_sha256": sha(folder / "roster.json"),
        "annotations_sha256": sha(earlier.previous.ANNOTATIONS),
        "source_bindings": meta.bindings,
        "layouts": LAYOUTS,
    })
    print(f"Sealed {len(records)} native held inputs with separate source groups and room cuts.", flush=True)


def cases(folder):
    source = read(folder / "sources.json")
    require(source["implementation_sha256"] == sha(Path(__file__)) and source["roster_sha256"] == sha(folder / "roster.json")
            and source["annotations_sha256"] == sha(earlier.previous.ANNOTATIONS), "Held source or code commitment changed")
    for row in read(folder / "roster.json"):
        raw = (folder / "pcm" / f"{row['id']}.pcm").read_bytes()
        require(len(raw) == 2 * SAMPLES and digest(raw) == row["pcm_sha256"], "Held native input changed")
        yield row, raw
