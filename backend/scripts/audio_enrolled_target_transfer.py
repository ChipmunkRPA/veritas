"""Qualify a clean-enrollment target mask under overlapping room speech."""

from __future__ import annotations

import argparse
import time
import wave
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from statistics import median
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.signal import istft, stft

from app.detectors.audio import decode
from scripts import audio_partial_residual_inputs as inputs
from scripts import audio_short_edit_cases_v172 as donors
from scripts import audio_spectra_long_room_holdout as long_room
from scripts import prepare_natural_overlap_transfer as ami

ROOT, BASE = long_room.ROOT, long_room.BASE
PARENT = BASE / "audio-spectra-long-room-final-2026-11-12"
STEREO = BASE / "audio-paired-stereo-2026-11-11"
ANNOTATIONS = long_room.ANNOTATION
ROOMS = {"selection": ("ES2004a", "IS1009a"), "confirmation": ("ES2010d", "IS1007a")}
RATE, FFT, HOP = 16000, 512, 128
TARGET_ATOMS, GENERIC_ATOMS, UPDATES = 12, 6, 28
SCHEMA = "chipmunk.audio-enrolled-target-transfer/1"
METHOD = {
    "source": "Reuse verified episode identities and edits from the earlier long-clip trial. Rerender every submitted mono mixture from the exact host, donor and new native AMI Array1-01 interval. Each 0.8-second edit intersects at least 400 ms of concurrent background words from two or more annotated voices. Global SNR is -5 dB. No new external models or sources.",
    "roles": "Same disjoint LibriSpeech hosts and distinct generator donor groups. Use two held meeting sessions separate from both the new point rooms and prior long-clip held rooms. References are four-second different utterances. True enrollment is the same canonical LibriSpeech speaker. Sham is a distinct speaker in the same role and not the ordinary different-voice donor. A common generic dictionary comes from four separate genuine confidence_cal speakers.",
    "scope": "The generated donor voices do not share declared host identity. A correct enrollment can suppress the generated substitution or an ordinary other speaker. Test this preservation directly and leave original samples available. Isolated AMI controls have no claimed target enrollment and are skipped by this check.",
    "features": "512-point periodic Hann power spectrum, 128-sample hops. Fit deterministic prototypes from the highest-energy 60% of each clean reference. Normalize weighted frame powers to unit norm, choose farthest cosine templates and make four cosine assignment/mean updates. Twelve templates for each target or sham, six each from four generic references. Equalize frequency by fourth root inverse generic median power with 1e-7 floor and 90th percentile weight cap.",
    "mask": "Fit nonnegative activations for submitted power by 28 squared-error multiplicative updates with templates fixed. Split target and generic fitted powers. Apply square root of target fraction on the original mixture phase with .20 amplitude floor. Fit correct and sham separately with the identical generic templates. Use no clean target or nuisance component in mask fitting.",
    "qualification": "Separate exact-component acoustic audit for whole, known edit and surrounding context. For ordinary same-host controls require median correct enrollment gain at least .5 dB in each held room, and correct gain above sham in at least two-thirds overall. The same test requires median synthetic edit retention no worse than -3 dB under correct enrollment in each room. Failure limits the result to source-selectivity measurement.",
}
read, write, sha, digest, require = long_room.read, long_room.write, long_room.sha, long_room.digest, long_room.require


def salt(value):
    return digest(f"{SCHEMA}:{value}".encode())


def spectrum(values):
    return stft(values, fs=RATE, nperseg=FFT, noverlap=FFT-HOP)[2]


def selected_frames(raw):
    power = abs(spectrum(raw))**2
    energy = power.sum(axis=0)
    chosen = energy > max(np.percentile(energy, 40), 1e-8)
    require(chosen.sum() >= 75, "Reference has too little usable energy")
    return power[:, chosen]


def normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)


def dictionary(power, weights, size):
    frames = normalize(power * weights[:, None])
    center = normalize(frames.mean(axis=1, keepdims=True))[:, 0]
    similarity = center @ frames
    ids = [int(similarity.argmax())]
    highest = frames[:, ids[0]] @ frames
    for _ in range(size-1):
        index = int(highest.argmin())
        ids.append(index)
        highest = np.maximum(highest, frames[:, index] @ frames)
    model = frames[:, ids].copy()
    for _ in range(4):
        labels = (model.T @ frames).argmax(axis=0)
        for k in range(size):
            if np.any(labels == k):
                model[:, k] = normalize(frames[:, labels == k].mean(axis=1, keepdims=True))[:, 0]
    return np.maximum(model, 1e-9)


def target_mask(mixed, target, generic, weights):
    power = abs(spectrum(mixed))**2 * weights[:, None]
    atoms = np.column_stack((target, generic))
    fitting = atoms.T @ power
    kernel = atoms.T @ atoms
    coefficients = np.broadcast_to(np.maximum(power.sum(axis=0), 1e-12)/atoms.shape[1], (atoms.shape[1], power.shape[1])).copy()
    for _ in range(UPDATES):
        coefficients *= fitting / np.maximum(kernel @ coefficients, 1e-12)
    keep = atoms[:, :TARGET_ATOMS] @ coefficients[:TARGET_ATOMS]
    all_power = atoms @ coefficients
    return np.sqrt(np.clip(keep/np.maximum(all_power, 1e-12), .04, 1.))


def apply(values, mask):
    decoded = istft(spectrum(values)*mask, fs=RATE, nperseg=FFT, noverlap=FFT-HOP)[1][:len(values)]
    require(decoded.shape == values.shape and np.isfinite(decoded).all(), "Masked waveform clock changed")
    return decoded


def db(a, b):
    return float(20*np.log10(max(1e-12, np.linalg.norm(a))/max(1e-12, np.linalg.norm(b))))


def word_room_sources():
    current = long_room.room_inputs()
    pins = read(STEREO / "run/freeze.json")["rooms"]
    consumed = read(STEREO / "run/roster.json")
    prior = read(PARENT / "roster.json")
    blocked = defaultdict(list)
    for row in prior:
        r = row["background"]
        blocked[r["room"]].append((r["capture_start_sample"], r["capture_start_sample"]+r["samples"]))
    for row in consumed:
        if row["treatment"] == "native":
            start = row.get("native_source_start")
            if start is not None:
                blocked[row["room"]].append((start, start+3*RATE))
    output = {}
    with ZipFile(ANNOTATIONS) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = set()
        for role, rooms in ROOMS.items():
            for room in rooms:
                record = next(x for x in meetings.iter() if x.get("observation") == room)
                members = {x.get("global_name") for x in record if x.get("global_name")}
                require(len(members) == 4 and not members & people, "Meeting participant ancestry crosses roles")
                people.update(members)
                if room in current:
                    path, pin, timeline, _, past = current[room]
                    blocked[room].extend(past)
                else:
                    name = f"{room}.Array1-01.wav"
                    path, pin = STEREO / "ami" / name, pins[room][name]["sha256"]
                    require(sha(path) == pin, "Raw AMI microphone bytes changed")
                    track = [ami.track_words(archive.read(f"words/{room}.{speaker}.words.xml")) for speaker in "ABCD"]
                    end = max(b for ranges in track for _, b in ranges)
                    by_speaker = np.zeros((4, end), dtype=bool)
                    for i, person in enumerate(track):
                        for a, b in person:
                            by_speaker[i, a:b] = True
                    timeline = by_speaker.sum(axis=0)
                output[room] = path, pin, timeline, role
    return output, blocked


def room_cut(spec, used, room, samples, edit):
    path, pin, timeline, role = spec
    ticks = samples//160
    first, last = (x//160 for x in edit) if edit else (0, ticks)
    selected = []
    with wave.open(str(path), "rb") as stream:
        channels, size, rate = stream.getparams()[:3]
        require((size, rate) == (2, RATE) and channels in {1, 2}, "Unexpected native AMI source format")
        end = min(len(timeline), stream.getnframes()//160)
        for tick in range(800, end-ticks, 800):
            a = tick*160
            if any(a < y and x < a+samples for x, y in used[room]):
                continue
            whole = timeline[tick:tick+ticks]
            overall = np.count_nonzero(whole >= 2)
            local = np.count_nonzero(whole[first:last] >= 2)
            if np.count_nonzero(whole) > .8*ticks and overall >= 120 and local >= 40:
                selected.append((salt((room, tick)), a, int(overall)*10, int(local)*10))
        require(bool(selected), "No verified room interval with local overlapping words")
        _, start, overlap, local = min(selected)
        stream.setpos(start)
        samples_raw = np.frombuffer(stream.readframes(samples), "<i2").reshape(-1, channels)
    require(len(samples_raw) == samples and (channels == 1 or np.array_equal(samples_raw[:, 0], samples_raw[:, 1])), "Native microphone stored differently")
    used[room].append((start, start+samples))
    data = samples_raw[:, 0].astype(np.float64)
    return data, {"meeting": room, "role": role, "capture_sha256": pin, "capture_start_sample": start,
                  "source_pcm_sha256": digest(samples_raw[:, 0].tobytes()), "overlap_ms": overlap, "edit_overlap_ms": local if edit else None}


def render():
    lock = read(PARENT / "freeze.json")
    require(sha(PARENT / "roster.json") == lock["roster_sha256"], "Earlier source episode roster differs")
    source, _, _, _ = inputs.catalogues()
    by_role = defaultdict(list)
    for item in source.values():
        if item["kind"] == "genuine":
            require(item["license"] == "CC-BY-4.0", "Genuine source rights changed")
            by_role[item["role"]].append(item)
    donor_file = donors.sources.MEDIA / "holdout/private-manifest.json"
    require(sha(donor_file) == lock["donor_manifest_sha256"], "Earlier donor source receipt differs")
    ai = {row["original_sha256"]: row for row in read(donor_file)["foregrounds"]}
    rooms, used = word_room_sources()

    @cache
    def genuine(pin):
        row = source[pin]
        pcm = decode.decode_audio(Path(row["source_path"]), source_sha256=pin)
        require(pcm.mapping["pcm_sha256"] == row["pcm_sha256"], "Qualified source decoded differently")
        return pcm.pcm.astype(np.float64)

    @cache
    def reference(pin):
        raw = genuine(pin)[:4*RATE]
        require(len(raw) == 4*RATE, "Short enrollment")
        selected_frames(raw)
        return raw

    def enrollment(options):
        for entry in sorted(options, key=lambda row: salt(row["raw_sha256"])):
            key = entry["raw_sha256"]
            try:
                return key, reference(key)
            except ValueError as error:
                if str(error) not in {"Short enrollment", "Reference has too little usable energy"}:
                    raise
        raise ValueError("No eligible enrollment utterance")

    generic = {}
    for group in sorted({r["study_group"] for r in by_role["confidence_cal"]}, key=salt)[:4]:
        key, ref = enrollment([r for r in by_role["confidence_cal"] if r["study_group"] == group])
        generic[key] = ref
    require(len(generic) == 4, "Incomplete generic reference bank")
    rows = defaultdict(list)
    for row in read(PARENT / "roster.json"):
        rows[row["role"], row["group"]].append(row)
    output, reference_pcm = [], generic.copy()
    for (role, group), examples in rows.items():
        job = examples[0]
        extent = job["background"]["samples"]
        membership = ROOMS[role]
        if job["arm"] == "background_only":
            target, sham = None, None
            room = membership[0 if job["background"]["room"].startswith("ES") else 1]
            recorded, info = room_cut(rooms[room], used, room, extent, None)
            cases = [(job, None, recorded)]
        else:
            host = source[job["host_sha256"]]
            identity, old_role = host["canonical_speaker"], host["role"]
            pool = by_role[old_role]
            target, positive_ref = enrollment([r for r in pool if r["canonical_speaker"] == identity
                                              and r["raw_sha256"] != job["host_sha256"]])
            ordinary_identity = source[job["ordinary_sha256"]]["canonical_speaker"]
            sham, false_ref = enrollment([r for r in pool if r["canonical_speaker"] not in {identity, ordinary_identity}])
            reference_pcm.update({target: positive_ref, sham: false_ref})
            require(source[target]["text_identity"] != host["text_identity"], "Enrollment repeats the test utterance")
            start, stop = job["edit_range"]
            base = genuine(job["host_sha256"])
            foreground = base[job["source_start"]:job["source_start"]+extent]
            ordinary = genuine(job["ordinary_sha256"])
            room = membership[int(group.rsplit("-", 1)[-1]) % 2]
            recorded, info = room_cut(rooms[room], used, room, extent, (start, stop))
            recorded *= np.linalg.norm(foreground*32768) * 10**.25 / np.linalg.norm(recorded)
            cases = []
            for example in examples:
                arm = example["arm"]
                if arm == "synthetic":
                    inserted = np.frombuffer(donors.exact_source(ai[example["donor_sha256"]], "holdout"), "<i2").astype(float)/32768
                    offset = example["donor_start"]
                elif arm == "ordinary_same_host":
                    inserted, offset = base, example["same_host_start"]
                else:
                    inserted, offset = ordinary, example["other_voice_start"]
                component = inputs.splice(foreground, inserted, start, offset, stop-start)*32768
                cases.append((example, component, recorded))
        gain = min(1., 31_129/max(float(np.max(abs(noise)+(0. if clean is None else abs(clean)))) for _, clean, noise in cases))
        for example, clean, noise in cases:
            first = np.zeros(extent) if clean is None else np.rint(gain*clean)
            second = np.rint(gain*noise)
            raw = long_room.pcm(first+second)
            first, second = long_room.pcm(first), long_room.pcm(second)
            output.append(({"id": example["id"], "group": group, "role": role, "family": example["family"],
                            "arm": example["arm"], "truth": example["truth"], "edit": example.get("edit_range"),
                            "target_reference": target, "sham_reference": sham, "background": info,
                            "host_sha256": example.get("host_sha256"), "ordinary_sha256": example.get("ordinary_sha256"),
                            "generated_sha256": example.get("donor_sha256"), "original_pcm_sha256": digest(raw),
                            "foreground_pcm_sha256": digest(first), "background_pcm_sha256": digest(second)}, raw, first, second))
    require(len(output) == 43 and sum(row["arm"] == "synthetic" for row, *_ in output) == 9, "Held input denominator differs")
    for role in ROOMS:
        expected = {r[key] for r, *_ in output if r["role"] == role for key in ("target_reference", "sham_reference") if r[key]}
        require(not expected & generic.keys(), "Generic reference ancestry crossed roles")
    return output, reference_pcm, list(generic)


def materialize(args):
    views, references, generic = render()
    args.run.mkdir(mode=0o700)
    reference_root = args.run / "references"
    reference_root.mkdir()
    pcm_root = args.run / "pcm"
    pcm_root.mkdir()
    for key, raw in references.items():
        (reference_root / f"{key}.pcm").write_bytes(reference_pcm(raw))
    for row, original, foreground, background in views:
        for key, raw in (("original", original), ("foreground", foreground), ("background", background)):
            (pcm_root / f"{row['id']}-{key}.pcm").write_bytes(raw)
    write(args.run / "roster.json", [row for row, *_ in views])
    write(args.run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)),
          "parent_roster_sha256": sha(PARENT / "roster.json"), "annotations_sha256": sha(ANNOTATIONS),
          "roster_sha256": sha(args.run / "roster.json"), "generic": generic,
          "references": {key: sha(reference_root / f"{key}.pcm") for key in references}})
    print("Frozen 43 exact new mixtures and separate target/sham reference inputs", flush=True)


def waveform(raw):
    return np.frombuffer(raw, "<i2").astype(np.float64)/32768


def reference_pcm(raw):
    values = np.rint(raw*32768)
    require(np.isfinite(values).all() and -32768 <= values.min() <= values.max() <= 32767, "Invalid signed enrollment samples")
    return values.astype("<i2").tobytes()


def checked_pcm(args, row, key):
    raw = (args.run / "pcm" / f"{row['id']}-{key}.pcm").read_bytes()
    require(digest(raw) == row[f"{key}_pcm_sha256"], "Frozen exact waveform changed")
    return waveform(raw)


def measure(args, lock, rows):
    @cache
    def reference(key):
        raw = (args.run / "references" / f"{key}.pcm").read_bytes()
        require(digest(raw) == lock["references"][key], "Exact reference changed")
        return selected_frames(waveform(raw))

    fixed = [reference(key) for key in lock["generic"]]
    weights = np.maximum(np.median(np.concatenate(fixed, axis=1), axis=1), 1e-7)**-.25
    weights = np.minimum(weights, np.percentile(weights, 90))
    generic = np.column_stack([dictionary(power, weights, GENERIC_ATOMS) for power in fixed])

    @cache
    def target(key):
        return dictionary(reference(key), weights, TARGET_ATOMS)

    output = []
    for row in rows:
        if row["target_reference"] is None:
            output.append({"id": row["id"], "status": "no_target_enrollment"})
            continue
        original, foreground, background = (checked_pcm(args, row, key) for key in ("original", "foreground", "background"))
        require(np.array_equal(original, foreground+background), "Frozen mixture components differ")
        a, b = row["edit"]
        support = {"whole": slice(None), "edit": slice(a,b), "outside": np.r_[0:a, b:len(original)]}
        arms = {}
        for key, ref in (("correct", row["target_reference"]), ("sham", row["sham_reference"])):
            mask = target_mask(original, target(ref), generic, weights)
            recovered, interference, rendered = (apply(values, mask) for values in (foreground, background, original))
            require(np.max(abs(recovered+interference-rendered)) <= 1e-10, "Exact component replay does not reconstruct output")
            arms[key] = {name: {"snr_gain_db": db(recovered[span], interference[span])-db(foreground[span], background[span]),
                                "retained_target_db": db(recovered[span], foreground[span])} for name, span in support.items()}
        output.append({"id": row["id"], "status": "measured", "arms": arms})
    return output


def summarize(rows, observed):
    buckets = defaultdict(list)
    for row, proof in zip(rows, observed, strict=True):
        if proof["status"] == "measured":
            buckets[row["role"], row["background"]["meeting"], row["arm"]].append(proof["arms"])
    cells = []
    for (role, room, arm), details in sorted(buckets.items()):
        outcomes = {side: {part: {key: median(item[side][part][key] for item in details)
                                 for key in ("snr_gain_db", "retained_target_db")} for part in ("whole", "edit", "outside")}
                    for side in ("correct", "sham")}
        cells.append({"role": role, "meeting": room, "arm": arm, "views": len(details), "median": outcomes})
    same = [item for row, item in zip(rows, observed, strict=True) if row["role"] == "confirmation" and row["arm"] == "ordinary_same_host"]
    benefit = sum(item["arms"]["correct"]["edit"]["snr_gain_db"] > item["arms"]["sham"]["edit"]["snr_gain_db"] for item in same)
    held_rooms = [row for row in cells if row["role"] == "confirmation"]
    gate = {"right_reference_beats_sham": benefit, "same_host_cases": len(same), "two_thirds_specific": benefit*3 >= 2*len(same),
            "each_room_genuine_median_snr_gain": all(row["median"]["correct"]["edit"]["snr_gain_db"] >= .5
                                                  for row in held_rooms if row["arm"] == "ordinary_same_host"),
            "each_room_edit_preserved": all(row["median"]["correct"]["edit"]["retained_target_db"] >= -3
                                           for row in held_rooms if row["arm"] == "synthetic")}
    gate["qualifies"] = gate["two_thirds_specific"] and gate["each_room_genuine_median_snr_gain"] and gate["each_room_edit_preserved"]
    return {"outcomes": cells, "acoustic_gate": gate, "scope": dict(Counter(item["status"] for item in observed)),
            "neural_authenticity_scores_measured": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "audit"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        materialize(args)
        return
    lock = read(args.run / "freeze.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(args.run / "roster.json"), "Source or method freeze changed")
    rows = read(args.run / "roster.json")
    started = time.monotonic()
    observed = measure(args, lock, rows)
    result = summarize(rows, observed)
    if args.phase == "measure":
        write(args.run / "measurements.json", observed)
        write(args.run / "result.json", {"schema": SCHEMA, "method": METHOD, **result,
              "commitments": {key: sha(args.run / f"{key}.json") for key in ("freeze", "roster", "measurements")},
              "seconds": round(time.monotonic()-started, 3)})
        print(result["acoustic_gate"], flush=True)
    else:
        expected = read(args.run / "result.json")
        require(read(args.run / "measurements.json") == observed and all(expected[key] == value for key, value in result.items()), "Acoustic replay differs")
        views, references, _ = render()
        require([row for row, *_ in views] == rows and all(digest(reference_pcm(raw)) == lock["references"][key]
                for key, raw in references.items()), "Original source or enrollment does not replay")
        print(f"Reproduced {len(rows)} native source inputs, {len(references)} enrollment inputs and measured acoustic cells.", flush=True)


if __name__ == "__main__":
    main()
