"""Test fixed dereverberation on newly authored matched audio triplets."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import time
import wave
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from scripts import audio_wpe_room_transfer as prior

stress = prior.stress
aachen = stress.aachen
RATE, SAMPLES, WINDOWS = stress.RATE, stress.SAMPLES, stress.WINDOWS
SCHEMA = "chipmunk.audio-wpe-fresh-room/1"
LEVELS = (0, -5)
ANCHORS = ((23200, 38560), (55200, 70560))
RIGS = {"meeting": (1, 4, "IS1004c"), "lecture": (3, 2, "ES2007c")}
AMI_PINS = {"IS1004c": "d9c81453ea85a228bf4b82e9757a19edfc760203d1a3af1c353473b3030613b6",
            "ES2007c": "895904c48583d74c311e42cf0bf49554e998cd5ab87b67268c97de09b856e619"}
PREFIX_SHA = "ef8e66c39f0948cdec5e0b76e0fbd60121e3cb0e89e8ba8be6ad035c4fa27f0b"
METHOD = {
    "foreground": "Select two new held host readers and two different publisher synthesis families per host. Prefer within-source-reader donor families for the first host, and use a separate new donor-only reader pool for the second host. Use native intervals of the already published partial recordings, without generating new speech. Each ordinary control uses a different fragment of its own host. Apply identical 20 ms fades and the same two 960 ms destinations for ordinary and synthetic variants. All dry versions exactly match outside the selected interval.",
    "background": "Use two held AIR measured head-rig impulse-response sets from lecture and meeting rooms and two AMI meetings with no participant from earlier WPE or directional work. Reuse each six-second background, gain and room across versions. Require at least 320 ms of overlapping annotated background speakers in both edit destinations for host groups. Use 0 and −5 dB stereo mean target/background SNR and add background-only controls.",
    "roles": "Choose up to four distinct development-only genuine reader groups for additive WPE operating points while reserving at least two separate genuine assessment readers. Every fit reader contributes matched ordinary and untouched variants. Hold the two matched reader groups and six isolated background excerpts. No scored assessment variant may enter point fitting.",
    "comparison": "Keep native mono at 5.356589748054317. Compare WPE at this same operating point, at the earlier eight-reader point 7.353627562469017, and at cutoffs allowing zero, one or two added genuine development groups. Apply all fixed points to new held scores. Count exact localized edit and ordinary reader groups, with file-condition rescues separate from pooled unique groups. A correct timestamp covers at least half the authored edit at interval IoU of at least .30.",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rank(key):
    return stress.sha(f"{SCHEMA}:{key}".encode())


def read_tar(prefix, labels_path, excluded, source_hashes):
    require(stress.digest(prefix) == PREFIX_SHA and
            stress.digest(labels_path) == aachen.publisher.PINS["part-a.labels.txt"], "Pinned publisher inputs differ")
    labels = {fields[0]: fields[1:] for line in labels_path.read_text().splitlines() if (fields := line.split())}
    by_reader = defaultdict(lambda: {"host": [], "fragment": []})
    with tarfile.open(prefix) as tar:
        try:
            for item in tar:
                match = aachen.PARTIAL.fullmatch(item.name)
                if not match or match[2] in excluded or item.offset_data + item.size > prefix.stat().st_size:
                    continue
                name, speaker = match.groups()
                length, label, *_ = labels[name]
                if label == "bonafide" and float(length) < 6.01:
                    continue
                if label != "bonafide" and not .96 <= float(length) <= 24:
                    continue
                body = tar.extractfile(item).read()
                raw = aachen.publisher.native(body)
                if stress.sha(body) in source_hashes or stress.sha(raw) in source_hashes:
                    continue
                pcm = np.frombuffer(raw, dtype="<i2")
                pin = {"wav": stress.sha(body), "pcm": stress.sha(raw), "id": name}
                if label == "bonafide":
                    by_reader[speaker]["host"].append((pin, pcm))
                else:
                    clock = aachen.publisher.labels_for(labels[name], len(pcm))
                    for a, b, truth in clock:
                        if truth == "spoof" and b-a >= 15360:
                            left = (a+b-15360)//2
                            by_reader[speaker]["fragment"].append((pin, pcm[left:left+15360], left, b-a, name.split("-")[1]))
        except tarfile.ReadError as error:
            require("end of data" in str(error), "Unexpected publisher archive failure")
    return by_reader


def source_groups(args):
    publisher = args.publisher
    _, _, excluded, hashes, _ = aachen.prior_sources(args.directional, publisher / "locked/roster.json")
    before = stress.read(args.aachen / "run/roster.json")
    report = stress.read(aachen.DOCS / "audio-spatial-mask-aachen-transfer-2026-11-13.json")
    require(stress.digest(args.aachen / "run/roster.json") == report["private_roster_sha256"], "Earlier reader exclusions differ")
    for row in before:
        if item := row["source"]:
            excluded.add(item["source"].split("_")[1])
            hashes.update([item["source_sha256"], item["source_pcm_sha256"]])
    extra_excluded = stress.read(args.stress_run / "roster.json")
    require(stress.digest(args.stress_run / "roster.json") ==
            stress.read(aachen.DOCS / "audio-room-stress-transfer-2026-11-13.json")["private_roster_sha256"], "Earlier room stress roster differs")
    # The earlier roster intentionally omits speaker identities. Rebuild its four new point identities from its pinned prefix.
    early_cases, _ = aachen.read_publisher(args.aachen / "llama-first-288m.tar", publisher / "part-a.labels.txt", excluded, hashes)
    previously_used = {row["group"] for row in extra_excluded if row["group"].startswith("new-control")}
    require(len(previously_used) == 4, "Earlier point identities incomplete")
    early_clean = sorted([case for case in early_cases if case["family"] is None], key=lambda row: stress.rank(row["source"]))
    for case in early_clean:
        if case["speaker"] not in excluded and previously_used:
            previously_used.remove(min(previously_used))
            excluded.add(case["speaker"])
            hashes.update([case["source_sha256"], case["source_pcm_sha256"]])
    require(not previously_used, "Unable to reconstruct four former point readers")
    require(stress.digest(publisher / "LICENSE.txt") == aachen.publisher.PINS["LICENSE.txt"], "Publisher terms differ")
    available = read_tar(args.prefix, publisher / "part-a.labels.txt", excluded, hashes)
    possible = [(reader, bank) for reader, bank in available.items() if bank["host"] and len({x[-1] for x in bank["fragment"]}) >= 2]
    require(possible, "No new matched host and donor reader group")
    within = min(possible, key=lambda pair: rank(pair[0]))
    cross_reader, cross = min(((reader, bank) for reader, bank in available.items() if reader != within[0]
                               and not bank["host"] and len({item[-1] for item in bank["fragment"]}) >= 2),
                              key=lambda pair: (-len({item[-1] for item in pair[1]["fragment"]}), rank(pair[0])))
    new_host, new_bank = min(((reader, bank) for reader, bank in available.items() if reader != within[0] and bank["host"]
                              and max(len(pcm) for _, pcm in bank["host"]) >= 7.1 * RATE), key=lambda pair: rank(pair[0]))
    possible = (within, (new_host, {**new_bank, "fragment": cross["fragment"]}))
    result = []
    used, families = {cross_reader}, set()
    for reader, bank in possible:
        host, pcm = min((item for item in bank["host"] if len(item[1]) >= 7.1 * RATE), key=lambda item: rank(item[0]["id"]))
        options = sorted(bank["fragment"], key=lambda item: (item[-1] in families, rank((item[0]["id"], item[2]))))
        first = options[0]
        second = next(item for item in options if item[-1] != first[-1] and item[0]["id"] != first[0]["id"])
        families.update([first[-1], second[-1]])
        result.append((f"paired-{len(result)}", "assessment", reader, host, pcm, (first, second)))
        used.add(reader)
    other = sorted([reader for reader, bank in available.items() if reader not in used and bank["host"]], key=rank)
    require(len(other) >= 5, "Too few new ordinary readers")
    development_n = min(4, len(other)-2)
    for index, reader in enumerate(other):
        role = "development" if index < development_n else "assessment"
        host, pcm = min(available[reader]["host"], key=lambda item: rank(item[0]["id"]))
        result.append((f"genuine-{index}", role, reader, host, pcm, ()))
    return result, len(excluded)


def room_inputs(args):
    path = args.aachen / "air_database_release_1_4.zip"
    pin = stress.read(aachen.DOCS / "audio-spatial-mask-aachen-transfer-2026-11-13.json")["freeze"]["aachen_archive_sha256"]
    require(stress.digest(path) == pin, "AIR archive changed")
    rooms = {}
    with ZipFile(path) as archive:
        require(b"MIT License" in archive.read("AIR_1_4/license.txt"), "AIR grant changed")
        for room, (foreground, background, _) in RIGS.items():
            paths, values = {}, {}
            for role, pos in (("lateral", foreground), ("balanced", background)):
                samples, pins = [], []
                for channel in (1, 0):
                    name = f"AIR_1_4/air_binaural_{room}_{channel}_1_{pos}.mat"
                    raw = archive.read(name)
                    data = loadmat(io.BytesIO(raw), simplify_cells=True)
                    require(data["air_info"]["fs"] == 48000 and data["air_info"]["head"] == 1, "Expected measured 48 kHz head rig")
                    pins.append({"member": name, "sha256": stress.sha(raw)})
                    samples.append(data["h_air"])
                response = resample_poly(np.column_stack(samples), 1, 3, axis=0)
                require(np.isfinite(response).all(), "Room contains invalid response")
                values[role] = response
                paths[role] = {"responses": pins, "peaks": np.argmax(abs(response), axis=0).tolist()}
            rooms[room] = (paths, values)
    return rooms, pin


def ami_inputs(args):
    annotation = stress.annotation_path()
    require(stress.digest(annotation) == aachen.ami.ANNOTATION_SHA256, "AMI release differs")
    rooms = {}
    with ZipFile(annotation) as archive:
        meeting_meta = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {m.get("observation"): {item.get("global_name") for item in m if item.get("global_name")}
                  for m in meeting_meta.iter() if m.get("observation")}
        excluded = set().union(*(people[key] for key in ("ES2004a", "IS1009a", "ES2010d", "IS1007a", "ES2005a", "ES2008a")))
        for room, (_, _, meeting) in RIGS.items():
            require(len(people[meeting]) == 4 and not excluded & people[meeting], "Former room participants overlap")
            excluded.update(people[meeting])
            file = args.ami / f"{meeting}.Array1-01.wav"
            require(stress.digest(file) == AMI_PINS[meeting], "Native background capture differs")
            words = [aachen.ami.track_words(archive.read(f"words/{meeting}.{ch}.words.xml")) for ch in "ABCD"]
            activity = np.zeros((len(words), max(b for channel in words for _, b in channel)), dtype=bool)
            for i, channel in enumerate(words):
                for a, b in channel:
                    activity[i, a:b] = True
            rooms[room] = file, activity.sum(axis=0)
    return rooms


def choose_background(spec, room, group, used, paired):
    path, counts = spec
    available = []
    for start in range(500, len(counts)-600, 25):
        if any(abs(start-s) < 600 for s in used):
            continue
        activity = counts[start:start+600]
        if np.count_nonzero(activity) < 450 or np.count_nonzero(activity >= 2) < 85:
            continue
        intervals = [int(np.count_nonzero(activity[a//160:b//160] >= 2) * 10) for a, b in ANCHORS]
        if not paired or min(intervals) >= 320:
            available.append((rank((room, group, start)), start, intervals))
    require(available, f"Unable to select annotated competing speech in {room} for {group}")
    _, start, overlaps = min(available)
    used.add(start)
    with wave.open(str(path), "rb") as stream:
        require(stream.getparams()[:3] == (1, 2, RATE), "Background native clock changed")
        stream.setpos(start * 160)
        raw = stream.readframes(SAMPLES)
    require(len(raw) == SAMPLES*2, "Background excerpt incomplete")
    return np.frombuffer(raw, dtype="<i2").astype(np.float64), {"start": start*160, "pcm": stress.sha(raw), "overlap_ms": overlaps}


def splice(host, donor, span):
    start, end = span
    source = donor.astype(np.float64)
    target = host[start:end].astype(np.float64)
    gain = math.sqrt((np.square(target).mean()+1) / (np.square(source).mean()+1))
    blend = np.r_[np.linspace(0, 1, 320), np.ones(end-start-640), np.linspace(1, 0, 320)]
    edited = host.copy()
    edited[start:end] = stress.peak_quantize((1-blend)*target + blend*source*gain)
    require(np.array_equal(edited[:start], host[:start]) and np.array_equal(edited[end:], host[end:]), "Splice escapes edit")
    return edited


def variants(host, donors):
    result = [("untouched", [], None, host[:SAMPLES])]
    reserve = host[SAMPLES+320:SAMPLES+320+15360] if len(host) >= SAMPLES+320+15360 else host[SAMPLES-15360:SAMPLES]
    require(len(reserve) == 15360, "Same-reader reserve too short")
    for index, interval in enumerate(ANCHORS[:len(donors) or 1]):
        result.append((f"ordinary-{index}", [], None, splice(host[:SAMPLES], reserve, interval)))
        if donors:
            _, source, _, _, family = donors[index]
            result.append((f"synthetic-{index}", [list(interval)], family, splice(host[:SAMPLES], source, interval)))
    return result


def render(waves, masker, room):
    foreground = [stress.convolve(x, room["lateral"]) for x in waves]
    background = stress.convolve(masker, room["balanced"])
    energy = stress.mean_energy(foreground[0])
    unit = background * math.sqrt(energy / stress.mean_energy(background))
    peak = max(np.max(np.abs(source)+math.sqrt(10**.5)*np.abs(unit)) for source in foreground)
    gain = 22500/peak
    rendered = [stress.peak_quantize(signal*gain) for signal in foreground]
    variants = {}
    for level in LEVELS:
        noise = stress.peak_quantize(unit*10**(-level/20)*gain)
        ratio = 10*math.log10(stress.mean_energy(rendered[0])/stress.mean_energy(noise))
        require(abs(ratio-level) < .02, "Native SNR differs from frozen level")
        values = [signal.astype(np.int32) + noise for signal in rendered]
        require(all(np.max(abs(x)) <= 22501 for x in values), "Native mixture exceeds its peak bound")
        variants[level] = [x.astype("<i2") for x in values]
    return variants


def prepare(args):
    groups, exclusions = source_groups(args)
    rooms, air_hash = room_inputs(args)
    backgrounds = ami_inputs(args)
    rows, signals, sources = [], [], []
    used = defaultdict(set)
    for group, role, reader, host_pin, host, donors in groups:
        versions = variants(host, donors)
        sources.append({"group": group, "role": role, "reader": reader, "host": host_pin,
                        "donors": [{"source": pin, "start": a, "truth_duration": duration, "family": family}
                                   for pin, _, a, duration, family in donors]})
        for room, (_, impulse) in rooms.items():
            mask, pin = choose_background(backgrounds[room], room, group, used[room], bool(donors))
            deliveries = render([entry[-1] for entry in versions], mask, impulse)
            for level in LEVELS:
                for (kind, truth, family, _), waveform in zip(versions, deliveries[level], strict=True):
                    rows.append({"case": len(rows), "group": group, "role": role, "room": room, "level": level,
                                 "kind": kind, "truth": truth, "family": family, "background": pin,
                                 "pcm": stress.sha(waveform.tobytes())})
                    signals.append(waveform)
    for room, (_, impulse) in rooms.items():
        for index in range(3):
            group = f"{room}-background-{index}"
            mask, pin = choose_background(backgrounds[room], room, group, used[room], False)
            waveform = stress.convolve(mask, impulse["balanced"])
            waveform = stress.peak_quantize(22500*waveform/np.max(abs(waveform)))
            rows.append({"case": len(rows), "group": group, "role": "assessment", "room": room, "level": None,
                         "kind": "background_only", "truth": [], "family": None, "background": pin, "pcm": stress.sha(waveform.tobytes())})
            signals.append(waveform)
    previous = stress.read(aachen.DOCS / "audio-wpe-room-transfer-2026-11-13.json")
    lock = {"schema": SCHEMA, "method": METHOD, "windows": WINDOWS, "levels": LEVELS, "edit_intervals": ANCHORS,
            "runner": stress.digest(Path(__file__)), "wpe_runner": stress.digest(Path(prior.__file__)),
            "old_mono": previous["freeze"]["prior_points"]["mono"], "old_wpe": previous["point"]["point"],
            "assets": previous["freeze"]["assets"], "rooms": {k: v[0] for k,v in rooms.items()},
            "air_archive": air_hash, "ami_sha": AMI_PINS, "publisher_prefix": PREFIX_SHA, "excluded_prior_readers": exclusions,
            "point_groups": sum(row[1] == "development" for row in groups),
            "held_paired_groups": sum(bool(row[-1]) for row in groups),
            "held_ordinary_only_groups": sum(row[1] == "assessment" and not row[-1] for row in groups)}
    return lock, rows, signals, sources


def score(args, lock, roster, signals, role):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    gate = SpeechActivity(args.weights / "silero_vad.jit")
    output, cache = [], {}
    t = time.monotonic()
    for row, stereo in zip(roster, signals, strict=True):
        if row["role"] != role:
            continue
        candidate = prior.wpe(stereo)
        alternate = prior.wpe(stereo, fft=np.fft.rfft, inverse=np.fft.irfft)
        require((candidate is None) == (alternate is None), "FFT replay changed WPE availability")
        lsb = 0 if candidate is None else int(np.max(abs(candidate.astype(np.int32)-alternate.astype(np.int32))))
        require(lsb <= 1, "FFT replay diverged by more than one PCM unit")
        scores = {}
        for arm, signal in (("mono", stress.peak_quantize(stereo.astype(np.float64).mean(axis=1))), ("wpe", candidate)):
            scores[arm] = []
            if signal is None:
                continue
            pcm = signal.astype(np.float32) / 32768
            speech = gate.spans(pcm)
            windows = []
            for a,b in WINDOWS:
                value = None
                if window_in_speech(speech, a/RATE, b/RATE):
                    key = stress.sha(signal[a:b].tobytes())
                    if key not in cache:
                        probability, = model.predict([pcm[a:b]])
                        require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid model response")
                        cache[key] = float(logit(probability))
                    value = cache[key]
                windows.append(value)
            scores[arm].append(windows)
        output.append({key: row[key] for key in ("case", "group", "role", "room", "level", "kind", "truth", "family")}
                      | {"responses": scores, "alternate_fft_max_lsb": lsb})
        if len(output) % 12 == 0:
            print(role,len(output),"cases",len(cache),"forwards",round(time.monotonic()-t,2),flush=True)
    print(role,len(output),"cases",len(cache),"forwards",round(time.monotonic()-t,2),flush=True)
    return {"rows": output, "forwards": len(cache)}


def fit(lock, development):
    values = defaultdict(list)
    for row in development["rows"]:
        require(row["role"] == "development" and not row["truth"], "Assessment entered calibration")
        mono, wpe = (stress.chosen(row, arm) for arm in ("mono", "wpe"))
        if wpe is not None and (mono is None or mono[0] <= lock["old_mono"]):
            values[row["group"]].append(wpe[0])
    require(len(values) >= 3, "Too few development ordinary groups eligible for additive points")
    ranks = sorted((max(scores) for scores in values.values()), reverse=True)
    return {"old_wpe": lock["old_wpe"], "old_mono": lock["old_mono"], **{f"budget_{i}": ranks[i] for i in range(3)}}


def summary(rows, lock, points):
    result = {}
    cohorts = {"pooled": rows, **{f"snr_{level}":[r for r in rows if r["level"] == level] for level in LEVELS},
               **{room: [r for r in rows if r["room"] == room] for room in RIGS},
               "background": [r for r in rows if r["kind"] == "background_only"]}
    for cohort, subset in cohorts.items():
        cases = {str(row["case"]): row for row in subset}
        indexed = [{**row, "group": str(row["case"])} for row in subset]
        mono, _ = stress.outcomes(indexed, "mono", lock["old_mono"])
        arms = {}
        for name, point in points.items():
            wpe, locations = stress.outcomes(indexed, "wpe", point)
            additions = wpe["located"] - mono["located"]
            ordinary = {cases[k]["group"] for k in mono["ordinary"] | wpe["ordinary"] if cases[k]["kind"] != "background_only"}
            background = {cases[k]["group"] for k in mono["background"] | wpe["background"]}
            arms[name] = {"useful_deliveries": len(wpe["located"] | mono["located"]),
                          "positive_groups": sorted({cases[k]["group"] for k in wpe["located"] | mono["located"]}),
                          "unique_edits": sorted({(cases[k]["group"], cases[k]["kind"]) for k in wpe["located"] | mono["located"]}),
                          "ordinary_groups": sorted(ordinary), "background_groups": sorted(background),
                          "added_deliveries": [{"group": cases[k]["group"], "level": cases[k]["level"],
                                                "edit": cases[k]["kind"], **locations[k]} for k in sorted(additions,key=int)],
                          "wrong_wpe_deliveries": len(wpe["wrong"]), "unassessed_wpe": len(wpe["unassessed"])}
        result[cohort] = {"positive_deliveries": sum(bool(row["truth"]) for row in subset),
                          "positive_groups": len({row["group"] for row in subset if row["truth"]}),
                          "ordinary_groups": len({row["group"] for row in subset if not row["truth"] and row["kind"] != "background_only"}),
                          "background_groups": sum(row["kind"] == "background_only" for row in subset),
                          "mono_useful_deliveries": len(mono["located"]),
                          "mono_positive_groups": sorted({cases[k]["group"] for k in mono["located"]}),
                          "mono_unique_edits": sorted({(cases[k]["group"], cases[k]["kind"]) for k in mono["located"]}),
                          "mono_ordinary_groups": sorted({cases[k]["group"] for k in mono["ordinary"] - mono["background"]}),
                          "mono_background_groups": sorted({cases[k]["group"] for k in mono["background"]}),
                          "wrong_mono_deliveries": len(mono["wrong"]), "unassessed_mono": len(mono["unassessed"]),
                          "arms": arms}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score"))
    parser.add_argument("run", type=Path)
    for flag in ("prefix","publisher","aachen","directional","stress-run","ami"):
        parser.add_argument(f"--{flag}",type=Path,required=True)
    parser.add_argument("--weights", type=Path, default=aachen.ROOT / "models/weights/audio-upgrade")
    args=parser.parse_args()
    lock, roster, signals, sources=prepare(args)
    lock_path, roster_path, source_path = (args.run / name for name in ("freeze.json","roster.json","sources.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        stress.write(lock_path,lock)
        stress.write(roster_path,roster)
        stress.write(source_path,sources)
        print("Frozen",len(roster),"deliveries",lock["held_paired_groups"],"matched groups",lock["point_groups"],
              "point groups",lock["held_ordinary_only_groups"],"independent held controls",stress.digest(lock_path),flush=True)
        return
    stress.compare(lock_path,lock)
    stress.compare(roster_path,roster)
    stress.compare(source_path,sources)
    d_path,p_path=(args.run / name for name in ("development.json","points.json"))
    if args.phase == "fit":
        dev=score(args,lock,roster,signals,"development")
        points=fit(lock,dev)
        stress.write(d_path,dev)
        stress.write(p_path,{"points":points,"development":stress.digest(d_path),"freeze":stress.digest(lock_path)})
        print("Points frozen",points,flush=True)
        return
    dev,point=stress.read(d_path),stress.read(p_path)
    require(point == {"points":fit(lock,dev),"development":stress.digest(d_path),"freeze":stress.digest(lock_path)},"Development points changed")
    held=score(args,lock,roster,signals,"assessment")
    stats=summary(held["rows"],lock,point["points"])
    stress.write(args.run / "public.json",{"schema":SCHEMA,"freeze":lock,"freeze_sha256":stress.digest(lock_path),
                 "roster_sha256":stress.digest(roster_path),"source_roster_sha256":stress.digest(source_path),
                 "point":point,"development":dev,"assessment":held,"summary":stats})
    print(json.dumps({name:{arm:len(value["added_deliveries"]) for arm,value in info["arms"].items()} for name,info in stats.items()}),flush=True)


if __name__ == "__main__":
    main()
