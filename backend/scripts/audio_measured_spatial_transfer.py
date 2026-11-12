"""Freeze and evaluate short voice edits after calibrated two-microphone cancellation."""

from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import fftconvolve, resample_poly

from scripts import audio_bbc_room_assets_v303 as bbc
from scripts import audio_remote_neighbor_replication as fresh
from scripts import audio_room_assets_v299 as surrey

BASE, RATE, SOURCE = fresh.BASE, fresh.RATE, fresh.source
OLD, QUALIFIED = fresh.OLD, fresh.qualified
read, write, sha, digest, require = fresh.read, fresh.write, fresh.sha, fresh.digest, fresh.require
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
PREVIOUS = BASE / "audio-remote-neighbor-replication-2026-11-12"
SCHEMA = "chipmunk.audio-measured-spatial-transfer/1"
STUDY = {"selection": ("surrey", ("ES2004a", "IS1009a")), "held": ("bbc", ("ES2010d", "IS1007a"))}
DURATION = 6 * RATE
FFT_GUARD = 2 * RATE
WINDOWS = [(RATE, 3 * RATE), (3 * RATE, 5 * RATE)]
STEMS = ("stereo", "foreground", "background", "original", "spatial")
ARMS = ("original", "spatial")
METHOD = {
    "scope": "Engineered stereo with measured stationary room response and authenticated front and one lateral source paths. Both real AMI background talkers are in one off-axis playback feed. A third physical path is not inferred. Ordinary mono, unknown stereo geometry, moving source or recorder, and differently placed competing voices are out of scope.",
    "sources": "Use fresh exact matched host, same-reader, other-reader and three synthesis families, disjoint speaker, transcript, utterance and background intervals across roles. Each role has two clone, three expressive and one stock inserts. Exclude exact sources and identities from the earlier replication, retain broader historical exclusions, and acknowledge exposed source banks. Surrey plus two distinct AMI sessions determine points. BBC plus two different AMI sessions are held. Include two room-only negative views per role.",
    "rendering": "Six-second 16 kHz sample clock. Resample the original 48 kHz measured responses to 16 kHz with fixed polyphase coefficients, then use exact causal convolution from sample zero. Match all stems and scale background to -5 dB mono SNR relative to the untouched foreground. Inserts are 0.80 or 0.96 s, have 20 ms fades, and contain at least half their authored time under separately annotated competing words.",
    "candidate": "Generalized sidelobe cancellation on input with at least two seconds of zero padding on either side and measured transfer coefficients. Normalize the target-blocking direction by front-channel energy. Fit no mixture coefficients. In the blocked channel apply the measured lateral path with ridge .08 relative to its mean channel power. Cap subtraction coefficient magnitude at 1.5. Raised cosine crossfades 100–200 and 6000–6800 Hz. Keep mean ear response outside those bands. Return the original sample count without translation, gain normalization, sample deletion, or speaker-window re-alignment.",
    "acoustic_preflight": "Each room needs at least +4 dB median reconstruction SNR improvement across edited cases and +3 dB inside exact authored ranges, at least 35 dB median isolated target preservation, and zero clipping. Report opposite lateral path and opposite rig as mismatched diagnostics. Known stems are used only to verify ground truth, never to construct the candidate PCM.",
    "scoring": "Use the identical pinned native voice classifier and speech gate on two exact 2-second slices starting at 1 and 3 seconds. Each arm must pass its own speech gate. Compare highest admitted response, plus same-time scoring limited to the intersection of both admission decisions. Keep earlier original research point and compare both arms at separate maximum-genuine selection points and descriptive equal-false-group budgets 0, 1, and 2. No threshold or window fitting on held outcomes.",
    "localization": "Useful requires at least .30 temporal IoU and .50 coverage of the authored range. Retain source clock and measured direct-path peak plus observed change energy spread. Report unassessed cases, matched-pair rank, and original-window geometry separately from production up-to-four-second windows.",
}


def rank(value):
    return digest(f"{SCHEMA}:{value}".encode())


def load_rigs(root):
    surrey_path = root / "audio-room-assets-v299/development"
    bbc_path = root / "audio-bbc-room-assets-v303"
    raw, surrey_asset = surrey.pinned_bytes(surrey_path, surrey.SURREY_ASSET)
    impulses = surrey.surrey_impulses(raw)
    measured = {
        "surrey": {
            key: impulses[label]
            for key, label in zip(("front", "left", "right"), ("FC", "SiL", "SiR"), strict=True)
        },
        "bbc": {},
    }
    receipts = {
        "surrey": {"asset": surrey_asset, "license": "CC-BY-4.0", "positions": {}},
        "bbc": {"license": "CC-BY-SA-4.0", "positions": {}},
    }
    for position, asset in bbc.ASSETS.items():
        raw, pin = surrey.pinned_bytes(bbc_path, asset)
        measured["bbc"][position], _ = bbc.parse_impulse(raw, position)
        receipts["bbc"]["positions"][position] = {"asset": pin}
    rigs = {}
    for room, positions in measured.items():
        rigs[room] = {}
        for position, response in positions.items():
            coefficients = np.asarray(response.coefficients, dtype="<f8")
            require(
                coefficients.ndim == 2 and coefficients.shape[1] == 2 and np.isfinite(coefficients).all(),
                "Room calibration must be synchronized stereo",
            )
            rendered = np.ascontiguousarray(resample_poly(coefficients, 1, 3, axis=0), dtype="<f8")
            require(
                len(rendered) <= FFT_GUARD and np.isfinite(rendered).all(),
                "Room impulse exceeds frozen guard",
            )
            rigs[room][position] = rendered
            receipts[room]["positions"].setdefault(position, {}).update(
                {
                    "native": response.identity(),
                    "rate_hz": RATE,
                    "samples": len(rendered),
                    "render_sha256": digest(rendered.tobytes()),
                    "mean_peak_sample": int(np.argmax(abs(rendered.mean(axis=1)))),
                }
            )
    return rigs, receipts


def controls(spec, key, width):
    return [
        a
        for a in SOURCE.options(spec, width, key + "-target")
        if any(s + 1600 <= a and a + width <= e - 1600 for s, e in WINDOWS)
    ]


def prepare_sources():
    meta = QUALIFIED.sources.Metadata()
    ids, _, seen, _ = QUALIFIED.history(meta)
    used, rooms = fresh.histories(meta, seen, ids)
    previous = meta.read("previous_frozen_replication", PREVIOUS / "result.json")
    require(
        previous["commitments"]["roster"] == sha(PREVIOUS / "roster.json"), "Previous exposure ledger differs"
    )
    for row in meta.read("prior_exact_replication", PREVIOUS / "roster.json"):
        for role in ("host", "donor"):
            if witness := row.get(role):
                used.add(witness["canonical_speaker"])
                for name in ("source_sha256", "decoder_input_sha256", "pcm_sha256"):
                    if name in witness:
                        seen.add(f"{'pcm' if name == 'pcm_sha256' else 'bytes'}:{witness[name]}")
        room = row["background"]
        start = room["capture_start_sample"]
        rooms[room["room"]][3].append((start, start + DURATION))
    for room, spec in rooms.items():
        unique = sorted(set(spec[3]))
        spec[3][:] = unique
    genuine = fresh.genuine_rows(meta, ids, seen, used)
    donors = {
        "clone": fresh.clone_rows(meta, ids, seen),
        "expressive": SOURCE.donors(meta, ids, seen),
        "stock": fresh.cue.recent.stock_rows(meta, seen, ids),
    }
    out, texts, contexts, cuts = [], set(), set(), set()
    with ExitStack() as stack, TemporaryDirectory() as directory:
        reader = QUALIFIED.source_reader.SourceReader(stack, meta.bindings)
        path = Path(directory) / "input"
        for panel, (rig, sessions) in STUDY.items():
            for index, room in enumerate(sessions):
                # Keep both roles balanced when the retained stock bank cannot fill four slots.
                families = ("clone", "expressive", "expressive") if index else fresh.FAMILIES
                for j, family in enumerate(families):
                    # The retained stock bank has fewer unused long spoken fragments.
                    width = 12_800 if family == "stock" else (12_800, 15_360)[(index + j) % 2]
                    key = f"spatial-{panel}-{index}-{j}-interior"
                    base, reserve, same_at, host = fresh.genuine(
                        genuine, reader, path, used, seen, texts, width, key, host=True
                    )
                    _, other, other_at, ordinary = fresh.genuine(
                        genuine, reader, path, used, seen, texts, width, key + "-other"
                    )
                    generated, at, synthetic = fresh.generated(
                        donors[family], reader, path, used, seen, texts, contexts, width, key
                    )
                    a, noise, background = fresh.capture(rooms[room], room, controls(base, key, width), width)
                    position = ("left", "right")[(index + j) % 2]
                    group = {
                        "panel": panel,
                        "group": key,
                        "rig": rig,
                        "position": position,
                        "background": background,
                        "family": family,
                        "host": host,
                        "authored_range": [a, a + width],
                    }
                    views = [("untouched", base, None)]
                    for arm, values, offset, witness in (
                        ("ordinary_same_host", reserve, same_at, host),
                        ("ordinary_other_voice", other, other_at, ordinary),
                        ("synthetic", generated, at, synthetic),
                    ):
                        pin = QUALIFIED.pcm_digest(values[offset : offset + width].astype(np.float32))
                        require(
                            pin not in cuts and f"pcm:{pin}" not in seen,
                            "Donor fragment has previous exact exposure",
                        )
                        cuts.add(pin)
                        edit = OLD.old.inputs.splice(base, values, a, offset, width)
                        views.append((arm, edit, witness | {"pcm_sha256": pin}))
                    out.append((group, noise, views))
                _, noise, background = fresh.capture(rooms[room], room, [32000], 12800)
                out.append(
                    (
                        {
                            "panel": panel,
                            "group": f"room-{panel}-{index}",
                            "rig": rig,
                            "position": ("left", "right")[index],
                            "background": background,
                            "family": None,
                            "host": None,
                            "authored_range": None,
                        },
                        noise,
                        [("background_only", np.zeros(DURATION), None)],
                    )
                )
    require(len(out) == 16 and len(cuts) == 36, "Incomplete role or fresh donor coverage")
    return out, meta.bindings


def power(values):
    return float(np.square(values, dtype=np.float64).sum())


def power_ratio(signal, error):
    return float(10 * np.log10(max(power(signal), 1.0e-16) / max(power(error), 1.0e-16)))


def to_bytes(values):
    return np.ascontiguousarray(values, dtype="<i2").tobytes()


def mono(values):
    return np.rint(values.mean(axis=1))


def convolve(values, impulse):
    return fftconvolve(values[:, None], impulse, axes=0)[:DURATION]


def tapered_band(freq):
    rise = (1 - np.cos(np.pi * np.clip((freq - 100) / 100, 0.0, 1.0))) / 2
    fall = (1 + np.cos(np.pi * np.clip((freq - 6000) / 800, 0.0, 1.0))) / 2
    return rise * fall


def blocking_filter(positions, direction):
    n = next_fast_len(DURATION + 2 * FFT_GUARD)
    front, lateral = (rfft(positions[site], n=n, axis=0) for site in ("front", direction))
    norm = np.sqrt(np.maximum(np.sum(abs(front) ** 2, axis=1), 1.0e-20))
    blocked = np.stack((front[:, 1], -front[:, 0]), axis=1) / norm[:, None]
    noise = (blocked * lateral).sum(axis=1)
    coefficient = (
        lateral.mean(axis=1)
        * np.conj(noise)
        / (abs(noise) ** 2 + 0.08 * np.mean(abs(lateral) ** 2, axis=1) + 1.0e-16)
    )
    coefficient *= np.minimum(1.0, 1.5 / np.maximum(abs(coefficient), 1.0e-16))
    return n, 0.5 - (coefficient * tapered_band(rfftfreq(n, 1 / RATE)))[:, None] * blocked


def apply_filter(values, filt):
    n, weights = filt
    # Use a zero guard on both sides and return the original native clock.
    padded = np.pad(values, ((FFT_GUARD, n - DURATION - FFT_GUARD), (0, 0)))
    return irfft((rfft(padded, axis=0) * weights).sum(axis=1), n=n)[FFT_GUARD : FFT_GUARD + DURATION]


def fingerprint_range(front, reference):
    change = np.square(front.mean(axis=1) - reference.mean(axis=1))
    cumulative = np.cumsum(change)
    if not cumulative[-1]:
        return None
    return [
        int(np.searchsorted(cumulative, portion * cumulative[-1])) + (portion > 0.5)
        for portion in (0.05, 0.95)
    ]


def build(root):
    rigs, room_receipts = load_rigs(root)
    groups, ancestry = prepare_sources()
    filters = {
        room: {side: blocking_filter(positions, side) for side in ("left", "right")}
        for room, positions in rigs.items()
    }
    rows = []
    for group, noise, views in groups:
        room, side = group["rig"], group["position"]
        other_room = "surrey" if room == "bbc" else "bbc"
        other_side = "right" if side == "left" else "left"
        impulse = rigs[room]["front"]
        foreground = [convolve(values, impulse) for _, values, _ in views]
        lateral = convolve(noise, rigs[room][side])
        require(power(lateral.mean(axis=1)) > 0, "Invalid room stem")
        if group["authored_range"]:
            lateral *= math.sqrt(power(foreground[0].mean(axis=1)) * 10**0.5 / power(lateral.mean(axis=1)))
        gain = 22500 / max(float(np.max(abs(stem) + abs(lateral))) for stem in foreground)
        noise_int = np.rint(lateral * gain)
        reference = np.rint(foreground[0] * gain)
        for (arm, _, donor), front in zip(views, foreground, strict=True):
            fg_int = np.rint(front * gain)
            stereo = fg_int + noise_int
            raw = mono(stereo)
            spatial = np.rint(apply_filter(stereo, filters[room][side]))
            require(
                max(abs(stereo).max(), abs(raw).max(), abs(spatial).max()) < 32768, "Authored audio clips"
            )
            data = dict(zip(STEMS, map(to_bytes, (stereo, fg_int, noise_int, raw, spatial)), strict=True))
            truth = group["authored_range"] if arm == "synthetic" else None
            changed = fingerprint_range(fg_int, reference)
            support = None
            acoustic = None
            if group["authored_range"] and arm != "untouched":
                a, b = group["authored_range"]
                end = min(DURATION, b + len(impulse) - 1)
                require(
                    np.array_equal(reference[:a], fg_int[:a])
                    and np.array_equal(reference[end:], fg_int[end:]),
                    "Change is outside causal measured room support",
                )
                require(
                    changed is not None and a <= changed[0] < changed[1] <= end, "Invalid recorded support"
                )
                support = [a, end]
            if truth:
                a, b = truth
                ref = fg_int.mean(axis=1)
                delta = {
                    name: values - ref
                    for name, values in {
                        "original": raw,
                        "spatial": spatial,
                        "wrong_side": apply_filter(stereo, filters[room][other_side]),
                        "wrong_rig": apply_filter(stereo, filters[other_room][side]),
                    }.items()
                }
                acoustic = {
                    "native_snr_db": {name: power_ratio(ref, value) for name, value in delta.items()},
                    "range_snr_db": {
                        name: power_ratio(ref[a:b], value[a:b]) for name, value in delta.items()
                    },
                    "target_preservation_db": power_ratio(
                        ref, apply_filter(fg_int, filters[room][side]) - ref
                    ),
                }
            row = group | {
                "id": f"{group['group']}-{arm}",
                "truth": truth,
                "arm": arm,
                "donor": donor,
                "room_causal_support": support,
                "change_energy_range": changed,
                "acoustic": acoustic,
                "pcm": {name: digest(value) for name, value in data.items()},
            }
            rows.append((row, data))
    return rows, ancestry, room_receipts


def acoustics(rows):
    result = {}
    for panel in STUDY:
        edits = [row["acoustic"] for row in rows if row["panel"] == panel and row["truth"] is not None]
        require(len(edits) == 6, "Expected six positive groups per role")
        delta = {
            span: {
                arm: median(row[span][arm] - row[span]["original"] for row in edits)
                for arm in ("spatial", "wrong_side", "wrong_rig")
            }
            for span in ("native_snr_db", "range_snr_db")
        }
        preservation = median(row["target_preservation_db"] for row in edits)
        result[panel] = {
            "gain_db": delta,
            "target_preservation_db": preservation,
            "pass": delta["native_snr_db"]["spatial"] >= 4
            and delta["range_snr_db"]["spatial"] >= 3
            and preservation >= 35,
        }
    return result


def freeze(args):
    rows, ancestry, room_receipts = build(args.archive)
    args.run.mkdir(mode=0o700)
    (args.run / "pcm").mkdir()
    for row, data in rows:
        for name, raw in data.items():
            (args.run / "pcm" / f"{row['id']}-{name}.pcm").write_bytes(raw)
    roster = [row for row, _ in rows]
    write(args.run / "roster.json", roster)
    lock = {
        "schema": SCHEMA,
        "method": METHOD,
        "runner_sha256": sha(Path(__file__)),
        "roster_sha256": sha(args.run / "roster.json"),
        "reused_runner_sha256": sha(Path(fresh.__file__)),
        "original_point": fresh.check_discovery()["current"],
        "rooms": room_receipts,
        "ancestry": ancestry,
        "assets": OLD.old.voice.ASSETS,
        "acoustics": acoustics(roster),
    }
    write(args.run / "freeze.json", lock)
    print(lock["acoustics"], flush=True)


def bound(args):
    lock, rows = (read(args.run / f"{name}.json") for name in ("freeze", "roster"))
    require(
        lock["method"] == METHOD
        and lock["runner_sha256"] == sha(Path(__file__))
        and lock["roster_sha256"] == sha(args.run / "roster.json")
        and lock["reused_runner_sha256"] == sha(Path(fresh.__file__))
        and lock["acoustics"] == acoustics(rows),
        "Frozen runner or roster changed",
    )
    return lock, rows


def exact_pcm(args, row, arm):
    value = (args.run / "pcm" / f"{row['id']}-{arm}.pcm").read_bytes()
    require(
        len(value) == DURATION * (2 if arm in ARMS else 4) and digest(value) == row["pcm"][arm],
        "Exact PCM commitment differs",
    )
    return value


def score(args, lock, rows, panel):
    require(
        all(result["pass"] for result in lock["acoustics"].values()),
        "Acoustic preflight does not qualify neural scoring",
    )
    if panel == "held":
        point_inputs(args)
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    OLD.old.spectra.incumbent.network_free()
    model = load_voice({"weights": str(args.voice), "assets": lock["assets"]})
    require(
        sha(args.voice / "silero_vad.jit") == lock["assets"]["silero_vad.jit"], "Pinned speech gate differs"
    )
    gate = SpeechActivity(args.voice / "silero_vad.jit")
    ledger, calls = [], Counter()
    began = time.monotonic()
    for row in rows:
        if row["panel"] != panel:
            continue
        arms = {}
        for arm in ARMS:
            raw = exact_pcm(args, row, arm)
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            spans = gate.spans(pcm)
            arms[arm] = []
            for a, b in WINDOWS:
                value = None
                if OLD.old.window_in_speech(spans, a / RATE, b / RATE):
                    (probability,) = model.predict([pcm[a:b]])
                    require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid score")
                    value = OLD.old.calibration.logit(probability)
                    calls[arm] += 1
                arms[arm].append({"range": [a, b], "sha256": digest(raw[2 * a : 2 * b]), "score": value})
        ledger.append({"id": row["id"], "arms": arms})
        if len(ledger) % 6 == 0:
            print(panel, len(ledger), "views", sum(calls.values()), "forwards", flush=True)
    write(
        args.run / f"{panel}.json",
        {
            "freeze_sha256": sha(args.run / "freeze.json"),
            "rows": ledger,
            "calls": dict(calls),
            "seconds": time.monotonic() - began,
            "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def point_inputs(args):
    point = read(args.run / "points.json")
    require(
        point["freeze_sha256"] == sha(args.run / "freeze.json")
        and point["selection_sha256"] == sha(args.run / "selection.json"),
        "Selection points differ",
    )
    return point


def winner(windows):
    best = max((w for w in windows if w["score"] is not None), key=lambda w: w["score"], default=None)
    return [best["score"], best["range"]] if best else [None, None]


def responses(args, rows, panel):
    ledger = read(args.run / f"{panel}.json")
    require(ledger["freeze_sha256"] == sha(args.run / "freeze.json"), "Scoring used a different input freeze")
    out = []
    for row, event in zip((row for row in rows if row["panel"] == panel), ledger["rows"], strict=True):
        require(
            event["id"] == row["id"] and set(event["arms"]) == set(ARMS), "Scoring ledger identity differs"
        )
        for arm, windows in event["arms"].items():
            raw = exact_pcm(args, row, arm)
            require(
                [tuple(w["range"]) for w in windows] == WINDOWS
                and all(
                    w["sha256"] == digest(raw[2 * a : 2 * b])
                    and (w["score"] is None or math.isfinite(w["score"]))
                    for w, (a, b) in zip(windows, WINDOWS, strict=True)
                ),
                "Score or reported window is not bound to exact PCM",
            )
        shared = {
            arm: [
                windows[i]
                for i in range(len(WINDOWS))
                if all(event["arms"][a][i]["score"] is not None for a in ARMS)
            ]
            for arm, windows in event["arms"].items()
        }
        out.append(
            row
            | {
                "responses": {arm: winner(windows) for arm, windows in event["arms"].items()}
                | {f"{arm}_shared": winner(windows) for arm, windows in shared.items()}
            }
        )
    return out


def fit(args, rows):
    selected = [row for row in responses(args, rows, "selection") if row["truth"] is None]
    require(
        len(selected) == 20 and len({row["group"] for row in selected}) == 8,
        "Unexpected control point roster",
    )
    points = {
        arm: max(row["responses"][arm][0] for row in selected if row["responses"][arm][0] is not None)
        for arm in ARMS
    }
    require(
        all(
            sum(
                any(row["group"] == group and row["responses"][arm][0] is not None for row in selected)
                for group in {row["group"] for row in selected}
            )
            >= 5
            for arm in ARMS
        ),
        "Too few assessed genuine control groups",
    )
    return {
        "freeze_sha256": sha(args.run / "freeze.json"),
        "selection_sha256": sha(args.run / "selection.json"),
        "values": points,
    }


def counts(rows, point):
    arms = (*ARMS, *(f"{arm}_shared" for arm in ARMS))
    pair_counts = {}
    for arm in arms:
        groups = {(row["group"], row["arm"]): row["responses"][arm][0] for row in rows}
        pairs = Counter()
        for row in rows:
            if row["truth"] is None:
                continue
            for ordinary in ("ordinary_same_host", "ordinary_other_voice"):
                left, right = row["responses"][arm][0], groups[row["group"], ordinary]
                if left is not None and right is not None:
                    pairs[f"{ordinary}_comparable"] += 1
                    pairs[f"{ordinary}_synthetic_higher"] += left > right
        pair_counts[arm] = dict(sorted(pairs.items()))
    return {
        "edited": sum(row["truth"] is not None for row in rows),
        "genuine_views": sum(row["truth"] is None for row in rows),
        "fixed_matched_selection": {
            arm: OLD.performance(rows, arm, point[arm.removesuffix("_shared")]) for arm in arms
        },
        "equal_false_groups": {
            str(b): {arm: OLD.equal_alarm(rows, arm, b) for arm in arms} for b in (0, 1, 2)
        },
        "useful_without_threshold": {
            arm: OLD.performance(rows, arm, -math.inf)["useful_ranges"] for arm in arms
        },
        "ranking": pair_counts,
    }


def result(args, lock, rows):
    point = point_inputs(args)
    require(point == fit(args, rows), "Frozen points cannot be reproduced")
    selected, held = (responses(args, rows, panel) for panel in STUDY)
    all_rows = selected + held
    families = {
        family: sum(row["truth"] is not None and row["family"] == family for row in held)
        for family in fresh.FAMILIES
    }
    result = {
        "schema": SCHEMA,
        "method": METHOD,
        "acoustic": lock["acoustics"],
        "roles": {
            panel: counts(scope, point["values"])
            for panel, scope in zip(STUDY, (selected, held), strict=True)
        },
        "earlier_original_fixed": OLD.performance(held, "original", lock["original_point"]),
        "family_held_edited": families,
        "calls": {p: read(args.run / f"{p}.json")["calls"] for p in STUDY},
        "commitments": {
            name: sha(args.run / f"{name}.json") for name in ("freeze", "roster", "points", *STUDY)
        },
    }
    for family in families:
        scoped = [row for row in held if row["family"] == family]
        result.setdefault("family_fixed", {})[family] = {
            arm: OLD.performance(scoped, arm, point["values"][arm]) for arm in ARMS
        }
    return result, all_rows


def audit(args, lock, rows):
    rebuilt, ancestry, rooms = build(args.archive)
    require(
        [row for row, _ in rebuilt] == rows and ancestry == lock["ancestry"] and rooms == lock["rooms"],
        "Exact source replay differs",
    )
    for row, raw in rebuilt:
        require(
            all(value == exact_pcm(args, row, arm) for arm, value in raw.items()),
            "Physical channel replay differs",
        )
    if (args.run / "result.json").exists():
        aggregate, decisions = result(args, lock, rows)
        require(
            aggregate == read(args.run / "result.json") and decisions == read(args.run / "decisions.json"),
            "Model decision replay differs",
        )
    print(
        f"Reproduced {len(rows)} stereo inputs, matched controls, all 16 kHz sample clocks, and existing score receipts",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "selection", "fit", "held", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--voice", type=Path, default=fresh.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args)
        return
    lock, rows = bound(args)
    if args.stage in STUDY:
        score(args, lock, rows, args.stage)
    elif args.stage == "fit":
        write(args.run / "points.json", fit(args, rows))
    elif args.stage == "report":
        aggregate, decisions = result(args, lock, rows)
        write(args.run / "result.json", aggregate)
        write(args.run / "decisions.json", decisions)
        print(aggregate["roles"]["held"], flush=True)
    else:
        audit(args, lock, rows)


if __name__ == "__main__":
    main()
