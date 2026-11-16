"""Compare original and estimated-background views on exact matched mixtures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_gtcrn_traffic as enhancement
from scripts import environmental_component_inputs as source
from scripts import environmental_sound_transfer as earlier
from scripts.audio_focused_context_transfer import digest, read, save
from scripts.audio_phase_corroboration import sample_bytes
from scripts.environmental_embedding_transfer import PINS

SCHEMA = "chipmunk.environmental-component-transfer/1"
SCENES = ("isolated", "genuine_6db", "genuine_balanced", "edited_6db", "genuine_opus_6db")
ARMS = ("mixture", "estimated_background", "both", "both_paired")
ARCHIVED = ("archived_magnitude", "archived_texture")
POINTS = ("wider", "strict")
FRAMES = earlier.N
METHOD = {
    "rate": earlier.RATE, "samples": FRAMES, "scenes": SCENES,
    "signals": "Original decoded input and aligned input minus the estimated speech waveform.",
    "mixture": "Retain identical foreground samples and shared output gain across the origins in a group.",
    "features": "The earlier per-band magnitude, flatness, and envelope consistency measures, independently on both views.",
    "fit": "Balanced ridge fit with 0.50 penalty. Both-view arms scale each standardized block by sqrt(block width). "
           "The paired variant adds a within-source difference objective with equal total pair and individual weights.",
    "selection": "One point per arm across all scenes. Use the third-highest ordinary-group maximum for the wider point "
                 "and second-highest for the strict point. Raise the point to at least the largest voice-only reading. "
                 "Flags strictly exceed these points. Choose an arm by the weakest selection generator first, then total "
                 "mixture hits, then fewer ordinary alerts. Score every arm independently on held inputs.",
}


def sha(path):
    return digest(path.read_bytes())


def checked_pcm(path, expected, samples=FRAMES):
    raw = path.read_bytes()
    if len(raw) != samples * 2 or digest(raw) != expected:
        raise ValueError("The exact audio input changed")
    return raw


def archived_voice_pairs(speech):
    roster = speech / "roster.json"
    if sha(roster) != PINS["speech_roster"]:
        raise ValueError("The archived genuine speech panel changed")
    rows = [row for row in read(roster) if row["placement"] == "clean"
            and row["family"] == "genuine" and row["role"] == "archived"]
    if len(rows) != 8 or len({row["group"] for row in rows}) != 8:
        raise ValueError("The fit and selection use separate foreground readers")
    return [{"source": row["group"], "author": row["group"], "genuine": sample_bytes(speech, row)} for row in rows]


def held_voice_pairs(matched):
    lock = read(matched / "freeze.json")
    cases = {r["id"]: r for r in lock["cases"]}
    results = []
    for pair in lock["pairs"]:
        edit = cases[f"{pair['edit']}/edited/clean"]
        ordinary = cases[f"{pair['ordinary']}/ordinary/clean"]
        if not edit["samples"] == ordinary["samples"] or edit["group"] != ordinary["group"]:
            raise ValueError("The paired voice clock or source group differs")
        count = edit["samples"]
        center = count // 2
        left, right = min((r for r in edit["truth"] if 0 < r[1] - r[0] <= FRAMES
                           and (count - 2 * FRAMES) // 2 <= r[0] < r[1] <= (count + 2 * FRAMES) // 2),
                          key=lambda r: abs(r[0] + r[1] - 2 * center))
        start = min(max(0, (left + right - FRAMES) // 2), count - FRAMES)
        if not start <= left < right <= start + FRAMES:
            raise ValueError("The foreground excerpt misses its annotated edit")

        def slice_pcm(row):
            path = matched / "pcm" / f"{row['pcm_sha256']}.pcm"
            return checked_pcm(path, row["pcm_sha256"], count)[2 * start:2 * (start + FRAMES)]

        results.append({"source": pair["edit"], "author": edit["group"], "family": edit["family"],
                        "start_sample": start, "edit_samples": [left - start, right - start],
                        "genuine": slice_pcm(ordinary), "edited": slice_pcm(edit)})
    if len(results) != 14 or len({row["family"] for row in results}) != 6:
        raise ValueError("Expected the 14 verified original utterances across six generator families")
    return results


def component_mixtures(backgrounds, voice, environment_db):
    """Use one output gain for every origin after setting the background level."""
    foreground = np.frombuffer(voice, "<i2").astype(np.float64)
    environment = [np.frombuffer(raw, "<i2").astype(np.float64) for raw in backgrounds]
    foreground_norm = np.linalg.norm(foreground)
    norms = np.asarray([np.linalg.norm(item) for item in environment])
    if not foreground_norm or (norms <= 0).any():
        raise ValueError("A source stem is silent")
    scales = foreground_norm * 10 ** (environment_db / 20) / norms
    drafts = [foreground + scale * audio for scale, audio in zip(scales, environment, strict=True)]
    gain = min(1., 30_000 / max(np.max(np.abs(draft)) for draft in drafts))
    return [np.rint(draft * gain).astype("<i2").tobytes() for draft in drafts], float(gain), scales.tolist()


def asset_binding(folder):
    assets = folder / "assets"
    model_sha, license_sha = sha(assets / enhancement.MODEL), sha(assets / "LICENSE")
    if (model_sha, license_sha) != (enhancement.MODEL_SHA, enhancement.LICENSE_SHA):
        raise ValueError("The speech estimator must use the pinned MIT release")
    return {"revision": enhancement.REVISION, "model_sha256": model_sha, "license_sha256": license_sha,
            "runtime": "onnxruntime 1.23.2 CPU single-thread",
            "code": {Path(module.__file__).name: sha(Path(module.__file__))
                     for module in (source, enhancement, earlier)} | {Path(__file__).name: sha(Path(__file__))}}


def prepare(folder, speech, matched, old):
    plan_path = folder / "environment-plan.json"
    acquisition_path = folder / "environment-acquisition.json"
    plan, acquisition = read(plan_path), read(acquisition_path)
    if acquisition["plan_sha256"] != sha(plan_path) or len(acquisition["rows"]) != 264:
        raise ValueError("The background acquisition does not match the frozen plan")
    sources = {(r["group"], r["lane"]): r for r in acquisition["rows"]}
    archived = archived_voice_pairs(speech)
    voices = {"fit": archived[:4], "selection": archived[4:], "held": held_voice_pairs(matched)}
    voice_bytes = {digest(row["genuine"]) for row in archived}
    if voice_bytes & {digest(row[key]) for row in voices["held"] for key in ("genuine", "edited")}:
        raise ValueError("A held foreground excerpt repeats development audio")
    assets = asset_binding(folder)
    cases = []
    deliveries = folder / "deliveries"
    deliveries.mkdir(mode=0o700)

    def store(raw, **metadata):
        if len(raw) != FRAMES * 2:
            raise ValueError("Prepared sample count differs")
        identifier = len(cases)
        (deliveries / f"{identifier:04d}.pcm").write_bytes(raw)
        cases.append({"id": identifier, "pcm_sha256": digest(raw), **metadata})

    counts = Counter()
    for group in plan["groups"]:
        index, role = group["group"], group["role"]
        voice = voices[role][counts[role] % len(voices[role])]
        counts[role] += 1
        lanes = list(group["parts"])
        backgrounds = [checked_pcm(folder / "environment" / f"{index:02d}-{lane}.pcm",
                                   sources[index, lane]["pcm_sha256"]) for lane in lanes]
        for lane, raw in zip(lanes, backgrounds, strict=True):
            store(raw, group=index, role=role, lane=lane, scene="isolated", environment_sha256=digest(raw))
        mixtures = [("genuine_6db", "genuine", -6), ("genuine_balanced", "genuine", 0)]
        if role == "held":
            mixtures.append(("edited_6db", "edited", -6))
        for scene, voice_kind, db in mixtures:
            foreground = voice[voice_kind]
            prepared, gain, scales = component_mixtures(backgrounds, foreground, db)
            for lane, raw, background, scale in zip(lanes, prepared, backgrounds, scales, strict=True):
                blend = {"group": index, "role": role, "lane": lane, "environment_sha256": digest(background),
                         "voice": voice["source"], "voice_kind": voice_kind, "voice_sha256": digest(foreground),
                         "common_gain": gain, "environment_scale": scale, "environment_relative_db": db}
                store(raw, scene=scene, **blend)
                if scene == "genuine_6db" and role == "held":
                    store(earlier.convert(folder, raw), scene="genuine_opus_6db", input_pcm_sha256=digest(raw), **blend)
    for role, panel in voices.items():
        for voice in panel:
            for lane in ("genuine", "edited") if role == "held" else ("genuine",):
                store(voice[lane], group=None, role=role, lane="voice_only", scene=f"{lane}_only", voice=voice["source"])
    for name in ("delivery.wav", "delivery.opus"):
        (folder / name).unlink()
    metadata = {role: [{key: digest(value) if isinstance(value, bytes) else value for key, value in row.items()}
                       for row in rows] for role, rows in voices.items()}
    save(folder / "case-roster.json", cases)
    save(folder / "freeze.json", {"schema": SCHEMA, "method": METHOD, "assets": assets,
                                  "sources": {"plan": sha(plan_path), "acquisition": sha(acquisition_path),
                                              "archived_heads": sha(old / "heads.json"),
                                              "matched_voices": sha(matched / "freeze.json")},
                                  "voice_sources": metadata, "cases_sha256": sha(folder / "case-roster.json")})
    print(f"Froze {len(cases)} aligned audio deliveries across 80 independent environmental source groups.", flush=True)


def load_panel(folder):
    lock = read(folder / "freeze.json")
    paths = {"plan": folder / "environment-plan.json", "acquisition": folder / "environment-acquisition.json"}
    roster = folder / "case-roster.json"
    if (lock["schema"] != SCHEMA or lock["assets"] != asset_binding(folder)
            or lock["method"] != json.loads(json.dumps(METHOD)) or lock["cases_sha256"] != sha(roster)
            or any(lock["sources"][key] != sha(path) for key, path in paths.items())):
        raise ValueError("The frozen procedure or inputs changed")
    return lock, read(roster)


def session_for(folder):
    import onnxruntime as ort

    if ort.__version__ != "1.23.2":
        raise ValueError("Unexpected local inference runtime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder / "assets" / enhancement.MODEL), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if {row.name: row.shape for row in session.get_inputs()} != enhancement.INPUTS:
        raise ValueError("Unexpected speech estimator input signature")
    return session


def measurements(folder, held):
    _, cases = load_panel(folder)
    cases = [row for row in cases if (row["role"] == "held") == held]
    session = session_for(folder)
    rows = []
    for row in cases:
        raw = checked_pcm(folder / "deliveries" / f"{row['id']:04d}.pcm", row["pcm_sha256"])
        wave = enhancement.float_pcm(raw)
        residual = enhancement.pcm(wave - enhancement.denoise(wave, session))
        rows.append(row | {"mixture": earlier.features(raw), "estimated_background": earlier.features(residual),
                           "estimated_background_sha256": digest(residual)})
        if len(rows) % 100 == 0:
            print(f"Measured {len(rows)}/{len(cases)} aligned views", flush=True)
    return rows


def vector(row, arm):
    views = (arm,) if arm in ARMS[:2] else ARMS[:2]
    return np.asarray([value for view in views for key in ("magnitude", "texture") for value in row[view][key]])


def calibrated_head(rows, arm):
    fit = [row for row in rows if row["role"] == "fit" and row["group"] is not None]
    matrix = np.stack([vector(row, arm) for row in fit])
    synthetic = np.array([row["lane"] != "real" for row in fit])
    weight = np.where(synthetic, .5 / synthetic.sum(), .5 / (~synthetic).sum())
    mean = weight @ matrix
    scale = np.maximum(np.sqrt(weight @ (matrix - mean) ** 2), 1e-5)
    if arm in ARMS[2:]:
        scale *= np.sqrt(matrix.shape[1] / 2)
    z = (matrix - mean) / scale
    system = (z.T * weight) @ z + .50 * np.eye(matrix.shape[1])
    target = (z.T * weight) @ (synthetic.astype(float) - .5)
    if arm == "both_paired":
        controls = {(row["group"], row["scene"]): point for row, point in zip(fit, z, strict=True) if row["lane"] == "real"}
        pairs = np.stack([point - controls[row["group"], row["scene"]]
                          for row, point in zip(fit, z, strict=True) if row["lane"] != "real"])
        system += pairs.T @ pairs / len(pairs)
        target += pairs.mean(axis=0)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": np.linalg.solve(system, target).tolist()}


def determine_points(rows, arm):
    controls = defaultdict(list)
    anchors = []
    for row in rows:
        response = row["scores"][arm]
        if row["lane"] == "real":
            controls[row["group"]].append(response)
        elif row["lane"] == "voice_only":
            anchors.append(response)
    if len(controls) != source.ROLE_COUNTS["selection"] or len(anchors) != 4:
        raise ValueError("Invalid independent point-selection groups")
    maxima = sorted(map(max, controls.values()))
    return {name: max(maxima[-index], *anchors) for name, index in zip(POINTS, (3, 2), strict=True)}


def rates(rows, points):
    cells = defaultdict(Counter)
    for row in rows:
        for arm, cutoffs in points.items():
            for point, cutoff in cutoffs.items():
                cell = cells[arm, point, row["scene"], row["lane"]]
                cell.update(total=1, flags=row["scores"][arm] > cutoff)
    return {"/".join(key): dict(cell) for key, cell in sorted(cells.items())}


def selection_order(rows, points):
    def rank(arm):
        cutoff = points[arm]["wider"]
        flags = Counter(row["lane"] for row in rows if row["scene"] in SCENES[1:3] and row["scores"][arm] > cutoff)
        hits = [flags[lane] for lane in source.FIT_GENERATORS]
        return -min(hits), -sum(hits), flags["real"], ARMS.index(arm)

    return sorted(ARMS, key=rank)


def fit(folder):
    work = measurements(folder, held=False)
    heads = {arm: calibrated_head(work, arm) for arm in ARMS}
    selected = []
    for row in work:
        if row["role"] == "selection":
            selected.append(public_row(row) | {"scores": {arm: earlier.raw(head, vector(row, arm)) for arm, head in heads.items()}})
    points = {arm: determine_points(selected, arm) for arm in ARMS}
    save(folder / "calibration.json", {"schema": SCHEMA, "freeze_sha256": sha(folder / "freeze.json"), "heads": heads,
                                       "order": selection_order(selected, points), "points": points, "selection": selected})


def public_row(row):
    return {key: value for key, value in row.items() if key not in ARMS[:2]}


def assess(folder, old, output):
    lock, _ = load_panel(folder)
    previous_heads = old / "heads.json"
    calibration = read(folder / "calibration.json")
    if calibration["freeze_sha256"] != sha(folder / "freeze.json") or lock["sources"]["archived_heads"] != sha(previous_heads):
        raise ValueError("The held comparison needs both committed calibrations")
    archived_heads = read(previous_heads)["models"]
    rows = measurements(folder, held=True)
    result = []
    for row in rows:
        values = {arm: earlier.raw(head, vector(row, arm)) for arm, head in calibration["heads"].items()}
        for key, saved in archived_heads.items():
            values[f"archived_{key}"] = earlier.raw(saved["head"], row["mixture"][key])
        result.append(public_row(row) | {"scores": values})
    points = calibration["points"] | {f"archived_{arm}": {"recorded": float(np.nextafter(saved["cutoff"], -np.inf))}
                                      for arm, saved in archived_heads.items()}
    save(output, {"schema": SCHEMA, "freeze": lock, "calibration": calibration, "points": points, "held": result,
                  "source_plan": read(folder / "environment-plan.json"),
                  "acquisition": read(folder / "environment-acquisition.json"), "summary": rates(result, points)})
    print(f"Completed {len(result)} held deliveries after independently locking all decision points.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "assess"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--speech", type=Path)
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--earlier", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.folder, args.speech, args.matched, args.earlier)
    elif args.phase == "fit":
        fit(args.folder)
    else:
        assess(args.folder, args.earlier, args.output)


if __name__ == "__main__":
    main()
