"""Evaluate blind stereo separation before using it to screen short voice edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, rfft

RATE = 16000
SAMPLES = 6 * RATE
WINDOW = 1024
STEP = 256
STEPS = 32
SOURCE = Path("docs/evaluation-panels/audio-measured-spatial-transfer-2026-11-12.json")
SCHEMA = "chipmunk.audio-blind-spatial-transfer/1"
METHOD = {
    "source": "Reused measured stereo panel. Surrey selection and BBC transfer use disjoint source groups and AMI sessions. One foreground and one lateral playback feed, which can contain several talkers. Reused cases have prior detector exposure.",
    "separation": "1024-sample square-root periodic Hann analysis, 256-sample hop, 32 iterative-projection AuxIVA sweeps with diagonal loading of 1e-5 times the covariance mean diagonal. Use the same two channels and all frequencies for fitting. Project each estimate back onto the stereo mean and overlap-add. No room response, directions, labeled edit, or source stem supplied to the separator.",
    "projection": "Return both unassigned candidates. Choose the candidate closest to the exact room-rendered foreground only to measure the best achievable reconstruction bound, and independently evaluate a stem-free channel-similarity choice. Never use true stems to choose a candidate for a future detector.",
    "routing": "For each candidate, project separately onto each channel, compare left-right difference energy to mean energy from 300–3400 Hz, and prefer the lower ratio. Abstain on rank-poor pairs or when the two ratios are within 1 dB.",
    "controls": "Compare to the submitted stereo mean, the previous measured-room reconstruction, and the exact foreground. Include background-only clips. Account for clipped estimates rather than silently discarding them. The selection and held cases use identical separator constants.",
    "gate": "Advance a stem-free voice scoring trial only if both roles achieve at least +3 dB median foreground reconstruction improvement for edited clips overall and within the edit, all six synthetic clips per role are routed, and none of those selected candidates clips. An oracle gain alone does not qualify.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, allow_nan=False, indent=2, sort_keys=True)
        out.write("\n")


def load_pcm(parent, row, name):
    payload = (parent / "pcm" / f"{row['id']}-{name}.pcm").read_bytes()
    channels = 2 if name in {"stereo", "foreground", "background"} else 1
    require(len(payload) == SAMPLES * channels * 2 and sha(payload) == row["pcm"][name], "Source PCM differs")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float64)
    return values.reshape(SAMPLES, 2) if channels == 2 else values


def analyze(signal, fft=rfft):
    window = np.sqrt(.5 - .5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW))
    padded = np.pad(signal, ((WINDOW // 2, WINDOW // 2), (0, 0)))
    blocks = sliding_window_view(padded, WINDOW, axis=0)[::STEP]
    return fft(blocks * window, axis=-1).transpose(2, 0, 1), window


def synthesize(spectrum, window, inverse=irfft):
    blocks = inverse(spectrum.transpose(1, 2, 0), n=WINDOW, axis=-1) * window
    output = np.zeros((SAMPLES + WINDOW, 2))
    weight = np.zeros(SAMPLES + WINDOW)
    for index, block in enumerate(blocks):
        start = index * STEP
        output[start:start + WINDOW] += block.T
        weight[start:start + WINDOW] += window ** 2
    section = slice(WINDOW // 2, WINDOW // 2 + SAMPLES)
    return output[section] / weight[section, None]


def separate(signal, *, fft=rfft, inverse=irfft):
    spectrum, window = analyze(signal, fft)
    band = (np.arange(len(spectrum)) * RATE / WINDOW >= 300) & (np.arange(len(spectrum)) * RATE / WINDOW <= 3400)
    covariance = spectrum[band].reshape(-1, 2)
    singular = np.linalg.eigvalsh(covariance.T.conj() @ covariance)
    if singular[0] <= 1e-4 * singular[1] or singular[1] <= 1:
        return None
    identity = np.eye(2, dtype=np.complex128)
    filters = np.broadcast_to(identity, (len(spectrum), 2, 2)).copy()
    floor = np.maximum(1e-12, np.mean(abs(spectrum) ** 2, axis=(1, 2)) * 1e-9)
    for _ in range(STEPS):
        for channel in range(2):
            estimated = np.einsum("fi,fti->ft", filters[:, channel], spectrum)
            inverse_radius = 1 / np.sqrt(np.maximum(np.mean(abs(estimated) ** 2, axis=0), 1e-12))
            cov = np.einsum("fti,t,ftj->fij", spectrum, inverse_radius, spectrum.conj()) / spectrum.shape[1]
            diagonal = np.maximum(floor, cov.trace(axis1=1, axis2=2).real * .5e-5)
            cov += diagonal[:, None, None] * identity
            right = np.broadcast_to(identity[:, channel, None], (len(cov), 2, 1))
            vector = np.linalg.solve(filters @ cov, right)[..., 0]
            energy = np.einsum("fi,fij,fj->f", vector.conj(), cov, vector).real
            filters[:, channel] = vector.conj() / np.sqrt(energy)[:, None]
    estimates = np.einsum("fij,ftj->fti", filters, spectrum)
    projection = np.linalg.inv(filters)
    mid = projection.mean(axis=1)
    centered = estimates * mid[:, None, :]
    residual = estimates * (projection[:, 0, :] - projection[:, 1, :])[:, None, :]
    require(np.max(abs(centered.sum(axis=2) - spectrum.mean(axis=2))) < 1e-5, "Projections do not conserve the input")
    source = synthesize(centered, window, inverse)
    require(np.max(abs(source.sum(axis=1) - signal.mean(axis=1))) < 1e-5, "Sample-domain conservation failed")
    power = np.sum(abs(centered[band]) ** 2, axis=(0, 1))
    side = np.sum(abs(residual[band]) ** 2, axis=(0, 1))
    ratio = 10 * np.log10(np.maximum(side, 1e-12) / np.maximum(power, 1e-12))
    route = int(np.argmin(ratio)) if abs(ratio[0] - ratio[1]) >= 1 and np.min(power) >= 1 else None
    return source, route, ratio


def snr(reference, waveform):
    reference_power = float(np.dot(reference, reference))
    if reference_power == 0:
        return None
    error = reference - waveform
    return round(10 * np.log10(reference_power / max(1e-12, float(np.dot(error, error)))), 6)


def measure(parent, roster, *, fft=rfft, inverse=irfft):
    rows = []
    for index, case in enumerate(roster):
        stereo = load_pcm(parent, case, "stereo")
        foreground = load_pcm(parent, case, "foreground")
        background = load_pcm(parent, case, "background")
        mean = load_pcm(parent, case, "original")
        room = load_pcm(parent, case, "spatial")
        require(np.array_equal(stereo, foreground + background) and np.array_equal(mean, np.rint(stereo.mean(axis=1))),
                "Parent mix or rounding differs")
        truth = case["truth"]
        span = slice(*truth) if truth else slice(None)
        reference = foreground.mean(axis=1)
        outcome = separate(stereo, fft=fft, inverse=inverse)
        results = {}
        ratio = None
        route = None
        views = {"mean": mean, "measured_room": room}
        if outcome is not None:
            waveforms, route, channel_balance = outcome
            ratio = np.round(channel_balance, 6).tolist()
            views |= {"output_0": waveforms[:, 0], "output_1": waveforms[:, 1]}
        for name, view in views.items():
            rounded = np.rint(view)
            results[name] = {
                "full_snr_db": snr(reference, rounded),
                "edit_snr_db": snr(reference[span], rounded[span]) if truth else None,
                "clips": int(np.count_nonzero((rounded < -32768) | (rounded > 32767))),
                "sha256": sha(np.clip(rounded, -32768, 32767).astype("<i2").tobytes()),
            }
        rows.append({"case": index, "role": case["panel"], "kind": case["arm"],
                     "route": route, "balance_db": ratio, "truth": truth, "results": results})
    return rows


def summarize(rows):
    summary = {}
    for role in ("selection", "held"):
        role_rows = [row for row in rows if row["role"] == role]
        edits = [row for row in role_rows if row["truth"]]
        routed = [row for row in edits if row["route"] is not None]
        require(len(role_rows) == 26 and len(edits) == 6, "Panel is incomplete")
        arms = {}
        for name in ("oracle", "blind", "measured_room"):
            selected = []
            for row in (routed if name == "blind" else edits):
                values = row["results"]
                outputs = [key for key in values if key.startswith("output_")]
                if name == "oracle" and not outputs:
                    continue
                chosen = (max(outputs, key=lambda key: values[key]["full_snr_db"]) if name == "oracle"
                          else f"output_{row['route']}" if name == "blind" else name)
                selected.append((values[chosen], values["mean"]))
            arms[name] = {
                "views": len(selected),
                "full_gain_db": round(median(a["full_snr_db"] - b["full_snr_db"] for a, b in selected), 4) if selected else None,
                "edit_gain_db": round(median(a["edit_snr_db"] - b["edit_snr_db"] for a, b in selected), 4) if selected else None,
                "clipped_views": sum(a["clips"] > 0 for a, _ in selected),
            }
        blind = arms["blind"]
        ordinary = [row for row in role_rows if row["kind"] != "synthetic"]
        summary[role] = {
            "arms": arms, "ordinary_routes": sum(row["route"] is not None for row in ordinary),
            "background_only_routes": sum(row["kind"] == "background_only" and row["route"] is not None for row in ordinary),
            "advance": blind["views"] == 6 and blind["full_gain_db"] >= 3 and blind["edit_gain_db"] >= 3 and blind["clipped_views"] == 0,
        }
    return summary


def binding(parent):
    public = read(SOURCE)
    roster = (parent / "roster.json").read_bytes()
    require(sha(roster) == public["commitments"]["roster"], "Parent roster differs from the published commitment")
    return {"schema": SCHEMA, "method": METHOD, "source_roster_sha256": sha(roster),
            "runner_sha256": sha(Path(__file__).read_bytes())}, json.loads(roster)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "replay"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock, roster = binding(args.parent)
    lock_path = args.output / "freeze.json"
    result_path = args.output / "result.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(lock_path, lock)
        return
    require(lock == read(lock_path), "Frozen experiment changed")
    if args.phase == "replay":
        rows = measure(args.parent, roster, fft=np.fft.rfft, inverse=np.fft.irfft)
        saved = read(result_path)
        require(rows == saved["rows"] and summarize(rows) == saved["summary"], "Alternate FFT replay differs")
    else:
        rows = measure(args.parent, roster)
        write(result_path, {"freeze_sha256": sha(lock_path.read_bytes()), "rows": rows, "summary": summarize(rows)})
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
