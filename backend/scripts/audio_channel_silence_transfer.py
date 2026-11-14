"""Re-run the exact channel-silence test on a verified local AMI stereo intake."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio
from app.evidence.audio_silence import measure as mono_silence

RECEIPTS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
CONDITIONS = (
    ("original", 3840, ()),
    ("left_120ms", 1920, (0,)),
    ("right_240ms", 3840, (1,)),
    ("attenuation", 3840, ()),
    ("both_muted", 3840, (0, 1)),
    ("right_240ms_aac", 3840, (1,)),
)


def run(qualified_dir: Path, intake: Path) -> dict:
    parent = json.loads((RECEIPTS / "audio-ami-native-stereo-complete-v291.json").read_text())
    receipt = intake.read_bytes()
    if hashlib.sha256(receipt).hexdigest() != parent["qualification_sha256"]:
        raise ValueError("The AMI intake receipt differs from the published commitment")
    cases = json.loads(receipt)["cases"]
    counts = {}
    with tempfile.TemporaryDirectory() as temp:
        wav, aac = (Path(temp) / name for name in ("fixture.wav", "fixture.m4a"))
        for index, row in enumerate(cases):
            path = qualified_dir / (row["case"] + ".wav")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["stereo_wav_sha256"]:
                raise ValueError(f"Qualified stereo input differs from the recorded intake for {row['case']}")
            with wave.open(str(path), "rb") as reader:
                original = np.frombuffer(reader.readframes(3 * SAMPLE_RATE), dtype="<i2").reshape(-1, 2)
            start = 11357 + 233 * (index % 12)
            for condition, samples, muted in CONDITIONS:
                end = start + samples
                modified = original.copy()
                modified[start:end, list(muted)] = 0
                if condition == "attenuation":
                    modified[start:end, 1] //= 16
                with wave.open(str(wav), "wb") as writer:
                    writer.setparams((2, 2, SAMPLE_RATE, 0, "NONE", ""))
                    writer.writeframes(modified.tobytes())
                path = wav
                if condition.endswith("_aac"):
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav),
                                    "-c:a", "aac", "-b:a", "96k", str(aac)], check=True, timeout=20)
                    path = aac
                decoded = decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), inspect_stereo=True)
                stereo = decoded.mapping["stereo_context"]
                selected = stereo["channel_silence"]["selected"]
                mono = mono_silence(decoded.pcm, decoded.mapping["timeline"])["selected"]
                stats = counts.setdefault(condition, dict(total=0, flags=0, exact=0, mono_exact=0, one_second_flags=0))
                stats["total"] += 1
                stats["flags"] += bool(selected)
                channel = ("left", "right")[muted[0]] if len(muted) == 1 else None
                stats["exact"] += any((r["start_sample"], r["end_sample"], r["silent_channel"]) == (start, end, channel)
                                      for r in selected)
                stats["mono_exact"] += any((r["start_sample"], r["end_sample"]) == (start, end) for r in mono)
                stats["one_second_flags"] += any(r["status"] == "strong_imbalance" for r in stereo["windows"])
    return {"schema": "chipmunk.stereo-channel-silence-transfer/1", "meeting_sessions": parent["sessions"],
            "source_excerpts": len(cases), "rows": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualified_dir", type=Path)
    parser.add_argument("intake_receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.qualified_dir, args.intake_receipt), indent=2))
