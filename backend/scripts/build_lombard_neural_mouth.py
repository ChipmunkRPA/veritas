# SPDX-License-Identifier: Apache-2.0
"""Build native Lombard neural redraws and geometry-matched ordinary controls."""

from __future__ import annotations

import argparse
import hmac
import json
import resource
import secrets
import shutil
import subprocess
import time
from collections import Counter
from contextlib import closing, contextmanager
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import grid_portrait_frames_v146 as portrait
from scripts import grid_portrait_geometry_v144 as geometry
from scripts import grid_reserve_neural_mouth as retained
from scripts import qualify_lombard_mouth_events as originals
from scripts.video_common_sources_v89 import read, require, sha, write

CONTROL_ROLES = ("original", "resample_ordinary", "delayed_pixels_ordinary")
NEURAL = "source_time_neural"
DELAY_FRAMES = 5
SPEC = {
    "roster": "All 13 preselected qualified source events. Construct and observe the two development sources before the 11 held sources. Preserve every attempted condition and keep speakers disjoint.",
    "render": "Rights-reviewed LivePortrait human weights, strict exact CPU float32 and unchanged bounded convolution. Use the same native source-time crop for both inputs. Reconstruction is an AI redraw and is not presumed synchronized.",
    "controls": "Lossless original, same-time ordinary 256-to-512 bilinear resample, five-picture delayed ordinary pixels. All share source crop, inverse affine, native mouth alpha, and edit interval. At the first and last edited picture use half opacity.",
    "delivery": "Lossless RGB H.264 and original mono float32 PCM in MOV. Preserve all exact native picture timestamps, zero origins, every source audio sample, and unchanged pixels outside the frozen edit geometry.",
    "blind_observer": "Muted stateless picture measurement, blind to case labels, source event observations and waveform. Find an unambiguous near-closed-to-open event against the publisher forced-phone neighborhood plus the fixed late control allowance. Keep ambiguous or missing observations.",
    "qualification": "Rejoin labels after muted observation. Require that a neural or ordinary re-opening remains within one native picture of its genuine counterpart, lies inside the native edit interval, and differs from the separately measured acoustic burst by at most 80 ms. This is local source-event agreement, not perceptual phoneme or full-clip synchrony.",
}


def command(*args, input=None):
    return subprocess.run(args, input=input, capture_output=True, check=True, timeout=45).stdout


def freeze(sources):
    roster = read(sources / "source-receipt.json")
    qualification = read(sources / "qualification.json")
    audio = read(sources / "audio-labels.json")
    visual = read(sources / "visual-labels.json")
    frozen = sources / "event-freeze.json"
    require(
        all(row["freeze_sha256"] == sha(frozen) for row in (qualification, audio, visual))
        and read(frozen) == originals.seal(sources),
        "Original event and source bindings differ",
    )
    admitted = set(qualification["scheduled_render_ids"])
    by_id = {row["ordinal"]: row for row in qualification["cases"]}
    rows = []
    for item, sound, picture in zip(roster["rows"], audio["rows"], visual["rows"], strict=True):
        k = item["ordinal"]
        require(k == sound["ordinal"] == picture["ordinal"], "Original source ordinals differ")
        if k in admitted:
            event = by_id[k]
            require(
                event["status"] == "qualified_original_event"
                and sound["audio_sha256"] == item["inputs"]["audio"]["sha256"]
                and picture["video_sha256"] == item["inputs"]["front"]["sha256"]
                and event["native_edit_frames"][0] >= DELAY_FRAMES,
                "Scheduled source does not support the matched control",
            )
            rows.append({"source": item, "event": event, "audio": sound, "visual": picture})
    require(
        len(rows) == len(admitted) == 13
        and Counter(r["source"]["split"] for r in rows) == {"development": 2, "held": 11}
        and len({r["source"]["group"] for r in rows if r["source"]["split"] == "held"}) == 6,
        "The frozen original schedule differs",
    )
    modules = (Path(__file__), Path(originals.__file__), Path(retained.__file__), Path(portrait.__file__))
    relative = (
        "portrait_expression_drive_v138.py",
        "spatial_conv_install_v141.py",
        "liveportrait_mouth_benchmark.py",
        "review_lombard_neural_mouth.py",
    )
    permissions = (
        "face_landmarker.task",
        "liveportrait-verified-assets.json",
        "liveportrait-source-LICENSE.txt",
    )
    return {
        "schema": "chipmunk.lombard-neural-mouth-freeze/1",
        "spec": SPEC,
        "sources": rows,
        "source_qualification_sha256": sha(sources / "qualification.json"),
        "tool_sha256": {p.name: sha(p) for p in (*modules, *(Path(__file__).with_name(n) for n in relative))},
        "asset_receipts": {n: sha(retained.ASSETS / "permissions" / n) for n in permissions},
        "ffmpeg": command("ffmpeg", "-version").decode().splitlines()[0],
    }


def table_for(path, row):
    table = probe_frames(path, timeout_s=15)
    require(
        table.time_base == row["visual"]["time_base"]
        and list(table.pts) == row["visual"]["pts"]
        and (table.width, table.height) == (720, 480)
        and not (table.rotation or table.interpolated or table.truncated),
        "Native picture clock differs from the frozen original",
    )
    step = (table.pts[1] - table.pts[0]) * table.tb
    require(
        table.pts[0] == 0 and step > 0 and all((b - a) * table.tb == step for a, b in pairwise(table.pts)),
        "Expected exact uniform source picture cadence",
    )
    return table, step


def read_pictures(path, table):
    pictures = []
    for frame in decode_frames(path, table, None, timeout_s=30):
        require(
            frame.index == len(pictures) and frame.pts_verified and frame.pts == table.pts[frame.index],
            "Decoder did not reproduce the native picture clock",
        )
        pictures.append(frame.rgb)
    require(len(pictures) == len(table.pts), "Missing delivered pictures")
    return pictures


def verified_audio(path, row):
    streams = json.loads(
        command(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name,sample_rate,time_base,start_pts,duration_ts,channels",
            "-of",
            "json",
            str(path),
        )
    )["streams"]
    n = row["audio"]["samples"]
    require(
        streams
        == [
            {
                "codec_name": "pcm_f32le",
                "sample_rate": "16000",
                "channels": 1,
                "time_base": "1/16000",
                "start_pts": 0,
                "duration_ts": n,
            }
        ],
        "Delivery changed the zero-origin float audio clock",
    )
    raw = command(
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "-"
    )
    require(
        len(raw) == n * 4 and originals.digest(raw) == row["audio"]["native_pcm_sha256"],
        "Delivery changed a source float audio sample",
    )
    return n


def encode(path, pictures, wav, table, step):
    require(table.tb.numerator == 1, "MOV timescale must reproduce the source tick")
    command(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{table.width}x{table.height}",
        "-framerate",
        str(1 / step),
        "-i",
        "pipe:0",
        "-i",
        str(wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264rgb",
        "-qp",
        "0",
        "-preset",
        "fast",
        "-bf",
        "0",
        "-threads:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-video_track_timescale",
        str(table.tb.denominator),
        "-c:a",
        "pcm_f32le",
        str(path),
        input=b"".join(f.tobytes() for f in pictures),
    )


def crop_geometry(native, interval, module, task):
    from scripts.liveportrait_mouth_benchmark import crop_from_mesh

    result = {}
    for i in range(interval[0] - DELAY_FRAMES, interval[1]):
        mesh, _ = retained.observed(module, task, native[i])
        if mesh is None:
            return None
        crop, matrix, _ = crop_from_mesh(native[i], mesh)
        points = np.asarray([(p.x, p.y, p.z) for p in mesh], dtype=np.float64)
        result[i] = crop, geometry.upscale_matrix(matrix), portrait.alpha_from_mesh(points, native[i].shape)
    return result


@contextmanager
def renderer():
    import torch

    from scripts import spatial_conv_install_v141 as tiling
    from scripts.portrait_expression_drive_v138 import ExpressionRenderer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(146)
    loads = []
    roots = Counter()
    saved_load = torch.nn.Module.load_state_dict

    def exact_load(model, state, strict=True, assign=False):
        require(strict and not assign, "Only strict retained checkpoint loads allowed")
        result = saved_load(model, state, strict=strict, assign=assign)
        actual = model.state_dict()
        require(
            actual.keys() == state.keys() and all(torch.equal(actual[k], state[k]) for k in state),
            "Loaded weights differ from reviewed checkpoints",
        )
        loads.append(len(state))
        return result

    torch.nn.Module.load_state_dict = exact_load
    try:
        model = ExpressionRenderer(retained.ASSETS)
    finally:
        torch.nn.Module.load_state_dict = saved_load
    require(len(loads) == 5, "Expected five exact retained state loads")
    patches = tiling.install(model.renderer.models, lambda _: None)

    def hook(name):
        def counted(*_):
            roots[name] += 1

        return counted

    handles = [m.register_forward_hook(hook(name)) for name, m in model.renderer.models.items()]
    try:
        yield model, loads, roots
    finally:
        for handle in handles:
            handle.remove()
        tiling.restore(patches)


def author(pictures, table, interval, role, crops, model):
    expected = list(pictures)
    hashes = []
    if role == "original":
        return expected, hashes
    start, stop = interval
    for i in range(start, stop):
        source, matrix, alpha = crops[i]
        driver = i - DELAY_FRAMES if role == "delayed_pixels_ordinary" else i
        driving = crops[driver][0]
        generated = model.render(source, driving) if role == NEURAL else cv2.resize(driving, (512, 512))
        portrait.generated_contract(generated)
        warped = cv2.warpAffine(
            generated,
            cv2.invertAffineTransform(matrix),
            (table.width, table.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        expected[i], mask = portrait.composite(
            pictures[i], warped, alpha, 0.5 if i in (start, stop - 1) else 1.0
        )
        require(mask.any(), "A fixed edit picture changed no pixels")
        hashes.append(
            {
                "index": i,
                "pts": table.pts[i],
                "driver_index": driver,
                "driver_pts": table.pts[driver],
                "alpha_sha256": portrait.fingerprint(alpha),
                "mask_sha256": portrait.fingerprint(mask),
                "rgb_sha256": portrait.fingerprint(expected[i]),
            }
        )
    return expected, hashes


def construct(sources, run, split, roles, model=None):
    lock = read(run / "freeze.json")
    require(lock == freeze(sources), "Rendering or source freeze changed")
    require(shutil.disk_usage(run).free >= 1024**3, "Need 1 GiB free to preserve lossless deliveries")
    cv2.setNumThreads(1)
    key = (run / "blind.key").read_bytes()
    mod, task = retained.mp_task()
    ledger = []
    with closing(task):
        for row in lock["sources"]:
            item = row["source"]
            if item["split"] != split:
                continue
            k, span = item["ordinal"], row["event"]["native_edit_frames"]
            original = originals.source(sources, item, "front")
            wav = originals.source(sources, item, "audio")
            table, step = table_for(original, row)
            require(
                originals.digest(originals.native_pcm(wav)[0]) == row["audio"]["native_pcm_sha256"],
                "The scheduled float source waveform differs",
            )
            pictures = read_pictures(original, table)
            crops = crop_geometry(pictures, span, mod, task)
            if crops is None:
                ledger.extend(
                    {"ordinal": k, "role": role, "status": "unavailable_geometry"} for role in roles
                )
                print("Unavailable source geometry", split, k, flush=True)
                continue
            for role in roles:
                token = hmac.digest(key, f"{k}:{role}".encode(), "sha256").hex()[:24]
                path = run / "deliveries" / f"{token}.mov"
                authored, hashes = author(pictures, table, span, role, crops, model)
                encode(path, authored, wav, table, step)
                ledger.append(
                    {
                        "ordinal": k,
                        "role": role,
                        "status": "authored",
                        "token": token,
                        "file_sha256": sha(path),
                        "time_base": table.time_base,
                        "frames": len(pictures),
                        "edits": hashes,
                    }
                )
                print("Authored", split, k, role, len(hashes), flush=True)
    return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "controls", "neural"))
    p.add_argument("sources", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("development", "held"), default="development")
    a = p.parse_args()
    if a.phase == "freeze":
        lock = freeze(a.sources)
        a.run.mkdir(mode=0o700)
        (a.run / "deliveries").mkdir(mode=0o700)
        (a.run / "blind.key").write_bytes(secrets.token_bytes(32))
        write(a.run / "freeze.json", lock)
        return
    if a.phase == "controls":
        result = construct(a.sources, a.run, a.split, CONTROL_ROLES)
        stats = {}
    else:
        require((a.run / f"controls-{a.split}.json").is_file(), "Complete controls before neural inference")
        if a.split == "held":
            require(
                (a.run / "review-development.json").is_file(),
                "Audit all development deliveries before held inference",
            )
        start = time.monotonic()
        with renderer() as (model, loads, roots):
            result = construct(a.sources, a.run, a.split, (NEURAL,), model)
        n = sum(len(r.get("edits", ())) for r in result)
        require(
            roots
            == {
                "appearance_feature_extractor": n,
                "motion_extractor": 2 * n,
                "warping_module": n,
                "spade_generator": n,
            },
            "Unexpected neural forward counts",
        )
        stats = {
            "model_loads": len(loads),
            "root_forwards": roots,
            "wall_seconds": time.monotonic() - start,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    write(
        a.run / f"{a.phase}-{a.split}.json",
        {"freeze_sha256": sha(a.run / "freeze.json"), "cases": result, **stats},
    )


if __name__ == "__main__":
    main()
