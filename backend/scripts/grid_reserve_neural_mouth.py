# SPDX-License-Identifier: Apache-2.0
"""Author and verify a small, source-qualified GRID neural mouth-redraw panel."""

from __future__ import annotations

import argparse
import importlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import build_grid_word_timed_controls as delivery
from scripts import grid_mouth_observer_v157 as observer
from scripts import grid_portrait_frames_v146 as historical
from scripts import grid_portrait_geometry_v144 as geometry
from scripts import grid_reserve_word_clocks as clocks
from scripts import portrait_geometry_runtime_v132 as runtime
from scripts.video_common_sources_v89 import read, require, sha, write

ASSETS = geometry.ASSETS
MODES = ("source_time_neural", "earlier_drive_neural", "earlier_pixels_ordinary")
DRIVER_SHIFT = 8
SPEC = {
    "schema": "chipmunk.grid-reserve-neural-mouth/1",
    "source_selection": "First eligible unscored literal-label and remapped-label reserve originals. Select on original "
                        "clock, publisher word and visible mouth closure only. Never substitute after rendering begins.",
    "bilabial_word_initials": ["b", "m", "p"],
    "original_closure": "Use eye-normalized median inner-lip gap. For each eligible word choose its closest observable "
                        "low aperture near the published onset with measured mouth reopening.",
    "maximum_closed_gap": .055, "minimum_reopening": .032,
    "closure_search_native_pictures": [-3, 3], "reopen_search_native_pictures": [1, 5],
    "authored_window_around_original_closure": [-3, 13],
    "driver_shift_pictures": DRIVER_SHIFT, "renderer": "Retained MIT LivePortrait human-only expression renderer, "
                 "CPU float32, 512 output, original bounded convolution arithmetic.",
    "comparison": "Independent post-encode stateless mesh observes original and finished clips. The same geometry asset "
                  "also registered the renderer inputs. Record closure support separately from authored provenance.",
}


def mp_task():
    sys.path.insert(0, str(runtime.SITE))
    module = importlib.import_module("mediapipe")
    require(module.__version__ == runtime.PINS["mediapipe"][0], "Retained MediaPipe runtime differs")
    frozen = read(geometry.PUBLIC)
    for name, field in (("face_landmarker.task", "face_task_sha256"),
                        ("liveportrait-verified-assets.json", "asset_catalogue_sha256"),
                        ("liveportrait-source-LICENSE.txt", "source_license_sha256")):
        require(sha(ASSETS / "permissions" / name) == frozen[field], "Retained geometry or renderer licence differs")
    return module, module.tasks.vision.FaceLandmarker.create_from_options(observer.observer_options(module))


def observed(module, task, pixels):
    result = task.detect(module.Image(image_format=module.ImageFormat.SRGB, data=np.ascontiguousarray(pixels)))
    if len(result.face_landmarks) != 1:
        return None, None
    mesh = result.face_landmarks[0]
    height, width = pixels.shape[:2]
    try:
        descriptor = observer.describe([(mesh[i].x*width, mesh[i].y*height) for i in observer.POINTS], pixels.shape)
    except ValueError:
        return None, None
    return mesh, [float.fromhex(x) for x in descriptor["descriptor_hex"]]


def closure(series, anchor, radius=3):
    positions = range(max(0, anchor-radius), min(anchor+radius+1, len(series)))
    candidates = []
    for k in positions:
        point = series[k]
        ahead = [x for x in series[k+1:k+6] if x is not None]
        if point is None or len(ahead) < 3:
            continue
        value = float(np.median(point[:3]))
        rise = max(float(np.median(x[:3])) for x in ahead) - value
        if value <= SPEC["maximum_closed_gap"] and rise >= SPEC["minimum_reopening"]:
            candidates.append((abs(k-anchor), -rise, value, k))
    return None if not candidates else min(candidates)[3]


def pins(qualified):
    paths = [Path(__file__), Path(delivery.__file__), Path(clocks.__file__), Path(observer.__file__),
             Path(geometry.__file__), Path(historical.__file__)]
    paths.extend((geometry.ROOT/"backend/scripts"/name) for name in (
        "portrait_expression_drive_v138.py", "spatial_conv_install_v141.py", "liveportrait_mouth_benchmark.py",
    ))
    proof = qualified / "private-word-clocks.json"
    require(sha(proof) == read(qualified/"qualification.json")["private_word_clocks_sha256"], "Qualified reserve clock bytes differ")
    return {"method": SPEC, "qualification_sha256": sha(proof),
            "source_code_sha256": {str(path.relative_to(geometry.ROOT)): sha(path) for path in paths}}


def preflight(qualified, out):
    frozen = pins(qualified)
    sources, _ = clocks.plan()
    timings = read(qualified/"private-word-clocks.json")["records"]
    require(len(timings)==16, "All frozen reserve roles required")
    out.mkdir(mode=0o700, parents=True)
    write(out/"protocol.json", frozen)
    measurements, selected = [], {}
    module, task = mp_task()
    try:
        for timing in timings:
            literal = timing["literal_label"]
            if literal in selected:
                continue
            ordinal = timing["ordinal"]
            source = sources[ordinal]
            table, pixels, _, _ = clocks.exact_native(source["input"])
            series = [observed(module, task, picture)[1] for picture in pixels]
            require(len(series)==75, "Require native original observation schedule")
            measurements.append({"ordinal": ordinal, "aperture": series})
            for word in timing["words"]:
                if word["word"][0] not in SPEC["bilabial_word_initials"]:
                    continue
                anchor = word["source_native_picture_indices"][0]
                event = closure(series, anchor)
                if event is None or event < 11 or event+13 >= len(table):
                    continue
                span = [event-3, event+13]
                if any(series[i] is None for i in range(span[0]-DRIVER_SHIFT, span[1])):
                    continue
                selected[literal] = {"ordinal": ordinal, "group": source["component_index"], "original_closure": event,
                    "published_word_source_samples": word["source_pcm_samples"], "published_word_source_seconds": word["source_seconds"],
                    "source_word_picture": anchor, "span": span, "source_sha256": timing["source_sha256"]}
                print("selected original", ordinal, "closure", event, "span", span, flush=True)
                break
    finally:
        task.close()
    write(out/"original-observations.json", measurements)
    require(len(selected)==2 and len({r["group"] for r in selected.values()})==2, "Two qualified fresh original closure groups unavailable")
    write(out/"freeze.json", {**frozen, "sources": list(selected.values()),
                            "original_observations_sha256": sha(out/"original-observations.json")})


def check(qualified, out):
    freeze = read(out/"freeze.json")
    require(all(freeze[k] == v for k,v in pins(qualified).items())
            and freeze["original_observations_sha256"] == sha(out/"original-observations.json"),
            "Frozen source, method or geometry changed")
    return freeze


def render(qualified, out):
    frozen = check(qualified, out)
    import torch

    from scripts import spatial_conv_install_v141 as tiling
    from scripts.liveportrait_mouth_benchmark import crop_from_mesh
    from scripts.portrait_expression_drive_v138 import ExpressionRenderer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(146)
    cv2.setNumThreads(1)
    output = out/"deliveries"
    output.mkdir(mode=0o700)
    sources, _ = clocks.plan()
    started = time.monotonic()
    loads, cases, roots = [], [], {}
    original_load = torch.nn.Module.load_state_dict

    def exact_load(model, state, strict=True, assign=False):
        require(strict and not assign, "Require strict exact checkpoint loads")
        receipt = original_load(model, state, strict=strict, assign=assign)
        actual = model.state_dict()
        require(actual.keys() == state.keys() and all(torch.equal(actual[k].cpu(), state[k].cpu()) for k in state),
                "Loaded tensor differs from its checkpoint")
        loads.append(len(state))
        return receipt

    torch.nn.Module.load_state_dict = exact_load
    try:
        renderer = ExpressionRenderer(ASSETS)
    finally:
        torch.nn.Module.load_state_dict = original_load
    require(len(loads) == 5, "Five strict tensor-exact licensed loads required")
    patches = tiling.install(renderer.renderer.models, lambda _: None)
    handles = []

    def count(name):
        def hook(*args):
            roots[name] = roots.get(name, 0)+1
        return hook

    for name, model in renderer.renderer.models.items():
        handles.append(model.register_forward_hook(count(name)))
    module, task = mp_task()
    try:
        with (out/"render.jsonl").open("x") as journal:
            for item in frozen["sources"]:
                ordinal = item["ordinal"]
                source = sources[ordinal]["input"]
                table, native, pcm, origin = clocks.exact_native(source)
                first, last = item["span"]
                crops = {}
                for index in range(first-DRIVER_SHIFT,last):
                    mesh, _ = observed(module, task, native[index])
                    require(mesh is not None, "A fixed native face geometry became unavailable")
                    crop, matrix, _ = crop_from_mesh(native[index], mesh)
                    values = np.asarray([(p.x,p.y,p.z) for p in mesh], dtype=np.float64)
                    crops[index] = crop, geometry.upscale_matrix(matrix), historical.alpha_from_mesh(values,native[index].shape)
                for mode in MODES:
                    authored = list(native)
                    for index in range(first,last):
                        crop, matrix, alpha = crops[index]
                        driver = index if mode == "source_time_neural" else index-DRIVER_SHIFT
                        require((table.pts[index]-table.pts[driver])*table.tb*25 == index-driver, "Driver source clock differs")
                        drive = crops[driver][0]
                        if mode == "earlier_pixels_ordinary":
                            generated = cv2.resize(drive, (512,512), interpolation=cv2.INTER_LINEAR)
                        else:
                            generated = renderer.render(crop, drive)
                            historical.generated_contract(generated)
                        warped = cv2.warpAffine(generated, cv2.invertAffineTransform(matrix), (table.width,table.height),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                        authored[index], mask = historical.composite(native[index], warped, alpha, .5 if index in (first,last-1) else 1.)
                        require(mask.any(), "A fixed redraw changed no native pixels")
                        journal.write(json.dumps({"ordinal": ordinal,"mode": mode,"frame_index": index,"driver_frame_index":driver,
                                                  "native_rgb_sha256": historical.fingerprint(authored[index]),
                                                  "changed_mask_sha256": historical.fingerprint(mask)})+"\n")
                        journal.flush()
                    dest = output/f"{ordinal:02}-{mode}.mkv"
                    delivery.encode(dest, authored, Path(source["wav_file"]))
                    proof = delivery.verify(dest, Path(source["source_file"]), native, table, pcm, origin, item["span"], authored)
                    cases.append({"ordinal": ordinal,"mode":mode,"file":dest.name,"proof":proof})
                    print("completed", ordinal, mode, flush=True)
    finally:
        task.close()
        for handle in handles:
            handle.remove()
        tiling.restore(patches)
    n = sum(row["proof"]["edited_frame_count"] for row in cases if row["mode"]!="earlier_pixels_ordinary")
    require(roots == {"appearance_feature_extractor":n,"motion_extractor":2*n,"warping_module":n,"spade_generator":n},
            "Observed renderer roots differ from completed frame schedule")
    write(out/"construction.json", {"freeze_sha256":sha(out/"freeze.json"),"model_loads":len(loads),"root_forwards":roots,
        "cases":cases,"runtime_s":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "render_journal_sha256":sha(out/"render.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight","render"))
    parser.add_argument("qualification",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    (preflight if args.phase=="preflight" else render)(args.qualification,args.output)


if __name__ == "__main__":
    main()
