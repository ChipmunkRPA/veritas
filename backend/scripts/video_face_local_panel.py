"""Build paired, labeled partial face swaps and matched ordinary edits."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts import video_common_sources_v89 as io
from scripts import video_face_temporal_delivery as delivery

SCHEMA = "chipmunk.video-face-local-panel/1"
GENERATORS = ("inswapper", "uniface")
LENGTHS = (10, 18, 28, 40)
STARTS = (14, 22, 28, 36, 20, 32)
ROLES = ("original", "gain", "face_blur", "picture_jump", "different_shot", *GENERATORS)


def digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def interval(index, role_index=0):
    length = LENGTHS[(index + 2 * role_index) % len(LENGTHS)]
    start = STARTS[(index + role_index) % len(STARTS)]
    return start, start + length


def source_pictures(group, root, family):
    from app.detectors.video.decode import probe_frames

    path = root / f"{group['index']:02d}-{family}.mp4"
    identity = group["files"][family]["sha256"]
    if io.sha(path) != identity:
        raise ValueError("Incorrect publisher picture bytes")
    table = probe_frames(path, cap=900)
    return (*delivery.sample(path), identity, table.pts[0] * table.tb)


def compatible_clocks(a, b, origin_a, origin_b):
    from fractions import Fraction

    return all(abs((pa * Fraction(ta) - origin_a) - (pb * Fraction(tb) - origin_b)) <= Fraction(1, 24)
               for (_, pa, ta), (_, pb, tb) in zip(a, b, strict=True))


def softened_face(frame, face):
    import cv2
    import numpy as np

    x, y, w, h = (int(value) for value in face.box())
    y0, y1 = max(0, y - h // 4), min(frame.shape[0], y + h + h // 4)
    x0, x1 = max(0, x - w // 4), min(frame.shape[1], x + w + w // 4)
    crop = frame[y0:y1, x0:x1]
    ys, xs = np.ogrid[y0:y1, x0:x1]
    rho = ((xs - x - w / 2) / (.65 * w))**2 + ((ys - y - h / 2) / (.66 * h))**2
    mask = np.clip((1 - rho) * 2.5, 0, 1)[..., None]
    blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(1.4, w / 35))
    result = frame.copy()
    result[y0:y1, x0:x1] = np.rint(crop * (1 - mask) + blur * mask).astype(np.uint8)
    return result


def camera_treatments(frames, alternate, begin, end, locator):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces

    local = frames[begin:end]
    yield "original", []
    yield "gain", [np.rint(np.clip(frame * 1.17 + 5, 0, 255)).astype(np.uint8) for frame in local]
    yield "picture_jump", frames[begin + 4:end + 4]
    yield "different_shot", alternate[begin:end]
    softened = []
    for frame in local:
        faces = detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detector=locator, min_size=48)
        softened.append(softened_face(frame, faces[0]) if faces else frame)
    if any(a is not b for a, b in zip(local, softened, strict=True)):
        yield "face_blur", softened


def prepare(roster_path, root, output, *, picks, purpose):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet

    cv2.setNumThreads(1)
    locator = load_yunet("yunet.2023mar", "cpu", "float32").obj
    roster = io.read(roster_path)
    selected = [row for row in roster["groups"] if row["index"] in picks]
    if len(selected) != len(picks) or len(selected) < 2:
        raise ValueError("The frozen group selection differs from the publisher receipt")
    output.mkdir(mode=0o700)
    rows, omissions = [], []
    for position, group in enumerate(selected):
        real, clock, base, origin = source_pictures(group, root, "ordinary")
        opposite = selected[(position + 1) % len(selected)]
        alternate, alternate_clock, alternate_sha, _ = source_pictures(opposite, root, "ordinary")
        begin, end = interval(position)
        cases = []
        for role, changed in camera_treatments(real, alternate, begin, end, locator):
            source, mapping = (alternate_sha, alternate_clock) if role == "different_shot" else (base, clock)
            cases.append((role, changed, None, [begin, end] if changed else [], source, mapping))
        for order, family in enumerate(GENERATORS):
            derived, match_clock, source, treated_origin = source_pictures(group, root, family)
            if not compatible_clocks(clock, match_clock, origin, treated_origin):
                omissions.append({"group": group["group"], "family": family, "reason": "incompatible_source_clocks"})
                continue
            a, b = interval(position, order)
            cases.append((family, derived[a:b], [a, b], [a, b], source, match_clock))
        for role, changed, truth, bounds, source, mapping in cases:
            untouched = real[:]
            if bounds:
                a, b = bounds
                if len(changed) != b - a:
                    raise ValueError("Treatment has a different frame count")
                untouched[a:b] = changed
            direct = output / f"{position:02d}-{role}-direct.mp4"
            tandem = output / f"{position:02d}-{role}-tandem.mp4"
            delivery.encode(untouched, direct)
            delivery.tandem(direct, tandem)
            for profile, clip in (("direct", direct), ("tandem", tandem)):
                rows.append({"file": clip.name, "group": group["group"], "family": role, "delivery": profile,
                             "source_sha256": base, "treated_sha256": source, "input_sha256": io.sha(clip),
                             "truth": truth, "edit_frames": bounds, "source_index": group["index"],
                             "geometry": {"width": delivery.WIDTH, "height": delivery.HEIGHT, "count": delivery.FRAME_COUNT}})
            if truth:
                a, b = truth
                if any(digest(untouched[i]) != digest(real[i]) for i in (0, a - 1, b, len(real) - 1)):
                    raise ValueError("Staged swap modified a flank")
                for row in rows[-2:]:
                    row["anchor_source_clocks"] = {str(i): (mapping[i] if a <= i < b else clock[i])
                                                   for i in (a - 1, (a + b) // 2, b)}
        print(f"Prepared {purpose} source {position + 1}/{len(selected)}", flush=True)
    io.write(output / "manifest.json", {"schema": SCHEMA, "roster_sha256": io.sha(roster_path), "picks": picks,
                                       "purpose": purpose, "rows": rows, "omissions": omissions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("roster", "sources", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--purpose", required=True, choices=("fit", "selection", "held"))
    parser.add_argument("--picks", type=int, nargs="+", required=True)
    args = parser.parse_args()
    prepare(args.roster, args.sources, args.output, picks=args.picks, purpose=args.purpose)


if __name__ == "__main__":
    main()
