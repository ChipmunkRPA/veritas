# SPDX-License-Identifier: Apache-2.0
"""Render ordinary local fills on the same source areas as the generated edits."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import image_region_rank_inputs as files
from scripts.image_jpeg_band_sources import small_subject_area


def render(intake, output, *, partial):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts.image_bg_real_acquisition_v221 import fingerprint, png_pixels

    output.mkdir(mode=0o700)
    cv2.setNumThreads(1)
    roster = files.read(intake / "intake.json")
    groups, cases = [], []
    for i, record in enumerate(roster["cases"]):
        source, name = record["source"], record["tag"]
        image_path, mask_path = intake / f"{name}.png", intake / f"{name}-subject.png"
        data = image_path.read_bytes()
        photo = png_pixels(data, source["size"])
        if (
            fingerprint(data, photo) != record["fingerprints"]["original"]
            or files.sha(mask_path) != record["subject_sha256"]
        ):
            raise ValueError("Pinned photo or subject differs from its intake")
        subject = png_pixels(mask_path.read_bytes(), source["size"])[..., 0] > 127
        if partial:
            subject = small_subject_area(subject, source["original"]["sha256"])
        support = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        group = f"ordinary-fill-{i:02d}"
        folder = output / group
        folder.mkdir()
        groups.append(
            {
                "tag": group,
                "source": source["original"]["sha256"],
                "family": "classical_fill",
                "split": "controls",
                "shape": list(support.shape),
                "rgb_sha256": record["fingerprints"]["original"]["rgb_sha256"],
                "first_changed_pixels": 0,
            }
        )
        empty = folder / "negative-mask.png"
        Image.fromarray(np.zeros_like(support)).save(empty)
        for role, algorithm in (("telea_fill", cv2.INPAINT_TELEA), ("navier_stokes_fill", cv2.INPAINT_NS)):
            processed = cv2.inpaint(photo, support, 4, algorithm)
            if np.any((processed != photo).any(axis=-1) & (support == 0)):
                raise ValueError("The matched ordinary edit changed the area outside its known support")
            for quality in files.QUALITY:
                media = folder / f"{role}-{quality}.jpg"
                Image.fromarray(processed).save(media, "JPEG", quality=quality, subsampling=2)
                cases.append(
                    {
                        "group": group,
                        "role": role,
                        "quality": quality,
                        "file": media.relative_to(output).as_posix(),
                        "sha256": files.sha(media),
                        "mask": empty.relative_to(output).as_posix(),
                        "mask_sha256": files.sha(empty),
                        "changed_pixels": 0,
                    }
                )
    files.save(
        output / "manifest.json",
        {
            "schema": files.SCHEMA,
            "groups": groups,
            "images": cases,
            "source_manifests": {"intake": files.sha(intake / "intake.json")},
            "partial_subject": partial,
        },
    )
    print(f"Rendered {len(cases)} ordinary fills from {len(groups)} exact source photos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true")
    render(**vars(parser.parse_args()))
