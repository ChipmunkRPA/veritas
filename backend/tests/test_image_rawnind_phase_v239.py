import hashlib
import io
import json
import stat
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import image_rawnind_phase_v239 as study


def authored_metadata():
    entries, groups, sources = [], {}, {}
    for extension in study.EXTENSIONS:
        for scene_number in range(3):
            scene = f"{extension}-scene-{scene_number}"
            clean_images, noisy_images = [], []
            for role, members, iso in (("_GT", clean_images, 100), ("", noisy_images, 200)):
                body = f"fixture:{scene}:{role}".encode()
                sha1 = hashlib.sha1(body, usedforsecurity=False).hexdigest()
                md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
                name = f"Bayer_{scene}{role}_ISO{iso}_sha1={sha1}.{extension}"
                identifier = len(entries) + 1
                entries.append(
                    {
                        "version": 1,
                        "restricted": False,
                        "dataFile": {
                            "filename": name,
                            "id": identifier,
                            "filesize": len(body),
                            "md5": md5,
                            "checksum": {"type": "MD5", "value": md5},
                        },
                    }
                )
                members.append({"filename": name, "sha1": sha1})
                sources[identifier] = body
            groups[scene] = {
                "test_reserve": scene_number == 2,
                "unknown_sensor": False,
                "clean_images": clean_images,
                "noisy_images": noisy_images,
            }
    catalog = {
        "data": {
            "datasetPersistentId": study.DOI,
            "versionState": "RELEASED",
            "versionNumber": 1,
            "versionMinorNumber": 0,
            "id": 177,
            "files": entries,
            "license": {"rightsIdentifier": "CC-BY-SA-4.0", "uri": study.LICENSE_URI},
        }
    }
    return catalog, {"Bayer": groups}, sources


def test_freeze_and_acquire_bind_both_checksums_and_owner_only_roster(tmp_path, monkeypatch):
    catalog, sidecar, sources = authored_metadata()
    monkeypatch.setattr(study, "CATALOG_FILES", len(catalog["data"]["files"]))
    chosen = study.select_roster(catalog, sidecar)
    assert len(chosen) == 4 and all("scene-2" not in row["file"] for row in chosen)
    assert [Path(row["file"]).suffix for row in chosen] == [".arw", ".arw", ".crw", ".crw"]
    catalog_bytes = study.compact(catalog)
    sidecar_bytes = yaml.safe_dump(sidecar).encode()
    wheel = b"fixture wheel bytes"
    for name, body in (
        ("CATALOG", catalog_bytes),
        ("SIDECAR", sidecar_bytes),
        ("ROSTER", study.compact(chosen)),
        ("WHEEL", wheel),
    ):
        monkeypatch.setattr(study, f"{name}_SHA", study.sha256(body))
    monkeypatch.setattr(study, "RAW_BYTES", sum(row["bytes"] for row in chosen))
    monkeypatch.setattr(study, "WHEEL_BYTES", len(wheel))
    payloads = {
        study.CATALOG_URL: catalog_bytes,
        f"{study.HOST}/api/access/datafile/{study.SIDECAR_ID}": sidecar_bytes,
        study.WHEEL_URL: wheel,
    }
    payloads.update({f"{study.HOST}/api/access/datafile/{row['id']}": sources[row["id"]] for row in chosen})
    requested = []

    def fetch(url, size):
        requested.append(url)
        if url not in (study.CATALOG_URL, f"{study.HOST}/api/access/datafile/{study.SIDECAR_ID}"):
            assert len(payloads[url]) == size
        return payloads[url]

    monkeypatch.setattr(study, "fetch_exact", fetch)
    trial = tmp_path / "owned"
    assert study.freeze(trial)["status"] == "frozen"
    assert stat.S_IMODE(trial.stat().st_mode) == 0o700
    assert len(requested) == 2
    assert json.loads((trial / "roster.json").read_bytes()) == chosen
    assert study.acquire(trial)["captures"] == 4
    assert len(requested) == 7
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in trial.iterdir())
    study.acquire(trial)
    assert len(requested) == 7
    original = trial / "capture-00.arw"
    original.write_bytes(b"X" * original.stat().st_size)
    with pytest.raises(ValueError, match="publisher size or checksum"):
        study.acquire(trial)


def test_roster_refuses_mismatched_author_checksum(tmp_path, monkeypatch):
    catalog, sidecar, _ = authored_metadata()
    monkeypatch.setattr(study, "CATALOG_FILES", len(catalog["data"]["files"]))
    sidecar["Bayer"]["arw-scene-0"]["clean_images"][0]["sha1"] = "0" * 40
    with pytest.raises(ValueError, match="publisher provenance"):
        study.select_roster(catalog, sidecar)


def authored_wheel(escape=False):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            f"{study.DIST_INFO}/METADATA",
            "Name: rawpy\nVersion: 0.28.0\nLicense-Expression: MIT\n"
            "License-File: LICENSE\nLicense-File: LICENSE.LibRaw\n",
        )
        archive.writestr(f"{study.DIST_INFO}/WHEEL", "Tag: cp312-cp312-macosx_11_0_arm64\n")
        archive.writestr(f"{study.DIST_INFO}/licenses/LICENSE", "Fixture license")
        archive.writestr(f"{study.DIST_INFO}/licenses/LICENSE.LibRaw", "Fixture attribution")
        archive.writestr("rawpy/__init__.py", "# Synthetic archive test. No import.")
        if escape:
            archive.writestr("../escape", "Reject this member before extraction")
    return data.getvalue()


def test_archive_checks_tags_licenses_and_safe_paths(tmp_path):
    study.unpack_decoder(authored_wheel(), tmp_path)
    assert (tmp_path / study.DIST_INFO / "licenses/LICENSE.LibRaw").is_file()
    with pytest.raises(ValueError, match="Unsafe wheel member"):
        study.unpack_decoder(authored_wheel(escape=True), tmp_path)
    assert not (tmp_path.parent / "escape").exists()


def test_raw_decode_requires_native_bayer_and_disables_automatic_geometry_changes():
    pixels = np.zeros((256, 320, 3), dtype=np.uint8)
    raw = SimpleNamespace(
        raw_pattern=np.array([[0, 1], [3, 2]]),
        color_desc=b"RGBG",
        sizes=SimpleNamespace(height=256, width=320),
    )
    observed = {}

    def postprocess(**settings):
        observed.update(settings)
        return pixels

    raw.postprocess = postprocess
    decoder = SimpleNamespace(
        imread=lambda _: nullcontext(raw),
        DemosaicAlgorithm=SimpleNamespace(LINEAR=0),
        FBDDNoiseReductionMode=SimpleNamespace(Off=0),
        ColorSpace=SimpleNamespace(sRGB=1),
        HighlightMode=SimpleNamespace(Clip=0),
    )
    assert study.decode(decoder, Path("local.arw")) is pixels
    assert observed["demosaic_algorithm"] == observed["user_flip"] == observed["median_filter_passes"] == 0
    assert observed["half_size"] is False and observed["no_auto_bright"] is True
    assert observed["noise_thr"] is None and observed["fbdd_noise_reduction"] == 0
    raw.raw_pattern = np.zeros((6, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="Bayer capture"):
        study.decode(decoder, Path("local.arw"))


def test_frozen_views_and_aggregate_hide_image_commitments():
    pixels = np.full((256, 320, 3), 128, dtype=np.uint8)
    views = dict(study.ordinary_views(pixels))
    assert tuple(views) == study.VIEWS
    assert views["native"] is pixels
    assert views["crop_odd"].shape == (252, 315, 3)
    assert views["resize_075"].shape == (192, 240, 3)
    results = {name: study.measure(pixels) for name, pixels in views.items()}
    rows = [{"capture": n, "view": name, **result} for n in range(4) for name, result in results.items()]
    result = study.summarize(rows)
    assert result["measured_views"] == 24 and result["new_neural_calls"] == 0
    assert result["views"]["native"]["status_counts"] == {"phase_not_supported": 4}
    assert all(key not in study.compact(result) for key in (b"rgb_sha256", b"source_size"))
    with pytest.raises(ValueError, match="Incomplete matched replay"):
        study.summarize(rows[:-1])
