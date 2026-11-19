from hashlib import sha256
from io import BytesIO
from shutil import which
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.core.config import settings
from app.detectors.image.faces import Face
from app.detectors.video import face_return, face_single_boundary
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.face_detail_inputs import align
from app.detectors.video.frames import probe
from app.engine.detector import Profile
from app.engine.stages import video
from app.evidence import face_single_boundary_binding as binding
from app.evidence import video_packets
from tests.test_video_face_ntsc_carrier import data_for, mux, verify_original

pytestmark = pytest.mark.skipif(not which("ffmpeg") or not which("ffprobe"), reason="FFmpeg tools required")


@pytest.fixture(scope="module")
def hd_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("native_face_720") / "native-720.mp4"
    mux(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
        "-frames:v",
        120,
        "-c:v",
        "libx264",
        "-threads:v",
        1,
        "-preset",
        "ultrafast",
        "-bf",
        2,
        "-crf",
        32,
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        30000,
        "-use_editlist",
        1,
        path,
    )
    return path


def test_720p_face_review_is_an_independent_full_opt_in(hd_source, monkeypatch):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native720_review", False)
    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: False)
    table = probe_frames(hd_source)
    info = probe(hd_source, include_codec_tag=True)
    encoding = info["codec"], info["pix_fmt"], info["codec_tag"]
    assert face_single_boundary.supported_native720(hd_source, table, encoding, 0)
    assert not face_single_boundary.supported_native720(hd_source, table, encoding, 1)
    assert not face_single_boundary.supported_native30(hd_source, table, encoding)
    ctx = SimpleNamespace(
        profile=Profile.FULL,
        outputs={"probe": {"info": info}},
        state={
            "frame_store": SimpleNamespace(path=hd_source, table=table),
            "registry": SimpleNamespace(manifest=SimpleNamespace(model=lambda _: None)),
        },
    )
    assert video._single_boundary_review(ctx)[binding.COVERAGE]["reason"] == "profile"
    monkeypatch.setattr(settings, "video_face_single_boundary_native720_review", True)
    eligible = video._single_boundary_review(ctx)[binding.COVERAGE]
    assert eligible["method"] == binding.NATIVE720_METHOD and eligible["reason"] == "models_unavailable"

    ctx.profile = Profile.DEEP
    assert video._single_boundary_review(ctx)[binding.COVERAGE]["reason"] == "profile"


def test_720p_evidence_preserves_both_pixel_planes_and_native_bounds(hd_source, monkeypatch):
    indices = [36, 37, 49]
    table = probe_frames(hd_source)
    source = {frame.index: frame.rgb for frame in decode_frames(hd_source, table, indices)}
    projected = {
        i: ((pixels.reshape(360, 2, 640, 2, 3).sum(axis=(1, 3), dtype=np.uint16) + 2) // 4).astype(np.uint8)
        for i, pixels in source.items()
    }
    lens = {"x": 100, "y": 86, "w": 140, "h": 140}
    eyes = ((141.25, 127.15), (205.65555, 128.005))
    face = Face(*lens.values(), 0.0)
    sample = [face_return.Sample(face, None, face_return.scene(projected[i], face)) for i in indices[:2]]
    similarity = face_return.scene_lower_quartile(*sample)
    assert similarity >= binding.MIN_SCENE
    raw = {
        "coarse": 38,
        "boundary": 37,
        "pts": table.pts[37],
        "time_base": table.time_base,
        "scene": similarity,
        "persistent": 0.60,
        "edge_supported": 0.5,
        "jump_excess": 0.5,
        "fine_margin": 0.004,
        "fine_scene": similarity,
    }
    monkeypatch.setattr(
        face_single_boundary.face_local_detail,
        "observe",
        lambda pixels, *_args, **_kwargs: (SimpleNamespace(vector=True), None, None, False),
    )
    monkeypatch.setattr(face_single_boundary, "proposals", lambda *_args, **_kwargs: [raw])

    def refine(*_args, working_frames, **_kwargs):
        for i in indices:
            assert np.array_equal(working_frames[i], projected[i])
        return raw | {
            "source_frames": {
                role: {
                    "box": lens,
                    "eyes": eyes,
                    "index": i,
                    "pts": table.pts[i],
                    "decoded_rgb_sha256": sha256(projected[i]).hexdigest(),
                    "aligned_rgb_sha256": sha256(align(projected[i], eyes)).hexdigest(),
                }
                for role, i in zip(binding.ROLES, indices, strict=True)
            }
        }

    monkeypatch.setattr(face_single_boundary, "_refine", refine)
    record, coverage = face_single_boundary.inspect(
        hd_source,
        table,
        None,
        None,
        check=lambda: None,
        timeout_s=lambda: 12.0,
        native_method=binding.NATIVE720_METHOD,
    )
    assert record["measurement_raster"] == binding.MEASUREMENT_RASTER
    assert record["affects_overall_confidence"] is False and record["pts"] == 37000
    for role, i in zip(binding.ROLES, indices, strict=True):
        row = record["source_frames"][role]
        assert row["box"] == {k: v * 2 for k, v in lens.items()}
        assert row["decoded_rgb_sha256"] == sha256(source[i]).hexdigest()
        assert row["working_rgb_sha256"] == sha256(projected[i]).hexdigest()

    anchors, rendered = video._render_single_boundary(
        {binding.KEY: record, binding.COVERAGE: coverage}, [], table
    )
    assets = {}

    def store_image(*args):
        image = BytesIO()
        args[-1].save(image, format="PNG")
        data = image.getvalue()
        checksum = sha256(data).hexdigest()
        assets[f"assets/{checksum}.png"] = data
        return {"sha256": checksum}

    monkeypatch.setattr(video_packets, "_packet_image", store_image)
    _, initial = data_for(hd_source, table, coverage)
    packet = video_packets._single_face_boundary_packet(
        None,
        None,
        None,
        None,
        initial["source"],
        rendered[binding.KEY],
        {row["frame_index"]: row for row in anchors},
        source,
    )
    bundle, evidence = data_for(hd_source, table, coverage, packet)
    assert "native frames (120), original images or crops (6)" in verify_original(
        hd_source, bundle, evidence, assets
    )

    before = record["source_frames"]["before"]
    before["working_rgb_sha256"] = "a" * 64
    evidence["source"][binding.COMMITMENT] = canonical_hash(record)
    with pytest.raises(ValueError, match="face transition working picture"):
        verify_original(hd_source, bundle, evidence, assets)

    before["box"] = before["box"] | {"x": before["box"]["x"] + 1}
    with pytest.raises(ValueError, match="native face crop differs"):
        binding.validate_record(record, evidence["source"], table_pts=table.pts)
