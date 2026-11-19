from hashlib import sha256
from io import BytesIO
from shutil import which
from struct import pack_into
from subprocess import DEVNULL, run
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image.faces import Face
from app.detectors.video import face_return, face_single_boundary
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.face_detail_inputs import align
from app.detectors.video.frames import probe
from app.engine.detector import Profile
from app.engine.stages import video
from app.evidence import face_single_boundary_binding as binding
from app.evidence import picture_hold_binding, video_packets
from app.evidence.mp4 import bounded_boxes
from app.reports import verify_video_source

pytestmark = pytest.mark.skipif(not which("ffmpeg") or not which("ffprobe"), reason="FFmpeg tools required")
ALT = binding.NATIVE2997_CARRIER_METHOD
LEGACY = binding.NATIVE2997_METHOD


def mux(*args):
    run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", *map(str, args)],
        check=True,
        stdin=DEVNULL,
        capture_output=True,
        timeout=30,
    )


def read_clock(source):
    table = probe_frames(source)
    info = probe(source, include_codec_tag=True)
    announced = info["codec"], info["pix_fmt"], info["codec_tag"]
    method = face_single_boundary.native_ntsc_method(source, table, announced, info["timeline_origin_s"])
    return table, method


def change_edit(source, target, *, extend=False):
    data = bytearray(source.read_bytes())
    size, _, boxes, one, _ = bounded_boxes(BytesIO(data))

    def children(parent):
        return boxes(parent.start, parent.end)

    movie = children(one(boxes(0, size), b"moov"))
    track = children(one(movie, b"trak"))
    edit = one(children(one(track, b"edts")), b"elst")
    assert data[edit.start : edit.start + 8] == bytes(7) + b"\1"
    if extend:
        for pos in (one(movie, b"mvhd").start + 16, one(track, b"tkhd").start + 20, edit.start + 8):
            assert int.from_bytes(data[pos : pos + 4], "big") == 360360
            pack_into(">I", data, pos, 363363)
    else:
        pack_into(">h", data, edit.start + 16, 0)
    target.write_bytes(data)


@pytest.fixture(scope="module")
def movies(tmp_path_factory):
    root = tmp_path_factory.mktemp("ntsc_source_clock")
    source = root / "legacy.mp4"
    mux(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30000/1001",
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
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        30000,
        "-movie_timescale",
        90000,
        "-use_editlist",
        1,
        source,
    )
    files = {"legacy": source}

    def remux(name, timescale, *options):
        target = root / f"{name}.mp4"
        mux(
            "-i",
            source,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-video_track_timescale",
            timescale,
            "-movie_timescale",
            90000,
            *options,
            target,
        )
        files[name] = target

    for name in (60000, 90000, 120000, 180000):
        remux(str(name), name, "-use_editlist", 1)
    remux("signed", 90000, "-use_editlist", 0, "-movflags", "+negative_cts_offsets")
    remux("unrebased", 90000, "-use_editlist", 0)
    remux("rounded", 1000000, "-use_editlist", 1)
    remux("jitter", 90000, "-use_editlist", 1, "-bsf:v", r"setts=pts=PTS+eq(N\,61)")
    for name, parent in (("dwell", "90000"), ("extended", "90000"), ("legacy_dwell", "legacy")):
        target = root / f"{name}.mp4"
        change_edit(files[parent], target, extend=name == "extended")
        files[name] = target
    target = root / "truncated.mp4"
    target.write_bytes(files["90000"].read_bytes()[:32])
    files["truncated"] = target
    return files


@pytest.mark.parametrize(
    "name, ticks", [("60000", 2002), ("90000", 3003), ("120000", 4004), ("180000", 6006), ("signed", 3003)]
)
def test_alternate_clock_preserves_native_files_and_relative_times(movies, name, ticks):
    old, old_method = read_clock(movies["legacy"])
    new, new_method = read_clock(movies[name])
    assert old_method == LEGACY and new_method == ALT
    assert new.pts == tuple(i * ticks for i in range(120))
    assert tuple(p * old.tb for p in old.pts) == tuple(p * new.tb for p in new.pts)


@pytest.mark.parametrize("name", ["unrebased", "rounded", "jitter", "dwell", "extended", "legacy_dwell"])
def test_a_nominally_similar_clock_cannot_pass_the_native_source_check(movies, name):
    table, method = read_clock(movies[name])
    assert method is None
    if name in {"dwell", "extended", "legacy_dwell"}:
        old, _ = read_clock(movies["legacy" if name == "legacy_dwell" else "90000"])
        assert table.to_json() == old.to_json()


def test_stage_admits_the_new_profile_only_when_enabled_and_always_checks_metadata(movies, monkeypatch):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "video_face_single_boundary_review", True)
    monkeypatch.setattr(settings, "video_face_single_boundary_native30_review", False)
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", False)
    monkeypatch.setattr(video.provenance_gate, "stop", lambda _: False)

    def review(path, profile=Profile.FULL):
        state = {
            "frame_store": SimpleNamespace(path=path, table=probe_frames(path)),
            "registry": SimpleNamespace(manifest=SimpleNamespace(model=lambda _: None)),
        }
        ctx = SimpleNamespace(
            profile=profile, outputs={"probe": {"info": probe(path, include_codec_tag=True)}}, state=state
        )
        return video._single_boundary_review(ctx)[binding.COVERAGE]

    assert review(movies["90000"])["reason"] == "profile"
    monkeypatch.setattr(settings, "video_face_single_boundary_native2997_review", True)
    assert review(movies["90000"]) == {
        "status": "unavailable",
        "reason": "models_unavailable",
        "method": ALT,
        "frames_scanned": 0,
        "measured_face_frames": 0,
    }
    assert review(movies["90000"], Profile.DEEP)["reason"] == "profile"
    assert review(movies["legacy"])["method"] == LEGACY
    for name in ("dwell", "legacy_dwell"):
        assert review(movies[name])["reason"] == "profile"

    table, _ = read_clock(movies["90000"])
    assert (
        face_single_boundary.native_ntsc_method(movies["truncated"], table, ("h264", "yuv420p", "avc1"), 0)
        is None
    )


def data_for(source, table, coverage, packet=None):
    source_info = {
        "sha256": sha256(source.read_bytes()).hexdigest(),
        "stream_selector": "v:0",
        "coordinate_space": "display-oriented-native-frame",
        "time_base": table.time_base,
        "width": table.width,
        "height": table.height,
        "frame_count": len(table),
        "timeline_origin_s": 0,
        "frame_table_sha256": canonical_hash(table.to_json()),
        "picture_hold_sha256": None,
        binding.COVERAGE_COMMITMENT: canonical_hash(coverage),
    }
    if packet is not None:
        source_info[binding.COMMITMENT] = canonical_hash(packet[binding.KEY])
    bundle = {
        "schema": verify_video_source.SCHEMA,
        "analysis": {"kind": "video"},
        "source_commitment": {"sha256": source_info["sha256"]},
    }
    document = {
        "version": video_packets.VERSION,
        "source": source_info,
        "packets": [packet] if packet else [],
        "coverage": {binding.COVERAGE: coverage},
        "picture_hold": picture_hold_binding.coverage(
            "unavailable", picture_hold_binding.HISTORICAL, table.time_base
        ),
    }
    return bundle, document


def verify_original(path, bundle, document, assets=None):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as output:
        output.writestr("evidence.json", canonical_json(document))
        for name, data in (assets or {}).items():
            output.writestr(name, data)
    with ZipFile(buffer) as archive:
        return verify_video_source.check(archive, bundle, path)


def test_original_replay_rechecks_explicit_edit_for_new_method_and_preserves_old_rules(movies):
    for name, method in (("90000", ALT), ("dwell", ALT), ("extended", ALT), ("legacy_dwell", LEGACY)):
        source = movies[name]
        table = probe_frames(source)
        coverage = {
            "method": method,
            "status": "no_match",
            "reason": None,
            "frames_scanned": 120,
            "measured_face_frames": 55,
        }
        bundle, evidence = data_for(source, table, coverage)
        binding.validate(evidence, canonical_hash, table_pts=table.pts)
        if name in {"dwell", "extended"}:
            with pytest.raises(ValueError, match="original MP4 presentation clock"):
                verify_original(source, bundle, evidence)
        else:
            assert "source frame clock matched" in verify_original(source, bundle, evidence)


def test_exact_alternate_boundary_and_crops_replay_from_canonical_packet(movies, monkeypatch):
    path = movies["90000"]
    table, method = read_clock(path)
    assert method == ALT
    indices = [36, 37, 49]
    frames = {frame.index: frame for frame in decode_frames(path, table, indices)}
    box = {"x": 100, "y": 86, "w": 140, "h": 140}
    face = Face(*box.values(), 0.0)
    eyes = [(141.25, 127.15), (205.65555, 128.005)]
    before, after = (
        face_return.Sample(face, None, face_return.scene(frames[i].rgb, face)) for i in indices[:2]
    )
    similarity = face_return.scene_lower_quartile(before, after)
    assert similarity >= binding.MIN_SCENE
    pins = {}
    for role, index in zip(binding.ROLES, indices, strict=True):
        frame = frames[index]
        assert frame.pts_verified and frame.pts == 3003 * index
        pins[role] = {
            "index": index,
            "pts": frame.pts,
            "box": box,
            "eyes": eyes,
            "decoded_rgb_sha256": sha256(frame.rgb).hexdigest(),
            "aligned_rgb_sha256": sha256(align(frame.rgb, eyes)).hexdigest(),
        }
    raw = {
        "coarse": 38,
        "boundary": 37,
        "pts": table.pts[37],
        "time_base": table.time_base,
        "scene": similarity,
        "persistent": 0.6,
        "edge_supported": 0.5,
        "jump_excess": 0.5,
        "fine_margin": 0.004,
        "fine_scene": similarity,
        "source_frames": pins,
    }
    mark = face_single_boundary.coverage(
        "localized", frames_scanned=120, measured_face_frames=60, native30=True
    )
    monkeypatch.setattr(face_single_boundary, "measure", lambda *args, **kw: (raw, mark))
    record, coverage = face_single_boundary.inspect(
        path, table, None, None, check=lambda: None, timeout_s=lambda: 10.0, native_method=method
    )
    assert record["pts"] == 111111 and record["time_s"] == table.seconds(37)
    assert record["affects_overall_confidence"] is False

    anchors, rendered = video._render_single_boundary(
        {binding.KEY: record, binding.COVERAGE: coverage}, [], table
    )
    assets = {}

    def save_image(*args):
        out = BytesIO()
        args[-1].save(out, format="PNG")
        content = out.getvalue()
        digest = sha256(content).hexdigest()
        assets[f"assets/{digest}.png"] = content
        return {"sha256": digest}

    monkeypatch.setattr(video_packets, "_packet_image", save_image)
    _, skeleton = data_for(path, table, coverage)
    packet = video_packets._single_face_boundary_packet(
        None,
        None,
        None,
        None,
        skeleton["source"],
        rendered[binding.KEY],
        {row["frame_index"]: row for row in anchors},
        {index: frames[index].rgb for index in frames},
    )
    bundle, document = data_for(path, table, coverage, packet)
    report = verify_original(path, bundle, document, assets)

    assert packet["semantics"] == packet["derivation"]["selection"] == ALT
    assert [i["anchor"]["pts"] for i in packet["frames"]] == [i * 3003 for i in indices]
    assert "native frames (120), original images or crops (6)" in report
    forged = record | {"pts": record["pts"] + 1}
    with pytest.raises(ValueError, match="frozen measurement range"):
        binding.validate_record(forged, document["source"], table_pts=table.pts)
