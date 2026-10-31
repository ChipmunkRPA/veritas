"""Perturbation suite (ML §2.7): deterministic transforms of benchmark items, applied to both classes.

- Every transform is a pure function of (native bytes, item_id). Randomness comes from a generator seeded by
  sha256("<item_id>:<transform>"), so a re-run reproduces the same bytes.
- Image transforms decode the native bytes to RGB 8-bit (as the norm variant does) and re-encode. `exif_forge` alone
  works on the native bytes and leaves the pixels untouched.
- Video and audio transforms run ffmpeg with argument lists and bit-exact flags; `noise_snr20` adds seeded white noise
  in numpy.
- `robustness_subset` picks the robustness subset from the manifest: 750 real images (R1–R5) and 750 fake images (by
  family), 300 videos and 200 audio clips, split evenly across strata and capped by availability.
- `run_perturb` writes `perturbed/<transform>/<item_id>.<ext>` atomically, appends to `perturbed/state.jsonl` and skips
  (item, transform) pairs already done; deterministic transform errors are recorded and not retried.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import wave
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.bench import media

IMAGE_TRANSFORMS = ("jpeg95", "jpeg75", "jpeg50", "webp80", "resize0.5", "crop70", "screenshot", "screenshot_jpeg",
                    "social_ig", "social_x", "launder", "noise3", "upscale2x", "rephoto_proxy", "exif_forge")
VIDEO_TRANSFORMS = ("h264_crf23", "h264_crf35", "social_video", "fps15", "crop_center70")
AUDIO_TRANSFORMS = ("mp3_64k", "opus_24k", "pstn8k", "noise_snr20")
ROBUSTNESS_TARGETS = {"image": (750, 750), "video": (150, 150), "audio": (100, 100)}  # (real, fake)
IMAGE_REAL_STRATA = ("R1", "R2", "R3", "R4", "R5")

SCREEN_W, SCREEN_H, STATUS_BAR, HOME_BAR = 1170, 2532, 47, 34
_BITEXACT = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact"]
_X264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
_VIDEO_ARGS = {
    "h264_crf23": ["-crf", "23"],
    "h264_crf35": ["-crf", "35"],
    "social_video": ["-vf", "scale=-2:720", "-r", "30", "-crf", "28", "-maxrate", "2M", "-bufsize", "4M"],
    "fps15": ["-r", "15", "-crf", "23"],
    "crop_center70": ["-vf", "crop=trunc(iw*0.7/2)*2:trunc(ih*0.7/2)*2", "-crf", "23"],  # even sizes for yuv420p
}
_AUDIO_ARGS = {
    "mp3_64k": (["-vn", "-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    "opus_24k": (["-vn", "-c:a", "libopus", "-b:a", "24k"], "opus"),
    "pstn8k": (["-vn", "-ar", "8000", "-c:a", "pcm_s16le"], "wav"),
}


class PerturbError(ValueError):
    """A transform cannot be applied to this item (wrong modality, missing donor or tool, undecodable input)."""


@dataclass(frozen=True)
class Output:
    data: bytes
    ext: str


@dataclass
class PerturbContext:
    tmp_dir: Path
    exif_donor: bytes | None = None
    ffmpeg: str | None = field(default_factory=lambda: shutil.which("ffmpeg"))
    ffprobe: str | None = field(default_factory=lambda: shutil.which("ffprobe"))
    timeout_s: float = 600.0


def transforms_for(modality: str) -> tuple[str, ...]:
    return {"image": IMAGE_TRANSFORMS, "video": VIDEO_TRANSFORMS, "audio": AUDIO_TRANSFORMS}.get(modality, ())


def rng_for(item_id: str, transform: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{item_id}:{transform}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def perturb(data: bytes, *, item_id: str, modality: str, transform: str, ctx: PerturbContext) -> Output:
    if transform not in transforms_for(modality):
        raise PerturbError(f"{transform} is not a {modality} transform")
    rng = rng_for(item_id, transform)
    if modality == "image":
        return _image(data, transform, rng, ctx)
    ctx.tmp_dir.mkdir(parents=True, exist_ok=True)
    src = ctx.tmp_dir / f"{item_id}.{transform}.src"
    src.write_bytes(data)
    try:
        if transform == "noise_snr20":
            return _noise_snr20(ctx, src, rng)
        if modality == "video":
            return Output(_ffmpeg(ctx, src, [*_VIDEO_ARGS[transform], *_X264], "mp4"), "mp4")
        args, ext = _AUDIO_ARGS[transform]
        return Output(_ffmpeg(ctx, src, args, ext), ext)
    finally:
        src.unlink(missing_ok=True)


# --- images ------------------------------------------------------------------------------------------------

def _rgb(data: bytes) -> Image.Image:
    try:
        return media.to_rgb8(media.open_image(data))
    except media.MediaError as e:
        raise PerturbError(str(e)) from None


def _encode(im: Image.Image, fmt: str, **options) -> bytes:
    out = io.BytesIO()
    im.save(out, format=fmt, **options)
    return out.getvalue()


def _png(im: Image.Image) -> Output:
    return Output(_encode(im, "PNG"), "png")


def _jpeg(im: Image.Image, quality: int) -> Output:
    return Output(_encode(im, "JPEG", quality=quality, subsampling=2, optimize=False), "jpg")


def _webp(im: Image.Image) -> Output:
    return Output(_encode(im, "WEBP", quality=80, method=4), "webp")


def _crop70(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = im.size
    cw, ch = max(1, round(w * 0.7)), max(1, round(h * 0.7))
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return im.crop((x0, y0, x0 + cw, y0 + ch))


def _half(im: Image.Image) -> Image.Image:
    return im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.Resampling.BILINEAR)


def _fit_long_side(im: Image.Image, long_side: int) -> Image.Image:
    """Area-resample down to `long_side`; platforms do not upscale smaller uploads (assumption)."""
    scale = long_side / max(im.size)
    if scale >= 1:
        return im
    size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    return Image.fromarray(cv2.resize(np.asarray(im), size, interpolation=cv2.INTER_AREA))


def _screenshot(im: Image.Image) -> Image.Image:
    """Fit inside a 1170×2532 black phone screen between a 47 px status bar and a 34 px home-indicator bar."""
    canvas = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    area_h = SCREEN_H - STATUS_BAR - HOME_BAR
    scale = min(SCREEN_W / im.width, area_h / im.height)
    size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    canvas.paste(im.resize(size, Image.Resampling.LANCZOS),
                 ((SCREEN_W - size[0]) // 2, STATUS_BAR + (area_h - size[1]) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((60, 16, 140, 32), fill=(255, 255, 255))  # clock
    draw.rectangle((1040, 16, 1100, 32), outline=(255, 255, 255), width=3)  # battery outline
    draw.rectangle((1045, 21, 1085, 27), fill=(255, 255, 255))  # battery level
    draw.rounded_rectangle(((SCREEN_W - 400) // 2, SCREEN_H - HOME_BAR + 12, (SCREEN_W + 400) // 2, SCREEN_H - HOME_BAR + 22),
                           radius=5, fill=(255, 255, 255))
    return canvas


def _noise3(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(im, dtype=np.float64)
    return Image.fromarray(np.clip(np.rint(arr + rng.normal(0.0, 3.0, arr.shape)), 0, 255).astype(np.uint8))


def _rephoto(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Perspective warp (camera tilt ±4° about both axes), Gaussian blur σ 0.8, gamma 0.9, moiré via a 0.97× nearest
    resample. The re-photographed JPEG q 85 encode happens in the caller."""
    arr = np.ascontiguousarray(np.asarray(im))
    h, w = arr.shape[:2]
    ax, ay = np.deg2rad(rng.uniform(-4.0, 4.0, 2))
    f = float(max(w, h))
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    homography = k @ ry @ rx @ np.linalg.inv(k)
    warped = cv2.warpPerspective(arr, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    blurred = cv2.GaussianBlur(warped, (0, 0), 0.8)
    lut = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** 0.9), 0, 255).astype(np.uint8)
    size = (max(1, round(w * 0.97)), max(1, round(h * 0.97)))
    return Image.fromarray(cv2.resize(cv2.LUT(blurred, lut), size, interpolation=cv2.INTER_NEAREST))


def exif_donor_from(data: bytes) -> bytes:
    """The raw EXIF block (with the `Exif\\0\\0` header) of a donor image."""
    try:
        exif = media.open_image(data).info.get("exif")
    except media.MediaError as e:
        raise PerturbError(f"exif donor: {e}") from None
    if not exif:
        raise PerturbError("the donor image has no EXIF block")
    return exif if exif.startswith(b"Exif\x00\x00") else b"Exif\x00\x00" + exif


def exif_forge(data: bytes, donor: bytes | None) -> Output:
    if not donor:
        raise PerturbError("exif_forge needs a donor EXIF block (--exif-donor)")
    kind = media.sniff(data)
    if kind == ("image", "jpg"):
        import piexif

        out = io.BytesIO()
        piexif.insert(donor, data, out)  # rewrites APP1 only; the entropy-coded pixels stay byte-identical
        return Output(out.getvalue(), "jpg")
    if kind == ("image", "png"):
        return Output(_encode(media.open_image(data), "PNG", exif=donor), "png")  # lossless re-encode with eXIf
    raise PerturbError(f"exif_forge supports JPEG and PNG natives, not {kind[1] if kind else 'unknown bytes'}")


def _image(data: bytes, transform: str, rng: np.random.Generator, ctx: PerturbContext) -> Output:
    if transform == "exif_forge":
        return exif_forge(data, ctx.exif_donor)
    im = _rgb(data)
    if transform in ("jpeg95", "jpeg75", "jpeg50"):
        return _jpeg(im, int(transform[4:]))
    if transform == "webp80":
        return _webp(im)
    if transform == "resize0.5":
        return _png(_half(im))
    if transform == "crop70":
        return _png(_crop70(im, rng))
    if transform == "screenshot":
        return _png(_screenshot(im))
    if transform == "screenshot_jpeg":
        return _jpeg(_screenshot(im), 85)
    if transform == "social_ig":
        return _jpeg(_fit_long_side(im, 1080), 72)
    if transform == "social_x":
        return _jpeg(_fit_long_side(im, 2048), 85)
    if transform == "launder":  # crop70 → resize0.5 → jpeg75 → webp80, each step decoding the previous output
        step = _png(_crop70(im, rng))
        step = _png(_half(_rgb(step.data)))
        step = _jpeg(_rgb(step.data), 75)
        return _webp(_rgb(step.data))
    if transform == "noise3":
        return _png(_noise3(im, rng))
    if transform == "upscale2x":
        if im.width * im.height * 4 > media.MAX_PIXELS:
            raise PerturbError("upscale2x output would exceed the pixel limit")
        return _png(im.resize((im.width * 2, im.height * 2), Image.Resampling.BICUBIC))
    if transform == "rephoto_proxy":
        return _jpeg(_rephoto(im, rng), 85)
    raise PerturbError(f"unknown image transform {transform}")


# --- video and audio ---------------------------------------------------------------------------------------

def _ffmpeg(ctx: PerturbContext, src: Path, args: list[str], ext: str) -> bytes:
    if ctx.ffmpeg is None:
        raise PerturbError("ffmpeg is not installed")
    out = src.with_name(f"{src.stem}.out.{ext}")
    cmd = [ctx.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-protocol_whitelist", "file", "-i", media.av_input(src),
           *args, *_BITEXACT, str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=ctx.timeout_s, check=False, stdin=subprocess.DEVNULL)
        if proc.returncode != 0 or not out.is_file():
            tail = proc.stderr.decode("utf-8", "replace").strip()[-300:]
            raise PerturbError(f"ffmpeg failed ({tail or proc.returncode})")
        return out.read_bytes()
    except subprocess.TimeoutExpired:
        raise PerturbError(f"ffmpeg timed out after {ctx.timeout_s:.0f} s") from None
    finally:
        out.unlink(missing_ok=True)


def _sample_rate(ctx: PerturbContext, src: Path) -> int:
    if ctx.ffprobe is None:
        raise PerturbError("ffprobe is not installed")
    proc = subprocess.run([ctx.ffprobe, "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0", "-show_entries",
                           "stream=sample_rate", "-of", "csv=p=0", media.av_input(src)], capture_output=True, timeout=60, check=False,
                          stdin=subprocess.DEVNULL)
    text = proc.stdout.decode("ascii", "replace").strip()
    if proc.returncode != 0 or not text.isdigit():
        raise PerturbError("no audio stream")
    return int(text)


def _noise_snr20(ctx: PerturbContext, src: Path, rng: np.random.Generator) -> Output:
    """Additive white noise at 20 dB SNR on the mono downmix at the native sample rate, written as 16-bit WAV."""
    if ctx.ffmpeg is None:
        raise PerturbError("ffmpeg is not installed")
    rate = _sample_rate(ctx, src)
    proc = subprocess.run([ctx.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file", "-i",
                           media.av_input(src), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"], capture_output=True,
                          timeout=ctx.timeout_s, check=False, stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise PerturbError(f"ffmpeg decode failed ({proc.stderr.decode('utf-8', 'replace').strip()[-300:]})")
    x = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    power = float(np.mean(x ** 2)) if x.size else 0.0
    if power <= 0:
        raise PerturbError("silent or empty audio")
    y = np.clip(x + rng.normal(0.0, np.sqrt(power / 10 ** (20 / 10)), x.shape), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.rint(y * 32767).astype("<i2").tobytes())
    return Output(buf.getvalue(), "wav")


# --- robustness subset -------------------------------------------------------------------------------------

def allocate(available: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` evenly across strata, capped by availability; the remainder goes to strata with room (by name)."""
    alloc = dict.fromkeys(available, 0)
    remaining = total
    open_strata = sorted(k for k, v in available.items() if v > 0)
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        for k in list(open_strata):
            take = min(share, available[k] - alloc[k], remaining)
            alloc[k] += take
            remaining -= take
            if alloc[k] >= available[k]:
                open_strata.remove(k)
            if remaining == 0:
                break
    return alloc


def robustness_subset(rows: Iterable[dict]) -> list[dict]:
    """The robustness subset. A manifest with a `robustness` column (vb-1.0-core, B-4c) names it: the flagged rows."""
    rows = list(rows)
    if any("robustness" in row for row in rows):
        return [row for row in rows if row.get("robustness")]
    strata: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row.get("role") or "core") != "core" or row.get("modality") not in ROBUSTNESS_TARGETS:
            continue
        real = row.get("label") == "real"
        if real and row["modality"] == "image" and row.get("real_stratum") not in IMAGE_REAL_STRATA:
            continue  # R6 hard negatives are not in the robustness subset (ML §2.7: R1–R5)
        stratum = (row.get("real_stratum") if real else row.get("family")) or "unknown"
        strata[(row["modality"], "real" if real else "fake")][stratum].append(row)
    chosen: list[dict] = []
    for modality, targets in ROBUSTNESS_TARGETS.items():
        for label, target in zip(("real", "fake"), targets):
            groups = strata.get((modality, label), {})
            alloc = allocate({k: len(v) for k, v in groups.items()}, target)
            for k in sorted(groups):
                ordered = sorted(groups[k], key=lambda r: hashlib.sha256(f"robustness:{r['item_id']}".encode()).hexdigest())
                chosen += ordered[:alloc[k]]
    return chosen


# --- runner ------------------------------------------------------------------------------------------------

@dataclass
class PerturbSummary:
    done: int = 0
    skipped: int = 0
    failed: Counter = field(default_factory=Counter)


def _state_events(vbench_dir: Path, *, allow_torn_tail: bool = True) -> list[dict]:
    path = vbench_dir / "perturbed" / "state.jsonl"
    events = []
    if path.is_file():
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as error:
                if allow_torn_tail and index == len(lines) - 1 and not content.endswith("\n"):
                    continue  # read-only recovery of a genuinely unterminated crash tail
                raise PerturbError(f"Invalid perturbation resume journal at line {index + 1}") from error
            if not isinstance(event, dict) or (event.get("t") in ("item", "error") and any(
                not isinstance(event.get(key), str) or not event[key] for key in ("item_id", "transform")
            )):
                raise PerturbError(f"Invalid perturbation resume journal record at line {index + 1}")
            events.append(event)
    return events


def _append(vbench_dir: Path, event: dict) -> None:
    path = vbench_dir / "perturbed" / "state.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = path.is_file() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if torn else "") + json.dumps(event, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _successful_records(events: Iterable[dict], selected: set[tuple[str, str]] | None = None) -> dict:
    prior_successes: dict[tuple[str, str], dict] = {}
    for event in events:
        if event.get("t") != "item":
            continue
        key = (event.get("item_id"), event.get("transform"))
        if selected is not None and key not in selected:
            continue
        if key in prior_successes:
            raise PerturbError(f"Duplicate successful resumed perturbation: {key[0]} / {key[1]}")
        record = event.get("row")
        if not isinstance(record, dict) or (record.get("item_id"), record.get("transform")) != key:
            raise PerturbError(f"Invalid resumed perturbation identity: {key[0]} / {key[1]}")
        relative = Path(record["path"]) if isinstance(record.get("path"), str) else None
        if (relative is None or relative.parent != Path("perturbed") / key[1]
                or relative.stem != key[0] or not relative.suffix):
            raise PerturbError(f"Invalid canonical resumed perturbation path: {key[0]} / {key[1]}")
        digest = record.get("sha256")
        if (type(record.get("bytes")) is not int or record["bytes"] <= 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise PerturbError(f"Invalid resumed perturbation byte pins: {key[0]} / {key[1]}")
        prior_successes[key] = record
    return prior_successes


def _reconcile_published_records(vbench_dir: Path, prior_successes: dict,
                                 selected: set[tuple[str, str]] | None = None) -> None:
    from app.bench.input_integrity import verify_local_input

    manifest = vbench_dir / "perturbed" / "manifest.parquet"
    if manifest.exists() or manifest.is_symlink():
        import pyarrow.parquet as pq

        fields = ("item_id", "transform", "path", "bytes", "sha256")
        try:
            path = verify_local_input(vbench_dir, "perturbed/manifest.parquet")
            manifest_rows = pq.read_table(path, columns=list(fields)).to_pylist()
        except Exception as error:  # PyArrow malformed-file/schema exception types vary by release.
            raise PerturbError("Cannot resume: perturbation manifest is unreadable or missing required columns") from error
        manifest_keys = set()
        for record in manifest_rows:
            if any(not isinstance(record.get(field), str) or not record[field] for field in ("item_id", "transform")):
                raise PerturbError("Cannot resume: invalid perturbation manifest identity")
            key = (record["item_id"], record["transform"])
            if selected is not None and key not in selected:
                continue
            if key in manifest_keys:
                raise PerturbError(f"Duplicate perturbation manifest record: {key[0]} / {key[1]}")
            manifest_keys.add(key)
            if key not in prior_successes:
                raise PerturbError(f"Manifest-only perturbation success has no matching journal record: {key[0]} / {key[1]}")
            previous = prior_successes[key]
            if any(type(record[field]) is not type(previous.get(field)) or record[field] != previous.get(field)
                   for field in fields):
                raise PerturbError(f"Conflicting perturbation manifest and journal pins: {key[0]} / {key[1]}")


def validate_manifest_publication(vbench_dir: Path) -> None:
    """Read only metadata; refuse publication that loses or changes recorded history.

    Existing published successes must match one unambiguous journal success.
    Journal-only successes are allowed after interrupted manifest publication.
    No native or derivative payload is opened by this global guard.
    """
    _reconcile_published_records(vbench_dir, _successful_records(_state_events(vbench_dir)))


def run_perturb(vbench_dir: Path, rows: Iterable[dict], ctx: PerturbContext, *, transforms: Iterable[str] | None = None,
                on_progress: Callable[[PerturbSummary], None] | None = None) -> PerturbSummary:
    from app.bench.input_integrity import verify_recorded_artifact

    wanted = set(transforms) if transforms is not None else None
    rows = list(rows)
    events = _state_events(vbench_dir)
    selected = {(row["item_id"], transform) for row in rows for transform in transforms_for(row["modality"])
                if wanted is None or transform in wanted}
    prior_successes = _successful_records(events, selected)
    _reconcile_published_records(vbench_dir, prior_successes, selected)
    # Check ALL selected successes before reading natives, transforming, making
    # directories or appending state. Never regenerate changed frozen artifacts.
    for key, record in prior_successes.items():
        try:
            verify_recorded_artifact(vbench_dir, record)
        except ValueError as error:
            raise PerturbError(f"Cannot resume perturbation {key[0]} / {key[1]}: {error}") from error
    done = {(e["item_id"], e["transform"]) for e in events if e.get("t") in ("item", "error")}
    if selected - done:
        # Appending after a torn tail would make the corrupt line permanent
        # interior history. Preserve it for reconciliation instead of rewriting.
        if _state_events(vbench_dir, allow_torn_tail=False) != events:
            raise PerturbError("Perturbation resume journal changed during preflight")
        for item_id, transform in sorted(selected - done):
            directory = vbench_dir / "perturbed" / transform
            if directory.is_symlink() or any(not path.name.endswith(".part")
                                            for path in directory.glob(f"{item_id}.*")):
                raise PerturbError(f"Unrecorded completed perturbation output: {item_id} / {transform}")
    summary = PerturbSummary()
    for row in rows:
        data: bytes | None = None
        for transform in transforms_for(row["modality"]):
            if wanted is not None and transform not in wanted:
                continue
            if (row["item_id"], transform) in done:
                summary.skipped += 1
                continue
            if data is None:
                data = (vbench_dir / row["native_path"]).read_bytes()
            try:
                out = perturb(data, item_id=row["item_id"], modality=row["modality"], transform=transform, ctx=ctx)
            except PerturbError as e:
                summary.failed[str(e)] += 1
                _append(vbench_dir, {"t": "error", "item_id": row["item_id"], "transform": transform, "reason": str(e)})
                continue
            rel = f"perturbed/{transform}/{row['item_id']}.{out.ext}"
            target = vbench_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(out.data)
            os.replace(tmp, target)
            try:
                info = _probe(out.data, target, row["modality"])
            except Image.DecompressionBombError:  # e.g. upscale2x of a > 89 MP photograph; the bench loader refuses such files too
                target.unlink(missing_ok=True)
                reason = "the output exceeds the Pillow decompression-bomb pixel limit"
                summary.failed[reason] += 1
                _append(vbench_dir, {"t": "error", "item_id": row["item_id"], "transform": transform, "reason": reason})
                continue
            _append(vbench_dir, {"t": "item", "item_id": row["item_id"], "transform": transform, "row": {
                "item_id": row["item_id"], "transform": transform, "modality": row["modality"], "label": row.get("label"),
                "path": rel, "bytes": len(out.data), "sha256": hashlib.sha256(out.data).hexdigest(), **info}})
            done.add((row["item_id"], transform))
            summary.done += 1
            if on_progress is not None and summary.done % 100 == 0:
                on_progress(summary)
    return summary


def _probe(data: bytes, path: Path, modality: str) -> dict:
    if modality == "image":
        with Image.open(io.BytesIO(data)) as im:
            return {"width": im.width, "height": im.height, "duration_s": None}
    info = media.probe_av(path)
    return {"width": info.get("width"), "height": info.get("height"), "duration_s": info.get("duration_s")}


def write_perturbed_manifest(vbench_dir: Path) -> Path:
    validate_manifest_publication(vbench_dir)

    import pyarrow as pa
    import pyarrow.parquet as pq

    s, i, f = pa.string(), pa.int64(), pa.float64()
    schema = pa.schema([("item_id", s), ("transform", s), ("modality", s), ("label", s), ("path", s), ("bytes", i),
                        ("sha256", s), ("width", i), ("height", i), ("duration_s", f)])
    rows = sorted((e["row"] for e in _state_events(vbench_dir) if e.get("t") == "item"),
                  key=lambda r: (r["transform"], r["item_id"]))
    out = vbench_dir / "perturbed" / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in schema.names} for r in rows], schema=schema), tmp)
    os.replace(tmp, out)
    return out
