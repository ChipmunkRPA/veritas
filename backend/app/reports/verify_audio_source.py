"""Compare published WAV intervals with a separately supplied source file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

from app.detectors.audio.decode import decode_audio, source_ranges, wav_bytes
from app.evidence.stereo_binding import SCHEMA as STEREO_CONTEXT_SCHEMA
from app.reports.public_verify import SCHEMA

SOURCE_FIELDS = ("samples", "sample_rate", "channels", "source_stream_index", "source_sample_rate", "source_channels",
                 "timeline_time_base", "timeline")


def _window_clips(window: dict, timestamps: list[dict]) -> Iterator[tuple[str, dict, list[dict] | None, dict]]:
    assets = window["artifacts"]
    yield f"{window['id']} selected clip", window["interval"], timestamps, assets["selected_clip"]
    yield f"{window['id']} context clip", window["context_interval"], None, assets["context_clip"]


def _clips(audio: dict, alignment: dict) -> Iterator[tuple[str, dict, list[dict] | None, dict]]:
    for channel in (audio, alignment):
        for packet in channel.get("packets", []):
            yield from _window_clips(packet, packet["anchor"]["source_timestamp_ranges"])
    for event in (audio.get("acoustic_context") or {}).get("events", []):
        yield from _window_clips(event, event["source_timestamp_ranges"])
    for window in (audio.get("sound_events") or {}).get("windows", []):
        yield f"{window['id']} sound clip", window, window["source_timestamp_ranges"], window["artifacts"]["selected_clip"]
    if (review := audio.get("short_voice_review")) and review["status"] == "flagged":
        yield "Short speech-window clip", review["interval"], review["source_timestamp_ranges"], review["artifacts"]["selected_clip"]
    if review := audio.get("long_voice_review"):
        from app.evidence.audio_long_review_binding import media_windows

        for window in media_windows(review):
            if clip := window["artifacts"].get("selected_clip"):
                yield "Long speech-window clip", window, window["source_timestamp_ranges"], clip
    if (review := audio.get("background_reduced_voice_review")) and review["status"] == "flagged":
        yield "Original background-reduced review clip", review["interval"], review["source_timestamp_ranges"], review["artifacts"]["selected_clip"]
    for name in ("repetition_context", "similarity_context"):
        for pair in (audio.get(name) or {}).get("pairs", []):
            for side in ("first", "second"):
                interval = pair[side]
                yield f"{pair['id']} {side} clip", interval, interval["source_timestamp_ranges"], pair["artifacts"][f"{side}_clip"]


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    """Re-decode the original and replay playback, stereo and waveform evidence."""
    kind = bundle.get("analysis", {}).get("kind")
    if bundle.get("schema") != SCHEMA or kind not in {"audio", "video"}:
        raise ValueError("Original-audio verification requires a public audio or video evidence package")
    evidence = json.loads(archive.read("evidence.json"))
    audio, alignment = (evidence, {}) if kind == "audio" else (evidence.get("audio", {}), evidence.get("av", {}))
    if not audio.get("coverage"):
        raise ValueError("The package has no frozen audio input and source timeline to compare")

    source = audio["source"]
    source_sha256 = bundle["source_commitment"]["sha256"]
    if source["sha256"] != source_sha256:
        raise ValueError("The audio input is bound to a different source")
    stereo = audio.get("stereo_context")
    decoded = decode_audio(original, source_sha256=source_sha256, max_seconds=600.0 if kind == "audio" else 120.0,
                           inspect_stereo=stereo is not None)
    with original.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source_sha256:
            raise ValueError("The supplied original changed during decoding")
    if (decoded.mapping["pcm_sha256"] != source["pcm_sha256"]
            or hashlib.sha256(decoded.wav).hexdigest() != source["wav_sha256"]):
        raise ValueError("Decoded audio differs from the committed mono PCM or WAV. Check the local FFmpeg version.")
    if any(decoded.mapping[key] != source[key] for key in SOURCE_FIELDS):
        raise ValueError("Decoded audio stream or source timeline differs from its commitment")
    if "audio_stream_indices" in source and source["audio_stream_indices"] != decoded.mapping.get("audio_stream_indices"):
        raise ValueError("Audio track coverage differs from the original container")
    if "source_codec" in source:
        from app.evidence.audio_short_review_binding import source_metadata_matches

        if not source_metadata_matches(source, decoded.mapping):
            raise ValueError("Audio codec scope differs from the original container")
    if audio["coverage"]["decode_limit_reached"] != (len(decoded.pcm) >= decoded.mapping["limit_samples"]):
        raise ValueError("Decoded audio duration disagrees with the recorded coverage limit")

    replay_detail = ""
    if stereo and stereo["schema"] == STEREO_CONTEXT_SCHEMA and stereo["stereo_pcm_sha256"] is not None:
        if stereo != decoded.mapping.get("stereo_context"):
            raise ValueError("Stereo channel and mono-downmix evidence differs from the original audio samples")
        replay_detail = f" Recomputed stereo channel and mono-downmix evidence for {len(stereo['windows'])} windows."
    covered = set()
    intervals = 0
    for label, span, timestamps, asset in _clips(audio, alignment):
        start, end = span["start_sample"], span["end_sample"]
        if any(type(i) is not int for i in (start, end)) or not 0 <= start < end <= len(decoded.pcm):
            raise ValueError(f"{label} has an invalid source interval")
        if timestamps is not None and timestamps != source_ranges(decoded.mapping, start, end):
            raise ValueError(f"{label} differs from its recorded source timestamps")
        # The ordinary verifier already checked the archive asset bytes against this digest.
        if hashlib.sha256(wav_bytes(decoded.pcm[start:end])).hexdigest() != asset["sha256"]:
            raise ValueError(f"{label} differs from the samples at its recorded source interval")
        covered.add(asset["sha256"])
        intervals += 1
    if listening := audio.get("stereo_listening"):
        from app.evidence.stereo_listening import selected_pcm, verify_payloads
        from app.storage.objects import ObjectIntegrityError

        if decoded.stereo_pcm is None:
            raise ValueError("The original audio did not yield the aligned stereo channels")
        plots = {entry["sha256"]: entry["file"] for entry in bundle["artifacts"] if entry["media_type"] == "image/png"}
        try:
            covered.update(verify_payloads(listening, stereo, decoded.pcm, selected_pcm(stereo, decoded.stereo_pcm),
                                           read_waveform=lambda sha: archive.read(plots[sha])))
        except ObjectIntegrityError as error:
            raise ValueError(str(error)) from error
        intervals += 3 * len(listening["windows"])
    if (review := audio.get("background_reduced_voice_review")) is not None:
        from app.detectors.audio.decode import read_wav
        from app.evidence.audio_background_review_binding import SPAN
        from app.evidence.audio_background_review_binding import validate as validate_background_review
        from app.evidence.audio_packets import _chart

        validate_background_review(review, decoded.mapping)
        if review["status"] == "flagged":
            assets = review["artifacts"]
            paths = {entry["sha256"]: entry["file"] for entry in bundle["artifacts"] if entry["media_type"] == "audio/wav"}
            pin = assets["processed_clip"]["sha256"]
            pcm = read_wav(archive.read(paths[pin]))
            if len(pcm) != SPAN or hashlib.sha256((pcm * 32768.0).astype("<i2").tobytes()).hexdigest() != review["processed_pcm_sha256"]:
                raise ValueError("Processed speech playback differs from its recorded PCM commitment")
            covered.add(pin)
            original_waveform = _chart(decoded.pcm, [], start=0, end=SPAN, selected=(0, SPAN), scores=False)
            if hashlib.sha256(original_waveform).hexdigest() != assets["waveform"]["sha256"]:
                raise ValueError("Background-reduced speech waveform differs from the original samples")
            replay_detail += " Matched the marked original waveform and the saved processed audio commitment."
    expected = {entry["sha256"] for entry in bundle["artifacts"] if entry["media_type"] == "audio/wav"}
    if covered != expected:
        raise ValueError("The source comparison did not account for every WAV asset in the package")
    similarity = audio.get("similarity_context")
    if similarity and similarity["schema"] == "chipmunk.audio-repeat-similarity/2":
        from app.evidence.audio_similarity import verify

        paths = {entry["sha256"]: entry["file"] for entry in bundle["artifacts"] if entry["media_type"] == "audio/wav"}
        verify(similarity, pcm=decoded.pcm, mapping=decoded.mapping, source=source,
               read_asset=lambda sha: archive.read(paths[sha]), repetition=audio.get("repetition_context"))
        replay_detail += " Recomputed the bounded waveform similarity result."
    if (review := audio.get("short_voice_review")) and review["status"] == "flagged":
        from app.evidence.audio_packets import _chart

        selected = (review["interval"]["start_sample"], review["interval"]["end_sample"])
        waveform = _chart(decoded.pcm, [], start=0, end=len(decoded.pcm), selected=selected, scores=False)
        if hashlib.sha256(waveform).hexdigest() != review["artifacts"]["waveform"]["sha256"]:
            raise ValueError("Short speech-window waveform differs from the recorded samples")
        replay_detail += " Recomputed the marked short speech-window waveform."
    if (review := audio.get("long_voice_review")) is not None:
        from app.evidence.audio_long_review import verify_source
        from app.evidence.audio_packets import _chart

        verify_source(review, pcm=decoded.pcm, mapping=decoded.mapping, waveform=_chart)
        replay_detail += f" Verified {len(review['windows'])} long speech-window input ranges and their marked waveforms."
        if "regions" in review:
            replay_detail += f" Verified {len(review['regions'])} selected speech ranges against the same original samples."
    interval_label = "interval" if intervals == 1 else "intervals"
    asset_label = "asset" if len(covered) == 1 else "assets"
    playback = (f"{intervals} original playback {interval_label} and accounted for {len(covered)} WAV {asset_label}"
                if audio.get("background_reduced_voice_review", {}).get("status") == "flagged"
                else f"{intervals} published playback {interval_label} across {len(covered)} WAV {asset_label}")
    return (f"Original SHA-256, {len(decoded.pcm):,} decoded mono samples and source timeline matched. "
            f"Verified {playback}.{replay_detail}")
