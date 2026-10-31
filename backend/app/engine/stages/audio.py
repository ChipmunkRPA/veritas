"""Standalone audio pipeline stages (ARCHITECTURE §3.1–3.2 `audio`, PRD DC-05; BUILD_SPEC A-1):
ingest → probe → provenance → decode → condition → detect → temporal → fuse → synthesise → render → publish.

- **probe** reads the first audio stream with ffprobe (container, codec, sample rate, channels, duration, bit rate, tags).
  A file without an audio stream fails the analysis. Persisted.
- **decode** extracts up to 600 s of 16 kHz mono PCM with ffmpeg into run state. Not persisted: re-derived on resume.
- **condition** labels every second speech, music, noise or silence (`detectors/audio/content.py`): shares, the speech spans
  that gate the voice windows, the slice id and the limiting conditions. A recording with less than 1 s of speech, or with
  more music than speech, is not assessable.
- **detect** runs the voice detectors on ≤ 4 s windows at the profile hop, keeping the windows at least half inside speech
  (at most 300). A recording that is not assessable runs none.
- **temporal** turns the window scores into the voice-lane series and segments (`video/segments.py`), lists the content
  runs as a table exhibit, and draws the waveform envelope and the spectrogram of the decoded signal
  (`detectors/audio/waveform.py`, L-7): descriptive pictures beside the voice lane, not detector outputs.
- **fuse** is the legacy voice fuser: the mean window probability of each voice detector, fused by weighted log-odds with
  the manifest `legacy_weight` (§4.4 fallback), published as an uncalibrated fallback `Decision`. A recording that is not
  assessable gets `not_assessable` with the content reason. With a compatible fuser artefact (none before vb-1.0) the
  calibrated decision replaces the fallback.
- **render** indexes the artefacts; **publish** stores the v2 result and a v1-shaped view (verdict, signals, audio facts).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.core.evidence import Signal
from app.detectors.audio import waveform
from app.detectors.audio.coverage import content_rows, coverage_seconds, scored_voice_spans
from app.detectors.common.fusion import fuse as legacy_fuse
from app.detectors.common.fusion import fuse_probability, prob_to_signal_fields
from app.detectors.video import segments as temporal_segments
from app.detectors.video.audio import SR, VOICE_WINDOW_S, extract_audio
from app.detectors.video.decode import _input
from app.engine.detector import Modality, StageOutputs
from app.engine.fusion import abstain
from app.engine.runner import StageContext
from app.engine.stages import common
from app.engine.stages import video as vid
from app.schemas.evidence import AnalysisResultV2, Decision
from app.schemas.legacy import FALLBACK_CAVEAT

PIPELINE_VERSION = "aud-2.0-a21"  # a21 (N-19q, finding 11; synthesis tpl-2.19): the heading of a check whose part in the estimate and stand-alone reading give two grades names its reading ("Weak support against AI generation in this estimate"); the finding of a check that entered the estimate stores its stand-alone words. a20 (N-19n, what the third part of N-19m left; supervisor check 29): under a configured prior the reason of a result that D-77 took names the other answer's part of the scale where its published estimate prints outside 35 to 64. a19: the v1 summary of an inconclusive result outside what was validated says so also where the cap changed no code (N-19n). a18: a decisive code needs the estimate as published in its answer’s part of the scale (D-77), which changes results under a configured prior (`VERITAS_PRIOR_AI`); the reason’s number is the printed whole points (N-19m, third part). a17: the checks that could not run on this server are one caveat with their names, in place of one caveat for each and of their ids in the caveat of a reduced profile; a caveat names a check as the page does, a step of the run and a provenance record in words, never a detector id, a stage name or the value of a kind (browser QA leads 3 and 4, design re-review D2-06); each check that scored and is not counted in the estimate carries the fusion's record of why (`outside_estimate`: not in the model, its gate closed, a weight of zero; browser QA Q2-12); a result of the fallback on an instance that holds a fusion model says that none was used for the analysis, not that none is loaded (N-19k; browser QA Q2-14); a16: the rule of D-76 (a decisive result needs a check whose published row is at least ln 2 toward its answer; it changes no recording of the packaged model, whose one row is beyond ln 2 in every decisive result) and each row of a check says whether the decision counts it (`counted`); a15: a check counts for a strong result, for "k of n detectors" and for "no check" when its published row is not zero; where every check ran or was not applicable and none that scored counts, the result is not assessable in words of its own, and a provenance record over such a result decides alone (N-19f, D-64 as clarified, code review R4-05, 2026-10-04); a14: a strong code needs two checks that scored, so the one-detector audio model never gives the strongest headline of either side, and a fallback result without a check of the sound is not assessable (N-18, D-64, 2026-10-03); a13: a not-assessable result has no caveat or reason that speaks of an estimate or a probability, and the voice lane series is a "window score" (N-17, D-68); a12: a recording that is not assessable under a loaded fusion model is that model's result, not the fallback's (N-17, QA P1-4, 2026-10-03); a11: with one voice detector the voice signal is that detector's, so its contribution, ledger row and finding agree (N-17, QA P1-4, 2026-10-03); a10: a finding that entered the estimate is worded by its contribution (N-17, D-63, 2026-10-03); a9: contributions are measured from each check's neutral point (N-17); a8: L-7: waveform envelope and spectrogram exhibits
FUSER_ID = "legacy.audio"
H2 = "H2_localised_manipulation"
ENTRY_STAGES = ("provenance", "detect")
EXHIBIT_STAGES = ("provenance", "detect", "temporal", "fuse")
AUDIO_MAX_SECONDS = 600.0
AUDIO_MAX_WINDOWS = 300  # 600 s at a 2 s hop
AUDIO_MAX_UNIT_SCORES = 1000
VOICE_SIGNAL = "audio_synthetic_voice"
VOICE_SIGNAL_ID = f"{FUSER_ID}:{VOICE_SIGNAL}"
LOSSLESS_CODECS = frozenset({"flac", "alac", "wavpack", "tta", "ape"})
CONTENT_TITLE = "Speech, music, noise and silence"
CONTENT_CAPTION = (
    "Each second of the recording labelled by a heuristic classifier (level, low-energy share, level spread and spectral "
    "flatness); consecutive seconds with the same label are merged. Only windows at least half inside speech were scored by "
    "the voice detectors. Scored spans mark samples included in a successfully completed voice-detector window; decoded "
    "audio outside those spans was not scored.")


# --- ingest-side stages --------------------------------------------------------------------------------------

def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def audio_facts(path: Path) -> dict[str, Any]:
    """ffprobe facts of the first audio stream. Raises ValueError when the file has none or cannot be read.

    `file` protocol only: an uploaded playlist (HLS / concat) cannot make ffprobe open a URL."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed")
    try:
        run = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-print_format", "json", "-show_format",
                              "-show_streams", _input(path)],
                             capture_output=True, text=True, timeout=60, check=True, stdin=subprocess.DEVNULL)
        doc = json.loads(run.stdout)
    except (subprocess.SubprocessError, ValueError) as e:
        raise ValueError("The audio file could not be read") from e
    streams = doc.get("streams") or []
    stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError("The file has no audio stream")
    fmt = doc.get("format") or {}
    codec = stream.get("codec_name")
    tags = {**{str(k).lower(): str(v)[:200] for k, v in (fmt.get("tags") or {}).items()},
            **{str(k).lower(): str(v)[:200] for k, v in (stream.get("tags") or {}).items()}}
    has_video = any(s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic") for s in streams)
    bit_rate = _number(fmt.get("bit_rate"))
    return {"container": fmt.get("format_name"), "codec": codec, "sample_rate": int(_number(stream.get("sample_rate")) or 0) or None,
            "channels": stream.get("channels"), "bit_rate": int(bit_rate) if bit_rate else None,
            "duration_s": _number(fmt.get("duration")) or _number(stream.get("duration")),
            "lossless": bool(codec) and (str(codec).startswith("pcm_") or codec in LOSSLESS_CODECS), "has_video": has_video,
            "tags": tags}


def probe(ctx: StageContext) -> dict[str, Any]:
    return {"facts": common.json_safe(audio_facts(common.media_path(ctx)))}


def provenance(ctx: StageContext) -> dict[str, Any]:
    return common.run_detectors(ctx, (Modality.AUDIO,), include={"provenance"},
                                outputs=StageOutputs(probe=ctx.outputs["probe"]["facts"]))


def decode(ctx: StageContext) -> dict[str, Any]:
    """Up to 600 s of 16 kHz mono PCM into run state (not persisted); the JSON output carries its sha256 for resume checks."""
    from app.detectors.video.adapters.common import AudioTrack

    pcm = extract_audio(common.media_path(ctx), max_seconds=AUDIO_MAX_SECONDS)
    if pcm is None:
        raise ValueError("The audio stream is unreadable (corrupt file or unsupported codec)")
    ctx.check("audio decoding")
    ctx.state["audio"] = AudioTrack("analysed", pcm, len(pcm))
    return {"sample_rate": SR, "samples": int(len(pcm)), "seconds": round(len(pcm) / SR, 3),
            "sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}


def detect(ctx: StageContext) -> dict[str, Any]:
    content = ctx.outputs["condition"]["content"]
    if not content["assessable"]:
        return {"detectors": [], "not_run": content["reason"], "exhibits": [], "artefacts": []}
    track = ctx.state.get("audio")
    if track is None:
        raise RuntimeError("the decoded audio is not available")
    gated = dataclasses.replace(track, gate=tuple((float(a), float(b)) for a, b in content["speech_spans"]),
                                max_windows=AUDIO_MAX_WINDOWS)
    facts = ctx.outputs["probe"]["facts"]
    return common.run_detectors(ctx, (Modality.AUDIO,), exclude={"provenance"},
                                outputs=StageOutputs(probe={**facts, "duration_s": facts.get("duration_s") or content["duration_s"]},
                                                     audio=gated),
                                max_units=AUDIO_MAX_UNIT_SCORES)


# --- temporal stage ------------------------------------------------------------------------------------------------

def _lane_of(entry: Mapping[str, Any]) -> str | None:
    return "voice" if entry["family"] == "voice" else None


def _windows(entry: Mapping[str, Any]) -> list[float]:
    return [float(u["raw"]) for u in entry["unit_scores"] if u["unit_ref"].startswith("window:") and u["raw_kind"] == "probability"]


def sole_voice_entry(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The one voice detector behind the recording's voice signal, when exactly one scored speech windows (N-17; browser QA
    P1-4). The signal is then that check: it takes the detector's id, so the contribution the fusion model gives the detector,
    the ledger row and the finding are one thing. With several voice detectors the signal is their weighted average and keeps
    the id of the legacy fuser; no fusion model holds such an average, so it stays outside the estimate."""
    scored = [e for e in entries if e["status"] == "ok" and e["family"] == "voice" and e["aggregate"] is not None and _windows(e)]
    return scored[0] if len(scored) == 1 else None


def voice_signal_id(entries: Sequence[Mapping[str, Any]]) -> str:
    """The id of the voice signal of these detector entries, the same in the temporal stage (lane sources, exhibits) and in
    the fuse stage (the signal)."""
    sole = sole_voice_entry(entries)
    return common.signal_id(sole["detector_id"], sole["aggregate"]) if sole is not None else VOICE_SIGNAL_ID


def _signal_pictures(ctx: StageContext) -> tuple[str | None, str | None]:
    """The waveform envelope (series) and the spectrogram (image) of the decoded signal; (None, None) without the signal."""
    track = ctx.state.get("audio")
    pcm = getattr(track, "pcm", None)
    if pcm is None or len(pcm) == 0:
        return None, None
    series, facts = waveform.envelope_series(pcm, SR)
    waveform_id = ctx.exhibits.series(series, x_unit="s", title=waveform.ENVELOPE_TITLE, media_ref="audio",
                                      caption=waveform.envelope_caption(facts))
    ctx.check("audio waveform")
    picture = waveform.spectrogram_png(pcm, SR)
    if picture is None:
        return waveform_id, None
    data, picture_facts = picture
    return waveform_id, ctx.exhibits.image(data, role="spectrum", media_type="image/png", title=waveform.SPECTROGRAM_TITLE,
                                           media_ref="audio", caption=waveform.spectrogram_caption(picture_facts))


def temporal(ctx: StageContext) -> dict[str, Any]:
    """Voice-lane series and segments from the window scores, and the content runs as a table exhibit. Persisted."""
    detected = common.entries(ctx, ("detect",))
    series = temporal_segments.collect(detected, lane_of=_lane_of, channel_of=lambda entry: "audio")
    voice_id = voice_signal_id(detected)
    lanes, found = vid._emit_lanes(ctx, series, {e["detector_id"]: e for e in detected}, None,
                                   published=lambda entry, track_id: [voice_id] if entry["family"] == "voice" else [],
                                   subject="whole recording")
    content = ctx.outputs["condition"]["content"]
    spans = scored_voice_spans(detected)
    rows = content_rows(content["runs"], spans)
    content_id = ctx.exhibits.table(["start_s", "end_s", "content", "voice detectors"], rows, title=CONTENT_TITLE,
                                    media_ref="audio", caption=CONTENT_CAPTION)
    waveform_id, spectrogram_id = _signal_pictures(ctx)
    return {"method": temporal_segments.METHOD, "params": temporal_segments.PARAMS, "lanes": lanes, "content_exhibit_id": content_id,
            "waveform_exhibit_id": waveform_id, "spectrogram_exhibit_id": spectrogram_id,
            "scored_spans": [list(span) for span in spans], "seconds_scored": coverage_seconds(spans),
            "segments": sorted(found, key=lambda s: (s["start_s"], s["end_s"])), **common.stage_exhibits(ctx, {"temporal"})}


# --- legacy voice fuser --------------------------------------------------------------------------------------------

def _voice_entries(ctx: StageContext) -> list[dict[str, Any]]:
    return [e for e in common.entries(ctx, ("detect",)) if e["status"] == "ok" and e["family"] == "voice" and e["aggregate"] is not None]


def voice_signal(ctx: StageContext, entries: list[dict[str, Any]], samples: int) -> Signal | None:
    """The fused synthetic-voice signal: each voice detector's mean window probability, weighted log-odds with the manifest
    `legacy_weight` (1.0 for a detector without a card)."""
    manifest = getattr(ctx.state.get("registry"), "manifest", None)
    per_model: dict[str, list[float]] = {}
    pairs: list[tuple[float, float]] = []
    for entry in entries:
        windows = _windows(entry)
        if not windows:
            continue
        card = manifest.detector(entry["detector_id"]) if manifest is not None else None
        weight = float(card.legacy_weight) if card is not None and card.legacy_weight is not None else 1.0
        per_model[entry["detector_id"]] = [round(p, 4) for p in windows]
        pairs.append((float(np.mean(windows)), weight))
    if not pairs:
        return None
    p = fuse_probability(pairs)
    n_windows = max(len(v) for v in per_model.values())
    spans = scored_voice_spans(entries)
    return Signal(
        id=VOICE_SIGNAL, name="Synthetic-voice detectors on the recording", category="model",
        observed=(f"fused p(synthetic)={p:.3f} over {n_windows} speech window(s) of ≤ {VOICE_WINDOW_S:g} s; per model: "
                  + ", ".join(f"{k}={np.mean(v):.2f}" for k, v in per_model.items())),
        reference="Anti-spoofing classifiers trained on spoofed vs bona fide speech; 0.5 decision boundary per window",
        weight=1.0, **prob_to_signal_fields(p),
        reasoning=("The voice carries spectral and prosodic artefacts that anti-spoofing models associate with text-to-speech or "
                   "voice-conversion output." if p >= 0.5 else
                   "The voice is classified as a bona fide recording by the anti-spoofing models."),
        details={"per_model_windows": per_model, "seconds_analysed": round(coverage_seconds(spans), 3),
                 "seconds_decoded": round(samples / SR, 3), "scored_spans": [list(span) for span in spans]})


def fuse(ctx: StageContext) -> dict[str, Any]:
    """Legacy voice fuser → uncalibrated fallback Decision (§4.4), `not_assessable` for music- or silence-dominated audio."""
    selection = common.select_fusion(ctx, "audio")
    content = ctx.outputs["condition"]["content"]
    facts = ctx.outputs["probe"]["facts"]
    samples = int((ctx.outputs.get("decode") or {}).get("samples") or 0)
    entries = _voice_entries(ctx)
    spans = scored_voice_spans(entries)
    signal = voice_signal(ctx, entries, samples)
    legacy_signals = [signal] if signal is not None else []
    verdict = legacy_fuse(legacy_signals, subject="recording", min_informative=1).model_dump(mode="json")
    detector_exhibits = [i for e in entries for i in e["exhibit_ids"]]
    # N-17 (QA P1-4): with one voice detector the signal is that detector's, so the fusion model's contribution reaches it
    sole = sole_voice_entry(entries)
    signals = [common.legacy_signal_v2(s.model_dump(mode="json"), n, entry=dict(sole) if sole is not None else None,
                                       fallback_detector=FUSER_ID, modality="audio", version=PIPELINE_VERSION, hypothesis=H2,
                                       family="voice", exhibit_ids=None if sole is not None else detector_exhibits)
               for n, s in enumerate(legacy_signals, 1)]
    vid.attach_temporal(ctx, signals)
    if content["assessable"]:
        decision = common.fallback_decision(verdict, informative=signal is not None, subject="recording")
    else:
        decision = Decision(code="not_assessable", verbal=abstain.NOT_ASSESSABLE_VERBAL, probability_ai_uncalibrated=None,
                            band="uncertain", confidence="low", reason=content["reason"], fusion_mode="fallback").model_dump(mode="json")
    caveats = [FALLBACK_CAVEAT, *verdict["caveats"]]
    view = {"verdict": common.legacy_verdict(verdict, decision), "signals": [s.model_dump(mode="json") for s in legacy_signals],
            "audio": {**facts, "seconds_analysed": round(coverage_seconds(spans), 3),
                      "seconds_decoded": round(samples / SR, 3), "scored_spans": [list(span) for span in spans],
                      "shares": content["shares"],
                      "assessable": content["assessable"], "not_assessable_reason": content["reason"]},
            "artifacts": [], "timings": {}}
    out = {"decision": decision, "assessments": common.LEGACY_ASSESSMENTS, "signals": signals, "caveats": caveats,
           "v1": common.json_safe(view), **common.stage_exhibits(ctx, {"fuse"})}
    if selection.mode == "artefact" and content["assessable"]:
        return common.artefact_fusion(ctx, "audio", selection, out, common.entries(ctx, ENTRY_STAGES))
    if selection.mode == "artefact":
        return _not_assessable_under_model(ctx, selection, out, content["reason"])
    return common.fallback_fusion(ctx, "audio", selection, out)


def _not_assessable_under_model(ctx: StageContext, selection: Any, out: dict[str, Any], reason: str) -> dict[str, Any]:
    """N-17 (browser QA P1-4): a recording with too little speech is not assessable while a fusion model is loaded. The result
    is that model's, with none of its checks applicable: no fallback caveat and no "no calibrated fusion model is loaded",
    which the fallback path printed for it. The checks of the model are entered as not applicable for the content reason,
    whatever windows they scored, so the model forms no estimate; documentary evidence decides as for any other result."""
    assert selection.fuser is not None
    held = list(selection.fuser.detectors)
    registry = ctx.state.get("registry")
    entries = [e for e in common.entries(ctx, ENTRY_STAGES) if e["detector_id"] not in held]
    entries += [{"detector_id": d, "name": (registry.detector_name(d) if registry is not None else None) or d, "version": "",
                 "hypotheses": [], "unit_scores": [], "metrics": {}, "provenance_override": None, "aggregate": None,
                 "status": "not_applicable", "reason": reason[:300]} for d in held]
    fused = common.artefact_fusion(ctx, "audio", selection, out, entries)
    if fused["decision"]["code"] != "not_assessable":
        return fused
    decision = {**fused["decision"], "reason": reason}
    # the model was validated on speech: a recording it cannot assess for its content is outside that, whatever its slice
    profile = {**fused["condition_profile"], "within_validated_envelope": False, "validation_support": "unsupported",
               "validation_measured_at": None}
    result = {**fused, "decision": decision, "condition_profile": profile}
    if isinstance(fused.get("v1"), dict) and "verdict" in fused["v1"]:
        result["v1"] = {**fused["v1"], "verdict": common.legacy_verdict(fused["v1"]["verdict"], decision)}
    return result


def synthesise(ctx: StageContext) -> dict[str, Any]:
    return common.synthesise_findings(ctx, kind="audio", entry_stages=ENTRY_STAGES, exhibit_stages=EXHIBIT_STAGES,
                                      family_by_v1_id={VOICE_SIGNAL: "A1_voice"})


def render(ctx: StageContext) -> dict[str, Any]:
    return common.index_artefacts(ctx, EXHIBIT_STAGES)


def publish(ctx: StageContext) -> dict[str, Any]:
    result: AnalysisResultV2 = common.build_result(ctx, kind="audio", pipeline_version=PIPELINE_VERSION, entry_stages=ENTRY_STAGES,
                                                   exhibit_stages=EXHIBIT_STAGES)
    return common.publish(ctx, result, dict(ctx.outputs["fuse"]["v1"]), pipeline_version=PIPELINE_VERSION, run_stages=ENTRY_STAGES)
