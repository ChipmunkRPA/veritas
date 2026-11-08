"""Provenance and metadata detectors as v2 plugins with the D-15 override strengths.

- **`prov.c2pa`** (I-2): offline validation against the bundled C2PA Trust List (`app/provenance/c2pa.py`).
  - An AI digital source type (IPTC tier `ai`) is decisive when the manifest is Trusted, self-declared when it is only
    valid. The legacy CAI interim list is reported, never trusted.
  - A generator named in the claim generator or a software agent of a valid manifest, without an AI source type, is
    self-declared.
  - A Trusted manifest declaring a capture source type whose hash binding matches the current bytes is a trusted-capture
    override.
  - Any other valid manifest (Trusted without a capture or AI declaration, or a signer not on the list) is weak evidence.
  - An Invalid manifest is modification after signing: informational, no override, its declarations are not relied on.
  - A manifest stored only at a remote URL is not fetched (offline) and carries no score.
- **`img.meta.generator`.** A generator named in structured metadata (SD parameters, Software, CreatorTool) or an IF §4.3
  M6 fingerprint (I-3) is self-declared, which floors s at 0.85 (D-15). Free-text mentions are weak.
- **`img.meta.exif`.** Capture-device EXIF with the v0.3 strengths, the EXIF table and the JPEG quality estimate.
- **`img.meta.consistency`** (I-3): ExifTool rules M1–M5 and M7 (`app/provenance/metadata_rules.py`) with the metadata tree,
  timestamps, edit history and thumbnail comparison exhibits. Contradicting fields score 0.75 (weak, uncalibrated);
  processing after capture is notable only.
- **`img.meta.qtables`** (I-4): the quantisation-table fingerprint against ExifTool JPEGDigest and the in-house reference
  set, and the make/model vs encoder rule (`app/provenance/qtables.py`). A software signature contradicting a covered
  camera model scores 0.75 (weak, uncalibrated); other outcomes are notable or info only.
- **`img.watermark.dwtdct`** (I-5): the public `dwtDct` messages of SDXL, the FLUX reference code and SD1
  (`app/provenance/watermarks.py`). A message at threshold is self-declared (D-15); no message is information only.
"""
from __future__ import annotations

import io

from PIL import Image

from app.detectors.image import jpeg_bitstream
from app.detectors.image.adapters.common import ALL_PROFILES, CPU_ONLY, H3, image_spec, original_bytes
from app.detectors.image.metadata import (
    GENERATOR_RE,
    PHRASE_LABEL,
    PLACE_PARAMETERS,
    PNG_TEXT_CHARS,
    PNG_TEXT_CHUNKS,
    SOURCE_TYPE_VALUES,
    camera_metadata,
    generator_markers,
    read_exif,
    read_png_text,
    xmp_packet,
)
from app.engine.detector import Detector, DetectorResult, InputUnit, RunContext, UnitScore
from app.provenance import c2pa as provenance
from app.provenance import metadata_rules, qtables, watermarks
from app.storage.files import untrusted_text

MAX_TABLE_ROWS = 80
C2PA_VERSION = "0.4.4"  # N-19p (SEC6-01): the titles of its thumbnails are filtered and cut to 200 characters; N-19o: N-19o (SEC5-01): the generator and the signer's name are cut to 200 characters in `observed`
METADATA_VERSION = "0.4.5"  # N-19p: 0.4.5 (R10-02, R10-03): the text of its tables keeps its words apart (tab and newline as a space); 0.4.4 (SEC6-02, R10-04): ExifTool reads a copy of a PNG through IEND, with one XMP-keyed chunk outside the bound, also where a chunk runs past the end, and reads the thumbnail from the same copy; the Pillow fallback reads PNG text through the bound and reads a PNG without EXIF; 0.4.3 (R10-01): the PNG keys of the M6 fingerprints are read wherever they stand, past the bound on PNG text; N-19o (SEC5-03, SEC5-02): the PNG text read is bounded in all, and the XMP history is read in one pass, each entry cut to 4 KB; N-19n (SEC4-01): the packet that the rules read is cut to `metadata.XMP_SCAN_BYTES`
GENERATOR_VERSION = "0.4.7"  # N-19p: 0.4.7 (R10-02, R10-03): text from the file keeps its words apart (tab and newline as a space); 0.4.6 (code review R10-01): the PNG keys that the check reads by name are read wherever they stand, past the bound on PNG text, and the result says how many text chunks the bound left unread (`png_text_unread`); N-19o: 0.4.5 (browser QA Q5-05): a mention in free text says in which fields it stands (`details.mentions`: "the EXIF image description", "the PNG text chunk “Comment”", each with what was found there); 0.4.4 (SEC5-01, SEC5-03): the PNG text read is bounded in all (64 chunks, 64 KB), and text from the file reaches no table or `observed` with a control or direction character; N-19n: 0.4.3 (SEC4-01): the reader of the IPTC source type is linear and bounded, and the packet is cut to `metadata.XMP_SCAN_BYTES` before it is scanned: a packet whose declaration stood past that limit was read before and is not read now; 0.4.2: the result says what was found and where (`details`); 0.4.1: a mention of AI in free text carries no weight in the fallback's score
EXIF_VERSION = "0.4.1"  # N-19p (R10-03): its exhibit "EXIF tags" is filtered (N-19o) and keeps the words of a tag apart; 0.4.0: I-1: JPEG facts from the bitstream (luma + chroma tables, subsampling); scores unchanged
C2PA_PRODUCES = ("table", "image")
NAME_ONLY_MARKERS = frozenset({"IPTC DigitalSourceType = trainedAlgorithmicMedia", "Generative-AI marker"})  # structural, not names


def _spec(det_id: str, name: str, *, produces: tuple[str, ...] = ("table",), version: str = "0.3.0"):
    return image_spec(det_id, name, "provenance", InputUnit.FILE, (H3,), produces, lane="cpu", devices=CPU_ONLY,
                      timeout_s=30.0, profiles=ALL_PROFILES, version=version)


def _generator_names(report: provenance.C2PAReport) -> list[str]:
    text = " ".join([report.claim_generator, *(a["software_agent"] for a in report.actions)])
    return [name for rx, name in GENERATOR_RE if name not in NAME_ONLY_MARKERS and rx.search(text)]


def _legacy_text(report: provenance.C2PAReport, legacy: provenance.TrustBundle) -> str:
    if report.state == "trusted":
        return "not consulted: Trusted under the C2PA Trust List"
    if report.legacy_trusted is None:
        return "not consulted"
    if report.legacy_trusted:
        return f"signer trusted under the {legacy.label}; reported separately, never treated as the trust list"
    return f"signer not on the {legacy.label} either"


def _c2pa_exhibits(ctx: RunContext, report: provenance.C2PAReport, official: provenance.TrustBundle,
                   legacy: provenance.TrustBundle) -> list[str]:
    signer = report.signer
    caption = (f"Validated offline with c2pa-rs {provenance.sdk_version()} against the {official.summary()}. "
               f"{provenance.REVOCATION_NOTE}")
    rows = [["state", provenance.STATE_LABELS[report.state]]]
    if report.state == "remote":
        rows += [["manifest URL", report.remote_url or "not stated"],
                 ["network", "not fetched: Veritas validates Content Credentials offline"]]
    else:
        rows += [["signer", signer.get("common_name") or "not stated"], ["issuer", signer.get("issuer") or "not stated"],
                 ["signed at", signer.get("time") or "not stated (no timestamp)"], ["algorithm", signer.get("alg") or "not stated"],
                 ["claim generator", report.claim_generator or "not stated"], ["title", report.title or "not stated"],
                 ["format", report.format],
                 ["digital source types", "; ".join(f"{s.term} ({s.tier}{', retired term' if s.retired else ''})"
                                                    for s in report.source_types) or "none declared"],
                 ["actions", ", ".join(a["action"] for a in report.actions) or "none"],
                 ["hash binding", report.binding_description],
                 ["legacy trust", _legacy_text(report, legacy)]]
    rows += [["trust list", official.summary()], ["revocation", provenance.REVOCATION_NOTE]]
    ids = [ctx.exhibits.table(["field", "value"], rows, title="C2PA Content Credentials", caption=caption)]
    if report.codes:
        ids.append(ctx.exhibits.table(["level", "code", "scope", "explanation"],
                                      [[c.level, c.code, c.scope, c.explanation] for c in report.codes],
                                      title="C2PA validation codes",
                                      caption="Status codes for the active manifest and the ingredient deltas, as reported by c2pa-rs."))
    if report.chain:
        ids.append(ctx.exhibits.table(
            ["depth", "title", "relationship", "format", "signer", "digital source types", "recorded validation failures", "manifest"],
            [[str(n.depth), n.title or "untitled", n.relationship, n.format or "not stated", n.signer or "not stated",
              ", ".join(n.source_types) or "none declared", str(n.failures), n.manifest or "no manifest"] for n in report.chain],
            title="C2PA ingredient chain",
            caption="The active manifest (depth 0) and its ingredients, followed recursively. Failures at depth 0 are this "
                    "validation's; for ingredients they are what the signer recorded when adding them (this validation's "
                    "ingredient deltas are in the codes table)."))
    for title, media_type, data in report.thumbnails:
        ids.append(ctx.exhibits.image(data, role="thumbnail", media_type=media_type, title=f"C2PA {title}", media_ref="original",
                                      caption="Embedded in the manifest by the signer: it shows the signed state, not necessarily "
                                              "the current pixels. Shown re-encoded, without the metadata of the embedded file."))
    return ids


def c2pa_result(ctx: RunContext, data: bytes | memoryview) -> DetectorResult:
    """Offline C2PA validation as a detector result, shared by the image and video detectors (D-15; IF §4.4)."""
    if not provenance.available():
        return DetectorResult.skipped("c2pa-python is not installed")
    mime = provenance.sniff_mime(data)
    if mime is None:
        container = provenance.unsupported_container(data)
        if container is not None:
            label = "WebM" if container == "webm" else "Matroska"
            return DetectorResult(status="not_applicable",
                                  reason=f"Content Credentials are not supported for this {label} container; they were not checked",
                                  metrics={"unsupported_container": 1.0, f"container_{container}": 1.0})
        return DetectorResult.failed("the file container is not one C2PA credentials can be read from")
    try:
        official, legacy = provenance.load_bundle(provenance.OFFICIAL), provenance.load_bundle(provenance.LEGACY)
    except provenance.TrustBundleError as e:
        return DetectorResult.failed(f"the bundled trust list could not be verified: {e}")
    report = provenance.read(data, mime, bundle=official, legacy=legacy)
    if report.state == "absent":
        return DetectorResult.not_applicable("no C2PA manifest")
    if report.state == "unreadable":
        return DetectorResult.failed("the C2PA manifest could not be read")
    state = report.state
    trusted = state == "trusted"
    metrics = {"trusted": float(trusted), "valid": float(state in ("trusted", "valid")), "invalid": float(state == "invalid"),
               "remote_reference": float(state == "remote"), "legacy_trusted": float(bool(report.legacy_trusted)),
               "ai_declared": float(report.ai_declared), "capture_declared": float(report.capture_declared),
               "covers_bytes": float(report.covers_bytes), "failure_codes": float(len(report.failure_codes)),
               "binding_matched": float(report.binding_status == "matched"),
               "binding_mismatch": float(report.binding_status == "mismatch"),
               "binding_not_verified": float(report.binding_status == "not_verified"),
               "signing_certificate_expired": float("signingCredential.expired" in report.failure_codes),
               "signer_untrusted": float("signingCredential.untrusted" in report.failure_codes),
               "ingredients": float(max(len(report.chain) - 1, 0))}
    exhibit_ids = _c2pa_exhibits(ctx, report, official, legacy)
    if state == "remote":
        return DetectorResult(
            status="ok", metrics=metrics, exhibit_ids=exhibit_ids,
            observed="remote manifest reference" + (f" {report.remote_url}" if report.remote_url else "") + "; not fetched",
            reference="Veritas validates Content Credentials offline and never fetches remote manifests",
            reasoning="The file points to Content Credentials stored at a remote address. They were not fetched, so they were "
                      "not assessed.")
    terms = [s.term for s in report.source_types]
    # N-19o (SEC5-01): the generator and the signer's name are the manifest's text; each is cut to 200 characters
    observed = (f"state={state}; signer='{untrusted_text(report.signer.get('common_name', ''))[:200]}'; "
                f"claim_generator='{untrusted_text(report.claim_generator)[:200]}'; "
                f"digitalSourceType={terms}")
    names = _generator_names(report)
    override = None
    if state == "invalid":
        aggregate = UnitScore("global", 0.0, "score", quality={"invalid": 1.0})
        reference = ("Certificate dates, signer trust, signature validity and content binding are separate checks. "
                     "A failure does not establish AI generation or originality.")
        failures = set(report.failure_codes)
        issues = []
        if "signingCredential.expired" in failures:
            issues.append("the signing certificate is expired")
        if "signingCredential.untrusted" in failures:
            issues.append("the signer is not trusted by the configured trust list")
        if report.binding_status == "mismatch":
            issues.append("the file content does not match the signed content hash")
        if not issues:
            issues.append("one or more signature or credential checks failed")
        reasoning = ("Content Credentials are present but are not accepted: " + "; ".join(issues) + ". "
                     "Content binding " + report.binding_description + ". "
                     "The declaration is not relied on to establish AI involvement; further detection continues.")
    elif report.ai_declared or names:
        override = ("decisive_ai" if trusted and report.ai_declared and report.covers_bytes
                    and not report.failure_codes else "self_declared_ai")
        aggregate = UnitScore("global", 1.0, "score", quality={"trusted": float(trusted), "valid": 1.0})
        declared = ", ".join(s.term for s in report.source_types if s.tier == "ai") or ("generator named: " + ", ".join(names))
        reference = "Camera and editor manifests carry capture or edit source types, never a generative-AI source type"
        if override == "decisive_ai":
            qualifier = " The manifest validates against the C2PA Trust List, so this is documentary evidence."
        elif trusted:
            qualifier = (" The manifest validates against the C2PA Trust List but declares no generative-AI source type, so the "
                         "named generator counts as self-declared.")
        else:
            qualifier = " The signer is not on the C2PA Trust List, so the declaration counts as self-declared."
        reasoning = f"The file's C2PA manifest declares generative AI ({declared}).{qualifier}"
    elif trusted and report.capture_declared and report.covers_bytes:
        override = "trusted_capture"
        aggregate = UnitScore("global", 1.0, "score", quality={"trusted": 1.0, "valid": 1.0})
        reference = "A trusted capture manifest whose hash binding matches binds the signed capture to these bytes"
        reasoning = (f"A C2PA manifest that validates against the C2PA Trust List declares a capture ({', '.join(terms)}), and "
                     "its hash binding matches the current bytes.")
    else:
        aggregate = UnitScore("global", 0.35, "probability", quality={"valid": 1.0, "outcome_weight": 0.5})
        if trusted:
            reference = "A trusted manifest without a capture or AI declaration records what the signer did, not how the content began"
            reasoning = ("A C2PA manifest validates against the C2PA Trust List but declares neither a capture of these bytes nor "
                         "generative AI" + (f" (source types: {', '.join(terms)})." if terms else "."))
        else:
            reference = "A valid manifest whose signer is not on the C2PA Trust List is weak evidence"
            reasoning = ("A C2PA manifest is intact and declares no generative AI, but its signer is not on the C2PA Trust List"
                         + ("; the signer is on the legacy CAI interim list, which is reported separately." if report.legacy_trusted
                            else "."))
    return DetectorResult(status="ok", unit_scores=[aggregate], aggregate=aggregate, observed=observed, reference=reference,
                          reasoning=reasoning, metrics=metrics, exhibit_ids=exhibit_ids, provenance_override=override)


class C2PADetector(Detector):
    spec = _spec("prov.c2pa", "C2PA Content Credentials", produces=C2PA_PRODUCES, version=C2PA_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        return c2pa_result(ctx, original_bytes(ctx))


class GeneratorMetadataDetector(Detector):
    spec = _spec("img.meta.generator", "Generator markers in metadata (EXIF, XMP, PNG text)", version=GENERATOR_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        data = original_bytes(ctx)
        with Image.open(io.BytesIO(data)) as im:
            exif, png_text = read_exif(im), read_png_text(im)
            fingerprints = metadata_rules.generator_fingerprints(png_text, exif.get("Make"), exif.get("Software"), im)
        # N-19n (Q4-01): the packet of a PNG is read from its chunk where it is compressed (`xmp_packet`)
        markers = generator_markers(exif, png_text, xmp_packet(data))
        # N-19p (R10-01): the text chunks that the bound left unread, said in the result where there are any
        unread = getattr(png_text, "unread", 0)
        unread_note = (f" {unread} of the file's PNG text chunks were not read: past the first {PNG_TEXT_CHUNKS} chunks or "
                       f"{PNG_TEXT_CHARS // 1024} KB of text the check reads only the keys it looks for by name.") if unread else ""
        unread_metric = {"png_text_unread": float(unread)} if unread else {}
        source_type = markers["source_type"]  # N-19n: the IPTC digital source type, where it declares AI generation
        if fingerprints or markers["structured_hits"] or markers["sd_params"] or source_type:
            hits = list(dict.fromkeys([name for name, _ in fingerprints] + markers["structured_hits"]))
            where = list(dict.fromkeys([field for _, field in fingerprints] + markers["where"]))
            declared = "; ".join(f"{name} in {field}" for name, field in fingerprints)
            # a file whose marker is the source type alone names no generator (the native Midjourney packets hold an
            # identifier and the source type): the marker's kind and its words follow (`decisive.marker_kind`)
            # N-19n (D3-07, 9): nor does the phrase "AI generated" in a field that software writes, nor do generation
            # parameters without a tool. A generator that a structured field names, or whose form the file has, does
            # A label that is no name of a tool (the phrase; the term of the source type in a field that holds no value of
            # the property, as Software = "trainedAlgorithmicMedia") names none either
            tools = [name for name in markers["structured_hits"] if name not in NAME_ONLY_MARKERS]
            phrases = [name for name in markers["structured_hits"] if name in NAME_ONLY_MARKERS]
            phrase = bool(phrases)
            held = "a phrase such as “AI-generated”" if PHRASE_LABEL in phrases else "the term trainedAlgorithmicMedia"
            names_generator = bool(fingerprints or tools)
            table = ctx.exhibits.table(["field", "value"], [["locations", "; ".join(where) or "structured field"],
                                                            ["named tools", "; ".join(hits) or ("generation parameters" if markers["sd_params"] else "none")],
                                                            ["fingerprints (M6)", declared or "none"],
                                                            ["IPTC digital source type", source_type or "none that declares AI generation"],
                                                            ["PNG text keys", ", ".join(list(png_text)[:10])]],
                                       title="Generator markers in metadata")
            aggregate = UnitScore("global", 1.0, "score", quality={"structured": 1.0})
            # N-19n (D3-07, 5): what the check found and where, each apart, for the finding's sentences: the tools that
            # structured fields name (Veritas's labels for them), the labels that name no tool, the fingerprints with their
            # fields, the labels of each group of places, the place of generation parameters, and the source type's value
            by_generator: dict[str, list[str]] = {}
            for name, field in fingerprints:
                by_generator.setdefault(name, []).append(field)
            details = {"tools": tools, "phrases": phrases,
                       "fingerprints": [{"generator": name, "fields": fields} for name, fields in by_generator.items()],
                       "places": [{"place": place, "labels": found} for place, found in markers["places"].items()],
                       "parameters": PLACE_PARAMETERS if markers["sd_params"] else None, "source_type": source_type}
            return DetectorResult(
                status="ok", unit_scores=[aggregate], aggregate=aggregate,
                observed=(declared or "; ".join(hits)) if names_generator
                else f"IPTC digital source type {source_type}" if source_type
                else f"{held} in a field that software writes" if phrase
                else "Stable-Diffusion style generation parameters",
                reference="Camera and editor files name a camera or an editor in these fields, never a generator",
                reasoning=("The file's structured metadata names a generative-AI tool or carries its generation parameters ("
                           + "; ".join(where or ["structured field"]) + ")." if names_generator else
                           f"The file's metadata declares AI generation in its IPTC digital source type ({source_type})." if source_type else
                           f"The file's metadata declares AI generation: a field that software writes holds {held} ("
                           + "; ".join(where or ["structured field"]) + ")." if phrase else
                           "The file's metadata holds generation parameters and names no tool (" + "; ".join(where or ["structured field"]) + ").")
                          + " Metadata can be added or edited by anyone, so this is self-declared evidence." + unread_note,
                metrics={**unread_metric, "structured_hits": float(len(hits)), "sd_params": float(markers["sd_params"]),
                         "fingerprints": float(len(fingerprints)), "names_generator": float(names_generator),
                         "source_type_trained": float(source_type == SOURCE_TYPE_VALUES[0]),
                         "source_type_composite": float(source_type == SOURCE_TYPE_VALUES[1]),
                         "phrase_declared": float(phrase)},
                details=details, exhibit_ids=[table], provenance_override="self_declared_ai")
        if markers["free_hits"]:
            # N-19n (browser QA, the read of the first part): a mention in free text takes no part in the fallback's score.
            # It was averaged in at half weight, so a caption that says "Not AI-generated." raised the score (three checks at
            # 20 gave 25), and the finding's "it sets no number in this result" was false there
            aggregate = UnitScore("global", 0.65, "probability", quality={"outcome_weight": 0.0})
            return DetectorResult(
                status="ok", unit_scores=[aggregate], aggregate=aggregate, observed="; ".join(markers["free_hits"]),
                reference="Captions and comments are user-editable and may mention AI tools without the image being synthetic",
                reasoning="A caption, description or comment field mentions a generative-AI tool or an 'AI generated' phrase. This "
                          "is a user-written hint, not a tool signature, so it is weak evidence only." + unread_note,
                metrics={"free_text_hits": float(len(markers["free_hits"])), **unread_metric},
                # N-19o (Q5-05): where each mention stands, for the finding's first sentence; a PNG key is the file's text
                details={"mentions": [{**m, "field": untrusted_text(m["field"])} for m in markers["mentions"]]})
        return DetectorResult(status="ok", observed="no generator markers in EXIF, XMP or PNG text chunks",
                              reference="Generators and editing tools often write their name into these fields",
                              reasoning="No generative-AI tool is named in the file's metadata; absence of markers is not evidence of "
                                        "authenticity." + unread_note, metrics={"structured_hits": 0.0, "free_text_hits": 0.0, **unread_metric})


def jpeg_metrics(jpeg: jpeg_bitstream.JpegStructure) -> dict[str, float]:
    out = {"jpeg_progressive": float("progressive" in jpeg.frame), "jpeg_components": float(len(jpeg.components))}
    for key, value in (("jpeg_estimated_quality", jpeg.quality_luma), ("jpeg_quality_chroma", jpeg.quality_chroma),
                       ("jpeg_ijg_quality", jpeg.ijg_quality)):
        if value is not None:
            out[key] = float(value)
    return out


def jpeg_rows(jpeg: jpeg_bitstream.JpegStructure) -> list[list[str]]:
    """Field/value rows for the JPEG structure table: frame, sampling, estimates, then the luma and chroma tables."""
    def estimate(value: int | None) -> str:
        return "not available" if value is None else str(value)

    rows = [["frame", f"{jpeg.frame}, {jpeg.precision}-bit samples"], ["dimensions", f"{jpeg.width} × {jpeg.height} px"],
            ["subsampling", f"{jpeg.subsampling} (sampling factors {jpeg.sampling_factors})"],
            ["quality estimate, luma", estimate(jpeg.quality_luma)], ["quality estimate, chroma", estimate(jpeg.quality_chroma)],
            ["standard libjpeg tables", f"equal at quality {jpeg.ijg_quality}" if jpeg.ijg_quality is not None
             else "not equal to any libjpeg quality"],
            ["table fingerprint (sha256, first 16)", jpeg.fingerprint[:16]]]
    for index, label in ((0, "luma"), (1, "chroma")):
        table = jpeg.table(index)
        if table is None:
            continue
        table_id = jpeg.components[index].table_id
        rows += [[f"{label} table {table_id} ({jpeg.table_bits[table_id]}-bit), row {r + 1}", " ".join(map(str, table[8 * r:8 * r + 8]))]
                 for r in range(8)]
    return rows


class ExifDetector(Detector):
    spec = _spec("img.meta.exif", "Capture-device metadata (EXIF)", version=EXIF_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        data = original_bytes(ctx)
        with Image.open(io.BytesIO(data)) as im:
            exif = read_exif(im)
        cam = camera_metadata(exif)
        metrics = {"exif_tags": float(len(exif))}
        exhibit_ids = []
        if exif:
            rows = [[str(k), str(v)] for k, v in sorted(exif.items())][:MAX_TABLE_ROWS]
            exhibit_ids.append(ctx.exhibits.table(["tag", "value"], rows, title="EXIF tags",
                                                  caption="EXIF is trivially forgeable and is often stripped by platforms"))
        jpeg = jpeg_bitstream.read(data)
        if jpeg is not None:
            metrics |= jpeg_metrics(jpeg)
            exhibit_ids.append(ctx.exhibits.table(["field", "value"], jpeg_rows(jpeg), title="JPEG structure",
                                                  caption="Read from the file's markers before the first scan. Tables are in "
                                                          "natural order; the quality estimates assume libjpeg scaling."))
        make, model, software = cam["make"], cam["model"], cam["software"]
        if make or model:
            coherent, edited = cam["coherent"], cam["edited"]
            aggregate = UnitScore("global", 0.45 if edited else (0.22 if coherent else 0.30), "probability",
                                  quality={"coherent": float(coherent), "edited": float(edited), "outcome_weight": 0.9 if coherent else 0.5})
            exposure = exif.get("ExposureTime", "")
            iso = exif.get("ISOSpeedRatings", exif.get("PhotographicSensitivity", ""))
            return DetectorResult(
                status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=exhibit_ids,
                observed=f"Make='{make}', Model='{model}', Software='{software}'"
                         + (f", Exposure={exposure}, ISO={iso}" if exposure else ""),
                reference="Generated images have no capture device; EXIF is, however, trivially forgeable",
                reasoning="Capture-device metadata is consistent with a camera original"
                          + (": make, model, exposure and ISO are all present and no editing software is recorded, which generators "
                             "and most re-encoders do not produce." if coherent else ".")
                          + (f" The Software tag shows post-processing in '{software}', so the file was edited after capture."
                             if edited else ""))
        if cam["editor_only"]:
            aggregate = UnitScore("global", 0.5, "probability", quality={"outcome_weight": 0.3})
            return DetectorResult(
                status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=exhibit_ids,
                observed=f"Software='{software}'", reference="Originals carry Make/Model; exported edits often do not",
                reasoning="The file was saved by an editing application and no capture-device metadata remains; this indicates "
                          "processing but not its nature.")
        return DetectorResult(status="ok", metrics=metrics, exhibit_ids=exhibit_ids,
                              observed="No EXIF Make/Model and no editing-software tag",
                              reference="Social-media and messaging apps strip metadata; generators usually write none",
                              reasoning="Absence of metadata is uninformative on its own: it is equally consistent with a re-shared "
                                        "photograph and with a synthetic image.")


ANOMALY_SCORE = 0.75  # IF §6.4 classical-heuristic cap; hand-set and uncalibrated (legacy_weight 0 on the fallback path)


class MetadataConsistencyDetector(Detector):
    spec = _spec("img.meta.consistency", "Metadata consistency (ExifTool rules M1–M5, M7)", produces=("table", "image"),
                 version=METADATA_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        analysis = metadata_rules.analyse(original_bytes(ctx))
        if not analysis.readable:
            return DetectorResult.failed("the file's metadata could not be read")
        rows, total = metadata_rules.tree_rows(analysis)
        shown = f"all {total} fields" if len(rows) == total else f"{len(rows)} of {total} fields, flagged fields first"
        exhibit_ids = [ctx.exhibits.table(metadata_rules.TREE_COLUMNS, rows, title="Metadata tree",
                                          caption=f"Read with {analysis.read.describe()}; {shown}. Severities and rule codes come "
                                                  "from rules M1–M7; an anomaly is a field that contradicts another.")]
        stamps = metadata_rules.timestamp_rows(analysis)
        if stamps:
            exhibit_ids.append(ctx.exhibits.table(metadata_rules.TIMESTAMP_COLUMNS, stamps, title="Metadata timestamps",
                                                  caption="Every date field, sorted by the time as written. A time without a UTC "
                                                          "offset is a wall-clock time and is compared as written."))
        if analysis.history:
            exhibit_ids.append(ctx.exhibits.table(metadata_rules.HISTORY_COLUMNS, metadata_rules.history_rows(analysis),
                                                  title="Edit history (xmpMM:History)",
                                                  caption="In document order, as recorded by the editing software. Editing is not "
                                                          "deception."))
        comparison = analysis.thumbnail
        if comparison is not None and comparison.composite_png is not None:
            turned = f", rotated {comparison.rotation}°" if comparison.rotation else ""
            exhibit_ids.append(ctx.exhibits.image(
                comparison.composite_png, role="comparison", media_type="image/png", title="Embedded thumbnail vs main image",
                media_ref="original",
                caption=f"Left: the embedded thumbnail ({comparison.width}×{comparison.height}, dark bands removed). Centre: the main "
                        f"image resized to the same size{turned}. Right: absolute difference ×4. SSIM {comparison.ssim:.2f}."))
        anomalies, notable = analysis.anomalies, analysis.notable
        metrics = {"tags": float(total), "anomalies": float(len(anomalies)), "notable": float(len(notable)),
                   "exiftool": float(analysis.read.source == "exiftool"),
                   "platform_pattern": float(any(h.code == "M7.platform_pattern" for h in analysis.hits))}
        if comparison is not None and comparison.ssim is not None:
            metrics["thumbnail_ssim"] = comparison.ssim
        reference = "The date fields, maker notes, embedded thumbnail and GPS block of an unedited camera file agree with each other"
        if anomalies:
            rules = sorted({h.rule for h in anomalies})
            aggregate = UnitScore("global", ANOMALY_SCORE, "probability", quality={})
            return DetectorResult(
                status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=exhibit_ids,
                observed="; ".join(h.observed for h in anomalies[:4]), reference=reference,
                reasoning="The file's metadata contradicts itself (" + ", ".join(f"{r} {metadata_rules.RULE_NAMES[r]}" for r in rules)
                          + "). This indicates that the file or its metadata changed after capture; it does not show what changed. "
                          "Some devices and apps write inconsistent fields, and anyone can edit metadata.")
        if notable:
            return DetectorResult(status="ok", metrics=metrics, exhibit_ids=exhibit_ids, reference=reference,
                                  observed="; ".join(h.note for h in notable[:4]),
                                  reasoning="The metadata records processing after capture. Editing is not deception, so this is "
                                            "recorded for review and does not count either way.")
        return DetectorResult(status="ok", metrics=metrics, exhibit_ids=exhibit_ids, reference=reference,
                              observed="no inconsistencies among the checked fields",
                              reasoning="The checked metadata fields agree with each other. Any software can write consistent "
                                        "metadata, so agreement is not evidence that the image is unedited.")


QTABLES_VERSION = "0.4.0"
TABLE_SEVERITY = {"info": 0.0, "notable": 1.0, "anomaly": 2.0}


class QuantisationTableDetector(Detector):
    spec = _spec("img.meta.qtables", "Quantisation-table fingerprint (make and model vs encoder)", version=QTABLES_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        data = original_bytes(ctx)
        structure = jpeg_bitstream.read(data)
        luma = structure.table(0) if structure is not None else None
        if structure is None or luma is None:
            return DetectorResult(status="not_applicable", reason="not a JPEG file with quantisation tables")
        with Image.open(io.BytesIO(data)) as im:
            cam = camera_metadata(read_exif(im))
        digest, db = qtables.jpeg_digest(data), qtables.load_db()
        outcome = qtables.assess(structure, cam["make"], cam["model"], digest, db)
        fingerprint, built = structure.fingerprint, db.built_from
        digest_text = (digest.name or (f"unknown tables (MD5 {digest.md5})" if digest.md5 else None)
                       or ("not reported" if digest.available else "not available (ExifTool not found)"))
        rows = [["claimed camera", outcome.claim or "no Make and Model"], ["ExifTool JPEGDigest", digest_text],
                ["standard libjpeg tables", f"equal at quality {structure.ijg_quality}" if structure.ijg_quality is not None else "no"],
                ["table fingerprint (sha256, first 16)", fingerprint[:16]],
                ["reference files for the claimed model", str(outcome.reference_files)],
                ["reference models using these tables", ", ".join(db.models_with(fingerprint)) or "none"],
                ["reference set", f"{built.get('files', 0)} camera originals of {built.get('models', 0)} models (vb "
                                  f"{'/'.join(built.get('strata', []))}, manifest {str(built.get('manifest_sha256', ''))[:12]})"],
                ["outcome", f"{outcome.severity} ({outcome.code}): {outcome.note}"]]
        exhibit_ids = [ctx.exhibits.table(["field", "value"], rows, title="Quantisation-table signature",
                                          caption="The tables show which encoder wrote the file. Equal tables identify the "
                                                  "encoder, not the content.")]
        if outcome.reference_fingerprint in db.tables:
            reference = db.tables[outcome.reference_fingerprint][0]
            label = f"the most common table of the {outcome.reference_files} reference files from {outcome.claim}"
        elif structure.quality_luma is not None:
            reference = [int(v) for v in metadata_rules.ijg_table(metadata_rules.STD_LUMA, structure.quality_luma)]
            label = f"the standard libjpeg luma table at quality {structure.quality_luma}"
        exhibit_ids.append(ctx.exhibits.table(["row", *map(str, range(8))], qtables.diff_rows(luma, reference),
                                              title="Luma quantisation table vs reference",
                                              caption=f"Each cell: observed step / reference step (ratio), in natural order; the "
                                                      f"reference is {label}."))
        metrics = {"qtable_severity": TABLE_SEVERITY[outcome.severity], "reference_files": float(outcome.reference_files),
                   "software_signature": float(outcome.signature is not None), "exiftool_digest": float(digest.available)}
        expected = "Camera originals from a model carry that model's firmware tables; applications write their own"
        if outcome.severity == "anomaly":
            aggregate = UnitScore("global", ANOMALY_SCORE, "probability", quality={})
            return DetectorResult(status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=exhibit_ids,
                                  observed=outcome.observed, reference=expected,
                                  reasoning=f"{outcome.note}. This shows how the file was saved, not what changed.")
        suffix = "." if outcome.severity == "info" else ". Editing is not deception, so this is recorded for review and does not count either way."
        return DetectorResult(status="ok", metrics=metrics, exhibit_ids=exhibit_ids, observed=outcome.observed, reference=expected,
                              reasoning=outcome.note + suffix)


WATERMARK_VERSION = "0.4.1"  # 0.4.1: chances in the table as natural frequencies (J-8)
WATERMARK_REFERENCE = "An image that never passed through these pipelines agrees with each message in about half of its bits"


def chance_words(p: float) -> str:
    """A chance as a natural frequency (DR P1-8: no "8.4e-01"): "about 8 in 10", "about 1 in 2,500", "below 1 in a billion"."""
    if p >= 0.095:
        return f"about {max(1, round(p * 10))} in 10"
    if p < 1e-9:
        return "below 1 in a billion"
    n = 1 / p
    nice = float(f"{n:.2g}")
    return f"about 1 in {nice:,.0f}"


class OpenWatermarkDetector(Detector):
    spec = _spec("img.watermark.dwtdct", "Public watermark messages (dwtDct, SDXL, FLUX reference, SD1)", version=WATERMARK_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        rgb = watermarks.read_rgb(original_bytes(ctx))
        if rgb is None:
            return DetectorResult(status="not_applicable", reason="the file could not be decoded as an image")
        readings = watermarks.scan(rgb)
        if readings is None:
            return DetectorResult(status="not_applicable", reason="smaller than 256 × 256 pixels, the decoder's minimum")
        rows = [[r.message.name, str(r.n_bits), f"{r.equal_bits} of {r.n_bits}", f"at least {r.threshold}", chance_words(r.chance),
                 "present" if r.present else "not present"] for r in readings]
        table = ctx.exhibits.table(["message", "bits", "equal bits", "threshold", "chance for an unmarked image", "outcome"], rows,
                                   title="Public watermark messages (dwtDct)",
                                   caption="Bits read from the stored pixels and compared with each public message. Each threshold "
                                           "keeps the chance that an unmarked image reaches it below 1 in a billion.")
        metrics = watermarks.metrics(readings)
        present = [r for r in readings if r.present]
        if not present:
            closest = max(readings, key=lambda r: r.equal_bits / r.n_bits)
            return DetectorResult(
                status="ok", metrics=metrics, exhibit_ids=[table], reference=WATERMARK_REFERENCE,
                observed=f"no public watermark message at threshold (closest: {closest.message.name}, {closest.equal_bits} of "
                         f"{closest.n_bits} bits)",
                reasoning="None of the three public messages is present. Most generators embed no such message, and resizing or "
                          "cropping removes it, so this is not evidence either way.")
        names = "; ".join(f"{r.message.name}, {r.equal_bits} of {r.n_bits} bits" for r in present)
        aggregate = UnitScore("global", 1.0, "score", quality={"structured": 1.0})
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=[table],
            reference=WATERMARK_REFERENCE, observed=f"public watermark message: {names}",
            reasoning="The pixels carry a fixed public message that open-source image generation code embeds in its output. The "
                      "message is public and unkeyed, so anyone can embed it in any image: this is self-declared evidence.",
            provenance_override="self_declared_ai")
