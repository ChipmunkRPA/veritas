"""Provenance and metadata evidence: C2PA Content Credentials, EXIF/XMP, PNG text chunks.

Documentary evidence is treated differently from statistical evidence:
- A C2PA manifest whose claim generator / actions declare generative AI is *decisive* (the file
  itself attests to its synthetic origin).
- Stable Diffusion / ComfyUI / Midjourney / DALL·E markers embedded in PNG text or XMP are
  *decisive* unless contradicted (they are written by the generator at save time).
- Absence of metadata is *neutral* (social platforms strip it); presence of coherent camera EXIF
  is *weak* support for authenticity (trivially forgeable).
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import struct
import zlib
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

from app.core.evidence import Signal

# Generator identifiers. Matched with word boundaries; a hit in a STRUCTURED field (Software, CreatorTool,
# claim_generator, DigitalSourceType, SD parameters) is decisive; a hit in FREE TEXT (captions, comments) is weak.
GENERATOR_PATTERNS = [
    (r"\bstable[\s_-]?diffusion\b|\bsd(?:xl)?[\s_-]?(?:1\.5|2\.1|3(?:\.5)?|turbo)\b|\bstability\s*ai\b|\bcomfyui\b|\bautomatic1111\b|\binvokeai\b|\bfooocus\b|\bsd[\s_-]?webui\b", "Stable Diffusion / ComfyUI"),
    (r"\bmidjourney\b", "Midjourney"),
    (r"\bdall[\s·\-]?e\b(?:\s*[23])?|\bgpt-?image\b|\bopenai\b", "DALL·E / OpenAI"),
    (r"\badobe\s+firefly\b", "Adobe Firefly"),
    (r"\bgoogle\s+imagen\b|\bimagen\s*[2-9]\b|\bgemini\s+(?:image|nano|flash\s+image)\b|\bsynthid\b", "Google Imagen / Gemini"),
    (r"\bflux\.?1\b|\bblack\s*forest\s*labs\b", "FLUX"),
    (r"\bleonardo\.?ai\b|\bideogram\b|\bplayground\s*ai\b|\bnightcafe\b|\bcraiyon\b|\bbing\s+image\s+creator\b|\bmicrosoft\s+designer\b", "Other generative tool"),
    (r"trainedalgorithmicmedia|compositewithtrainedalgorithmicmedia", "IPTC DigitalSourceType = trainedAlgorithmicMedia"),
    (r"c2pa\.ai_generative_training|\bgenerative\s*ai\b|\bai[\s_-]generated\b", "Generative-AI marker"),
]
STRUCTURED_EXIF = ("Software", "ProcessingSoftware", "Make", "Model", "HostComputer")
# N-19n (browser QA Q4-01; D-15 as amended on 2026-10-05): the IPTC source type is a structured field by its property,
# under whatever prefix the packet gives the namespace. The pattern named it under `Iptc4xmpExt` only, so the native
# Midjourney PNG, whose packet writes `iptcExt:DigitalSourceType` (the prefix of the Exiv2 library), was read as free text
IPTC_EXT_NAMESPACE = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
SOURCE_TYPE_PREFIXES = ("Iptc4xmpExt", "iptcExt")  # the specification's prefix, and the one that Exiv2 writes
_STRUCTURED_XMP_NAMES = ("xmp:CreatorTool", "tiff:Software", "photoshop:Credit", "plus:ModelReleaseStatus", "dc:creator", "xmpMM:History",
                         "stEvt:softwareAgent", "exif:Software")
_NAMESPACE_PREFIX = re.compile(r"xmlns:([A-Za-z_][\w.-]{0,63})\s*=\s*[\"']" + re.escape(IPTC_EXT_NAMESPACE.rstrip("/")) + r"/?[\"']")
SOURCE_TYPE_VALUES = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")  # the two that declare AI generation
SOURCE_TYPE_LABEL = "IPTC DigitalSourceType = trainedAlgorithmicMedia"
PHRASE_LABEL = "Generative-AI marker"  # the pattern of "AI generated", "generative AI": a phrase, which names no tool
XMP_PNG_KEY = "XML:com.adobe.xmp"
# N-19n (SEC4-01, P1; the security re-review of bc49948c): a packet is the uploader's, and it is cut to this before anything
# scans it, as the document reader caps its input (N-10). The uncompressed chunk of a PNG and the packet among a file's raw
# bytes had no bound (an inflated chunk had 4 MB), so the cost of every reader below grew with the upload. A real packet is
# under 64 KB (the packet of a JPEG stands in one segment of at most 65,502 bytes); a longer one is read up to the limit
XMP_SCAN_BYTES = 64 * 1024
# The value of the IPTC source type is a short NewsCode address: an element's content is read up to this length, the white
# space around the value included. The pattern read it as `>\s*([^<]*?)\s*<`, three quantifiers over one run: where no "<"
# followed, the match tried on the order of n³ splits of the run before it failed (5 s at 3,200 spaces; a pure-Python
# regex cannot be pre-empted, so the detector's thread kept a core after its watchdog returned)
SOURCE_TYPE_VALUE_CHARS = 300
MAX_DECLARED_PREFIXES = 4  # a packet declares one prefix for the namespace; each one more is a name the patterns try at every position


def source_type_names(xmp: str) -> list[str]:
    """`<prefix>:DigitalSourceType` for the prefixes of the IPTC Extension namespace: the two in common use and those the
    packet declares for it."""
    declared = list(dict.fromkeys(m.group(1) for m in _NAMESPACE_PREFIX.finditer(xmp)))[:MAX_DECLARED_PREFIXES]
    return [f"{prefix}:DigitalSourceType" for prefix in dict.fromkeys([*SOURCE_TYPE_PREFIXES, *declared])]


def structured_xmp(xmp: str) -> re.Pattern[str]:
    """The structured fields of an XMP packet, each with what follows it up to the next tag (300 characters at most)."""
    names = "|".join(re.escape(name) for name in (*_STRUCTURED_XMP_NAMES, *source_type_names(xmp)))
    return re.compile(rf"(?:{names})[^<\n]{{0,300}}", re.IGNORECASE)


STRUCTURED_XMP = structured_xmp("")  # the pattern of a packet that declares no prefix of its own


def source_type(xmp: str) -> str | None:
    """The file's IPTC digital source type where it declares AI generation: `trainedAlgorithmicMedia` or
    `compositeWithTrainedAlgorithmicMedia`, as an attribute, an element or an `rdf:resource` of the property; None for any
    other value (`digitalCapture`) and without the property. The value is the last part of the IPTC NewsCodes address.
    Linear in the length of the packet (SEC4-01): the element's content is one bounded run, trimmed here and not by the
    pattern, so a property name that no "<" follows fails after `SOURCE_TYPE_VALUE_CHARS` characters."""
    names = "|".join(re.escape(name) for name in source_type_names(xmp))
    values = re.finditer(rf"(?:{names})(?:\s+rdf:resource)?\s*(?:=\s*([\"'])(.*?)\1|>([^<]{{0,{SOURCE_TYPE_VALUE_CHARS}}})<)", xmp,
                         re.IGNORECASE | re.DOTALL)
    found = {(m.group(2) if m.group(2) is not None else (m.group(3) or "").strip()).rstrip("/").rsplit("/", 1)[-1].lower() for m in values}
    # a file that holds both is worded by the whole-image value
    return next((value for value in SOURCE_TYPE_VALUES if value.lower() in found), None)
GENERATOR_RE = [(re.compile(p, re.IGNORECASE), name) for p, name in GENERATOR_PATTERNS]
EDITOR_RE = re.compile(r"photoshop|lightroom|gimp|affinity|pixelmator|capture one|snapseed|facetune|faceapp|reface|deepfacelab|faceswap", re.IGNORECASE)


def _scan_text(blob: str) -> list[str]:
    hits = []
    for rx, name in GENERATOR_RE:
        if rx.search(blob):
            hits.append(name)
    return list(dict.fromkeys(hits))


def c2pa_declares_ai(c2: dict) -> tuple[list[str], bool]:
    """(generator names found in the manifest, whether an action declares trainedAlgorithmicMedia)."""
    blob = json.dumps(c2.get("raw", {}))[:20000]
    hits = _scan_text(blob + " " + str(c2.get("claim_generator", "")))
    return hits, any("trainedAlgorithmicMedia" in d for d in c2.get("digital_source_types", []))


# N-19n (Q4-02; D-15 as amended): the PNG text keys that a tool writes at save time. "comment" stood among them, against
# the rule above ("a hit in FREE TEXT (captions, comments) is weak"): a camera photograph saved as a PNG with the Comment
# "compare with the Midjourney version" was headed as naming an AI generator. A generator that writes its parameters into
# the Comment keeps its marker by the fingerprint that reads them (NovelAI: `provenance/metadata_rules.py`)
PARAMETER_PNG_KEYS = ("parameters", "prompt", "workflow")
TOOL_PNG_KEYS = ("software", "generator", "model")
STRUCTURED_PNG_KEYS = (*TOOL_PNG_KEYS, *PARAMETER_PNG_KEYS)
# the keys of the IF §4.3 M6 fingerprints that `metadata_rules.generator_fingerprints` reads besides those above
INVOKEAI_PNG_KEYS = ("invokeai_metadata", "invokeai_workflow", "invokeai_graph", "sd-metadata", "dream")
FINGERPRINT_PNG_KEYS = (*INVOKEAI_PNG_KEYS, "fooocus_scheme", "comment")
# N-19p (code review R10-01): the keys that `img.meta.generator` and `img.meta.consistency` read by name, lower case; the bound
# on PNG text does not drop them
PNG_KEYS_READ = frozenset((*STRUCTURED_PNG_KEYS, *FINGERPRINT_PNG_KEYS))
_PARAMETER_TERMS = ("steps", "sampler", "cfg", "seed", "ksampler", "checkpoint", "negative prompt", "class_type")


# N-19n (D3-07, 5): the check's groups of places, as its `where` names them, in the order it lists them
PLACE_PARAMETERS, PLACE_EXIF, PLACE_XMP, PLACE_SOURCE_TYPE, PLACE_PNG = (
    "PNG generation parameters chunk", "EXIF Software/Make/Model", "XMP CreatorTool/history", "XMP IPTC digital source type",
    "PNG Software chunk")


# N-19o (browser QA Q5-05): the free-text fields in a reader's words, for the finding of a mention ("the EXIF image
# description"); another EXIF tag is named by its tag, a PNG text chunk by its key, the rest of an XMP packet as the packet
_EXIF_TEXT_WORDS = {"ImageDescription": "the EXIF image description", "UserComment": "the EXIF user comment",
                    "XPComment": "the EXIF comment (Windows)", "XPSubject": "the EXIF subject (Windows)",
                    "XPTitle": "the EXIF title (Windows)", "XPKeywords": "the EXIF keywords (Windows)", "Artist": "the EXIF artist field",
                    "Copyright": "the EXIF copyright field"}
MENTION_FIELD_CHARS = 40  # a PNG key is the file's: it is cut to this in the field's words


def _mention_fields(exif: dict, png_text: dict, rest_of_xmp: str) -> list[dict]:
    """The free-text fields that hold a hit, each with the check's labels for what it found there, in the order the check
    reads them (N-19o, Q5-05)."""
    places = [(_EXIF_TEXT_WORDS.get(str(k), f"the EXIF field {k}"), str(v)) for k, v in exif.items() if k not in STRUCTURED_EXIF]
    places += [(f"the PNG text chunk “{str(k)[:MENTION_FIELD_CHARS]}”", v) for k, v in png_text.items() if k != XMP_PNG_KEY]
    places.append(("the XMP packet", rest_of_xmp))
    return [{"field": field, "labels": found} for field, text in places if (found := _scan_text(str(text)))]


def generator_markers(exif: dict, png_text: dict, xmp: str) -> dict:
    """Generator markers in structured fields (Software, CreatorTool, the IPTC source type, SD parameters) and in free text
    (captions, descriptions, comments). `source_type`: the IPTC digital source type where it declares AI generation
    (`source_type`); `structured_hits` then holds no label for it, so that they are the tools that structured fields name.
    `places` (N-19n): the labels that each group of structured fields holds, so that a finding can say where a tool is
    named; what the check looks for is the same."""
    fields = structured_xmp(xmp)
    structured = [m.group(0) for m in fields.finditer(xmp)]
    structured_blob = " ".join([*(str(exif.get(k, "")) for k in STRUCTURED_EXIF), *structured,
                                *(v for k, v in png_text.items() if k.lower() in STRUCTURED_PNG_KEYS)])
    parameters = "".join(png_text.get(k, "") for k in PARAMETER_PNG_KEYS).lower()
    sd_params = any(k.lower() in PARAMETER_PNG_KEYS for k in png_text) and any(t in parameters for t in _PARAMETER_TERMS)
    declared = source_type(xmp)

    def labels(texts) -> list[str]:
        return list(dict.fromkeys(hit for text in texts for hit in _scan_text(str(text)) if hit != SOURCE_TYPE_LABEL or declared is None))

    places = {PLACE_PARAMETERS: labels(v for k, v in png_text.items() if k.lower() in PARAMETER_PNG_KEYS),
              PLACE_EXIF: labels(exif.get(k, "") for k in STRUCTURED_EXIF), PLACE_XMP: labels(structured),
              PLACE_PNG: labels(v for k, v in png_text.items() if k.lower() in TOOL_PNG_KEYS)}
    where = []
    if sd_params:
        where.append(PLACE_PARAMETERS)
    if any(_scan_text(str(exif.get(k, ""))) for k in STRUCTURED_EXIF):
        where.append(PLACE_EXIF)
    if places[PLACE_XMP]:
        where.append(PLACE_XMP)
    if declared is not None:
        where.append(PLACE_SOURCE_TYPE)
    if any(_scan_text(v) for k, v in png_text.items() if k.lower() in TOOL_PNG_KEYS):
        where.append(PLACE_PNG)
    # the packet of a PNG stands in its text chunks too: it is read once, as XMP
    free_blob = " ".join([*(str(v) for k, v in exif.items() if k not in STRUCTURED_EXIF),
                          *(v for k, v in png_text.items() if k != XMP_PNG_KEY), fields.sub("", xmp)])
    hits = [hit for hit in _scan_text(structured_blob) if hit != SOURCE_TYPE_LABEL or declared is None]
    free_hits = _scan_text(free_blob)
    return {"structured_hits": hits, "sd_params": sd_params, "where": where, "free_hits": free_hits, "source_type": declared,
            "places": {place: found for place, found in places.items() if found},
            "mentions": _mention_fields(exif, png_text, fields.sub("", xmp)) if free_hits else []}


def camera_metadata(exif: dict) -> dict:
    """Capture-device facts: make/model/software, edited (an editor in Software), coherent (full capture EXIF, unedited)."""
    make, model, software = exif.get("Make", ""), exif.get("Model", ""), exif.get("Software", "")
    edited = bool(EDITOR_RE.search(software))
    coherent = bool(make and model and exif.get("ExposureTime") and (exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")) and not edited)
    return {"make": make, "model": model, "software": software, "edited": edited, "coherent": coherent,
            "editor_only": bool(not (make or model) and software and edited)}


def read_exif(im: Image.Image) -> dict:
    out: dict = {}
    try:
        exif = im.getexif()
        for k, v in exif.items():
            tag = TAGS.get(k, str(k))
            if isinstance(v, bytes):
                v = v[:64].decode("utf-8", "ignore")
            out[tag] = str(v)[:200]
        # IFD0 sub-IFD (ExifIFD)
        try:
            for k, v in exif.get_ifd(0x8769).items():
                tag = TAGS.get(k, str(k))
                if isinstance(v, bytes):
                    v = v[:64].decode("utf-8", "ignore")
                out[tag] = str(v)[:200]
        except Exception:
            pass
    except Exception:
        pass
    return out


# N-19o (security re-check SEC5-03): the PNG text that the checks read is bounded in all, not per chunk: 32,000 compressed
# chunks of a 1.43 MB file made the generator check take 8.3 s and the consistency check 1.5 s
PNG_TEXT_CHUNKS, PNG_TEXT_CHARS = 64, 64 * 1024


class PngText(dict):
    """The PNG text that the checks read, by key; `unread`: the text chunks that the bound left unread."""
    unread = 0


def read_png_text(im: Image.Image) -> PngText:
    """The text chunks of a PNG (and other string entries of `im.info`), each cut to 2,000 characters: the first chunk of
    each key that the checks read by name (`PNG_KEYS_READ`), wherever it stands (N-19p, R10-01: a generator marker past the
    bound was dropped and the decision changed), and of the others at most `PNG_TEXT_CHUNKS`, `PNG_TEXT_CHARS` in all.
    `unread` counts the others that the bound leaves out."""
    info, named, left = PngText(), set(), PNG_TEXT_CHARS
    others = 0
    for k, v in (im.info or {}).items():
        if not isinstance(v, (str, bytes)) or k.lower() in {"icc_profile", "exif"}:
            continue
        key = str(k)
        if key.lower() in PNG_KEYS_READ:
            if key.lower() not in named:
                named.add(key.lower())
                info[key] = (v.decode("utf-8", "ignore") if isinstance(v, bytes) else v)[:2000]
            continue
        if others >= PNG_TEXT_CHUNKS or left <= 0:
            info.unread += 1
            continue
        s = (v.decode("utf-8", "ignore") if isinstance(v, bytes) else v)[:min(2000, left)]
        info[key] = s
        others += 1
        left -= len(s)
    return info


def read_xmp_bytes(data: bytes) -> str:
    """The XMP packet among a file's raw bytes, cut to `XMP_SCAN_BYTES` (SEC4-01: it ran to the end tag, wherever that stood)."""
    i = data.find(b"<x:xmpmeta")
    j = data.find(b"</x:xmpmeta>")
    if i >= 0 and j > i:
        return data[i : min(j + 12, i + XMP_SCAN_BYTES)].decode("utf-8", "ignore")
    return ""


def read_png_xmp(data: bytes) -> str:
    """The XMP packet of a PNG from its `XML:com.adobe.xmp` chunk (iTXt, also zTXt or tEXt), inflated where the chunk is
    compressed (N-19n: a compressed packet is not among the file's raw bytes, and reached the check as PNG text alone).
    The chunks are walked without decoding the picture; "" for any other file, without the chunk, or if it cannot be read.
    The packet is cut to `XMP_SCAN_BYTES`, plain or inflated (SEC4-01: a plain chunk had no bound)."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ""
    at, key = 8, XMP_PNG_KEY.encode()
    try:
        while at + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[at:at + 8])
            body = data[at + 8:at + 8 + length]
            at += 12 + length
            if kind == b"IEND":
                break
            if kind not in (b"iTXt", b"zTXt", b"tEXt") or not body.startswith(key + b"\x00"):
                continue
            rest = body[len(key) + 1:]
            if kind == b"tEXt":
                return rest[:XMP_SCAN_BYTES].decode("utf-8", "ignore")
            if kind == b"zTXt":  # compression method, then the deflated text
                return zlib.decompressobj().decompress(rest[1:], XMP_SCAN_BYTES).decode("utf-8", "ignore")
            compressed, rest = rest[0], rest[2:]  # iTXt: flag, method, language tag, translated keyword, text
            text = rest.split(b"\x00", 2)[2]
            return (zlib.decompressobj().decompress(text, XMP_SCAN_BYTES) if compressed else text[:XMP_SCAN_BYTES]).decode("utf-8", "ignore")
    except (struct.error, zlib.error, IndexError):
        return ""
    return ""


def xmp_packet(data: bytes) -> str:
    """The file's XMP packet: where it stands among the raw bytes (a JPEG's segment, an uncompressed PNG chunk), else the
    PNG's chunk, inflated."""
    return read_xmp_bytes(data) or read_png_xmp(data)


def read_xmp(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return xmp_packet(data)


def read_c2pa(source: Path | bytes, mime: str | None = None) -> dict:
    """Active-manifest summary for the v1 path from a file path or bytes, read offline: no remote manifests, no OCSP (I-2).
    The v2 detector uses `app.provenance.c2pa` directly."""
    try:
        import c2pa  # type: ignore

        from app.provenance import c2pa as provenance

        try:
            bundle = provenance.load_bundle(provenance.OFFICIAL)
        except provenance.TrustBundleError:
            bundle = None
        context = c2pa.Context.from_dict(provenance.settings(bundle))
        try:
            if isinstance(source, bytes):
                reader = c2pa.Reader(mime or "application/octet-stream", io.BytesIO(source), context=context)
            else:
                reader = c2pa.Reader(str(source), context=context)
            try:
                raw = reader.json()
                manifest = json.loads(raw) if isinstance(raw, str) else raw
                state = str(reader.get_validation_state()) if hasattr(reader, "get_validation_state") else "unknown"
            finally:
                with contextlib.suppress(Exception):
                    reader.close()
        finally:
            with contextlib.suppress(Exception):
                context.close()
        active = manifest.get("active_manifest")
        m = (manifest.get("manifests") or {}).get(active, {}) if active else {}
        gen = m.get("claim_generator") or (m.get("claim_generator_info") or [{}])[0].get("name", "")
        actions = []
        dst = []
        for a in m.get("assertions", []):
            if str(a.get("label", "")).startswith("c2pa.actions"):
                for act in (a.get("data") or {}).get("actions", []):
                    actions.append(act.get("action", ""))
                    if act.get("digitalSourceType"):
                        dst.append(act["digitalSourceType"])
                    if act.get("softwareAgent"):
                        gen = f"{gen}; {act['softwareAgent'] if isinstance(act['softwareAgent'], str) else act['softwareAgent'].get('name','')}"
        return {"present": True, "validation_state": state, "claim_generator": gen, "actions": actions, "digital_source_types": dst, "raw": manifest}
    except Exception as e:  # no manifest, or library error
        msg = str(e)
        absent = any(marker in msg.lower() for marker in ("no jumbf", "manifestnotfound", "not found"))  # c2pa 0.37: "ManifestNotFound: no JUMBF data found"
        return {"present": False, "error": None if absent else msg[:200]}


def metadata_signals(path: Path, im: Image.Image) -> tuple[list[Signal], dict]:
    signals: list[Signal] = []
    exif = read_exif(im)
    png_text = read_png_text(im)
    xmp = read_xmp(path)
    c2 = read_c2pa(path)
    meta = {"exif": exif, "png_text": {k: v[:300] for k, v in png_text.items()}, "xmp_present": bool(xmp), "c2pa": {k: v for k, v in c2.items() if k != "raw"}}

    # --- C2PA ---
    if c2.get("present"):
        hits, ai_dst = c2pa_declares_ai(c2)
        if hits or ai_dst:
            signals.append(
                Signal(
                    id="c2pa_ai_manifest", name="C2PA Content Credentials declare generative AI", category="provenance",
                    observed=f"claim_generator='{c2.get('claim_generator','')}'; digitalSourceType={c2.get('digital_source_types')}",
                    reference="Camera/editor manifests carry digitalCapture/edit actions, never trainedAlgorithmicMedia",
                    ai_probability=0.98, weight=3.0, direction="supports_ai", strength="decisive", decisive=True,
                    reasoning="The file carries a cryptographically bound C2PA manifest in which the producing tool itself declares the content as AI-generated (" + ", ".join(hits or ["trainedAlgorithmicMedia"]) + "). This is documentary evidence from the generator, not a statistical inference.",
                    details={"validation_state": c2.get("validation_state"), "actions": c2.get("actions")},
                )
            )
        else:
            state_ = str(c2.get("validation_state", "")).lower()
            valid = state_ in ("valid", "trusted", "validationstate.valid", "validationstate.trusted")
            signals.append(
                Signal(
                    id="c2pa_manifest", name="C2PA Content Credentials present (non-AI)", category="provenance",
                    observed=f"claim_generator='{c2.get('claim_generator','')}', validation={c2.get('validation_state')}",
                    reference="A valid capture/edit manifest binds the pixels to a signed provenance chain",
                    ai_probability=0.15 if valid else 0.35, weight=1.5 if valid else 0.5,
                    direction="supports_authentic" if valid else "neutral", strength="strong" if valid else "weak",
                    reasoning="A C2PA manifest is present and does not declare generative AI. "
                    + ("Its signature validates, so the pixels have not been altered since signing." if valid else "Its validation state is not confirmed, so it is only weak evidence."),
                    details={"actions": c2.get("actions")},
                )
            )

    # --- generator markers: structured fields are decisive, free text is weak ---
    # N-19n: one rule with the v2 check (`generator_markers`): the lists of structured fields stood here a second time
    markers = generator_markers(exif, png_text, xmp)
    structured_hits = [*markers["structured_hits"], *([SOURCE_TYPE_LABEL] if markers["source_type"] else [])]
    sd_params, where = markers["sd_params"], markers["where"]
    if structured_hits or sd_params:
        signals.append(
            Signal(
                id="generator_metadata", name="Generative-AI tool named in structured metadata", category="provenance",
                observed="; ".join(structured_hits) if structured_hits else "Stable-Diffusion style generation parameters",
                reference="Camera/editor files name a camera or editor in these fields, never a generator",
                ai_probability=0.97, weight=3.0, direction="supports_ai", strength="decisive", decisive=True,
                reasoning="The file's structured provenance fields name a generative-AI tool or carry its generation parameters (" + "; ".join(where or ["structured field"]) + "). These fields are written by the producing software at save time.",
                details={"hits": structured_hits, "locations": where, "png_keys": list(png_text.keys())[:10]},
            )
        )
    else:
        free_hits = markers["free_hits"]
        if free_hits:
            signals.append(
                Signal(
                    id="generator_mention_freetext", name="Generative-AI terms in free-text metadata (caption/comment)", category="metadata",
                    observed="; ".join(free_hits), reference="Captions and comments are user-editable and may mention AI tools without the image being synthetic",
                    ai_probability=0.65, weight=0.5, direction="supports_ai", strength="weak",
                    reasoning="A caption, description or comment field mentions a generative-AI tool or an 'AI generated' phrase. This is a user-written hint, not a tool signature, so it is weighted as weak evidence only.",
                    details={"hits": free_hits},
                )
            )

    # --- camera EXIF ---
    make, model, software = exif.get("Make", ""), exif.get("Model", ""), exif.get("Software", "")
    if make or model:
        edited = bool(EDITOR_RE.search(software))
        coherent = bool(make and model and exif.get("ExposureTime") and (exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")) and not edited)
        signals.append(
            Signal(
                id="camera_exif", name="Camera EXIF present" + (" and coherent" if coherent else ""), category="metadata",
                observed=f"Make='{make}', Model='{model}', Software='{software}'" + (f", Exposure={exif.get('ExposureTime','')}, ISO={exif.get('ISOSpeedRatings', exif.get('PhotographicSensitivity',''))}" if exif.get("ExposureTime") else ""),
                reference="Generated images have no capture device; EXIF is, however, trivially forgeable",
                ai_probability=0.45 if edited else (0.22 if coherent else 0.30), weight=0.9 if coherent else 0.5,
                direction="neutral" if edited else "supports_authentic", strength="moderate" if coherent else "weak",
                reasoning="Capture-device metadata is consistent with a camera original" + (" — make, model, exposure and ISO are all present and no editing software is recorded, which generators and most re-encoders do not produce." if coherent else ".") + (f" The Software tag shows post-processing in '{software}', so the file was edited after capture." if edited else ""),
                details={"edited_with": software if edited else None},
            )
        )
    elif software and EDITOR_RE.search(software):
        signals.append(
            Signal(
                id="editor_software", name="Editing software tag without camera data", category="metadata",
                observed=f"Software='{software}'", reference="Originals carry Make/Model; exported edits often do not",
                ai_probability=0.5, weight=0.3, direction="neutral", strength="weak",
                reasoning="The file was saved by an editing application and no capture-device metadata remains; this indicates processing but not its nature.",
            )
        )
    else:
        signals.append(
            Signal(
                id="no_metadata", name="No capture metadata", category="metadata",
                observed="No EXIF Make/Model, no C2PA, no generator markers",
                reference="Social-media and messaging apps strip metadata; generators usually write none",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="Absence of metadata is uninformative on its own: it is equally consistent with a re-shared photograph and with a synthetic image.",
            )
        )
    return signals, meta
