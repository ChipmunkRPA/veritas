# Structured source labels: local evidence and limits

This reader recognizes structured AIGC metadata and preserves it as
**unverified documentary context**. It does not authenticate a provider, decode
its private watermark, locate manipulated pixels or assign AI confidence.
The trusted, asset-bound Content Credentials gate remains separate.

## Implemented formats

| Format | Inspected location |
|---|---|
| PNG | AIGC text chunks and supported XMP text packets, including bounded compression |
| JPEG | APP1 XMP and the wrapped AIGC object in EXIF UserComment |
| WebP | RIFF XMP and EXIF UserComment |
| WAV | RIFF AIGC chunks and supported XMP chunks |
| MP3 | Supported ID3v2.3/v2.4 TXXX frames with the AIGC description |
| FLAC | AIGC Vorbis comments |
| Ogg Vorbis / Opus | Initial supported comment packets, with page/packet bounds |
| MP4 / MOV / M4A | Keyed AIGC metadata in the supported movie user-data metadata container |
| DOCX / OOXML | The custom AIGC property in the document metadata member |
| PDF | The Information dictionary AIGC entry and supported, bounded catalog XMP |

Other formats and layouts are explicitly unassessed. This release does not
claim TIFF/HEIC, every Matroska/AVI layout, arbitrary Markdown headers, every
ID3 flag/encoding variant, or all private provider watermarks. Plain article
text, email bodies, whitespace, filenames and logos are not treated as these
structured fields. A source provider can use a different marking format or no
surviving mark at all.

Exact known-name aliases normalize claims for OpenAI, Anthropic/Claude,
Google/Gemini, Meta/Muse, xAI/Grok, MiniMax/Hailuo and DeepSeek, plus
Alibaba/Qwen/Wan, ByteDance/Doubao/Seedance, Tencent/Hunyuan, Baidu/ERNIE and
Kuaishou/Kling. Version 2 now includes known-name normalization for the 82 families in
the [provider catalog](provider-signals-catalog.json), including Suno, Bria,
Microsoft, Adobe, Resemble, ElevenLabs, Kimi, Cartesia and Inworld. The voice examples published by
Volcengine also motivate recognizing its `volcengine` spelling. Both historical
version 1 and current version 2 records render in the normal view and reports.
This is normalization of an uploaded field, **not evidence that
each provider uses this format or that the claimed name is correct**.

The three supported label values retain their separate meanings: declared,
possible and suspected AI involvement. None becomes an authenticated assertion
or a numerical probability. Reserved fields are not silently interpreted as
verified signatures. Unknown codes and arbitrary provider strings are withheld
from the new structured records; content IDs are represented only by presence
indicators. The original upload remains unchanged.

## Evidence and analysis behavior

The reader rechecks the uploaded file's SHA-256 before reading its metadata.
Each recognized record has an SL identifier, a source locator, the payload
checksum, the bound upload checksum, the claimed known provider if recognized,
and explicit unauthenticated/no-confidence fields. Byte offsets identify the
metadata container, not a pixel, spoken interval or authored sentence.
The payload hash covers the extracted label field before JSON parsing; decoded
text is encoded as UTF-8. It is not the checksum of an entire compressed chunk,
XML packet or UTF-16 frame. The separate source checksum binds the original file.

The records are stored in the reproducibility manifest, outside detector
signals and statistical fusion. Unsigned labels never activate the positive
Content Credentials shortcut. Normal applicable checks continue. Missing,
unsupported or malformed labels cannot supply an original-file verdict.
Partial coverage retains usable claims with an explicit incomplete status.

The normal analysis view, all report profiles, API responses and JSON exports
carry matching identifiers and checksums. The JSON export also includes the
original stored result checksum. A historical result with no such check says
that it was not recorded; re-fusion does not invent a new examination of the
original input. Previously stored report and analysis bytes remain immutable.

Resource controls include 64 KiB metadata packets, 8 KiB label fields, at most
eight displayed claims, an aggregate read/step budget and parser time checks.
Compressed metadata has an output bound. XML entities/DTDs and unsupported
encodings are refused. PDF metadata inspection has a 16 MiB file limit; archive
directory/member bounds and duplicate-member checks apply to OOXML. These
limits can leave a valid file's labels unassessed, which is reported explicitly.
Cancellation is preserved; ordinary metadata parse errors remain partial
evidence instead of becoming authorship conclusions.

The OOXML directory is checked before the archive library reads it: both its
declared size/count and its actual bounded entries must agree. ZIP64 overrides,
multiple-disk layouts, inconsistent offsets/comments and unsupported metadata
compression remain incomplete checks. Only bounded stored/deflate custom
properties are opened, using the already verified source descriptor.

No model weights, serving calibration or statistical thresholds change.
Inspection is inside the prepaid analysis pipeline. No customer files are sent
to external providers. Stripe test/live payment enforcement is unchanged.

## Sources and implementation

Source: Secretariat of the National Technical Committee 260 on Cybersecurity.
The independently written reader follows the documented metadata structures in
the [image guide](https://www.tc260.org.cn/tc260/sjzn/202508/7227bfecdf344c14abc647fbbcf69edf/files/37774d2b3149456ba7179a81bc07d3d5.pdf),
[audio guide](https://www.tc260.org.cn/tc260/sjzn/202508/5afb978a1a294937bf66fd96ccdd0f6a/files/bb1fb60134144c0e98554a15a03f0098.pdf),
and [document guide](https://www.tc260.org.cn/tc260/sjzn/202508/f0a84ada0c11458e9cb6da56c1cb2426/files/760387fb30e94c058c31b8c2b2230020.pdf).
The [official guide index](https://www.tc260.org.cn/tc260/sjzn/list_2.shtml)
also distinguishes labeling from metadata security and content detection.
No guide implementation code or new model/dependency was imported.

Provider-specific private verification remains work requiring actual supported
access, response semantics, privacy controls and real provider fixtures. The
[provider coverage review](provider-provenance-coverage.md) distinguishes those
gaps from working local Content Credentials and metadata inspection.

## Local validation

408 selected backend checks pass, including signed-credential short circuits,
unsigned-label fall-through, all four publication flows, API/JSON/report HTML,
payment enforcement, incomplete evidence and public model-name filtering.
After the final label-hash wording change, 72 focused reader/flow/export checks
were repeated and passed. Four authored unsigned-label flows retain an
unavailable or inconclusive assessment when detector weights are deliberately
unavailable; they do not become AI/original verdicts from the metadata.

74 selected frontend tests, type checking, the production build and API schema
comparison pass. The source-label panel rendered in an isolated local browser
preview. Image and document PDFs were built from actual isolated analysis
snapshots; all three final pages were visually checked, with clean vocabulary
and internal-identifier lints and no refused assets. A missing-record message
remains available for historical results. Empty evidence sets preserve coverage
notes without forcing an empty chapter page.

These are authored software and export tests, not real provider watermark
accuracy validation. No private provider key, external upload, serving model
replacement or new model dependency is involved. Outside human reviewers remain
zero. Statistical detection quality remains separate development work.

A further 57 reader and full-flow checks pass after the OOXML preflight
hardening, including malformed-directory refusal before archive opening,
unsupported compression refusal and reuse of the verified file descriptor.
