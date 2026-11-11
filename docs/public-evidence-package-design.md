# Public evidence packages

The customer export uses `chipmunk.public-evidence/1`, a separately signed derivative of a frozen report. Internal `veritas.evidence/2.0` packages remain immutable and retain offline verification support; customer routes do not download them. Migration 0024 labels existing rows with the legacy schema without changing their stored ZIPs, hashes or signatures.

## Contents and commitments

- `report.pdf` is the exact approved, signed report. No older executive summary is implicitly included. The report binds to its analysis revision, source, original result, snapshot and evidence set.
- `evidence.json` is the exact frozen evidence record used by the report, including its set hash. `assets/` contains every declared PNG, PCM WAV or silent H.264 MP4, with content-hash filenames. Arbitrary internal artifacts and the model registry are excluded.
- `result.json` is an allowlisted public projection. It has its own hash; the original result hash remains a separate commitment. Source quotations are preserved rather than rewritten to hide words resembling model names.
- `bundle.json` records these bindings, all asset hashes, the signer, and draft/issue status. Custody entries are explicitly only original event-hash commitments: original event bodies and chain completeness cannot be verified from this derivative.
- `source/original.*` is optional and excluded by default. When selected, it contains exact unredacted user bytes, including embedded metadata. Source crops and excerpts are already included in ordinary evidence assets.
- `manifest.sha256`, its Ed25519 signature and the public key support offline integrity checks. The README distinguishes integrity from inference accuracy and explains how to obtain the verification key separately.

## Verification

`python -m app.cli verify package.zip --public-key trusted-instance.pem` verifies the archive, manifest, signed report, supported evidence version, exact asset membership, source/result/revision bindings, PNG dimensions, WAV sample dimensions, and MP4 frame counts/timestamps/track format. Missing, altered, extra or mismatched files are refused. Without a separately trusted key the outcome is only `self_consistent`, never authenticated verification.

Add `--source-video original.mp4` to replay the native source frame clock, original PNGs, face crops and aligned inputs. Video evidence version 11 can also contain a refined face-change range under `face_boundary` and its three source views under `face_boundary_frames`. The verifier checks the matching review check's signed measurement, both time ranges, the measured transition support and the source images. The refined range carries zero overall confidence weight.

Add `--source-document original.pdf` to verify the recorded excerpts, input ranges, source positions, repeated passages and native PDF evidence images against a local original. This optional replay supports TXT, DOCX, EML with a plain-text body, and native-text PDFs up to 25 MiB. For OCR-dependent documents, inspect the signed frozen evidence. The replay checks source correspondence without scoring authorship.

`reproduce` explicitly refuses this public format, even if the optional original is present. The package does not include the internal inputs needed to repeat inference. A valid signature is not proof of AI authorship or manipulation.

## Local validation

Tests cover pinned/unpinned/wrong keys, tampering and re-signed inconsistent content, source-quote preservation, historical package verification, migration preservation, and real authorization/case scoping. Multimedia fixtures exercise frozen frames, PCM clips and document excerpts with and without originals; their offline verifier runs without application runtime or network access.

The live image example is analysis `171d1fe2-bed0-4f19-aec1-cd0d5f34f02e`, with approved report `VR-2026-000004`. Its package `fb6a1c1f-48ad-4ffe-b5e4-9e153d00defc` contains 25 files, including all 16 declared evidence images. The downloaded report bytes and evidence record equal the PDF and API outputs. The ordinary browser export also completed with report `VR-2026-000005` and package `cb0887dc-02c3-4fad-aceb-66c9f1ad1b28`, using template r53. Verification records are stored locally under ignored `backend/data/image-evidence-demo/`.

Video clips retain signed source PTS mappings and explicit lossy/silent transforms; they are viewing context, not additional scored frames. The offline MP4 reader is bounded standard-library code and imports no model runtime. Document ratio evidence distinguishes full forward contexts from the selected tokens used in each local ratio; opposite-direction responses and bounded omissions are retained.

Image evidence version 3 adds a separate whole-image candidate-selection record. Its exact response, fixed threshold, source hash and scalar-map hash are bound to the frozen check; this selection never replaces the quality mask or zeroes the response. Coverage counts all qualifying measured components before the localiser/display caps, distinguishes suppression from post-selection omissions, and retains the actual 8-bit scalar PNG and quality PNG in the declared assets. These are the persisted, quantised measurement grids, not native-resolution pixel-edit probabilities. PDF manifests commit to the same grids and plates as JSON and the public ZIP; the dependency-free offline validator also checks gate direction, counts and map bindings.

The live version-3 image pair is `fbee815b-3801-4d44-8781-7b6353600a07` and `aa5cd758-b4b1-4d65-b025-d8c1ab762ba0`, with signed r59 reports `VR-2026-000018` and `VR-2026-000019`. Both source-included and source-excluded packages passed trusted-key offline verification. The 35 unique PNG assets, nine exact source crops, 29 PDF-embedded plates, API/JSON equality and ordinary-view mask/overlay/source navigation were checked. The first control remains immutable after a new analysis corrected the distinction between an unselected region and a missing check. Records are under ignored `backend/data/image-evidence-v3-demo/`; these engineering examples do not establish detector accuracy.
