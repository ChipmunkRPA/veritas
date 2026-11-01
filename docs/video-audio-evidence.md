# Video-contained voice and audio/video evidence

Internal implementation and verification record, 2026-10-31. These checks establish input correspondence and export integrity. They are not detector accuracy measurements.

## Authority and compatibility

`chipmunk.video-evidence/1`, version `video-evidence-3`, keeps the existing visual `packets` and adds `audio` and `av` content blocks. Each child has `status`, `reason`, `source`, `coverage`, `packets`, and `limitations`; voice also has `overview` and `omitted_windows`. A child cannot claim its own schema, version, set ID, set hash, analysis, revision, or result identity. The parent commits every child field and presentation asset through its canonical hash. All child asset URLs use the parent analysis ID and set ID.

Versions 1 and 2 remain readable without mutation or retroactive enrichment. Reads never build evidence. Version 3 is published only by a new completed analysis; missing old inputs are explicitly unavailable. The reader accepts all three versions. The version-3 writer was enabled after the ordinary view passed 52 frontend tests and a TypeScript/Vite build; service activation and live acceptance are coordinated separately with the report template revision.

## Original and decoded audio

The video decode stage freezes the already-decoded 16 kHz mono PCM as signed 16-bit WAV and its source timestamp mapping, including silent or too-short input. It does not decode a second copy to invent evidence. `manifest.sampling.video_audio_evidence` commits that input and the hashes of successful voice and AV raw outputs. These internal source records are omitted from generated report prose.

Voice windows reuse the standalone audio content builder: exact sample intervals, complete recorded responses, selected and padded context WAVs, waveform and score plots, source-clock ranges, and checks linked to the corresponding public signal IDs. The default selection retains up to six windows reaching the uncalibrated 0.5 inspection boundary. Remaining windows and sample coverage remain in JSON. PCM waveform appearance alone is not evidence of synthesis. A whole-window response does not identify words, a speaker, or the location of a splice.

## Served correlation measurement

The served mouth/audio correlation check freezes the actual inputs while they still exist in the run:

- One source frame choice per 25 Hz tick, preserving repeated frame indices on variable-frame-rate input. Every choice retains native RGB hash, frame PTS, source audio PTS, the exact integer mouth crop rectangle, actual 32×24 gray-patch hash, and measured aperture proxy.
- Each 1-second measured window includes 25 mouth inputs and 55 audio RMS ticks. All 31 candidate lags from −15 to +15 ticks (−600 to +600 ms) are retained. Positive lag means audio later than the image. The selected lag, peak correlation, low-correlation state, offset-change/collapse overlap, and track baseline are preserved.
- Each packet includes all 25 actual gray patches as lossless PNGs, three representative native frames and exact source mouth crops, plotted lag and alignment curves, selected unshifted 1-second WAV, full 2.2-second lag-search WAV, and a bounded silent MP4 viewing context.
- The MP4 is a re-encoded visual aid, not original compressed bytes or additional detector evidence. Exact source pixels remain separate PNGs. MP4 source PTS, timescale conversion, sample durations, selected packet bindings and the last-duration basis are committed. The WAVs carry the measured audio separately; the silent MP4 does not demonstrate audiovisual synchronization by itself.

At most four AV windows are selected: recorded offset-change/collapse windows first, then chronological informational examples. Other measured-window counts and unsupported runs are disclosed. A selected lag or high correlation is not an AI probability or proof of authenticity. Mouth shading, tracking/interpolated landmarks, occlusion, unrelated speech, container offsets, dropped frames and clock drift can affect this measurement. A track is a measurement grouping, not an identity conclusion.

The research neural synchronization path lacks a complete frozen mel/feature-input proof and therefore cannot be presented as supported AV packet evidence. Missing PCM mapping, discontinuous source audio, no audio stream, no successful voice windows and no usable correlation windows have distinct explicit states; none is interpreted as a negative detection.

## Validation and exports

`video_audio.verify_source_content` verifies voice windows against the parent's frozen raw-run commitments. `video_av.verify_source_assets` verifies raw measurements, actual nearest-frame choices, native and downsampled pixel hashes, exact PCM intervals, all RMS/lag values, and MP4/source-table timing. These checks read content-addressed objects, not mutable analysis state, and do not rerun source-video decoding or inference. During building, native frames are streamed; only the tiny patches and three representative full-size frames per packet are retained.

The API's existing allowlist includes all declared child PNG/WAV/MP4 assets, with the same authentication and byte-range behavior. PDF capture freezes the complete parent document. Its public manifest lists exactly the JSON presentation assets, including media that a static PDF cannot play. Source images/audio and excerpts are withheld in newsroom/redacted reports; non-content anchors and commitments remain. The public ZIP contains the same frozen parent JSON and all declared assets.

Offline public verification uses only the standard-library MP4 reader, `audio_binding`, `av_binding`, canonical JSON and signature verification. It checks the signed source/result/set authority, cross-channel PCM agreement, window/sample/lag clocks, signal-reference joins, asset hashes and intrinsic dimensions. It cannot reproduce detector inference or prove the AI verdict. Full source reconstruction from the original requires a separately authorized internal record.

## Reproducible engineering checks

Run from `backend`:

```sh
.venv/bin/python -m pytest -o addopts='' -q tests/test_video_av_evidence.py tests/test_video_nested_evidence.py
```

The FFmpeg fixture uses 100 generated 96×64 frames at 25 fps, video origin 5.0 s, a 16 kHz audio stream beginning at 5.4 s, and a controlled +200 ms envelope offset. It measures the real grayscale/RMS correlation pipeline; a separate fixed voice response exercises content packaging without claiming voice-classifier accuracy. A VFR derivative drops every fifth-frame position and verifies repeated frame selections, exact timestamps and all actual inputs. Tamper tests cover lag curves, raw runs, missing tick inputs, source timestamps, PCM intervals, pixel inputs, parent identity, withheld source content, historical compatibility, and a signed public ZIP verified in a fresh process with model, database, configuration and networking access blocked.

Ignored local QA outputs are under `backend/data/video-evidence-demo/video3-controlled/`: `evidence.json`, PNG/WAV/MP4 assets, the generated source, a labelled 14-page PDF evidence-section proof, rendered pages and contact sheet. This is an engineering fixture, not a benchmark or a live API result.

The live workflow fixture is `backend/data/video-evidence-demo/video-audio-evidence-demo.mp4` (SHA-256 `d0014735457e2d490918ab436697dca7d4d9e998ab14d07e4dfa4cfb10bea302`). It deliberately combines the existing silent visual demo with the first 9.041666667 s of a separate locally generated TTS WAV at zero start offset. All 217 H.264 packets retain their original PTS/DTS/duration/size/content hashes; the audio is AAC at 128 kbps. The adjacent manifest records every input/output checksum, FFmpeg version, command and transformation. The audio does not belong to the person shown. This assembled fixture must not be described as a natural synchronized recording or an accuracy benchmark. The original silent demo remains the channel-absence control.

Actual local full-profile runs on 2026-10-31 used neutral upload names and the single inference worker:

| Input | Analysis | Run time | Recorded evidence |
| --- | --- | --- | --- |
| Assembled video and unrelated TTS (`sample-av-01.mp4`) | `78fdcf6e-0269-4b41-9729-5fc191b53cf6`, revision 2 | 30.199 s | 1 visual packet, 4 voice windows / 4 voice packets, 11 measured correlation windows / 4 AV packets |
| Original silent video (`sample-video-02.mp4`) | `980bd904-8e3d-4f68-a3ec-f3ddbe921116` | 22.402 s | 1 visual packet; voice and AV explicitly `absent`, no fabricated windows |

The first assembled-video attempt (`d15b8f17-6f86-4ac4-9462-06d604b582e6`) failed during evidence publication because the validator compared exact presentation seconds with canonical six-decimal seconds. Its 144,893 decoded samples correspond to 9.0558125 s; canonical JSON carries 9.055812 s. The corrected check accepts only the exact sample-derived value or its canonical representation; sample integers, source PTS and PCM hashes remain exact. A regression rejects a one-sample shift. The failed record was preserved and a new revision completed after the correction. AAC decoder padding explains why the actual decoded sample span can extend slightly beyond the container's declared audio duration.

Independent CPU checks re-decoded 63 original frames for the assembled-video result and matched all 100 selected mouth-patch inputs, all 144,893 PCM samples and 145 declared assets. The silent control matched its three selected original frames and 11 assets. Both API evidence documents equal their JSON exports; the final public API/export privacy scan found zero internal model names. The assembled video's immutable evidence hash is `db643e72b844db371d92527db329fcb427f3fa5861e49e371b90a636399758a2`. Detailed results are stored in each analysis folder's `input-verification.json` under `backend/data/video-evidence-v3-demo/`.

The controlled PDF section passed an independent check of all 82 embedded PNG pixel arrays and 90 asset commitments. Live browser playback and seeking, the 31 lag values, 55 audio RMS blocks, 25 input patches per AV packet, nine representative source/crop images and the silent-channel absence state passed separate ordinary-view checks (`browser-verification.json`).

Final live export acceptance used template `veritas-report-2.0.r61`: assembled-video report `VR-2026-000021` has 52 pages, and silent-control report `VR-2026-000022` has 28 pages. Both report snapshots contain the exact API/JSON evidence set. Their embedded manifests commit all 145 and 11 declared assets respectively, including the WAVs and silent MP4 viewing clips. All 124 and 10 PNG assets were matched pixel-for-pixel inside the actual PDFs. The full reports were rendered and inspected; source frames/crops, waveform windows, lag plots, all selected mouth-patch plates, coverage and limitations remain visible. Four public ZIPs (each report with and without the original source) passed the separately trusted instance-key signature check and offline verifier, with exact source inclusion and asset hashes. Known internal identities were absent from final public JSON, PDF text/manifests and ZIP text members; the exact existing heading “FUSED ERROR RATES IN TESTING” was treated as generic combiner prose, without changing exported bytes.

An earlier signed `r60` report, `VR-2026-000020`, exposed a detector label in its generated fusion-baseline narrative. Its original PDF, snapshot and signature were preserved; the live file endpoint now returns HTTP 409 and its public metadata reports `downloadable: false`. The corrected report was generated as a new revision, not a rewrite. The export checks also verified that the pre-existing 26 analysis records and 19 signed reports, including the completed image evidence reports, remained unchanged. Proofs, immutable baselines, PDFs, ZIPs and page renders are retained under `backend/data/video-evidence-v3-demo/exports/veritas-report-2.0.r61/`; the reusable CPU-only verifier is `backend/data/video-evidence-v3-demo/verify_exports.py`. These selected fixtures establish source correspondence, playback and export integrity, not detector accuracy. No new inference was needed for the report wording fixes.

## Background acoustic context and cross-format locators

New `audio-evidence-2` and `video-evidence-4` publications add a source-bound `acoustic_context` record to the same frozen mono PCM used by voice checks. Historical sets remain readable without inventing missing measurements. Non-overlapping 8,000-sample blocks cover the decoded input, including a partial final block. Measurements include RMS/peak level, near-full-scale sample fraction, low/high frequency energy and spectral flatness. The 10th-percentile energy reference describes the mixed recording; it is not an isolated room-noise estimate.

At most eight B-series observations select near-silent intervals, adjacent-block level changes of at least 12 dB, or near-full-scale samples. Every selected interval retains integer sample bounds, original source PTS ranges, exact/context WAVs, an annotated waveform and a frequency plot. Spectrograms use all 512-sample Hann STFT frames at a 256-sample hop, with bounded-memory peak pooling into at most 880 time columns. The method parameters, all acoustic blocks, omitted event count and related voice-window IDs remain in JSON. Runtime export validation recomputes measurements, plots and clips from frozen PCM; the offline verifier validates the signed coordinates and media commitments without importing the analysis runtime.

These are descriptive acoustics of a mixed track. Speech, music and environmental sound are not separated or classified; pauses, gain changes, music and normal recording distortion can produce these observations. Near-full-scale samples do not prove capture clipping. Neither an acoustic event nor its overlap with a voice score proves synthesis or an exact splice boundary. They do not change the existing fused decision.

The ordinary audio view and video's voice tab expose the same B-series intervals, source clocks, plots and playback. PDF evidence locators map V/A/AV/B/T IDs to exact `evidence.json` pointers. Every plot and source plate repeats its packet/frame ID and source time or text position, including across page breaks. PDFs contain the exact frozen PNG pixels; JSON and ZIP retain their hashes and the associated playable media. Document positions remain Unicode code-point offsets and PDF page, DOCX paragraph or MIME anchors, rather than word-level authorship claims.

Template `veritas-report-2.0.r64` also removes architecture labels, internal fusion IDs, runtime-library labels and implementation revision details from generated report prose. The same API projection preserves source quotations and metadata. A later full-page audit found such labels in earlier report templates, so **all pre-r64 PDF/package downloads are now withheld**, superseding earlier template approvals described above. Their stored bytes, snapshots and signatures remain unchanged; requesting a new report creates a clean version.

Live verification artifacts are under `backend/data/multimedia-background-demo/r64/`. The reusable `backend/scripts/verify_multimedia_exports.py` compares API/JSON/snapshot evidence, validates all source assets and PDF image pixels, scans public output for implementation identities, checks signatures and offline ZIPs, renders every PDF page, and verifies the historical baseline. New audio revision `013e8d85-742c-4564-8ea8-56a8b72ac6f9` retains four acoustic observations; video revision `322b73cd-98f1-4551-b464-eedac9fddafa` retains one. The video still uses the deliberately unrelated TTS engineering fixture described above, not a detector-accuracy benchmark.
