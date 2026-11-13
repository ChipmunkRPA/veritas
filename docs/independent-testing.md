# Independent testing and collaboration

We welcome independent evaluations of images, video, audio, and writing. The
current detection-improvement program is in [this plan](detection-improvement-plan.md).
An implementation by our team is not an independent review; an external test
must identify who ran it, how inputs were selected, and what was independently
checked. We do not currently claim recruited reviewers or completed community pilots.

Repository-managed GitHub Actions is disabled at the project owner's request.
Run verification locally and push reviewed updates directly. Historical Linux
workflow receipts below describe past runs; they are not instructions to start
or rerun a workflow.

## Start with a bounded check

The [four-modality receipt example](#four-modality-receipt-example) runs with
the Python standard library and fictional inputs. It shows how to reproduce a
comparison and detect a changed operating point.

The [standalone measurement reviewer](#recompute-selected-image-and-video-measurements)
replays a PNG seam or video frame schedule directly from a supplied original
and its signed evidence record. Start there to check those measurements on a
local export.

### Replay delivery and earlier-copy comparisons

Recount the [two JPEG90 delivery histories](image-regional-jpeg90-carrier-2026-11-13.md),
[strong paired-video picture intervals](video-reference-measured-interval-2026-11-13.md),
[12-second audio duration trial](audio-clean-reference-duration-2026-11-13.md)
and [GAO excerpt source binding](writing-gao-archive-comparison-2026-11-13.md).
This replay checks published decisions and overlap counts without source media.
The linked guides cover native replay, the separate audio equalization test
and the Word extraction scope.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/delivery-reference.json" --suite delivery-and-reference-transfers
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/delivery-reference.json" --format-issue
```

### Replay the single-upload transfer results

Recount the [native image prediction trial](image-bayer-prediction-transfer-2026-11-13.md),
[brief face changes in four-second videos](video-brief-face-120-carrier-2026-11-13.md),
[overlapping speech trial](audio-overlap-pitch-transfer-2026-11-13.md) and
[writing typography trial](writing-typography-transfer-2026-11-13.md).
The local receipts cover useful locations, misses, ordinary controls and
which inputs each method could assess. The replay uses only the Python
standard library and the committed scalar records. It does not repeat model
inference or validate general accuracy.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/single-upload.json" --suite single-upload-transfers
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/single-upload.json" --format-issue
```

### Replay local processing and short edit comparisons

The suite recounts [lower-depth PNG resampling](image-quantized-phase-transfer-2026-11-13.md),
[color changes between video copies](video-reference-chroma-transfer-2026-11-13.md),
[short voice edits under competing speech](audio-local-context-transfer-2026-11-13.md)
and [localized writing temperature responses](writing-temperature-transfer-2026-11-13.md).
It uses published scalar measurements. The linked guides distinguish exact
source replay, ordinary flags, misses and tests that did not qualify for serving.
The [GAO excerpt comparison](writing-gao-archive-comparison-2026-11-13.md)
also has a separate reader that checks the preserved source and text positions.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/processing.json" --suite processing-and-short-edits
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/processing.json" --format-issue
```

### Replay the expanded local evidence checks

Run this from a clean checkout with Python 3.11 or later. It recounts all
350 [PNG dimension cases](image-native-cfa-dimensions-transfer-2026-11-13.md),
48 [paired video comparisons](video-paired-local-area-2026-11-13.md),
108 [short voice edit deliveries](audio-phoneme-short-edit-intake-2026-11-13.md)
and [reorganized Word tables](document-word-table-alignment-2026-11-13.md).
The video guide also rebuilds the paired carrier measurements from eight
pinned public recordings, using about 2.8 MB of downloads.
The [public single-upload video replay](video-scene-return-public-replay-2026-11-13.md)
rebuilds the separately shipped picture-return detector's HEVC test from
eight publisher recordings. The [mouth color trial](video-mouth-photometric-transfer-2026-11-13.md)
has a standalone reader for false alarms, misses and exact timestamps.
The [shifted image grid trial](image-native-generated-grid-transfer-2026-11-13.md)
publishes all 104 scalar outcomes, including the rotation and
quantization controls.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/evidence.json" --suite local-evidence-expansion
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/evidence.json" --format-issue
```

### Replay fresh short-edit comparisons

This suite recounts the [shifted image grids](image-native-generated-grid-transfer-2026-11-13.md),
[mouth color and texture](video-mouth-photometric-transfer-2026-11-13.md),
[short voice windows under competing speech](audio-phoneme-short-edit-score-2026-11-13.md),
and [matched forum revisions](writing-forum-revision-transfer-2026-11-13.md).
It checks the published counts and source commitments using the Python standard
library. The guides describe the native checks run on retained local inputs
and which inputs are needed to repeat them.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/short-edit-transfer.json" --suite short-edit-transfer-2026-11-13
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/short-edit-transfer.json" --format-issue
```

### Replay delivery and localization checks

Recount the [rotation controls](image-native-rotated-stencil-transfer-2026-11-13.md),
[re-encoded video trims](video-trimmed-reencode-transfer-2026-11-13.md),
[paired speech with different backgrounds](audio-paired-speech-continuity-transfer-2026-11-13.md)
and [single-upload writing intervals](writing-interval-transfer-2026-11-13.md)
from the published measurements. This verifies counts and decision
arithmetic. Use the respective guide to repeat native media or model checks.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/delivery.json" --suite delivery-and-localization
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/delivery.json" --format-issue
```

### Replay source comparisons and local geometry

Recount the [PNG rotation test](image-native-rotation-location-2026-11-13.md),
[VP9 picture changes](video-scene-return-2026-11-13.md#vp9-webm-transfer),
[clean-reference speech test](audio-clean-reference-transfer-2026-11-13.md),
and [GAO archive passage comparison](writing-gao-archive-comparison-2026-11-13.md)
from their published numeric receipts. The separate
[mouth color trial](video-mouth-temporal-chroma-2026-11-13.md) documents
the synchronized-redraw coverage gap.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/reference.json" --suite reference-and-rotation
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/reference.json" --format-issue
```

### Replay difficult edits across four media

Run from a clean checkout with Python 3.11 or later. The readers use only
published ledgers. They reproduce the reported score and overlap arithmetic
without downloading media, rerunning models, or creating GitHub issues.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/difficult-edits.json" --suite difficult-edits
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/difficult-edits.json" --format-issue
```

The suite covers [feathered image edits](image-masked-context-transfer-2026-11-13.md),
[mouth edits with preserved speech timing](video-mouth-chroma-coupling-2026-11-13.md),
[short voice replacements under competing speech](audio-unassigned-stereo-voice-2026-11-13.md),
and [exact writing locations](writing-position-qwen-confirmation-2026-11-12.md).
The first three ledgers include negative or mixed results. Use a separate
reproduction for native media, new inference, or fresh accuracy estimates.
Python 3.11 and 3.12 reproduced the same four reader outputs locally.
The [independent evaluation form](https://github.com/ChipmunkRPA/veritas/issues/new?template=independent-evaluation.yml)
accepts the output and the exact claim being checked.
The [longer writing source replay](writing-f1000-revision-spans-2026-11-13.md)
separately retrieves original and revised publisher XML, then verifies their
exact sentence locations without running a classifier.
The [single-upload stereo repeat replay](stereo-repetition-2026-11-13.md#verify-a-signed-export)
uses a signed package and the original audio or video to recompute both
channels and check the exact WAV excerpts.
The [gain and polarity check](stereo-repetition-2026-11-13.md#gain-and-polarity)
also verifies two distinct WAVs and the exact sample relationship between them.
The [public AMI replay](audio-gain-public-replay-2026-11-13.md) downloads and
verifies eight publisher WAVs and runs the shipped checks over 528 controlled
deliveries. It does not require private development audio or model weights.
When submitting its result through the evaluation form, include the commit,
`readout_sha256`, FFmpeg version, and any source downloads or counts that
differed. The receipt and those details are sufficient without the audio.
The [trimmed video comparison](video-trimmed-reference-transfer-2026-11-13.md#replay)
and [shifted JPEG comparison](image-reference-shifted-jpeg-transfer-2026-11-13.md)
can each recheck a downloaded JSON result against the two supplied originals.
The [cross-format draft comparison](document-mixed-format-comparison-2026-11-13.md#reproduce)
rebuilds all six TXT, Word and PDF directions from public publisher sources.
It verifies both source locations and checks that a format change alone
produces no wording highlights.
The [Word table replay](document-word-table-comparison-2026-11-13.md#reproduce)
checks paired cell ranges and all formatting-only controls from the
published ledger. It can also compare an exported result against both DOCX files.
The [two-room voice study](audio-spatial-mask-aachen-transfer-2026-11-13.md)
publishes fixed comparison points, an alternate FFT replay and a reader for
checking the decisions without distributing speech or separated waveforms.
The [room and competing-speech ladder](audio-room-stress-transfer-2026-11-13.md#replay)
and [mouth luminance detail study](video-mouth-luma-condition-2026-11-13.md#replay-the-published-ledger)
include standalone standard-library readers for both findings and misses.
The [JPEG tile main bank](image-regional-tile-rescue-2026-11-13.md#coverage-and-replay)
provides anonymous per-file pixel counts and recomputes the ordinary flags and
edit overlap without the images or weights.
The [entropy-conditioned writing trial](writing-entropy-conditioned-2026-11-13.md#replay)
replays exact sentence outcomes and genuine-work flags from scalars without
the text, token IDs or weights.
The [JPEG detector transfer](image-flame-jpeg-transfer-2026-11-13.md#replay)
and [blind dereverberation trial](audio-wpe-room-transfer-2026-11-13.md#replay)
publish source-group results and standard-library checks for both operating
points. The voice replay binds the unchanged mono measurements to the earlier
public ledger.
The [public-report revision replay](writing-revision-trajectory-2026-11-13.md#replay)
rebuilds all 156 records and source positions from two pinned publisher CSVs.
It separately counts text-identical operation labels. The [H.264 prediction
study](video-mouth-codec-prediction-2026-11-13.md#local-replay) supplies the
original frame and crop commitments and a standalone replay of the decisions.
The [standalone video picture-change replay](video-scene-return-2026-11-13.md#reproduce)
recomputes the cut and control counts from a public ledger. With the retained
deliveries it also reopens all timestamps and the exact supporting images using
FFmpeg. The [earlier-copy JPEG structure trial](image-reference-structure-jpeg-2026-11-13.md)
publishes source-group results and the native comparison and export replays.
The [neighbor-predicted mouth study](video-mouth-blind-prediction-2026-11-13.md)
recomputes the decision totals and publishes before, during and after
frame commitments. The [single-upload image residual trial](image-residual-template-transfer-2026-11-13.md)
replays the JPEG observations and source-mask overlap with the retained
qualification archives.
The [Full writing worker replay](writing-revision-worker-transfer-2026-11-13.md#replay)
checks which ranges reached the app on genuine government-report excerpts and
recomputes the separate candidate points from source-bound scalar records.
The [sustained writing response replay](writing-sustained-response-2026-11-13.md)
checks the longer ranges against the full 140-delivery worker ledger.
The [camera-native interpolation replay](image-native-cfa-closure-2026-11-13.md#replay)
uses a published CC0 camera roster to compare known locations, ordinary
controls and incompatible image deliveries.
The [matched utterance speech replay](audio-odss-matched-room-2026-11-13.md#reproduce)
recomputes held background and voice flags on separate reader groups.
The [public-source waveform replay](audio-odss-public-waveform-replay-2026-11-13.md)
rebuilds all 176 deliveries from a clean cache with bounded publisher
downloads, including the exact six-second excerpts used for background speech.
The [PhonemeDF transfer replay](audio-phonemedf-matched-transfer-2026-11-13.md#reproduce)
tests the same frozen operating points on four new readers, two generators
and two competing-speech levels.
The [Aachen flank comparison](audio-flank-background-transfer-2026-11-13.md#reproduce)
publishes paired detection outcomes from a separate room and source
speakers and tests whether adjacent audio can help estimate competing speech.
The [stereo path comparison](audio-stereo-path-return-transfer-2026-11-13.md#replay)
recounts recording-path flags under competing speech from an anonymous ledger.
The [earlier-copy mouth comparison](video-paired-native-mouth-transfer-2026-11-13.md#replay)
recomputes time and changed-pixel overlap independently on timing-preserving redraws.
The [earlier-copy video extent study](video-reference-extent-2026-11-13.md#replay)
recounts the time-and-pixel overlap on short authored edits and format controls.
The [public design-system revisions](writing-uswds-revision-transfer-2026-11-13.md)
replay passage-location flags against pinned source positions and genuine editorial controls.
The [generated image replacement pilot](image-native-generated-stencil-transfer-2026-11-13.md)
rebuilds all 48 regions and exact masks using the source captures and retained
generated outputs. Independent generation requires the pinned local runtime.
The [interpolation-control follow-up](image-native-stencil-interpolation-transfer-2026-11-13.md)
includes 13 additional CC0 camera captures and an anonymous scalar audit of
ordinary changes that resemble generated replacements.

### Replay the latest evidence transfer results

The extent suite checks all four current ledgers from a clean checkout.
It recomputes the image interpolation controls, video region overlap,
noisy-speech decisions and Word table results. The separate public audio
command above rebuilds the waveforms.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/evidence-extent.json" --suite evidence-extent-2026-11-13
```

The earlier transfer suite checks the earlier-copy JPEG comparison, timing-preserving mouth
redraws, background estimation, and all 140 writing worker deliveries. It
replays the published decisions and misses using only the standard library.
Use two interpreters or two checkouts to compare their receipts. Python 3.11
and 3.12 matched on all four local reader outputs.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/evidence-transfer.json" --suite evidence-transfer-2026-11-13
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/evidence-transfer.json" --format-issue
```

Use `--suite specialist-transfer` in the replay command below to check these
four newer comparisons together. The resulting receipt can be reviewed
without downloading the source reports or media.

### Replay the latest localization comparisons

The four-study suite covers the complete JPEG tile trial, native and H.264
mouth detail, the audio room and competing-speech ladder, and mixed-writing
token scores. It uses only the published measurements and Python 3.11 or
later. It does not require model weights, a test account or a paid analysis.

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/localization-limits.json" --suite localization-limits
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/localization-limits.json" --format-issue
```

The [large-image copy comparison](image-affine-native-scale-2026-11-11.md#reproduce)
provides a licensed roster, exact source commitments, and a separate reader
for the PNG and JPEG localization and ordinary-control results.
The [between-block JPEG copy comparison](image-offgrid-copy-transfer-2026-11-12.md)
provides a source-separated confirmation and tests whether the signed
evidence can reproduce both crop positions and their color measurements.
The [eight-photo public JPEG replay](image-jpeg-public-replay-2026-11-12.md)
starts from a clean checkout and pinned public downloads. It reproduces
both shipped JPEG fallback checks and verifies native crops without the
original study's source cache.

The [earlier-image orientation comparison](image-reference-orientation-transfer-2026-11-12.md)
reconstructs right-angle rotations and reflections. Its Pillow-only reader
checks the original sources, native crops, and submitted difference mask.
The [feathered-edit crop study](image-crop-layout-transfer-2026-11-12.md)
offers a standard-library review of the result ledger and a native replay that
separates region size limits from score and layout errors.
The [signed residual phase study](image-phase-locked-residual-2026-11-12.md)
checks two pixel-grid periods on known generated regions and ordinary views.
Neither period supplied a useful location at its frozen point.
The [earlier-JPEG comparison](image-reference-delivery-envelope-2026-11-12.md)
checks soft changes, ordinary re-encoding and global contrast. Its export
reviewer reopens both originals and verifies the displayed crop and overlay.
The [exact JPEG block alignment study](image-reference-exact-jpeg-2026-11-12.md)
replays a fallback for earlier PNGs against edited, unchanged and unrelated
deliveries. The [mouth-contour phase study](video-mouth-contour-phase-2026-11-12.md)
replays two single-upload video methods that missed synchronized neural redraws.
The [photographic PNG supplement](image-png-layout-guard-2026-11-12.md)
replays the scope check on paired browser layouts and ordinary photos.
Its public ledger compares the separate indications against 74 exact
Full app results and records native crop commitments.
The [PNG Exif preview comparison](image-png-exif-preview-2026-11-12.md)
reopens a submitted PNG and its saved thumbnail, then verifies both crops
and the measured difference from the signed evidence. The [single-file image
candidates](image-local-edit-candidates-2026-11-12.md) document tested texture
and JPEG95 comparisons that stay outside the app.
The [JPEG coefficient-transition study](image-jpeg-signed-transitions-2026-11-12.md)
replays local generated edits against ordinary processing on the same pixels.
It includes the pinned downloads and exact edit masks for a separate
BG-REAL comparison.
The [exact PNG copy check](image-exact-small-copy-2026-11-12.md) replays matched
native crops and surrounding differences. It includes ordinary photos, generated
edit controls, repeated structures and duplicate graphics in Chrome captures.
The [cross-camera HDR comparison](image-gainmap-cross-camera-2026-11-12.md)
reopens ten licensed originals and verifies the gain-map offsets, the exact
regions in derived views, and false highlights on untouched photos.

The [earlier-video comparison](video-reference-comparison-2026-11-11.md#replay-without-the-app)
re-decodes two supplied clips and checks the JSON, native picture timestamps,
before/during/after previews, and exact soundtrack excerpts. On supported
pairs it also replays the short local picture range and its native crops.
An additional Python and FFmpeg command independently checks the exported local
interval and all six native crops without installing the app or model weights.
The [paired soundtrack timing study](video-paired-soundtrack-transfer-2026-11-12.md)
replays measured shifts between identical-picture copies and verifies the
exact earlier and submitted audio excerpts and their corresponding picture.

The [brief face-change study](video-face-return-fine-scene-2026-11-12.md)
measures whether additional background support recovers an interval around a
face change. Its reader checks the exact boundaries and original face crops
and reports separate outcomes for authentic scene cuts.
The [two-range face-change study](video-face-two-returns-2026-11-12.md)
checks two separate edits in the same file alongside single edits, ordinary
processing and genuine picture cuts. Its replay verifies native pictures and
the authored edit locations separately from the signed app output.
The [720p30 extension](video-face-two-returns-hd-2026-11-12.md) independently
reopens the source clock and both guarded boundaries on the higher-resolution
delivery, including two successive encodes.
The [paired native face edges study](video-face-paired-edge-transfer-2026-11-12.md)
recalculates how a change and return refine an existing face interval. Its
native replay checks exact before, during and after pictures in H.264 and in
the separate HEVC confirmation. Genuine speaker cuts remain visible in the
baseline counts.
The [six-picture proposal study](video-face-unrouted-short-transfer-2026-11-12.md)
recounts exact interval gains and ordinary cutaways separately from inherited
Full findings. Native replay reopens the representative source frames and
both nearby motion measurements.
The [picture-cut context study](video-spatial-cut-context-transfer-2026-11-12.md)
replays both ends of ordinary speaker cutaways flagged by the face check.
Its evidence reader verifies the surrounding picture against the original.
The [subject-context comparison](video-face-subject-context-2026-11-12.md)
checks whether the area below the face helps recognize ordinary speaker cuts.
The [near-head transfer](video-face-near-head-transfer-2026-11-12.md) adds
independent source groups and measures confounds from zoom and brightness.
The [brief face-check coverage verification](video-brief-face-coverage-2026-11-12.md)
replays a localized case, a measured scan without a range and an unassessed
scan through prepaid Full analysis and the signed reports.
The [zoom and brightness comparison](video-face-context-correction-2026-11-12.md)
separates camera adjustments from speaker cuts and local face swaps.
The [scale-only follow-up](video-face-scale-context-holdout-2026-11-12.md)
measures the same explanation at known edit times and at ranges selected by
the Full worker on a different set of speaker pairs.
The [compression repeat study](video-face-compression-stability-2026-11-12.md)
rebuilds single and tandem encodes, checks exact face crops, and reproduces
both appearance measurements at the original time interval.
The [motion-aligned mouth detail study](video-mouth-temporal-detail-2026-11-12.md)
checks native mouth crops and PCM intervals. It measures both missed
source-compatible redraws and flags on ordinary video.
The [fixed-position picture detail study](video-mouth-fixed-residual-2026-11-12.md)
checks the same native coordinates across an interval and its two flanks.
Its replay checks exact crops and PCM ranges, including a separate reserve
of redraws that preserve source-time mouth motion.
The [source-aligned writing study](writing-source-aligned-transfer-2026-11-12.md#replay)
provides a standard-library reader for 204 decisions and source positions
for reconstructing the publisher versions and matched revision controls.
The [clipped token-rank comparison](writing-clipped-rank-transfer-2026-11-12.md)
recalculates 484 decisions on matched writing revisions from a public scalar
receipt. Native source references and offsets support a separate input audit.
The [source-conditioned comparison](writing-source-conditioned-2026-11-12.md)
checks exact edited sentences against unchanged and incorrect earlier context.
Its public replay compares the resulting regions with a simple lexical control.
The [additional writing-response verification](writing-unselected-responses-2026-11-12.md)
checks source locations and actual responses for scored text beyond the
selected passage cards, including known edits that the detector missed.
The [two-sentence draft review](writing-draft-grammar-2026-11-12.md)
replays paired responses and genuine revisions. The report and export point to
the exact changed sentences in both drafts.
The [single-file vocabulary comparison](writing-local-lexical-transfer-2026-11-12.md)
replays sentence rankings and exact ranges across two held publishers. It
checks whether any gain survives ordinary same-work substitutions.
The [short HTML reply comparison](writing-html-reply-short-layouts-2026-11-12.md)
checks the same writing inputs in plain text and HTML-only email. Its
native replay verifies the exact MIME body and scored text ranges.
The [mixed PDF writing check](writing-mixed-pdf-page-transfer-2026-11-12.md)
replays writing responses and exact source locations on intact selectable
pages of mixed PDFs. Its receipt records ordinary agreement and skipped
cross-page inputs.
The [paired writing check beside a paragraph break](writing-paired-center-transfer-2026-11-12.md)
compares changes in exact sentence and context readings. Its replay checks
both draft positions and reports held genuine revisions separately. A second
implementation in Node.js verifies the same decisions and source spans.
The [semantic continuity study](writing-semantic-continuity-transfer-2026-11-12.md)
recomputes exact passage overlap and false flags for fresh PLOS Biology
excerpts and two generated rewrite families.
The [paired writing response comparison](writing-instruct-base-passage-transfer-2026-11-12.md)
checks response differences on identical target tokens and replays the held
publisher's false flags and missed rewrites.
The [paired neural speech edit pilot](audio-partialedit-dense-transfer-2026-11-12.md)
checks overlapping speech windows across two edit generators, matched genuine
utterances and two background mixes. The replay recounts localized edits and
flagged genuine time.
The [overlapping speech confirmation](audio-dense-evidence-confirmation-2026-11-12.md)
recounts matched gain-only and traffic deliveries from six generator families.
It reports edit coverage and genuine time separately for the sparse, dense and
combined window rules.
The [narrower speech evidence study](audio-focused-window-transfer-2026-11-12.md)
recounts 98 new exact deliveries with rain and train sound, including isolated
backgrounds and marked genuine speech in partly edited files.
The [frequency-view study](audio-subband-partial-transfer-2026-11-12.md) replays
separate band points and tests whether either rescues edits in the same noisy
deliveries. Its native check also recomputes the filtered PCM window hashes.
The [adjacent speech-frame study](audio-adjacent-frame-transfer-2026-11-12.md)
replays the local evidence windows, genuine-group false flags, and exact
sample counts under the same loud rain and train conditions.
The [transient-reduction comparison](audio-transient-reduction-transfer-2026-11-12.md)
recreates the processed inputs on the original sample clock and checks whether
two neighboring windows recover any additional known edits.
The [enhanced-audio blend comparison](audio-enhancement-blend-transfer-2026-11-12.md)
verifies the same native time intervals and checks whether restoring some of
the background preserves useful locations or reduces false flags.
The [native word-clock qualification](grid-reserve-word-clocks-2026-11-12.md)
aligns publisher word intervals with original embedded speech and replays four
ordinary mouth changes with unchanged soundtracks.
The [fresh neural mouth fixtures](grid-reserve-neural-mouth-2026-11-12.md)
replay two original-time and two shifted redraws against their source
closures, native pictures, and unchanged soundtracks.
The [detector readout on those fixtures](grid-reserve-neural-mouth-readout-2026-11-12.md)
repeats current and experimental timing checks on those exact clips and
verifies source crops where an inspection range appears.
The [speech-sound and mouth-shape comparison](grid-phone-viseme-timing-2026-11-12.md)
replays signed timing errors on 96 controlled shifts and 144 ordinary
controls. It also verifies the source pictures and wrong-direction responses
from the separate neural-redraw panel.
The [mouth-state timing study](grid-articulatory-timing-2026-11-12.md)
compares local timing estimates on 108 authored shifts and 162 ordinary
controls. Its reader reproduces signed displacement errors, including the
local filtering controls that produced false alerts.
The [marked earlier-video timing study](grid-paired-closure-timing-2026-11-12.md)
replays independently measured mouth movement and speech changes at a marked
event, with coherent retiming and ordinary processing controls.
The [single-video release timing study](grid-single-clip-release-timing-2026-11-12.md)
reconstructs 120 native deliveries and verifies how often nearby acoustic
transients can support the same visible speech event.
The [phoneme qualification](video-phone-release-qualification-2026-11-12.md)
compares blind and supplied-transcript decoding, and replays acoustic timing
against the original PCM and publisher word intervals.
The [fresh release timing study](grid-fresh-release-qualification-2026-11-12.md)
seals independent waveform and picture observations before moving the audio
and compares both blind checks with a supplied-reference waveform.
The [marked earlier-copy transfer](grid-marked-release-transfer-2026-11-12.md)
compares two waveform templates selected using just a marked source frame
and verifies the reported native assets against both copies.
The [independent marked timing holdout](grid-marked-release-holdout-2026-11-12.md)
tests the shorter excerpt as a fallback on six separate speaker groups and
replays every picture, crop, and sound excerpt used to measure the offset.

The [earlier-audio comparison](audio-reference-comparison-2026-11-11.md#replay-without-the-app)
replays changed sample ranges, difference waveforms, and playable excerpts
from two continuous, clock-matched recordings.

The [two-check writing replay](writing-checkpoint-agreement-2026-11-11.md#local-replay)
checks whether both detectors responded to the same source ranges and reproduces
the agreement and false-flag counts on the retained publisher panels.
The [editorial transfer](writing-agreement-ordinary-edit-2026-11-11.md#replay)
adds dated before-and-after revisions and separate scientific publisher controls.
The [email reply layout replay](writing-reply-paragraph-transfer-2026-11-12.md#replay)
reopens the raw MIME part, verifies the measured text against native body
positions, and records when reformatting changes sentence segmentation.
The [exact per-email replay](writing-reply-native-batches-2026-11-12.md)
repeats the second check in source order and follows the resulting locations
through the local paid worker, public exports, and source-bound evidence bundle.
The [HTML email source replay](writing-html-reply-transfer-2026-11-12.md)
reopens the decoded HTML part and verifies exact passage locations while
keeping trailing quoted replies out of the second check.
The [scanned PDF study](writing-ocr-scan-transfer-2026-11-12.md) measures
which exact classifier inputs survive recognition and checks whether a
signed report can be replayed against the original scan.
The [conditional writing study](writing-conditional-context-transfer-2026-11-12.md)
compares the same target tokens after three contexts and independently
recalculates held location counts from the saved measurements.
The [archive boundary comparison](writing-archive-changed-edge-2026-11-12.md)
reconstructs exact submitted and reference coordinates after a word changes
next to a lost paragraph break and checks open-set and ambiguous controls.

The [audio supplement example](#review-an-audio-supplement-by-acoustic-condition)
uses authored scores to show how an acoustic routing error can add a genuine
false flag.

The [mono speech-interference study](audio-mono-dereverb-transfer-2026-11-11.md)
reconstructs matched noisy inputs and checks which true and false voice flags
change when reducing predictable late sound.
The [competing-voices disagreement study](audio-disagreement-transfer-2026-11-12.md)
reconstructs exact mixtures across calibration and two held meeting rooms.
It compares four voice policies and records when the second opinion cannot act.
The [stereo removed-speech comparison](audio-removed-source-transfer-2026-11-12.md)
rebuilds the sound taken out by interference suppression. The replay measures
target-speech leakage separately for centered and displaced recordings.
The [short edits in longer voice recordings](audio-overlap-window-transfer-2026-11-12.md)
replays four window rules on exact six-second waveforms, with known edit times
and matched loud competing speech.
The [long-room voice confirmation](audio-spectra-long-room-holdout-2026-11-12.md)
checks native waveform bindings, the exact shared and exclusive window samples,
and held performance at fixed and equal false-alarm counts.
The [enrolled-speaker extraction study](audio-enrolled-target-transfer-2026-11-12.md)
checks how much of each known insert remains after suppressing competing speech.
The [matching-speaker source qualification](audio-same-speaker-clone-sources-2026-11-12.md)
verifies exact foreground, enrollment, and clone audio from the same declared
source speaker in separately assigned selection and held groups.
The [matching-speaker enrollment test](audio-same-speaker-enrollment-transfer-2026-11-12.md)
replays 42 mixtures and measures preservation of the known clone interval
when two or more genuine background speakers are talking.
The [original-audio boundary study](audio-original-pcm-boundary-transfer-2026-11-12.md)
replays exact ranges and compares useful locations at fixed and matched
false-group counts without separating the soundtrack.
The [variable-duration audio transfer](audio-variable-boundary-transfer-2026-11-12.md)
compares four span durations on new expressive inserts and paired genuine
edits in two rooms with competing speech.
The [short native-response transfer](audio-short-native-response-transfer-2026-11-12.md)
runs short windows on held room recordings and compares the locations at
fixed and matched false-alarm counts.
The [MP3 residual study](audio-mp3-residual-transfer-2026-11-12.md)
checks whether a compact background-sensitive feature adds useful edit
locations. Its standard-library reviewer verifies the group operating point,
extra useful locations, and the ordinary flags that prevented release.
The [vocal-event source roster](audio-vocal-event-sources-2026-11-12.md)
reproduces a small set of publisher-labeled generated events and individually
licensed recorded controls, with exact byte ranges, file hashes, and decoding.
The [paired vocal-event mixtures](audio-vocal-event-mixtures-2026-11-12.md)
replay changes in the voice decision when a known event is added to the same
speech. The ledger reports the WAV and MP3 renderings separately.
The [RADAR transfer and longer-file confirmation](audio-radar-transfer-2026-11-12.md)
start from bounded prefixes of two public archive splits. A model-free
reader reopens official labels, native clocks and all matched short
mixtures, then reproduces the decision counts from the saved responses.
The [WAV transfer replay](audio-radar-transfer-2026-11-12.md#local-evidence)
compares native decode and exact voice inputs in WAV and FLAC. The
[publisher-labeled partial speech comparison](audio-partial-publisher-transfer-2026-11-12.md)
uses a separate public archive prefix, six generator families and matched
traffic and café scenes. Its reader checks the decision geometry without
model weights and can also verify the local PCM inputs.
The [enhanced-speech comparison](audio-noise-views-transfer-2026-11-12.md)
keeps the development and held noise recordings separate. Its public
replay recounts the fixed decisions and shows the best held gain allowed
by each added genuine false-alarm budget.
The [staggered speech and noise-reduction comparison](audio-noise-window-transfer-2026-11-12.md)
separately replays native scheduling and spectral reduction on 18 labeled
partial edits from six generators. It verifies the original-response
calibration, localized ranges and additional genuine time highlighted.
The [short speech and wider context comparison](audio-short-coarse-transfer-2026-11-12.md)
reconstructs the duration-specific operating points and compares held
sample locations and flags with and without wider-window support.
The [two-enhancer replication](audio-nemotron-noise-transfer-2026-11-12.md)
provides a model-free replay for four held sound scenes and a separate
confirmation at identical decision points. It distinguishes additional
flags from false alarms inherited from the original detector.
The [speech-context comparison](audio-remote-neighbor-transfer-2026-11-12.md)
compares recurring acoustic contexts on the original mixtures and verifies
fresh and previously tested source groups separately.
The [larger replication](audio-remote-neighbor-replication-2026-11-12.md) keeps
the same cutoffs on 12 new edits from three generator families and compares
localization within the same admitted time coverage.
The [measured-room stereo comparison](audio-measured-spatial-transfer-2026-11-12.md)
reconstructs the exact PCM, verifies foreground preservation, and measures
short-edit localization under calibrated, wrong-side, and wrong-room conditions.
The [controlled room-speech attenuation study](audio-room-reference-transfer-2026-11-12.md)
reconstructs each changed waveform independently and checks whether improved
foreground reconstruction recovers short synthetic inserts.
The [temporal voice-head comparison](audio-partial-residual-transfer-2026-11-12.md)
replays the fitted and unfitted checks on exact short edits with competing
speech. The public reader verifies location counts without audio or weights.
The [environmental sound embedding comparison](environmental-embedding-transfer-2026-11-12.md)
recounts separate isolated and mixed operating points and held rankings.
It preserves the trade-off between additional generated-background flags
and ordinary backgrounds flagged during genuine speech.

The [video tracking evidence fixture](video-face-tracking-context.md) exercises
the same recorded values in a saved packet and report without downloading
detector weights. The [matched image cue](image-bg-real-detail-deficit-v265-result.md)
and [passage context replay](text-context-shift-v266-result.md) give reviewers
two recent examples of what a failed localization candidate looks like, with
frozen methods, exact aggregate results, and limitations. Full-media replay
requires the separately obtained source inputs and retained local bindings.
The [multi-scale image crop trial](image-crop-multiscale-transfer-2026-11-12.md)
measures what exact crop geometry can cover and which regions the detector
actually selects on held photos and screenshots.
The [native tile agreement trial](image-native-tile-agreement-transfer-2026-11-12.md)
rebuilds each image tile in two layouts and compares regional localization
with false flags on ordinary photographs and screenshots.
The [screenshot viewport comparison](image-viewport-routing-transfer-2026-11-12.md)
tests the same image check on the whole screenshot, a proposed photo crop, and
the exact photo area, then verifies the effect of the surrounding frame.
The [fixed-photo framing test](image-viewport-frame-perturbation-2026-11-12.md)
changes only the frame around matched clean and edited photos and replays both
the original and newly selected regions.
The [earlier-copy screenshot comparison](image-reference-embedded-screenshot-2026-11-12.md)
aligns the submitted photo within a larger screenshot, then verifies the native
differences inside the inset. It found more edits than the current comparison
and fell short of the cross-editor coverage target.
The [embedded JPEG preview comparison](image-thumbnail-localization-2026-11-12.md)
uses a retained preview within a single upload to locate localized differences.
It replays the measured regions and distinguishes ordinary edits, processing
controls, and incompatible previews.
The [EXIF orientation transfer](image-thumbnail-orientation-2026-11-12.md)
replays rotated main images and both preview-storage conventions. It checks
display-oriented main crops and original stored preview pixels separately.
The [mirrored EXIF transfer](image-thumbnail-mirror-2026-11-12.md) adds four
reflections and checks both stored preview conventions. Its public readback
distinguishes useful feathered-region locations from missed compact edits.
The [separate whole-image panel](image-whole-fresh-holdout-2026-11-12.md)
compares the current Full result and a positive supplementary cue on 34
generated and ordinary source groups in two delivery formats. Its reviewer
replays the outcome counts and records unavailable Full results separately.
The [native PNG and screenshot studies](image-whole-png-transfer-2026-11-12.md)
replay the separate camera, generated-image and screenshot results and check
whether existing image-geometry rules distinguish photo-rich screenshots.
The [whole-image view replay](#verify-the-inspected-views-from-a-jpeg)
checks the actual exported crop pixels and their bounds against the submitted JPEG.
The [embedded-preview footprint study](image-thumbnail-extent-transfer-2026-11-12.md)
compares the crop extents on 222 deliveries and verifies the exact marked
cells and both source crops from the current evidence.
The [Ultra HDR gain-map pilot](image-gainmap-localization-2026-11-12.md)
tests two ways to locate local changes in JPEGs that retain a gain map but
have no embedded thumbnail. Its source replay records a camera-original false
flag and the limited device coverage.
The [two-photo difference pilot](image-embedded-pair-difference-transfer-2026-11-12.md)
replays source-pixel differences when one PNG contains two matching photo areas.
The [separate confirmation](image-embedded-pair-difference-confirmation-2026-11-12.md)
checks new layouts and documents the ambiguity when a third matching area is present.
The [third-copy holdout](image-embedded-pair-third-guard-2026-11-12.md)
tests when to withhold the comparison while retaining the exact differences
in valid pairs.
The [screenshot-size transfer](image-embedded-pair-canvas-transfer-2026-11-12.md)
replays 870 exact pixel inputs and measures ambiguous matches involving a
third view resized by different amounts.
The [dense third-view comparison](image-embedded-pair-dense-third-2026-11-12.md)
tests a separate patch verification rule with additional source photos
and records the original whole-area verification failures.
The [quality-90 earlier-copy study](image-reference-jpeg90-transfer-2026-11-12.md)
rebuilds both JPEG deliveries, checks mapped native crops, and verifies the
paid result's API, PDF, and JSON evidence in the local app.

The [strong-background speech test](audio-noise-speech-gate-strong-noise-v269-result.md)
documents how to reproduce a comparison between the current speech admission
rule and a fixed fallback. Its aggregate distinguishes genuine and synthetic
coverage at two noise levels. The [downstream comparison](audio-noise-speech-rescue-downstream-v270-result.md)
scores the rescued clips and the already admitted comparison group against
their own clean versions.

The [science-explainer intake](text-elife-source-intake-v272-result.md) gives
reviewers a new five-source writing cohort with work-level attribution, an
immutable publisher revision and exact replacement targets. The receipt records
all nine dispositions and contains no excerpts. The [rewrite result](text-elife-rewrites-v273-result.md)
records all four unavailable attempts. The [detector comparison](text-elife-passage-score-v275-result.md)
reports localization, coverage and raw position ranking separately.
The [fresh transfer roster](text-elife-transfer-intake-v286-result.md) adds 26
groups with separately checked work licenses and fixed rewrite targets. The
[transfer rewrite construction](text-elife-transfer-rewrites-v289-result.md)
keeps the source roles and all 52 planned attempts in the record.
The [relative paragraph comparison](text-elife-relative-paragraph-v292-result.md)
reports recovered edit locations and false highlights on separate confirmation articles.
The [editorial-control intake](text-elife-editorial-controls-v301-result.md)
reports exact source and offset checks for a new set of ordinary edits, including
the coverage shortfall that stopped the study before scoring.
The [writing and measured-room transfer checks](writing-and-room-transfer-2026-11-11.md)
record the false-flag and recall trade-off from an archived writing replay
and the separate acoustic geometry preflight.
The [fresh editorial and mixed-passage test](text-editorial-transfer-2026-11-11.md)
reports newly measured false highlights and source-localization results from two generator families.
The [paragraph context transfer](writing-paragraph-context-transfer-2026-11-11.md)
compares isolated and neighboring paragraph responses on 11 new source articles,
with separate counts for originals, ordinary edits, and two generators.
The [eight-word insert comparison](writing-phrase-context-stability-2026-11-12.md)
replays exact phrase locations across two source contexts and compares their
responses with unchanged articles and ordinary substitutions.
The [Python PEP replication](writing-pep-phrase-replication-2026-11-12.md)
replays the same frozen checks on a separate set of English technical prose.
The [fiction transfer](writing-fiction-paragraph-transfer-2026-11-11.md) applies
the same frozen rule to a new genre and counts ordinary human replacements.
The [practical DOCX comparison](writing-practical-notice-transfer-2026-11-11.md)
binds both human and generated replacements to exact extracted paragraph positions.
The [correspondence holdout](writing-correspondence-transfer-2026-11-11.md)
uses separate archival sources and synthetic multipart email containers. It
counts both useful paragraph locations and ordinary replacements flagged.
The [modern bulletin holdout](writing-modern-bulletin-transfer-2026-11-11.md)
includes exact quote and template variants across six separate publishers.
The [article context holdout](writing-article-context-transfer-2026-11-11.md)
compares a paragraph-only read with the usual surrounding input and a fixed
paragraph-omission experiment on six source-disjoint web features.
The [transcribed speech transfer](writing-speech-paragraph-transfer-2026-11-11.md)
checks whether an isolated paragraph preserves known rewrite locations while
reducing false highlights on oral histories and prepared public addresses.

The [image edit construction check](image-flux-reference-edit-v271-result.md)
uses public, authored RGB fixtures to verify exactly which pixels change when a
generator output is composed into a source region. A separate verifier checks
the pinned bundle without running the model or accessing project photos. The
[licensed photo pilot](image-flux-photo-preparation-v276-result.md) carries the
same geometry checks into two exact edits with six matched ordinary controls.
The [paired detector comparison](image-flux-detector-comparison-v280-result.md)
reports the same edit footprints under PNG and JPEG75, including false highlights.
The [six-photo expansion](image-flux-photo-expansion-v285-result.md) extends
that design to new licensed source groups. Its [format-specific comparison](image-flux-product-comparison-v288-result.md)
uses the actual PNG and JPEG routes and checks the resulting masks against exact truth.
The [copy corroboration fixture](image-copy-corroboration-v295-method.md)
checks whether a proposed region has a matching donor elsewhere in the image
while retaining the original highlight.
The [licensed photo comparison](image-copy-transfer-v298-result.md) measures
donor recovery on fresh ordinary edits and unchanged controls in two formats.
The [native copy follow-up](source-evidence-native-coverage-2026-11-11.md#images)
compares the supplied-region and automatic proposal paths on the same images,
with separate counts for translated and reflected copies.
The [regional patch response comparison](image-relative-patch-transfer-2026-11-11.md)
reports a compact paired-feature method's localization and false highlights
when the edit generator and source groups change.
The [small-insertion transfer](image-bg-real-insertions-2026-11-11.md)
compares source-localized results with unedited originals and matched
processing controls. Its exact precompression footprint is kept for both formats.
The [two-editor background patch panel](image-background-patch-transfer-2026-11-11.md)
tests PNG and JPEG75 with region-matched blur and processing controls.
The [native copy comparison](image-translation-patch-transfer-2026-11-11.md)
tests two crop sizes on 16 held-out source groups and reports hard controls
with natural repetitions in both PNG and JPEG90.
The [boundary blend comparison](image-boundary-blend-transfer-2026-11-11.md)
reconstructs the exact masks for sharp and feathered edits and verifies the
format-specific current checks against matched controls.
The [noise-guided image comparison](image-noise-region-candidate-2026-11-11.md)
matches the same input digests and evaluates a separate checkpoint at its
publisher point and two points fitted on disjoint controls.
The [fine-detail comparison](image-wavelet-region-transfer-2026-11-11.md)
adds 12 independently qualified photo groups and compares two sealed regional
checks on exact sharp and feathered edits after PNG and JPEG delivery.
The [residual color comparison](image-residual-color-transfer-2026-11-11.md)
reports exact regional overlap and control flags for three spatial-band arms
on two panels of sharp and feathered image edits.
The [semantic context comparison](image-semantic-context-transfer-2026-11-11.md)
compares a forensic patch response with its response relative to distant,
semantically similar patches on two licensed panels.
The [three-layer image comparison](image-multidepth-transfer-2026-11-11.md)
publishes exact region counts and a standard-library replay of all 384 held
decisions without the source media. A second verifier rebuilds native masks
when the evaluator has the original intake.
The [learned residual comparison](image-residual-patch-transfer-2026-11-11.md)
rebuilds exact edited-pixel masks for 496 deliveries, including seven separately
qualified SDXL and Kandinsky sources. The [remote crop transfer](image-remote-crop-editor-transfer-2026-11-11.md)
replays the same inputs and verifies whether suppressing broad processing
responses changes useful localizations.
The [paired PNG and JPEG crop comparison](image-crop-delivery-consistency-2026-11-12.md)
checks whether converting the same PNG to JPEG corroborates the native crop,
with separate counts for lost edit locations and added ordinary flags.
The [flexible single-image boundary comparison](image-boundary-extent-transfer-2026-11-12.md)
has a standalone public count verifier and native source replays. It records
small- and large-region tradeoffs and the separate follow-up on generated edits.
The [photo and graphics stress test](image-small-boundary-control-stress-2026-11-12.md)
replays 120 exact views and separates correct paste locations from ordinary
photo and graphics flags. Its public verifier needs no media or weights.
The [photo-admission screenshot test](image-small-boundary-screenshot-admission-2026-11-12.md)
seals the screenshots that reach the fallback before candidate scoring. Its
public reader reproduces the added paste locations and screenshot flags.
The [interior flatness comparison](image-small-boundary-interior-transfer-2026-11-12.md)
adds separate low-texture pastes and flat, translucent, and patterned
interface panels, with another standard-library result replay.
The [image flip comparison](image-regional-flip-transfer-2026-11-11.md) tests
whether the sensitive detector reproduces a region under a reversible transform.
It separately reports useful regions, processing controls and untouched photos.
The [stereo speech coverage comparison](audio-channel-speech-gap-transfer-2026-11-11.md)
checks exact channel samples and annotated speech in intervals left unscored
by the mono voice check. The [blind writing probe](writing-blind-rewrite-probe-2026-11-11.md)
checks source-character positions and human false highlights on new rewrites.
The [dated journal body comparison](writing-journal-body-transfer-2026-11-11.md)
replays exact earlier and later text from 21 distinct work groups and checks
localization on 32 separately generated one-sentence edits.
The [archived PEP writing pilot](writing-pep-editorial-intake-2026-11-11.md)
replays five dated work histories and compares exact sentence locations on
generated rewrites and ordinary editorial revisions.
The [archived technical prose transfer](writing-archived-passage-transfer-2026-11-12.md)
replays 125 held passages, including one historical revision, and compares a
fixed sentence ranker and its combination with the current check.
The [multi-paragraph writing transfer](writing-multiparagraph-localization-transfer-2026-11-12.md)
replays native article passages and constructed email equivalents, including
unrelated paragraphs and quoted text.
The [independent sentence comparison](writing-multiblock-independent-2026-11-12.md)
checks all eligible interior sentences in multi-paragraph text. Its reader
replays exact edit coverage, fitted comparisons, and false alarms on original
and ordinary revised passages.
The [held eLife confirmation](writing-elife-exact-sentence-confirmation-2026-11-12.md)
replays the same frozen sentence point on a separate publisher and two new
generation families, with exact source bindings and matched ordinary revisions.
The [adjacent-context writing study](writing-shared-sentence-transfer-2026-11-12.md)
measures whether overlapping context windows locate entire rewrites, with
ordinary editorial controls and the geometry of measured source sentences.
The [public-archive writing comparison](writing-archive-source-evidence-2026-11-12.md)
replays exact archive matching and source offsets, including a publisher
correction, missing-source controls, and quoted text that must be ignored.
The [joined paragraph study](writing-archive-joined-layout-2026-11-12.md)
replays source identification and exact changed-word coordinates when an
archival text copy loses its paragraph breaks.

The [offline video interval verifier](video-native-interval-replay-2026-11-11.md)
checks the source clock and exact before, during and after pixels with Python
and FFmpeg. Its local receipt distinguishes native source verification from
retrospective scoring.
The [short face-return transfer](video-short-return-margin-transfer-2026-11-11.md)
checks a fixed margin at three placement phases, then counts distinct secondary
ranges in a separate retrospective analysis. Its reviewer checks native anchors
and control outcomes independently of the scorer.
The [single-file mouth texture trial](video-mouth-native-texture-transfer-2026-11-11.md)
counts separately localized and missed redraws, ordinary retouching flags,
and source-compatible visible motion. The [paired H.264 transfer](video-reference-h264-earlier-transfer-2026-11-11.md)
verifies two independently encoded video copies, including tandem re-encodes
and unrelated clips with the same picture clock. The [second held H.264 group](video-reference-h264-grid-reserve-2026-11-11.md)
checks native crops and clocks on separate 360 × 288, 25 fps sources with
ordinary brightness changes, blur, and multiple encoding controls.
The [scene-referenced face motion comparison](video-webcam-motion-transfer-2026-11-11.md)
checks whether genuine head movement can explain flagged ranges and tracks
which clips lack enough stable background for that measurement.
The [lip-sync coverage comparison](video-lipsync-mouth-coverage-2026-11-11.md)
checks held mouth frames on real moving footage, matched to original and
recompressed controls, with a separate authored false-collapse regression.
The [local audiovisual timing comparison](video-lipsync-local-contrast-transfer-2026-11-12.md)
reconstructs 160 exact soundtrack deliveries and checks signed timing estimates,
ordinary controls, native mouth crops, and PCM slices.
The [mouth-pose comparison](video-mouth-pose-recurrence-2026-11-12.md)
tests short neural redraws and independently reopens the seven pairs with
source-compatible mouth motion, including their native pictures and PCM.
The [six-picture face-swap comparison](video-face-six-picture-transfer-2026-11-12.md)
re-creates the authored H.264 swaps and hard cuts and compares denser sampling
with the additional consecutive support rule.
The [six-picture duration transfer](video-face-six-duration-transfer-2026-11-12.md)
rebuilds 64 held H.264 deliveries byte for byte and verifies native ranges and
source crops at two additional clip lengths.
The [six-picture codec comparison](video-face-six-codec-transfer-2026-11-12.md)
audits VP9 and HEVC deliveries and replays the same false cut ranges against
matched H.264 inputs and native background pixels.
The [background comparison on VP9 and HEVC](video-face-codec-background-transfer-2026-11-12.md)
rebuilds held swaps and ordinary cuts, then replays their source crops,
timing, and background measurements.
The [720p30 face-change study](video-face-720p30-transfer-2026-11-12.md)
compares the Full analysis with two sampling rates on short H.264 clips.
It replays the exact source evidence and records the multiple-face
abstentions and the remaining missed longer inserts.
The [timing and codec follow-up](video-timing-coverage-and-codec-holds-2026-11-11.md)
records when a single timing window leaves change unassessed and compares
exact and approximate picture holds on the same lossy and lossless deliveries.
The [extended repeated-picture check](video-extended-picture-hold-2026-11-12.md)
includes test clips that FFmpeg generates locally and a replay of every
inspected frame when the original video accompanies the signed export.
The [repeated sequence comparison](video-exact-sequence-transfer-2026-11-12.md)
replays matching decoded frames in two separate ranges, with static holds,
different timing, and lossy copies as separate controls.
The [similar moving-picture comparison](video-lossy-sequence-transfer-2026-11-12.md)
checks for repeated picture content and motion after re-encoding. Its held
results include controls for ordinary cuts, holds, reversals and changed
frame spacing. The signed export can replay both locations from the upload.
The [360p transfer](video-motion-sequence-360p-transfer-2026-11-12.md) applies
the fixed method to H.264 and VP9 at 25 fps, with eight published controls
covering real repeated activity and editing. It verifies the new 225-frame
prefix and replays the exact earlier 180-frame package format.
The [18-second repeat study](video-motion-late-transfer-2026-11-12.md)
replays both time ranges and all four source pictures, and reports which
opening matches survive a later clock gap. A standard-library reader
checks the held outcomes for both codecs.
The [longer timing replay](video-longer-timing-transfer-2026-11-11.md)
tests native windows with actual timing-change opportunity and separately reports
measured windows, redraw overlaps, ordinary controls, and unassessed inputs.
The [speaker-attribution comparison](video-speaker-attribution-transfer-2026-11-11.md)
checks visual speech separation, lip geometry and existing timing on known
voiceovers, timing interventions and continuous camera controls.

The [stereo audio preflight](audio-native-stereo-preflight-v274-result.md)
checks exact channel recovery and source clocks on five authored WAVs. The
same tool can also verify the four publisher background archives and the
fixed development and holdout windows. The [paired stereo benchmark](audio-native-stereo-pairs-v277-result.md)
records construction and exact readback of all 385 planned files. The
[development speech-gate comparison](audio-native-stereo-gate-v279-result.md)
compares mono admission and a fixed acoustic channel choice without using
authenticity scores for selection. The [downstream comparison](audio-native-stereo-authenticity-v282-result.md)
reports correct, wrong and inconclusive directions on those recovered clips
and on the comparison group. The [AI-only policy replay](audio-ai-only-rescue-v283-result.md)
counts the directions that a fixed mono-plus-channel rule would change. The
[200-case holdout](audio-native-stereo-holdout-v284-result.md) compares that rule
with mono on separate sources and reports the effects by noise condition.
The [AMI intake](audio-ami-native-stereo-v291-result.md) adds 24 naturally
recorded two-microphone windows for a separate genuine-audio test.
Its [stereo transfer comparison](audio-ami-stereo-transfer-v293-result.md)
reports identical decisions and unchanged coverage across both rules.
The [room-transfer preflight](audio-room-transfer-preflight-v296-method.md)
verifies a renderer for asymmetric background noise and confirms the archive
inventory for a future paired comparison with two voice generators.
The [measured room intake](audio-room-assets-v299-result.md) qualifies the
documented Surrey source positions and records the open second-room geometry.
The [BBC holdout intake](audio-bbc-room-assets-v303-result.md) verifies a
separate room with documented front and lateral positions.
The [matched Surrey and BBC comparison](audio-measured-surrey-activity-transfer-2026-11-11.md)
reports speech-window admission on the same foregrounds and strong background noise.
The [stereo suppression test](audio-stereo-reference-suppression-2026-11-11.md)
adds 36 matched noise-only controls and measures whether channel-difference
filtering recovers synthetic or genuine speech windows.
The [waveform fallback transfer](audio-waveform-fallback-transfer-2026-11-11.md)
tests repeat localization across three held meeting groups and twelve separate
speakers, with a standalone reviewer that checks source clocks and paired spans.
The [second voice representation](audio-wavlm-strong-traffic-2026-11-11.md)
keeps the admission decisions fixed and compares clean and −10 dB traffic
responses from two voice representations, with retained misses and exclusions.
The [two-background transfer](audio-wavlm-two-background-transfer-2026-11-11.md)
tests the same rule with distinct background recordings and records how each
condition changes flags and unassessed coverage.
The [independent voice veto comparison](audio-independent-noise-veto-2026-11-11.md)
replays 212 held audio views to quantify both removed false alerts and lost
correct detections under strong noise, Opus, and audio restoration.
The [paired noise correction](audio-paired-nuisance-transfer-2026-11-11.md)
checks separate fit, calibration, selection and held roles and replays 77 exact
room deliveries under the same fixed correction.
The [temporal corroboration trial](audio-temporal-corroboration-2026-11-11.md)
uses three native windows per track and holds the speech gate fixed when
comparing the outcomes under two additional background recordings.
The [simulated call-processing comparison](audio-communication-transfer-2026-11-12.md)
verifies the exact original and processed windows and reconstructs the decision
counts with a standard-library reviewer.
The [fresh call-processing replication](audio-communication-replication-2026-11-12.md)
uses separate foreground groups and meeting participants and verifies the
retained locations and all false-alarm group counts.
The [Full analysis call-processing test](audio-communication-serving-grid-2026-11-12.md)
replays the comparison on the four-second windows the app serves. The pairing
lost the confirmation cohort's only overlapping original flag and left the
same ordinary false-alarm groups.
The [short-window voice test](audio-edge-window-transfer-2026-11-12.md)
adds clips with an edit near either end under competing speech. It verifies
separate ordinary operating points and exposes the localization and false-alarm
trade-off when transferring the earlier experimental point to a denser grid.
The [prospective grouped replication](audio-edge-window-grouped-2026-11-12.md)
sets both grids to the same ordinary-group allowance before testing new speech
and room combinations. Its replay verifies missed and recovered intervals and
the exact source sample positions.
The [short speech codec comparison](audio-short-codec-transfer-2026-11-12.md)
replays the lossless match and recalculates separate MP3 and Opus points,
held interval matches, and ordinary-group errors using public window responses.
The [separate MP3 confirmation](audio-short-mp3-confirmation-2026-11-12.md)
reuses the stricter MP3 points on different host recordings and source groups.
It verifies the offset between decoded excerpt time and the original MP3 clock.
The [two-microphone noise trial](audio-natural-stereo-coherence-2026-11-11.md)
compares score directions after a fixed filter with two recorded meeting
backgrounds, six genuine readers and three synthetic generator families.
The [stereo component comparison](audio-stereo-component-transfer-2026-11-11.md)
uses two other recorded meetings and explicitly counts admitted genuine false
alarms after two fixed decision rules.

The [video brightness readout](video-luma-context-v278-result.md) uses the
original selected appearance flags, known edit controls and exact native
frame bindings to document where a luma change extended beyond the face.
The [fresh camera qualification](video-vcd2-unseen-controls-v287-result.md)
adds eight unused camera types with independently verified native clocks.
The [matched ordinary controls](video-vcd2-matched-controls-v290-result.md)
verify 24 unchanged, chroma-converted and recompressed views on those clocks.
The [first current-app triplet](video-vcd2-current-app-v294-first-result.md)
records measured appearance coverage and the separate frame-check budget skip.
The [later reference snapshot](video-vcd2-current-app-v294-reference-snapshot.md)
preserves all 20 completed controls and four cases pending resource admission.
The [aligned face detail and motion comparison](video-aligned-face-detail-probe.md)
reuses committed landmarks to evaluate two new measurements on separate
development and confirmation groups. It includes the local replay command and
reports the false alarms that kept both measurements experimental.
The [whole-clip face-swap comparison](video-full-clip-face-swap-transfer.md)
uses pinned crop classifiers on exact common frame inputs, with genuine paired
and camera controls and a separate publisher swap.
The [whole-clip pooling transfer](video-whole-clip-pooling-transfer-2026-11-11.md)
compares four frozen rules across two source-disjoint delivery cohorts. Its
anonymized scalar replay reproduces every operating point and paired count
using only Python. The [facial color-step comparison](video-face-chroma-transfer-2026-11-11.md)
tests a separate native-region measurement on the same verified source inputs.
The [same-source delivery comparison](video-face-delivery-transfer-2026-11-11.md)
holds the source pictures and times fixed across two deliveries and tests the
frozen rule that selected a different check for each delivery.
The [paired face-swap transfer](video-face-pe-transfer-2026-11-11.md) includes a
standard-library score replay on 108 clips. The [prospective confirmation](video-face-pe-confirmation-2026-11-11.md)
applies the frozen rule to separate source groups using identical native faces in both arms.
The [outside publisher and camera transfer](video-face-pe-publisher-2026-11-11.md)
checks known full and partial swaps and ordinary delivery changes. It reports
the four partial variants as one source and camera outcomes by recording group.
The [recorded-noise repetition comparison](audio-repeat-robustness-transfer-2026-11-11.md)
checks exact and similar repeated audio across recorded backgrounds and four
codecs, with matching ordinary controls and anonymized per-group results.
The [frequency-domain separation comparison](audio-frequency-separation-transfer-2026-11-11.md)
keeps the production voice scores and calibrations fixed while measuring one
stereo projection under two kinds of competing speech.
The [voice timestamp replay](audio-clock-voice-window-replay-2026-11-11.md)
holds decoded samples fixed while introducing timestamp breaks and checks the
resulting voice coverage through the signed app report.

The [active PNG transfer](image-active-png-transfer-2026-11-11.md) uses exact
source-aligned masks to check which gains survive on small edited patches with
untouched backgrounds. The [color-residual comparison](image-color-residual-transfer-2026-11-11.md)
shows how many misses come from region selection before a review point is applied.
The [integer image reference comparison](image-reference-comparison-2026-11-11.md#integer-crops-and-translations)
preserves both original coordinate spaces for a corroborated crop or translation.
The [short face texture comparison](video-face-texture-return-2026-11-11.md)
checks appearance and texture on native video clocks with ordinary camera edits.
The [denser timing comparison](video-face-native-boundary-2026-11-11.md) measures
boundary error on a different set of recordings while preserving the existing flags.
The [fixed one-second rule](video-face-short-boundary-confirmation-2026-11-11.md)
tests the resulting timing rule prospectively on separate moving faces and camera controls.
The [short email contrast study](writing-short-email-contrast-transfer-2026-11-11.md)
compares individual sentences and matched context with exact MIME positions.
The [contextual token study](writing-context-token-transfer-2026-11-11.md)
checks whether exact sentence features improve selection on source-separated emails.

The [portable PACC-P replay](audio-pacc-portable-replay.md) runs the 98 genuine
speech controls using publisher files, separately obtained model assets and
caller-owned directories. It compares aggregate decisions and clean-to-noisy
transitions against the published result. Qualification runs without model
weights, and the aggregate comparison can be shared without sharing recordings
or individual scores.

The [fresh GRID transfer preparation](native-mouth-transfer-progress-v134.md)
separates publisher metadata, complete archive checksums, native video/audio
clocks and neural qualification. Fifteen new authored transport/selection checks
use synthetic bytes/ZIPs and mocked requests in the project dev environment;
they need no media, model weights, credentials or actual network requests:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_grid_acquisition_continuation_v135.py backend/tests/test_grid_clip_plan_v136.py
```

These check missing-range boundaries, unchanged-prefix assembly, rejected short/
ignored responses, bounded ZIP-directory reads and complete source/utterance
selection. They do not measure detection quality or establish the complete fresh
cohort's input eligibility. Read the [explicit acquisition continuation](grid-acquisition-continuation-v135-method.md)
and [full clip planning method](grid-clip-plan-v136-method.md) before interpreting
their receipts. Raw source identities/media and individual fingerprints remain
private. No outside reviewer is claimed.

The [expression-driving prototype](portrait-expression-drive-v138-design.md)
also has nine NumPy/pytest checks for the independent displacement equation,
unchanged source expression, invalid inputs and model-free imports:

```sh
PYTHONPATH=backend python -m pytest -q -o addopts='' backend/tests/test_portrait_expression_drive_v138.py
```

Those checks do not run pretrained networks or qualify generated motion. The
CPU bridge is unqualified and is not part of serving confidence. Its model
execution, native composite checks and achieved-motion comparison remain open.

The [complete native mouth construction/replay](native-mouth-clips-v120-complete.md)
now includes all 283 ready clips and 4,700 source-bound difference masks. Those
private-media checks are internal lineage validation, not an outside review.
For the next small cross-modal feature core, 13 authored checks need no media,
weights or credentials in the project dev environment:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_mouth_articulation_core_v122.py
```

They cover exact native-time context, speech-layer order, gap/padding rejection,
weighted projection/reconstruction and protection against learning the reference
from calibration/confirmation or edited rows. They establish arithmetic/role
contracts, not detection quality or a calibrated probability.

For exact source/output mask correspondence, the seven authored checks in the
[native clip replay](native-mouth-clip-review-v121-method.md) need no private
media, model weights, credentials or network in the project dev environment:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_review_native_mouth_clips_v121.py
```

They reject wrong mask positions, nonbinary masks, extra changed pixels and
incorrect pixel-edge transforms. The full real-source audit needs separately
retained clips/source plans; its passing authored tests are not evidence of
outside human review, achieved synchronization or detector accuracy.

For native mouth-edit support and counterfactual response arithmetic, the
[new input bank](native-mouth-inputs-v116-result.md) and
[renderer contract](native-mouth-renderer-v117-method.md) offer 24 authored
checks in the project development environment. They require no source media,
weights, credentials or network. These checks cover native context preservation,
exact-time conditioning shifts, repeated-image equality and nonresponse failures;
they do not establish neural accuracy. Three separate installed-runtime checks
require the pinned v7 generator dependency overlay. The actual v118 native
[renderer response experiment completed](native-mouth-renderer-v118-result.md);
the earlier admission denial remains historical. That success qualifies response
and containment, not achieved phonetic alignment or detector accuracy. The
separate [CPU second-generator qualification](portrait-native-qualifier-v133-method.md)
later started after RAM admission, but hit its sampled RSS limit; see the
[actual stop](evaluation-panels/portrait-native-resource-stop-v133.json). Its
earlier [readiness receipt](evaluation-panels/portrait-native-readiness-v133.json)
remains historical. The [bounded-convolution context](portrait-tiled-runtime-v139-method.md)
preserves that failed attempt and tests an explicit memory implementation change
at the same weights, precision, resolution and resource limits.
The new [actual bounded CPU run passed](portrait-tiled-runtime-v139-result.md),
including exact own repeats and byte-identical first-crop comparison with the
preserved partial original. This is construction-runtime qualification, not fresh
detector accuracy or achieved phonetic alignment.

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_native_mouth_inputs_v116.py backend/tests/test_native_mouth_renderer_v117.py
```

For the newest native speech-state/source routing contract, run the 21 authored
checks in the [source encoder result](av-source-encoder-v115-result.md). They use
NumPy, pytest and the project code, with no media, model download or credentials.
The complete real-source state review requires separately retained private
inputs. It verifies saved outputs and routing, not detector accuracy. The
[native video selection result](video-native-appearance-v112-result.md) likewise
separates zero false flags from two inconclusive controls and reports incomplete
time support; it is not reserved confirmation or full production qualification.

The [photo attribution/pixel-lineage review](image-source-lineage-v20-result.md)
adds 49 authored local checks requiring no model, real photo, credentials or
network request. Its actual private-panel run covers all 523 attribution records
and 465 source-pixel comparisons, with separate full reconstruction and dual
decoders. These verify source correspondence, not detection accuracy or training
permission. Additional useful contributions are permitted original/edit pairs
with photographer attribution, exact edit positions and source-group separation;
do not post customer media or assume a dataset-wide license clears every image.

For a model-free audit of photo licenses and source grouping, the
[image training-data review](image-source-rights-v19-result.md) provides a
public-only local command path. It downloads about253MB of official/pinned
metadata and no photographs. All2,242 metadata source records and per-role
counts reproduce without the private pairings; attribution/pixel lineage and
training permission remain unestablished. Its52 authored local tests also cover
used-source exclusion and inconsistent role files. GitHub Actions stays disabled.

For the new localizers' source-coverage and pixel-projection checks, use the32
authored tests below (NumPy, Pillow, Torch and pytest; no weights or photos):

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_image_imdet_v17.py backend/tests/test_relayformer_v18.py
```

Their [native execution record](new-localizers-and-editing-methods-2026-11-08.md)
is separate:24 actual forwards on authored patterns, not a real-edit accuracy
evaluation. Neither candidate is admitted to serving confidence.

For an image-coordinate review, use the public authored crop/interpolation
check. It needs NumPy, SciPy, Pillow and OpenCV, with no photographs, pretrained
weights or account credentials:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_image_iml_zoom_public_v16 /tmp/new-image-crop-review.json
```

For a smaller audio/video timing review, the signed-lag and sample-block check
needs only NumPy:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_mouth_audio_public_v14 /tmp/new-mouth-timing-review.json
```

Use a fresh receipt filename; these commands refuse overwrite. Include your
commit, environment and generated receipt in a proposed evaluation issue.
These checks validate arithmetic. To evaluate actual detection accuracy, use
the source-separated manifest contract below with separately permitted real
and manipulated inputs, retaining false alarms, misses and unassessed cases.
Do not attach private source media or claim that an authored check measures
real-world detector accuracy.
The image arithmetic check passes while the [real-source exact-mask audit](image-iml-zoom-v16-result.md)
fails at rounding boundaries. Its result and supplementary sensitivity remain
explicit; passing continuous math does not establish discrete mask parity.

## Verify image crops from a signed export

From `backend`, with the project environment active and the instance key obtained
separately, run the verifier against the evidence ZIP and your original image.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-image original.png
```

This optional check runs locally. It compares the original SHA-256 and
then checks every unannotated inspection crop and both sides of a same-image
match at their recorded EXIF-oriented coordinates. Pillow is enough for
the crop comparison. For a geometric copy match in `image-evidence-13`, the
project Python environment also recomputes both masks with the recorded
alignment and starting box. This replay does not rerun sparse proposal
generation. [Geometric copy evaluation](image-affine-copy-transfer-2026-11-11.md)

When a signed JPEG export includes a saved editor thumbnail comparison, the
same command uses the project environment to verify the submitted JPEG, both
source crops and the recorded difference footprint. The thumbnail positions
and PNG references are in the signed `result.json` projection. Earlier ZIPs
continue to verify against their original contents. [Editor thumbnail verification](image-editor-preview-2026-11-12.md)

When the export includes an aligned repeat-pass mask, the project environment
also recomputes the overlap of every inspected component with that mask, even
when the check withheld all crops. It uses the recorded response and quality
masks without running a detector. [Regional repeat pass](image-regional-flip-transfer-2026-11-11.md)

New image coverage records can include how much of the original contains
clipped or very dark pixels, low contrast, low-detail JPEG blocks, and patterns
consistent with resizing. Those categories can overlap. The recorded
reliability mask shows where the check applied exclusions on its measurement
grid. With the original supplied, the same command independently recomputes
those native shares and the exact grid mask, including analyses with no
selected regions. This check uses the submitted pixels and JPEG structure
without running detector weights.

The ordinary archive
verification remains available with only the ZIP and a key. The pixel check
provides source evidence for a separate review of the detector’s interpretation.
For a source-free setup check, run the project's authored fixture tests. They
cover mirrored and translated pairs, EXIF rotation, an incorrect source file,
and a signed crop whose declared position was shifted.

```sh
python -m pytest tests/test_public_package_integration.py -k '(image and multimedia_zip) or native_source_verifier'
```

When an earlier aligned image is available, the separate [reference comparison](image-reference-comparison-2026-11-11.md)
can produce a full-size mask and unaltered crops from both files. It runs on the
reviewer's machine and commits to both source hashes.

## Verify audio clips and timestamps from a signed export

Run the optional comparison with the local project environment, FFmpeg and
FFprobe installed. Supply the exact uploaded file. For a video export, give
the original video containing the soundtrack.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-audio original.mkv
```

The verifier checks the original SHA-256, re-decodes the first audio stream
to the committed mono 16 kHz samples, and compares its source timeline and
all published WAV intervals. This includes voice and lip-sync windows,
background sound, acoustic events, and both sides of repeated-audio evidence
when present. Timeline gaps stay visible in the comparison. The output counts
each playback interval separately, even when two intervals share one WAV asset.
For current measured stereo context, it also recomputes both physical channel
levels, their correlation and the mono level drop in each bounded source
window. It replays the bounded waveform similarity result when the signed
package contains that measurement.
The ordinary verifier also checks a reported audio clock discontinuity against
the soundtrack timeline committed in a signed video export.
The same file decoded with a different FFmpeg build can produce different samples.
This check verifies source correspondence for independent listening. It does
not rerun the detectors, recreate charts, or validate an AI authorship claim.

```sh
python -m pytest tests/test_audio_repetition_context.py tests/test_audio_similarity_context.py tests/test_video_nested_evidence.py
```

## Verify video frames and crops from a signed export

With the local project environment, FFmpeg and FFprobe installed, supply the
exact original video and the separately obtained instance key.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-video original.mkv
```

This checks the source hash and native presentation timestamps, then compares
unannotated frame PNGs and crops to the decoded pixels. When present, it also
checks the measured mouth patches, the committed aligned face inputs and the
source clock of saved context clips. It reports how many frames and inputs
matched. An inferred timestamp cannot verify as a measured frame. Native
replay accepts frames up to 16 megapixels. Different FFmpeg builds can produce
different results, particularly for older exports.

Review the report's interpretation separately. The verifier does not rerun the
detectors or recreate annotations and context-clip pixels. When the package has
frozen soundtrack evidence, run both checks together against the same original.
Each channel reports its own result.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem \
  --source-video original.mkv --source-audio original.mkv
```

### Recompute selected image and video measurements

After the trusted-key verification above, extract the signed `evidence.json`
entry and run the standalone reviewer from `backend`. Use separate work files
for different exports.

```sh
review_dir=$(mktemp -d)
unzip -p image-evidence.zip evidence.json > "$review_dir/image.json"
python -I scripts/review_evidence_measurements.py image original.png "$review_dir/image.json"

unzip -p video-evidence.zip evidence.json > "$review_dir/video.json"
python -I scripts/review_evidence_measurements.py video original.mkv "$review_dir/video.json"
```

The reviewer imports no application code and reads the original locally. The
image path needs Pillow. It recomputes the four sharp seam responses from
native color steps and checks their combined response. The output gives the
largest absolute difference from the packet. The tolerance is 0.00005 for
float32 accumulation and packet rounding. This reproduces the published
rectangle's measurement. Whole-image candidate ranking is a separate test.

The video path needs FFprobe and a complete native clock of at most 108,000
frames. It reconstructs the 12 Hz schedule from the source presentation
timestamps, checks the recorded input indices and ticks, and derives the two
transition timestamps and the before, during, and after frame references from
the recorded supports. Use the signed-source verifier above to verify the
actual frame pixels and aligned crops. Face comparison scores and candidate
ranking require detector re-execution.

The output includes source and evidence digests, packet counts, and either
`matched` or `no_measurement_recorded`. A mismatch exits with status 1. A
source without a published seam or timing refinement yields
`no_measurement_recorded`. Keep the JSON and the tool versions with a review.
Share the output only if disclosure of those source commitments is permitted.

Local replay of the existing paid-flow test fixtures reproduced all five seam
responses in the pasted PNG and all 21 scheduled positions, 21 measured
inputs, and three displayed frame roles in the known short swap. The original
PNG, JPEG, and original video had no recorded measurement in this replay.
On the 120-view [second PNG panel](image-boundary-prominence-holdout-2026-11-11.md),
the reviewer checked all combined responses and the four side responses for
25 highlighted rectangles. The largest absolute difference was 0.00000160.
The authored regression tests reject altered values, schedule positions,
source ticks, and displayed frame references.

```sh
python -m pytest tests/test_evidence_measurement_review.py
```

### Verify the inspected views from a JPEG

Download the analysis JSON and keep the exact original JPEG. Under
`whole_image_review.crops`, download each PNG from its `url` using the same
authorized browser session. Save each PNG in a local directory as its
`sha256` value followed by `.png`. Run this from `backend` with Python 3.11 or
later and Pillow. The app and detector weights are unnecessary.

```sh
python -I scripts/review_evidence_measurements.py image-views original.jpg \
  analysis.json --assets downloaded-crops
```

The reviewer checks the original file digest, recomputes the three 528 × 528
views, and compares their RGB pixels and source bounds against the export.
It reports the crop digests and measured pixel counts. An ordinary result
with no elevated response reports `no_flag`, while unavailable coverage
reports `skipped`. Both have zero published views to compare.

This uses the customer JSON and crop bytes as supplied. Verification covers
the input pixels and coordinates. Rechecking the detector response requires
the separate inference runner, and authenticating the JSON itself requires
an independently trusted copy of the export.

The standalone command reproduced all three crops from an actual Full
analysis and returned `no_flag` for its matched-scope ordinary control.
The regression test also rejects a shifted rectangle and a valid PNG from the
wrong position.

### Verify an earlier-copy image comparison

Download the comparison JSON in the local app and obtain both original files
from the reviewer who supplied them. From the backend directory, use the
project's local Python environment to replay the comparison.

```sh
.venv/bin/python -m scripts.review_image_reference_export \
  chipmunk-verify-image-comparison.json earlier.png submitted.jpg
```

The first two comparison schemas also support the standalone Python and Pillow
reviewer.

```sh
python -I scripts/review_image_reference_export.py \
  chipmunk-verify-image-comparison.json earlier.png submitted.jpg
```

For a crop taken after a right-angle rotation or reflection, use the additive
oriented crop reviewer. It also opens the earlier comparison schemas. It
independently verifies the unique crop placement, both native coordinate
mappings, all retained mask pixels and the exact crop and overlay bytes.
[Oriented crop evaluation](image-reference-oriented-crop-2026-11-12.md)

```sh
.venv/bin/python -I scripts/review_image_reference_oriented_crop_export.py \
  chipmunk-verify-image-comparison.json earlier.png submitted.png
```

When a rotated or reflected crop was also resized, use the reviewer below. It checks the projected RGB grid and the submitted mask, maps each evidence crop back to the earlier file, and reruns the bounded search to verify the recorded geometry. It also accepts the preceding comparison schemas. [Combined transformation evaluation](image-reference-oriented-resize-transfer-2026-11-12.md)

```sh
.venv/bin/python -I scripts/review_image_reference_oriented_resize_export.py \
  chipmunk-verify-image-comparison.json earlier.png submitted.png
```

The reviewer checks both file SHA-256 commitments and native dimensions after
orientation. It checks the overlap against the recorded integer offset,
compares both decoded crops with the corresponding source pixels, and checks
the embedded PNG hashes and binary mask bounds and pixel totals. Where the export
includes an amber difference overlay, the reviewer checks every pixel against the
full mask sliced at the submitted crop's coordinates. For the resize
schemas, the reviewer reruns the resize comparison using the two source files
and checks the recomputed framing, regions, and entire mask. It independently
maps the submitted region back to the exact earlier-copy crop using rational
coordinates. The crop-and-resize schemas verify the earlier area's exact offset
and size. The scoped PNG fallback also verifies the reproduced resize, pixel
match count, and exact support tiles. The reviewer reports empty comparable
and unassessed results separately. A mismatch exits with status 1. [Resize evaluation](image-reference-resize-transfer-2026-11-11.md) · [JPEG crop and resize](image-reference-crop-resize-transfer-2026-11-11.md) · [Exact PNG crop and resize](image-png-exact-crop-resize-2026-11-11.md)

The comparison JSON is a standalone, unsigned export. This replay checks its
geometry and source binding. The resize replay also checks the algorithm's
selected pixels. Establishing when or how either source was created requires
separate evidence. On the isolated browser fixture, the reviewer matched both crops and
counted 7,693 pixels within the reported mask bounds. Authored tests reject
shifted coordinates, missing alignment support and rehashed altered crops.
The reviewer also accepted all 180 genuine exports from the earlier integer
alignment panel. It checked 106 source crops and counted 1,036,494 retained
mask pixels across 99 comparable views. The other 81 views retained their
reported unassessed reasons. [Source and transforms](image-reference-comparison-2026-11-11.md#integer-crops-and-translations)

### Verify an earlier-draft comparison

Download the standalone comparison JSON and keep the original earlier and
later files. From the backend directory, replay both source hashes, the
decoded text positions, and the exact changed-token ranges locally.

```sh
.venv/bin/python -m scripts.review_document_revision_export \
  chipmunk-verify-draft-comparison.json earlier.eml later.eml
```

Use matching `.eml`, `.txt`, `.docx` or `.pdf` files. The output gives the verified range
counts by role. Word replay also checks the top-level body paragraph numbers,
XML body-child positions and paragraph-local character ranges. HTML-only email
replay checks the original MIME part and the decoded HTML positions of changed
words. Both HTML drafts need two, three, five or six simple body paragraphs
followed by an explicit blockquote. The comparison measures wording in the
reply body and carries zero authorship confidence weight. PDF replay also
checks the changed words against both sources' page positions, original crops
and highlighted crops. Each PDF must be unlocked, at most 1 MiB and 12 pages,
and have mapped native text on every page. The browser shows the recorded
original and highlighted crops. The downloadable PDF shows the highlighted
crops. The JSON preserves both crops for the selected previews and page
coordinates for every changed token. The comparison locates differences and
leaves revision order and authorship to other evidence.

In an authored panel of 28 HTML draft pairs, the comparison located all 16
wording and punctuation changes. It returned no reply-body changes on 12
quote, attachment, and formatting controls. Separate MIME and XML readback
verified 24 changed-token positions and 48 decoded HTML ranges. In a local
browser test after a simulated checkout, the highlighted words matched the PDF
and the replayed JSON. The comparison created no second job or order.

The authenticated local app can also compare both writing responses for one
changed sentence flanked by two unchanged sentences in a short English `.txt`,
`.docx`, `.eml`, or `.pdf` pair. All three sentences must be in one paragraph
or one PDF text block. For Word files, the check uses top-level body
paragraphs. For email, it uses the selected plain-text body or the mapped body
of a supported HTML-only reply and excludes marked quotes, signatures, headers
and attachments. For PDFs, both drafts must have one upright page with
selectable text, complete word boxes and consistent top-to-bottom reading
order. Linked changes preserve the exact wording positions in both drafts.
The check adds zero weight to authorship confidence. Replay verifies the source
spans, input hashes and arithmetic of the saved responses. Reproducing the
responses requires a separate inference run.

Rewrapping 53 previously tested article draft pairs in Word preserved the 34
eligible pairs, the exact model inputs and the selected locations. The check
reproduced five flags on 23 eligible generated edits and zero on 11 eligible
ordinary edits. A separate local run used both installed writing checks through
the authenticated Word route for two generated edits and two ordinary edits.
All four decisions matched the earlier runs, and their PDFs and JSON replays
matched the source positions. [Original article panel](writing-journal-body-transfer-2026-11-11.md)

The same 53 earlier article pairs, delivered separately as plain-text and
HTML-only replies, retained all 34 eligible paired inputs in each format.
Four authenticated runs with both installed checks verified linked source
positions, MIME part labels, PDF and JSON. The test isolates the effect of
packaging and quote handling on previously scored writing. [Email comparison](writing-paired-email-revision-2026-11-11.md#email-format-verification)

Four paired selectable PDFs preserved the exact text inputs from their
earlier journal runs. Their JSON and PDF exports matched the original
word boxes and page crops. [Selectable PDF drafts](writing-paired-email-revision-2026-11-11.md#selectable-pdf-drafts)

### Replay document evidence against the original

New document evidence records distinguish source ranges with a local response,
text read only as context, and text with no exact local record. The analysis
view, PDF and JSON use the same extracted Unicode positions. Counts include
every mapped character excluding whitespace. The analysis view and JSON retain
up to 256 ranges, and the PDF shows up to 12. Source replay verifies the
retained positions and character counts against the supplied original. The
server also replays the map against the frozen local responses.

An isolated local app run used a short multipart email containing a controlled
rewrite, a genuine quotation and a signature. With the current classifier
available, all 307 mapped characters had local responses. A second run of the
identical file with the classifier skipped showed all 307 characters as having
no exact local record. The evidence API, ordinary JSON export and rendered PDF
agreed on the three exact source ranges in both runs. Both runs used simulated
payment and a separate local database.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.pdf
```

With the project dependencies installed, this compares the original's SHA-256,
re-extracted text and source mapping with the signed record. It checks the exact
excerpts, positions, supporting input ranges, repeated passages and any PDF
page images in the package. It supports plain text, DOCX, source-mapped EML,
and PDF. PDFs with OCR evidence require the project's verified OCR data and
a compatible local renderer and recognizer. Replay checks the exact OCR input
and the source pixels behind the crop. The replay does not score authorship.

## Choose a review to reproduce

### Save a four-modality replay receipt

From the repository root with Python 3.11 or later:

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/review.json"
```

The command runs one published ledger reader each for image, video, audio,
and writing. Use `--modality video` to replay a single ledger. The receipt
records the checkout revision when Git is available, exact input hashes,
the Python environment, and each result. It contains no media, credentials,
local absolute paths, or raw diagnostic output. The command saves rejected,
missing, and timed-out checks and exits unsuccessfully if any selected check
fails. Share the JSON in the [independent evaluation form](https://github.com/ChipmunkRPA/veritas/issues/new?template=independent-evaluation.yml)
if you want to record a reproduction or discrepancy.
For detailed diagnostics, run the recorded ledger reader directly with its
listed arguments.

To compare a second machine's receipt with yours, run:

```sh
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/review.json" --compare other-review.json
```

Use the same suite and modalities on both machines. The
command checks the input, result, and diagnostic commitments and identifies
where they differ. It reports missing modalities and failed runs separately
and exits successfully only when all selected replays match. Different
operating systems, Python versions and Git revisions can match when the
hashed reader inputs and outputs agree. The command reports a runner version
difference separately. Comparison reads JSON and runs no
commands from either receipt. Equal receipts establish that the recorded
commitments agree. The files do not authenticate who ran the checks or
establish new detector performance.

The selected ledgers cover [JPEG preview regions](image-thumbnail-extent-transfer-2026-11-12.md),
[later video repeats](video-motion-late-transfer-2026-11-12.md),
[matched audio overlays](audio-vocal-event-mixtures-2026-11-12.md), and
[the eLife sentence comparison](writing-elife-exact-sentence-confirmation-2026-11-12.md).
For source labels, decoder output, or predictive performance, follow the
native instructions for the claim you are reviewing.

### Review localization across four media types

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/localization.json" --suite localization
```

This suite replays four completed comparisons and preserves each study's
held denominator and rejected decisions. It covers
[multiple image regions](image-boundary-multiple-transfer-2026-11-12.md),
[local soundtrack displacement](video-paired-soundtrack-transfer-2026-11-12.md),
[partial neural speech edits](audio-partialedit-dense-transfer-2026-11-12.md),
and [earlier-source writing responses](writing-source-conditioned-2026-11-12.md).
It uses saved observations and checks the reported outcomes without downloading
source files or weights. Each study links to its separate native-source
instructions. Share the receipt and the claim or discrepancy you reviewed
through the independent evaluation form.

### Replay four transfer studies

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/transfer.json" --suite transfer
```

This suite checks a successful [photographic PNG expansion](image-png-layout-guard-2026-11-12.md)
and three candidates that stayed outside the app. It verifies the saved
[lip-sync timing](grid-phone-viseme-timing-2026-11-12.md),
[speech window](audio-short-coarse-transfer-2026-11-12.md) and
[mixed-writing vocabulary](writing-local-lexical-transfer-2026-11-12.md)
results. The receipt includes the checked-in data and helper hashes used
by each reader, plus hashes of its output and diagnostics.

### Replay recent source evidence

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/source-evidence.json" --suite source-evidence
```

This suite recalculates the [PNG preview comparison](image-png-exif-preview-2026-11-12.md),
[paired HEVC face timing](video-face-paired-edge-transfer-2026-11-12.md),
[overlapping speech windows](audio-dense-evidence-confirmation-2026-11-12.md),
and [mixed PDF writing coverage](writing-mixed-pdf-page-transfer-2026-11-12.md).
The receipt captures every public input and helper used for the comparisons.
It checks saved measurements and source commitments using Python's standard
library. The native replay instructions in each study explain how to verify
the media evidence against the supplied source.

Compare your result with the [published local reference](evaluation-panels/source-evidence-replay-reference-2026-11-12.json)
and prepare a short issue excerpt with these commands. A difference is useful
evidence, including when the command exits with status 1. The formatter reads
the receipt without running the embedded reader commands or publishing an issue.

```sh
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/source-evidence.json" \
  --compare docs/evaluation-panels/source-evidence-replay-reference-2026-11-12.json
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/source-evidence.json" --format-issue
```

The reference records our local run. Add your own receipt and the specific
claim you verified or challenged to the [independent evaluation form](https://github.com/ChipmunkRPA/veritas/issues/new?template=independent-evaluation.yml).

### Replay the latest detector comparisons

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/recent.json" --suite recent-checks
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/recent.json" \
  --compare docs/evaluation-panels/recent-checks-replay-reference-2026-11-12.json
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/recent.json" --format-issue
```

This suite replays the held [cross-camera HDR cue](image-gainmap-cross-camera-2026-11-12.md),
[six-picture face check](video-face-unrouted-short-transfer-2026-11-12.md),
[transient-reduced speech check](audio-transient-reduction-transfer-2026-11-12.md),
and [writing comparison across a paragraph break](writing-paired-center-transfer-2026-11-12.md).
Python 3.11 and 3.12 reproduced all four outputs locally. The suite runs without
model weights, media downloads or network access. It replays published measurements.
The same four outputs matched twice on Linux arm64 with Python 3.12.15.
The second run mounted only the 19 committed public files listed by the
reference, including the runner, in a read-only container with networking
disabled. The local image ID was `sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
This reproduces the arithmetic on another operating system. Source authorship,
detector accuracy, and the interpretation of findings still need separate review.
[Linux replay receipt](evaluation-panels/recent-checks-linux-replay-2026-11-12.json)

### Replay four follow-up checks

```sh
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/followups.json" --suite evidence-followups
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/followups.json" \
  --compare docs/evaluation-panels/evidence-followups-linux-replay-2026-11-12.json
```

The suite covers [HDR perimeter controls](image-gainmap-perimeter-transfer-2026-11-12.md),
[short mouth-motion repeats](video-mouth-sequence-reuse-transfer-2026-11-12.md),
[short speech inputs under loud background noise](audio-reflected-local-transfer-2026-11-12.md),
and [plain-text and HTML email bodies](email-body-alternatives-transfer-2026-11-12.md).
The reader outputs and input hashes agreed on macOS Python 3.11 and 3.12,
and in two Linux arm64 runs on Python 3.12.15. The Linux container had a
read-only checkout of 22 pinned inputs, disabled networking, and no source
media or weights. A matching receipt reproduces the published arithmetic.
Source preparation and inference replay require the additional inputs in
each linked study. [Linux replay receipt](evaluation-panels/evidence-followups-linux-replay-2026-11-12.json)

### Rebuild native Word source controls

Reconstruct the ordinary Word draft controls and verify their exact decoded
positions from ten pinned public PLOS articles. The tool can fetch the
publisher XML once and repeat the same checks from a local cache. It also
checks the available earlier sides of the generated-rewrite pairs. The
generated later sides stay outside this reconstruction.
[Publisher source replay and commands](writing-word-public-source-replay-2026-11-12.md)

### Replay brief face changes on the eight-second clock

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_short_prefix_transfer \
  docs/evaluation-panels/video-face-short-prefix-transfer-2026-11-12.json
```

The reader compares the current Full face checks, a four-second prefix, and
three overlapping four-second windows on the same 54 held deliveries. The
native procedure additionally checks original picture timestamps and crop
pixels. It requires the exact publisher sources described in the study.
[Face-change study and replay instructions](video-face-short-prefix-transfer-2026-11-12.md)

### Verify an earlier-audio timing export

For a comparison that shows a moved excerpt, keep both original recordings and download the comparison JSON. The verifier independently decodes their clocks and reruns the waveform match. It checks the signed offset, both time ranges, the supporting windows, the waveform values and the bytes of both playable clips. It needs FFmpeg, FFprobe, NumPy and OpenCV. The supplemental export is unsigned, so this verifies its evidence against the supplied copies. [Local timing evaluation](audio-reference-local-shift-2026-11-12.md)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.verify_audio_local_timing_export \
  earlier.wav analyzed.wav comparison.json
```

The verifier checked the 34 localized WAV pairs from the published held panel and all 68 exported excerpts. In a separate check, the public timing, RADAR, Lombard source and writing result readers printed identical output on macOS Python 3.11.9 and offline Linux Python 3.12.15. The shared output SHA-256 was `9d2680b4ca0e7dd6041ff90c364eb9f89b8010e093fbfe1d3a03645362f8183d`. This cross-platform replay uses saved evaluation ledgers. [Timing reader](../backend/scripts/review_audio_reference_local_shift_panel.py) · [RADAR reader](../backend/scripts/review_audio_selective_fusion_transfer.py) · [Lombard reader](../backend/scripts/review_lombard_mouth_events.py) · [Writing reader](../backend/scripts/review_writing_position_qwen.py)

### Replay the stereo export and the latest held checks

Run from the repository root with Python 3.11 or later. The first command also
needs FFmpeg and FFprobe, both original audio files, and a JSON export with a
measured stereo section.
It reopens the original recordings and verifies exact source intervals,
channel samples, playable excerpts and waveforms. Version 7 includes the
exact timing matches for each recorded channel. The other commands read
public receipts and recalculate the held decisions without model weights.

```sh
python3 -B -IS backend/scripts/verify_stereo_reference.py earlier.wav analyzed.wav stereo.json
python3 -B -IS backend/scripts/review_audio_modulation_partial_transfer.py docs/evaluation-panels/audio-modulation-partial-transfer-2026-11-12.json
PYTHONPATH=backend python3 -B -S -m scripts.review_video_lipsync_repeat_audio docs/evaluation-panels/video-lipsync-repeat-audio-2026-11-12.json
python3 -B -IS backend/scripts/review_writing_two_context_transfer.py docs/evaluation-panels/writing-two-context-transfer-2026-11-12.json
python3 -B -IS backend/scripts/review_writing_position_confirmation.py docs/evaluation-panels/writing-position-confirmation-2026-11-12.json
python3 -B -IS backend/scripts/review_writing_position_qwen.py docs/evaluation-panels/writing-position-qwen-confirmation-2026-11-12.json
python3 -B -IS backend/scripts/review_audio_selective_fusion_transfer.py docs/evaluation-panels/audio-selective-fusion-transfer-2026-11-12.json docs/evaluation-panels/audio-selective-radar-transfer-2026-11-12.json
PYTHONPATH=backend python3 -B -S -m scripts.review_video_mouth_sector_persistence docs/evaluation-panels/video-mouth-sector-persistence-2026-11-12.json
```

The [image line-pattern audit](image-line-fingerprint-transfer-2026-11-12.md)
reopens the licensed photo and edit sources to check the location arithmetic.

Three further reviews cover [stereo timing under competing speech](audio-reference-stereo-channel-timing-2026-11-12.md),
[original character locations in signed writing evidence](document-original-text-locations-2026-11-12.md),
and [head and mouth motion in synchronized video edits](video-mouth-rigid-coupling-2026-11-12.md).
The writing verifier needs the original document and signed package. The stereo
challenge needs the separately acquired publisher speech and room measurements.
The video result reader can recalculate the held decisions from the public
receipt alone. The [image region proposal audit](image-patch-proposal-audit-2026-11-12.md)
replays both blind and earlier-copy localization on separately acquired photos
and exact edit masks.

To save a single cross-media receipt, run the `short-edits` suite. It uses
the video, audio and writing readers above and the [public image thumbnail
extent reader](image-thumbnail-extent-transfer-2026-11-12.md). All four output
hashes matched on macOS Python 3.11 and two offline Linux arm64 Python 3.12.15
runs from a read-only set of nine public files. The receipt reproduces the
reported arithmetic. Source labels and detector inference need the study inputs.
[Linux receipt](evaluation-panels/short-edits-linux-replay-2026-11-12.json)

```sh
python3 -B -IS backend/scripts/replay_public_evaluations.py short-edits.json --suite short-edits
python3 -B -IS backend/scripts/replay_public_evaluations.py short-edits.json \
  --compare docs/evaluation-panels/short-edits-linux-replay-2026-11-12.json
```

Use the [evaluation issue form](../.github/ISSUE_TEMPLATE/independent-evaluation.yml) to
share the commit, exact command, and discrepancy or reproduced claim. Keep
private sources and customer reports local.

### Replay recent evaluations from a clean checkout

From the repository root, use Python 3.11 or later. These commands read the
checked-in receipts and need only the standard library.

```sh
python3 -B -IS backend/scripts/review_audio_nemotron_noise_transfer.py review docs/evaluation-panels/audio-nemotron-noise-transfer-2026-11-12.json
python3 -B -EsS backend/scripts/review_audio_nemotron_noise_confirmation.py review docs/evaluation-panels/audio-nemotron-noise-confirmation-2026-11-12.json
python3 -B -IS backend/scripts/review_audio_reduced_background_localization.py review docs/evaluation-panels/audio-reduced-background-localization-2026-11-12.json
python3 -B -IS backend/scripts/review_writing_punctuation_stability.py docs/evaluation-panels/writing-punctuation-stability-2026-11-12.json
```

- The audio readers recalculate added flags, genuine false alarms and the
  advancement decisions from saved scores on two separate held panels.
  [Method and native replay](audio-nemotron-noise-transfer-2026-11-12.md#replay)
- The longer speech reader reconstructs the checked sample ranges and reports
  useful edit locations and genuine time flagged with four window rules.
  [Method and native replay](audio-reduced-background-localization-2026-11-12.md)
- The writing reader recalculates 108 cases, including the useful edit
  locations the proposed punctuation guard would suppress.
  [Method and replay](writing-punctuation-stability-2026-11-12.md#replay)

The image editor-thumbnail and 720p video retry receipts publish aggregate
counts. This command checks their recorded count relationships.

```sh
python3 -B -IS - <<'PY'
import json
from pathlib import Path

panels = Path('docs/evaluation-panels')
image = json.loads((panels / 'image-editor-preview-2026-11-12.json').read_bytes())
video = json.loads((panels / 'video-face-hd-sparse-retry-2026-11-12.json').read_bytes())

roles = image['by_role']
image_cases = sum(row['views'] for row in roles.values())
useful_images = roles['generated']['useful']
assert image_cases == image['cases'] == 166
assert all(sum(row['status'].values()) == row['views'] for row in roles.values())
assert useful_images == image['counters']['useful_generated'] == 6
malformed = image['malformed_rejected']
assert len(malformed) == 6 and all(malformed.values())
assert image['counters']['malformed_acceptances'] == 0
assert sum(row['status'] == 'comparable' and row['regions'] == 0
           for row in image['native_controls']) == image['counters']['native_controls'] == 2

strata = video['held']['strata'].values()
video_cases = sum(row['files'] for row in strata)
retries = sum(row['retry'] for row in strata)
before = sum(row['before'] for row in strata)
after = sum(row['after'] for row in strata)
assert video_cases == 48 and retries == 0
assert sum(row['truth'] for row in strata) == 32 and before == after == 14
assert after - before == video['held']['gains']['added_correct'] == 0
assert video['held']['advances'] is False and video['admitted_to_serving'] is False
print(f'Image: {image_cases} cases, {useful_images} useful generated-edit locations')
print(f'Video: {video_cases} cases, {retries} retries, {before} useful ranges')
PY
```

For native image replay, obtain the earlier image panel and the two pinned
Wikimedia originals, and install Pillow, NumPy and OpenCV. The source check
uses a supplied original, evidence export and crop PNGs.
[Image replay instructions](image-editor-preview-2026-11-12.md#local-verification)
For native video replay, use the retained source archive, the prepared held
run with its exact video deliveries, FFmpeg/FFprobe and the backend Python
environment. Recreating the Full comparison also requires the pinned detector
assets. [Video replay instructions](video-face-hd-sparse-retry-2026-11-12.md#replay)

### Replay four measured panels from a clean checkout

Run these commands from the repository root with Python 3.11 or later. They use
public scalar receipts from local experiments. The commands need only the
standard library, with no media, model downloads, accounts, or running app.

```sh
python3 -B -IS backend/scripts/review_image_thumbnail_mirror_transfer.py docs/evaluation-panels/image-thumbnail-mirror-2026-11-12.json
python3 -B -IS backend/scripts/review_grid_articulatory_timing.py docs/evaluation-panels/grid-articulatory-timing-2026-11-12.json
python3 -B -IS backend/scripts/review_audio_short_codec_transfer.py
python3 -B -IS backend/scripts/review_writing_multiblock_independent_transfer.py docs/evaluation-panels/writing-multiblock-independent-2026-11-12.json
```

The checks recalculate [mirrored-image region overlap](image-thumbnail-mirror-2026-11-12.md),
[signed video timing errors](grid-articulatory-timing-2026-11-12.md),
[audio codec points and false alarms](audio-short-codec-transfer-2026-11-12.md),
and [writing span overlap and fit arithmetic](writing-multiblock-independent-2026-11-12.md).
Each linked method explains the extra inputs required to replay source media.
For an independent result, use the issue form below and include the commit,
command, output, and the claim you verified or challenged.

### Choose a source or method audit

These are bounded entry points for contributors. Automated Linux reproduction
is separate from an outside person's review. A reproduced calculation does not
establish source truth, neural input correspondence or detection quality.
The numerical paths and synthetic colour experiment use this public repository;
the writing source audit and real-video renderer supplement additionally require
separately acquired study data.

| Package | What can be reproduced | Current evidence and limitation |
| --- | --- | --- |
| [Photo attribution and pixel lineage](image-source-lineage-v20-result.md) | Public authored identity/crop/decoder checks; complete real-panel replay requires separately retained permitted photos and private source plans | All 523 metadata responses reconstructed; 465 accepted source identities compared. 397 exact RGB matches, 68 differences; all 465 geometry replays and 930 dual-decoder checks match. No model inference, training admission, camera-authenticity proof or outside human review. Current generator-output terms remain separately scoped. |
| [Image source rights/grouping](image-source-rights-v19-result.md) | Standard-library public metadata/count reviewer after pinned metadata acquisition; no images, model weights or private pairings required | Full local private reconstruction and public-only replay pass for2,242 photos;597 pass a preliminary license filter.523 queued for attribution/lineage review, none admitted to training. Misnamed FLUX original files excluded from role assignment; actual generation splits disjoint. Not detector accuracy, final rights clearance or outside human participation. |
| [Full-source localizer v18](image-relayformer-v18-result.md) | Frozen native/source reviewer, independent OpenCV/IEEE ranking and contingency counts; exact panel additionally requires private sources and permitted cached weights | All384 inputs/798 project forwards complete. Full384-map repeat and both genuine-point refits pass. Useful regions0/48 per format vs9/48 native and6/48 JPEG current; rejected. Raw photographs/maps remain private; no fresh accuracy, serving admission or outside human review. |
| [Controlled writing-span learning](writing-spans-v7-result.md) | Public source construction, native loaders, two small learners, calibration/points, and full local source/refit reviewer; exact panel requires private source plan and separately acquired licensed CSV | All992 planned documents and full source/native/refit replay passed. All864 diagnostic arm masks recounted; both learned heads and three calibrators/points refit exactly. Interval found38/128 useful mixed locations vs45/128 current and lost27 previously useful documents; rejected as replacement. Consumed source groups, no serving confidence, fresh-author or outside human review claim. |
| [Magnified image crops](image-iml-zoom-v16-result.md) | Public authored grid/point arithmetic and complete aggregate results; native source/grid replay requires the retained private panel | All384 candidate variants and96 ordinary-reference runs completed. Candidate found0/48 useful edits per format versus9/48 native and6/48 JPEG current; rejected. Strict real-source independent-mask audit remains failed. Full post-result canonical/sensitivity audit retains25/768 ungated metric discrepancies and one changed point; rejection unchanged. Actual Linux continuous arithmetic passes, without outside human review or fresh accuracy. |
| [Supervised image-region head](image-patch-head-v14-result.md) | Frozen native/source replay, separate NumPy dense-head and coordinate arithmetic, and [actual Linux synthetic checks](evaluation-panels/image-patch-head-v14-linux-synthetic-review.json) | All1,344 development cases completed; candidate rejected for fewer useful regions. Original strict mean audit failed and remains failed. A separately frozen complete diagnostic replay reproduces all native features/maps and original outcomes; no mean-sensitivity mask changes. Raw sources/vectors/head remain private; no fresh accuracy or outside human review. |
| [Fresh JPEG calibration confirmation](image-fresh-v12-result.md) | Public frozen scoring/audit code and complete aggregate results; source/map replay requires the separately retained private panel and cached permitted weights | All 96 fresh inputs completed; 192 independent probability calculations and 384 source-coordinate means match. Candidate rejected: 12/48 edited flags vs 10/48 current, but 5/48 genuine false flags vs 2/48. Current masks meet IoU >= .3 on 3/48 edits. No serving promotion, public photos/maps or outside human review. |
| [Four-media specialist calibration](specialist-calibration-design.md) | Public authored128-file/512-observation version2 example, distinct context heads and calibrated combination; source-role/missing-evidence/duplicate-vote/source-geometry checks |63 local selected tests and [actual Linux integration](evaluation-panels/specialist-calibration-v2-linux-review.json) pass. Unrelated genuine regions no longer subtract from positive evidence elsewhere. This is symbolic-score integration only, not real-media calibration or detection quality. |
| [Region-aligned image calibration](image-region-calibration-v9-result.md) | Standard-library312 source/plateau-certificate bindings, two genuine point refits,120 display audits,384 metric sets and14 deliberate alterations | Complete private-grid/source replay passed. Area-aligned points still located0/48 edits per format vs12/48 current; rejected. Cached consumed responses, no new neural inference or outside human review. |
| [Regional image patch drift](image-patch-drift-v8-development-evaluation.md) | Standard-library312-case source/role binding, genuine-only point refits,576 pixel-metric sets, family counts and24 advancement checks; thirteen deliberate alterations | Separate local native/vector/pixel audit and [actual Linux scalar review](evaluation-panels/image-v8-patch-drift-linux-review.json) passed. Both readouts located0/48 edits per format vs12/48 current and were rejected. No public photos/features or fresh accuracy claim. |
| [Face-transition development review](video-v16-sface-development-evaluation.md) | Standard-library source/range/overlap/count review and ten deliberate packet alterations; no images or face embeddings | Local and actual Linux arithmetic passed. Located 55/60 edits vs 19/60 with 21/60 genuine review flags for both. Consumed-source development only. |
| [Fresh face-transition confirmation](video-v17-sface-confirmation-evaluation.md) | Standard-library source/donor exclusions, all120 ranges, paired/fallback counts and twelve deliberate packet alterations | Combined method located60/60 vs25/60, same26/60 genuine review flags. New cue supplied58 edit ranges; two use declared fallback. All ten requirements and [actual Linux replay](evaluation-panels/video-v17-sface-linux-review.json) passed. [Local integration and activation](local-face-appearance-v17-rollout.md) are complete; repository default remains legacy. |
| [Whole-image generation cue](image-smaug-v7-development-evaluation.md) | Standard-library all224 scalar responses, directional counts and ten original gates; optional numerical intervals | Actual Linux replay passed. Whole-generation indications improved, but local-edit detection was 0/24 per format and the original mixed-image advancement gate failed. No serving promotion. |
| [Overlapping-speech separation review](audio-separation-v14-development-evaluation.md) | Complete outcomes, numerical/refit and native-audit evidence, with public-source feature references | Separation arms rejected under the original useful-detection requirements. Better loss or fewer calls alone did not justify replacing the serving calibration. |
| [Mouth-window review](mouth-public-portable-review-v12.md) | Standard-library arithmetic, or seven-package exact fitting-only tree refit; no media or neural weights | Both Linux modes passed. Candidate still rejected for excess ordinary-clip flags. |
| [General-video review](video-demamba-v13-result.md) | NumPy-only all192 outcomes per method, losses, grouped intervals and11 requirements from nine lossless members | Linux arithmetic passed. Candidate rejected; four of16 native RGB checks remain failed. |
| [Video colour-path audit](video-v14-colour-rendering-audit.md) | Self-created synthetic colour chart; optional real-source renderer/crop/tensor replay without neural weights | Local32-frame supplement passed. Linux synthetic run completed but did not reproduce the Mac-specific hypothesis; different bundled codecs and separate encoding remain explicit. Original native failures remain. |
| [Fixed native-YUV comparison](video-v15-native-yuv-protocol.md) | Self-authored planar samples with exact full native recovery before renderer comparison | Mac/Linux native samples and clocks exact; explicit renderer RGB hashes still differ across hosts. Original RGB hypothesis/native gate remain failed. |
| [Audio review](audio-public-portable-review-v13.md) | All4,248 feature records,1,449 development outcomes and optional exact fitting-only numerical refit | Linux arithmetic passed; strict exact cross-platform refit failed. Diagnostics changed no directional calls; exact check was not relaxed. |
| [Writing-control source audit](writing-v4-human-controls.md) | Original human-only history replay and overlap audit, using separately obtained study data | Local source audit only;45 supported/eight unresolved controls, no classifier scores. Source text and edit logs are not in the public package. |

For a useful review, identify your commit and environment, choose which claims
you independently inspect, retain failures and return the generated receipt
with any proposed issue or pull request. Exact commands and data/dependency
boundaries are linked above. No account credentials, customer uploads or model
downloads are needed for the three numerical paths or synthetic colour experiment.

## Useful contributions

- Reproduce a published failure on a permitted input, including its source hash,
  exact check coverage, app revision, hardware, and relevant source positions.
- Compare a proposed detector or preprocessing change against the current check
  on identical, source-separated inputs. Include ordinary editing, compression,
  background sound, and other plausible non-AI explanations.
- Audit a calibration, sample split, mask or timestamp measurement, license
  declaration, or claimed numerical parity. A code audit and a detection-quality
  evaluation answer different questions; label them separately.
- Test whether ordinary users can follow findings to evidence and understand
  inconclusive outcomes. Document the task and feedback without private uploads.

Use the repository's **Independent evaluation or detection failure** issue form.
Issues and pull requests are reviewable proposals; they do not automatically
change a serving model, calibration, or stored result. Do not submit media that
you do not have permission to share. Public references, hashes, methods, and
aggregate outcomes are sufficient when source media cannot be redistributed.

The [public evaluation source catalogues](evaluation-panels/README.md) identify
the exact new image/video panels with public references, hashes, and edit
positions. Their dataset metadata retains its separate attribution and terms.

## Portable evaluation contract

`backend/app/bench/independent.py` is standard-library-only. Its command-line
wrapper runs without downloaded weights and does not start the app:

```sh
python3 backend/scripts/independent_review.py audit manifest.json
python3 backend/scripts/independent_review.py freeze manifest.json development.json thresholds.json frozen.json
python3 backend/scripts/independent_review.py summarize manifest.json confirmation.json frozen.json summary.json
python3 backend/scripts/independent_review.py compare manifest.json baseline.json baseline-frozen.json candidate.json candidate-frozen.json comparison.json
```

Before sharing a result, an evaluator with the original files can verify
the actual submitted and parent bytes against every planned digest.

```sh
python3 -B -IS backend/scripts/independent_review.py verify-inputs \
  manifest.json private-file-index.json input-verification.json --root "$media_dir"
```

Create the private index with `schema: "chipmunk.independent-file-index/1"`,
the canonical `manifest_sha256` printed by `audit`, and a `cases` list.
Each case has its exact manifest `id`, `input` and `source`. Use paths
relative to the supplied directory. Include both splits and repeat the
parent path for related variants. File names, paths, and individual
hashes stay out of the output. Reviewers with equivalent source access
can perform their own verification. The receipt records checked bytes.
Confirm labels, source rights and absence of earlier exposure separately.

To compare two runs of the same frozen method, submit both confirmation
receipts to `replicate`. The tool reports changes in responses, flags,
assessment coverage and localization counts by condition and true class.
Each receipt must commit to the same manifest, implementation, calibration
and freeze. A numerical difference can occur without changing a flag.
Matching overlap counts can still describe different positions. Keep the
individual receipts and positions local when the inputs are private.

```sh
python3 -B -IS backend/scripts/independent_review.py replicate \
  manifest.json reference.json second-run.json frozen.json replication-summary.json
```

Use `replicate-locations` when both reviewers can run locally against the
same private truth and the two original location receipts from before
`count-locations`. It counts exact matches of pixel, sample, timestamp, or
character unions, including where equal-size highlights land in different
places. For PBM image masks, supply one local directory with both sets of
mask files, using distinct filenames for different bytes. The output includes
aggregate counts and commitments to all three private receipts. It leaves
source coordinates, case IDs, mask filenames and individual reasons local.

```sh
python3 -B -IS backend/scripts/independent_review.py replicate-locations \
  manifest.json private-truth.json private-reference.json private-second-run.json \
  frozen.json location-replication.json --image-mask-dir "$local_masks"
```

Outputs refuse overwrite. Keep the original manifests, frozen decisions, and
receipts. Store private evaluation media outside Git. The summary deliberately
omits individual source identifiers and media; review it before public sharing.

For a new evaluation, add `--print-digest` to the `freeze` command to obtain the
canonical digest. Put it in the confirmation receipt as `freeze_sha256`.
`summarize` and `compare` check a digest when present. To require it, pass
`--require-freeze-commitment`. The digest binds the exact threshold and other
frozen fields. A dated, separately retained record establishes when the freeze
was made. Historical receipts can still run without the new flag.

The manifest has `schema: "chipmunk.independent-evaluation/1"` and a `cases` array.
Each case records:

| Field | Meaning |
| --- | --- |
| `id` | Unique input variant |
| `modality` | `image`, `video`, `audio`, or `text` |
| `split` | `development` or `confirmation` |
| `group` | Shared parent-source group, consistent across variants and modalities |
| `label` | Integer 0 for declared genuine control, 1 for known generation/manipulation |
| `family`, `condition` | Generator/control family and transformation or recording condition |
| `source_sha256`, `input_sha256` | Parent source and exact scored-file digests |
| `lineage` | Known linked original, speaker, synthetic voice, topic, transcript, or source identifiers |
| `license` | `id`, explicit `evaluation_permitted`, and explicit `redistribution_permitted` |

Known lineage, group identities, and source/input hashes cannot cross splits.
These are declared identities: the checker cannot discover undisclosed training
overlap, authenticate a publisher's labels, grant rights, or infer subject consent.
Check near duplicates and connected source lineage before freezing the selection.
Keep source families and transformed copies together. Explain any unknown overlap.

Each system receipt records `manifest_sha256` (canonical JSON digest from the
audit), `split`, `system`, `implementation_sha256`, a nonempty `weight_sha256`
array, `score_kind` (`raw` or `calibrated`), `response_units`, and `predictions`.
Calibrated receipts also require `calibration_sha256`.

For checks based on direct measurements, set `method_kind: "model_free"` and
`weight_sha256: []`. The freeze records the method and implementation digest.
Existing weighted receipts use their pinned weight digests.

Every planned case appears exactly once with its `id`, `input_sha256`, and
`status`.

- `ok`: a finite `response`, oriented towards the declared positive class. Raw
  measurements retain their actual units, including signed responses. Calibrated
  estimates must lie between 0 and 1.
- `inconclusive`, `not_applicable`, or `failed`: `response: null` and a specific `reason`.

Raw responses are not confidence probabilities. Missing responses must not be
written as zero. Candidate calibrations have their own identities; changing a
system, checkpoint, implementation, or calibration after the freeze is refused.
The `thresholds.json` file maps each evaluated modality to a fixed **internal
benchmark flag point**. This is not a public screening cutoff or a rule for the
customer headline. Operating points use the declared response units; signed raw
measurements are not clipped or mislabeled as probabilities. Select operating points only on development data and explain
the choice; the wrapper intentionally does not optimize them on confirmation.

All planned inputs remain in alert/miss and coverage denominators, including
errors and abstentions. AUC covers only scored inputs. Brier is reported only for
declared calibrated responses and scored cases. AI without an alert includes
unassessed inputs; it must not be described as a genuine-content verdict.
Paired comparisons resample declared source groups and report each media type and
condition separately. A small bootstrap interval describes this panel, not future
customers or unknown generators. Report all original/edit counts alongside rates.

### Four-modality receipt example

From the repository root, run with Python 3.11 or later. The example contains
32 authored input records and no media, app dependencies, model weights, or
accounts. It demonstrates counts and localization for images, video, audio, and
writing. The values are fictional and measure receipt arithmetic.

```sh
scratch=$(mktemp -d)
example="$scratch/four-media"
review=backend/scripts/independent_review.py
python3 -IS "$review" demo "$example"
python3 -IS "$review" compare "$example/manifest.json" \
  "$example/baseline-confirmation.json" "$example/baseline-freeze.json" \
  "$example/candidate-confirmation.json" "$example/candidate-freeze.json" \
  "$scratch/replayed.json" --require-freeze-commitment
cmp "$example/comparison.json" "$scratch/replayed.json"
python3 -IS "$review" replicate-locations "$example/manifest.json" \
  "$example/location-truth.json" "$example/candidate-locations.json" \
  "$example/candidate-repeat-locations.json" "$example/candidate-freeze.json" \
  "$scratch/replayed-locations.json"
cmp "$example/location-replication.json" "$scratch/replayed-locations.json"
```

Each method retains 24 confirmation cases and four unavailable outcomes.
The comparison records false flags, recovered edits and source overlap.
The location replay shows two shifted highlights per medium whose sizes and
overlap counts match, even though they point to different places.
Changing a frozen operating point while using the original confirmation receipt
causes verification to stop before writing an output. Try the same review
commands on your own receipts after replacing the fictional source roster.

The local clean-source test used Python 3.11.9 with site packages disabled.
The example comparison reproduced byte for byte and a changed video point was
rejected. A separate clean-source run of the published mouth review reproduced
100 recorded trees, 492 point windows, 2,321 development windows and 88 case
outcomes. The audio identity reviewer also reproduced 5,328 planned inputs
from public source metadata. Those two checks used public files from commit
`b0aa41f`. [Mouth review instructions](mouth-public-portable-review-v12.md) ·
[Audio metadata commands](#model-free-checks)

### Count exact locations on your own files

Use `count-locations` to calculate the overlap between your known edit
locations and the regions a check reports. It counts the union of
overlapping regions once and checks bounds against the submitted file
geometry. The output works with `summarize` and `compare` above.
The command reads JSON only. Keep the native files, coordinates and any
identifying annotations in your own workspace. Share the aggregate
summary after reviewing its labels.

```sh
scratch=$(mktemp -d)
review=backend/scripts/independent_review.py
python3 -IS "$review" demo "$scratch/example"
python3 -IS "$review" count-locations "$scratch/example/manifest.json" \
  "$scratch/example/location-truth.json" "$scratch/example/candidate-locations.json" \
  "$scratch/candidate-counts.json"
cmp "$scratch/example/candidate-confirmation.json" "$scratch/candidate-counts.json"
```

For your own data, copy the structure of the demo files. The truth file
uses `chipmunk.independent-location-truth/1`, the manifest digest and the
split. Its `cases` bind each planned location to the case ID, exact input
hash, source `extent`, and `regions`. The source and target union sizes
must agree with `localization_truth` in the manifest. The response uses the
normal receipt fields, with `localization.regions` in place of the two
overlap counts for each measured case.

| Media | Extent | Region |
| --- | --- | --- |
| Image | `[width, height]` in submitted pixels | `[left, top, right, bottom]` |
| Video | `[decoded_frame_count]` | `[first_frame, end_frame]` |
| Audio | `[decoded_sample_count]` | `[first_sample, end_sample]` |
| Writing | `[extracted_codepoint_count]` | `[start, end]` in the same extracted text |

All coordinates start at zero and use exclusive ends. An assessed
check with no located region uses an empty list. An unassessed check uses
its status and reason, without a region list. Confirm the clock, frame
order or text extraction against the original before entering positions.

For an irregular image region, use a binary PBM mask in the original
image geometry. Truth or response can use a mask in place of `regions`.
The other side can be another mask or a set of rectangles. Use the
`mask` field `{"file": "case-a.pbm", "sha256": "<file SHA-256>"}` and
add `--image-mask-dir "$scratch/masks"` to the command. The basename,
file digest, native dimensions and target pixel count are checked before
the overlap is counted. The masks and their names are omitted from the
counted output. Keep the original mask files and private annotations
locally. The accepted PBM uses an exact `P4\nWIDTH HEIGHT\n` header,
most-significant-bit first pixels, one for included pixels, and zero
padding to each row's next byte. For a PNG mask with black background and
white edit pixels, this optional conversion produces the exact encoding.

```sh
mkdir -p "$scratch/masks"
python3 - "$mask_png" "$scratch/masks/case-a.pbm" <<'PY'
import sys
from PIL import Image

with Image.open(sys.argv[1], formats=["PNG"]) as image:
    if image.mode not in ("1", "L") or set(image.convert("L").tobytes()) - {0, 255}:
        raise SystemExit("Expected a black-and-white PNG with white edit pixels")
    width, height = image.size
    stride = (width + 7) // 8
    mask = bytearray(stride * height)
    for y in range(height):
        for x in range(width):
            if image.getpixel((x, y)):
                mask[y * stride + x // 8] |= 128 >> (x % 8)
with open(sys.argv[2], "xb") as out:
    out.write(f"P4\n{width} {height}\n".encode("ascii"))
    out.write(mask)
PY
shasum -a 256 "$scratch/masks/case-a.pbm"
```

## Review an audio supplement by acoustic condition

The [separated-stream presence comparison](audio-stream-presence-transfer-2026-11-11.md)
adds a natural meeting panel selected from public word annotations before
inference. Its separate verifier reconstructs exact input clocks, confirms
the development and confirmation roles and checks the published arithmetic.
The local recordings and per-interval outputs remain with the evaluator.

The [hard-overlap source panel](audio-hard-overlap-transfer-2026-11-11.md)
is reconstructible from pinned public VoxPopuli test partitions and the
commercially reviewed model assets. It includes a separate standard-library
verifier for the saved local audio hashes, interval grids, count distributions
and route totals. The method and aggregates are public. Keep native source audio
and per-recording responses in the evaluator's local workspace.

The [independent speaker-count comparison](audio-independent-overlap-count-2026-11-11.md)
includes a fixed acoustic routing replay with recorded input hashes, the author
checkpoint and scaler commitments, and separate controlled and natural-room
counts. An evaluator can replay the original five-second count and the two
level-normalized variants with the published method and locally held source
permissions. The decision review below works from a receipt without these weights.

The standard-library reviewer replays the effect of adding a second audio
response only where the current check was inconclusive. The authored example
uses fictional source groups, score values and routes. It reproduces the
counting logic without model weights or audio files.

```sh
review_dir=$(mktemp -d)
python3 -I backend/scripts/review_audio_supplement.py demo "$review_dir/example"
cat "$review_dir/example/summary.json"
```

The example recovers one synthetic input. Two extra genuine flags in the
mostly-single-interferer stratum belong to one source group. A third genuine
flag comes from a source annotated as overlapping speech that the router
classified as mostly one interfering voice. The summary counts source groups,
individual inputs, routing, and assessment coverage separately.

For your own evaluation, use the [portable manifest](#portable-evaluation-contract)
with audio cases only. Use `clean`, `mostly_single_interferer`,
`overlapping_interferers`, or `unknown` for the reference `condition`. Put
known foreground speaker and background recording or meeting links in
`lineage` so that they stay in one split. Use the foreground identity for
the parent `source_sha256` when a mix has a separate background recording.
Labeling and source permission remain the contributor's responsibility.

Use portable development and confirmation receipts with two additional
top-level fields. `incumbent_sha256` and `router_sha256` identify the incumbent
and routing implementations and their pinned configuration. Add
`incumbent_direction` and `observed_condition` to every prediction. Use
`ai`, `original`, `inconclusive`, or `unassessed` for the incumbent. Use the
four acoustic condition values for the measured route. Obtain that route
from the candidate's measurement and record an unavailable route as `unknown`.

Choose the internal operating point for each enabled noisy route on
development data. Use a JSON object such as the generated example's
`{"mostly_single_interferer": 0.6}`. Freeze it before running confirmation.

```sh
freeze_sha256=$(python3 -I backend/scripts/review_audio_supplement.py freeze \
  manifest.json development.json points.json frozen.json)
```

Put the printed digest in the confirmation receipt's top-level `freeze_sha256`.
Keep the freeze and its recorded time with the study. The replay validates
matching content, and the study log establishes when the freeze was made.

```sh
python3 -I backend/scripts/review_audio_supplement.py compare \
  manifest.json confirmation.json frozen.json summary.json
```

A scored response strictly above the frozen point can promote an incumbent
inconclusive result to an AI flag when the measured route is enabled. The
summary evaluates the result by reference condition and includes all planned
cases. For a genuine stratum, compare the `ai` count with `planned` and
`incumbent_assessed` to see false flags and coverage together. The aggregate
contains counts and whole-receipt digests. Review the aggregate before sharing
it, especially small groups, and keep media, per-case scores and private
manifests local.

## Evidence localization

The portable receipt can also count localization outcomes without carrying source
media, coordinates, or private text. To include them, add `localization_truth`
to each planned case with `unit`, `source_units`, and `target_units`. Use
`pixels` for images, `frames` for videos, `samples` for audio, and
`unicode_codepoints` for writing. Use zero target units for controls and the
exact union of affected units for positive inputs. In the corresponding
prediction, add a `localization` object with its own `status`, plus measured
`reported_units` and `overlap_units` for `ok`, or a reason for an unassessed
outcome. An assessed input with no region has zero reported and overlapping
units. The summary reports per-condition counts, control highlights, and
positive IoU, F1, and recall, including misses and unassessed cases in the
positive means. File response and localization may have different coverage.
The comparison command also reports paired localization differences by media
type, condition and truth label. It averages related variants within each
declared source group and resamples source groups for its intervals. It reports
individual case transition counts alongside the group-weighted difference.
Read the coverage difference with overlap and false regions so a missing
source mapping cannot be mistaken for an improvement.
For an image with 1,000 source pixels and a 100-pixel truth mask, a report
covering 80 pixels with 60 pixels inside the mask contributes an IoU of
60 / 120. Keep the exact region geometry in your private evaluation records.

The receipt checks arithmetic bounds. A separate audit should verify the
underlying masks, clocks, and text positions:

- Images: original dimensions and orientation, reference-mask hash, exact scoring
  grid and projection, displayed-region policy, IoU/pixel F1 including misses,
  and authentic inputs receiving highlights. Bounding rectangles are not masks.
- Video: known manipulated frame ranges, exact presentation timestamps, track
  coverage, interval overlap including misses, and before/during/after frames.
- Audio: original and decoded hashes, sample rate, sample bounds, independent
  source groups, measured mixture SNR, and whether the background was separated
  or remains part of a mixed recording.
- Writing: immutable extracted text, Unicode offset convention, scored context
  windows, source passage positions, and whether word-level labels are actually
  available. A passage response is not proof of individual AI-written words.

## Model-free checks

For a first independent contribution, the corrected audio source graph and
execution freeze can be reviewed with the standard library alone:

```sh
python3 backend/scripts/review_audio_identity_plan_v9.py \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz /tmp/new-source-review.json
python3 backend/scripts/review_audio_identity_execution_v9.py review \
  docs/evaluation-panels/audio-v9-execution-freeze.json \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz \
  /tmp/new-execution-review.json --project .
```

Use new output paths. Report the commit, your environment, receipt hashes and
which checks you independently inspected. A successful metadata/code review
does not establish native inference parity, predictive accuracy or customer
usability. The [writing source preflight](writing-source-preflight-v2.md) similarly
distinguishes source-position verification from detector validation and records
unresolved source terms before reuse.

The [complete corrected audio result](audio-identity-v9-development-evaluation.md)
also supplies all consumed representations, fitted parameters and outcomes for
a numerical review with optional CPU refit. That command needs Python numerical
dependencies but no source audio or pretrained-model downloads. It reproduces
the original feature/fit artifacts and all conclusions/gates; it does not
re-execute native model forwards or establish fresh accuracy. The native replay
receipt is separate. The candidate failed development and remains disabled.

The [conditional audio result](audio-conditional-v10-development-evaluation.md)
adds a separate nonlinear/quality comparison with all outcomes, small fitted
numerical trees and 121 native replay receipts. Its public-only numerical refit
uses the already published vectors/quality packet, without private manifests or
pretrained downloads. Both eligible arms failed; reproducing that rejection is
a useful contribution. Please distinguish arithmetic reproduction, source/audio
re-execution, predictive evaluation and usability feedback in any review.

The [full-vector audio comparison](audio-full-representation-v11-development-evaluation.md)
also remains disabled after four failed requirements. Its small public packet
reuses previously published vectors/cached probes. The supplied command refits
only fitting sources and reproduces all conclusions/gates without private
manifests or model downloads. Native source re-execution covers 125 fixed/new-probe
cases and is a separate recorded check.

For a model-free starting point with a clean installation, use the
[portable v13 numerical review](audio-public-portable-review-v13.md). It has a
separate seven-package requirements file and blocks app/neural imports. Local
verification reconstructed every feature record, both complete fitting-only
objects and all 1,449 outcomes in a new environment. The result remains rejected;
this is a reproducibility check, not fresh accuracy or outside participation.

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q backend/tests/test_independent_review.py
```

Those tests use explicitly synthetic response fixtures. They verify the audit
contract and denominators, not the predictive accuracy of any detector.

The [image context/region-point comparison](image-region-point-v15-result.md)
preserves all four arms and a rejected primary, with complete native/source
replay and an actual 12-test Linux arithmetic receipt. Its frozen data were
already consumed; no new outside human reviewer or fresh-quality claim follows.
The [broader provider-signal inventory](provider-signal-broad-search-2026-11-08.md)
identifies additional qualification routes and explicit documentation gaps;
provider policy review is separate from authentic file/response testing.

The [background-context video experiment](video-background-context-v25-result.md)
adds 18 authored spatial/temporal and source-role tests, with an actual isolated
Linux receipt. The native280-file study and separate600-outcome audit are
complete, but its primary fails replacement rules. Source footage/geometry/
descriptors stay private; the public tests reproduce arithmetic and safety
contracts, not native face inference or fresh accuracy.

The [public LTX audio marking contract](ltx-marking-audio-v1-result.md) adds
80 authored native/file cases and a complete separate sample/codec audit. The
[actual Linux receipt](evaluation-panels/ltx-marking-audio-v1-linux-array-review.json)
passes 20 arithmetic/contract tests without media or neural inference. No
provider-authentication, natural-speech accuracy or outside human review follows.
The [provider follow-up](provider-signal-followup-2026-11-08.md) records 82
families and explicit gaps, separately from operational verification.

The [audio-aware mouth study](mouth-audio-context-v14-result.md) provides another
model-free starting point: `python -m scripts.review_mouth_audio_public_v14`
with a new receipt path and `PYTHONPATH=backend`. Only NumPy is required; it
checks authored signed lags, exact audio blocks and no-wrap/degenerate cases
against separately written scalar equations. It does not replay private audio
or train the small head. The [actual Linux math receipt](evaluation-panels/mouth-audio-context-v14-linux-math-review.json)
also passes. The complete local 272-track/6,373-window study rejects
the candidate for excess ordinary flags. Outside human review remains pending.
