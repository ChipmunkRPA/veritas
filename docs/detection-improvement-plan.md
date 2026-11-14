# Detection improvement and independent evaluation

Started 5 November 2026. GCP deployment is deferred. This work improves detection
and the evidence supporting it before expanding hosting.

## Requirements and order

On 7 November the user added three priorities: verified provider provenance
before statistical detection; stronger customer disclosures and separate expert
engagement; and recent-paper/code review feeding independently tested methods.
The named provider review includes OpenAI, Anthropic, Google/Gemini, Meta/Muse,
xAI/Grok, MiniMax and DeepSeek, with further major Chinese providers in scope.
See [actual provider coverage and limits](provider-provenance-coverage.md) and
the [recent-method review](recent-methods-review-2026-11.md). Proprietary verifier
access is not simulated, unsigned labels/spacing cannot stop detection, and
no customer file is automatically sent to an external provider.

On 7 November the user directed region/check/context-specific calibration and a
tested combined result **across all four modalities**. The
[shared evaluation implementation and required real-data work](specialist-calibration-design.md)
now define that path. It supplements the full requirements below; an authored
software smoke does not complete any modality's real-data calibration or release.

1. **Image edits:** compare a commercially permitted alternate localizer against
   the currently selected PNG splice check and the fallback image localizer.
   Include small edits, native PNG, and JPEG recompression. Improve actual edit
   overlap at a comparable authentic-image review rate before adopting a change.
2. **Video face swaps and lip sync:** investigate missed swaps and incomplete
   coverage. Compare candidate changes separately, using new source groups and
   ordinary cuts, lighting changes, compression, and correctly synchronized
   speech as controls. Preserve exact source timestamps and before/during/after
   evidence. Scope audio, face manipulation, and fully generated video separately.
3. **Audio:** expand noisy-speech evaluation across unseen generators, speakers,
   background sources, and recording conditions. Keep source families together
   across data splits; use each candidate's own calibration and keep rollback.
4. **Writing:** test fresh human/AI article pairs and realistic edited or mixed
   passages. Evaluate passage position and context as well as document scores.
   Do not manufacture calibrated authorship confidence from raw responses.
5. **Independent collaboration:** provide a reproducible, model-independent
   evaluation contract, contributor instructions, and privacy-safe failure-case
   reporting. Publish methods and aggregate results with limitations. Do not
   publish customer uploads or imply independent reviewers have participated
   before they have done so.

## Current decision policy

The owner's later instruction is to reconsider earlier rejections when a
substantial detection gain accompanies a modest false-alarm increase. Do not
treat every additional false flag as an automatic veto. Compare complete paired
gains/losses, misses, ordinary-input flags, assessed coverage and uncertainty;
keep complementary checks and rollback when their errors differ. An improved
candidate still needs its own calibration, fresh transfer evidence and local
customer-flow verification before becoming a default.

Preserve the original frozen thresholds, gates and outcomes below and in past
capsules as historical evidence. Report their original pass/fail separately from
the current adopt/retain/defer judgment; do not retrospectively tune consumed
test cases or rewrite a failure as a pass. Internal predeclared research settings
are not public screening promises or calibrated population probabilities.

## Latest checks added and evaluated locally

- [Adjacent soundtrack repeats](video-adjacent-sound-transfer-2026-11-14.md)
  brought the combined short-video result from seven to 15 of 44 known
  repeat pairs without adding ordinary flags. The AAC transfer added no
  useful clean cases and found one noisy edit shared by both carriers.
  The method remains outside the app.

- [PDF copyable text vs. glyph text](document-pdf-copyable-text-2026-11-14.md)
  located changed amounts, omissions, inserts, and replacements even when
  the rendered page was identical. It also reported a legitimate OCR
  correction. Locations and coverage appear in browser, PDF, and JSON.
  The check carries zero AI-authorship weight.

- [Small JPEG regions below the area rule](image-small-region-repeat-2026-11-14.md)
  found one useful edit and eight additional ordinary highlights when the
  area rule was removed. The horizontal repeat recovered no additional
  edits. The browser and PDF now show the recorded area denominator.

- [Progressive JPEG earlier-copy comparison](image-reference-progressive-2026-11-14.md)
  located four additional object removals across two photographs and six
  more deliberate local retouches on the same photos. The comparison added
  no highlights on unchanged or globally processed controls. It carries
  zero AI-authorship weight.

- [Agreement between three views of a sentence](writing-context-agreement-2026-11-14.md)
  added no useful locations on the two journal panels. The optional
  longer-passage check now retains complete centers when some inputs fail or
  exceed the time budget. Evidence shows the missing source ranges.

- [Mouth-hold soundtrack under strong background sound](video-mouth-hold-background-transfer-2026-11-14.md)
  found zero added reliable periodicity recoveries in held testing.
  Speech activity survived environmental noise but could not distinguish
  foreground speech from competing voices. New local video evidence
  distinguishes digital silence from nonperiodic sound and reports its
  level on the same clock as the waveform and mouth crops.

- [Sound during a short mouth hold](video-mouth-hold-sound-2026-11-13.md#sound-evidence-on-two-second-clips)
  extends source-clock audio evidence to an existing two-second H.264
  picture check. Simulated checkout, signed original replay and the
  report matched on a held interval with a shifted audio stream.

- [Denser and narrower lip-sync timing](video-lipsync-window-schedule-2026-11-14.md)
  tested the two schedule changes separately and together on acted and
  conversational speech. They all measured more edited clips and located
  zero of 44 local audio shifts. The denser-only and combined changes
  flagged 18 and 27 of 132 ordinary controls, respectively. The timing
  schedule stays unchanged.

- [Speech and mouth timing coverage](video-timing-window-coverage-2026-11-14.md)
  records which exact video windows could not be assessed and why. A held
  mouth and an off-camera speaker retained their own dispositions through
  the browser, PDF, JSON, and signed source replay. This improves visibility
  into coverage and keeps detection rules unchanged.

- [Short speech after a longer-window skip](audio-short-admission-transfer-2026-11-14.md)
  adds an optional local inspection path for brief speech. On the matched
  source panel it found one noisy edit and flagged three ordinary noisy
  recordings. The new route carries zero weight in the overall assessment.

- [Matched resynthesis and competing-speech controls](audio-sine-resynthesis-transfer-2026-11-14.md)
  measured 152 new deliveries. The current review flagged 8 of 12 clean
  resyntheses that preserve the words. The two interval-trained heads located
  the same short edit in clean and mixed deliveries and added three noisy
  original flags. The serving voice calibration stays in place.

- [Interval-trained noisy speech review](audio-interval-learning-2026-11-13.md)
  added three useful locations across two test banks at −5 dB, and nine
  additional false flags on ordinary files. Both trained heads stay offline.
  The existing short voice review now shows checked and below-speech-minimum
  windows with native source timestamps in the browser, PDF and JSON.

- [Side-by-side face tracking](video-face-split-track-2026-11-13.md)
  measures both sides of the picture separately. The study compares
  individual tracks, the largest face, and a guard for changes outside the face.

- [Face tracking through changes in guest size](video-face-position-track-2026-11-13.md)
  recovered six exact intervals across three of eight source recordings,
  with no added flags on 32 matched ordinary controls. The measurements
  cover a qualified four-second video profile and carry zero AI-confidence weight.

- [Short speech-window attenuation](audio-voice-attenuation-2026-11-13.md)
  added two useful locations on 28 noisy edits and ten ordinary flags
  with the simpler rule. The comparison stays offline. The app and PDF
  now link flagged short speech windows to overlapping acoustic evidence.

- [Denser earlier-copy speech review](audio-dense-clean-reference-2026-11-13.md)
  located 14 of 16 short synthetic alterations under competing speech,
  up from six on the same earlier-copy pairs. At −5 dB it highlighted one
  of 24 untouched or processing controls, down from five. It is a local
  trial for six-second aligned comparisons with a clean earlier recording.

- [Native PNG regional review](native-regional-serving-v47-result.md#optional-app-integration)
  is enabled on the local instance. The paid-flow replay added one location
  to a previously evaluated edit and none to the matching untouched photo.
- [PNG regional repeat](image-regional-flip-confirmation-2026-11-13.md)
  is enabled locally for the supported PNG size. It reduced ordinary
  highlights from 37 to 26 of 68 and useful masks increased from 11 to 12
  of 40. It carries zero AI-confidence weight.
- [Two reflected views](image-regional-two-reflections-2026-11-13.md)
  added no useful edit locations on the 108-image replay. The browser
  and Standard PDF now show the existing horizontal repeat mask with
  source bounds and overlap counts for both supported and withheld candidates.
- [Progressive JPEG regional evidence](image-regional-progressive-jpeg-2026-11-13.md)
  adds support for standard quality-75 progressive files in the existing
  development profile. All 90 matched decodes and six paired regional runs
  agreed exactly with baseline JPEG. A separate known insertion reproduced its
  crop, mask and signed report through simulated checkout.
- New analyses apply an English-language coverage check before statistical
  writing models run. Clearly unsupported, detected mixed, or uncertain
  language yields an unassessed authorship result. Purchase disclosures
  explain the scope before checkout. The language check samples up to 48
  passages in addition to the complete extract, so short language changes can
  escape it. Verified signed provenance is evaluated separately.
- [Sound during a measured mouth hold](video-mouth-hold-sound-2026-11-13.md)
  binds a playable mono excerpt, waveform and periodicity measurements to
  an existing verified HD picture interval. A controlled clear-speech insert
  registered on seven of seven intervals. None of the same inserts registered
  with strong added noise. This evidence carries zero weight in AI confidence.
- [Mouth-hold sound with a timestamp break](video-mouth-hold-sound-2026-11-13.md#timestamp-breaks-outside-the-selected-sound)
  keeps usable audio before or after a gap, and leaves sound unassessed
  when a gap crosses the selected pictures. Three real MP4s matched in the
  local worker and signed exports. A separate filter probe gained two of
  20 noisy periodic observations at best and lost four of ten clean ones.
- [Blind pixel prediction](image-crossfit-prediction-2026-11-13.md) measured
  900 held deliveries against exact local image masks. It added at most two
  useful masks per encoding on the primary set, added ordinary-processing
  alerts and found no edits in the independent-source set. It stays offline.
- PDF text visibility checks every page during document extraction for searchable text
  that the PDF explicitly does not draw. The evidence view, report and JSON retain page
  coverage and locations in unrotated PDF coordinates. Whole-document writing confidence
  is withheld when this layer is present or any page could not be checked for visibility.
  Searchable scans can contain such a layer, so its presence supplies zero evidence of AI
  authorship. Original-file replay still supports archived first-12-page checks.
- [Matched-rate speech diagnostic](audio-background-rate-diagnostic-2026-11-13.md)
  reproduced five false alerts on 12 genuine speakers in loud traffic at both native
  48 kHz and the existing supported 16 kHz. The evidence view and PDF disclose this
  concrete failure mode of the optional background-reduced check.
- [Background-reduced speech safeguards](audio-background-safeguard-2026-11-13.md)
  tested 204 uploads over rain, train, road traffic and clean speech. The optional
  review added 24 hits on 63 noisy synthetic deliveries and 11 flags on 81 noisy
  genuine deliveries. Checking the original voice band removed one false flag and
  lost four hits, plus two clean synthetic flags. Reversing signal polarity removed
  no false flags. The study preserves the measured limits and keeps both probes offline.
- [Persistence of noisy speech flags](audio-background-persistence-2026-11-13.md)
  tested 116 held uploads. Requiring both halves to agree cut false flags on
  noisy genuine clips from three to two and cut synthetic flags from eight
  to six. Overlapping windows and restricted frequency bands removed no
  false flags. All three probes remain offline.
- [Paired speech under loud backgrounds](audio-paired-loud-noise-2026-11-13.md)
  checked 193 English deliveries at the earlier fixed score points. On
  environmental noise the union added one synthetic flag and zero false
  flags beyond the optional background reduction. Competing speech added
  three detections across two families and six genuine false flags. The
  paired check remains offline. A public reviewer recounts anonymous
  directions without audio files or model assets.
- [Native competing-voice intervals](audio-competing-speech-localization-2026-11-13.md)
  measured 128 excerpts from eight separate, previously unused AMI
  meetings. Exact unpadded input marked no minimal-overlap controls in
  its 32-clip bank, but localized 18 of 32 high-overlap cases versus 19
  for right padding. On the separate archived comparison, withholding AI
  flags removed three genuine-foreground false flags and eight or nine
  correct synthetic flags. Both approaches remain in evaluation. The two
  public replays recover the original PCM ranges from the publisher.
- [Speech-window coverage](audio-speech-admission-transfer-2026-11-13.md)
  now shows measured selection reasons and exact timestamps in new audio
  and video analyses. In a four-room, 96-clip study, gain admitted 15 of
  24 speech-heavy excerpts versus 12 on original audio. The gain view
  remains outside the voice-origin assessment. The [matched downstream
  replay](audio-gain-gate-voice-transfer-2026-11-13.md) added zero detections
  on 193 archived and 68 newly prepared quiet deliveries. The three
  newly admitted synthetic windows all returned inconclusive.
- [Grammatical attachment changes](writing-dependency-transfer-2026-11-13.md)
  added zero locations on 40 held sentence edits. A part-of-speech comparison added
  one location and one false alert. Both remain experimental.
- [Typed contextual sentence review](writing-typed-decision-transfer-2026-11-13.md)
  increased useful locations from 10 to 13 across 40 held English edits and
  untouched highlights from 11 to 12 of 20. It remains experimental.
- [Fresh report calibration and contrast](writing-context-margin-transfer-2026-11-13.md)
  found zero known locations out of 40 at both frozen context points, even
  with no flags on 20 ordinary excerpts. The existing check found ten and
  flagged eight ordinary excerpts. This rules out adding the candidate at
  either tested point.
- [Sentence rotation](writing-sentence-rotation-transfer-2026-11-13.md)
  added no useful locations on 64 known single-sentence rewrites from two
  generators and stays offline. New analyses count completed sentence-position
  readings in the passage coverage map. Standard PDF reports and JSON carry
  the same counts and source ranges. Short-passage PDF rendering now handles
  the three-sentence record alongside the existing longer variant.
- [Short H.264 face localization](video-brief-face-avc-clock-confirmation-2026-11-13.md)
  increased correct ranges from nine to ten on 16 edited deliveries from eight
  source groups. Neither route flagged any of the 40 ordinary clips. The new
  route is a local trial with zero confidence weight. Both this trial and the
  existing short WebM review explain the two boundary readings and show the
  original frames in the browser, PDF and JSON.
- [Scheduling both boundaries together](video-brief-face-paired-shortlist-2026-11-13.md)
  added no edit locations or ordinary flags on 56 retained H.264 files.
  All 11 missed edits were in clips already carrying one short range.
  The app and Standard PDF now show the six source frames compared at
  each boundary, matching the JSON. Simulated checkout and signed replay
  verified those lists on both encodes of the known edit.
- [A second distinct H.264 face range](video-face-distinct-avc-2026-11-13.md)
  adds 10 exact locations on the retained bank and one on the separate
  eight-source panel, with no additional ordinary flags on 103 controls
  across both runs. The wider search is a local opt-in. The source-separated
  test fell short of the confirmation gate. Both ranges and coverage
  replayed through prepaid local analysis, JSON, PDF and signed originals.
- [Moving face positions in the first H.264 scan](video-face-moving-shortlist-2026-11-13.md)
  recovered four exact intervals on development clips with zero extra
  flags on 101 unique ordinary deliveries. It added no locations or
  flags in the separate 56-file confirmation and remains a local opt-in.
  The customer flow verified one recovered range and its six images
  through simulated prepayment, JSON, PDF and signed replay. The held
  misses had competing face detections around the authored boundaries.
- [Follow the dominant face when smaller detections appear](video-face-primary-track-2026-11-13.md)
  recovered four exact ranges in the previously missed source without
  adding any flags across 121 ordinary clips. A separate picture-in-picture
  test recovered 11 of 14 edited deliveries across six sources with zero
  flags on 21 ordinary controls. The option remains a local trial.
- [JPEG75 and JPEG85 detail loss](image-jpeg-detail-quality-transfer-2026-11-13.md)
  adds 19 useful local blur ranges per format, without adding any highlight
  on 306 controls per format. It requires an earlier matching copy and
  contributes zero AI-confidence weight.
- [Video insertion verification](video-insert-retention-2026-11-13.md)
  withholds a single range when the pictures outside it do not reliably
  match. Across two held collections it retained 20 correct ranges and
  withheld 32 incorrect single ranges on multiple or mixed edits. It also
  withheld two of 29 previously correct ranges, both on the same source.
- [Single-upload public replay](independent-testing.md) now reproduces the
  latest four media studies in one command on macOS and Linux. The replay
  verifies recorded measurements and decisions. Independent external
  review remains open.
- [48 kHz background review](audio-background-lossless-rate-transfer-2026-11-13.md)
  transfers 18 of 19 earlier useful noisy synthetic indications, but adds
  five flags on 12 genuine native-rate speakers under strong roadside
  sound. This route remains outside the app.
- [Repeated percentage statements](writing-percentage-consistency-2026-11-13.md)
  links both positions when the same complete statement uses two different
  percentage values. It located both values in all 48 seeded disagreements
  in the TXT study and gave no findings on 608 ordinary and constructed
  controls. Supported Word, plain-text and mapped HTML email, and selectable
  PDF documents record source-level locations. It carries zero AI-authorship
  weight. Dates, cohorts, denominators and citations that differ in the text
  are treated as different claims.
- [JPEG detail loss](image-jpeg-detail-loss-2026-11-13.md) adds a narrow
  earlier-copy check. It located moderate local blur in 14 of 34 cases,
  versus one with the existing comparison. It added useful moderate or
  strong regions across 13 sources and selected none of 442 other pairs.
  Browser, PDF and JSON identify the measured area and retain the source crops.
- [Multiple video insertions](video-temporal-multiple-inserts-2026-11-13.md)
  improved complete locations from zero to nine of 24 in the held test.
  It added no flags across 48 negative controls and kept eight correct
  single-insert results. The successful multi-insert results came from
  three of eight sources, so the method remains offline.
- [Randomized audio edit positions](audio-random-clock-stock-2026-11-13.md)
  confirmed the five-window review can reach both clip edges. The older
  sparse point located two of 16 edits at −5 dB versus one today, with
  three additional ordinary flags. A separate enhanced-audio view helped
  on some positions and added eight ordinary flags, so serving is unchanged.
- [Standalone writing edit intensity](writing-edit-intensity-transfer-2026-11-13.md)
  found two additional locations among 255 known edits, both variants from
  one journal article, alongside seven newly flagged ordinary work groups.
  The candidate stays offline. A [four-media replay](independent-testing.md#replay-delivery-and-earlier-copy-comparisons)
  verifies this round's public evaluation ledgers.
- [Video insertion localization](video-temporal-insert-confirmation-2026-11-13.md)
  adds a short earlier-copy comparison. It recovered both boundaries of all
  16 held generated insertions and 13 of 16 conventional insertions. The
  existing route assessed none of those insertions. The new check issued no
  insertion report on 40 unchanged, equal-length or wrong-reference controls.
  Source timestamps and pictures before, during and after appear in the
  browser, PDF and JSON. It makes a physical comparison with zero AI weight.
- [Paragraph relocation](document-mixed-format-comparison-2026-11-13.md#reordered-paragraphs-in-matching-txt-or-word-files)
  links the same body paragraph across matching TXT or Word drafts. The
  replay verifies 64 exact locations and 208 controls with no relocation
  reported. The standalone writing comparison based on overlapping model
  responses added zero useful locations across 200 known edits and stayed
  outside the application.
- [JPEG fine-detail differences](image-jpeg-ac-difference-2026-11-13.md)
  on an earlier copy improved generated edit localization from 65 to 66 of
  68 confirmation pairs. Its strongest measured result was conventional
  local blur, from 1 to 26 of 34. One source contributed the generated
  improvement, so the method stays in evaluation for now.
- [Stock-voice splices under competing speech](audio-stock-voice-transfer-2026-11-13.md)
  tested the same speech classifier with different window schedules and
  points on 124 fresh deliveries. At the current point, the two-window
  schedule recovered 7 of 16 edits at 0 dB versus 5 of 16 for five windows,
  with 2 versus 7 flags across 24 ordinary controls. Both recovered 3 of 16
  edits at −5 dB. The authored edit positions favor the two-window schedule.
  The current audio setting stays in use while randomized positions are tested.

## First new image comparison: original predeclared design

- Candidate: official IML-ViT checkpoint, subject to source, checkpoint, and
  commercial-license verification. Preserve original files and pin SHA-256.
- Development may reuse consumed evaluation pairs, explicitly labeled as
  development. Freeze preprocessing, map selection, and gates before any fresh
  confirmation inference. An engineering smoke is not validation.
- Confirmation: at least 48 new source-image pairs, selected deterministically
  from the TGIF/TGIF2 author test splits, excluding all source IDs in earlier
  local TGIF comparisons. Include Adobe Firefly, SD2, and FLUX Fill. Acquisition
  failures stay visible; do not replace cases based on detector responses.
- Score native images and deterministic JPEG quality-75 versions of both
  original and edited images. Variants of a source remain one statistical group.
  Recompressed edit masks refer to the original known edit coordinates, rather
  than treating whole-image JPEG changes as manipulation ground truth.
- Compare displayed-mask IoU and pixel F1, edit misses, authentic images with a
  displayed region, assessed coverage, errors, latency, and memory. Misses count
  as zero overlap. Report each generator/format independently and bootstrap
  paired differences by source group. Checkpoints do not enter the overall
  confidence estimate on the strength of localization results alone.
- Adoption requires more edits overlapping the selected regions and higher mean
  IoU, with at most one additional authentic image highlighted in each format on
  the same confirmation sources. Record uncertainty; a small passing comparison
  supports a scoped local release, not a universal accuracy claim.
- Preserve baseline weights, immutable past results, model-name filtering, and
  corresponding source coordinates across UI, PDF, and JSON.

## Completion evidence

Every change needs pinned inputs and weights, a locked development decision,
complete held-out outcomes including failures, relevant regression tests, and
an explicit adopt/reject/defer record. A public contribution process is not
evidence of recruited collaborators or completed independent review. Track
remaining requirements here and in linked experiment records; do not call the
overall improvement program complete while any required media evaluation or
integration remains unfinished.

## Progress on 13 November

- The [WAV and M4A timing check](audio-reference-local-shift-2026-11-12.md#wav-or-flac-paired-with-aac-m4a)
  uses the declared AAC sample extent to exclude trailing decoder padding
  before comparing timing. It located 34 of 48 known shifts with no flags
  on 216 ordinary pairs. The local app shows both playable intervals
  and the extent in the browser, PDF and JSON.
- The [speech word-position comparison](audio-paired-word-change-2026-11-13.md)
  added 16 ordinary flags on reprocessed or noisy audio when used alone.
  Requiring acoustic agreement cleared them, but removed useful edits.
  Neither channel located an edit under strong construction noise.
- The [native video insertion comparison](video-temporal-insert-native-2026-11-13.md)
  located four generated and three ordinary inserts against earlier
  footage. The four publisher clips place the inserted content later than
  their annotations imply. The paired check stays experimental.
- The [ordered surprisal experiment](writing-surprisal-dynamics-transfer-2026-11-13.md)
  added no useful locations to the 12 of 40 found by the current writing
  comparator and added two ordinary-file flags. It stays outside the app.
- The [exact JPEG90 confirmation](image-reference-same-jpeg-confirmation-2026-11-13.md)
  adds a local opt-in for otherwise unassessed pairs. Across both held
  banks it improved useful locations from 108 to 111 of 114, with no
  added flags on 285 untouched or global-processing controls. The fourth
  new region covered 48.95% of its known edit. Each bank's original
  source-breadth decision remains recorded.

- The [longer audio bandwidth comparison](audio-bandwidth-return-2026-11-13.md#transfer-to-eight-and-twelve-seconds)
  adds eight- and twelve-second lossless inputs and native 16-bit FLAC.
  On two additional AMI recordings, twelve-second deliveries assessed all
  64 edited files, versus 50 assessed at six seconds. They located 36
  edits versus 35 at six seconds, with three ordinary observations at
  each duration. The check carries zero AI-confidence weight.
- The [re-encoded video soundtrack comparison](video-soundtrack-reencode-2026-11-13.md)
  found 30 of 40 expressive and eight of 16 conversational timing shifts
  per audio carrier on matching H.264 pairs. It produced no timing finding
  on 154 controls per carrier, including synchronized mouth and full-frame
  changes. The local evidence shows both source timelines and carries
  zero AI-confidence weight.
- The [Word paragraph comparison](writing-gao-archive-comparison-2026-11-13.md#word-changes-across-paragraphs)
  localized 38 of 48 authored changes across paragraph breaks on 24
  held report sources. No unchanged reflow, boundary deletion or quote
  received a change. The browser, PDF and JSON show exact native positions.
- The [Word boundary omission check](writing-gao-archive-comparison-2026-11-13.md#word-omissions-at-paragraph-breaks)
  located 40 of 48 known omissions across adjacent paragraphs on 24
  separate reports. It added no omission findings on 120 ordinary layout,
  quote and hidden-text controls. Browser, PDF and JSON show the two
  native character positions and the omitted wording from the archive.
- The [public bandwidth replay](audio-bandwidth-return-2026-11-13.md#replay-from-public-source-ranges)
  now downloads only the pinned 12.3 MB needed from the AMI publisher and
  reproduces all 192 audio measurements from a clean checkout.
- The [robust Mel trial](audio-trimmed-mel-normalization-2026-11-13.md)
  retained the submitted waveform and changed only feature normalization.
  It found none of 12 known short voice edits under competing speech
  at the locked point and remains experimental.
- The [spectral return timing trial](audio-acoustic-return-timing-2026-11-13.md)
  tested whether similarity across the two audio flanks could select short
  edited intervals. Combining it with dense speech scores found none of
  12 known edits at the locked point and added no gain at equal ordinary
  group counts. It stays experimental.
- The [PartialSpoof source audit](audio-partialspoof-source-audit-2026-11-13.md)
  verified the published protocols and 121,461 timing records. The release
  supports preparation of a speaker-held training panel, subject to native
  audio clock checks. It omits per-edit generator identity, so a separate
  source with named generators is needed for confirmation.
- The [PartialSpoof native intake](audio-partialspoof-native-intake-2026-11-13.md)
  scanned 3,333 native WAVs from bounded publisher ranges. It verified six
  short edits and two speaker-matched controls. Those bounded ranges do
  not support a supervised fit.
- The [Half-Truth native audio test](audio-had-native-intake-2026-11-13.md#positive-only-localization-check)
  found useful locations on eight of eight known Mandarin edits with
  one-second windows, versus two with the stock two-second windows.
  The public test prefix contains no fully genuine recordings.
  The shorter check remains experimental.
- The [English named-generator census](audio-melo-short-source-census-2026-11-13.md)
  verified 914 eligible genuine and generated prompt pairs. All 11
  paired readers appeared in earlier local source plans, so the archive
  prefix supplies development material and cannot serve as a fresh
  reader-held test.
- The [DailyTalkEdit audit](audio-dailytalkedit-source-audit-2026-11-13.md)
  verified 2,475 dialogue records without downloading audio. The test
  split contains no edits at or below 0.8 seconds, the corpus uses two
  original actors and one generator, and upstream rights remain unclear.
- The [matched English speech-edit comparison](audio-sine-preview-2026-11-13.md)
  found two of ten cut-and-paste edits with the dense voice windows,
  versus zero with the current windows. Neither window located the ten
  seamless infilling edits. Both left all ten original controls from
  the preview's sole reader unflagged. The dense check stays experimental.
- The [aligned earlier-copy speech comparison](audio-aligned-speech-difference-2026-11-13.md)
  found seven of ten seamless edits and 13 of 18 clean VoiceCraft edits
  versus zero with the current earlier-copy check. A processing guard
  removed the nine SINE resampling and equalization flags. It could not
  assess any noisy edits or the second clean PartialEdit generator.
  It remains experimental.
- The [PDF Content Credentials check](pdf-content-credentials-2026-11-13.md)
  verifies a whole-file AI declaration locally before acquiring writing models.
  Signed document coverage, modified PDF controls and payment-first behavior
  were checked with a public provider example. The option is local and off by
  default.
- The [fresh lip-sync study](video-lipsync-repeat-fresh-actors-2026-11-13.md)
  added four useful locations per carrier on expressive speech with no added
  ordinary flags. It added none on the separately sourced conversational clips
  and retained a synchronized-copy error. The combination stays experimental.
- The [four-second H.264 face comparison](video-brief-face-avc-confirmation-2026-11-13.md)
  located 12 of 12 short swap deliveries versus 10 of 12 with the current
  Full check and added no flags on 30 ordinary deliveries. Both exact
  additions are encodes of the same edit. The H.264 proposal stays experimental.
- The [general-video insertion source audit](video-temporal-insert-source-intake-2026-11-13.md)
  verified 100 labeled triplets and identified 34 with equal declared
  original and generated frame rates. Four smallest triplets form a
  reproducible native preflight. This intake scored no media.
- The [JPEG90 reflection check](image-regional-jpeg90-reflection-2026-11-13.md)
  retained all four useful locations in the 11-photo confirmation and reduced
  ordinary highlights from 15 to 10. The existing JPEG75 path flagged none of
  those controls, so the new check remains experimental.
- The [JPEG90 paired-source audit](image-jpeg90-reflection-source-context-2026-11-13.md)
  reproduced the held source masks and found that 12 of 13 ordinary
  highlighted components lacked corroboration from the aligned earlier copy.
  Nine were pixel-identical. The one corroborated region was a known
  conventional paste. Standalone JPEG90 reflection stays experimental.
- The [paired JPEG90 transfer](image-jpeg90-paired-corroboration-2026-11-13.md)
  measured 154 pairs across 14 photos outside the preceding JPEG90 banks.
  The earlier-copy check located 24 of 28 generated changes and 26 of 42
  ordinary local changes. Intersecting with the learned mask lost both
  useful standalone locations. The combined candidate stays experimental.
- The [exact JPEG90 block comparison](image-reference-same-jpeg-2026-11-13.md)
  uses the complete app as the comparator and locates 46 of 46 held edits
  versus 44. Both additions are from one of 23 photos. Whole-image or
  untouched highlights stay at 21 of 115. The experimental fallback needs
  confirmation across additional photographs.
- The [selectable PDF archive comparison](writing-gao-archive-comparison-2026-11-13.md)
  matches indexed report wording and places verified changes on the original
  pages, with the source crop in the browser and report. A two-page transfer
  matched 19 of 20 untouched excerpts and located 37 of 40 provider edits.
- The [JPEG90 agreement test](image-regional-jpeg90-quality-agreement-2026-11-13.md)
  removed all 15 ordinary flags in its separate 11-photo confirmation, but
  retained only two of four useful locations, from one photo. It remains offline.
- The [repeat-speech video study](video-lipsync-repeat-motion-2026-11-13.md)
  found 19 of 40 short edits with the combined check versus 16 of 40 for
  the original experimental arm. Both flagged one of 90 ordinary controls.
  The panel was reused after the signed-motion trial. No release change.
- The [voice resonance trial](audio-formant-local-transfer-2026-11-13.md)
  found zero of 12 competing-speech edits at the fixed points versus one
  for the current check. The new voice cues stay experimental.
- The [foreground isolation diagnostic](audio-source-isolation-diagnostic-2026-11-13.md)
  tested 12 synthetic voice edits and 40 ordinary deliveries under competing
  speech. The current check located one edit, blind separation located none,
  and the exact benchmark foreground located five using the same two-second
  windows. The exact foreground added no ordinary flags. Blind separation
  stays experimental and supplies no confidence.
- The [upper-frequency audio check](audio-bandwidth-return-2026-11-13.md)
  located 34 of 64 temporary low-pass changes from a single upload and
  reported one range on 128 ordinary controls. Twelve inputs per cutoff
  lacked sufficient comparable sound. The opt-in displays physical audio
  evidence and gives it zero AI-confidence weight.
- The [changed-word supervision test](writing-word-edit-transfer-2026-11-13.md)
  added no useful report locations to the current eight of 40 and flagged
  one more ordinary group in each of three held cohorts. No release change.
- The [exact location replication](independent-testing.md#portable-evaluation-contract)
  independently compares source-coordinate unions from two runs while
  keeping both sets of positions and image masks local.
- The [JPEG90 comparison](image-regional-jpeg90-carrier-2026-11-13.md)
  kept all 11 useful JPEG75 locations after a second save. Direct quality-90
  saves held the useful count at 11 while raising ordinary highlights from
  13 to 30 of 154. The direct workflow failed and remains experimental.
- The [Word GAO comparison](writing-gao-archive-comparison-2026-11-13.md)
  shows matched body paragraphs and native character ranges in the browser,
  PDF and JSON. It linked 23 of 24 unchanged excerpts and localized 45 of 48
  edited excerpts in the full archive. A cross-paragraph change abstained.
- The [measured paired-video interval](video-reference-measured-interval-2026-11-13.md)
  retains the initial window and shows a second interval of strong picture
  change on supported H.264 pairs. It improved exact intervals from zero
  to 15 of 16 conventional changes in the separate confirmation. The neural
  comparison gained one timing-qualified location and missed its frozen gate.
- The [12-second clean audio comparison](audio-clean-reference-duration-2026-11-13.md)
  localized nine of 18 synthetic and five of nine ordinary replacements per
  background level. The first pass flagged four equalization controls per
  level. A separate broad equalization check removed those flags and kept
  the localized replacements. The paired evidence has zero AI weight.
- The [replication comparison](independent-testing.md#portable-evaluation-contract)
  lets reviewers submit two runs of the same frozen method and share aggregate
  differences in scores, flags, coverage and region overlap.
- The [lower-depth resampling check](image-quantized-phase-transfer-2026-11-13.md)
  adds exact native PNG evidence when local value steps retain a camera-style
  pixel pattern. It found nine of 13 quantized translations on another camera
  panel, with no new resampling flag on 104 comparison deliveries. The
  evidence shows the original crop and carries zero AI confidence.
- The [earlier-copy chroma trial](video-reference-chroma-transfer-2026-11-13.md)
  found one additional useful redraw location and zero additional locations
  among timing-qualified redraws. It stays offline under the frozen scope.
- The [voice context trial](audio-local-context-transfer-2026-11-13.md)
  found zero of 12 short synthetic edits across three generator families
  under competing speech, with five genuine groups flagged. The existing
  check found one of 12 and flagged three genuine groups. The candidate
  stays offline.
- The [writing temperature trial](writing-temperature-transfer-2026-11-13.md)
  found zero of 48 fresh paired edits from Gemini and GPT-5.4 Nano, with
  one flag on 52 genuine excerpts. The existing context proxy localized
  16 of 48 and flagged 20 genuine files. The new cue stays offline.
- The [exact GAO excerpt comparison](writing-gao-archive-comparison-2026-11-13.md)
  newly attributed 23 of 24 untouched short excerpts to the correct
  published transcript. It preserved 46 of 48 strict edited locations
  and attributed none of 96 edge, quote, ordering or mixed-source controls.
- The [independent input verifier](independent-testing.md#portable-evaluation-contract)
  reads each planned input and parent file to verify the declared SHA-256.
  It emits a digest and counts that reviewers can share without private paths.
- The [four-second VP9 face-change check](video-brief-face-120-carrier-2026-11-13.md)
  recovered 15 of 16 six-frame swaps across all eight source groups, with
  zero additional flags on 40 ordinary deliveries. It displays the source
  interval and original frames and crops on the optional local path.
  AVC gained four exact locations across two groups and remains offline
  for this specific check. The VP9 test measures transfer on earlier sources.
- The [full Bayer prediction trial](image-bayer-prediction-transfer-2026-11-13.md)
  ranked 22 of 24 known edits in useful locations. Rotations and ordinary
  quantized processing scored above all the generated replacements, and
  none cleared the frozen point. This candidate remains experimental.
- The [two-pitch audio trial](audio-overlap-pitch-transfer-2026-11-13.md)
  reproduced 180 native measurements. It found no edits at the frozen
  point on 126 held deliveries. Incorrect peaks often followed competing
  speech rather than the changed voice. The candidate remains experimental.
- The [short writing typography trial](writing-typography-transfer-2026-11-13.md)
  added five useful locations on 23 previously studied rewrites with no
  extra genuine flags. It added none on the source-held diplomatic panel
  and flagged one additional genuine work. The simpler normalized score
  found three additional locations there, all from the same generator.
  Both checks remain experimental.
- The [contributor location counter](independent-testing.md#count-exact-locations-on-your-own-files)
  checks the case, input hash and extent before counting unions and overlap
  for image, video, audio and writing evaluations. Images support irregular
  binary masks as well as rectangles. Contributors can share aggregate
  results while keeping private files, masks and positions locally.
- The [native PNG local rotation check](image-native-rotation-location-2026-11-13.md)
  found 35 of 39 authored small local rotations across 13 cameras.
  None of 130 comparison deliveries produced a rotation observation.
  The optional local check shows the exact 16-bit source crop and
  carries no AI confidence.
- The [mouth temporal color test](video-mouth-temporal-chroma-2026-11-13.md)
  found no useful locations among 17 known redraws. The flicker arm
  flagged eight of 75 ordinary videos. Both arms remain experimental.
- The [VP9/WebM transfer](video-scene-return-2026-11-13.md#vp9-webm-transfer)
  extends the single-upload short picture-change check to the validated
  720p carrier. It found 15 of 16 authored changes with no flags on 58
  ordinary controls. The report preserves the actual millisecond source
  timestamps and exact source pictures.
- The [clean earlier-audio comparison](audio-clean-reference-transfer-2026-11-13.md)
  localized four of six synthetic replacements and two of three ordinary
  replacements at each of three competing-speech levels, with no false
  flags on 15 matched controls per level. The optional comparison requires
  a suitable clean copy and supplies no AI confidence.
- The [GAO highlights comparison](writing-gao-archive-comparison-2026-11-13.md)
  found the correct archived report for 73 of 120 edited files and 18 of
  20 untouched files. It found 23 precise differences in edited copies
  and none in untouched copies. Forty-nine files lacked a qualified match.
  It reports source wording differences with zero AI confidence.
- The [re-encoded trim check](video-trimmed-reencode-transfer-2026-11-13.md)
  adds local video comparison when two clips have matching motion and the
  analyzed clip is shorter. It localized 27 of 33 authored edits and flagged
  none of 44 controls. All admitted offsets matched and 168 exact source
  frames replayed. The check reports both timelines and crops.
- The [rotated interpolation trial](image-native-rotated-stencil-transfer-2026-11-13.md)
  explained all seven ordinary rotation flags while preserving all 16
  useful generated-edit locations. Three quantized translations still
  flagged, and the shipped phase-shift check supplies no candidate region
  for these rotations. The method remains experimental.
- The [paired noisy voice trial](audio-paired-speech-continuity-transfer-2026-11-13.md)
  found two of six synthetic replacements under louder competing speech.
  Both new methods flagged too many matched ordinary controls, including
  untouched speech. The methods remain experimental.
- The [single-upload writing interval trial](writing-interval-transfer-2026-11-13.md)
  added zero strict locations across 120 held edited reports. Its two-stream
  rule flagged zero of 20 untouched reports but located only one edit that
  the primary comparison already found. The app keeps its existing check.
- The [short voice-edit comparison](audio-phoneme-short-edit-score-2026-11-13.md)
  located nine of 16 edited deliveries with two-second windows, including
  none of four with a competing voice at −5 dB. The tested background
  reduction found the same nine and added one ordinary flag. The new
  processing stays outside the app's confidence calculation.
- The [forum revision transfer](writing-forum-revision-transfer-2026-11-13.md)
  tested the two-check writing comparison against earlier copies from eight
  dated threads. The interior-context expansion raised useful locations
  from six to nine of 14 generated rewrites. No eligible genuine revision
  flagged, and two genuine revisions fell outside the comparison scope.
- The [single-upload mouth color test](video-mouth-photometric-transfer-2026-11-13.md)
  located 2/17 redraws and flagged 29/75 ordinary files. Requiring
  local texture support located 1/17 and flagged zero ordinary files.
  All 92 held clips were measured, and 111 flagged native crops replayed.
  Neither arm advanced.
- The [public picture-change replay](video-scene-return-public-replay-2026-11-13.md)
  reconstructs the existing single-upload test from eight pinned
  publisher files, including all carrier hashes, frame times and
  supporting crops.
- The [picture-change evidence view](video-scene-return-2026-11-13.md)
  marks the measured texture tiles on a copy of the during frame in
  the browser and Standard PDF. The source frames and exact tile
  indices remain available in the same evidence set.
- The [shifted image grid trial](image-native-generated-grid-transfer-2026-11-13.md)
  recovered 6/8 sharp and 5/8 feathered generated regions where the
  existing grid recovered none. Ordinary flags increased from 1/64
  to 13/64, mostly on rotation and quantized interpolation. The
  added grid and the bit-depth veto remain experimental.
- The [local picture area check](video-paired-local-area-2026-11-13.md)
  handles supported matched short H.264 clips. On eight source groups it
  raised useful large-edit regions from 3/8 to 8/8, kept the 16 small and
  texture edit locations, and flagged zero of 16 ordinary delivery controls.
  One of the controls lacked photometric support. Public media reconstruction
  replayed all 48 paired results.
- The [expanded native PNG check](image-native-cfa-dimensions-transfer-2026-11-13.md)
  adds three supported landscape and square sizes. Those three sizes
  localized 48 of 60 planted interpolation edits and highlighted none of
  150 controls. The tested portrait size remains outside scope.
- [Reordered Word tables](document-word-table-alignment-2026-11-13.md)
  now compare uniquely matched columns and rows. The test recovered
  31 changed cells from genuine publisher revisions and flagged none of
  12 controls. Each side retains its own native coordinates.
- The [matched short voice edit panel](audio-phoneme-short-edit-intake-2026-11-13.md)
  prepared 108 waveform deliveries with exact same-word replacements and
  ordinary controls under annotated competing speech. Native replay and the
  subsequent detector trial matched all delivery commitments.
- [Matching Word tables](document-word-table-comparison-2026-11-13.md) now
  show wording changes in earlier-draft comparison with native cell and
  paragraph coordinates. The browser, PDF, JSON and source replay found
  all 15 edited cases and zero of six formatting controls.
- The [paired video area study](video-reference-extent-2026-11-13.md)
  expanded the key crop from 8.8% to 91.2% of the changed middle picture
  pixels with the exact time interval. Both methods localized 20/20 prior
  neural edits on a different carrier bank. The larger region found four
  of six large conventional edits on new source groups and missed its
  frozen five-of-six requirement. It stays offline.
- The [interpolation-control transfer](image-native-stencil-interpolation-transfer-2026-11-13.md)
  retained ten useful generated-edit locations across two delivery types on
  13 new camera models, while reducing ordinary highlights from 30 to 16
  across 104 controls. Rotation and quantization still produced false
  alarms. The method stays experimental. Native and scalar replays matched
  all 130 outcomes.
- The [public-source noisy-speech replay](audio-odss-public-waveform-replay-2026-11-13.md)
  rebuilt all 176 earlier stereo deliveries and 432 comparison views from
  bounded publisher downloads. It verifies input preparation and acoustic
  measurements separately from classifier accuracy.
- The [generated-image replacement pilot](image-native-generated-stencil-transfer-2026-11-13.md)
  compared the shipped interpolation shift check with two phase-loss checks.
  The shift check found 0/12 generated replacements. Phase loss found 12/12
  and flagged 11/30 ordinary edits. Requiring supporting texture retained
  10/12 useful locations and reduced ordinary flags to 6/30. Both phase-loss
  arms remain offline pending transfer beyond the tested generator and profile.
- The [earlier-copy mouth comparison](video-paired-native-mouth-transfer-2026-11-13.md)
  found 15/17 time windows and 11/17 useful time-and-pixel locations with
  luminance alone. Texture corroboration found 16/17 time windows and 10/17
  combined locations. Neither arm passed the frozen rule because the tile
  was too small on the independently timing-qualified second-family edit.
  The comparison remains offline.
- The [stereo recording-path study](audio-stereo-path-return-transfer-2026-11-13.md)
  located no synthetic insert under the fixed rule and flagged four gradual
  movement controls. All 203 input and decision replays matched. The check
  remains offline.
- The [USWDS writing study](writing-uswds-revision-transfer-2026-11-13.md)
  tested real editorial revisions and two generated rewrite families. Using
  the paired reading alongside the incumbent localized 5/9 rewrites versus
  4/9 with the incumbent and flagged 1/6 genuine revisions in both cases.
  The one added location came from a single source and generator. The paired
  reading remains offline.
- [Word tracked text changes](document-word-track-changes-2026-11-13.md)
  displays the original inserted and deleted wording with exact XML locations.
  The browser, Standard PDF, JSON and signed source replay agree. These
  recorded changes carry no AI confidence weight.
- The [full writing worker evaluation](writing-revision-worker-transfer-2026-11-13.md#complete-held-evaluation)
  completed all 140 held deliveries. It precisely localized four of 105
  changed versions, selected all 20 genuine reports for inspection and
  returned an inconclusive assessment on every delivery. The separate
  [sustained-response trial](writing-sustained-response-2026-11-13.md)
  added one precise location with five genuine flags. Its agreement arm
  added none. Both remain offline.
- The [camera-native interpolation trial](image-native-cfa-closure-2026-11-13.md)
  located eight of eight sharp, one-pixel-offset copies with zero flags on
  40 held ordinary views. Classical fills were missed. The trial measures
  local interpolation continuity in narrowly prepared 16-bit images.
- A [ten-camera PNG transfer](image-native-cfa-png-transfer-2026-11-13.md)
  located 54 of 60 odd-phase 128-pixel pastes and 24 of 60 at 64 pixels.
  It highlighted none of 100 assessed ordinary controls and declined
  30 incompatible inputs. The local development app now shows the supported
  interpolation observation and full-depth source crop. Chrome, the PDF,
  JSON, and signed source replay agreed across four held deliveries.
- The [flanking-audio background trial](audio-flank-background-transfer-2026-11-13.md)
  tested four new source readers in the Aachen room. All three approaches
  flagged two of 16 noisy synthetic deliveries and zero of 16 noisy genuine
  deliveries. The extra cleanup never proposed a new synthetic flag in
  that room and stays offline.

- The standalone video picture-change check now states whether the source
  frames were checked, whether it found a range and why the check was
  unavailable. Chrome, the PDF and JSON matched on a positive, a supported
  no-match, an unsupported input and a Quick-profile upload. Two paired H.264
  encodes retained all 14 exact ranges and flagged zero of 40 matched ordinary
  controls. A further 18 ordinary H.264 controls yielded zero flags.
  [Local test](video-scene-return-2026-11-13.md)
- The [neighbor-predicted mouth study](video-mouth-blind-prediction-2026-11-13.md)
  found one of 17 authored redraws with the simpler cue and zero with the
  motion-adjusted cue. The arms flagged 17 and eight of 75 ordinary files.
  Neither is enabled.
- The [single-upload residual template](image-residual-template-transfer-2026-11-13.md)
  found zero useful locations on 50 known removals. Adding it raised ordinary
  highlights from five to ten of 150 at the frozen points. It remains offline.
- Writing details in the browser and PDF now explain whether each scored
  input gave an AI-like flag, whether readings disagreed and where to review
  the source. The first three new worker batches localized two of 61 changed
  deliveries under the strict annotation rule and selected all 12 genuine
  controls for inspection. All 84 document assessments were inconclusive.
  [Worker comparison](writing-revision-worker-transfer-2026-11-13.md#three-completed-held-batches)
- [Selective voice cleanup](audio-phonemedf-matched-transfer-2026-11-13.md)
  flagged 36 of 48 noisy synthetic recordings versus 21 with the original
  input, and 10 of 24 noisy genuine recordings versus seven. At −5 dB it
  flagged 15 of 24 synthetic recordings versus eight, and four of 12 genuine
  recordings versus three. It remains outside the confidence score.

- The [standalone video picture-change check](video-scene-return-2026-11-13.md)
  filled the face-routing gap on eight held layout-only cuts. It also located
  six of eight speaker-plus-layout cuts and flagged none of 40 controls.
  The local option uses exact source frames and carries no AI confidence weight.
- The [structure-supported earlier-copy JPEG check](image-reference-structure-jpeg-2026-11-13.md)
  raised useful sharp and feathered object-removal locations from 3/25 to
  20/25 each, with the same 42/150 ordinary highlights. The optional local
  comparison preserves both source crops and carries no AI confidence weight.
- The [Full worker writing calibration controls](writing-revision-worker-transfer-2026-11-13.md)
  confirmed inspection highlights on all 16 untouched reports, while every
  document-level conclusion was inconclusive. The 95 saved packets and exact
  source ranges support a separate held comparison of passage selection. The
  first held batch highlighted all four genuine reports, covered at least half
  of one edited passage in 19/22 changed versions, and found zero strict
  locations. All three fitted selections found zero strict locations and
  remain experimental.
- The [matched utterance speech study](audio-odss-matched-room-2026-11-13.md)
  improved synthetic flags from 17/32 to 28/32 with frontal background
  cancellation, while increasing ordinary flags from 2/16 to 9/16.
  The same-script, two-generator comparison remains experimental.

- The [cross-format draft check](document-mixed-format-comparison-2026-11-13.md#multi-page-validation)
  now covers up to 12 supported PDF pages and can show a single revision across
  several source pages. The three-page replay localized all 40 PDF edits and
  returned zero changes on all 40 matched unchanged PDF deliveries. Chrome
  verified crops on two different pages and the PDF and JSON downloads.
- The [earlier-copy JPEG tone comparison](image-reference-monotone-jpeg-2026-11-13.md)
  increased sharp and feathered removal localization from 1/8 to 4/8 each
  on the separate confirmation bank. It added six ordinary-file highlights,
  including broad regions after grading and resizing, and stays experimental.
- The [ordinary webcam-cut comparison](video-scene-only-cut-transfer-2026-11-13.md)
  found all eight surrounding-layout cuts with the off-face check at supplied
  intervals. On unmarked uploads the Full face pipeline surfaced none of those
  eight cuts because none produced a coarse face interval. It surfaced 14
  exact cut ranges across 16 ordinary speaker changes, with 42 source frames
  independently decoded. A standalone check needs new held controls.
- The [matched new-room speech comparison](audio-wpe-fresh-room-transfer-2026-11-13.md)
  increased useful deliveries from 3/16 to 4/16 without new false reader or
  background groups at its most conservative point. Both arms found the same
  two of four unique edits. The additional location was another rendering of
  the same edit, and the lower points flagged more genuine readers. The
  candidate stays experimental.
- The [H.264 mouth prediction comparison](video-mouth-codec-prediction-2026-11-13.md)
  located five of 17 neural redraws but flagged 37 of 75 ordinary files.
  Motion-vector dispersion located zero. Both remain offline.
- The [outside writing benchmark](writing-revision-trajectory-2026-11-13.md)
  found one additional half-covered span on 105 text-changed deliveries,
  no additional strict location, and no improvement in ordinary highlights.
  Fifteen further publisher-labeled versions exactly matched their originals.
- The [released FLAME JPEG comparison](image-flame-jpeg-transfer-2026-11-13.md)
  found no useful region on 26 object removals. Its published point with local
  region filtering highlighted 38 of 48 held ordinary files. Its separately
  fitted ordinary point returned no held regions. It stays offline.
- The [blind dereverberation comparison](audio-wpe-room-transfer-2026-11-13.md)
  raised the source-pooled full-room and noisy union from four to five of
  seven edits at the prior mono point and from two to six of eight false
  genuine groups. Its independently fitted point added zero edits. It stays
  experimental pending a prospective matched-room retest.
- The [entropy-conditioned writing trial](writing-entropy-conditioned-2026-11-13.md)
  measured all 39 known edit locations and added zero useful locations. It
  added flags on three eLife control works and stays offline.
- The [JPEG tile fallback](image-regional-tile-rescue-2026-11-13.md) added no
  useful locations across 46 known edits and raised ordinary highlights
  from 13 to 22 of 154 files. Both source banks completed and all 200 JPEG
  and mask bindings replayed. The fallback stays experimental.
- The [paired room and background-voice stress test](audio-room-stress-transfer-2026-11-13.md)
  measured five of seven dry or early-room locations versus three of seven
  with the full room response. Neither tested separator met its held advancement
  gate across background levels.
- The [mouth luminance detail trial](video-mouth-luma-condition-2026-11-13.md)
  measured all 23 native neural edits and all 17 H.264 redraws. Both arms
  localized zero edits and flagged zero ordinary files at the frozen points.
  Frame, crop and carrier-clock commitments replayed. The checks stay offline.
- The [cross-format draft comparison](document-mixed-format-comparison-2026-11-13.md)
  adds exact wording evidence for TXT, DOCX and supported PDFs. Public source
  replay checked 60 edited and 60 unchanged comparisons in all six directions.
- The [JPEG regional check](image-regional-jpeg-transfer-2026-11-13.md)
  usefully localized 10 of 40 known edits versus zero for the aligned luma
  check. It highlighted five of 120 ordinary deliveries versus three.
  Development opt-in applies only to the evaluated JPEG profile.
- The [mouth chroma-grid trial](video-mouth-chroma-grid-2026-11-13.md)
  localized zero of 23 held neural mouth edits and zero of 17 after H.264
  re-encoding. The candidate remains offline.
- The [public AMI replay](audio-gain-public-replay-2026-11-13.md) independently
  reconstructs the fixed-gain comparison from eight publisher WAVs. It
  localized all 96 additional authored repeats across two further meetings
  and flagged zero of 144 matched controls.
- The [earlier-copy video trim check](video-trimmed-reference-transfer-2026-11-13.md)
  measured all 18 known changed ranges after lossless trims and declined all
  18 wrong-source, lossy, or changed-clock controls.
- The [earlier-copy JPEG shift check](image-reference-shifted-jpeg-transfer-2026-11-13.md)
  localized all 24 known edits while separately showing the uncovered margins.
- [Directional separation for competing voices](audio-spatial-mask-voice-transfer-2026-11-13.md)
  added two useful locations and two ordinary-group flags while retaining the
  current mono locations. Both gains came from the same lateral layout. The
  [two-room transfer](audio-spatial-mask-aachen-transfer-2026-11-13.md) covered
  six synthesis families and added zero of seven edit locations. The check
  stays in evaluation.
- [Sentence omission](writing-omission-attribution-2026-11-13.md) added one
  eLife location and no ordinary flags there. On F1000Research it added no
  useful locations and one ordinary-group flag. It remains experimental.
- [Cross-fitted JPEG color direction](image-color-line-transfer-2026-11-13.md)
  missed all 40 known edits across two panels and remains offline.
- The [single-upload fixed-gain stereo check](stereo-repetition-2026-11-13.md#gain-and-polarity)
  found 96 of 96 level- or polarity-adjusted repeats missed by the exact
  channel and mono searches. It added zero findings on 144 matched controls
  across the same two AMI meetings. It is available on supported local audio
  and video soundtracks and carries no AI confidence weight.
- [Multi-frame mouth tracking](video-mouth-track-cycle-2026-11-13.md) found one
  of ten qualified LivePortrait redraws, missed the qualified MuseTalk redraw,
  and flagged eight of 123 ordinary files in its stronger localization arm.
  The two candidate checks remain offline.
- [Directional JPEG texture](image-directional-structure-transfer-2026-11-13.md)
  located one of twelve feathered edits, flagged seven of 72 ordinary files,
  and found no useful region on sixteen second-bank edits. It remains offline.
- [Longer writing on eLife](writing-elife-long-position-transfer-2026-11-13.md)
  found no useful location on 21 generated sentence substitutions and flagged
  two of twelve ordinary works at the frozen longer-passage point. The local
  opt-in scope remains unchanged.
- The [single-upload stereo repeat check](stereo-repetition-2026-11-13.md)
  located all 32 exact one-channel copies that the downmix missed, with no
  matches on 16 unchanged recordings. It shows and replays both intervals
  for supported audio and video soundtracks without changing AI confidence.
- The [fresh synchronized video source study](lombard-fresh-musetalk-2026-11-13.md)
  rebuilt 30 native deliveries across six speakers. One neural redraw
  preserved the measured speech event and passed every timing control. The
  study qualifies a second-generator detector fixture.
- The [transparent PNG study](image-png-hidden-rgb-2026-11-13.md) verified
  hidden color regions without exposing the pixels. It localized five of
  twelve small targets and remained offline.
- The [longer writing comparison](writing-f1000-position-transfer-2026-11-13.md)
  raised the combined exact locations from two to four among 18 matched
  rewrites and added one flagged genuine source work among nine.

- The [three-sentence writing check](writing-position-app-2026-11-12.md) now
  runs on the local Apple GPU. It reproduced all 68 CPU decisions and passed
  the prepaid worker, browser, PDF, JSON, and signed source replay.
- The [masked image-context study](image-masked-context-transfer-2026-11-13.md)
  retained three useful feathered locations versus two for the visible-only
  head and reduced flagged ordinary groups from five to one. All three
  locations came from one generator family in the reused confirmation panel.
  A [new-source transfer](image-masked-context-new-editors-2026-11-13.md)
  added no useful locations on six neural object removals or the one
  independent photograph with two diffusion deliveries. The study stays
  outside the detector.
- The [longer writing source study](writing-f1000-revision-spans-2026-11-13.md)
  qualified nine previously unused genuine article revision pairs and bound 274
  passage locations and classifier inputs to official XML. Two retrievals
  reproduced the source receipts. No classifier was scored on this panel.
- The [two-output stereo study](audio-unassigned-stereo-voice-2026-11-13.md)
  found four of eight off-center edits when both outputs had to support a
  finding, with five newly flagged genuine groups. At the same retrospective
  budget of one false group it found two versus zero for the downmix. In
  the two-competitor layout this rule lost two current locations. It stays
  outside serving.
- The [mouth color study](video-mouth-chroma-coupling-2026-11-13.md) found zero
  of ten source-event-aligned neural redraws and no ordinary flags for either
  color variant. It remains offline.
- The [four-media replay](independent-testing.md#replay-difficult-edits-across-four-media)
  reproduces the public image, video, audio and writing measurements and
  creates a compact issue receipt. It can run without app dependencies or
  private media.

## Progress on 12 November

- The [synchronized mouth-redraw test](lombard-neural-mouth-redraw-2026-11-12.md) verified retained local timing on ten of eleven neural clips. The existing single-upload timing check had insufficient windows on all 44 held deliveries. Earlier-copy comparison found all eleven changed picture ranges. The tested single-upload candidates produced too many genuine flags for adoption.
- The [blind stereo comparison](audio-blind-spatial-transfer-2026-11-12.md) located two of six known short edits on the reused held panel, versus zero on the original stereo mix. Both methods flagged two ordinary groups. In the [third room](audio-blind-huddersfield-transfer-2026-11-12.md) the routing rule discarded the better component for all eight off-center targets, so it remains outside serving.
- The [longer article transfer](writing-long-position-transfer-2026-11-12.md) gained two locations at the short-passage point with one extra genuine work flagged. The longer-document operating point gave no added locations. The [JPEG contour trial](image-jpeg-contour-gap-transfer-2026-11-12.md) gave no useful localization before applying its points. Both remain in evaluation.

- The [local audio timing check](audio-reference-local-shift-2026-11-12.md) compares a short recording with its earlier copy. It recovered 34 of 48 known moves and added zero ranges on 216 held controls. The local app and downloadable evidence use the same two time ranges and waveforms.
- The [three-sentence writing check](writing-position-app-2026-11-12.md) adds source-linked evidence in the browser, PDF and JSON on eligible CPU runs. It added six correct sentence locations and no genuine groups across three historical publisher evaluations.
- The [JPEG texture ranking](image-jpeg-texture-rank-transfer-2026-11-12.md) remains experimental after useful localization on only one fresh photo group. The [Lombard source qualification](lombard-original-mouth-event-2026-11-12.md) provides genuine controls for a separate synchronized mouth-redraw test.

- The [positive-only audio combination](audio-selective-fusion-transfer-2026-11-12.md)
  added ten AI flags and one new genuine false-flag group in the archived
  confirmation. The same fixed rule added zero flags on 32 unused published
  RADAR recordings. The serving calibration is unchanged.
- The [learned JPEG residual check](image-residual-encoder-transfer-2026-11-12.md)
  found three useful regions across two encodes of the same held photograph.
  It frequently flagged ordinary deliveries and missed every held LaMa edit.
- The [mouth-region persistence check](video-mouth-sector-persistence-2026-11-12.md)
  improved one CREMA-D comparison and missed all nine neural redraws that
  preserved the measured mouth timing. Listener and phoneme synchrony and
  broader redraw coverage remain open.
- The [frozen writing check on Qwen edits](writing-position-qwen-confirmation-2026-11-12.md)
  found two additional exact rewrite locations across two journal titles
  and flagged none of the 14 original and genuine revision groups.
- The [eight-second H.264 and VP9 check](video-face-short-carrier-transfer-2026-11-12.md)
  localized 27 of 30 and 30 of 30 exact face changes, respectively. It added
  zero ordinary flags in either format. The local opt-in passed paid-path,
  browser and source-evidence replay. The [repeated speech and mouth study](video-lipsync-repeat-audio-2026-11-12.md)
  localized only 3 of 32 conventional soundtrack edits per format and stays
  offline. Neural mouth redraws still need separate evaluation.
- The [stereo earlier-copy comparison](audio-reference-comparison-2026-11-11.md#changes-in-the-recorded-stereo-channels)
  exposed 36 changes that canceled in mono and verified both original channels
  against the downloaded evidence. The [single-file modulation check](audio-modulation-partial-transfer-2026-11-12.md)
  missed all 22 edits under loud competing speech and stays offline.
- The [image line-pattern check](image-line-fingerprint-transfer-2026-11-12.md)
  missed 144 known changes and flagged 13 of 576 ordinary deliveries. The
  [writing position check](writing-two-context-transfer-2026-11-12.md) added
  three exact locations with zero new ordinary work flags, all from one
  generator. The [historical revision confirmation](writing-position-confirmation-2026-11-12.md)
  added one further exact location with zero new ordinary flags and missed
  all four rewrites from the second generator in that panel.
- The [picture-cut context check](video-spatial-cut-context-transfer-2026-11-12.md)
  explained 13 of 15 routed ordinary cutaways at their exact authored
  boundaries. It left the separately reported brief face changes intact and
  carries no overall confidence weight.
- The [cross-camera HDR study](image-gainmap-cross-camera-2026-11-12.md)
  expanded gain-map extraction from three to eight of ten originals. The
  fixed union localized 12 of 16 controlled edits and flagged 17 of 64
  ordinary views, including two originals. The cue remains experimental.
- The [transient-reduction audio study](audio-transient-reduction-transfer-2026-11-12.md)
  added no locations under rain or train noise. Only 16 of 65 eligible
  edited windows under train noise passed the speech-activity check.
- The [enhanced-audio blend study](audio-enhancement-blend-transfer-2026-11-12.md)
  found two additional train-masked edits with full enhancement and flagged
  two genuine recordings. Requiring support after restoring part of the
  original signal removed both gains and both false flags.
- A [separate Node.js replay](writing-paired-center-transfer-2026-11-12.md#replay)
  reproduced the writing study and the exact positions in four app exports.
- The [fresh PLOS Biology writing study](writing-semantic-continuity-transfer-2026-11-12.md)
  found two of 17 exact edits with the existing passage check. The simpler
  semantic cue added one location and flagged five of ten ordinary work
  groups. Both semantic candidates stay outside the app.
- The [six-picture face-change study](video-face-unrouted-short-transfer-2026-11-12.md)
  found all 16 exact ranges missed by Full, across eight HEVC source groups.
  Both the bounded shortlist and the dense search added zero ordinary ranges
  across 40 controls. The local opt-in passed simulated paid analysis,
  browser and export checks, and native replay with the existing checks enabled.
- The [JPEG coefficient-transition study](image-jpeg-signed-transitions-2026-11-12.md)
  added three feathered JPEG90 locations and seven ordinary source-group
  flags. A fresh 120-view BG-REAL comparison found no useful locations or
  ordinary flags. The two source replays matched and the cue stays offline.
- The [frequency-view speech study](audio-subband-partial-transfer-2026-11-12.md)
  added three gain-only locations and zero under louder rain or train sound.
  The two separately frozen points added no genuine file flags. It did not
  resolve the loud-noise gap and stays outside the app.
- The [adjacent speech-frame study](audio-adjacent-frame-transfer-2026-11-12.md)
  found one additional rain-masked edit, flagged two genuine readers under
  rain, and found no train-masked edit. The cue stays outside the app.
- The [exact small PNG copy check](image-exact-small-copy-2026-11-12.md)
  found 44/48 known copies and added zero flags on 66 held controls and
  eight interface captures. New local opt-in reports show both native crops
  and the measured surrounding differences. The cue carries no authorship
  confidence weight.
- The [paired writing response](writing-instruct-base-passage-transfer-2026-11-12.md)
  added zero useful locations on 24 known rewrites and flagged five of 15
  genuine article groups at the frozen operating point. It stays outside
  the app.
- The [paired draft check beside a paragraph break](writing-paired-center-transfer-2026-11-12.md)
  located four of 12 known edits and flagged zero of eight ordinary revisions.
  Local paid verification confirmed the exact source spans in the browser,
  PDF and JSON. The opt-in needs both versions of a short English TXT draft.
- The [narrower speech evidence study](audio-focused-window-transfer-2026-11-12.md)
  compared 98 exact deliveries. Replacing a broad displayed range only when
  it contains a corroborated narrow range preserved every useful sparse
  location, found 3/18 new rain edits and 0/18 train edits, and marked nearly
  identical genuine time. The optional Full review retains the original
  readings and verifies the selected playback and waveform against the upload.
- The [PNG Exif preview comparison](image-png-exif-preview-2026-11-12.md)
  can show a difference between a supported PNG and the thumbnail embedded
  in the same file. It located 11/12 feathered edits in one test group and
  0/6 smaller edits in the second. The optional check shows the original
  and preview crops and their difference without adding authorship weight.
  The separate [texture and JPEG95 trials](image-local-edit-candidates-2026-11-12.md)
  stay outside the app.
- The [selectable-text check for mixed PDFs](writing-mixed-pdf-page-transfer-2026-11-12.md)
  found 11 of 44 known edited passages in files previously skipped by the
  second check, with agreement on 13 of 56 genuine or ordinarily edited
  controls. The optional local check requires an intact native page and
  confirms its source ranges in the app, PDF, JSON and signed export. It
  leaves OCR text and cross-page contexts outside its scope.
- The [overlapping speech confirmation](audio-dense-evidence-confirmation-2026-11-12.md)
  compared 60 matched deliveries across six editing families. The union
  localized 15/18 gain-only and 12/18 traffic edits, up from 7/18 and 6/18.
  It added one genuine flag and 6.73 points of flagged genuine time, exceeding
  the four-point frozen limit. Dense alone lowered flagged genuine time but
  also lost some of the existing locations. The serving review stays in place.
- The [paired native face edges study](video-face-paired-edge-transfer-2026-11-12.md)
  refined two of three previously missed H.264 swap ranges to the exact native
  boundaries. It added no ranges on 79 ordinary deliveries. The fixed H.264
  advancement gate failed on the number of gains and the share of eligible
  edits assessed. A separate HEVC confirmation located three of eight swaps,
  up from zero, with no added ordinary ranges on 53 controls. It missed all
  four six-frame edits. The local HEVC opt-in reproduced all 61 serving-stage
  decisions. Prepaid browser, PDF and JSON checks displayed the exact interval
  and three original frames. Signed replay checked both edges and surrounding
  motion. The H.264 candidate remains experimental.
- The [environmental sound comparison](environmental-embedding-transfer-2026-11-12.md)
  held the voice fixed while changing background origin across 348 exact
  deliveries. Mixed-condition spectral fitting found 21/36 generated
  backgrounds under louder speech with 6/18 ordinary flags. The learned
  and combined checks did not meet the held conditions either. The
  public replay distinguishes the fixed results and matched-alarm ranking.
- The [photographic PNG supplement](image-png-layout-guard-2026-11-12.md)
  added 12 indications on pixel-identical PNG deliveries of previously
  tested generated JPEGs across three families. On eight new native
  generated PNGs, the main Full checks already indicated all eight and
  the supplement agreed on four. The screen-layout check kept all seven
  raw false indications out of the supplement across 24 authored browser
  deliveries. The local opt-in shows its inspected crops and reports
  a scope skip on detected screen layouts.
- The [short HTML reply extension](writing-html-reply-short-layouts-2026-11-12.md)
  preserved 14 useful locations and 13 existing genuine flags over 100
  matched plain-text and HTML-only emails. It assessed two more reply
  layouts and omitted all 12 unsupported controls. Local prepaid analysis
  and report replay preserved exact MIME positions.
- Voice evidence cards and Standard PDFs link directly to overlapping
  acoustic observations and sound-category windows. The return links use
  the same saved sample ranges. Browser and rendered PDF checks verified
  both directions, with no change to the recorded detector responses.
- The [speech-sound and mouth-shape check](grid-phone-viseme-timing-2026-11-12.md)
  located one of 96 local audio shifts and flagged three of 144 ordinary
  controls. It missed all four known picture retimes. The method remains
  experimental, and the portable and native source replays are available.
- The [local vocabulary comparison](writing-local-lexical-transfer-2026-11-12.md)
  used two independent cues at the sentence level. Combined, they recovered
  zero additional rewrites and flagged five additional genuine groups across
  the two held publishers. Neither cue entered the app.
- The [earlier-copy image comparison](image-reference-delivery-envelope-2026-11-12.md)
  added nine useful feathered edit locations across 14 controlled JPEG-to-PNG
  pairs. Two of 60 ordinary delivery controls highlighted after global
  contrast changes. The optional local browser, PDF and JSON views share
  the same crops and mask, with exact geometry replay across 184 held pairs.
- The [two-sentence draft comparison](writing-draft-grammar-2026-11-12.md)
  added three locations across two held panels, all from one generator,
  with no new genuine groups flagged. The local English TXT comparison
  preserves the exact ranges in both drafts and contributes zero confidence
  weight. The separate grammar cues remained experimental.
- The [scale-only video follow-up](video-face-scale-context-holdout-2026-11-12.md)
  explained all ten authored zoom intervals and preserved all ten genuine
  cutaways. The Full app selected none of the zooms, leaving no measured
  benefit for an additional report qualifier.
- The [staggered audio test](audio-noise-window-transfer-2026-11-12.md)
  located six of 36 noisy edits with dense native sampling versus four with
  sparse sampling. Adding staggered ranges and spectral noise reduction
  located the same six and added one genuine group flag. It remains outside
  the app. The held edits cover six generators and previously seen speaker
  groups with separately sourced exact inputs.
- The [short speech and wider context comparison](audio-short-coarse-transfer-2026-11-12.md)
  added one useful gain-only location and zero under strong noise. It
  preserved the six dense-grid noisy locations and their genuine flags.
  This retrospective duration comparison stays outside the app.
- The [public replay tool](independent-testing.md#save-a-four-modality-replay-receipt)
  now compares the exact inputs and outputs recorded on two systems. Python
  3.11 and 3.12 reproduced the four localization outputs locally.
- The [second speech-enhancement comparison](audio-nemotron-noise-transfer-2026-11-12.md)
  added flags on seven synthetic source groups across five generators and
  one genuine group. With the same decision points and separate recordings,
  it added flags on nine synthetic and one genuine RADAR files. Both trials
  preserved existing clean flags. The earlier restaurant comparison still
  has seven inherited genuine flags. The newer enhancer is eligible for an
  isolated evidence integration and has no customer confidence weight.
- The [720p30 face-change extension](video-face-two-returns-hd-2026-11-12.md)
  found five additional swap ranges, including three after tandem
  recompression, on eight separate retained source groups. It added no
  locations on 32 ordinary or cut files and 16 mixed single-edit controls.
  Full worker, signed source replay and browser tests reproduced both ranges.
- The [cross-format earlier-image comparison](image-reference-cross-format-local-edits-2026-11-12.md)
  located ten of 12 feathered AI patches. Five compression or tone controls
  also showed overlays. A second-panel stability guard removed those nuisance
  flags and missed all six held feathered FLUX edits. It stays experimental.
- The [matched substantial writing revision test](writing-clipped-rank-transfer-2026-11-12.md)
  found zero new locations from clipped token ranks and produced additional
  genuine flags. The alternate classifier added two rewrite locations and
  seven genuine flags. Both previously missed PDF passages remained missed.
- The [second brief face-change range](video-face-two-returns-2026-11-12.md)
  localized 16 of 16 authored swaps across eight two-edit videos, up from
  eight. It added no locations on 32 single-swap and mixed controls and
  no flags on 24 ordinary or genuine-cut deliveries. The prior method had
  one flag on a recompressed original. The narrowly scoped local path
  preserved both ranges and their source frames in the app and exports.
- The [public JPEG replay](image-jpeg-public-replay-2026-11-12.md)
  reproduced 27 useful locations on 48 known edits, up from 13 with the
  translation check alone, and produced zero flags on 48 ordinary
  deliveries. A clean checkout and public sources supply the inputs.
- The [PDF writing coverage table](writing-pdf-page-coverage-2026-11-12.md)
  records which pages have local responses, context-only text, missing
  mappings and unchecked image areas. Browser, PDF, JSON and signed source
  replay agree. The writing check missed both known rewritten passages in
  the mixed-page verification file.
- The [overlapping speech-window comparison](audio-partial-publisher-transfer-2026-11-12.md)
  found five of 12 partial edits with traffic, up from two, and added two
  genuine flags. It located none under café noise. It stays experimental.
  The [existing long speech review](audio-radar-transfer-2026-11-12.md#local-evidence)
  now also accepts equivalent native PCM WAV. A parity check reproduced
  171 window inputs from both containers, and the full app repeated the
  existing evidence on a matched positive and genuine control.
- The [between-block JPEG copy check](image-offgrid-copy-transfer-2026-11-12.md)
  added 19 correct locations across 64 edited test deliveries and produced
  zero flags on 124 assessed ordinary deliveries in a separate confirmation.
  The other 16 ordinary deliveries were outside the encoding profile. The
  development app displays the two native crops, coordinates and measured
  color differences for the supported JPEG profiles. The full local flow
  reproduced the same evidence in the PDF, JSON and signed package. The copy
  evidence carries zero weight in the AI confidence score.
- The [fixed-position video detail check](video-mouth-fixed-residual-2026-11-12.md)
  located four of 52 neural redraws, compared with two using the direct
  comparison. It missed all nine redraws with verified source-time mouth
  motion and flagged 18 of 56 ordinary GRID views and four of eight
  H.264 re-encodes. It stays experimental. Native readback verified 171
  frame crops and 24 aligned audio intervals.
- The [separately calibrated speech-enhancement comparison](audio-noise-views-transfer-2026-11-12.md)
  raised synthetic traffic-clip coverage from eight to all 16. The
  enhanced-only rule added one correct flag. The combined rule added flags
  on 11 synthetic and 12 genuine sources across the two noise scenes. Both
  stay experimental. The independent replay checked all 240 prepared PCM files.
- The [source-aligned writing comparison](writing-source-aligned-transfer-2026-11-12.md)
  located 14 of 14 controlled rewrites, up from two. Adding it raised flags
  on genuine works from one to three of ten. A simple edit-distance rule
  gave identical results and flagged substantial publisher revisions.
  The comparison stays experimental for AI authorship.
- The [single-video mouth-region check](video-mouth-local-hold-2026-11-12.md)
  localized 26 of 28 known held-mouth edits on its two-second H.264 profile.
  None of the 81 assessed controls produced an alert, and three cuts were
  unassessed. The app, PDF, JSON and signed source replay agree on the range
  and the three before, during and after picture and crop pairs.
- The [earlier-photo screenshot comparison](image-reference-screenshot-holdout-2026-11-12.md)
  localized 17 of 18 known edits versus zero with the prior app. All 18
  unchanged photo placements were recovered without a highlight. The
  [Word archive comparison](writing-archive-word-transfer-2026-11-12.md)
  recovered all 53 expected source matches and made zero claims on 43
  controls. Both comparisons report measured differences with their
  native source positions and carry zero weight in the AI estimate.
- The [independent speech diagnostic](audio-vocal-event-mixtures-2026-11-12.md#can-a-separate-speech-classifier-catch-those-misses)
  caught none of the five wrong-original deliveries with competing vocal
  events and stays experimental. The
  [paired-video soundtrack replay](video-paired-soundtrack-transfer-2026-11-12.md#replay)
  can now start from the pinned public recordings. The local replay
  regenerated and verified all 198 held deliveries.
- The [JPEG copy check](image-jpeg-copy-blocks-transfer-2026-11-12.md)
  located 15 of 18 small on-grid copies at JPEG75, up from three, and 17
  of 18 at JPEG90, up from two. It added no flags on ordinary controls
  and supplied no additional off-grid locations. Full local analysis and
  signed source replay verified both original crop locations.
- The [720p video extension](video-mouth-hd-hold-2026-11-12.md) localized
  12 of 14 known mouth-region holds and flagged none of 53 ordinary
  controls. It covers up to four seconds at the start of qualifying
  H.264 MP4. The original two-second scope remains available.
- The [RADAR speech comparison](audio-radar-transfer-2026-11-12.md)
  corroborated 10 of 16 synthetic longer recordings on the initial split
  and 18 of 20 on the separate confirmation split, with zero corroborations
  on 36 genuine recordings. Short-file AI flags fell from 11 to three
  with added traffic and to zero with added café sound. Source acquisition
  and native replay use bounded public archive downloads.
- The [writing revision-class comparison](writing-assisted-class-transfer-2026-11-12.md)
  found no new useful locations on 32 rewrites and added genuine flags
  that the incumbent did not produce. It stays experimental.

## Progress on 11 November

- The [residual color comparison](image-residual-color-transfer-2026-11-11.md)
  assessed 384 PNG and JPEG75 deliveries. It found zero useful edit regions.
  Even before applying the cutoff, the best two-band region reached useful
  overlap on two of 96 edits. An [offline reviewer](independent-testing.md#verify-an-earlier-copy-image-comparison)
  now checks both source files, native crop pixels and mask geometry for the
  local earlier-copy comparison export.
- The [waveform fallback transfer](audio-waveform-fallback-transfer-2026-11-11.md)
  increased repeat localization from 629/720 to 634/720 with no ordinary
  flags. All five gains came from two source groups. Both methods localized
  79/144 under the strongest traffic noise. Keep the fallback experimental.
- The [transcribed speech transfer](writing-speech-paragraph-transfer-2026-11-11.md)
  reduced human flags from 24/27 to 12/27 and retained all 12 useful rewrite
  locations. Four of six oral-history speakers still had a false highlight.
  This supports studying passage ordering while keeping authorship confidence
  outside the isolated-paragraph rule.
- The [video margin transfer](video-short-return-margin-transfer-2026-11-11.md)
  localized 46/60 swaps versus 28/60 for the incumbent, with zero added flags
  on 120 controls. It missed the advancement criterion for earlier edit
  placements. A retrospective second-range policy located all 36 shortest
  swaps and would add 32 distinct correct ranges after deduplication. Freeze
  and test that policy on different source groups before app integration.
- The [stream-presence transfer](audio-stream-presence-transfer-2026-11-11.md)
  separated lab and natural results. The stability cue found 37/52 controlled
  three-voice cases with zero flags on 26 two-voice controls. It found zero of
  18 natural three-voice windows from three AMI rooms. The release includes
  independent annotation selection and native sample-clock verification.
  The paired separator experiment supported 3/18 natural three-voice cases
  with 1/18 false flags. Its 19/52 controlled support also fell below the
  advancement requirement.
- The [local resampling cue](image-local-resampling-transfer-2026-11-11.md)
  localized 11/72 intentionally resized PNG patches with 3/216 ordinary PNG
  flags. It found zero corresponding JPEG75 regions and zero unmodified
  generated edits. The supported cue is specific to native resampling geometry.
- The [short face-return comparison](video-face-return-transfer-2026-11-11.md)
  found 10/12 swaps at 0.375 seconds versus 0/12 for the current appearance
  check, but added 42/66 ordinary false flags at its original frozen point.
  Requiring a positive rank margin after scoring preserved those 10 short hits
  and removed the false flags. A [second panel](video-short-return-replication-2026-11-11.md)
  reproduced 10/12 short localizations versus 0/12. It also added four false
  flags across 66 ordinary views from two source groups, above the frozen limit.
  The observed true margins exceeded the false margins. A higher margin needs
  its own source-separated evaluation before it enters the app.
- The [writing token-curvature test](writing-token-curvature-2026-11-11.md)
  scored 3/5 generated edits above both paired human controls versus 5/5 with
  the current context response. Both ranked two exact locations first, in
  different examples. The new score also rose on four of eight human edits.

- The [similar-voice audio transfer](audio-hard-overlap-transfer-2026-11-11.md)
  tested 504 new mixtures from 63 distinct VoxPopuli speakers. The strongest
  prior normalized rule reduced mistaken three-voice routes from 181 to 163 of
  378. It retained all 124 correct baseline routes, while 20 of 21 source
  triplets still had a routing error. The independent verifier can compare the
  exact PCM hashes, frame arithmetic and decision to the public aggregate.
- The [three-stream separation pilot](audio-separation-third-voice-pilot-2026-11-11.md)
  recovered 14/16 quieter and 12/16 fainter third voices. All 16 two-voice
  controls produced a substantial unmatched output. A separate voice-presence
  check is required before those streams can be interpreted independently.
- The [image edge-spread check](image-edge-spread-transfer-2026-11-11.md) added
  a feathered PNG location and two feathered JPEG75 locations beyond the
  wavelet check. The opposing-flank method flagged 32/144 PNG controls and
  63/144 JPEG75 controls. It failed the advancement rule in both deliveries.
- The [spatial video color-boundary check](video-chroma-boundary-transfer-2026-11-11.md)
  measured all 24 known swaps and 48 controls in its transfer panel. It flagged
  zero swaps and zero controls at the frozen point. Only five top windows
  overlapped a known edit by at least half when ignoring the point.
- The [shared-anchor writing study](writing-shared-anchor-ranking-2026-11-11.md)
  put 8/9 AI rewrites and 7/10 genuine human revisions first in a forced
  ranking. Using agreement as an evidence filter removed the only useful AI
  location along with two human flags. Keep selection and authorship confidence
  outside that agreement rule.

- The [independent audio speaker count](audio-independent-overlap-count-2026-11-11.md)
  withheld two of six retained three-voice routing errors while keeping all seven
  synthetic flags recovered in the separate-room test. A consumed diagnostic
  normalized the recording level and withheld uncertain zero counts. It left
  one of the six routing errors and kept five recovered flags. New disjoint
  triplets exposed zero baseline errors. Seek another panel that contains
  difficult overlapping speech before adopting the change.
- The [shifted JPEG likelihood check](image-shifted-jpeg-likelihood-2026-11-11.md)
  located 8/48 edits at quality 90 and flagged 11/120 controls. The existing
  aligned check found 46/48 and flagged 60/120. The candidate found no new useful
  locations and no useful quality-75 regions. Keep it experimental.
- The [cheek and jaw boundary replay](video-selected-face-boundary-2026-11-11.md)
  could measure four of 39 incumbent flags and corroborated none. All three
  genuine stress views were unmeasured. Reliable patch correspondence and texture
  coverage are the next requirements for testing this cue.
- The [journalism transfer](writing-editorial-transfer-2026-11-11.md) located
  3/6 known rewrites versus 2/6 for the current check. Both flagged the same
  9/16 human documents. The two methods found different edit locations. Keep
  the existing local opt-in and test new genres before extending coverage.

- The [matched short-passage check](writing-two-sided-usgs-transfer-2026-11-11.md)
  located 7/15 edited passages across the NIST and USGS panels versus 5/15 with
  the current passage check. It flagged 5/26 human documents versus 12/26. The
  local opt-in uses the existing classifier response, selects only an eligible
  middle sentence, and retains the three measured sentences for inspection.
- The [selected-window video landmark check](video-selected-landmark-reliability-2026-11-11.md)
  retained all 36 correct face-swap ranges and cleared zero of three genuine
  camera false flags. The 60 matched controls stayed unflagged. Three other
  camera deliveries remained unassessed.
- The [image lens-displacement pilot](image-chromatic-field-pilot-2026-11-11.md)
  could measure the lens field in two of four native images and one of four
  images after either JPEG re-encode. That coverage ended the trial before
  localization on the held-out edited images.
- The [measured speaker-count test](audio-measured-speaker-routing-2026-11-11.md)
  recovered seven missed synthetic voice clips with no additional false flag
  in the separate-room panel. Six of 14 controlled three-voice mixtures
  incorrectly qualified for the two-voice route. The supplement remains experimental.

- The [cross-layer voice head](audio-cross-layer-room-transfer-2026-11-11.md)
  improved AUC from 0.917 to 0.958 and from 0.948 to 0.979 on two new competing-speech
  mixtures of retained foregrounds. The frozen operating points recovered zero of
  six missed synthetic clips. The comparison ran the original speech gate and
  serving calibration. The separate-room confirmation with frozen room points
  detected all eight synthetic clips in the mostly-single-interferer scene with
  one added false alarm for the unpaired relation head. The same head flagged
  ten of twelve genuine controls with overlapping speakers. A new acoustic
  condition check needs independent validation before scoped adoption.
- The [two-sided writing check](writing-two-sided-context-2026-11-11.md)
  located three of six edited sentences versus one with the current check on
  six fresh NIST authors, with five versus six human false flags. The matched
  full-window candidate found two edits and flagged two human files. This
  comparison advances to a different technical publisher.
- The [registered video exposure comparison](video-registered-light-context-2026-11-11.md)
  retained all 36 correct swap ranges and cleared zero of 14 genuine camera flags.
  Registration was unavailable in 31 of 50 flagged views.
- The [independent measurement reviewer](independent-testing.md#recompute-selected-image-and-video-measurements)
  recomputes the sharp PNG seam arithmetic and the short-swap sampling schedule
  from supplied originals. The reviewer reproduced the existing signed examples
  and supports local testing without the detector models.

- The [six-frame paired face check](video-face-pe-confirmation-2026-11-11.md)
  found 45/48 whole-clip swaps versus 26/48 on separate source groups, with
  zero flags on 24 originals in both arms. An opt-in local app analysis now
  exports the source clocks and crops. The [outside publisher test](video-face-pe-publisher-2026-11-11.md)
  missed one full swap and all four deliveries of one short swap. It flagged
  two of 24 camera controls from one of eight recording groups, versus zero
  for the reference. The full-clip median hid an in-range response change.
  A [short-window test](video-face-short-window-2026-11-11.md) localized one of
  two authored photo swaps and flagged none of 11 controls. It missed all four
  publisher-video variants. The current appearance check found all six edited
  deliveries and flagged two synthetic controls. The [wider-window replay](video-native-interval-replay-2026-11-11.md)
  recovered the longer synthetic photo edit, with all four moving publisher
  variants still missed. The same work adds a portable way to verify interval
  endpoints and original before, during and after pixels. Keep the current
  appearance check and seek an additional moving source. The [lip-sync coverage fix](video-lipsync-mouth-coverage-2026-11-11.md)
  excludes unchanged-mouth windows that could create a false timing collapse.
  The moving portions of the local test retained their eligible coverage. The
  [short-clip follow-up](video-timing-coverage-and-codec-holds-2026-11-11.md)
  found 166 single-window timing measurements across 176 retained views. All
  timing-change decisions remained unassessed, and the app now says so at the
  measured source interval. The approximate picture-hold candidate found
  12/18 lossy holds and flagged 15/48 ordinary views. Keep the exact check.
  A [longer timing replay](video-longer-timing-transfer-2026-11-11.md) exposed
  timing flags formed across unmeasured gaps. Separating those windows cleared
  three ordinary flags and four neural-view flags that depended on the same
  unsupported comparison. Three of those four had overlapped the known redraw
  enough to count in the prior result. Continuous footage still needs better
  speaker attribution and timing sensitivity.
  The [speaker-attribution test](video-speaker-attribution-transfer-2026-11-11.md)
  improved target-voice separation in five of six measured mixtures but recovered
  no complete local timing pattern. The existing check's latest flag fell at the
  same unchanged time in the original and displaced clips.
- The [image patch and crop comparisons](image-relative-patch-transfer-2026-11-11.md)
  each completed inference. Neither localized a useful region on the held-out
  generator's six small edits. The [separate insertion transfer](image-bg-real-insertions-2026-11-11.md)
  found 4/8 useful PNG regions for relative crops versus 0/8 current, with zero
  unchanged-original flags in both arms. It also flagged 4/8 processing controls
  and localized zero JPEG75 edits, so the candidate remains experimental.
  The [separate remote-context holdout](image-bg-real-insertions-2026-11-11.md#separate-holdout-for-shared-processing-artifacts)
  removed all three processing-control flags and retained two useful locations
  on eight other PNG sources. This result remains specific to one generator and
  one matched processing control. On the [two-editor panel](image-background-patch-transfer-2026-11-11.md),
  the gate kept three useful PNG locations and flagged 23/48 controls. It found
  zero JPEG75 locations. The existing routes remain preferred. The app now
  presents [two source crops for non-mirrored copy matches](source-evidence-native-coverage-2026-11-11.md#automatic-copy-similarity-without-mirroring)
  with zero weight in the AI estimate. The [native patch addition](image-translation-patch-transfer-2026-11-11.md)
  raised correct source-pair localization from 9/32 to 28/32 for PNG and from
  7/32 to 19/32 for JPEG90, with zero pairs on 56 controls in each format.
  The [boundary blend comparison](image-boundary-blend-transfer-2026-11-11.md)
  found zero useful locations on 12 feathered edits in each format for both
  the boundary candidate and current routes. The same candidate found five
  sharp PNG patches and six sharp JPEG patches, with 36/72 control highlights
  under JPEG. Keep the boundary check experimental. The [noise-guided model](image-noise-region-candidate-2026-11-11.md)
  added three useful feathered locations across the two formats at its publisher
  point, with 130/144 control highlights. The separate control-fit points found
  no useful feathered locations. Retain the current regional routes. The app
  and new PDFs show the exact reliability mask when the check recorded one.
  The [fine-detail comparison](image-wavelet-region-transfer-2026-11-11.md)
  found one useful feathered location in each delivery on the earlier panel.
  Neither arm found a useful region on the 12 new source groups. The
  [offline source verifier](independent-testing.md#verify-image-crops-from-a-signed-export)
  compares the original pixels at each reported crop location.
  The [semantic context test](image-semantic-context-transfer-2026-11-11.md)
  reduced PNG highlights without finding any useful edit region. The
  [offline reference comparison](image-reference-comparison-2026-11-11.md)
  found all eight feathered edits in each delivery when the reviewer had an
  aligned earlier image. Ordinary multiple compression and broad processing
  can still produce local differences.
- The [strong-traffic voice comparison](audio-wavlm-strong-traffic-2026-11-11.md)
  raised noisy synthetic flags from 1/27 to 4/27 on admitted inputs, with zero
  flags on eight assessed originals. It used one noise recording and earlier
  foregrounds. On a [separate two-background test](audio-wavlm-two-background-transfer-2026-11-11.md),
  the paired method added zero flags. The current rule falsely leaned AI on
  4/5 assessed genuine inputs under one background and 1/6 under the other.
  A [background corroboration replay](audio-background-corroboration-2026-11-11.md)
  reduced those five false flags to three and preserved all 25 existing synthetic
  flags. The fresh challenge had limited genuine-speech coverage under strong
  noise. The [stronger frozen transfer](audio-frozen-corroboration-transfer-2026-11-11.md)
  reproduced five false flags across two backgrounds and neither check removed
  one. Each lost one synthetic detection. Keep both experimental.
  A [separate speech-frame representation](audio-speech-masked-representation-2026-11-11.md)
  assigned an incorrect original direction to eight synthetic recordings under
  one held-out background, where the current check assigned zero. The separate
  head remains experimental. The [temporal corroboration trial](audio-temporal-corroboration-2026-11-11.md)
  admitted clean and noisy speech from more genuine readers and agreed with the
  current check on all 60 selected views. The corroboration rule stays experimental.
  A [stereo filtering trial](audio-natural-stereo-coherence-2026-11-11.md) added
  nine synthetic flags across two recorded backgrounds and three genuine flags
  in one. Keep the current classifier and calibration.
  A [stereo component comparison](audio-stereo-component-transfer-2026-11-11.md)
  recovered no additional synthetic flags and reproduced severe false alarms
  on two other recorded backgrounds.
- The [paragraph context comparison](writing-paragraph-context-transfer-2026-11-11.md)
  localized 6/18 rewrites versus 4/18 for the strict raw rule and flagged
  3/21 human controls versus 4/21. The test used 11 new source articles and two
  generator families. The [fiction transfer](writing-fiction-paragraph-transfer-2026-11-11.md)
  localized 10/22 rewrites versus 13/22 for the strict rule. Both flagged zero
  of 33 human controls. Keep the current writing check. The ordinary document
  view and PDF now show the reason and coordinates when a hybrid PDF page was
  only partly checked. The [practical DOCX panel](writing-practical-notice-transfer-2026-11-11.md)
  recovered six of eight rewrites with broad review and flagged five of 18 human
  controls. The stricter and paragraph rules recovered none. All 26 files
  retained their exact text and source positions through normal extraction. The
  [correspondence holdout](writing-correspondence-transfer-2026-11-11.md)
  improved from 14/16 to 16/16 useful locations, but flagged 11/24 human files.
  Keep the current estimate and show the selected body part and omitted attachment
  count in the ordinary email report.
  The [modern bulletin test](writing-modern-bulletin-transfer-2026-11-11.md)
  found all eight eligible rewrites with the current review rule and six with
  the practical point, with ten of 30 human controls flagged under either rule.
  The app and exports now distinguish a mixed-role highlight from a supporting
  context that read a quote or signature outside the highlight.
  The [article context test](writing-article-context-transfer-2026-11-11.md)
  retained 8/8 useful locations and reduced ordinary flags from 15/18 to 2/18
  when a paragraph-only read also supported inspection. An opt-in local article
  view exposes those paired responses. The existing overall estimate stays in use.
- The [portable evaluation contract](independent-testing.md#evidence-localization)
  now checks region, frame, sample and passage overlap counts separately from
  file-level directions. Reviewers can report model-free checks and share
  coverage and overlap aggregates without publishing private source material.

## Progress on 10 November

- Completed native feature extraction, three compact heads, six separate study
  calibrations, both locked held-out roles and a separate arithmetic replay.
  Across the two held-out roles, interaction localized 9/40 edits versus 4/40
  visual-only, with 9/80 ordinary-input flags versus 7/80. Keep interaction and
  complementary visual evidence for transfer; 31/40 edits still missed by
  interaction. These consumed, narrowly connected source contexts do not qualify
  deployment or customer probabilities. The remaining work in the earlier
  construction snapshot below is superseded for these completed stages.
  [Complete calibrated comparison and limitations](native-mouth-calibration-and-transfer-v129-results.md).
- Fixed a new 32-clip, sixteen-source GRID transfer selection from publisher
  directory metadata before media/detector response selection. Full archive
  acquisition and complete native RGB/PCM banks are running sequentially, with
  partial snapshots clearly separated from cohort completion. VidTIMIT media
  remains excluded by its full NC-ND metadata. The same MIT human generator
  assets and an isolated geometry runtime are retained. The separate CPU neural
  qualifier passed RAM admission but was stopped by the unchanged 6-GiB sampled
  RSS limit; its failed attempt remains preserved. The new bounded-convolution
  context keeps the same inputs/weights/precision/resolution and resource limits.
  Its actual run now passes in 25.63 seconds at 2.37 GiB maximum sampled RSS,
  with byte-identical comparison to the preserved first original crop and exact
  own repeats. This qualifies the runtime, not fresh detection or achieved lip
  timing. [Actual bounded CPU result](portrait-tiled-runtime-v139-result.md).
  A richer expression-driving
  bridge has passing public arithmetic checks but remains neurally unqualified.
  No new prediction result or default follows from these data/software checks.
  [Current preparation report](grid-transfer-progress-v138.md) and
  [model-free contribution entry points](independent-testing.md).
- Completed all native mouth construction and separate mask replay: 283 ready
  clips across 48 sources, five explicit short-source unavailable edits, 94 neural
  mouth-edit clips and 189 original/ordinary controls. All 4,700 masks, 21,767 pictures
  and 11,808,637 audio samples/clocks match. All 16 stages exit zero without guards
  or sampling errors. The reference-interaction feature math adds 13 passing
  authored checks but no real-media fit or detector accuracy claim. Full blind
  feature extraction, separately calibrated comparisons, placement/generator
  transfer and customer-flow tests remain open.
  [Complete construction/replay result](native-mouth-clips-v120-complete.md).
- Complete-clip construction is running under the frozen native mouth plan.
  The first three batches pass: 103 ready full clips and five unavailable edits
  for the short source, with 3,140 actual neural inputs. All ready files pass
  full decoded RGB/clock/PCM/origin checks. The separate source-to-mask replay
  is frozen and remains unexecuted at this partial snapshot. Detector scores,
  alignment quality and held-out accuracy remain open.
  This is a historical partial snapshot; the full completion is recorded above.
  [Partial construction evidence](native-mouth-clips-v120-progress.md).
- Lossless native audiovisual assembly passes two authored cases and one complete
  fitting-source rewrite: all 76 pictures and 43,912 audio samples/timestamps
  reproduce exactly, with stream offsets preserved and zero neural calls.
  [Actual codec result and limits](native-av-mux-v119-result.md).
- The frozen native renderer now executes successfully after the owner's fresh
  RAM check: four renders, 17 actual neural inputs, exact seeded repeat and
  changed output under earlier/zero conditioning. The historical readiness
  snapshot remains unchanged. Achieved synchronization and detector accuracy
  remain unqualified. [Actual native result](native-mouth-renderer-v118-result.md).
- Fixed a production lip-sync coverage bug: ticks inside long source-picture
  gaps could reuse pictures hundreds of milliseconds away. Such windows now
  remain unassessed before crops/models, with regular 24/25/30/60-Hz routing
  retained. The 39 timing/evidence and 79 assessment/export checks pass locally.
  Model selection and confidence are unchanged. [Fix and evidence](video-lipsync-gap-coverage.md).
- Completed the next native mouth-render input bank: all 48 recordings retained,
  with 1,128 exact RGB/timestamp-bound pictures for 47 planned edits and their
  before/after context. The short source stays unsupported. Eight decoder-only
  batches pass; no new model ran. The counterfactual renderer procedure is now
  frozen in an explicit retained research runtime, with the initial missing-
  dependency preparation failure preserved. Native rendering, achieved alignment
  and articulation detection remain unqualified. Neural admission still requires
  6 GiB available RAM; its earlier read-only check found 2.84 GiB and started no
  model stage. A later owner-requested admission succeeded, as recorded above.
  The 27 targeted input/renderer/runtime checks pass locally.
  [Input result](native-mouth-inputs-v116-result.md) and
  [actual pending admission](native-mouth-renderer-readiness-v118.md).
- Completed the fixed native-video selection comparison: 138 clips across 23
  source groups, 9,244 actual neural calls. Raw and calibrated appearance arms
  both flag all 46 constructed swaps, with zero false flags among 92 controls
  and two inconclusive controls. Local support also ties; mean bounding-range
  IoU is 0.551, with many unexamined source frames. Keep the current check and
  retain the candidate for broader testing; this panel does not establish a
  replacement gain or general accuracy. The original saved-record/scalar bug
  and failed attempt are preserved; a new frozen context fixes only that
  handoff. [Full outcomes and limitations](video-native-appearance-v112-result.md).
- Passed the authored CPU video/speech contracts, then actually encoded all 48
  existing speech sources in 56 forwards. Eight bitwise source repeats and all
  3,642 native picture/audio routes pass, including explicit padding. A complete
  data-only saved-state review passes. Voice-derived states stay private on
  Seagate. This qualifies source inputs for future mouth rendering, not a new
  lip-sync detector or customer confidence. Native rendering, achieved alignment,
  articulation comparison and full-flow qualification remain unfinished.
  [Executed source result and next stage](av-source-encoder-v115-result.md).
- All 95 targeted software checks pass locally. Updates are pushed manually;
  GitHub Actions remains disabled. No outside reviewer, outreach or production
  deployment is claimed. The broader image/audio/writing and serving evaluation
  requirements above remain open.

## Progress on 8 November

- Completed photo attribution and native pixel lineage before image learning:
  all 523 metadata cases checked; 465 exact current photo/license matches, 58
  unresolved cases retained. All 465 source comparisons and separate local
  replays complete: 397 exact RGB matches, 68 differences, 930 matching dual-
  decoder checks. All 49 authored tests pass. Current generator-output terms
  require further release-specific review; no training/model admission or new
  detector accuracy is claimed. Only 19 unscored test sources also match exact
  pixels, so the 48-source confirmation requirement still needs more sources.
  [Complete source-chain findings and limits](image-source-lineage-v20-result.md).

- Audited actual per-photo image metadata before regional fine-tuning:2,242
  distinct source photos,597 preliminary BY/BY-SA-compatible records,523 queued
  for private attribution/lineage review after common-generator/source exclusions.
  Publisher training rows represent1,558 source photos; three FLUX original
  CSVs identically contain331 test photos and cannot assign training roles.
  Actual generation splits are disjoint. Full separate local reconstruction and
  public-only replay pass, plus52 authored tests. No photo is admitted to training;
  only24 unconsumed compatible three-generator test sources remain, so additional
  permitted sources are needed for a48-source confirmation.
  [Complete findings and reproduction](image-source-rights-v19-result.md).

- Completed the new source-preserving image localizer comparison and complete
  native replay:384 inputs,798 project-source forwards, no skipped/failed
  images. All384 float32 maps and separate IEEE/component reconstruction match;
  both genuine points and288 gated pixel-count sets reproduce. Candidate still
  finds0/48 useful regions per format versus9/48 native and6/48 JPEG current,
  with larger wrong regions on some genuine sources. Rejected; no fresh reserve
  or confidence admission. Ungated proposals also generate frequent genuine
  false highlights. [Complete result and next methodological question](image-relayformer-v18-result.md).

- Finished the controlled writing-span comparison:992 documents, complete
  native/source replay,864 separately recounted diagnostic arm masks, exact
  refits of both heads and three calibrators/points. Interval localization
  recovers more known intervals but loses27 previously useful mixed locations
  and remains blind to every short insert. Rejected as a replacement; confidence
  unchanged. [Result and limitations](writing-spans-v7-result.md).
- Reviewed newer word-edit and image-segmentation methods, excluded noncommercial
  paired text/checkpoints, and acquired two separately permitted image candidates.
  Both execute on authored inputs:24 actual forwards total,32 source/geometry
  tests, exact repeats and fixed optimization bounds. The new localizer loads
  all410 released tensors and preserves full image coverage with explicit
  padding/source projection. Sample video-tail loss remains unintegrated.
  No project-media inference or improved quality is claimed; a complete frozen
  real-source comparison and fresh confirmation are still required.
  [Sources, executed contracts and remaining work](new-localizers-and-editing-methods-2026-11-08.md).
- The owner requires local verification and direct pushes. GitHub Actions stays
  disabled; historical Linux receipts remain historical and are not rerun.

## Progress on 7 November

- Ran the public AudioSeal generator/detector locally without provider access:
  32authored cases,41actual forwards, all model tensors verified. A separate
  replay checks1,024 time bins and8gapless codec inputs. Unmarked-carrier false
  responses and noisy misses are retained; no real-speech/provider accuracy or
  serving admission claim follows. Public-family work is executable now and
  does not wait on private APIs.
  [Actual local native result and remaining qualification](public-audio-watermark-v1-result.md).
- Implemented and tested two new source-resolution noise/texture localizers,
  with and without a fit-genuine intervention projection. All 1,344 consumed-
  source feature cases completed. On 108 diagnostic edits, each candidate has
  only one IoU>=.3 case versus six current; both are rejected. A separate replay
  matches 1,008 score grids and exact source masks. This is an actual new feature
  comparison, not another calibration-point adjustment or fresh accuracy claim.
  [Complete method, controls and result](image-residual-v13-result.md).
- Added source-bound structured AIGC label evidence across supported image,
  video, audio and document formats, normal views, all report profiles, API and
  JSON. Claims stay unverified and outside confidence; missing/malformed fields
  do not clear originality or bypass models. Original checksums and privacy
  indicators accompany the records. 408 selected backend and 74 frontend checks
  pass; no private watermark decoder or statistical model is admitted.
  [Working coverage and validation](structured-source-labels.md).
- Completed fresh JPEG confirmation of the frozen regional candidate on 48 new
  source pairs, with 102 actual native forwards. Candidate flags 12/48 edits
  versus 10/48 current, but 5/48 genuine false flags versus 2/48; it also loses an
  edited-file flag and has worse Brier than primary-only. Rejected on three
  fixed gates. Current masks meet IoU >= .3 on 3/48 edits; no pixel-selection gain.
  Separate source/map/component replay and 192 independently computed file
  probabilities match exactly, with no extra neural inference. No serving
  promotion or tuning on this now-consumed holdout.
  [Full fresh result](image-fresh-v12-result.md).
- Implemented the shared image/video/audio/passage calibration framework and
  explicit source-disjoint specialist/fusion/calibration/confirmation roles.
  Local authored command flow passes for16 context heads/four calibrated
  combiners;63 selected tests cover leakage, coordinates, missing local truth,
  skipped checks, duplicate-family evidence and the version2 safeguard preventing
  unrelated genuine regions subtracting from positive evidence. Real-data adapters, support,
  new comparisons and fresh confirmation remain required across all four media.
- Tested area-aligned regional calibration on all312 consumed response grids.
  Both formats still locate0/48 edits versus12/48 current, with one genuine
  highlight each. All plateau/source/pixel checks pass, but quality gates fail.
  The pixel-maximum/display-area mismatch alone does not fix misses. No new
  neural inference, reserve scoring or serving change. [Complete result](image-region-calibration-v9-result.md).
- Completed the regional patch-drift comparison:312 consumed inputs, exact
  native/pixel replay and genuine-only points published before development.
  Both readouts located0/48 edits per format versus12/48 current. All four arms
  rejected; no fresh confirmation or service change. Two operational failures
  were retained, a resource-only retry was prospectively published, and all34
  repeated cases matched exactly. [Complete negative result and reproduction](image-patch-drift-v8-development-evaluation.md).
- Finished fresh face-transition confirmation and local integration:60/60
  located edits versus25/60, same26/60 genuine review flags;58 new-cue ranges
  and two declared fallbacks. Actual-adapter parity, paid local flows,
  UI/API/PDF/JSON filtering and rollback passed. The main local app uses the
  appearance cue; repository default remains legacy. [Local rollout](local-face-appearance-v17-rollout.md).
- Public audio packages, complete rejected speech/writing comparisons and
  separate-host replay entry points are available in
  [independent testing](independent-testing.md). Outside human reviewers remain
  zero, outreach is not authorized, and GCP is still deferred. Regional image
  edits, difficult audio, lip sync, general video and writing remain open work.

## Progress on 6 November

- Published/remote-verified the full two-source audio freeze as220e488 before
  starting original-recording extraction. The live append-only run preserves
  all unassessed inputs; no fitting/calibration or new accuracy result yet.
  Added a separate source/bag arithmetic reviewer outside the frozen79 files;
  seven hand-computed numerical tests pass. Full reconstruction and exact refits
  await the completed recording run. [Execution](audio-separation-v14-execution.md).

- Completed the two-source audio execution runner and prospective lock: all
  4,248 original consumed inputs, unchanged roles/62 unassessed,79 project/six
  installed code pins, own bag fitting/calibration, both raw/mapped stability
  and original16 requirements. Both estimates retain full clocks, waveform and
  feature hashes; native review also checks the inspected upstream graph on
  original and required band inputs. All67 selected tests pass, and13 actual
  freeze alterations are refused. No project recording forward/fitting has begun.
  Publish and verify the complete freeze before extraction; reserved1,080 inputs
  and serving defaults remain unchanged. [Execution](audio-separation-v14-execution.md).

- Prepared a materially different overlapping-voice method: a pinned
  5,066,929-parameter two-source separator, both source estimates inspected by
  the frozen voice backbone, and a weak mixture-label objective that does not
  label both estimates AI. Strict345-tensor load and all three synthetic upstream
  graph comparisons pass exactly. Retained the card's3.0/4.0 ShareAlike version
  discrepancy and checkpoint8000/training-card16000 metadata discrepancy; changing
  only metadata rate has no forward effect. Twenty focused tests pass, including
  both analytic gradients, source-permutation invariance and synthetic fit/calibration
  smokes. No project recording, project fitting/calibration, reserve, installed
  dependency or serving change yet. Complete the
  source-bound execution freeze before forwards. [Preparation](audio-separation-v14-preparation.md).

- Completed all four writing focus/context arms:40 authors/674 foci, no failures
  or token omissions. Candidate/focus recovers306 versus295 pure-Bot passages at
  the first fixed research reference, but highlights136 versus52 pure-User-origin
  passages. At the second:287 versus276,108 versus38. Four of nine frozen gates
  fail, rejecting the replacement without source-disjoint confirmation or serving
  changes. Removing context from the current model also loses substantial AI
  recovery. All160 native replays and separate reconstruction of every aggregate,
  rank, interval and decision pass; earlier674 context scores match within1.2e−7.
  User-origin highlights in assisted text are not unassisted-human false-alarm
  rates. [Complete result and diagnosis](writing-focus-v3-development-result.md).

- Implemented a prospective writing2x2 comparison: current/candidate models on
  exact focus sentences and three-sentence contexts, all40 consumed authors and
  674 foci. Forty-five selected tests pass. Independent native token-ID audits
  retain every input; candidate inference has not begun. Freeze binds12 project
  files, installed/runtime/source/model assets, both diagnostic references and
  author-group continuation gates. Native single-window replay is predeclared.
  No fitting, source-text redistribution, fresh confirmation or serving adoption
  follows from this development design. [Protocol and execution](writing-focus-v3-execution.md).

- Completed the new speech-backbone comparison:4,248 feature records and1,449
  development outcomes. Combined noise/music improves, but correct AI648→522,
  correct original356→437, false AI58→4 and inconclusive387→486. Three original
  gates fail; feature-only fails eleven. Reject both without reserved scoring or
  serving changes. Initial duplicate thread initialization failure is preserved;
  the corrected journal's first33 successes match exactly. All593 native replays,
  numerical reconstruction and both exact refits pass. Lossless public feature
  packets restore the full receipt; empty-cache public-only refits and12 actual
  alteration checks pass. Guard removal alone still cannot recover four families.
  [Full result](audio-wavlm-v13-development-evaluation.md).

- Prepared writing focus/context comparison inputs: all40 consumed documents,
  40 author groups and674 exact historical sentence/context bindings; no omitted
  AI-labeled characters. Twelve position/provenance tests pass. A125M-parameter
  MIT-declared alternative strictly loads, with zero classifier forwards so far.
  User-origin spans inside assisted documents are not unassisted controls; do not
  mislabel their highlights as population false alarms. No new training or source
  text redistribution grant is inferred from the retained data terms conflict.
  [Preparation and next comparison](writing-focus-v3-preparation.md).

- Completed the fresh image-crop comparison: all192 inputs, zero failures.
  Native overlapping edits12→20 but genuine highlights2→9; JPEG edits12→13
  and highlights3→6. Both fail the unchanged allowance and remain disabled.
  Twelve consumed native replays match exactly. All288 TGIF originals/eight
  historical web examples are now excluded from future confirmation.
  [Complete aggregate result](fused-crop-v6-confirmation-result.md).

- Completed audio v12: all4,248 extractions/all1,449 development outcomes,
  zero failures. Correct conclusions913/920 versus service1,004, with2 false AI
  each and534/527 inconclusive versus387. Both fail the same four of sixteen
  gates. All508 native checks, both exact refits and public-only numerical
  reconstruction pass. Aggregate results and full lossless reproduction packets are published. No reserved confirmation or service changes.
  [Complete aggregate result](audio-layer-v12-development-evaluation.md).

- Prepared a different noise-aware speech backbone with94.4million frozen
  parameters, publisher-checksummed weights and model-card-linked CC-BY-SA-3.0
  terms (separate code MIT). Strict load and two synthetic-only repeated forwards
  passed. All58 selected tests pass. Freeze binds69 project/five installed files,
  every4,248 original consumed input, source roles, original16 gates and reserved
  confirmation. Thirteen actual-metadata alterations are rejected. Publish before
  source forwards/fitting; no detection gain or outside review claimed.
  [Prospective preparation](audio-wavlm-v13-execution.md).
  After remote verification as45e8ff9, all4,248 extractions completed (4,186
  assessed/62 unassessed, zero failures), and both fixed heads/calibrations were
  saved. Initial comparison stopped at34 on duplicate thread initialization;
  the failed attempt is retained and a single-owner callable launcher is bound
  before continuation. No completed quality result or serving change yet.

- Prepared fresh image-crop confirmation: all48 unused pairs/all48 publisher
  masks acquired, exact RGB-reference/geometry checks passed and no source/file
  reuse after248ID exclusions. Minimum perceptual distance12 to history/20 within
  panel. All20 method/runtime/test pins and original gates frozen;78 selected
  regressions pass. Publish/verify before twelve consumed native replays and192
  fresh inputs. No fresh detection result or service change yet.
  [Execution record](fused-crop-v6-confirmation-execution.md).

- Verified the owner's institutional AITDNA archive against its published
  checksum and all ZIP CRCs. Its span file differs in bytes but all362 decoded
  records/schema match the acquired copy. The institutional CC-BY-SA metadata
  and bundled GPL-2.0 README retain the terms conflict; no source replacement,
  new writing inference/training or outside review occurred.
  [Institutional follow-up](writing-source-preflight-v2.md).

- Completed the separate FUSED crop-localization development experiment: all
  96 inputs/384 crop forwards, no failures. Native overlapping edits rise10→12
  with genuine highlights2→3; JPEG overlapping edits3→10 with highlights1→2.
  Both formats advance to separately frozen fresh confirmation, not service
  adoption. All88 targeted tests pass.48 new source pairs are selected without
  inference, excluding240 prior originals/eight historical web examples.
  [Results and next work](fused-crop-v6-development.md). A primary-source review confirmed
  commercial grants for SAFIRE/CAT-Net but owner-only downloads, and unresolved
  separately hosted DetectiveSAM weight terms; no restricted weights acquired.
  [Source review](image-localizer-v6-source-review.md).

- Prepared audio v12 before new source forwards: fixed hidden-state indices
  8/16/24 versus unprojected-final control, unchanged objective, independent guards
  and sixteen gates. Rehashed all 4,248 waveform files; source roles, 62 unassessed
  records, 450 old probe references and reserved confirmation remain unchanged.
  All 64 method/protocol files and original baseline digests are bound. All 160
  regressions and nine actual-freeze alteration checks passed. Publish before
  extraction; no accuracy, serving change or outside review claimed.
  [Preparation and execution](audio-layer-v12-execution.md).

- Completed full-vector audio v11: all 1,449 inputs, 1,420 assessed/29 unassessed,
  zero failures. Correct conclusions 944 versus 872 PCA16 and 1,004 service;
  false AI 2/false original 0/inconclusive 503. Clean retention recovers, but four
  unchanged gates fail; no fresh scoring or enablement. All 125 native cases,
  every 26 new probe and exact fitting-only refit passed. Public reproduction,
  nine actual-capsule alteration checks and 127 regressions passed. Guard removal
  still cannot recover six families. Next investigate preselected earlier/middle/
  later frozen features. [Complete result](audio-full-representation-v11-development-evaluation.md).

- Prepared one full-representation audio v11 comparison after v10 rejection:
  retain all1024 frozen coordinates with fitting-only diagonal normalization
  at the old16-dimensional variance budget, unchanged objective/regularization,
  independent calibration and all sixteen gates. Original4,248 consumed inputs,
  424 exact cached probes and reserved confirmation remain pinned. All119 selected
  regressions and seven actual-public-freeze alteration checks passed;58 code
  files are frozen. Publish before project fitting. No gain, serving change,
  fresh scoring or outside review claimed. [Preparation](audio-full-representation-v11-execution.md).

- Completed conditional audio v10: all1,449 outcomes,1,420 assessed/29 unassessed,
  zero failures. Nonlinear and added-quality arms reduced false AI58→0 but correct
  conclusions fell1,004→804/609 and inconclusive rose387→644/840. Both failed the
  same five of sixteen unchanged requirements and remain disabled. All121 native
  cases and every22 new probe matched; fitting-only CPU refit/JSON-native prediction
  parity passed exactly. Complete public results reference existing vectors and
  include all outcomes/parameters for numerical reproduction without model/media
  downloads. No fresh scoring or outside human participation. Guard removal alone
  still cannot recover five generators; next test representation/mapping rather
  than weaken uncertainty rules. [Result](audio-conditional-v10-development-evaluation.md).

- Prepared audio v10 before project fitting: a compact nonlinear control, a
  separate arm adding two unchanged waveform descriptors and a non-adoptable
  quality-only nuisance check. All4,248 exact quality measurements and402 cached
  filtered-probe measurements passed. Freeze binds54 project/twelve installed
  learner files and unchanged commercial checkpoints/source roles/baseline.
  All96 selected tests, public metadata review and seven actual-freeze alteration
  checks passed. Original sixteen gates, independent calibration and confirmation
  restrictions remain. No improvement or outside review is claimed.
  [Design and execution](audio-conditional-v10-execution.md).

- Completed corrected audio v9: all 4,248 consumed features (zero failures),
  independent fit/calibration and all 1,449 development outcomes. False AI fell
  58 → 3, but correct calls fell 1,004 → 872 and inconclusive rose 387 → 573.
  Five of sixteen unchanged requirements failed, so reject without fresh scoring
  or serving changes. All 99 predetermined native cells/fifteen probes matched;
  complete numerical reconstruction and CPU refit passed with zero coefficient
  error. Public-only numerical replay reproduced exact original feature/fit
  artifacts with an empty model cache. No outside human review is claimed.
  A post-result fixed-probability ceiling proves guard removal alone cannot
  restore the required AI retention for five generator families. The next
  method must improve mapping/representation, not just weaken uncertainty checks.
  [Full result, conditions and next work](audio-identity-v9-development-evaluation.md).

- Prepared an additive audio verification supplement during corrected extraction,
  before fitting/comparison. The fixed native plan covers all 99 development
  dataset/generator/truth/condition cells without score selection. Complete
  numerical review reconstructs guard ranks, both stability checks and all
  outcomes, with an optional fitting-only CPU refit. All 115 selected regressions
  passed. This does not change the original learner or sixteen acceptance gates;
  no completed-candidate result or accuracy gain is claimed.
  [Verification protocol](audio-identity-v9-review-protocol.md).

- Audited all 362 mixed-writing source records and original extraction positions:
  zero failures. Forty previously consumed authors are excluded; 59 unused authors
  supply 138 mixed and 53 declared human-only records before further connected-source
  selection. Exact pinned publisher cards expose conflicting dataset license
  declarations; record the ambiguity before new training/redistribution. The
  installed detector's card consistently permits commercial use. Nine focused
  tests passed. No inference, calibration, accuracy gain or outside review is
  claimed. [Source audit and remaining work](writing-source-preflight-v2.md).

- Froze corrected audio v9 execution before new extraction/fitting, retaining
  the unchanged learner and all sixteen gates. The runner binds all 4,248 consumed
  inputs and refuses reserved confirmation. Checkpoint/runtime/feature checks
  identify 1,846 exact-input cache candidates; nineteen native checks are required
  before reuse and 2,402 inputs require new extraction. All 132 selected tests,
  the 51-file metadata/code review and eight actual-freeze alteration checks
  passed. No model gain, serving change or outside review is claimed.
  [Execution and remaining stages](audio-identity-v9-execution.md).
  Published the freeze as `4e1c59c` before any new forward; all nineteen native
  cache replays subsequently matched exactly and permitted exact-input reuse.
  Consumed extraction is in progress; no corrected fitting or fresh scoring yet.

- The wider audio identity audit superseded v8 before fitting. Checked1054
  originals across all868 earlier source references; no reserved recording,
  non-silent2–4second prefix or known script was reused. However, the publisher's
  cloning code identifies LibriSpeech parent speakers, revealing six cross-role
  groups missed by literal dataset-key checks. Stopped v8 at1882 complete records
  without fitting or fresh scoring. The corrected metadata plan retains all120
  confirmation recordings in118 groups, retires four connected consumed sources,
  and keeps known cloned/genuine identities together.5328 repaired inputs rebuilt
  from the public capsule alone;14 identity/role/privacy tests and five actual
  capsule alteration checks passed. No model improvement, serving change or
  outside validation is claimed. [Correction and next work](audio-source-identity-v9.md).

- Prepared the next audio v8 experiment before new fitting/inference: account for
  all540 consumed source references, retain476 eligible originals in174fit/
  84calibration/106development connected groups, and reserve120 new foregrounds
  across genuine speech, ElevenLabs-v4, Kokoro and Chatterbox. Prepared5364 exact
  four-second inputs with66 role-separated noise/music/genuine-speech backgrounds.
  All662 source-byte/full-production-PCM identities are distinct, with known
  voice/script/prefix connections kept together. The initial insufficient-noise
  preparation is preserved; a fixed publisher-checksummed public-domain subset
  extension supplies eligible originals without changing duration rules.
  Published metadata checks cover every recipe and40 frozen implementation files;
  39 focused tests passed. No new performance result, serving change or outside
  reviewer participation is claimed. [Method and source review](audio-multisource-v8-development.md).
  After the prospective freeze was published as`6ad06da`, complete independent
  source reconstruction and a public-capsule-only rebuild both matched all5364
  inputs;72 fixed native decoder checks passed.43 focused tests passed. An
  additive141-source publisher-location supplement, full digest receipts and an
  offline model-free reconstruction helper improve external reproducibility.
  [Source replay and remaining stages](audio-multisource-v8-execution.md).

- Completed all1080 frozen fresh audio v7 inputs across two new generator
  source panels and genuine noise/music/overlapping-speech controls. Correct
  calls761→823 and Brier0.121088→0.113554, but false AI35→38, false original
  39→52 and worse equal-level speech Brier failed four of sixteen gates.
  Rejected under unchanged requirements; current calibration/overall confidence
  retained. All18 preselected native source replays matched exactly and the
  complete numerical record is published. This exposes modern-TTS false
  original calls under noise and genuine multi-speaker false alarms; it is
  not outside validation or completion of the other media work.
  [Full result and next work](audio-guard-v7-confirmation-evaluation.md).

- Prepared the prospective v7 fresh audio sources without classifier inference:
  60 unused genuine speakers,30 ElevenLabs voices/complete scripts,30 Chatterbox
  prompt IDs/dialogues, six separate genuine background speakers and twelve
  unused noise/music recordings. Known shared script openings leave110
  connected foreground groups, not120 independent prompts. Source/full-decoded/
  prefix checks and publisher checksums passed; public references and a
  prospective1080-input design/gates are available. Exact variants and scoring
  code still require a final freeze before inference. This is source preparation,
  not completed confirmation or outside testing. [Design](audio-guard-v7-confirmation.md).

- Audio v7 separated fitting from class-set calibration and compared with the
  exact current guarded service on all720 consumed development inputs. All12
  frozen development gates passed:563 correct versus500, the same1false AI/0
  false-original calls, and156 inconclusive versus219. The candidate still
  leaves83/360 AI inputs inconclusive; music5 log loss and AUC worsened.
  All720 source bindings/297 probe views passed audit, and public CPU refit
  reproduced the fitted coefficients exactly. This permits a separately
  frozen fresh comparison only; no serving/overall-confidence change or
  outside validation. [Complete result](audio-guard-v7-evaluation.md).

- V11 froze a49-variable video-window learner combining retained vision heads
  with measured mouth/pose/quality. Generated all160 complete fitting clips on
  the original16 fit actors, separate from4 point/8 development actors. Source
  audit passed9580 frames,3200 exact masks, selected original PCM,1280 neural
  input bindings and128 fixed output replays. All24 point clips have valid
  measurements. All160 fitting clips supplied3560 windows, and the classifier
  completed all88 development cases:32/32 locations versus14/32, but22/56
  ordinary flags versus0/56, including4/8 untouched originals. Rejected on the
  unchanged false-flag gates. A post-hoc scalar bound retains at most16locations
  at<=1ordinary flag; a cutoff change alone cannot meet the frozen criteria.
  Published all6373 eligible windows and fitting coefficients; independent CPU
  refit matched exactly. Serving/overall confidence unchanged; no fresh or
  outside validation. [Complete result](mouth-temporal-v11-evaluation.md).

- V10 created/audited24 ordinary point clips with exact common source clocks,
  preserving all four point actors (three incomplete final audio spans trimmed,
  no padding). Fixed source-relative temporal windows, actual window-level
  calibration and two predeclared combinations completed all88 consumed clips.
  Primary located21/32 versus V8 14/32 but flagged12/56 ordinary controls;
  secondary26/32 with24/56. Both failed the fixed ordinary-flag gates and remain
  disabled. Native replay passed4704 inputs/9408 head bindings; stdlib review
  reconstructs all5626 eligible windows, including unhighlighted candidates.
  First-family sensitivity improved, while ordinary-motion discrimination is
  now the demonstrated blocker. No new/fresh actors, weight training, package
  upgrade, global-confidence or serving change. [V10 result](mouth-context-v10-evaluation.md).

- V9 source-matched ranking/teacher retention trained the same two small
  vision blocks on the immutable V8 panel, with 768 fitting pairs and 4,096
  fitting-only teacher targets. All teacher/source/full-forward/native-video
  replays passed. It lost V8's gain: 0/32 located versus 14/32, both 0/56
  ordinary flags. Rejected under the frozen gates, preserving V8 and the app.
  A post-hoc scalar-ranking bound proves that no point allowing <=1 ordinary
  flag could recover more than one first-family edit or any second-family
  edit on this consumed panel. Next work needs source-relative temporal
  evidence or better separation, plus calibration at the actual window unit;
  point relaxation alone cannot meet the retained requirements. No fresh
  actors or outside review. [V9 result and diagnostic](mouth-paired-v9-evaluation.md).

- V8 actually trained the final two small vision blocks on 1,600 two-family
  stills (16 fit/four point actors, eight excluded video actors). All source,
  native-input and learned-state replays passed. On 88 consumed clips it
  localized 14/32 AI edits versus 0/32, with 0/56 ordinary flags versus 1/56.
  All successes were MuseTalk (14/16); LivePortrait remained 0/16. The frozen
  both-family requirement failed, so the candidate is retained as research,
  not enabled or advanced as a passing replacement. Published source/role/
  learning bindings, full results and a standard-library retained-window
  reviewer; no fresh actors or outside review. [V8 results](mouth-learning-v8-evaluation.md).

## Progress on 5 November

- Completed a new 48-source paired PNG/JPEG IML-ViT comparison, including a pinned
  commercial-permission record and bitwise reference parity. Rejected replacement
  in both formats because detection overlap and false highlights worsened.
  [Results and limits](iml-vit-localization-evaluation.md).
- Added a portable four-media evaluation contract, complete-denominator and
  split-leakage tests, contributor documentation, and an evaluation issue form.
  These enable independent contributions; no external reviewer has participated
  in these local experiments yet.
- A whole-image-residual tiling prototype added one false review cue and recovered
  no additional development edits in either tested format. Its frozen development
  result is retained; do not promote it or consume confirmation inputs to claim
  a gain. A separately versioned tile-local normalization experiment is planned.
- Reserved 48 further source-disjoint image pairs for the next viable frozen
  method. One secondary prompt mask is unavailable in the publisher archive.
  The original and edited image bytes remain required; exact changed pixels are
  still the primary reference. The missing mask is explicit in the manifest and
  no source is substituted. The tile-local normalization variant subsequently
  scored them after its rule was frozen. Its one-edit development gain did not
  generalize: no additional confirmation edit overlapped in either format.
  [Complete tiling results](tiled-splice-evaluation.md). Neither variant is enabled.
- The next video comparison targets the lighting-related false alarms of the
  consecutive-crop method. A photometric-normalization candidate will use the
  existing commercially permitted checkpoints, consumed development data, and
  new source/donor-disjoint confirmation clips with ordinary-edit controls.
  It will not change the overall confidence calculation during comparison.
  Thirty new source groups have supplied all 120 planned original, swapped, and
  ordinary-edit clips. The unchanged current check completed its baseline. Both
  consecutive-crop arms were scored on 120 consumed v5 development cases. Raw
  crops were selected as primary there, before fresh candidate inference;
  normalization remains a secondary comparison. Fresh confirmation completed:
  raw crops localized 19/60 swaps versus 7/60 for current, with the same 21/60
  genuine/control review flags and no lost correct locations. The normalized
  arm added no improvement. [Results](video-sequence-v6-evaluation.md).
  Next: integrate the primary method with exact input bindings and verify UI,
  PDF, JSON and the paid analysis flow before local enablement. It must retain
  zero weight in overall confidence until separately calibrated for that purpose.

- Integrated the primary raw video sequence method behind an explicit local
  mode with rollback. Production-adapter parity passed all 120 consumed v6
  inputs. Real prepaid worker/report/package tests passed on the swap, its
  untouched source, and rollback; same-input overall decisions were identical.
  The small review check runs even when the old large classifier is resource
  skipped, while overall confidence still discloses that missing check. Native
  inputs, before/during/after roles and aligned-crop commitments travel through
  evidence version 7. See the [release record](local-face-sequence-rollout.md).
  No voice, lip-sync, text or independent-review requirement is completed by
  this video-only integration.
- Completed a 720-input, consumed-data audio development comparison of original,
  enhanced and combined views. The combined view improved probability scoring
  but lost correct conclusions under strong noise and music, failing the frozen
  usefulness gate. Both replacements were rejected; the running detector and
  calibration are unchanged. Fresh multi-generator source metadata and rights
  were checked, but no fresh audio confirmation was scored. See the
  [complete result and limits](audio-noise-view-v4-evaluation.md).
- A distinct original-waveform quality calibration completed the same 720
  consumed inputs. It also failed the fixed strong-noise usefulness and
  condition-specific error gates, despite better pooled probability scoring.
  Rejected without changing serving behavior or scoring fresh confirmation
  audio. [V5 result](audio-quality-v5-evaluation.md).
- A frozen-backbone residual-head comparison completed all 720 inputs with
  unchanged raw-response replay and zero extraction failures. It improved
  common-assessed Brier from 0.057960 to 0.038711 but failed the fixed strong-noise
  usefulness gate. Rejected without scoring fresh confirmation or changing the
  serving calculation. [V6 result](audio-residual-v6-evaluation.md).
- Expanded the public source catalogue to audio: 120 original references,
  12 background recordings and 720 exact recipes/connected-fold assignments.
  Added an offline reconstruction helper and locally reproduced all 720 float
  payloads, WAV files and production PCM hashes. Original WAV header timestamps
  are explicit metadata; the first timestamp-mismatch attempt remains recorded
  locally. This enables outside replication; no outside review has occurred.
- Built and audited 48 known AI mouth-edit/non-AI controls from eight consumed
  CREMA-D actors, using permitted generator weights and replacement face
  preprocessing. All native pixels, masks, clocks and audio passed replay.
  The current sequence statistic localized 0/16 AI edits; the mouth-region
  prototype localized 1/16 and failed its frozen gates. Timing checks flagged
  none and remain contextual observations. Rejected without enabling a method
  or scoring fresh actors. Published source references, asset pins and
  generation/audit tooling; outside review remains outstanding.
  [Results and limits](mouth-sequence-v4-evaluation.md).
- Tested a distinct learned face-forgery temporal contrast on the same 48
  consumed mouth cases. A separate audit caught a cached-frame ordering error;
  the invalid run remains retained. After correcting and re-freezing only the
  input binding, all 2,016 native/crop/tensor checks and 48 baseline replays
  passed. The valid candidate localized 0/16 AI edits and failed its gates.
  Rejected without fresh actors or serving changes. Published all input hashes,
  raw logits and a standard-library window/denominator review tool. This enables
  outside arithmetic review but does not constitute it.
  [V5 result and failure record](mouth-artifact-v5-evaluation.md).
- Fitted a separate direct mouth-chip head on 16 consumed actors with four
  different point-selection actors, excluding all eight video-development
  actors. All 800 training frames and 2,016 video input bindings passed source
  replay. The head flagged two AI clips at the wrong range and three ordinary
  controls; it correctly localized 0/16 edits and was rejected under the frozen
  gates. Published a compact numerical fitting capsule and a CPU refit helper;
  local coefficients, normalization and point reproduced with zero differences.
  No fresh actors, customer detector or confidence changes. Outside review is
  still outstanding. [V6 results](mouth-trained-v6-evaluation.md).
- Added a second permitted mouth-rendering family with an isolated Mac runtime,
  strict checked weights, deterministic audio-conditioned generation and matched
  ordinary resampling/blur controls. All forty new clips were prepared. Source
  replay passed 2,740 frames, 800 masks, all 320 neural input bindings and a fixed
  sample of 32 bitwise neural-output replays. The unchanged current cue localized
  0/16 new AI edits and flagged one strong-blur control. Across two families:
  88 assessed correlated clips, 32 AI edits, zero correct locations, one control
  flag. This extends consumed-source development, not fresh confirmation or
  model admission. All source metadata, rights, recipe and review measurements
  are published; outside review remains pending. [V7 result](mouth-multifamily-v7-evaluation.md).

## Additional completed development on 8 November

- The [image context/region-point factorial](image-region-point-v15-result.md)
  passes the new mean arithmetic requirement and complete native/source replay,
  but still loses useful localization versus current; all four arms remain
  rejected. Its actual12-test Linux arithmetic receipt is separate from native
  image inference and outside human review.
- The [background-context video comparison](video-background-context-v25-result.md)
  completed 80 genuine calibration files,160 consumed diagnostic clips and 40
  ordinary source substitutions, with exact source clocks and 600 audited
  outcomes. It changes62 to 63 localized swaps, but appearance recalibration alone
  also reaches63; scene alerts22 to 20 still span11 of 20 components, and a new
  jump-cut alert violates a frozen rule. Primary rejected; no fresh reserve or
  serving promotion. All 18 focused tests and actual Linux arithmetic pass.
- The [broader provider search](provider-signal-broad-search-2026-11-08.md)
  initially covered 67 families/96 scoped records with 19 explicit gaps. Newly documented
  verification routes are distinct from qualified integrations; no external
  customer verifier was enabled.

Regional image localization, difficult audio, lip sync, general video, writing
and outside human participation still require work. These completed negative
comparisons narrow the methodology; they do not complete the improvement goal.

The [LTX public-pattern contract](ltx-marking-audio-v1-result.md) now completes
80 authored audio cases, 32 actual codec decodes, a full separate sample audit
and 20 actual Linux contract tests. Clean expected patterns work; all eight
noisy cases miss, and wrong payloads trigger generic presence. Keep it isolated,
without confidence or early stopping. The
[follow-up inventory](provider-signal-followup-2026-11-08.md) now covers 82
families/105 records with 28 review gaps, including further speech, mobile and
regional-provider searches. The separate
[lip-sync permission review](lip-sync-source-review-2026-11-08.md) admits no new
checkpoint. Serving defaults, rollback, voice calibration and GCP are unchanged.

The [audio-aware mouth comparison](mouth-audio-context-v14-result.md) now completes
all 272 audio tracks and 6,373 windows, with separate full audio/feature/mask/time
replay. It locates 31/32 edits versus 26/32 visual-only but doubles ordinary flags
from 12/56 to 24/56; rejected. The original short-control failure and 178 exact
persisted-prefix reproductions remain retained. New public NumPy-only arithmetic
review and an actual Linux run complement 20 local authored tests; outside human participation and fresh
confirmation remain outstanding. No serving or confidence promotion.

The [magnified image-crop comparison](image-iml-zoom-v16-result.md) completes all
384 candidate input variants and 96 actual ordinary-reference runs, with 1802
new neural forwards. Crop localization finds0/48 useful edits per format versus
9/48 native and6/48 JPEG current; both formats are rejected. The original strict
independent-mask audit failed at a rounding boundary and remains failed. A
separately published post-result full384-source/1920-grid sensitivity audit
reproduces canonical masks/counts, retains25/768 ungated metric discrepancies
and one changed genuine-only point, and confirms rejection under both numerical
paths. All25 authored tests and actual Linux continuous arithmetic pass; neither
clears that strict real-source failure. No fresh reserve or serving promotion.

The [new passage/mixed-audio source review](passage-and-mixture-method-review-2026-11-08.md)
pins selected boundary-learning and component-target code. Its audio checkpoint
is explicitly noncommercial and excluded; a compact independently calibrated
passage implementation and properly labeled speech/background evaluation remain
future work. Short model-free contributor commands are now prominent in the
[independent-testing guide](independent-testing.md). No outreach or outside human
review is claimed, and the full four-media improvement program remains active.
