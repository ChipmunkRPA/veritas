# Detection improvement and independent evaluation

Started 5 November 2026. GCP deployment is deferred. This work improves detection
and the evidence supporting it before expanding hosting.

## Requirements and order

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

## First new image comparison: predeclared design

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

## Progress on 6 November

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
