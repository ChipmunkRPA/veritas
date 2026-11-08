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

On7 November the user directed region/check/context-specific calibration and a
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

## Progress on 7 November

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
