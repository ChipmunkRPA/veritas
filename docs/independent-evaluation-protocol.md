# Local independent evaluation protocol

Protocol `local-audio-evaluation/1`, frozen 2026-11-01, before scoring or inspecting any new holdout response; the exact freeze timestamp and SHA-256 are recorded with the experiment. This is a prospective plan, not a statement that a model has passed. The manifest, this protocol and the scoring implementation must be hashed before inference. Decisions and any deviations must be recorded alongside the results.

## Purpose and scope

Compare the installed voice candidate with the existing voice detector on new, externally labelled recordings, and fit an **offline** calibration candidate. Compare clean speech with noise and music added under controlled conditions. Separately, run publicly labelled images, videos and texts through the application to identify practical failures.

“Independent” here means that this run's fitting, calibration and final evaluation data are separated before model responses are observed. It does **not** mean that overlap with pretrained detector data has been ruled out. The [Pella publisher card](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector) does not disclose its training corpus. The [existing voice detector card](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification) names an intermediate LibriSpeech-trained encoder and reports ASVspoof2019 evaluation. Thus LibriSpeech has known source-family overlap with the existing encoder; the exact recording overlap is unknown. A new calibration split cannot remove that limitation.

This first experiment is restricted to one four-second interval per source file, English speech, the selected publishers and a fixed background level. It does not establish performance over complete long recordings, all languages, all voice clones, arbitrary backgrounds or the soundtrack of a video. Existing application confidence and historical evidence remain unchanged unless a later, separately reviewed admission supports a specific scope.

## Data plan frozen before scores

The audio agent's selected plan contains 160 original speech files and 480 derived inputs. The original files, rather than windows or perturbations, determine the basic sample count.

| Split | Real source files | AI source files | Group separation |
| --- | ---: | ---: | --- |
| Fit | 16 LibriSpeech development speakers | 16 publisher-labelled Kokoro files | Six Kokoro voices, assigned before inference |
| Calibration | 24 other LibriSpeech development speakers | 24 publisher-labelled Kokoro files | Six different Kokoro voices |
| Final test | 40 LibriSpeech test speakers | 40 publisher-labelled Chatterbox files | Test speakers/voices withheld; Chatterbox withheld as a complete generator family |

Each real speaker supplies only one original file. Synthetic sources come from the publicly released `dictation-speech` material with explicit synthetic labels, generator, voice and transcript metadata. The implementation records the exact repository, immutable revision where available, original URL, source row key and licence evidence in the manifest; this document does not substitute for those per-file records. LibriSpeech's declared CC-BY-4.0 and the synthetic collection's declared CC-BY-SA-4.0 must be recorded from their publishers. Background clips are selected from MUSAN only where the individual clip's permission is explicitly CC0 or CC-BY; retain its attribution. Do not treat a corpus-level licence as blanket clearance of differently licensed clips.

Use an immutable seed and deterministic ordering to select source files and assign the twelve Kokoro voices to six fitting and six calibration voices. Before scoring, reject files shorter than four seconds, undecodable files, label conflicts and unresolvable provenance; write the exclusion reason. Keep all failures encountered after scoring begins in the evaluation denominators. Do not replace a difficult file because its response is inconvenient.

Check connected source identities before locking the manifest: source file, original/decoded hashes, speaker, synthetic voice, source URL, transcript and any known parent recording. Shared text or source lineage must not cross fit/calibration/test; all versions of a source remain together. Report the number of speakers, voices, source files and connected groups in every split. If a constraint leaves fewer eligible items than planned, report the actual counts and the shortfall. Do not silently weaken the split rule.

Fit and calibration share a generator family but have disjoint voices and source texts. The final test is a new-generator stress test, not a claim that all generator families have been independently evaluated. There is no calibration example from Chatterbox.

## Preprocessing and background conditions

Derive the first four seconds of each eligible source into 16 kHz mono PCM using a documented deterministic conversion. Preserve the original file separately and record native SHA-256, decoded PCM SHA-256, time/sample bounds and derived file SHA-256. All systems receive the same derived input before their own production preprocessing and speech gate.

For each original, derive exactly these conditions:

1. Clean four-second interval.
2. The same interval with one licensed noise clip mixed at 10 dB SNR.
3. The same interval with one licensed music clip mixed at 10 dB SNR.

Compute and record the SNR convention (speech/background RMS and treatment of silence), mixture gain, background source/time bounds, actual measured SNR and peak scaling. Preserve the relative SNR if peak limiting requires scaling; do not clip the mixture. Background recordings are disjoint across splits. A fixed background recording reused within one split is a nuisance factor, not another independent observation. Record such reuse, and do not claim uncertainty covers every possible background environment.

The existing detector keeps its original speech heuristic and aggregation path. The candidate keeps its independent speech activity gate and its released preprocessing. Record differences in assessment coverage. Do not bypass a model's gate on hard test cases. Sound-category responses, including a “speech synthesizer” category, are descriptive outputs and never ground-truth labels or extra evidence of a correct AI decision.

## Systems fixed before evaluation

All model revisions, weight hashes, scoring code, speech gate settings and input conversion are pinned. Raw model responses and final calibrated responses are separate results.

| System | Fitting | Calibration | Purpose |
| --- | --- | --- | --- |
| S0: existing service | No new fit | Existing frozen service calibration | What the current application does |
| S1: existing detector, recalibrated | Fixed single-detector response | Platt fit on the new calibration split only | Isolate the benefit of refreshing calibration |
| S2: new detector, calibrated | Fixed single-detector response | Platt fit on the same calibration split only | Compare the candidate fairly with S1 |
| S3: existing + new detector | Nonnegative monotone combination fitted on `fit` only | Platt fit on `calibration` only | Test whether the candidate adds useful information |

Any regularisation choice for S3 is either fixed in the scoring implementation before inference or selected by grouped cross-validation within `fit`. Do not select it on calibration or test results. A small fit split may fail a grouped fit; record that as a failed candidate, not permission to merge splits. No threshold, score polarity, missing-value rule or gate is selected from final-test labels.

Calibrators use the application's class-balanced 50:50 reference prior. Their outputs are estimates under that prior, not measured real-world success rates or probabilities under every service traffic mix. Balance each original source's total weight across its three conditions, so perturbations do not triple its effective weight. Use the same weighting convention for S1 and S2 and document S3's weighting. Each calibrated score must carry its system/calibration identity; never relabel the candidate's raw output using the old model's calibration.

Do not average in a missing model as a negative result. Define S3's handling of missing scores before evaluation, and report its assessment coverage. The primary operational comparison includes every test source and condition; an additional common-assessable intersection table is allowed but must retain the excluded counts.

## Metrics and uncertainty

Freeze both operating rules before test:

- Customer tendency: AI at calibrated probability at least 0.8; original at at most 0.2; otherwise inconclusive. Retain production applicability/conflict checks where evaluating S0's actual customer conclusion.
- A separate false-alarm operating point: threshold estimated only from calibration real sources for a nominal 5% false-positive target. Use a documented finite-sample rank rule. If calibration support cannot establish the threshold, report it as unestablished. Do not rename the nominal target as the realised test FPR.

Report for clean, noise and music separately and for the source-balanced combined view:

- Number of original files and independent source/speaker/voice groups; real and AI counts.
- Model coverage: assessed, no speech, not applicable, decoding error, failed and timed out.
- Raw-score ROC AUC and EER on the assessable cohort, with coverage alongside them. EER is a descriptive test metric, not a tuned deployment threshold.
- Calibrated Brier score, log loss and a reliability table. ECE must state the binning and counts; sparse-bin ECE is not proof of calibration.
- At the 80% rule: counts of AI, original and inconclusive calls; false AI calls among real files; AI files incorrectly cleared as original; recall; decision coverage; and error rate among decided cases, separately for both decision directions.
- Realised FPR and TPR at the calibration-selected 5% target threshold, if established.
- Paired differences S2 minus S1 and S3 minus S1, and the removal comparison S2 versus S3. S0 is a practical historical baseline; superiority over S0 alone does not establish a new-model benefit.

Use a paired grouped bootstrap, with an explicitly recorded seed and 2,000 replicates, resampling whole source/speaker/voice groups and carrying all derived conditions together. Keep a pairing across systems. Where repeated background clips induce additional dependence, state the conditional nature of the interval and avoid claiming an environment-population interval. Give 95% intervals for metric differences. Wilson intervals may accompany unperturbed independent source-level count rates, but do not apply binomial intervals to 480 dependent derived clips as if all were independent. Undefined or single-class bootstrap replicates and their number must be disclosed.

## Prospective admission and removal decision

This experiment's output is **offline-only by default**. It can reject an unsuitable candidate and identify a promising one; its 40 real/40 AI test originals and one held-out synthetic family cannot establish general production confidence. The repository's direct slice support uses at least 100 cases per class, and even zero errors on 40 independent real sources has a wide upper error bound. Additional background copies do not satisfy either shortfall. No result from this fixed small run alone authorises broad production promotion.

Evidence required for a future, scoped promotion is predeclared as follows:

1. Traceable labels, licence/provenance review, no cross-split source identities, deterministic replay, and a scoring path matching the service.
2. Adequately supported final-test classes and relevant recording/background strata, with multiple real sources and additional held-out synthetic families. Eligibility scope must be derived honestly from this coverage; one new-generator result cannot clear all video soundtracks or all languages.
3. Candidate benefit over the fairly recalibrated old-model baseline: lower source-balanced Brier and log loss, with paired 95% intervals supporting improvement. AUC/recall and decision coverage are reported to expose a gain obtained merely by abstaining more.
4. At the fixed 80% rule, no material degradation in either false-AI or false-original error. For promotion, the paired upper bound on the FPR increase must not exceed two percentage points in each supported background condition. Report the decided-case risk and both-direction count intervals rather than assuming 80% means 80% accuracy.
5. At the separate nominal 5% FPR operating point, realised real-file FPR must have a 95% upper bound no greater than 7.5% in each scope being admitted. Insufficient precision is “not established,” not a pass. Confirm AI recall improvement without using test to retune the threshold.
6. No inherited old calibration, slice map, bootstrap interval or benchmark claim for a newly fitted candidate. Promotion needs new versioned frozen artefacts with the evaluated scope and a source-bound admission record.

Only consider disabling the old detector if the candidate alone maintains the combined system's benefit: paired comparisons must rule out more than a one-point Brier increase and a two-point recall loss, without increasing false AI/original errors or sacrificing important coverage. If intervals remain too wide, retain the old detector. Disablement, if eventually justified, is reversible; preserve weights, original reports and old calibration artefacts for historical replay.

Failure or insufficient evidence leaves the new voice check as a separate uncalibrated inspection signal with zero fusion weight. Offline calibration files and every unsuccessful evaluation remain available for review. Do not alter a threshold to turn this run into a pass. Once final-test results are read, that test is consumed for selection; any revised method needs a new holdout for confirmation.

### Predeclared second-stage confirmation

The first stage is not a permanent bar to admission. If S2 or S3 improves the point estimates of both source-balanced Brier and log loss against S1, and no tested background condition shows a material error/coverage regression, a second independent confirmation stage is permitted. If the result shows clear harm, saturation without discrimination or no useful improvement, reject the candidate without expanding merely to search for a favourable result. An unresolved first-stage interval is not a pass.

Before scoring stage two, select a single candidate from the fixed systems, freeze its existing fitted coefficients, calibrator, operating thresholds, preprocessing and gates, and write a new source manifest with a fixed sample size. Neither models nor thresholds may be changed after stage-one results. Stage two uses entirely new source recordings, speakers/voices, source texts and background clips, with no source identity or derived-file connection to stage one. The confirmation set must include at least 100 real and 100 AI original sources, multiple independent real sources and at least two synthetic families not used in fitting or calibration. Report generator, speaker and background coverage explicitly; adequate counts of dependent variants cannot replace independent originals or missing families.

Evaluate the frozen candidate against the frozen S1 baseline and relevant ablations once on that set. Report stage two separately; do not pool the selected stage-one test into confirmatory confidence intervals or keep adding files until a threshold passes. Apply all admission criteria above to the stage-two result and only to the scope it supports. If the fixed confirmation fails, lacks precision, or suitable labelled sources cannot be obtained, record that outcome and retain the current service configuration. A successful, adequately supported confirmation can support a versioned local deployment and reversible old-model disablement, followed by actual API/evidence/export validation; it is not automatic broad deployment to untested modalities or languages.

## Runtime isolation and artefact safety

Write all candidate files under `backend/data/independent-evaluation/audio/`. Do not write trial fusers into `backend/data/fusion/` or `backend/app/engine/fusion/release/`.

The current loader automatically chooses compatible highest-semver fusers from its configured directories. Its compatibility check verifies schema, modality, nonempty `reviewed_by`, enabled detector identities and pinned weight hashes; it does **not** enforce the performance criteria above. The fitting function's default automated review flag is nonempty, so it can satisfy that loader check. An offline candidate must not become live merely by being saved. Never invent a human-review sidecar or repurpose the historical fitted model's validation fields.

The existing source files, analyses, signed reports and audit events are immutable baselines. New local test analyses may be added through the API. Keep public model-name redaction and existing access controls intact.

## Multimodal web-sample verification

The separate image/video/text set is an end-to-end smoke test, not a calibration or population-accuracy benchmark. Each sample needs a primary-source label, original source URL, acquisition URL, file SHA-256, source/group identity, declared generator where applicable and permission/provenance notes. “Looks obviously AI” is not a label. Include real controls, avoid detector publishers' hand-picked success examples and disclose unknown training overlap.

Preserve original samples. If a bounded excerpt or conversion is necessary, record the exact transformation and evaluate that derived input honestly. Report every attempted file, including errors, unassessable results, scope-limited conclusions and inconclusive outputs. Separate source/metadata declarations of AI use from content-detector evidence. A visual-only video conclusion is not a whole-video or audio conclusion. Save the source label, actual customer conclusion, percentage or its absence, relevant content-check outputs, provenance basis and local analysis link for each row.

Small web-sample counts are reported as counts with denominators, never as verified service-wide accuracy. A failure is evidence for further work; it must not be removed from the report or “fixed” by tuning on the same examples and then calling them an independent test.

## Pre-score amendment 1: background availability and conversion details

Effective edition `local-audio-evaluation/1.1`, recorded 2026-11-01T02:22:44Z, before audio scoring and before either evaluator inspected any holdout response. The prior edition's SHA-256 was `ced27ebfc9fe543ea981cdc7e0044a0e2f773785edfe4591add14d86b5f4e74d`; its plan is retained above so the change is visible. This amendment supersedes only the background-source restriction and clarifies the deterministic preprocessing. Sample counts, split assignment, systems, operating thresholds, metrics and admission criteria do not change.

The MUSAN music metadata service returned an HTTP 501 scan-size limit, so the three music backgrounds come instead from individual OpenGameArt works with explicit CC0 permissions, independently checked on their primary publisher pages:

| Split | Work and source | Author credited by publisher | Permission |
| --- | --- | --- | --- |
| Fit | [Regret — short emotional piano](https://opengameart.org/content/regret-short-emotional-piano), `piano_nostalgia.wav` | Wolfgang_ | CC0 |
| Calibration | [Can Be So Beautiful](https://opengameart.org/content/can-be-so-beautiful), `can_be_so_beautiful.wav` | cinameng; page credits James Gargette | CC0 |
| Test | [At the End of Hope](https://opengameart.org/content/at-the-end-of-hope), `at_the_end_of_hope_loop.wav` | Emma_MA | CC0 |

Noise backgrounds remain separate MUSAN/free-sound clips whose individual upstream licence records explicitly designate them public domain. Record that designation accurately; do not relabel a public-domain notice as a specific CC0 dedication. Retain original download URLs, file hashes, publisher permission evidence and attribution. No background clip is selected using detector responses. Each split uses its own noise and music file; intervals within a split share those source recordings, so uncertainty remains conditional on this small background selection.

After deterministic decoding/resampling to 16 kHz mono, normalize the first four-second source interval to RMS 0.1 (−20 dBFS). Set each background to 10 dB SNR using full-window RMS, including silence. Derive clean, noise and music conditions, then apply the same peak-avoidance gain to all three so their maximum absolute sample value is at most 0.98 without clipping. Save and verify the actual measured SNR, common gain, background interval and exact derived PCM/file hashes. This experiment therefore concerns normalized four-second speech; it does not validate arbitrary recording gain or longer-file aggregation.

Run both production adapters with their own original applicability, speech gates, window selection and aggregation. The four-second window/two-second hop configuration is retained: for an exactly four-second input the production window planner selects one covering window, not extra overlapping pseudo-observations. Preserve source sample bounds, unrounded detector responses and the six-decimal service-compatible responses used for fitting. No fitting or threshold decision may use additional precision selected after test results are observed.

If a later confirmation seeks admission for complete recordings, it must include the actual production multi-window aggregation across varied recording lengths and backgrounds, or explicitly restrict admission to the validated short-window scope. A calibration fitted here cannot silently become a confidence estimate for arbitrary-duration audio or video soundtracks.
