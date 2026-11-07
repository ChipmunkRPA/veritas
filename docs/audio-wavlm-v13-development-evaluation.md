# Retain the current audio detector after overlap failures

6 November 2026. **Reject both frozen candidates without scoring reserved
confirmation.** Combining the new features with the current score improves
ordinary noise and music results. Overlapping speech still loses too many AI
conclusions, and the full comparison fails three unchanged requirements. The
feature-only classifier performs substantially worse. Keep the serving detector,
its calibration, rollback and overall confidence unchanged.

## Fixed experiment and complete outcomes

The [prospective protocol](audio-wavlm-v13-protocol.md), 69 implementation files,
runtime, source roles and weight digests were published as `45e8ff9` before any
project speech forward. The new 94.4-million-parameter backbone stays frozen.
Both small logistic classifiers learn only from fitting sources, receive their
own independent calibration and retain the original sixteen adoption gates.
The feature-only arm retains the existing assessment scope and raw stability
safety check; its name does not imply an independent complete service.

All 4,248 source extractions completed: 4,186 assessed, 62 unassessed, zero
extraction failures. Roles remain 1,926 fitting / 873 calibration / 1,449
development inputs. Development has 161 originals in 106 connected groups,
each with nine correlated acoustic variants. It is consumed development, not
fresh testing or 1,449 independent people. Unknown speaker/pretraining overlap
remains unverified. All 1,080 reserved confirmation inputs remain unscored.

| Measurement | Current service | New features only | Features plus current score |
| --- | ---: | ---: | ---: |
| Correct AI conclusions / 756 AI inputs | 648 | 30 | 522 |
| Correct original conclusions / 693 genuine inputs | 356 | 28 | 437 |
| Genuine inputs incorrectly called AI | 58 | 0 | 4 |
| AI inputs incorrectly called original | 0 | 4 | 0 |
| AI without an AI conclusion, including abstentions | 108 | 726 | 234 |
| Inconclusive / 1,449 planned inputs | 387 | 1,387 | 486 |
| Total correct directional conclusions | 1,004 | 58 | 959 |
| Inconclusive rate | 26.71% | 95.72% | 33.54% |
| Balanced Brier error, lower is better | 0.094859 | 0.170619 | 0.072194 |
| Balanced log loss, lower is better | 0.317112 | 0.524796 | 0.243317 |

The combined arm gains 81 correct original conclusions but loses 126 correct
AI conclusions. Its 54 fewer false AI calls accompany 99 more inconclusive
results and 45 fewer total correct conclusions. It passes 13 of 16 gates:
per-generator AI retention, background-speech usefulness and background-speech
class retention fail. The feature-only arm passes five gates and fails eleven.
Fewer false alarms through widespread abstention do not establish better detection.

The combined Brier difference versus service is -0.022665, connected-group
bootstrap 95% interval [-0.029246, -0.016489]. These probability-error estimates
condition on consumed development, fixed fitting/calibration and selected
backgrounds; they exclude fitting and background-population uncertainty. They
are not per-upload confidence intervals or cloud-service accuracy claims.

## Where the useful gains and losses occur

Each row has 161 originals. Entries show correct directional conclusions, with
inconclusive counts in parentheses. Incorrect calls account for the remainder.

| Condition | Current service | Features only | Combined |
| --- | ---: | ---: | ---: |
| Clean | 154 (7) | 12 (149) | 154 (7) |
| Noise, 0 dB | 97 (62) | 5 (155) | 111 (48) |
| Noise, 5 dB | 112 (49) | 7 (153) | 120 (41) |
| Noise, 15 dB | 139 (22) | 10 (151) | 144 (17) |
| Music, 5 dB | 95 (66) | 8 (153) | 108 (53) |
| Music, 15 dB | 141 (20) | 8 (153) | 150 (11) |
| Background speech, 0 dB | 75 (75) | 2 (158) | 31 (130) |
| Background speech, 5 dB | 93 (50) | 2 (158) | 44 (117) |
| Background speech, 15 dB | 98 (36) | 4 (157) | 97 (62) |

Noise at 0 dB gains 14 correct conclusions with the same two false AI calls.
Music at 5 dB gains 13 correct conclusions with no false calls in either method.
Clean conclusions are preserved. These are actual conditional gains on these
consumed sources, not evidence for a selective deployment chosen after results.

Across background-speech variants, correct AI falls 234 to 97 while correct
original rises 32 to 75. False AI falls 56 to 2, but inconclusive rises 161 to
309. Both interference levels where speech is equal or relatively loud lose
substantial useful AI detection. The experiment identifies the failing condition;
it does not establish a physical cause for each individual error.

The explicitly [post-result fixed-probability ceiling](evaluation-panels/audio-v13-probability-ceiling.json)
shows that removing guards alone cannot satisfy the original retention gates.
Even that hypothetical upper bound permits only 611 AI conclusions overall and
174 with background speech, versus service 648 and 234. Four generator families
remain below their unchanged allowed loss. This diagnostic changes no operating
point, calibration, decision policy or adoption rule.

## Verification and the retained failed attempt

The first comparison stopped at input 34 because the CLI and original-model
loader both initialized PyTorch inter-op threads. Its 33 successful records and
failed record remain intact. A minimal reproduction confirmed the initialization
error; the same case passed through the unchanged callable functions. The
[published execution supplement](evaluation-panels/audio-v13-thread-ownership-supplement.json)
binds the dedicated launcher before continuing. No frozen implementation,
feature, fitted object, calibration, filter, stability check or gate changed.

The new journal completed all 1,449 inputs with zero execution failures, and its
first 33 complete records match the original successful prefix exactly. This
does not erase the first attempt's operational failure. All 593 native replays
passed: the union of 99 predeclared source cells and every one of 511 required
filtered probes. Original status/raw/vector/window parity and new source/probe
feature bytes match. A separate arithmetic review reconstructed normalization,
calibration ranks, all probabilities/stability decisions, counts and statistics;
both fitting-only refits reproduced their complete original objects exactly.
The declared statistical/bootstrap/gate implementation and refit learner remain
shared. These are local checks, not outside human review.

## Public reproduction

The [aggregate result](audio-wavlm-v13-development-evaluation.json) retains all
condition/generator counts, uncertainty, gates and execution limits. The complete
packet is `docs/evaluation-panels/audio-v13-development-review.json.gz`, accompanied
by five role-separated feature parts. Lossless float32 transport reconstructs
the original complete feature receipt byte-for-byte; float64 parameters are
unchanged. The package references earlier public source/raw-vector resources and
includes all outcomes, filtered probes, fitted/calibrated objects and receipts.
It excludes raw recordings, transcripts, customer files, provider identities,
private paths and full pretrained checkpoints. Source/model terms remain attached;
derived speech features are not asserted to be anonymous.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.publish_audio_wavlm_v13 \
  review docs/evaluation-panels/audio-v13-development-review.json.gz \
  /tmp/new-audio-v13-review.json --refit
PYTHONPATH=backend backend/.venv/bin/python -m scripts.publish_audio_wavlm_v13 \
  tamper docs/evaluation-panels/audio-v13-development-review.json.gz \
  /tmp/new-audio-v13-tamper.json
```

Empty-cache public-only replay reproduced the original feature receipt, both
complete fitted objects and every outcome/statistic. Thirteen transport tests
passed. Twelve actual-packet alterations were refused after the unchanged packet
passed, including coherently changed feature/probe payloads and transfer hashes.
Numerical replay
requires numerical dependencies and the referenced public files, but no model
downloads, source audio or private manifest. Actual outside reviewers remain zero.

## Next work

Investigate temporal or speaker-resolved information rather than weaken guards
to rescue this result. Any source separation must be tested for artifacts on
genuine single-voice, overlapping-voice, noise and music controls, with its own
calibration and verified commercial permissions. Known condition labels cannot
be used as an inference-time routing oracle. A new method needs a new prospective
freeze; the rejected result and original confirmation reserve remain unchanged.
Image/video/writing improvements and actual outside testing remain unfinished.
GCP stays last.
