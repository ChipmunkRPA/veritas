# Full audio representation: better than the projection control, insufficient for replacement

6 November 2026. **Rejected under four of sixteen unchanged development gates.**
Full-vector fitting recovers useful conclusions compared with the rejected PCA16
control, including clean genuine speech. Against the current service, generated
speech retention and difficult-background usefulness still worsen. The candidate
remains disabled; no reserved confirmation was scored.

## Frozen method and sources

The [prospective protocol](audio-full-representation-v11-protocol.md) and
[execution freeze](audio-full-representation-v11-execution.md) were published
and remote-verified as `e2c1877` before project fitting or new neural forwards.
Retain all 1,024 frozen head-orthogonal coordinates, with fitting-only diagonal
normalization and a total variance budget at most sixteen. The weighted objective,
target smoothing, regularization, nonnegative raw slope, independent calibration
and sixteen gates are unchanged from v9. The completed PCA16 object is a fixed
representation control, not a new fit. No new pretrained model or dependency.
The existing commercial-permission checkpoint checks remain in force.

Same 4,248 consumed inputs: 1,926 fitting, 873 calibration, 1,449 development,
from 214/97/161 originals in 169/82/106 connected groups. Nine conditions per
original are correlated variants, not nine independent recordings or users.
Known genuine/cloned parent speakers stay together. Unknown biometric and
pretrained-corpus overlap remain unverified. This is reused development, not
a fresh accuracy study or outside human validation.

The complete feature matrix remains 4,186 assessed and 62 unassessed. Development
completed all 1,449 planned inputs: 1,420 assessed, 29 unassessed, zero failed.
No source, role or condition was substituted. All 1,080 reserved confirmation
inputs remain fixed and unscored. Unassessed inputs remain inconclusive.

## Complete outcome

| Measurement | Current service | Fixed PCA16 control | Full vector |
|---|---:|---:|---:|
| Correct AI conclusions / 756 AI inputs | 648 | 574 | 544 |
| Correct original conclusions / 693 genuine inputs | 356 | 298 | 400 |
| Genuine inputs incorrectly called AI | 58 | 3 | 2 |
| AI inputs incorrectly called original | 0 | 1 | 0 |
| AI without an AI conclusion, including abstentions | 108 | 182 | 212 |
| Inconclusive / 1,449 planned inputs | 387 | 573 | 503 |
| Total correct directional conclusions | 1,004 | 872 | 944 |
| Inconclusive rate | 26.71% | 39.54% | 34.71% |
| Balanced Brier error, lower is better | 0.094859 | 0.059157 | 0.062859 |
| Balanced log loss, lower is better | 0.317112 | 0.201372 | 0.212161 |

Against PCA16, full-vector fitting gains 102 correct original conclusions but
loses 30 correct AI conclusions: net 72 more correct conclusions, 70 fewer
abstentions and two fewer false calls. Its probability errors are slightly worse
than that control. This supports a limited representation finding, not a
universally better detector.

Against service, false AI falls 58 → 2 and correct original rises 356 → 400,
but correct AI falls 648 → 544. Net correct conclusions fall by 60 and abstentions
rise by 116. Lower false alarms do not override the unchanged usefulness gates.
Twelve gates pass; per-generator AI retention, strong-condition usefulness,
background-speech usefulness and background-speech class retention fail. Clean
retention now passes. No acceptance rule changed after observing results.

Brier difference versus service is -0.032000, connected-group bootstrap 95%
interval [-0.042039, -0.022515]. Log-loss difference is -0.104951, interval
[-0.133181, -0.076796]. These are conditional on reused development sources,
fixed fitting/calibration and selected backgrounds. They exclude fitting and
background-population uncertainty and are not customer confidence intervals.

## Background interference

Each individual condition has the same 161 originals. Values are correct
conclusions followed by inconclusive results in parentheses.

| Condition | Current service | Fixed PCA16 control | Full vector |
|---|---:|---:|---:|
| Clean | 154 (7) | 147 (14) | 154 (7) |
| Noise, 0 dB | 97 (62) | 87 (74) | 90 (71) |
| Noise, 5 dB | 112 (49) | 98 (61) | 105 (56) |
| Noise, 15 dB | 139 (22) | 120 (41) | 136 (25) |
| Music, 5 dB | 95 (66) | 75 (86) | 85 (76) |
| Music, 15 dB | 141 (20) | 112 (49) | 135 (26) |
| Background speech, 0 dB | 75 (75) | 51 (110) | 56 (105) |
| Background speech, 5 dB | 93 (50) | 79 (81) | 77 (83) |
| Background speech, 15 dB | 98 (36) | 103 (57) | 106 (54) |

Full-vector fitting recovers the seven clean genuine conclusions lost by PCA16
and improves over that control in eight of nine conditions, but not background
speech 5 dB. Against service, overlapping-speech correct conclusions fall 266 →
239: correct AI falls 234 → 177 while correct original rises 32 → 62. False AI
falls 56 → 2; abstention rises 161 → 242. Equal-level overlap remains difficult.
Background speech and acoustic context tags do not establish foreground AI origin.
Both wrong calls occur in genuine background-speech mixtures; no AI input is
called original. The [complete aggregate](audio-full-representation-v11-development-evaluation.json)
retains every generator, condition, background stratum, uncertainty and gate.

## Verification and public reproduction

Fitting-only CPU refit reproduced the complete original fitted object exactly:
all 1,026 coefficients, head direction, centers and scales had zero differences.
All calibration ranks, cached-service verdicts, mapped probabilities, both
raw/mapped stability checks, operational counts and summary gates were rebuilt.
There were 427 required filtered probes: 401 exact reuses from the pinned
424-probe v9/v10 union and 26 new probes.

Re-ran all 99 predetermined native cells plus every 26 new-probe case: 125 unique
cases, zero mismatches. This includes actual decoding/PCM, speech activity/windows,
raw and frozen-vector outputs, new mapped original scores, and required probe
vectors/scores. Extra cases audit provenance, not a new accuracy panel.

The [public capsule](evaluation-panels/audio-v11-development-review.json.gz) is
845,203 bytes, SHA-256
`5b2c2d06e785d60ed754adc175b77a8de57161a67165df164a497f0b3876ba30`.
It preserves exact original small fit/head JSON, every sanitized outcome/new
probe vector, summaries and audit receipts. Existing original/cached probe vectors
are referenced instead of duplicated. No source audio, transcripts, raw provider
identities, private/customer files, local paths or full pretrained checkpoints.
Derived-data terms remain separate from the source-code license.

With the project's numerical dependencies installed, from the repository root:

```sh
PYTHONPATH=backend OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python backend/scripts/publish_audio_full_results_v11.py review \
  docs/evaluation-panels/audio-v11-development-review.json.gz \
  /tmp/new-audio-v11-result-review.json --project . --refit
```

Use a new output path. Keep referenced v9/v10 public capsules beside the new
packet. Public-only refit passed with an empty model-cache location: no private
manifest, source audio or pretrained download. This checks native receipts without
rerunning the neural model or establishing outside review. Original fit/head bytes
reproduce exactly; sanitized outcome/summary bytes differ from lineage originals.

All nine deliberate actual-capsule alterations were refused: changed coefficient,
source group, probability, cached probe PCM, new probe vector, complete denominator,
native coverage, unassessed input called original and embedded private path.
All 127 selected regression tests passed. Tests verify audit/implementation
behavior, not detection accuracy.

## Next work

The [fixed-probability ceiling](evaluation-panels/audio-v11-guard-only-bound.json)
shows that removing every class/stability veto still cannot restore the required
AI retention for six families: Chatterbox, ElevenLabs, NaturalSpeech2, PflowTTS,
StyleTTS2 and Fleurs Azure. This is post-result diagnosis, not a selection rule
or permission to weaken uncertainty handling.

Full use of the final pooled vector helped genuine-speech coverage but did not
solve masked synthetic speech. Next prospectively test fixed earlier/middle/later
network features, preserving original waveforms and the backbone. Earlier layers
may retain acoustic details suppressed by the fixed layer mixture; this remains
a hypothesis. Freeze layer selection/fitting before any new forward, retain
service and full-vector controls, source roles, independent calibration and all
adoption gates. Short-window development cannot establish whole-recording or
whole-video confidence.

Image/video/writing improvements and real outside human testing remain open.
The app, USD9.99 prepayment and authentication remain unchanged. GCP stays deferred.
