# Original-sample enrollment reliability: genuine-only feasibility

The original-audio-only gate retained clean speech and discarded too much genuine
foreground under strong noise. This genuine-only feasibility trial measures whether
the source-conditioned check can tell if the enrolled reader is present.

Three previously unattempted source-qualification slots were sealed. One
development and one held target qualified. The second held target failed fixed
native duration, chapter, source and signal requirements and was not substituted.
A separate model-free attempt removed an inherited unused fourth splice-donor
requirement and still admitted only the same two targets. The one-held-source
scope was fixed before encoding. The original two-held-source goal remains unmet.

Each target has one twelve-second genuine host and two different four-second
genuine references. Reused role-specific readers supply wrong references and
ordinary nuisance speech. Retained traffic and AMI captures supply −10 dB noise.
The eighteen inputs include assigned genuine foreground, other genuine speech and
isolated background, each clean and with both noise conditions.

The reviewed ECAPA encoder reads exact original 750 ms samples every 500 ms. Two
reference similarities and a wrong-reference margin supply the candidate signal.
It performs no separation, reconstruction, filtering, waveform normalization,
alignment or padding. Known source components support offline grading only. The
point is the maximum likeness on all 138 development target-absent windows, locked
before held encoding. The candidate also requires a positive reference margin.

A measurable target window contains known foreground power above 1e-5 in at least
half its complete 20 ms blocks. The frozen criteria require at least 80% retention
and at most 10% combined absent-window admission in every held condition.

| One held source | Current speech gate | Candidate retained target | Other genuine admitted | Isolated background admitted |
|---|---:|---:|---:|---:|
| Clean | 16/16 | 16/16 | 0/23 | 0/23 |
| −10 dB traffic | 0/16 | 3/16 | 0/23 | 0/23 |
| −10 dB competing speech | 16/16 | 0/16 | 0/23 | 1/23 |

The three traffic windows are genuine coverage gains. Under competing speech the
candidate would discard all sixteen measurable foreground windows already admitted
by the current gate. Both noisy retention criteria fail.

A **separate post-hoc diagnostic** chose the most optimistic held threshold under
the same absent-window budget. It retained at most 10/16 traffic windows with 4/46
absent admissions and 3/16 competing-speech windows with 3/46 absent admissions.
A second grouped-rank implementation verified those bounds. These use held labels
and are not prospective results. No new point or inference was adopted.

The run took 3.79 seconds and 378 encoder calls including an exact repeat. Independent
direct-encoder and speech-gate replay matched 414 original windows, 1,656 cosine
comparisons and 4,968,000 window sample values. Native review verified twelve
original utterances including nuisances, 1,280,000 original sample values and all
3,456,000 delivered additive sample values. Both runs made zero separator,
synthesis or forensic-classifier calls.

**The overlapping windows come from one held foreground reader and recording.**
The repeated noise conditions share that foreground, and their nuisance/background
sources are reused. These counts do not represent independent people or establish
population accuracy. The readers and corpus have historical project exposure;
encoder training independence and distinct physical AMI rooms are unestablished.
The genuine-only trial does not establish synthetic preservation or detection gain.

The [prior component audit](audio-learned-enrollment-genuine-preflight-2026-11-19.md)
documents the extraction checkpoint and embedded encoder's reviewed terms. Genuine
sources retain [LibriSpeech CC BY 4.0](https://www.openslr.org/12/), meeting sources
retain [AMI CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml), and the
two retained traffic recordings were qualified under CC0. No weights or media are
included in this publication.

The companion count-only receipt omits participant identities, recordings,
recording hashes, per-source clocks, likeness scores, operating-point value and
feature or response vectors. The standard-library reviewer recounts denominators,
admission transitions and frozen gates.
It checks the released post-hoc count bounds; private response ledgers are required
to verify threshold optimality. Exact native and model replay completed locally
and requires the retained private custody package. The candidate stays experimental.

```sh
python3 -B -I -S backend/scripts/review_audio_original_enrollment_admission.py \
  docs/evaluation-panels/audio-original-enrollment-admission-2026-11-19.json
```
