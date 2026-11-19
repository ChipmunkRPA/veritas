# Local nonlinguistic audio features under strong competing speech

This pilot added **zero correctly located noisy edits** and **six clean genuine-file
alerts** across one held host. The research gate failed. The current models,
operating points, confidence calculation and serving behavior remain unchanged.

## Frozen sources and comparison

Nine declared readers and books were assigned to separate fitting, point-selection
and held roles before participant inference. Each role contains two hosts and one
shared ordinary different-reader donor. These form **three connected source
components, including one held component with two held hosts**. Shared donors,
carriers and repeated conditions remain correlated observations.

Fifteen original utterances were qualified from a newly acquired, fully
checksum-verified official MLS shard. The retained source-role search inspected
645 metadata files and conservatively excluded exact reader mentions and the
previous development-shard readers. Historical aliases and model-pretraining
independence remain unresolved. Source audio has Opus ancestry. The twelve-second
16 kHz PCM16 fixtures preserve the explicitly qualified primary decode and do not
establish a lossless-original claim.

Three retained AMI captures provide fitting, point-selection and held interference.
Their twelve annotated participants are disjoint across roles. Physical room
independence is unestablished. Backgrounds have prior project exposure. Before
inference, annotation-only qualification fixed an assignment with foreign speech
throughout every unchanged 960 ms target. Noise is nominally 10 dB louder than the
untouched foreground at that target. This trial tests competing speech, not traffic.

Each host has two licensed stock-generator families, Piper and Kitten, each tested
with direct and fixed Opus-roundtrip donors. Ordinary controls cover untouched
speech, same- and different-reader splices, two resampling roundtrips and a local
Opus roundtrip. Shared fades, local RMS matching, gain and background preserve every
sample outside each edit. The **126 deliveries contain 48 synthetic, 72 ordinary
and six isolated backgrounds**. Twelve base host/family events repeat across two
carriers and two conditions. Wording and voice are unaligned with the host.

## Representation, fit and point separation

[Localized nonsemantic-representation research](https://www.isca-archive.org/interspeech_2025/das25_interspeech.html)
motivated an alternative to the existing speech representation. That paper uses
TRILL/TRILLsson. This pilot uses the already retained PANNs Cnn10 AudioSet encoder
and supplies no reproduction or transferred accuracy claim. Recent
[neural-codec research](https://arxiv.org/abs/2610.07216) motivated explicit ordinary
codec controls; codec artifacts alone cannot establish generation.

The frozen encoder reads original 200 ms crops through its fixed 16→32 kHz
frontend, without learned enhancement or reconstruction. A fixed random projection
and local context differences produce 64 features. Fit-only normalization and a
ridge-regularized 65-parameter logistic head use 2,138 eligible fitting cells.
The point uses the second-highest ordinary point-host maximum, floored by isolated
background responses. It allows at most one point-host group to exceed the point.
No held feature enters fitting or point selection.

Three adjacent admitted cells above that point produce a 600 ms inspection range.
Its context stays within the original recording; the first and final 600 ms remain
outside this candidate's context-supported schedule. The incumbent includes the
actual confirmed two-second check and unchanged adjacent 750 ms rule. The augmented
arm retains every incumbent interval. A useful interval covers at least half the
known edit and has temporal intersection over union of at least 0.30.

## Held outcomes

| Held condition | Useful incumbent | Useful candidate | Useful augmented | Ordinary flags incumbent → augmented | Isolated background flags |
|---|---:|---:|---:|---:|---:|
| Clean | 6/8 | 4/8 | 6/8 | 0/12 → 6/12 | — |
| −10 dB competing speech | 0/8 | 0/8 | 0/8 | 4/12 → 4/12 | 0/2 → 0/2 |

Each eight-delivery synthetic denominator represents four base generated events
and two carriers. The augmented arm leaves two clean and all eight noisy deliveries
without a useful location. It adds two clean synthetic alerts confined to the wrong
time. All six new genuine flags concern different ordinary variants of one host,
including its untouched recording. Their total new highlighted time is 3.6 seconds.

Every file had some mixed-track speech assessed, and no known edit was uncovered
under the combined checks. Mixed-speech admission does not establish that the target
voice was isolated or intelligible. Zero noisy recovery therefore cannot be treated
as evidence of originality. Including every incumbent confirming support interval
adds no useful location in any of the six role/condition strata.

The frozen continuation criteria required new noisy events across both held hosts
and families with bounded additional ordinary alerts. Both the gain requirements
and the overall ordinary-alert limit failed. A passing small-pilot gate would have
permitted larger confirmation only. This closure authorizes no model adoption.

## Verification, rights and public recount

The feature stage used 1,693 forwards including an exact repeat in 12.78 seconds.
The unchanged baseline used 1,246 forwards including an exact repeat in 500.75
seconds. Independent source and renderer reviews verified all fifteen original
rows and 24,192,000 delivered samples, including exact non-edit equality for 108
edited copies, all eighteen Opus roundtrips and the fixed clocks. Independent
feature/point review reconstructed 7,560 cell bindings, fit normalization, the
convex objective, point and all selectors. The separate outcome review reproduced
all 126 paired decisions and 189 detailed count groups, including 2,898 brief and
2,480 admitted two-second source-window bindings.

Two generator startup attempts failed before fixture closure because the retained
speech frontend needed its C-level data directory set. Failed attempts and four
successful preliminary Piper calls were preserved. The corrected closed render
used fourteen successful generation calls including both family repeats, and
verified existing partial donor bytes unchanged. No detector response changed
sources, controls, clocks, models or rules.

PANNs weights retain [CC BY 4.0 publisher terms](https://zenodo.org/records/3987831)
and [MIT code terms](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE.MIT).
Sources retain [MLS CC BY 4.0](https://www.openslr.org/94/) and
[AMI CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml). The
[earlier stock-source audit](audio-stock-voice-transfer-2026-11-13.md) documents the
Piper and Kitten grants and local generator-runtime scope. No source recordings,
generated speech or model weights are distributed with this report.

The [count-only receipt](evaluation-panels/audio-nonlinguistic-local-strong-noise-2026-11-19.json)
omits identities, recording fingerprints, source paths and per-case offsets,
feature/response vectors, learned coefficients and numerical operating point. The
standard-library reviewer checks its pinned bytes, paired transitions, denominators,
event/host counts and gates, and rejects fifteen authored corruptions. Full native
and model replay requires the retained private custody materials. The limited
source components and reused backgrounds provide no population accuracy estimate.

```sh
python3 -I -S backend/scripts/review_audio_nonlinguistic_local_strong_noise.py --self-test
```
