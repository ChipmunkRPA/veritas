# Lip-sync preparation: connected sentences and preserved stream timing

The new study view retains all **48 audiovisual sources** and removes known
sentence overlap across its roles: 11 sentence types crossed the earlier roles;
none cross the new ones. All four historically observed sentence contexts,
covering 18 sources, are fitting-only. The earlier inventories and roles remain
unchanged. This prepares a better comparison; it is not a new detector result.

The [procedure](av-sentence-support-v104-method.md), implementation, authored
tests and source/history bindings were frozen and manually published as
`f2ea083`. Remote freeze blob `1ed6ebf2dfe72a2a8c8f2b5b339695bd4a775dd6`
was verified before preparation. The [aggregate result](evaluation-panels/av-sentence-support-result-v104.json)
binds the complete new private plan.

| New study role | Sources | Sentence contexts | Native 20-frame edit support |
| --- | ---: | ---: | ---: |
| Fitting | 18 | 4 historically observed | 18 |
| Confidence calibration | 10 | 2 | 9 |
| Selection | 10 | 3 | 10 |
| Confirmation | 10 | 3 | 10 |

Sentence, actor and known ancestry connections stay within one role. An exhaustive
bounded assignment balances the 30 remaining sources at ten per reserved role,
using only metadata counts and a fixed tie order. Thirty-six sources have a new
study role; their prior assignments are retained. One short source lacks the
24 common native pictures required for a 20-picture visual edit plus two context
pictures on each side. It remains in the inventory with an unavailable reason.
No padding, repetition or response-dependent replacement was used.

## Timing findings

Audio begins **25–27 ms before the first video picture**. The plan preserves
both declared source origins and the signed offset. It maps all **3,642 common
picture centers** to their rational source times and audio-relative times.
Before/during/after positions refer to actual native source pictures. Twenty
native pictures are not automatically twenty pictures at 25 Hz.

The native audio is 44.1 kHz, while its recorded frame timestamps have only
1 ms precision. At **129 picture centers**, two declared audio-frame intervals
overlap within that precision. None has a missing interval. Those ambiguous
memberships remain recorded; they are not dropped or silently resolved to a
sample-exact phase. A uniform native-sample lattice is also recorded, explicitly
conditional on contiguous sample order and pending native PCM verification.
Common support cannot extend beyond either declared or conditional sample extent.

`audio_conditioning_ready` remains false for every source. No waveform is decoded
by this preparation, and no resampler, generator or achieved phonetic alignment
has been validated. The next timing step must check native PCM/sample counts
and explicitly qualify the conditioning path before mouth generation.

## Local verification and limits

All 41 relevant tests passed, including 18 new authored cases for connected
roles, deterministic balance, source offsets, native context, short inputs and
quantized audio clocks. Preparation completed in 0.56 seconds, with maximum
observed self RSS 161,202,176 bytes and no bound crossed. These per-source
observations are not continuous or guaranteed peak measurements. Existing
neural-study limits remain unchanged; zero neural calls occurred.

A [separate read-only recorded-data check](evaluation-panels/av-sentence-support-recorded-check-v104.json)
verified the retained memberships, sample extents, edit support and every one of
the 3,642 rational time relationships. Additional assertions checked all source
actor/sentence labels and historical memberships against the locked original
filenames and historical manifests. These checks use the same saved metadata;
they are not an independent decoder, source-authorship adjudication, second
model implementation or outside human review.

The two-to-four sentence contexts per role are a small development panel, not
enough by themselves to establish general performance. Unknown historical-audio,
near-duplicate, identity and pretraining connections remain unknown. Generator
quality, actual lip-sync detection, matched ordinary controls, separate learned
calibration and fresh confirmation remain unfinished. No customer confidence,
model default, payment behavior or deployment changes from this preparation.

Source identities, sentence pairings, detailed clocks and individual fingerprints
remain private. Public materials contain code, aggregate counts and whole-artifact
commitments. No raw media, model weights, workflow execution or outreach was
added; the separate full face-swap input construction continues locally.
