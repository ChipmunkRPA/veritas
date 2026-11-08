# Fresh speech and strong-background watermark qualification

The publicly frozen [v2 protocol](public-audioseal-speech-v2-protocol.md) completed
all **576 cases, 32 generator forwards and 581 detector forwards** locally.
Its code/assets/policy were published at
`1c6661e3fc53e78e695521868fcc2982900d4606` before model construction or inference.
The native point was not changed after seeing results. The
[aggregate result and execution receipts](evaluation-panels/public-audioseal-speech-v2-result.json)
retain every condition, miss and false response.

**Decision: research only.** This check does not qualify for automatic early
stopping, AI-origin confidence or a serving default. It detects a public
watermark family that can deliberately be added to genuine speech. It does
not authenticate a provider or establish AI generation, even when it detects
the mark. Existing voice screening and its calibration remain unchanged.

## Cohort and conditions

Exactly32 LibriSpeech speakers not found among423 known canonical prior project
speaker identities; each contributes an eight-second original. The source pool
is the first eight pinned train.clean.360 shards. Sources are English read
speech, not representative conversations or a multilingual population. External
pretraining overlap is unknown. A separate zero-overlap check includes known
background speakers.

Three states: no added mark, full eight-second mark, and a deliberate mark only
at source2–6seconds. Six conditions use the same32 voices. Background noise,
music and other speech are reused project sources, with47 distinct selected
originals (19noise,13music,15speech) from the66-file licensed pool. White noise
is deterministic. Each background is as loud as the unmarked foreground by
RMS (0dB); the realization/gain is matched across mark states. One shared
clipping-safe gain applies per speaker. The separate MP3 condition is64kbps
encoding of clean gained speech, not MP3 plus background.

The576 cases are **not576 independent speakers**. No test contains a production
provider's authenticated watermark. The encoder deliberately marks genuine
recordings; these are not synthetic-voice classification labels.

## False responses, missed marks and localization

Every count below has32 sources in its denominator. A detected case has at
least one selected250ms bin. A miss selects none; finding one bin does not
mean the entire marked interval was recovered. Interval IoU compares selected
time against the exact deliberate mark. Perfect overlap is1.

| Condition | No-added-mark controls flagged | Full mark missed | Partial mark missed | Mean partial interval IoU | Mean partial time selected outside mark |
|---|---:|---:|---:|---:|---:|
| Clean speech | 1/32 | 0/32 | 0/32 | .9104 | .0234s |
| White noise0dB | 0/32 | 29/32 | 29/32 | .0059 | 0s |
| Environmental noise0dB | 1/32 | 11/32 | 15/32 | .2262 | .0156s |
| Music0dB | 2/32 | 5/32 | 10/32 | .2308 | .0469s |
| Other speech0dB | 0/32 | 1/32 | 3/32 | .3288 | .0156s |
| MP3,64kbps | 0/32 | 0/32 | 0/32 | .9062 | 0s |

Clean and compressed deliberately marked speech were detected consistently in
this panel, with useful partial localization. Strong backgrounds substantially
reduced sensitivity and recovered marked time; white noise was particularly
severe. Other-speech mixtures often yielded some detection while recovering
only a small part of the marked span. No-added-mark controls also produced
false responses. These outcomes contradict using this fixed readout as a
universal definitive origin gate or treating absence as originality.

Small counts leave wide uncertainty. Wilson95 intervals for no-added-mark
control flag rates are0.55–15.74% for1/32 and1.73–20.15% for2/32. Even0/32
has an upper bound of10.72%. These intervals do not account for every source
or background dependency and are not customer accuracy guarantees. All18
condition/state cells and full/partial coverage figures are in the aggregate.

## Separate review and retained preparation stops

A separate model-free process verified576 native inputs,18,432 bins, all joined
source intervals, matched background/partial geometry, original source bindings,
known-mark overlap and96 bit-identical codec re-decodes. Independent bin means
had maximum difference0.0; every aggregate denominator matched. This review
performed zero neural forwards or provider requests. It is code/arithmetic
review; **outside human reviewers remain0**.

Native execution completed in101.87seconds, sampled process-tree RSS lower
bound889,176,064bytes. Review completed in18.68seconds at499,154,944bytes.
Neither activated a resource guard. Native/review recorded2/1 transient process
sampling errors respectively; these measurements are not guaranteed peak
memory totals. Both remained bounded to900seconds,6GiB sampled RSS and at
least1536MiB available RAM/free disk.

The first source preparation stopped before inference because signed16 byte
hashes were compared with stored normalized-float32 PCM hashes. The first
flagged source had identical sample values under both decoding paths. The
32 collected original foregrounds and their inputs were retained and rechecked;
background verification now uses the actual production decoder mapping. No
foreground or detector point was replaced in response to model outputs. Earlier
rejected short candidate bytes were not retained, which limits exact replay of
initial duration eligibility. A separate monitor-access stop occurred before
source work. These attempts and successful recovery profiles remain recorded
in the aggregate and private journals.

## Reproduction and next work

The acquisition helper was actually run: seven pinned assets,35 source files
and80 dependency files verified, two small publisher notices downloaded, zero
package installations/source imports/checkpoint loads. The assets are isolated
copies; previous studies and the serving environment are unchanged. Use the
[asset lock](../backend/scripts/public_audioseal_assets_v2.json) and public
publisher sources rather than unrestricted installers/loaders.

```sh
PYTHONPATH=backend python -m scripts.acquire_public_audioseal_v2 /path/to/new-assets --download
PYTHONPATH=backend python -m scripts.acquire_public_audioseal_v2 /path/to/new-assets --verify-only
```

The new preparation, freeze/run and separate review commands are public. Exact
study replay also requires the retained private source plan/originals and
background plan; those are not included in this public release. Raw speech,
per-case pairings and per-sample responses remain local. An authorized reviewer
can inspect those against the published source-plan and case-journal checksums.
No outside person has done so yet. The code can also support a separately
frozen panel assembled by a reviewer under appropriate source permissions.

Twenty-two model-free tests and Ruff passed. There is no new serving model or
confidence calibration. Further work needs additional source domains, multiple
mark instances/keys and genuinely compatible provider positives. Any proposed
noise treatment needs its own predeclared control and fresh confirmation panel;
this result must not be retuned into a passing headline. GCP remains deferred.
