# Controlled writing span learning, v7: development result

**Keep the current passage check.** The small interval learner improves some
multi-region and pure-AI source coverage, but is not a replacement: it loses
27 mixed-document locations the current method usefully identified, and
locates only1/32 larger AI inserts versus18/32 current. The short-insert blind
spot remains. Neither small learner changes serving behavior or customer
confidence.

## What was fixed before inference

The [protocol](writing-spans-v7-protocol.md), implementation and
[public freeze](evaluation-panels/writing-spans-v7-freeze.json) were published
by direct push as `89c6e009b91a4c54df26afd1e688d8e1d4a551d9`, and the remote
freeze blob was verified as `5ff5aa83aeedaf2c6e76624c87a4262111a9c064`
before the first project-source forward or fit. The private lock SHA256 is
`61dfe6aa86e6a07fc1a70c04394686b747790f746fca40bd3b5bd406d34ea68a`.
All29 authored tests and the supported native/overflow execution probe passed
locally first. Repository GitHub Actions remained disabled throughout.

The cached Apache-2.0 ModernBERT encoder stayed frozen; own token and interval
heads were fitted on288 documents from32 prompt/duplicate components. Each
arm had its own separate288-document calibration role. Another32 components
provided only128 genuine controls to choose genuine-group maxima points.
No AI control was inferred to select those points. Heads, calibration, point
responses and points were sealed before scoring288 diagnostic documents.

This consumes128 previously used development components and992 controlled
160-word documents. These are not fresh authors or independently verified
authorship. Nine related compositions share each non-point component; they
are not nine independent people. Actual current three-sentence-context passage
scores are the comparator, not the entire SaaS fusion.
The own token head has769 parameters; the interval head has154,628. These
counts exclude the shared frozen pretrained encoder.

## Results at the fixed genuine-only research points

Each diagnostic condition contains32 documents. A useful location requires
source non-whitespace character IoU>=.3. Known-interval recovery means at
least half of that particular AI interval's non-whitespace characters were
highlighted. These are internal research criteria, not public confidence
thresholds or guaranteed population error rates.

| Measure | Current passage check | Matched token head | Interval head |
| --- | ---: | ---: | ---: |
| Useful mixed-document locations, /128 | 45 | 1 | 38 |
| Mean mixed character IoU | .20089 | .02065 | .21963 |
| Known mixed AI intervals recovered, /224 | 39 | 0 | 78 |
| Useful pure-AI locations, /32 | 22 | 3 | 27 |
| Useful12-word AI insert locations, /32 | 0 | 0 | 0 |
| Useful60-word AI insert locations, /32 | 18 | 0 | 1 |
| Useful mostly-AI/human-gap locations, /32 | 22 | 1 | 21 |
| Useful multi-region locations, /32 | 5 | 0 | 16 |
| Genuine source groups with any highlight, /32 | 3 | 3 | 1 |
| Genuine documents with any highlight, /128 | 5 | 5 | 4 |
| Human characters wrongly highlighted in genuine documents | 1,643 | 160 | 2,076 |

The interval head finds20 mixed locations not usefully located by current,
but loses27 previously useful locations: the net is seven fewer. Its slightly
higher mean mixed IoU does not establish an improvement: the paired
source-group bootstrap95% difference interval is **[-.04757,.08466]**.
It beats the matched token learner under the frozen numerical comparison,
but fails two requirements against current: more useful mixed locations,
and preservation of previously useful mixed locations. Both comparisons were
required. Gates are unchanged after observing the results.

Fewer flagged genuine groups also conceals a cost: a false interval highlight
can cover more human text. Here1 flagged group produces2,076 wrong character
highlights across its genuine compositions, versus1,643 across3 groups for
current. These counts must be shown together. Neither fewer alerts nor a
better calibration loss is sufficient to adopt a detector.

All three methods locate **none** of the known AI characters in the32 short
insert cases. The apparent highlights in that condition are human-origin
characters. A whole-document AI-leaning response would not fix or substantiate
that source-localization gap. The data do not justify word-origin proof,
an originality verdict for unhighlighted words, or calibrated customer
authorship probabilities.

## Execution review

The complete local replay **passed**. All992 compositions reconstruct from the
original CSV and connected source components. Every complete native feature,
offset/input proof, current response, head proposal and calibrated interval
matches exactly. A separate set-based calculation reproduces all864 diagnostic
arm masks and character counts. Both fitting-only head refits reproduce every
saved parameter and epoch loss exactly; all three calibration refits and
genuine-only points match. The summary and unchanged gate decisions replay.
This is execution review using the same local native libraries, not an outside
human or an independent author population. Outside human reviewers: **0**.

| Phase | Completed documents | Actual encoder forwards | Actual current-classifier forwards | Guard duration, seconds | Sampled tree RSS lower bound, bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit, including both small learners | 288 | 289 | 0 | 40.32 | 1,388,593,152 |
| Separate calibration | 288 | 289 | 704 | 255.64 | 2,936,143,872 |
| Genuine-only points | 128 | 129 | 345 | 123.41 | 2,828,877,824 |
| Diagnostics | 288 | 289 | 692 | 252.76 | 2,924,724,224 |
| Full source/native/refit replay | 992 | 996 | 1,741 | 659.32 | 3,369,402,368 |

Every phase exits0 with no resource guard and no sampling errors, within its
900-second/6-GiB bound. No planned project document is skipped or failed.
Audit/tap forwards are counted, not described as zero neural inference.
The [aggregate package](evaluation-panels/writing-spans-v7-result.json)
binds all private phase journals, controls, head hashes, summary and terminal
review, including exact resource profiles. It contains no source text,
row pairings, token vectors or learned weight values.

The earlier authored native probe failed on a combining accent omitted by the
tokenizer. That attempt and its failed profile are retained. A separate corrected
probe explicitly requires refusal before any forward on that unsupported
input; supported ASCII and full-overflow probes pass. The token-only preflight
and all planned project-source runs cover all their source characters. That
is not evidence for universal Unicode input coverage.

## What this suggests next

The interval approach may supply complementary multi-region evidence, but this
study does not test a combined serving policy. A future combination must be
specified separately, preserve the current larger-insert capability, use its
own fitting/calibration roles, and face new-source confirmation. Do not choose
a fallback or combine these arms using diagnostic labels, then describe the
same documents as fresh validation. The short-insert failure needs additional
method and source work; it is not solved by changing a public threshold.

## Source credit and scope

Iqbal, Md Sadiq; Kashem, Mohammod Abul (2026), *TXD-22: A Large-Scale Benchmark
Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification*,
Mendeley Data,V1, [DOI10.17632/prcjcggtjf.1](https://data.mendeley.com/datasets/prcjcggtjf/1),
CC BY4.0. Changes: exact original answer slices recomposed into controlled
160-word documents with explicit source-character annotations. Original data,
source mappings, row pairings, token vectors and learned head weights remain
private. External pretrained-training overlap and author identities remain
unknown. Publisher labels and consent statements are not independently verified.

The source builder, model/native loaders, learners, calibrators, fixed-point
selection, metric summary and complete local reviewer are public. Reconstructing
this exact private panel additionally requires the retained old source-group
plan and separately acquired licensed CSV; the repository does not contain an
independent public author benchmark. Aggregates cannot prove source truth.
The preliminary SaaS report, payment flow, serving confidence and rollback
files remain unchanged. GCP deployment and external review are not part of
this local comparison.
