# Text formatting changes passage evidence: completed local comparison

The frozen comparison completed **288 controlled documents, 1,440 requested
views and all 800 distinct effective text inputs**, with no unassessed response.
Changing only paragraph layout changed the current passage check's highlights
in **91/288 documents**. The interval check changed in **97/288** and their
union in **120/288**. Ordinary horizontal spacing changed none. This identifies
a concrete weakness in passage evidence; it is not a whole-document verdict
flip rate or evidence of a provider's hidden watermark.

The [original protocol](text-format-comparison-v94-protocol.md), models,
heads, separate calibration maps, genuine-only research points, source roles
and inputs remained fixed. Every unchanged original exactly reproduced its
earlier current/interval proposals, native feature and offset hashes and
actual current input-token hash before its alternate views were interpreted.
No new calibration, fitting or serving adoption occurred.

## Changes under the fixed research points

Each entry counts related documents with any changed source-character highlight,
out of the same 288 documents from **32 known source/prompt components**.
These are not 288 independent authors. A changed highlight can add or remove
either AI- or human-origin source characters, so stability alone is not accuracy.

| Method | Horizontal spacing | Single-space text | Wrap at 48 codepoints | Paragraphs every 40 words |
| --- | ---: | ---: | ---: | ---: |
| Current passage check | 0 | 58 | 58 | 91 |
| Interval head | 0 | 57 | 57 | 97 |
| Mask union | 0 | 83 | 83 | 120 |
| Mask intersection | 0 | 34 | 34 | 55 |

The unchanged-original self-comparison is zero for all four methods. Horizontal
space changes were already normalized away by actual app extraction. Wrapped
and single-space requests produced the same effective inputs and therefore
share responses; they are not independent confirmations of this effect.
Paragraph changes preserve every non-whitespace codepoint and existing token
boundary, but can change sentence contexts and the model's tokenized input.
This audit does not isolate one causal shortcut or reproduce the complete
published artifact-reliance benchmark.

Character gains and losses show why flattening formatting is not an automatic
solution. The counts below cover the complete mixed/pure controlled panel,
not just wholly genuine documents. A wrongly highlighted human-origin character
is not a count of falsely accused authors or false-positive files.

| View and method | AI characters added | AI characters lost | Human characters added | Human characters removed |
| --- | ---: | ---: | ---: | ---: |
| Single spaces, current | 1,470 | 9,037 | 2,931 | 1,153 |
| Single spaces, interval | 10,203 | 8,342 | 5,407 | 1,144 |
| 40-word paragraphs, current | 1,841 | 20,976 | 526 | 1,962 |
| 40-word paragraphs, interval | 6,051 | 22,349 | 3,302 | 2,624 |
| 40-word paragraphs, union | 4,335 | 25,887 | 3,488 | 3,836 |
| 40-word paragraphs, intersection | 2,116 | 15,997 | 250 | 660 |

Across shared scored characters, the current method's mean absolute mapped
response change is **0.084870** after flattening and **0.097040** after the
40-word paragraph intervention; the largest changes are **0.693671** and
**0.712221**. These are changes in the separately mapped interval responses,
not percentage-point changes in a customer's whole-file confidence.

Coverage also changes for the interval head. Under the paragraph intervention,
35,359 source characters have a response only in the original and 15,297 only
in the alternate view. Missing coverage remains missing rather than zero or
an originality claim. The full [aggregate result](evaluation-panels/text-format-response-result-v94.json)
includes these counts, source-group counts, overlap and score-change summaries
for every view and method. Union/intersection have masks only; neither has an
invented combined probability. Mean highlight Jaccard includes assessed empty
agreement, which does not establish correct authorship.

## Execution and separate review

All eight 36-document CPU float32 batches exited zero, with no resource guard
or live-sampling error. Their summed runtime was **741.61 seconds**. The largest
sampled process-tree RSS was **2,910,732,288 bytes**; samples are not guaranteed
peaks. The fixed 900-second/6-GiB neural limits and RAM/disk reserves were kept.
Permission inspection was checked on one owned standard-library child before
the first scientific worker was launched outside the sandbox.

The workers recorded **808 encoder forwards and 1,928 current-classifier
forwards**, including native/direct audits. There are two pretrained backbones
and the retained small head, not four independent models. The 640 equivalent
requested views are not described as 640 avoided neural forwards.

A [separate standard-library reviewer](text-format-independent-review-v99-protocol.md)
passed after its own procedure and freeze were manually published and the
remote blob verified. It imports neither the scorer nor application/scientific
libraries. It independently checks source-character correspondence, actual
recorded original parity, response/resource commitments, all **5,760 method/view
pairs** and the complete aggregate arithmetic. Integer counts and hashes match
exactly; floating sums match the unchanged 1e-9 absolute/1e-12 relative allowance.
Its 17 focused tests passed locally. The review itself took 2.11 seconds with
maximum sampled tree RSS 113,852,416 bytes, exit zero and no guard/sampling error.
It ran **zero new neural forwards** and did not refit or independently recalculate
the private calibration. The [review receipt](evaluation-panels/text-format-independent-review-v99.json)
contains aggregate status and whole-artifact hashes.

This is a separate local implementation of recorded-response arithmetic, not
an outside human evaluation, independent decoder/model implementation or new
authorship verification. Outside human reviewers: **0**. The earlier 900-second
before-launch RAM stop is preserved as history; it was not overwritten or
called a model failure. This later launch completed all previously unattempted
model batches without retry or source substitution.

## What to improve next

Keep the current passage check and retain the interval learner as a complementary
candidate. A union does not solve formatting sensitivity, and a more stable
intersection can simply miss more AI source content. No method is selected or
rejected on this stability count alone; detection gains, wrong human highlights,
lost AI coverage and context all need to be compared together.

The next development comparison should preserve original source offsets while
testing layout-consistent inference and explicitly reporting disagreements
between supported views. Paragraph context should not be silently discarded,
and source/extraction formatting should not be described as AI-authorship
evidence. Fit and calibrate any changed method separately, then compare on new
natural articles, email and extracted documents with source/generator separation.
The known short-insert blind spot remains unfinished. These reused controlled
160-word English compositions cannot validate those deployment contexts.

Source credit remains TXD-22 v1, Md Sadiq Iqbal and Mohammod Abul Kashem,
[DOI 10.17632/prcjcggtjf.1](https://data.mendeley.com/datasets/prcjcggtjf/1),
publisher-declared CC BY 4.0. The existing pinned Apache-2.0 checkpoints and
retained notices were used offline. Labels, authorship/consent assertions and
external pretraining overlap were not independently verified. Source text,
pairings, per-source fingerprints, character annotations, response journals,
hidden states and learned parameters remain private.

Customer confidence, payment-before-analysis, model-name filtering, reports,
serving defaults and rollback files remain unchanged. Publication uses reviewed
manual pushes; no Actions workflow, outreach, real payment or GCP deployment
occurred.
