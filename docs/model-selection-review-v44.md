# Reopen model decisions and compare useful tradeoffs

The owner asked us to revisit earlier decisions across images, video, audio and
writing, retain multiple candidates and decide after comparative tests. This
review supersedes earlier decisions about whether a candidate deserves another
comparison. It does not rewrite a historical experiment or turn its failed
frozen rule into a pass.

The [decision register](evaluation-panels/model-decision-review-v44.json) reviews
**28 studies: 3 image, 12 audio, 5 writing and 8 video**. Each entry binds its
source document and states the revised research disposition. Existing serving
models and rollback remain available. No new customer confidence is admitted.

## Revised selection approach

- Compare useful findings, misses, each false-alarm category, affected area or
  text length, coverage, calibration quality and execution cost together.
- A modest increase in an error category is a tradeoff to measure. It does not
  automatically reject a large detection or localization gain. Preserving every
  previous successful case is not a universal requirement for a new candidate.
- Retain multiple models, mappings and actual combinations. A weaker standalone
  total can conceal complementary cases. An observed dominated point can still
  be useful on a different subset; keep its results and inspect that support.
- Compare all arms on the same planned sources. Count failures, unassessed and
  inconclusive outcomes. Do not count abstention as an original-file conclusion.
- Use development/calibration comparisons to choose models and combinations.
  Then freeze the chosen methods before the untouched final benchmark. A panel
  used repeatedly to choose methods becomes development, not fresh validation.
- Keep commercial permission, source/weight integrity, valid geometry and honest
  scope as requirements. Correlated models are not independent votes. Agreement
  or a scalar raw response does not become a customer authorship probability.

The [comparison helper](../backend/scripts/evaluation_tradeoffs_v44.py) retains
every method and describes observed tradeoff frontiers without an automatic
rejection. It refuses missing/invalid metrics and keeps unknown case outcomes
separate from misses. Point-estimate dominance is not statistical superiority.

## Native images: reopen the adapted candidate

On 39 calibration photos in 32 source components, the strictest originally
frozen native setting has 66.7% useful localization by component mean versus
20.8% current. False-alarm groups are:

| Control, out of 32 components | Current | Adapted candidate |
| --- | ---: | ---: |
| Original | 3 | 6 |
| Tone | 7 | 7 |
| Blur | 11 | 5 |
| Any of those controls | 12 | 7 |

The old requirement of no increase in **every** control type rejected this
candidate because of originals, despite its much greater useful localization
and lower any-control false-flag count. Reopen it for unused-source comparison.
The [original 628-call result](image-regional-calibration-v43-results.md) stays
unchanged. This is localization, not 66.7% whole-file accuracy or confidence.
JPEG behavior is substantially weaker and remains a separate comparison.

The new review computes the observed frontier for all 28 recorded map settings
and the current check in each format. No setting is retrospectively declared
validated. Subsequent comparisons can include several settings and models;
their final validation cohort must remain separate from that selection.

## Writing: actual combinations recover additional regions

The old interval learner had 38 useful mixed-document locations versus 45
current and was rejected as a replacement. It nevertheless found 20 locations
missed by current. We now measured **actual source-character masks** from six
fixed standalone/union/intersection methods, using the retained responses and
unchanged component points. No label chooses a method for a document.

All 288 consumed diagnostic documents remain: 128 mixed, 128 genuine controls
and 32 pure-AI constructions from 32 connected prompt components. All 864
standalone mask measurements reproduce exactly before combination. Another
864 combined masks are evaluated. No neural model is rerun or fitted.

| Method | Useful mixed locations /128 | Recovered AI intervals /224 | Genuine documents highlighted /128 | Genuine groups highlighted /32 | Genuine human characters wrongly highlighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 45 | 39 | 5 | 3 | 1,643 |
| Token head | 1 | 0 | 5 | 3 | 160 |
| Interval head | 38 | 78 | 4 | 1 | 2,076 |
| Current OR interval | **67** | **102** | **7** | **3** | **3,322** |
| Current AND interval | 13 | 17 | 2 | 1 | 397 |
| All three OR | 67 | 103 | 11 | 5 | 3,465 |

The two-model union adds 22 useful mixed-document locations and loses none of
the current method's 45 on this panel. Its human-highlight burden roughly
doubles. Adding the third model yields no extra useful mixed-document location
and increases genuine alerts. This is evidence for further two-model testing,
not permission to hide the extra false-highlighted text.

The 65 documents on which either standalone method was useful were not assumed
to be the union's accuracy: the union was explicitly reconstructed and measured,
and it usefully locates 67. Combining individually weak masks can recover more
of an interval; combining masks can also worsen IoU. Tests cover both concerns.

These are post-result development measurements on controlled 160-word documents,
not fresh articles or emails. The combination has no separately fitted
calibration and has not been tested on new writers/generators. Neither empty
highlights nor agreement establishes word origin or whole-document originality.

## Audio and video candidates reopened

| Area | Evidence worth retaining | Revised next comparison |
| --- | --- | --- |
| Audio decision layer, v7 confirmation | Correct conclusions 761 to 823; false AI 35 to 38; false-original 39 to 52; inconclusive 245 to 167. | Compare the complete coverage/error tradeoff; do not reject automatically for exceeding the former one-error allowance. Strong-noise AI called original remains material. |
| Audio WavLM complement, v13 | False AI 58 to 4; noise/music improve; overlapping speech and AI-call coverage worsen. | Compare current, complement and independently specified context combinations; inference routing must not use the known test condition. |
| Audio layers/context/separation | Several mappings improve proper scores or false alarms while changing abstention and AI recall. | Keep a shared candidate pool and operating curves. Near-total-abstention variants remain low priority, not universal architecture failures. |
| Mouth/audio correspondence, v14 | Located edits 26/32 to 31/32; ordinary flags 12/56 to 24/56. | Retain as a sensitive review cue and compare its actual combined evidence with visual-only and current checks. Timing mismatch alone is not AI proof. |
| General video AEGIS | Useful raw ranking on some generators; the original monotone calibration collapsed on another domain. | Compare calibration across multiple generator/source domains, with current synthesis checks and other permission-qualified candidates. Raw ranking is not confidence. |
| Face appearance and background context | Existing appearance remains strong for change localization; ordinary insertions can trigger it. Some context combinations make small gains. | Compare incremental and complementary value, genuine edits and compute cost. Keep authorship claims separate from locating a change. |

Zero-useful-location crop experiments, poorly ranked binary passage checks and
near-all-inconclusive video/audio mappings remain lower priority on the evidence
available. Their code/assets are retained; these results do not prove that their
architectures can never work. Restrictions on noncommercial weights and the
historical numerical/monitor failures are not waived by this selection change.

## Verification and publication

The [complete aggregate reanalysis](evaluation-panels/multimodal-tradeoff-review-v44.json)
contains all image settings, all six writing methods, per-composition outcomes,
gains/losses and source-journal commitments. It contains no source text, source
IDs, raw images/audio, feature vectors or learned weights. The
[review implementation](../backend/scripts/review_multimodal_tradeoffs_v44.py)
can replay the retained private writing inputs and public image result.

Forty-two focused local tests and Ruff pass. A second complete reanalysis
reproduced the published aggregate byte for byte, including all six writing
methods and the image frontiers. This is a local cached-response replay, not
new neural inference or outside review. The first new review attempt
expected a per-row status on a phase receipt and refused before output; its
failure is retained. The corrected validator requires the original complete
phase status, exact document count, diagnostic role and freeze binding. Separate
successful outputs are retained, including the extension that counts actual
combined-mask gains/losses. This changes no original source, point or outcome.

Local observed modes remain audio `candidate`, text `candidate` and face review
`appearance`; repository defaults are unchanged. The protected audio calibration,
customer model-name filtering, USD 9.99 prepayment and report limitations remain
unchanged. No workflow, external outreach, charge or GCP resource is used.
Outside human reviewers remain zero; this is local re-evaluation, not external
certification. Fresh multi-model confirmation and full product-flow testing
are still required before a new default or confidence contribution.
