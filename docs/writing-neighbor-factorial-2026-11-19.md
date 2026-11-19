# Does a neighboring rewrite move a writing highlight?

Changing one sentence moved the rolling-context reading for its unchanged
neighbor in all 16 matched comparisons. The largest shift was 0.555926 on
the classifier’s internal scale. A reading of the sentence alone and a
reading using the same outer sentences stayed identical in those comparisons.
Neither experimental selection rule entered the app. [Anonymous measurements](evaluation-panels/writing-neighbor-factorial-2026-11-19.json)

## Sources and method

The study used exact 2021 parent and child chapter snapshots from *The Turing
Way*, copyright The Turing Way Community, under CC BY 4.0. It retained 12
sentences per excerpt, including one dated revision and 11 unchanged
sentences. The intake excluded chapter paths and shared 12-word sequences
from the previous Turing Way test. Four source groups set the operating
points and five separate groups served as held controls. The publisher's
history establishes chronology and exact wording. Authorship tools used by
individual contributors and checkpoint exposure are unknown. [Publisher reuse guidance](https://book.the-turing-way.org/reproducible-research/licensing/licensing-documentation/) · [Pinned archive](https://github.com/the-turing-way/the-turing-way/tree/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Two local generators attempted to rewrite the focal sentence and one
adjacent sentence independently. With the surrounding prose in the prompts,
2 of 20 outputs passed the wording and alignment checks. Both changed a
factual detail. Several other responses rewrote surrounding prose. A separate
target-only arm excluded one fragmented target before generation. Of its 18
outputs, 14 passed structural checks. Six preserved meaning, seven changed
a claim or qualifier, and one remained unclear. The study froze this review
before detector scoring. It used the first eligible meaning-preserving
response at each position when available and kept an otherwise structurally
eligible response separately labeled for the intervention comparison.

The four complete source groups supplied an original, a focal rewrite, a
neighbor rewrite and a file with both rewrites. The actual dated revision
provided a separate ordinary control in each held work. The experiment
compared the current optional passage-interval check, a sentence-only
ranking and a ranking requiring the sentence-only and shared-endpoint
inputs to select the same exact interior sentence. The second input reads
that sentence between the file's first and last sentences. The two views
share classifier weights. The development rule allowed one more ordinary
alert group than the current check. Only one development group produced an
agreed selection, leaving no observed score below it to set a stricter
point. [Sentence localization research](https://aclanthology.org/2024.findings-acl.495/) · [Mixed authorship evaluation](https://aclanthology.org/2025.acl-long.1069/)

## Held results

The 12 rewritten files contained 16 edit positions across deliveries. Those
positions represent eight distinct source-and-sentence sites. Four files
contain one meaning-preserving edit. Every two-edit file includes at least
one rewrite that changed a claim. The ordinary comparison includes the
original and dated revision for each of the five source groups.

| Check | Useful positions across deliveries / 16 | Distinct sites / 8 | Wrong highlights | Ordinary groups flagged / 5 |
| --- | ---: | ---: | ---: | ---: |
| Current optional passage check | 0 | 0 | 0 | 0 |
| Sentence only | 2 | 1 | 0 | 0 |
| Sentence and shared endpoints agree | 4 | 2 | 3 | 3 |

The sentence-only rule found the same edit in a one-edit file and its
two-edit counterpart. Agreement found one distinct site from each
generator. Two of its wrong highlights landed on the unchanged sentence
next to the edit. At the fixed point, it located 2 of 4 meaning-preserving
single edits, compared with 1 of 4 for sentence only. The small sample from
one handbook leaves transfer and practical false-alarm rates uncertain.
The application keeps its existing detector and confidence scoring.

## Recount

The public receipt contains anonymous sentence identities, length counts,
scalar readings and selection geometry. The reviewer recalculates the
operating points, matched positions, distinct sites, ordinary alerts and
paired reading differences with the standard library. The local run used
381 distinct inputs and one CPU thread, taking about 28 seconds after
loading the classifier. The public receipt supports arithmetic review.
Exact source wording, individual revision references and model inputs stay
in the private evaluation workspace.

```sh
python3 -B -I -S backend/scripts/review_writing_neighbor_factorial.py \
  docs/evaluation-panels/writing-neighbor-factorial-2026-11-19.json
```
