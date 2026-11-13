# Paired writing checks on dated forum revisions

Eight historical source threads yielded eight ordinary wording revisions and
14 matched one-sentence rewrites. In a broad comparison, combining primary
and paired responses recovered 13 of 14 generated locations, compared with
five for the primary response alone. The combination also flagged four
genuine later drafts, compared with three for the primary response alone.
[Source and result commitments](evaluation-panels/writing-forum-revision-transfer-2026-11-13.json)

## Source selection

We requested the 90 highest-voted questions dated 2016–2019 from each of
English Language & Usage, Travel, Super User and Ask Ubuntu. A supplemental
intake returned 137 of 280 accepted-answer histories before the API limited
access. Selection used the returned data and chose at most one dated
revision pair per thread, considering questions first and then accepted
answers. A pair needed named contributors, version-specific CC BY-SA terms,
an actual wording change within one paragraph, and unique exact neighboring
sentences. The source had to segment identically in the app. Retained source
groups did not share contributors or a normalized 12-word sequence with
each other or with the previous dated forum pairs. The selected excerpts
also shared no 12-word sequence with the retained OpAI-Bench, F1000 and
USWDS test inputs. Exact model-training exposure is unknown.
[Revision API](https://api.stackexchange.com/docs/types/revision) ·
[Version-specific license policy](https://stackoverflow.com/help/licensing)

The selected threads comprise five Travel questions, one Super User
question and one answer each from Super User and English Language &
Usage. Two pinned Apache 2.0 generators each attempted two seeded rewrites
per selected source, using the previous USWDS protocol. Seven Granite and
seven OLMo rewrites met the sentence admission rule before detector
scoring. Every accepted fixture preserves the text outside the replaced
sentence. The matched pairs measure localization within short excerpts.
Neither the original forum prose nor generated text is included in the
public ledger. [Prior USWDS test and checkpoint terms](writing-uswds-revision-transfer-2026-11-13.md)

## Broad comparison

Both unchanged local checkpoints read the same 96 distinct native
contexts. The comparison includes the usual two-sentence contexts at
the edges. A useful location overlaps at least half the replaced
sentence and reaches at least 0.30 non-whitespace character IoU.

| Method | Earlier originals flagged | Genuine later versions flagged | Useful generated locations |
| --- | ---: | ---: | ---: |
| Primary check | 4/8 | 3/8 | 5/14 |
| Primary check intersecting exact changed tokens | 0/8 | 2/8 | 5/14 |
| Both checks increase on the paired context | 0/8 | 2/8 | 11/14 |
| Primary or paired increase | 4/8 | 4/8 | 13/14 |
| Both checks agree at their original comparison points | 0/8 | 0/8 | 0/14 |

The union recovered five additional Granite and three additional OLMo
locations. It flagged one genuine later version that the same rule left
unflagged in the earlier version. Direct factual draft comparison showed
changed tokens in every genuine later draft and met the same span rule in
10 of 14 rewrites. Its shorter token spans measure wording differences.

## Scope matching the draft comparison

The existing optional app check was replayed with exact native source
hashes and retained CPU responses. It found six of 14 generated
locations, with zero flags in six eligible genuine later versions.
Two genuine later versions fell outside the existing supported input scope.
Allowing other directly affected three-sentence contexts within that same
scope marked the exact changed sentence in nine of 14 generated files and
zero of the same six genuine later versions. Those nine exact sentence
anchors meet the overlap rule. Scored context is a separate, larger range.
Applying the rule to that larger range would count eight of the nine.
The broader method's two genuine flags occurred only in two-sentence
edge contexts.

The app's optional authenticated earlier-draft check now uses the
interior approach for short plain-text files in the existing
single-sentence scope. Both versions need the same paragraph layout and
unchanged sentences around the edit. The browser, PDF and JSON keep the
exact changed sentence linked to its wording change and separately show
each scored context and its uncalibrated responses. The check carries
zero authorship confidence weight. When the optional contexts exceed
the available scoring coverage, the app can return the existing
changed-sentence result.

The local authenticated HTTP comparison ran both pinned serving
checkpoints on all 22 genuine later versions and generated rewrites.
Its scope and decisions matched the offline comparison. We reopened
and verified the PDF and JSON for each of the three newly recovered
locations against both drafts. Of the 128 individual responses compared,
the largest absolute difference was below 0.000001 for the primary
checkpoint and 0.0483 for the second checkpoint. The offline evaluator
specifies float32 and SDPA for that second checkpoint. The serving loader
uses the snapshot defaults. The raw scores shown in an app report come
from the serving route.

## Replay

A second local CPU pass regenerated the same score ledger byte for byte.
The clean-checkout standard-library reader checks all 30 files and source
commitments and recomputes both the broad arms and the narrower scope
from the retained model responses, source eligibility records and
positions. Rebuilding eligibility and inference from scratch also
requires the exact captured publisher inputs, admitted generated
fixtures, scratch intake scripts and pinned local weights. The ledger
publishes the dated revision links and hashes and omits excerpt text.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_forum_revision_transfer \
  docs/evaluation-panels/writing-forum-revision-transfer-2026-11-13.json
```
