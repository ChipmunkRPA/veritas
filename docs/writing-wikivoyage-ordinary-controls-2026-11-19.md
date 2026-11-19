# Writing checks on archived travel-guide revisions

The current earlier-copy comparison located all seven archived editorial
changes across 28 TXT and DOCX comparisons. It reproduced 32 Word paragraph
positions and reported zero differences across 28 unchanged earlier-copy
controls. The standalone passage check assessed all seven earlier excerpts and
seven revised excerpts and highlighted zero passages. This panel measures
ordinary edits and false alerts. It contains no known AI-generated positives.
[Count-only receipt](evaluation-panels/writing-wikivoyage-ordinary-controls-2026-11-19.json)

## Source and qualification

The roster used the 24 publisher-listed feature guides from 2015 and 2016. For
each guide, we froze an archive window from 75 days before its feature month
through 45 days after the month's first day and read up to 30 revisions in
chronological order. Seven histories had more revisions than the cap. All 521
returned wikitext revisions matched their publisher SHA-1. Of 497 exact parent
links, the primary screen checked 132 eligible, visible revisions by different
registered contributors. It found four replacement pairs. A separate reserve
checked 210 adjacent revisions by the same contributor in the other 20 guides
and added four replacement pairs. Neither stratum yielded a qualified complete
sentence insertion.

All eight replacement pairs matched unique ranges in both official renders.
Each retained 12 unchanged sentences, with at least three on either side of the
changed sentence or sentences. Exact offsets refer to the canonical visible
prose extracted from the publisher render. The check excluded quotations,
embedded media and explicitly sourced paragraphs. Manual review removed one case
whose edit note named third-party sources for its fare estimates. The seven
remaining guides include punctuation, grammar and factual updates. Three
contributor accounts each appeared in two retained works. All retained passages
cleared an overlap check against 116,404 known normalized 12-token fingerprints
from 578 retained project excerpts. Earlier project panels used separate 2014
and 2020–2021 featured guides. Other past project exposure and model pretraining
content are unknown. [Featured guide roster](https://en.wikivoyage.org/wiki/Previous_Destinations_of_the_month#2016) ·
[Revision API](https://www.mediawiki.org/wiki/API:Revisions) ·
[Earlier scored panel](writing-token-curvature-2026-11-11.md)

The exact pre-upgrade revisions are available under CC BY-SA 3.0. The current
Wikivoyage guide describes CC BY-SA 4.0. Wikimedia's upgrade guidance preserves
the earlier licence for archived revisions from before the change. Both permit
commercial reuse under attribution and share-alike terms. The selected revisions,
attribution and locally adapted TXT and DOCX excerpts remain in the private
source ledger. The published receipt contains anonymous counts only.
[Wikimedia guidance for pre-upgrade revisions](https://meta.wikimedia.org/wiki/Terms_of_use/Creative_Commons_4.0/FAQ/en) ·
[Wikivoyage reuse policy](https://en.wikivoyage.org/wiki/Wikivoyage:How_to_re-use_Wikivoyage_guides) ·
[CC BY-SA 3.0 terms](https://creativecommons.org/licenses/by-sa/3.0/)

## Source comparison and standalone screening

The earlier-copy feature received the literal excerpts in all four TXT and DOCX
directions. A comparison counted as a match when it identified at least one
changed token and every changed token fell inside the independently traced
replacement ranges. The comparison could mark just the changed words or
punctuation. The 28 carrier comparisons cover seven works.

The standalone trial screened each excerpt individually. The one-core offline
run scored 210 distinct sentence and surrounding-context inputs and replayed
the existing pooled, within-document and report settings. All three assessed
14 excerpts with no alerts. The additional report-specific scope did not apply
to these guides. Earlier-copy comparison documents changes between files. It
carries zero AI assessment weight. No detector or calibration changes were made.

## Recount

The checker reproduces the source funnel arithmetic, all four carrier
comparisons for each work, and standalone alerts for the earlier and edited
excerpts. It uses only the standard library and the anonymous receipt. Repeating
the underlying model run requires reconstructing the original source panel.

```sh
python3 -B -I -S backend/scripts/review_writing_wikivoyage_ordinary_controls.py \
  docs/evaluation-panels/writing-wikivoyage-ordinary-controls-2026-11-19.json
```
