# Checking whether a DOI points to the cited work

A reference can print a genuine DOI for a different publication. A local
comparison checked the visible title and author against a frozen Crossref
snapshot. It recovered the exact changed DOI in **52 of 60 held substitutions**
across five citing works. Crossref resolved all 60 substituted identifiers.
The measurement concerns citation accuracy and carries zero weight in the AI
assessment. [Reference verification research](https://arxiv.org/abs/2605.07723)

## Method

Read one reference at a time from a labeled bibliography. The eligible
paragraph contains one visible DOI and between 32 and 640 Unicode codepoints.
The DOI must identify an entry in the local registry snapshot. An exact
normalized title of six to 65 words and its first recorded family name, which
must appear before the title, can identify a different registered work. The
title must start within the early part of the citation. Both works must have
sufficient title metadata, their title word sets must differ by at least 55%
under Jaccard distance, and the registry must show no qualifying version or
format relationship between them. Ambiguous or missing evidence stays
unassessed. The observation preserves the printed DOI and matched title
intervals as separate, half-open Unicode ranges.

The frozen intake used seven article-level CC BY 4.0 research works. Four
PLOS ONE articles supplied reference entries from mixed-citation XML. Three
eLife articles supplied entries rendered from their stored author, title,
venue, year and DOI fields. The trial preserved reference order and used a
common heading and entry numbering. Two PLOS ONE works supplied development
cases. The other two and all three eLife works were held out. The positive
cases replace a DOI already supported by its reference with a different real
DOI from another work in the same split. A fixed hash order chooses the donors.
Separate controls contain one out-of-snapshot DOI, two DOIs, or a removed DOI.
All identifiers queried for the benchmark came from public source material.
[PLOS license terms](https://plos.org/terms-of-use/) · [Pinned eLife XML](https://github.com/elifesciences/elife-article-xml/tree/143b1639fc47a80b3897c4c046e5c076d1110904/articles) · [Crossref metadata terms](https://www.crossref.org/documentation/retrieve-metadata/)

## Results

| Measurement | Development | Held |
| --- | ---: | ---: |
| Exact substituted DOI located | 15/17 | 52/60 |
| Substituted DOI available in Crossref | 17/17 | 60/60 |
| Original reference matched the registered work | 16/40 | 306/369 |
| Wrong-work observations on original references | 0/40 | 0/369 |
| Original references left unassessed | 24/40 | 63/369 |
| Unknown, multiple or absent DOI controls left unassessed | 6/6 | 15/15 |

Of the eight held misses, five had insufficient evidence that the destination
title was distinct and three lacked one uniquely matched reference. The
registry-only comparator checks availability and makes no wrong-work calls.
Crossref returned **364 of 370 distinct requested identifiers**. This is the
coverage of a lookup set assembled for these public references. Coverage for
new uploads depends on which registry records are available locally. Ordinary
citation errors, incomplete publisher deposits, and alternative versions
remain relevant. The substitutions are labeled by known citation edits.

Four paired DOCX and TXT replays covered ordinary and substituted references
from one held work per publisher. All **222 DOI positions** and **22 matched
title positions** agreed. The DOCX ledger retained the original body paths and
within-paragraph character ranges. The local evaluation used the existing
document extractor with Python 3.12.6 and python-docx 1.2.0 on Apple Silicon.

## Independent recount

The numeric receipt retains anonymous citation groups, source, DOI and
response hashes, position ranges, and the counts needed to reapply the frozen
rule. The standard-library command recounts those saved measurements offline.
[Numeric receipt](evaluation-panels/writing-doi-title-verification-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_doi_title_verification \
  docs/evaluation-panels/writing-doi-title-verification-2026-11-18.json.gz
```
