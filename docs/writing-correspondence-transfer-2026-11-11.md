# Passage review on archived official correspondence

The holdout used seven memoranda and one letter from two State Department economic-policy volumes. The documents have eight distinct attributed federal senders. The Office of the Historian releases the master files in the public domain. The selected bodies omit editorial notes, quotations and contact information. The human cases passed an exact normalized 12-token overlap screen against 45,073 text entries in 36 retained project packets. Pretraining exposure remains unknown. [Official collection](https://history.state.gov/about/faq/what-is-frus) · [Publisher master files and rights](https://github.com/HistoryAtState/frus)

Each source supplied an original, a full human paragraph replacement, a two-sentence human replacement, and a target-paragraph rewrite from each of two Apache-2.0 generator families. The human replacements use length-matched prose from elsewhere in the same document. All 16 generated rewrites passed the unchanged source-copy, length and source-scope checks. The unchanged source text and exact edited intervals are preserved in the private run.

Each of the 40 files used a synthetic multipart EML container around the archival body. The app selected the quoted-printable UTF-8 plain body, preserved its exact Unicode positions, and excluded an alternate HTML body and an attachment. These fixtures evaluate correspondence prose and email extraction. They do not measure accuracy on modern mailbox traffic or authenticate senders.

## Consumed-only fit

The fit replayed 120 previously measured cases from science explainers, fiction and practical notices, with zero new classifier calls. Each genre point had to flag at most one fifth of its human source groups. A group counted if any of its original or edited human views was flagged. Among qualifying points, the fit favored useful localization averaged over source groups. The pooled point enforced the same constraint separately in each genre. The practical-notice point was assigned to the entire correspondence holdout before the holdout sources were acquired or scored. The raw and strict reference rules retained their earlier points. [Science](writing-paragraph-context-transfer-2026-11-11.md) · [Fiction](writing-fiction-paragraph-transfer-2026-11-11.md) · [Practical notices](writing-practical-notice-transfer-2026-11-11.md)

## Held-out results

| Outcome | Current raw review | Strict raw | Pooled point | Transferred practical point |
| --- | ---: | ---: | ---: | ---: |
| Human originals flagged | 4/8 | 0/8 | 0/8 | 3/8 |
| Human paragraph replacements flagged | 3/8 | 0/8 | 0/8 | 3/8 |
| Human two-sentence replacements flagged | 6/8 | 0/8 | 0/8 | 5/8 |
| All human files flagged | 13/24 | 0/24 | 0/24 | 11/24 |
| Human source groups with a flag | 7/8 | 0/8 | 0/8 | 5/8 |
| Qwen rewrites usefully localized | 7/8 | 2/8 | 2/8 | 8/8 |
| SmolLM rewrites usefully localized | 7/8 | 4/8 | 4/8 | 8/8 |
| All eligible rewrites usefully localized | 14/16 | 6/16 | 6/16 | 16/16 |

Useful localization required at least 0.30 non-whitespace character intersection over union with the known edited paragraph. The classifier made 190 CPU forwards. Every file had complete scored source coverage. Relative to current raw review, the transferred practical point added one useful location per family, kept all 14 useful locations and cleared two human flags without adding any. It still flagged five of eight human source groups, exceeding the development constraint on this holdout. The stricter and pooled points missed ten rewrites each. These results do not support a customer-facing point or confidence change. [Exact aggregate and source commitments](evaluation-panels/writing-correspondence-transfer-2026-11-11.json)
