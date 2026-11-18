# Source locations for saved HTML articles

Chipmunk Verify records original HTML character ranges for eligible article
passages in the browser, PDF, JSON and signed evidence packages. Positions are
Unicode codepoints in the decoded upload. The reader attaches ranges when a
whole extracted block matches one article paragraph, heading or list entry
unambiguously. Where that match fails, reports retain extracted-text positions.
The reader handles one marked article or main area up to 512 Ki decoded
characters. It uses static markup and leaves linked resources and scripts
unloaded. The same writing input and paragraph boundaries still go to the
detectors, and the existing checkout rules apply.

## Local replay

Four NOAA and three NASA English-language articles originally published before
2021 provided authentic controls. The local study used full saved snapshots,
with each edited variant changing one known paragraph and preserving the raw
HTML outside it. Seven original files and ten generated rewrites passed the
frozen admission checks. Four replies failed those checks. The study used
government-published text and kept source HTML, article excerpts, third-party
assets and generated transcripts out of the replay artifact. Publications may
have been updated before the snapshots were taken. [NOAA reuse information](https://www.noaa.gov/office-education/outreach-communication/faq) [NASA Earth Science reuse information](https://science.nasa.gov/earth/faq/)

On all 17 admitted files, the production reader reproduced the previous
classifier text and paragraph boundaries verbatim and recovered the exact
original HTML position of the focus span. The held comparison below separately
tested replacing the classifier input with the article-only projection. It used
the pinned classifier's three-sentence windows and the first six qualifying
locations. A located edit required at least 50% focus recall and 30%
intersection-over-union in non-whitespace characters. The corresponding
unchanged paragraph in each control was measured by the same rule.

| Result | Existing extraction | Experimental article-only input |
| --- | ---: | ---: |
| First generator rewrites localized | 1/5 | 2/5 |
| Second generator rewrites localized | 0/5 | 1/5 |
| Ordinary control files with any flagged span | 7/7 | 7/7 |
| Ordinary unchanged focus paragraphs flagged | 0/7 | 1/7 |

The false alarms and small sample leave the article-only input unsuitable for
promotion. This comparison covers one existing classifier and the bounded
location selection. It provides no estimate of full-pipeline performance or
general HTML authorship accuracy. The source-location improvement applies only
to already selected passages with an exact mapping and carries no weight in
the AI assessment.

The local browser test opened a completed $9.99 simulated prepaid analysis.
Its six evidence cards included 11 matching original HTML locations across
the browser, PDF and JSON. The signed package verified with and without the
submitted original. A previously signed HTML package also verified against
its original using the historical extraction. [App replay receipt](evaluation-panels/writing-saved-html-app-2026-11-18.json)

The [held replay](evaluation-panels/writing-saved-html-scope-2026-11-18.json.gz)
retains publisher URLs, exact positions, input hashes and classifier responses.
Recount it without weights, source text or network access.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_saved_html_scope review \
  docs/evaluation-panels/writing-saved-html-scope-2026-11-18.json.gz
```
