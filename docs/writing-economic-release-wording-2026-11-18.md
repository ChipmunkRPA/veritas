# Compare named federal releases with their published wording

Chipmunk Verify can compare a single attributed upload with a locally
retained BEA or Census Bureau release. The comparison shows the official
title and release date, the source link, the digest of the exact archived
publisher page, the wording before and after, and the changed sentences
at their original locations. It carries zero weight in the AI assessment.
An ordinary human edit can produce the same finding.

## Scope

The upload begins with the official title and a `Source: https://…` line,
followed by the excerpt. The source line selects a record only from the
local verified pack. User-provided URLs trigger no network requests.
The check requires a verified title and official URL, a unique ordered
pair of 12-token anchors at the edges of the submitted excerpt, and
at least 60% exact token alignment in both excerpts. Bodies of 64–800
tokens qualify. The official span is also capped at 800 tokens. Changes
are bounded between those anchors. The matching rules are scope and
correspondence checks for this comparison.

The current pack has 22 retained publisher pages from 2017–2020. Its
builder verifies exact HTML and editorial text digests, canonical URLs,
publisher titles and release dates from the reviewed study intake. The
retained source prose excludes marked quotations, contact lines, tables
and paragraphs that link to non-agency domains. The pack uses federal
editorial prose. BEA classifies its site content as public domain unless
otherwise stated. Census public-use guidance asks users to identify the
source, and 17 U.S.C. § 105 addresses copyright in U.S. government works.
The current publisher page may incorporate later corrections. The report
therefore identifies the retained text and source page by their hashes.
[BEA reuse policy](https://www.bea.gov/help/faq/145) ·
[Census public-use guidance](https://www.census.gov/about/policies/citation.html) ·
[17 U.S.C. § 105](https://www.copyright.gov/title17/92chap1.html)

In the development instance this works for TXT, DOCX, ODT and PDFs with
supported native text and page order. The PDF report, customer analysis
view, JSON export and signed evidence package share the original positions.
The independent package verifier reconstructs the comparison from the
submitted original and the earlier text retained in the package. The GAO
check keeps its previous format and has its own archive configuration.

## Local validation

The readback added the verified title and source line to each existing
economic test excerpt. The 22 unchanged copies produced no differences.
The 15 generated rewrites recovered all 33 known changed sentences and
marked four additional sentences. All 17 same-source human substitutions
were also reported as wording differences. Across the 54 excerpts, each of
five separate controls produced zero matches: anonymous excerpt, missing
title, missing URL, incorrect URL, and the complete title and URL of a
different release from the same family. These measurements test a
comparison with a known counterpart. [Held rewrite study](writing-economic-paraphrases-2026-11-17.md) ·
[Wording comparison receipt](evaluation-panels/writing-economic-release-wording-2026-11-18.json.gz)

One attributed Census rewrite also passed an isolated simulated paid
DOCX analysis. That run verified an official source link in the PDF, five
wording ranges tied to three Word sentences, identical evidence in the
API, JSON export and signed package, and successful local replay against
the DOCX. The original file was omitted from the signed package.

## Rebuild and use locally

Use the exact retained economic study archive inputs. The build checks the
original reservation hash and each source, and fails if a live page has
changed. The report pack stays local.

```sh
export PYTHONPATH=backend
runner=backend/.venv/bin/python
"$runner" -B -m scripts.build_economic_release_archive "$study/intake" "$out/pack.json"
"$runner" -B -m scripts.review_economic_release_wording "$study/held.json.gz" \
  "$out/pack.json" "$out/receipt.json.gz"
```

For the development service, set `VERITAS_DOCUMENT_ECONOMIC_ARCHIVE_PATH`
to the absolute pack path and `VERITAS_DOCUMENT_ECONOMIC_ARCHIVE_SHA256`
to the builder's reported SHA-256, then restart the local service. Submit
one supported file with the official title as its first paragraph, the
`Source: https://…` line as the next paragraph, and the selected excerpt
after it. The panel appears when the attribution and wording qualify.
