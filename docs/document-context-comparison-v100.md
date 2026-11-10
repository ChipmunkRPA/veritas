# Show the actual contexts behind a passage response

New document evidence sets (`document-evidence-3`) show the lowest and highest
recorded classifier responses among distinct saved inputs containing a selected
passage. Both exact source excerpts, source character positions and input
commitments appear in the ordinary evidence view, PDF and JSON. The browser
marks the selected passage inside both excerpts using Unicode codepoint offsets.
No additional inference or payment is performed by this comparison.

This is a response to the weakness identified in the
[fixed text-formatting comparison](text-format-comparison-v94-result.md) and
the [recent paper review](recent-detection-research-2026-11-09.md). It does
**not** fix paragraph invariance, isolate formatting from content, reproduce a
paper, or improve measured authorship accuracy. The compared inputs contain
different surrounding text. Their responses apply to the complete inputs,
not to individual highlighted words. No authorship confidence is assigned to
the comparison and it does not change the overall confidence.

## Construction and source verification

The comparison uses successful recorded classifier contexts bound to the same
extracted source. Ratio scores are excluded. Identical actual input text is
deduplicated, and inconsistent responses for the same input are refused.
At least two distinct compatible inputs must contain the complete selected
passage. Otherwise the result remains unassessed. Stable responses do not
establish originality.

The two displayed endpoints retain the complete recorded response range;
omitted input counts remain visible. At most six check comparisons appear,
ordered by response spread, with any additional count disclosed. This display
rule introduces no warning threshold or new AI direction. Responses displayed
to three decimal places remain unrounded in JSON, apart from the explicitly
six-decimal spread representation.

The report verifier reconstructs the comparison from the original saved
result and source, rather than accepting a self-consistently rehashed score.
The public offline verifier checks structure, source commitments, range
arithmetic and references using the standard library. It does not rerun a
classifier or independently authenticate the original response. Redacted PDF
views withhold both context quotations. Existing frozen evidence versions 1
and 2 remain readable and are not silently regenerated or upgraded.

## Completed local checks

The [aggregate replay receipt](evaluation-panels/document-context-recorded-model-v100.json)
records all **800 retained effective inputs** and **6,286 passage locations**:
6,282 compatible comparisons and four correctly unassessed locations. The
comparison helper and validator passed on every location, with zero new neural
calls and no Torch/Transformers imports. Nonzero ranges are not AI detections
or a measured error rate. These are previously consumed controlled compositions,
not fresh natural articles or independent authors.

This replay reconstructs context strings with the earlier reader recipe and
checks recorded focus geometry and raw responses. It does not independently
reproduce tokenized model inputs or execute a second model implementation.
Runtime was 3.56 seconds; maximum observed self RSS was 153,141,248 bytes.
Observations were per input, not continuous peak measurements. Existing neural
budgets, model settings, calibration and frozen experimental artifacts remain
unchanged.

Additional local software verification:

- 202 backend test cases passed with no failures, errors or skips, including
  evidence integrity, original-result binding, historical formats, public
  packages, full/redacted PDF exports and prepayment gates.
- All 246 frontend tests passed; the production TypeScript/Vite build passed.
  The existing large-bundle advisory remains. One obsolete plain-view navigation
  test was corrected to match the already visible findings; production navigation
  was unchanged.
- Actual app-generated authored-fixture PDFs were rasterized and inspected:
  ten pages full, nine redacted. This checks report layout and quotation
  withholding, not model accuracy.
- An isolated real-browser rendering of the actual component passed at
  1366 x 1050 and 390 x 844, with two correct passage marks at each size,
  no horizontal overflow or page errors. This was an authored component
  fixture, not a new paid end-to-end analysis. Its first temporary page had
  an import error; corrected imports passed. The browser connector itself
  timed out, so this check used local Playwright.
- The idle local API was refreshed with its original command and environment
  preserved, and returned healthy. The frontend is available on port 5230.
  No analyses or charges were started by that refresh.

Source quotations, per-input replay records, browser captures and QA PDFs remain
private. Published material contains code, aggregate counts and whole-artifact
commitments only. No GitHub Actions, outreach or GCP deployment was performed.

## Still to validate

Layout-consistent input construction needs its own controlled comparison and
separate calibration before adoption. Fresh natural articles, emails, mixed
documents and short generated inserts remain necessary. This context display
exposes a limitation; it does not close those detection gaps. Recent image
relation-learning, lip-focused interaction and separately labeled speech/event
methods are still proposed experiments, as distinguished in the paper review.
