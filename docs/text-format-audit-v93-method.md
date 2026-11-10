# Formatting stress audit: paired inputs and original positions

This independent implementation follows the evaluation idea in
[Measuring Artifact Reliance in Machine-Generated Text Detectors, 24 October 2026](https://aclanthology.org/2026.docinsights-1.13/).
It is a stricter whitespace-only input audit, not a reproduction of that paper's
punctuation/profile interventions, placebo-corrected score or reported accuracy.
No author implementation or checkpoint is used.

## Controlled paired views

Use all 288 already consumed v7 diagnostic documents from 32 known source/prompt
components. Retain every original composition and its exact character labels.
The underlying TXD-22 publisher source is CC BY 4.0; its attribution and
limitations remain in the [v7 result](writing-spans-v7-result.md). This is not
a new-author test or validation on natural emails/articles.

Prepare five input views per document:

1. Unchanged original.
2. Horizontal-whitespace control: replace runs of whitespace other than CR/LF
   by two ordinary spaces, retaining existing line and paragraph structure.
3. Single spaces between existing non-whitespace tokens.
4. Wrapping at 48 Unicode codepoints, without splitting long tokens.
5. Paragraph breaks every 40 existing non-whitespace tokens.

Require every non-whitespace character to remain unchanged and in order.
Additionally preserve every existing token boundary: joining `AI generated`
into `AIgenerated`, or splitting an existing token, is rejected despite an
unchanged non-whitespace subsequence. Do not normalize case, punctuation,
Unicode composition or invisible non-whitespace characters. This preserves
textual content under the stated rules; it does not guarantee identical
semantics for code, tables, emails or other layout-dependent material.

## Actual input and source correspondence

Run each view through the app's existing string extractor. Verify its effective
text still preserves the same content/boundaries. Deduplicate only identical
effective strings within a document, requiring identical source positions and
labels. Keep every requested view and its effective-input reference, even when
it requires no distinct detector call. An extractor-normalized control is not
evidence of detector robustness before any detector has run.

Store an immutable original-character index for each effective non-whitespace
codepoint. Whitespace has no authorship position. A flagged span projects to the
actual original non-whitespace positions, not a filled bounding range. Preserve
human/AI character counts across views and refuse unknown provenance on any
retained non-whitespace character. Offsets are Unicode codepoints, end exclusive;
they are not byte or browser UTF-16 indices.

This map permits later comparison of retained highlights despite changed
paragraphs, sentence segmentation and context. A whitespace-only span cannot
supply authorship evidence, and an unassessed response cannot clear its source.

## Execution and next comparison

The builder verifies the original published v7 freeze, complete diagnostic
journal, input hash and sealed heads/calibration/point files. Preparation binds
the new utility, builder, tests, extractor and this procedure before any new
model calls. Source text, pairings, individual text fingerprints, annotations
and mappings remain private. Public summaries contain aggregate counts and
whole-artifact hashes only. Preparation makes no network, child-process or
neural call and refuses an existing/partial output folder.

Per-case data-only observations use a 900-second bound, 256-MiB process RSS cap,
2-GiB initial available RAM, 3-GiB initial disk and 1.5-GiB running RAM/disk
reserves. These are discrete observations, not continuous or peak guarantees.
Neural comparison needs a separate frozen worker and the unchanged larger
neural budget; no scientific model or earlier method is edited by preparation.

The next comparison must keep the current passage and interval heads and their
existing maps/points fixed, report effective-input changes, score changes and
source-aligned highlight changes for every view, and retain failures/skips.
Agreement between variants does not establish accurate authorship; disagreement
is an instability finding, not evidence of a provider watermark. Any changed
model or proposed combination needs new fitting/calibration and genuinely new
source confirmation before serving adoption. The actual model-response test,
fresh natural articles/emails and customer integration remain pending.
