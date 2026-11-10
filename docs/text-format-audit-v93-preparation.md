# Formatting stress audit: input preparation completed

The subsequent [fixed-model comparison and separate local review](text-format-comparison-v94-result.md)
have now completed all 800 effective inputs. This preparation receipt below
records the earlier non-neural stage; preparation alone did not establish
detector stability.

All **288 consumed diagnostic documents** from 32 known source/prompt components
were prepared under the [fixed whitespace-only procedure](text-format-audit-v93-method.md).
Five views per document give 1,440 requested views. After actual app extraction,
these contain **800 distinct effective classifier texts**. The future comparison
can therefore reuse responses for 640 equivalent requested views; this is not
a count of completed or avoided neural forwards.

| Input view | Documents whose effective text changes, /288 |
| --- | ---: |
| Unchanged original | 0 |
| Horizontal-whitespace control | 0 |
| Single spaces | 224 |
| Wrapping at 48 codepoints | 224 |
| Paragraphs every 40 existing tokens | 288 |

All non-whitespace characters, existing token boundaries and source labels
remain exact. Effective views retain mappings to the original Unicode codepoint
positions. Whitespace has no authorship label, and projections retain actual
source characters rather than filling bounding intervals.

The [public preparation receipt](evaluation-panels/text-format-input-preparation-v93.json)
and [input freeze](evaluation-panels/text-format-input-freeze-v93.json) contain
aggregates and whole-artifact hashes. Source text, individual fingerprints,
pairings, character maps and labels remain private. Preparation finished in
1.64 seconds; maximum per-case observed process RSS was 76,709,888 bytes and
minimum observed available RAM 5,916,098,560 bytes. These discrete observations
are not continuous measurements or guaranteed peaks. Twelve authored checks
and lint passed locally, including changed-word refusal, Unicode correspondence,
real extraction equivalence and source-provenance preservation.

**No detector response was computed by preparation.** The subsequent comparison
kept the old heads, maps and research points fixed. Input equivalence after extraction
is not detector robustness; a paragraph-sensitive response is not AI evidence
or a provider watermark. These are reused controlled 160-word documents, not
fresh authors, natural articles or email accuracy. No model, serving default,
customer report, payment flow or infrastructure changed.
