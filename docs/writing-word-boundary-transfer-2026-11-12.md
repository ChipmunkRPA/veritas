# Word draft comparison across a paragraph break

Chipmunk Verify can compare one revised sentence beside a preserved paragraph break in two short English Word drafts. It reads the changed sentence alone and with its two unchanged neighbors. The comparison displays the linked wording changes and all six sentence locations in the Word body, in the browser, PDF and JSON. The check uses uncalibrated before-and-after responses and carries zero authorship confidence weight.

## What changed

The Word draft comparison already locates differences in words and punctuation. The previous paired writing check gave no response when an unchanged neighbor was in a different Word paragraph. The new check requires one changed sentence, identical immediate neighbors, the same paragraph assignment and adjacent top-level body paragraphs in both versions. Empty paragraphs and tables between the neighbors put the pair outside scope. Source replay reads both original Word packages and recalculates the saved direction from the recorded responses. It verifies the paragraph and Unicode ranges and rejects altered source files or edited ranges.

Enable the local experimental check with `VERITAS_DOCUMENT_PAIRED_BOUNDARY_REVIEW=true` where the earlier-draft feature and writing agreement review are already enabled. Open the completed Word analysis, select **Compare with an earlier draft**, choose the earlier `.docx`, and select **Include experimental paired writing check**. The draft comparison leaves the original analysis assessment intact.

## Frozen local results

The test re-encoded 22 earlier TXT draft pairs as Word and reproduced all saved responses bit for bit. A separate transfer panel used ten CC BY 4.0 PLOS Biology source works already used in a different single-file experiment. Seventeen known one-sentence rewrites and ten same-article ordinary sentence substitutions were converted to DOCX, with a deterministic paragraph split next to the known changed sentence in both versions. No PLOS pairs set the operating point. Each positive location corresponded exactly to the edited sentence. [Prior paired calibration](writing-paired-center-transfer-2026-11-12.md) · [PLOS source selection and terms](writing-semantic-continuity-transfer-2026-11-12.md) · [PLOS licenses and copyright](https://journals.plos.org/plosbiology/s/licenses-and-copyright)

| Result | Granite edits | OLMo edits | Ordinary replacements flagged |
| --- | ---: | ---: | ---: |
| Previous Word scope across paragraph break | 0/10 | 0/7 | 0/10 |
| Frozen sentence and context rule | 3/10 | 0/7 | 1/10 |
| Original same-paragraph response rule on these inputs | 3/10 | 0/7 | 1/10 |
| Previously fitted context-only point on these inputs | 5/10 | 1/7 | 1/10 |

The three flagged edits came from different works. The ordinary flag and all fourteen misses limit how broadly the response can be interpreted. The test measures short article excerpts with constructed single-sentence revisions and controlled Word layout. The previously fitted context-only point found more edits here. Earlier TXT testing showed one extra ordinary work flag at that point, so the Word extension uses the same two-scale rule as TXT. No new model weights were installed. [Pinned checkpoint licenses and terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Five normal Full-profile jobs completed after simulated Stripe payments. The test included known positive cases, an OLMo miss and the ordinary false alarm. Browser and PDF evidence showed the same source positions as JSON. Offline replay matched all ten native source files and thirty sentence positions, and the historical out-of-scope result still replayed. Chrome exercised the comparison from the plain report, followed the wording links and downloaded JSON and PDF equal to the route response.

The receipt records source and input hashes, Word positions, token counts and saved responses. The exact excerpts, generated text and Word files stay local. The model-free reviewer recomputes the recorded decisions, checks geometry and compares prior and current raw readings. A separate earlier-draft export reviewer can verify a user's own source files.

```sh
python3 -B -I -S backend/scripts/review_writing_word_boundary_transfer.py \
  docs/evaluation-panels/writing-word-boundary-transfer-2026-11-12.json
```

[Local measurements and replay receipt](evaluation-panels/writing-word-boundary-transfer-2026-11-12.json)
