# Can two local rewrites locate an AI-edited sentence?

An uploaded sentence is rewritten once, then the probe rewrites its own result.
The first reading measures semantic similarity between the uploaded sentence
and its first rewrite. The second reading subtracts the similarity between the
first and second rewrites. This follows the idea of an instance-specific
reference in iterative semantic-difference research. The comparison uses
independent implementation and already installed Apache-2.0 checkpoints.
The probe sees only sentences from the single upload. [Iterative semantic-difference paper](https://www.sciencedirect.com/science/article/pii/S0925231226010532)
[Pinned local probe](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
[Pinned sentence encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

Both readings examine the same exact sentences and use only valid two-step
responses. A focus must contain 14–58 words. Each probe response must be one
sentence, retain the written numbers, stay within the fixed length limits,
and reach the minimum semantic similarity. This does not verify the facts in
either rewrite. The comparisons keep the original Unicode character ranges.
Each file needs at least three reading pairs and can return one unique top
focus. Each method uses its own operating point, set at the second-highest
ordinary work maximum before probing held inputs.

The calibration cohort contains 13 NASA and NOAA articles and 13 matched
controls made by rearranging genuine sentences. The held cohort contains 20
distinct works, four per publisher, selected by group hash before probing.
The 56 held deliveries include 20 unchanged originals, 14 constructed
ordinary controls, four naturally archived revisions to *The Turing Way*, and
18 locally generated rewrites across three editor families. The Qwen rewrite
family shares the probe model. The SmolLM2 and Granite families are separate.
The historical text predates 2022, and the authoring process in every original
cannot be confirmed. All cases come from earlier project studies. Calibration
and held works and exact probe sentences are disjoint for this comparison.
[NASA and NOAA source audit](writing-alignment-preference-2026-11-18.md)
[BEA and Census source audit](writing-economic-paraphrases-2026-11-17.md)
[The Turing Way editorial sources and CC BY 4.0 terms](writing-rtf-draft-2026-11-18.md)

The baseline uses the locally configured passage interval selector with
document-relative and economic-report review. Other cues and overall verdicts
are outside this comparison. A useful location covers half the labeled edit
and has at least 0.30 intersection over union after removing whitespace. Each
label can count once. An addition advances only if it locates new edits across
at least two work groups and two editor families, with at most one additional
ordinary source group flagged.

## Held results

| Reading | Known edited sentences located | Untouched originals flagged | Constructed ordinary edits flagged | Archived revisions flagged |
| --- | ---: | ---: | ---: | ---: |
| Existing passage selector | 0/23 | 1/20 | 0/14 | 0/4 |
| Existing plus one-step similarity | 1/23 | 3/20 | 2/14 | 0/4 |
| Existing plus two-step similarity | 0/23 | 1/20 | 0/14 | 0/4 |

Both candidates assessed all 56 files. Four files fell below the existing
selector's 12-sentence minimum and remain in the counts. The one-step reading
matched one SmolLM2 edit, added two misplaced highlights on edited files, and
flagged four more ordinary files from two source groups. The two-step reading
made no selections. An eligible candidate range existed for 18 of the 23 known
edits. Before applying the cutoff, the highest one-step focus matched two known
edits and the highest two-step focus matched one. The result gives little
support for using a second rewrite to improve the ranking. Both methods remain
outside the app.

## Replay

The [numeric receipt](evaluation-panels/writing-rewrite-response-2026-11-18.json.gz)
retains publisher links, source-input hashes, exact sentence ranges, measured
similarities and aggregate counts. It omits source prose and probe rewrites.
Recalculate the decisions from the saved readings without models or network
access.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_rewrite_response review \
  docs/evaluation-panels/writing-rewrite-response-2026-11-18.json.gz
```

The generation pass read 152 distinct calibration and 251 distinct held
sentences. A formatting fix added the existing encoder's required input-hash
field before semantic measurements. The calibration probe responses were
rebound only after checking that the two input plans matched in every field
except the implementation hash. The held generation ran after calibration.
