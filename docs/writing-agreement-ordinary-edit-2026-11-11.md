# Writing agreement on dated editorial revisions

Requiring both checks cleared six human file flags across the two panels and removed both useful synthetic locations. The results support displaying agreement as evidence. Requiring agreement to keep an automated finding would lose detection coverage on these sources. [Exact aggregate and source commitments](evaluation-panels/writing-agreement-ordinary-edit-2026-11-11.json)

The earlier 100-file study found that two-checkpoint agreement cleared three human flags, all from one publisher, while preserving 14 useful edit locations. This follow-up checks whether that gain transfers to real editorial revisions and a separate scientific publisher. The scores are local responses used at fixed comparison points. [Earlier comparison](writing-checkpoint-agreement-2026-11-11.md) · [Professional editing and false alarms](https://aclanthology.org/2026.emnlp-industry.109/)

## Versioned source panel

The frozen intake requested 40 questions dated 2016 through 2019 from each of Stack Exchange Academia, History, The Workplace, and Writing. Each revision had to precede 2020, name an attributable user, expose the exact body, and specify a Creative Commons Attribution-ShareAlike license. Each pair keeps four consecutive sentences in the normalized retained body text. One interior sentence has a lexical change and the other three match exactly. Excerpts have 68–134 words. The procedure excludes markup, quotations, code, and formatting-only changes. It sorts by question ID and chooses the newest qualifying pair in each thread. Sources sharing a known contributor or a normalized 12-word fragment are held. The overlap screen includes the previous agreement panel, eLife materials, the retained English LLMTrace test archive, and the OPAI intake. Model-training exposure remains unknown. [Revision API](https://api.stackexchange.com/docs/types/revision) · [Revision license policy](https://stackapps.com/help/licensing)

Eight source groups qualified. They comprise two Academia threads, one History thread, three Workplace threads, and two Writing threads. These are distinct sources from four communities on one publication platform. A separate fixed roster of eight CC0 18F Content Guide pages produced no qualifying pair. Both versions of each selected pair precede the release of current general-purpose chat systems. Their dated histories provide ordinary-edit controls. The history does not independently certify how every contributor authored the text. The aggregate gives each selected exact revision URL, attribution, license, source position, and byte commitment. Publisher content stays local. [18F license](https://guides.18f.org/content-guide/license/)

Two Apache-2.0 generator families attempted to replace the earlier target sentence in each pair. Both use the exact previously pinned local quantized assets, one deterministic seeded attempt per family per source, and the same constrained rewrite request. Admission requires a completed single sentence with a nontrivial lexical edit, a valid length, and the same four-sentence source geometry. Each generated fixture preserves every character outside the changed sentence. Granite admitted seven attempts and TinyLlama admitted none. After completing the first comparison, a separately frozen OLMo run admitted three of eight attempts on the same target sentences. That family extends generator coverage within the measured sources. No detector response selected the sources or controlled admissions. [Generator asset provenance](writing-grammar-seams-2026-11-11.md)

Both classifiers score exactly the app's native three-sentence contexts, including its two-sentence edge contexts. The comparison keeps the first six positive windows in source order. The agreement arm retains the first check's positives only where the second check marks the identical input. Both model-specific internal points stay at 0.5. A useful location overlaps the known replaced sentence at at least 0.30 non-whitespace character intersection over union. The aggregate also tracks whether the exact focus was selected, wrong locations, wholly unflagged positives, and before-to-after false-flag transitions. These describe the comparison and supply no authorship confidence.

## Independent publisher controls

The eLife arm uses the previously unscored confirmation-role inputs from an earlier preparation-only intake. It keeps four of five previously frozen paragraphs with the same target and both neighboring paragraphs. The prior paper-level CC BY 4.0 source receipts and exact ordinary controls are reused. The eLife comparisons include untouched source prose and constructed copyedit and paragraph splice controls. They do not treat these constructed edits as observed publisher revisions. No generated edits are scored in this arm. [Source and control qualification](text-elife-editorial-controls-v301-result.md)

## Local results

| Case type | Files | Current flags | Second flags | Agreement flags | Current → agreement useful edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forum before edit | 8 | 1 | 0 | 0 | — |
| Forum after human edit | 8 | 1 | 0 | 0 | — |
| Granite replacement | 7 | 2 | 0 | 0 | 1 → 0 |
| OLMo replacement on same targets | 3 | 2 | 0 | 0 | 1 → 0 |
| eLife original prose | 7 | 5 | 7 | 3 | — |
| eLife constructed paragraph splice | 6 | 5 | 6 | 3 | — |
| eLife constructed copyedit | 1 | 1 | 1 | 1 | — |

The forum pair flagged by the current check had a flag in both versions. No forum edit introduced a new false-flag transition. Agreement reduced the number of flagged eLife source groups from five to three of seven. The tests cover short forum excerpts and longer scientific excerpts with related controls. The samples and admitted replacement families are limited, and the eLife arm supplies no information about synthetic detection there.

The model-free reviewer checked 16 public revision bodies, all 40 scored files, and all 276 context occurrences across the three phases. It reproduced the source slices, admission bindings, exact score inputs, and outcome arithmetic. Initial intake and offline generation startup retries ran before any classifier scoring. The completed comparison retained every planned generator disposition.

## Replay

Run the source acquisition and preparation from the repository root. Keep the downloaded record archive and source roster. The live question list can change. The result's pinned revisions and exact publisher records identify the observed cohort.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$run" "$elife_run"
backend/.venv/bin/python -B -m scripts.writing_agreement_revision_sources fetch "$run"
backend/.venv/bin/python -B -m scripts.writing_agreement_revision_sources prepare "$run" \
  --history "previous-agreement=$previous_agreement" \
  --history "llmtrace=$llmtrace" \
  --history "elife-v301=$elife_v301" \
  --history "elife-v286=$elife_v286" \
  --history "elife-v272=$elife_v272" \
  --history "opai-v257=$opai_v257"
```

The previous-agreement file contains the 100 `text` and `text_sha256` values from the source-bound input replay in `scripts.writing_checkpoint_agreement.frozen_inputs`. The other history files are the project-local exact JSONL inputs recorded in the aggregate. Generation uses the existing isolated MLX environment and a Metal device. The classifier steps use the ordinary backend environment and the exact pinned CPU checkpoints. The independent reviewer uses the standard library and the saved classifier responses.

```sh
transfer() { backend/.venv/bin/python -B -m scripts.writing_agreement_revision_transfer "$@"; }
transfer freeze-generation "$run"
for family in tiny granite; do
  backend/data/text-upgrade/mlx-env/bin/python -B -m scripts.writing_agreement_revision_transfer generate "$run" --family "$family"
done
transfer freeze-scoring "$run" --current "$current" --second "$second"
for arm in current second; do
  transfer score "$run" --arm "$arm" --current "$current" --second "$second"
done
transfer report "$run" --current "$current" --second "$second"

supplement() { backend/.venv/bin/python -B -m scripts.writing_agreement_revision_olmo "$@"; }
mkdir "$run/olmo"
supplement freeze "$run/olmo"
backend/data/text-upgrade/mlx-env/bin/python -B -m scripts.writing_agreement_revision_olmo generate "$run/olmo"
supplement admit "$run/olmo"
for stage in current second report; do
  supplement "$stage" "$run/olmo" --current "$current" --second "$second"
done

for stage in freeze current second report; do
  backend/.venv/bin/python -B -m scripts.writing_agreement_elife_controls "$stage" "$elife_run" "$elife_source" "$current" "$second"
done
backend/.venv/bin/python -B -m scripts.review_writing_agreement_revision_transfer "$run" \
  --elife "$elife_run" --elife-source "$elife_source" --olmo "$run/olmo"
```
