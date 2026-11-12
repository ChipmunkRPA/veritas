# Isolated sentence corroboration on dated editorial revisions

Requiring both writing checks to agree on a native window lost both useful generated locations in the earlier editorial transfer. The second check’s exact center-sentence score crossed the same comparison point on six of ten known generated sentences. Its native-window score crossed the point on zero of ten. Allowing corroboration from either input recovered one useful location and reintroduced four false flags on scientific prose. The comparison supports retaining the existing evidence-only behavior. [Earlier editorial transfer](writing-agreement-ordinary-edit-2026-11-11.md) · [Audited counts](evaluation-panels/writing-agreement-focus-replay-2026-11-11.json)

## Exact input replay

Reopen all 40 retained files from the earlier study. This includes 16 pre-2020 forum versions from eight distinct groups, 14 original and constructed ordinary eLife controls from seven groups, and 10 admitted target-only synthetic replacements from two generators. Before scoring, the runner verifies the public commitments, publisher source records, exact archived ranges, generator receipts, and previous model responses. The forum bodies have CC BY-SA terms and the scientific sources use CC BY 4.0. This replay reuses both exact files and prior responses. It supplies no fresh source holdout. [Source terms and input preparation](writing-agreement-ordinary-edit-2026-11-11.md#versioned-source-panel)

Take the exact center of each recorded native window as a separate input when it has at least 12 whitespace-delimited words. This produced 269 eligible occurrences across 148 distinct focus inputs, with seven native positions ineligible for the added reading. Both unchanged, pinned Apache-2.0 classifiers scored every distinct input on CPU. The isolated scores are raw responses. Shortening a window changes both context and input length, so the observed score shifts do not distinguish their effects. [Checkpoint identities and scope](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Keep the current detector’s first six native positives in source order. Compare requiring a positive second response on the same native input, on its isolated center, or on either. All four rules keep the same native display spans and internal response point. Useful localization requires at least 0.30 non-whitespace character intersection over union with the exact replacement. Exact focus is also counted separately. The frozen criterion for pursuing an independent follow-up required preserving all useful current locations, clearing two currently flagged human groups and reintroducing at most one human group relative to native agreement in either control panel.

## Results

| Case type | Files | Current native | Same native window agreement | Exact focus agreement | Either scope agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dated human forum files flagged | 16 | 2 | 0 | 0 | 0 |
| eLife human files flagged | 14 | 11 | 7 | 11 | 11 |
| Generated edits usefully localized | 10 | 2 | 0 | 1 | 1 |
| Generated edits with wrong location | 10 | 2 | 0 | 0 | 0 |
| Generated edits with exact focus selected | 10 | 1 | 0 | 1 | 1 |

The isolated view recovered one exact Granite focus and still missed the useful OLMo location. The second isolated response crossed the point on four of seven exact Granite and two of three exact OLMo targets. Most of those targets had no matching current positive at the exact focus. In eLife, either scope flagged five of seven source groups, up from three under same-window agreement. Across all 269 eligible occurrences, the second check shifted from negative to positive 90 times and from positive to negative once. The isolated view broadly changes the response and lacks the selectivity required by the frozen criterion.

The CPU runs made 74 model forwards across both classifiers. The standard-library audit independently reconstructed exact focus inputs, 276 native-window occurrences, 269 eligible focus occurrences, 148 unique input commitments and all 32 scope and arm counts. A macOS resource wrapper reported a denied system counter after the first completed score receipt. The second stage ran without that wrapper. No memory peak is claimed. Model weights, source prose, and individual scores stay local.

## Replay

Use the prior exact panel and eLife preparation directories. Use the current and second checkpoints named in the earlier evaluation. The runner checks their pinned asset hashes. Model scoring is offline, while the last command verifies the records without importing either classifier.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$run"
focus() { backend/.venv/bin/python -B -m scripts.writing_agreement_focus_replay "$@"; }
focus freeze "$run" "$parent" "$elife_source"
focus current "$run" "$parent" "$elife_source" --model "$current"
focus second "$run" "$parent" "$elife_source" --model "$second"
focus report "$run" "$parent" "$elife_source"
backend/.venv/bin/python -B -m scripts.review_writing_agreement_focus_replay "$run" "$parent" "$elife_source"
```
