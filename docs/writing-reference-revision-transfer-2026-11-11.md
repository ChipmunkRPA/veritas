# Mixed writing on dated article revisions

This study tests whether a writing flag persists on the same classifier input in an earlier draft and whether a changed sentence produces a new response. Chipmunk Verify already offers a model-free draft comparison for matching TXT, EML and DOCX inputs in the authenticated local application. This study uses its exact word and punctuation ranges and checks whether paired model responses add reliable localization. [Existing draft comparison](writing-paired-email-revision-2026-11-11.md)

## Sources and fixed comparisons

The complete Crossref catalog for January–June 2017 contained 98 version-2 F1000Research records with a CC BY 4.0 term and a version-specific publisher XML link. An initial source-only scout of 28 pairs found no citation-free seven-sentence windows with a single exact interior edit. The separate expanded rule excluded all 28 scout works before collecting the remaining 70 pairs. It used two identical preceding and two identical following sentences and omitted marked bibliographic cross-references. The original and revised publisher XML both had to supply the correct DOI, author list, online publication date and license. The selected author body prose excluded other markup and quoted or imported blocks. The intake also checked for identical 12-token sequences in six earlier project datasets and against other selected groups. [Crossref journal catalog](https://api.crossref.org/journals/2046-1402/works) · [F1000Research versioning and licensing](https://f1000research.com/about) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Review of the frozen source intake identified one citation-format-only candidate. That unscored roster remains archived locally. The amended rule disregarded inline author-year citations when qualifying a lexical revision. Nine distinct author groups qualified. Both versions are dated 2016–2017. The two versions establish a dated ordinary-writing control and an exact comparison target. Publisher attribution and publication dates cannot establish all tools an author may have used.

Both pinned Apache-2.0 generation checkpoints attempted a single rewrite of the exact center sentence for every article. The generation lock fixed the prompt, seeds, asset hashes and runtime before the first response. The acceptance rule required a complete, single-sentence rewrite with adequate lexical changes. All other sentences stayed identical. The same two existing writing classifiers read complete native inputs locally with one CPU thread. Response points stayed fixed. The comparison includes no fitting and gives paired responses no confidence weight. The experiment uses the idea of measuring known edit locations across versions and independently preserves the exact source lineage. No data, weights or code came from the papers. [Recent revision benchmark](https://arxiv.org/abs/2606.06481) · [Mixed-text segmentation research](https://aclanthology.org/2026.findings-eacl.326/) · [Pinned detector identities](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Current uses the existing source-ordered native review windows. Same-window agreement requires both checkpoints to cross the same raw comparison point on identical input. Changed-window agreement keeps only those agreement windows whose exact input differs from the earlier version. The paired response marks the changed center when both checkpoints independently at least double their own AI-like odds across matched contexts. Combined takes both changed-window agreement and the paired focus. For generated cases, useful localization requires non-whitespace character IoU of at least 0.30 and recall of at least 0.50. The frozen follow-up rule requires preservation of current useful locations, two added source groups across both generation families, and at most one added human revision group flagged relative to same-window agreement. Before-versus-self serves as a scope control.

## Results

| Outcome | Current | Same-window agreement | Changed-window agreement | Paired response | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier human versions flagged | 3/9 | 1/9 | 0/9 | 0/9 | 0/9 |
| Later human revisions flagged | 2/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| Generated edits usefully located | 1/11 | 1/11 | 1/11 | 4/11 | 4/11 |
| Generated edits with a wrong location only | 1/11 | 0/11 | 0/11 | 0/11 | 0/11 |
| Generated edits with no flagged location | 9/11 | 10/11 | 10/11 | 7/11 | 7/11 |
| Generated edits whose exact center was selected | 0/11 | 0/11 | 0/11 | 4/11 | 4/11 |

The exact draft comparison localized 37 change hunks within the recorded center sentences across all 20 edited pairs, including one human deletion with an empty later range. It also found changes on both human and generated inputs as expected. The same-window check flagged one of the nine earlier human versions. It flagged none of the nine revised versions, so this article panel did not reproduce the earlier eLife pattern of persistent false flags after same-window agreement. The exact reference window for the earlier flag was the same by construction. [Earlier editorial results](writing-agreement-focus-replay-2026-11-11.md)

The paired direction recovered three additional generated edits from three distinct source groups and retained the existing useful location. It flagged none of the dated human revisions. Granite accounted for all four hits among nine admitted rewrites. OLMo accounted for zero hits among two admitted rewrites. Each generator attempted nine cases. This improvement merits replication with more admitted cases from another generator and publisher. The frozen cross-generator criterion did not pass. The experiment made no serving change.

The two CPU runs scored 105 distinct native inputs each with 54 total model forwards. The first scoring invocation stopped before inference because a tuple did not round-trip through JSON as the same type. The corrected run reproduced all 18 generation responses exactly and retained the same source selection and rule. The source replay reproduced the licensed roster, and the standard-library audit checked 140 publisher XML hashes, all exact inputs and 136 outcome counts without model inference. Source prose, generated excerpts and per-passage responses stay local. [Aggregate and commitments](evaluation-panels/writing-reference-revision-transfer-2026-11-11.json)

## Replay

Use the retained publisher catalog and XML archive, the six exact history files listed in the source receipt and the existing pinned local checkpoints. Pass those history inputs as `--history NAME=PATH`. Generation requires the repository’s existing MLX environment and access to Metal. The classifier steps use the backend environment, CPU and offline snapshots.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
backend/.venv/bin/python -B -m scripts.writing_reference_revision_sources verify "$run" --expanded "${history_args[@]}"
for phase in freeze-scoring current second report; do
  backend/.venv/bin/python -B -m scripts.writing_reference_revision_transfer "$phase" "$run" --current "$current" --second "$second"
done
backend/.venv/bin/python -B -m scripts.review_writing_reference_revision_transfer "$run"
```

To rebuild inference receipts, copy the source plan, catalog freeze, source receipt, retained originals and publisher archive to a new run directory. Run `freeze-generation`, then `generate --family granite` and `generate --family olmo` under the MLX environment before the classifier steps. To check the completed receipt in place, run only the source verifier and final standard-library verifier.
