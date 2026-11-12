# Archived Python PEP revisions for single-file writing checks

Python Enhancement Proposals offer a separate publishing system from the journal and forum panels. The official PEP process uses a versioned source repository, documented authors and explicit public-domain or CC0 terms. This intake required those terms in both exact historical files. The files and normalized prose stayed local. [PEP process and licensing](https://peps.python.org/pep-0001/#copyright) · [Fixed historical repository anchor](https://github.com/python/peps/tree/475d6cdf8caf61cfe02ec527f84d9489566349e6)

The metadata freeze selected 200 pre-2020 work histories in SHA-256 order. For each work it considered up to 12 dated one-parent commits, excluding revert and known bot records. Admission required an unchanged documented author roster, a distinct recorded editor across admitted works, and no matching normalized 12-token fragments in nine retained project packets or another admitted work. Git authorship and dates identify the published historical revisions. They do not verify who composed every sentence. Model pretraining overlap was unmeasured.

The initial comparison required five consecutive body sentences with a material middle replacement and four exactly unchanged surrounding sentences. The next freeze retained that geometry and added explicit handling of the historical four-space PEP text layout, simple inline literal names, and numbered reference marks. The next design allowed the two outermost sentences to differ while retaining both immediate neighbors. A separate diff intake compared short replacements in an actual parent-to-child prose diff and used exact equal prose before and after the replacement block as anchors.

## First source block

| Frozen method | Mechanical work groups | Admitted work groups |
| --- | ---: | ---: |
| Plain-text, exact five-sentence neighborhood | 1 | 1 |
| Legacy-layout rendering, exact neighborhood | 1 | 1 |
| Immediate neighbors match | 2 | 2 |
| Parent-to-child prose diff with exact outside anchors | 6 | 4 |

The four methods reuse the same first 200 metadata work histories. Their admissions are separate and cannot be added together. The prose diff reached six work groups. Two shared a recorded author or editor with an earlier admitted work. A second metadata freeze retained all 296 remaining work histories and applied the unchanged prose-diff method. The two blocks share no work IDs. The combined roster also screens for documented author, recorded editor, and 12-token overlap across blocks.

The extension reached six mechanical pairs and admitted five within its own block. Four of those shared an author or recorded editor with a previously admitted work. One new work joined the first four. The combined pilot therefore used five groups across 496 inspected work histories. [Source commitments and results](evaluation-panels/writing-pep-editorial-intake-2026-11-11.json)

## Single-file writing pilot

The ordinary controls are the five exact earlier excerpts and the five exact later revisions. For the synthetic cases, each of two pinned Apache-2.0 generators attempted one rewrite of the earlier center, with the same exact surrounding sentences. A generation had to produce a complete, materially changed single sentence and preserve its native center offsets. The historical excerpts and generated rewrites stay in the local evaluation directory. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) · [SmolLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The current check uses the same native context windows for all files. The matched-context comparison ranks the raw responses for up to three eligible interior sentences and can mark one exact sentence. The candidate separately scores the left-plus-center, center-plus-right, and left-plus-right sentence pairs. It subtracts the two-flank log odds from the weaker of the two center-containing log odds. All three reads contain two complete sentences. The flank pair skips the center, which can itself change the classifier response. The highest eligible sentence must exceed a point fixed on the 32 source maxima from 64 previously used journal control files. The two rankers use separate points. This is a local response measurement. It contributes no authorship confidence. [Localized detection research](https://arxiv.org/abs/2607.21458) · [Current classifier and license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

Both generators completed five attempts. Four per generator qualified. A useful location requires at least 0.30 non-whitespace character intersection over union and 0.50 recall of the known replacement.

| Held outcome | Current context | Matched native-context ranker | Center-deletion ranker | Current plus center deletion |
| --- | ---: | ---: | ---: | ---: |
| Qwen rewrites localized | 0/4 | 1/4 | 0/4 | 0/4 |
| SmolLM2 rewrites localized | 0/4 | 1/4 | 0/4 | 0/4 |
| Earlier ordinary versions flagged | 0/5 | 0/5 | 0/5 | 0/5 |
| Later genuine revisions flagged | 1/5 | 0/5 | 0/5 | 1/5 |

The matched ranker also marked one SmolLM2 rewrite at the wrong location. Before applying the calibrated points, the matched ranker placed the center first on five of eight rewrites and the center-deletion ranker did so on four of eight. The center-deletion arm gave no qualifying added location. The five groups could not meet the prespecified six-admissions-per-generator requirement. The matched-context result supports a larger evaluation and remains descriptive. There was no serving change.

The classifier ran 165 CPU forwards across 660 distinct exact input strings. Independent replay verified 82 fixtures, 1,055 source-span and input bindings, all 18 held outcomes, and both cutoff calculations. A changed input hash failed verification. The retained source replays also reconstructed all five frozen intake receipts. [Public aggregate](evaluation-panels/writing-pep-editorial-intake-2026-11-11.json)

## Reproduce the source intake

Supply an isolated bare checkout of the public PEP repository and the retained prior text packets bound in the freeze. The source review uses Git and the existing text extractor and segmentation environment. The commands perform no detector inference.

```sh
cd backend
.venv/bin/python -B -m scripts.writing_pep_revision_intake verify "$initial" "$pep_repo"
.venv/bin/python -B -m scripts.writing_pep_legacy_intake verify "$legacy" "$initial" "$pep_repo"
.venv/bin/python -B -m scripts.writing_pep_adjacent_intake verify "$adjacent" "$initial" "$legacy" "$pep_repo"
.venv/bin/python -B -m scripts.writing_pep_prose_diff_intake verify "$diff" "$initial" "$legacy" "$adjacent" "$pep_repo"
.venv/bin/python -B -m scripts.writing_pep_prose_diff_extension verify "$extension" "$initial" "$diff" "$pep_repo"
```

To rebuild a receipt in a new directory, use `freeze` and `qualify` in that order with the same arguments. The first intake expects an existing output directory.

## Reproduce the detector pilot

Use the pinned offline generator environments and the original classifier snapshot, then build each stage in a new private directory. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONPATH=.` from `backend`. The two `generate` invocations use the previously pinned MLX Python environment. All other stages use the backend Python environment.

```sh
.venv/bin/python -B -m scripts.writing_center_deletion_transfer freeze "$run" "$extension" "$current_snapshot"
"$mlx_python" -B -m scripts.writing_center_deletion_transfer generate "$run" "$extension" "$current_snapshot" --family qwen
"$mlx_python" -B -m scripts.writing_center_deletion_transfer generate "$run" "$extension" "$current_snapshot" --family smol2
for phase in calibrate lock-holdout holdout report; do
  .venv/bin/python -B -m scripts.writing_center_deletion_transfer "$phase" "$run" "$extension" "$current_snapshot" || exit 1
done
.venv/bin/python -B -m scripts.review_writing_center_deletion_transfer "$run" "$extension"
```
