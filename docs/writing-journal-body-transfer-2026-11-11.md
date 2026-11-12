# Paired writing checks on dated journal body revisions

Two journal catalogs supplied fresh dated revisions with exact author body-text witnesses. The freeze selected 40 version-2 records from each complete 2018–2019 catalog by DOI hash, before reading either article body. Both versions had to use CC BY 4.0 and match their DOI, ordered author list, and publisher date. The author work groups are distinct from the earlier F1000Research and PeerJ/Copernicus panels. Wellcome Open Research and Gates Open Research share a publishing platform with F1000Research. [Wellcome Crossref catalog](https://api.crossref.org/journals/2398-502X/works) · [Gates Crossref catalog](https://api.crossref.org/journals/2572-4754/works) · [Wellcome published XML archive](https://wellcomeopenresearch.org/published-xml-urls) · [Gates version policy](https://gatesopenresearch.org/about/policies)

The XML body reader omits marked bibliographic references and excludes quotes, figures, tables, and unsupported inline material. An eligible excerpt has a single material sentence edit with two exact unchanged sentences on both sides. It checks the exact body geometry and rejects 12-token overlap with eight prior evaluation packets, as well as overlapping recorded author names in selected works. The qualification replay admitted 10 Wellcome and 11 Gates works. Three unavailable XML files affected two work pairs. Four pairs failed matching author or date checks, two failed other XML metadata checks, and 51 lacked the exact edit geometry. Prior model training exposure is unknown.

The generator and scoring stages reuse the frozen paired-writing method on the new excerpts, with no previously scored inputs. Each of two Apache-2.0 generator snapshots makes one center-sentence rewrite per work. The scoring compares the current passage check, same-window agreement, paired earlier-versus-later odds, and the union of the paired center with same-window agreement on changed inputs. Useful localization requires at least 30% non-whitespace character intersection over union and 50% recall of the known replacement. The previously frozen replication rule requires at least six admitted cases per generator, two useful additions against the current check, no lost current locations, and at most one extra genuine false flag against agreement. [Prior method and pinned assets](writing-paired-revision-replication-2026-11-11.md)

The app scope replay also measures a prospective layout-only relaxation. It flattens paragraph separators for eligibility and checks that the resulting detector input hashes still match the original native windows. This measures possible extra coverage under the same paired rule. The release gate for this study requires at least two useful additions on distinct works and at most one extra false flag. Individual source files, excerpts, generator replies, and model scores stay local.

## Results

Both generator lanes admitted 16 of 21 attempted rewrites. Each lane reuses the same publisher work groups. The two checkpoints scored 264 distinct complete native inputs in 132 total CPU forwards, with zero reused responses. The independent numerical replay verified all 74 cases, their exact input hashes and source offsets, and the outcome counts. [Evaluation receipt](evaluation-panels/writing-journal-body-transfer-2026-11-11.json)

| Cohort | Admitted | Current useful | Same-window useful | Paired useful | Combined useful |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wellcome, Qwen | 8 | 1 | 1 | 1 | 1 |
| Gates, Qwen | 8 | 1 | 1 | 0 | 1 |
| Wellcome, SmolLM2 | 9 | 2 | 2 | 2 | 2 |
| Gates, SmolLM2 | 7 | 0 | 0 | 3 | 3 |
| Generated total | 32 | 4 | 4 | 6 | 7 |
| False flags on ordinary later revisions | 21 | 0 | 0 | 0 | 0 |

The combined method preserved all four useful current locations and added three. SmolLM2 passed the prior replication criterion on these work groups. The paired arm alone lost one Qwen location that current and same-window agreement detected. The 21 ordinary earlier versions also received zero flags from every arm. These counts describe exact one-sentence professional revisions from two journals on one platform.

The current local app eligibility admitted 23 of 32 generated pairs and 11 of 21 ordinary pairs. It identified five generated centers and zero ordinary centers. The layout relaxation admitted nine more generated pairs and six more ordinary pairs. It added one correct center and zero ordinary flags, short of the two-work release gate. The layout extension stays experimental. The existing app check provides evidence for a supplied earlier draft with zero authorship confidence weight.

An authenticated local route test used both installed checkpoints on one admitted generated edit and one admitted ordinary revision per journal. All four responses matched the study direction. Each comparison PDF and JSON showed the exact center and passed the source replay, with zero authorship confidence contribution.

## Replay

From the backend directory, supply the retained private run folder and the two exact detector snapshots. The source replay checks the original roster, file digests, and source selection. The scored replay independently rebuilds native input commitments and outcome counts without model inference. Use the same MLX runtime and offline snapshots identified in the prior paired-writing method for generation.

Keep `catalog-freeze-initial.json` for this recorded run. A historical input reader correction allowed the two prior JSON arrays to be compared alongside the six JSONL packets. The source replay requires the original roster and historical hashes to match exactly. Keep `scored/frozen-generation-controller.py` with the original generation receipts. A scoring controller correction writes the case inventory before hashing it. The generation receipt, controller, source and method remain hash-bound through the scoring freeze.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m scripts.writing_journal_body_holdout verify "$source"
.venv/bin/python -B -m scripts.writing_journal_body_transfer verify "$source/scored" "$source" \
  --current "$current" --second "$second"
```
