# Eight-word writing localization on archived Python PEPs

The flank-adjusted writing check's earlier gain did not repeat on archived Python enhancement proposals. It located zero of 29 controlled eight-word AI inserts and added two false-alarm work groups to the current comparison. The two-context check without subtraction found one insert, with five false-alarm groups. The model, phrase lengths, neighboring contexts, operating points, and advancement rule are the same as in the earlier test. [Earlier article study](writing-phrase-context-stability-2026-11-12.md) · [Independent result ledger](evaluation-panels/writing-pep-phrase-replication-2026-11-12.json)

## Source and control selection

The source-only roster uses 20 works at the pinned December 2019 Python PEP Git snapshot. Each file has a named credited author and an explicit public-domain or CC0 notice in its own copyright section. It provides two nonoverlapping five-sentence excerpts with exact raw ranges. The second supplies an ordinary eight-word substitution for the first. The qualifier excludes earlier PEPs, recorded author overlap with earlier PEPs and the previous phrase study, and normalized twelve-word fragments shared with 18 retained prior inventories. Author credits identify published works. The method does not establish individual sentence authorship or absence from model pretraining. [PEP process and licensing](https://peps.python.org/pep-0001/#copyright) · [Earlier PEP qualification](writing-pep-editorial-intake-2026-11-11.md)

Two pinned commercially permitted generators each make one full-sentence rewrite attempt per work. The same deterministic rule chooses one punctuation-free, contiguous eight-word fragment near the original relative position. Surrounding characters and all five native sentence boundaries must stay exact. A retained insertion can disrupt the sentence's syntax or meaning. The paired unchanged and ordinary controls test false alarms on the same subjects and source structure. [Frozen model revisions and licenses](writing-phrase-context-stability-2026-11-12.md#sources-and-edit-construction)

The first comparator uses the current three-sentence windows. The experimental arms rank eight-word candidates using one contextual read, two contextual reads, and the lower of two contextual responses after subtracting the stronger nearby four-word response. A useful location covers at least half the known changed characters and reaches 0.30 non-whitespace intersection over union. The unchanged gate needs gains on two work groups across both generators, no more than one additional genuine group flag, and at least six eligible edits per generator. The archived 11-group selection role supplies the frozen points. No PEP response can modify those points.

## Held results

Both generators completed their 20 attempts. Thirteen from the first and 16 from the second met the unchanged fragment rule. Forty matched ordinary files completed, including both controls for each of 20 source works.

| At the frozen points | Current windows | One-context phrase | Two-context phrase | Two-context with flank contrast | Current plus flank contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful locations on 29 generated edits | 0 | 0 | 1 | 0 | 0 |
| Flagged untouched files out of 20 | 2 | 0 | 2 | 2 | 3 |
| Flagged same-work substitutions out of 20 | 2 | 0 | 3 | 1 | 3 |
| Work groups with an ordinary flag | 2 | 0 | 5 | 3 | 4 |

The current window geometry could not reach useful overlap on any of the 29 inserts. Each exact-phrase arm could propose a useful range on all 29 before its score was applied. The sole two-context hit came from the second generator. It came from a different generator than the one hit in the earlier Kubernetes test, with too many ordinary work groups flagged here to support adoption. The combined arm missed both the gain and false-alarm requirements. These results apply to controlled eight-word pasted fragments in English technical prose. The candidate remains experimental.

At the retrospective budget of two false held groups, one-context persistence and two-context persistence each located one of 29. The flank-adjusted check located zero. One-context persistence also ranked one useful location above every genuine group. Those comparisons describe ranking on this panel and establish no new operating point. Independent replay verified 200 archived sentence ranges, 132 original calibration inputs, 385 new held inputs, 877 candidate positions, and 345 held decisions. The held classifier made 66 CPU forwards.

## Reproduce

Use the retained pinned PEP Git archive, original phrase-study private input and response ledgers, and the same pinned local classifier and Apple Metal environments. Start a new private run directory. The native reviewer replays exact source locations and both control types, recomputes the prior points from the 11 committed calibration groups, verifies each native model input, and independently reconstructs the held decisions without model inference. The public receipt excludes source prose and generated text.

```sh
cd backend
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B -m scripts.writing_pep_phrase_replication freeze "$run" "$snapshot"
for family in qwen smol2; do
  "$mlx_python" -B -c 'import runpy, sys; sys.path.append(".venv/lib/python3.12/site-packages"); runpy.run_module("scripts.writing_pep_phrase_replication", run_name="__main__")' generate "$run" "$snapshot" --family "$family" || exit 1
done
for phase in fixtures holdout; do
  .venv/bin/python -B -m scripts.writing_pep_phrase_replication "$phase" "$run" "$snapshot" || exit 1
done
.venv/bin/python -B -m scripts.review_writing_pep_phrase_replication "$run" "$receipt"
```

The public ledger verifier checks the earlier point commitment and held decision arithmetic using only the Python standard library. It consumes the recorded per-case dispositions. The native reviewer checks the source bytes and independently recomputes those dispositions.

```sh
python3 -B -I -S backend/scripts/review_writing_pep_phrase_ledger.py \
  docs/evaluation-panels/writing-pep-phrase-replication-2026-11-12.json \
  docs/evaluation-panels/writing-phrase-context-stability-2026-11-12.json
```
