# Relative paragraph review on fresh science explainers

Use the exact v289 source split and all 78 planned rows. Development has 13 originals, 12 eligible rewrites from 10 source groups and 14 unavailable attempts. Confirmation has 13 originals, 10 eligible rewrites from eight groups and 16 unavailable attempts. Recheck every original and mixed Unicode interval, ordered role assignment, and absence of exact normalized 12-token overlap across the two roles. No replacement is inferred for an unavailable attempt. [Sealed inputs](evaluation-panels/text-elife-transfer-rewrites-complete-v289.json)

Run the same current CPU classifier, sentence context, calibration map and baseline research point used by v275. The current-only worker invokes the exact native current proposal method with token and coverage binding. It leaves the separate interval encoder and head unloaded. Each role has three exclusive resource-guarded batches of 15, 15 and 9 planned rows. Complete all development batches and seal the paragraph point before starting confirmation. The point code reads only the separate journals for the 13 development originals. It requires at least 12 fully covered originals. Every confirmation worker verifies the seal against those original-only receipts before model loading. [Current passage runtime](text-elife-passage-score-v275-method.md)

Split the submitted text at its paragraph boundaries. Average the highest saved raw current response covering each non-whitespace character in each paragraph. Require full article coverage. The paragraph signal is the leading mean minus the runner-up mean. Equal leading means give zero and cannot pass. Seal the second-largest lead observed among the fully covered development originals, including ties as zero. Add one whole-paragraph highlight only when the existing check is quiet and the unique leader's margin strictly exceeds the sealed point. The rule receives the submitted text and its own current responses. The frozen generated interval is used only after scoring to evaluate localization. [Retrospective ranking diagnostic](text-elife-paragraph-rank-v281-result.md)

On confirmation, require at least three additional useful locations at character intersection over union of at least 0.30, across at least two distinct source groups and with at least one from each generator. Permit one additional flagged original, at most two additional mixed paragraphs with no generated overlap, and at most 5% incremental highlighting of known unchanged characters across the ten eligible mixed inputs. Require full coverage on at least 12 of 13 originals and 9 of 10 eligible rewrites. Report the individual criteria, both families, coverage, useful and false highlights, and all unavailable attempts. Keep exact output intervals and source information local. This study applies to constructed science-explainer examples and supplies no public confidence threshold or serving change.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 preflight
backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 freeze
for batch in 0 1 2; do
  backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 run development "$batch" || break
done
backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 seal
for batch in 0 1 2; do
  backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 run confirmation "$batch" || break
done
backend/.venv/bin/python -B -m scripts.text_elife_relative_paragraph_v292 readout
```
