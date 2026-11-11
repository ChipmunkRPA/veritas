# Passage evidence on the sealed eLife rewrite cohort

Score the exact v273 cohort in its sealed source order. Each of five eLife articles contributes one original and one attempt per generator family. The cohort has five original controls, five eligible rewrites from one generator, one from the other, and four unavailable attempts. Recheck every mixed article against its v272 original and require its half-open Unicode interval to preserve the exact original prefix and suffix. Carry unavailable attempts into the readout without an inference input. [Sealed rewrites](evaluation-panels/text-elife-rewrites-complete-v273.json) · [Source intake](evaluation-panels/text-elife-source-intake-complete-v272.json)

Use the exact v260 guarded worker, response extraction, current passage classifier, interval head, calibration maps and research operating points. Run two sequential batches of three and two source groups. Score the current check on the ordinary paragraph-preserving view and the interval head on the whitespace-canonical view, projecting each method's proposals back to the same source coordinates. Compare those two masks individually. The inherited worker may also record canonical current responses when the two views differ. No point, model or method is selected from this cohort. [Prior passage runtime](text-opai-bench-score-v260-method.md) · [Frozen operating points](evaluation-panels/writing-spans-v7-result.json)

Report each generator's denominator and the five shared human controls. Count fully assessed, partially assessed and unavailable cases on non-whitespace proposal positions. A complete direction match requires full coverage and a highlight that intersects the known generated paragraph for a mixed article, or no highlight for a human control. A useful location requires character intersection over union of at least 0.30. A recovered interval covers at least half its generated characters. Also count false review of unchanged originals and highlighted human characters in both original and mixed articles.

Compare raw positional discrimination on the same covered positions for both methods. Each position receives the largest raw response among proposals covering it. Within each mixed article with jointly covered human and generated characters, report the probability that a randomly chosen generated position ranks above a randomly chosen human position, giving ties half credit. Average eligible articles equally within each family. This rank statistic measures localization on covered positions and does not represent a confidence score. The interval head's partial coverage and the second family's one eligible rewrite limit interpretation.

The comparison does not calibrate a new confidence score, run on customer data, or alter serving. Run without restarting a completed or interrupted batch. The public readout contains counts and commitments. Source text and native responses remain in the private evaluation directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.text_elife_passage_score_v275 preflight
backend/.venv/bin/python -B -m scripts.text_elife_passage_score_v275 freeze
backend/.venv/bin/python -B -m scripts.text_elife_passage_score_v275 run 0
backend/.venv/bin/python -B -m scripts.text_elife_passage_score_v275 run 1
backend/.venv/bin/python -B -m scripts.text_elife_passage_score_v275 readout
```
