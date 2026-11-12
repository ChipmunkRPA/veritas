# Localizing eight-word edits across neighboring contexts

The two-context phrase check located one of 15 held eight-word AI inserts. The current three-sentence windows located zero. Applying local flank contrast retained the one useful location and flagged none of the 24 ordinary files. The current windows flagged six ordinary files across five work groups. The result supports a narrower search geometry and leaves the candidate in research. [Results and independent replay](evaluation-panels/writing-phrase-context-stability-2026-11-12.json)

## Frozen comparison

The test compares three ways to rank exact eight-word spans in a single uploaded excerpt. The first uses the prior one-context persistence score. The second requires the same span to score across both preceding-plus-focus and focus-plus-following inputs. The third subtracts the stronger four-word flank before requiring agreement across both inputs. Both neighboring contexts preserve the focus sentence exactly. They share classifier weights, so their agreement measures context stability. Each span also needs four available words on either side. The score uses the smaller of the mean over all eight and the mean over its central four. The current comparison retains qualifying three-sentence windows in source order. [Prior phrase comparison](writing-email-phrase-persistence-2026-11-11.md)

The experimental arms select at most one uniquely strongest span. Eleven previously used archived work groups supply ordinary originals and matched same-work phrase substitutions for each operating point. The point is the second-largest source maximum and must be strictly exceeded. Useful localization requires at least 0.30 non-whitespace intersection over union and 0.50 recall against the known changed range. The trial also counts whether each proposed geometry could ever meet both overlap conditions before applying scores. Advance requires two new useful work groups across both generators, at most one new false human group, and six admitted edits per generator.

## Sources and edit construction

The preflight qualified twelve articles published in 2020, from the Kubernetes English blog archive pinned in August 2021. All twelve works have recorded bylines separate from the two older archive studies and from each other. A source check screened exact normalized 12-word fragments against twelve previous text inventories. The two five-sentence excerpts per work have nonoverlapping archived source ranges. The untouched excerpt and a same-work phrase substitution using the second excerpt serve as ordinary controls. Source license text was verified from the pinned archive. The archived date and source checks do not establish absence from model pretraining. [Kubernetes website license](https://github.com/kubernetes/website/blob/e20b8746f2444ae9fe58f49fd0f0acbfbabbb938/LICENSE) · [Prior archive qualification](writing-archived-passage-transfer-2026-11-12.md)

The first frozen lane made 24 direct attempts to paraphrase exactly eight words and admitted zero responses. A second freeze kept the same source file byte for byte and used a completed full-sentence rewrite. It selected a contiguous eight-word fragment nearest the original phrase's relative center, with a deterministic leftmost tie-break. Only files preserving all surrounding text and native sentence boundaries entered scoring. The same fixed score and calibration rule applied to both lanes. Human controls had prior project exposure from the first lane. The synthetic files are controlled eight-word insertions that may disrupt grammar or meaning. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM2 model card and terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Held outcomes

| At the frozen review points | Current windows | One-context phrase | Two-context phrase | Two-context with flank contrast | Current plus flank contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful locations on 15 generated edits | 0 | 0 | 1 | 1 | 1 |
| Useful locations on seven Qwen edits | 0 | 0 | 1 | 1 | 1 |
| Useful locations on eight SmolLM2 edits | 0 | 0 | 0 | 0 | 0 |
| Flagged untouched files, out of 12 | 3 | 0 | 0 | 0 | 3 |
| Flagged same-work ordinary substitutions, out of 12 | 3 | 1 | 1 | 0 | 3 |
| Work groups with an ordinary file flagged | 5 | 1 | 1 | 0 | 5 |

The three-sentence windows could not reach the required overlap on any of the 15 edits, even before applying the score rule. The exact phrase candidates covered a viable region in all 15. At the retrospective budget of five false held work groups, one-context and flank contrast each located three edits, two-context persistence located two, and current windows located zero. Those figures compare held ranking and provide no new operating point. The one frozen gain came from one work and one generator, below the advancement requirement.

All three operating points reproduced numerically from the first lane. The second lane used 61 CPU forwards over 132 distinct calibration and 219 distinct held native inputs, plus the paired direct-output checks included in that forward count. A separate replay checked 230 raw archive sentence ranges, 357 held phrase positions and 195 arm decisions. Each replay re-derives model input strings and coordinates from the exact fixture and separately recomputes the token reductions. This trial covers brief pasted passages in English technical prose. It supplies no authorship confidence weight.

## Replay

The [runner](../backend/scripts/writing_phrase_context_stability.py) accepts a new private run directory and the pinned local primary classifier snapshot. The previously retained Kubernetes Git archive and source rosters supply the input set. The two generation commands require the pinned MLX environment on Apple Metal. The app environment contributes the existing text extraction and segmentation dependencies. The [independent reviewer](../backend/scripts/review_writing_phrase_context_stability.py) checks archived ranges, recomputes the token-to-word reduction, verifies the fixture coordinates and replays all location decisions without loading weights. Source prose and private response packets stay local.

```sh
cd backend
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B -m scripts.writing_phrase_context_stability freeze "$run" "$snapshot"
for family in qwen smol2; do
  "$mlx_python" -B -c 'import runpy, sys; sys.path.append(".venv/lib/python3.12/site-packages"); runpy.run_module("scripts.writing_phrase_context_stability", run_name="__main__")' generate "$run" "$snapshot" --family "$family" || exit 1
done
for phase in fixtures calibration holdout; do
  .venv/bin/python -B -m scripts.writing_phrase_context_stability "$phase" "$run" "$snapshot" || exit 1
done
.venv/bin/python -B -m scripts.review_writing_phrase_context_stability "$run" "$receipt"
```
