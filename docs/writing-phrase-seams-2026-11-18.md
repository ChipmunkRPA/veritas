# Can word-pair patterns locate a short AI copyedit?

The current passage check and the experimental word-pair check each located **0 of 3** short, meaning-preserving AI edits. Neither flagged any of the 12 ordinary controls. The experimental check found measurable spans elsewhere in every file, but its vocabulary rule excluded all 39 spans that could have matched a known edit. Lowering the score cutoff would recover none. [Aggregate receipt](evaluation-panels/writing-phrase-seams-2026-11-18.json)

## Source and test design

Six earlier English Wikipedia excerpts each had 13–15 sentences and an independently verified later revision from 2011–2014. The two ordinary controls per article were the untouched earlier excerpt and the later revision's exact focal sentence inserted into that same excerpt. The second version tests genuine archived wording in a fixed context. Only one of the six ordinary edits fit the same 2–7-word replacement scope as the AI examples. These six work groups came from earlier source qualification. They were excluded from fitting and calibration. [Archive qualification](writing-wiked-origin-collisions-2026-11-18.md) · [WikEd corpus terms](https://www.junczys.net/publications/pdf/mjd.poltal2014.draft.pdf) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

Two pinned Apache-2.0 editors received only the earlier focal sentence, with four attempts per article and editor. The direct rule required one interior replacement, two to seven changed words on at least one side, unchanged numbers and the same characters outside the replacement. Of 48 new replies, one passed the direct rule and two reproduced the archived sentence exactly. A separate locked rule took the first eligible replacement from a multiple-edit reply and inserted only that phrase into the untouched earlier sentence. It supplied eight more candidates. A reviewer checked the nine sentences for factual equivalence before seeing detector readings. Three qualified across three articles, one direct and two spliced. Two changed facts and four were uncertain. Archived wording collisions did not enter accuracy testing. [Pinned editor A](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) · [Pinned editor B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## How the check was reproduced

1. Casefold words and count them and their adjacent pairs across the first 600 unique original reports in `default/test/reports_gemini-2.5-flash.csv`. Skip source overlap of 12 normalized words with the assessment and calibration text. Only version-zero rows with zero labeled AI-token share enter the counts. Count pairs only across whitespace. [Pinned release](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) · [Data reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse) · [Original report source terms](https://gov-report-data.github.io/#license)
2. Read one document at a time with the app's text extractor and sentence reader. Consider one to seven consecutive words within a sentence, with one adjacent word on each side. All those words must occur at least three times in the reference corpus, and every gap must contain only whitespace. Score an ordered pair with `log((count(pair) + 1) / (count(left) × count(right) / token_count + 1))`. Subtract the stronger outer pair score from the median of the internal pair scores. Use zero as the internal score for single-word candidates. This pilot tests the word-pair calculation independently of published token-level models. [Research on source-aligned word labels](https://arxiv.org/html/2609.32625v1)
3. For each width, center the candidate on the median of the per-file maxima from 12 separate genuine works. Divide by 1.4826 times the median absolute deviation, with a 0.25 floor on the divisor. Keep the single unique file maximum only when it strictly exceeds the second-highest calibration-file maximum of 1.741920504295. This was a fixed experimental rule. Report a correct location only when it covers at least half of the changed non-whitespace characters and reaches 0.30 intersection over union against that change.

The editors ran offline with MLX 0.32.4 and MLX-LM 0.31.3, using seed 1701834, four-bit group-64 affine quantization, temperature 0.78, top-p 0.93 and a 110-token limit. The current passage check used Python 3.12.6, PyTorch 2.13.0 and two CPU threads. It evaluated the same 15 documents and 205 distinct isolated or contextual inputs with its pinned classifier. Its pooled, within-document and report-confirmed selections all gave the same result. [Pinned current classifier](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615) · [Current passage selection](writing-interval-localization-audit-2026-11-18.md)

## What failed

| Result | Current check | Word-pair check |
| --- | ---: | ---: |
| Views assessed | 15/15 | 15/15 |
| Exact edits located | 0/3 | 0/3 |
| Earlier excerpts flagged | 0/6 | 0/6 |
| Archived copyedit controls flagged | 0/6 | 0/6 |

Before applying the word-pair eligibility rules, 39 candidate spans across all three edited files could have matched the known changes. The whitespace rule alone left seven spans across one file. The reference vocabulary rule alone left zero. Exact text and focal ranges transferred through both TXT and DOCX extraction for all 15 views. These were source-position checks, separate from detector accuracy and checkout. The small panel cannot estimate a general detection or false-alarm rate. The experimental check adds no app evidence or confidence weight.

The receipt supports a public arithmetic check. The private input lock and result hashes identify the local run. Rerunning inference on the same examples requires the retained local inputs. The public files omit article text, revision identifiers, exact sample locations and model output text.

```sh
python3 -B -S - <<'PY'
import json
from pathlib import Path

receipt = Path('docs/evaluation-panels/writing-phrase-seams-2026-11-18.json')
result = json.loads(receipt.read_bytes())
panel, ranges = result['held'], result['candidate_geometry']
assert sum(result['generation']['direct_reply_outcomes'].values()) == 48
assert sum(result['generation']['blind_review'].values()) == 9
assert panel['earlier_excerpts'] + panel['archived_copyedits'] + panel['labeled_edits'] == 15
assert ranges['before_candidate_rules'] == 39
assert ranges['whitespace_rule_alone'] == 7
assert ranges['vocabulary_rule_alone'] == ranges['both_rules'] == 0
assert set(result['native_format_positions_verified'].values()) == {15}
for method in result['measurements'].values():
    assert method == {'assessed': 15, 'located_edits': 0, 'earlier_flags': 0, 'archive_flags': 0}
print('Aggregate counts and comparison scope agree.')
PY
```
