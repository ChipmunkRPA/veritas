# Rare words and punctuation in short writing edits

An open-vocabulary variant restored usable locations for all three edits that the earlier word-pair check could not represent. It could also represent all four CDC transfer edits. The new check still located **0 of 7** verified edits. The first usable span ranked 28th or lower in every file, before applying the fixed score cutoff. On 48 ordinary views, the current check flagged three and the candidate flagged 20, adding alerts in 12 previously unflagged work groups. [Aggregate receipt](evaluation-panels/writing-open-vocabulary-2026-11-18.json) · [Earlier word-pair pilot](writing-phrase-seams-2026-11-18.md)

## What changed in the experiment

The candidate scored every contiguous one- to seven-word span within a native sentence, including positions next to punctuation and at sentence boundaries. It used Unicode compatibility normalization, casefolding and digit normalization for its counting keys. Exact character positions came from the submitted text. For every adjacent pair, it calculated the smoothed log ratio of observed to expected counts for both the whole words and the last two characters of the left word paired with the first two of the right. Counts were conditional on the gap class, with separate sentence boundary markers. The whole-word contribution was weighted by `m / (m + 8)` for the smaller of the two marginal token counts. The pair score was limited to −4 to 4. The span score was the median of its interior scores minus the stronger of its two outer scores, with zero for a one-word span's interior. This independently implemented calculation uses no pretrained weights. [Research on open-vocabulary word models](https://aclanthology.org/N18-1130/)

The reference counts came from the first 600 unique version-zero original government reports in the same benchmark release as the earlier pilot. The fit skipped any exact sequence of 12 normalized words shared with an evaluation or calibration excerpt. At each width, the candidate centered scores on the median of 12 separate ordinary file maxima and divided by 1.4826 times their median absolute deviation, with a 0.25 divisor floor. The frozen point was the second-highest normalized maximum, 2.630640856318. The candidate selected at most one uniquely highest span above it. A useful location required at least half the changed non-whitespace characters and 0.30 intersection over union. [Pinned report data and reuse terms](writing-phrase-seams-2026-11-18.md#how-the-check-was-reproduced)

## Scope and results

The 55 views came from three retained English panels. The seven verified edits include five small edits and two broader CDC transfer edits, across two editor families. The 48 ordinary views include 27 earlier excerpts and 21 archived revisions or corrections. We excluded a Turing Way group with identical published and generated wording, and counted a duplicate CDC earlier view once. The inputs were held out of fitting for this trial and had been evaluated previously with other methods. Archive records establish that wording existed at a given date. They cannot establish each contributor's writing process. [Wikipedia copyedits](writing-phrase-seams-2026-11-18.md#source-and-test-design) · [Turing Way revisions](writing-turing-editorial-history-2026-11-18.md) · [CDC edits and erratum](writing-mmwr-minimal-edits-2026-11-18.md)

| Outcome | Current check | Open-vocabulary check |
| --- | ---: | ---: |
| Verified edits located | 0/7 | 0/7 |
| Verified edit files flagged at a wrong location | 1/7 | 4/7 |
| Earlier ordinary files flagged | 2/27 | 13/27 |
| Archived ordinary files flagged | 1/21 | 7/21 |
| Work groups with an ordinary alert | 2/27 | 13/27 |

The candidate scored 182,822 native spans. None of the seven edits had a usable span in its first six results, even without a cutoff. With the target word count supplied only for analysis, none of four generated edit locations or 11 archived edit locations ranked first in the shared one-, two- and four-word strata.

The new check flagged 13 of 40 short ordinary views and seven of eight full-length CDC ordinary views. The current check flagged two and one. Longer files offered more chances for an extreme score, but the excess alerts were also present in the 12–15-sentence views. Every flagged archived revision had already been flagged on its matched earlier version. These results distinguish a workable search range from a useful ranking and leave the candidate outside the app.

The receipt records grouped counts and run-level digests. It omits source text and positions, contributor identities, and per-file hashes. The exact native rerun requires the private inputs. The public counts can be checked with standard-library Python.

```sh
python3 -B -S - <<'PY'
import json
from pathlib import Path

path = Path('docs/evaluation-panels/writing-open-vocabulary-2026-11-18.json')
data = json.loads(path.read_bytes())
panels = list(data['panels'].values())
assert sum(p['verified_edits'] for p in panels) == 7
assert sum(p['earlier'] + p['archived'] for p in panels) == 48
assert [sum(p['ordinary_flags'][m] for p in panels) for m in ('current', 'candidate')] == [3, 20]
assert data['ordinary_alert_groups']['candidate_only'] == 12
assert data['candidate_ranking']['verified_edits_in_top_six'] == 0
assert sum(g['verified_edits'] for g in data['matched_widths'].values()) == 4
assert sum(g['archived_edits'] for g in data['matched_widths'].values()) == 11
print('Aggregate counts and comparison scope agree.')
PY
```
