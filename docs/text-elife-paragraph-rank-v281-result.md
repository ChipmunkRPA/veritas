# Paragraph ranking on archived article scores

The rewritten paragraph ranked uniquely first in **3 of 6** mixed articles. One of those targets already ranked first in the corresponding original, and the other two wins came from different rewrites of the same source. Thus **one of the five source groups** supplied a new top-ranked target. Rank improved in four pairs and held steady in two, with no top ties. The original target ranked first in one of five human controls. [Fixed diagnostic](evaluation-panels/text-elife-paragraph-rank-rule-v281.json) · [Aggregate](evaluation-panels/text-elife-paragraph-rank-complete-v281.json)

The diagnostic used the existing raw responses. It took the highest response covering each non-whitespace character, averaged those values within each frozen paragraph and compared that paragraph with the other paragraphs in the same article. All 11 prepared articles had full coverage. Four attempted rewrites remained unavailable. The same-source repetition and the high-ranking original show how broad character ranking can overstate the precision of a single suggested passage. Any future customer-facing rank rule needs fresh controls and rewrites. [Prior frozen scoring](text-elife-passage-score-v275-result.md)

The detailed local replay contains per-source predictions. Store it in an owner-only evaluation folder and publish only aggregated counts. The command requires the exact private v275 observations and v272 originals.

```sh
PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/python -B \
  -m scripts.text_elife_paragraph_rank_v281 readout docs/evaluation-panels/text-elife-paragraph-rank-rule-v281.json
```
