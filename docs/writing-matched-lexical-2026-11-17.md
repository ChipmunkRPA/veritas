# Learning word and phrase choices from matched rewrites

This experiment tests whether a small classifier can locate a short
rewrite in one uploaded document. It learns word, phrase and character
features from 5,434 aligned earlier and revised sentence pairs across
800 U.S. government reports. The original and edited versions have
the same subject and surrounding material. Each held file is scored
on its own. The paired fit measures the change in feature weights
between matched sentences. Separate arms test absolute phrase scores,
character patterns, a within-document comparison and agreement
between the two training generators.
[Source and label audit](writing-token-supervision-2026-11-17.md)
· [Research on generation and editing traces](https://aclanthology.org/2026.emnlp-main.1725/)

The fit uses independently authored code and aligned pairs from the
pinned commercial-use benchmark intake. The cited paper and its
restrictions were reviewed for methodology. The fitted data come from
OpAI-Bench and its verified published originals.

The 96 source-disjoint ordinary calibration reports each supply their
maximum sentence score. Their empirical 95th and 90th percentiles set
two internal operating points. On the 96 matched edited files, the
selection rewards two per useful location, with a deduction of four
per ordinary file flagged and one per misplaced highlight. The wider
absolute phrase reading won on development with 60 of 679 correct
locations and nine ordinary files flagged. The two points for every arm
were frozen before held evaluation.

## Held and transfer results

Each cell shows correct edit locations, followed by untouched files
flagged. A useful region covers at least half the labeled non-whitespace
characters and reaches 0.30 intersection over union. Each labeled
edit can count once. The comparison retains up to six candidates per
method and includes the current passage check in the combined result.

| Cases | Current app | Absolute phrase reading | Combined |
| --- | ---: | ---: | ---: |
| All retained held pairs | 11/911 · 10/133 | 25/911 · 16/133 | 36/911 · 25/133 |
| Held report pairs from the previously unused benchmark generator | 0/705 · 8/96 | 22/705 · 5/96 | 22/705 · 12/96 |
| Independent short paraphrases from two local editors | 5/149 · 0/55 | 0/149 · 2/55 | 5/149 · 2/55 |

The combined rule added 25 useful locations across the broad held set
and flagged 15 additional ordinary files. It added zero locations on
the separate 55-pair short-rewrite transfer and flagged two additional
ordinary files. The within-document arm added one transfer location
with one additional ordinary flag at its tighter point. Without a
cutoff, the preferred reading's highest sentence overlapped an edit in
14 of the 55 short-rewrite files. An eligible sentence drawn at random
would be expected to hit 11.36 of those files, given their label
positions. The lexical check stays in evaluation. The short-rewrite
transfer is separate from fit and calibration and was already used to
test the earlier model.

## Reproduce the comparison

The public receipts bind the native source rosters, trained coefficient
file and original text positions. Source text and fitted vocabulary
stay local. The independent reader reconstructs the operating points
and uses a different one-to-one assignment algorithm to count useful
locations, missed locations and ordinary alerts.

```sh
cd backend
panels=../docs/evaluation-panels
freeze="$panels/writing-matched-lexical-freeze-2026-11-17.json.gz"
for split in held transfer; do
  python3 -B -S -m scripts.writing_matched_lexical review "$freeze" \
    "$panels/writing-matched-lexical-$split-2026-11-17.json.gz"
done
.venv/bin/python -B -m scripts.writing_matched_lexical fit \
  "$plans/fit.json.gz" "$plans/calibration.json.gz" "$run"
.venv/bin/python -B -m scripts.writing_matched_lexical held \
  "$plans/held.json.gz" "$incumbent" "$run" "$run/held.json.gz"
.venv/bin/python -B -m scripts.writing_matched_lexical held \
  "$transfer_inputs" "$transfer_incumbent" "$run" "$run/transfer.json.gz"
```

Rebuild the exact source plans, incumbent measurements and transfer
inputs using the [prior sentence and word study](writing-token-supervision-2026-11-17.md#reproduce-the-local-comparison).
