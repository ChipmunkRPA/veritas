# Dependency patterns around single-sentence AI polishes

The experiment tests whether a localized polish changes the sentence's
grammatical structure relative to its actual neighbors. Recent research found
stronger stylometric effects for generated writing than for AI copy edits and
included maximum dependency depth among its measurements. This study measures
the location of a known edit within a single five-sentence upload. [AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not](https://aclanthology.org/2026.emnlp-main.1725/)

A 12 MB English tagging and parsing checkpoint runs on CPU. One arm measures
length, punctuation and the proportions of 13 parts of speech. The second arm
adds dependency depth, attachment distance, passive constructions, branching,
and the directional counts of eight groups of grammatical relations. Each
reading subtracts the mean features of the two native neighbors. The fit
learns from same-source edited and original sentence differences. At evaluation
the scorer sees a single text and ranks its interior sentences. No reference
version, words as categorical features, source identifier or incumbent score
enters the candidate at inference. [MIT English parser wheel and model card](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0)

Fit covered 46 edit pairs from 23 of 24 earlier report works. Each arm's
internal operating point was the second-highest work maximum among 32
assessable works from 35 earlier dated journal revision pairs. Evidence
coverage and points were frozen before held readings. These scores are
research measurements with zero contribution to authorship confidence.
[Dated revision controls](writing-bidirectional-transfer-2026-11-13.md)

Fresh assessment used 20 previously unused GAO report works with exact
five-sentence matching for the untouched source and the two independently
polished versions. Target positions one, two and three had seven, six and
seven works. The order excluded prior report works and works in the Qwen
shard. Texts with shared normalized 12-word sequences were excluded.
The intake considered 30 of 138 initially eligible works, including five
without a shared isolated edit, three with incompatible source geometry,
and two with source phrase overlap. The final source commitments also
showed no exact group overlap with any pre-existing writing ledger. [Pinned Apache-2.0 publisher release](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

## Mac results

| Fresh held outcome | Existing context check | With POS and punctuation | With grammatical dependencies |
| --- | ---: | ---: | ---: |
| Gemini polishes located | 12/20 | 13/20 | 12/20 |
| Nano polishes located | 10/20 | 10/20 | 10/20 |
| All known edits located | 22/40 | 23/40 | 22/40 |
| Untouched excerpts flagged | 12/20 | 13/20 | 12/20 |

Both candidates assessed all 40 new edits and all 20 new originals. The
dependency arm returned no locations at its frozen operating point. The
POS arm found one additional edit and flagged one additional original.
Neither arm added an accurate location on the 120 previously exposed GAO
edits. Coverage was four of eight work groups for the genuine forum versions,
four of six for USWDS versions, and zero for the eLife three-sentence controls.
At least two eligible centers are required to choose a unique best one.
Both checks remain offline.

The comparison uses the first six native context ranges at the incumbent
research point. A useful range must cover at least half of the edited
non-whitespace characters and have at least .30 intersection over union
against the publisher-labeled edit sentence. These results measure one
sample of localized English public-report copy edits. The high original
flag rate in this study underscores the limit of using a highlight to
infer authorship. Neither result estimates a population-wide error rate.

## Replay and local rebuild

The public receipt retains rounded aggregate parser features, source
commitments, position geometry, frozen fit and local incumbent responses.
It includes the fitted numerical coefficients and omits original passages,
individual token identities and parser weights. The replay uses the Python
standard library and reproduces operating points, source counts, location
decisions and advancement. Refit coefficients allow a small platform
arithmetic tolerance while location decisions must match exactly.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.review_writing_dependency_transfer \
  docs/evaluation-panels/writing-dependency-transfer-2026-11-13.json
```

For a source rebuild, use an empty local directory, the pinned report release
and the retained local case roster and incumbent measurements from the
two-sided masked-word run. Install spaCy 3.8.7 and the pinned English parser
in an isolated overlay accessible through `PYTHONPATH`. The runner verifies
the original wheel digest and actual package versions. The primary stage
uses the pinned production snapshot entirely offline.

```sh
export PYTHONPATH=backend:"$parser_overlay" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=backend/.venv/bin/python
$python -B -m scripts.writing_dependency_transfer freeze "$run" \
  --cache "$report_cache" --previous "$bidirectional_run" --wheel "$parser_wheel"
$python -B -m scripts.writing_dependency_transfer development "$run"
$python -B -m scripts.writing_dependency_transfer point "$run"
$python -B -m scripts.writing_dependency_transfer held "$run"
$python -B -m scripts.writing_dependency_transfer primary "$run" \
  --previous "$bidirectional_run" --snapshot "$primary_snapshot"
$python -B -m scripts.writing_dependency_transfer finish "$run" --output "$run/receipt.json"
```
