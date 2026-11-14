# Using surrounding sentences to locate a known AI polish

Adding a typed context check to the existing writing check localized **13 of
40** known sentence edits, up from **10 of 40**, and flagged **12 of 20**
untouched excerpts, up from **11 of 20**. The three additional locations
represent three different reports and both generators. The second version
of the candidate, which saw each sentence by itself, added one location
and two untouched flags. The contextual candidate remains a research check.
[Frozen response receipt](evaluation-panels/writing-typed-decision-transfer-2026-11-13.json)

## Inputs and measurement

The candidate gives the same compact classifier a fixed two-choice question
about authorship of the target sentence. One version supplies only the
target. The other includes the exact previous and next sentences. Both
versions run the two answer orders, average the AI responses, and pick the
unique highest of three measurable interior sentences. Each version has
its own internal operating point. The macOS test used a matching FP16
conversion with pinned model assets and an offline MLX runtime.
[Apache-2.0 source checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual/tree/052592a15d198d9ad47da779604259b10b47b7aa) ·
[Pinned FP16 conversion](https://huggingface.co/aac6fef/laya-multilingual-mlx/tree/f2b4faf51023039425946074e2cf1361d2db11d5)

The 35 earlier historical journal works each supplied two authentic
versions. Their 70 excerpts were reconstructed after a reboot and matched
against the prior study's text hashes, native ranges and whitespace
offsets. All 28 F1000Research XML versions also matched the original XML
hashes. Both arms assessed all 35 works. Each cutoff was the second-highest
per-work maximum, set before scoring any holdout candidate input.
[Gates and Wellcome archive intake](writing-journal-body-transfer-2026-11-11.md) ·
[F1000Research publisher replay](writing-long-position-transfer-2026-11-12.md#independent-publisher-replay)

The fresh holdout used 20 public GAO works with one untouched five-sentence
excerpt and two exact-layout versions in which either Gemini 2.5 Flash or
GPT-5.4 Nano polished one labeled sentence. Admission excluded listed
prior works, works in the retained Qwen shard, and overlaps of normalized
12-word sequences. The 20 admitted works came from 41 examined of 118
initially eligible works. Target positions one, two and three occurred in
seven, six and seven works. Both methods assessed every excerpt in this
holdout. Exact earlier retrospective packets were unavailable.
[Pinned Apache-2.0 benchmark](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[Underlying report terms](https://gov-report-data.github.io/#license)

The comparison re-ran the existing classifier's native three-sentence
contexts at the research operating point used in the earlier studies.
A useful location covers at least half of the known edit's non-whitespace
characters and reaches 0.30 intersection over union with the edit.
The fixed advancement rule required at least four additions across
three works and both generators, at most two additional flags of untouched
excerpts, and coverage of at least 36 edited cases.

| Fresh held outcome | Existing check | With target only | With target and neighbors |
| --- | ---: | ---: | ---: |
| Gemini edits located | 6/20 | 6/20 | 7/20 |
| Nano edits located | 4/20 | 5/20 | 6/20 |
| All known edits located | 10/40 | 11/40 | 13/40 |
| Untouched excerpts flagged | 11/20 | 13/20 | 12/20 |

The context arm ranked the known edited sentence highest in 24 of 40
submissions before applying its cutoff. The isolated arm did so in 11
of 40. The context arm demonstrates better localization potential and
fell one edit short of the frozen advancement rule. These results are
specific to short English excerpts from government reports, with operating
points derived from journal prose. Performance on email, full articles,
other genres and languages remains unmeasured. These readings contribute
zero to the app's authorship confidence.

## Replay

The portable evaluation packet contains source commitments, ranges,
whitespace offsets, and aggregate classifier readings. Original prose,
rewritten excerpts, and model weights stay in local storage. The replay
needs only the Python standard library.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.writing_typed_decision_transfer review \
  docs/evaluation-panels/writing-typed-decision-transfer-2026-11-13.json
```

For a native rerun, restore the verified benchmark files, authentic journal
pair roster, and exact publisher XML in local directories. The two Python
environments provide the pinned backend and MLX dependencies recorded in
the receipt. The candidate scored 309 unique calibration inputs and 252
unique holdout inputs in two answer orders. The CPU comparator scored
220 unique native contexts.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.writing_typed_decision_transfer freeze "$run" \
  --cache "$report_cache" --journal-pairs "$journal_pairs" --publisher-xml "$publisher_xml"
"$mlx_python" -B -m scripts.writing_typed_decision_transfer calibration "$run" --snapshot "$candidate"
backend/.venv/bin/python -B -m scripts.writing_typed_decision_transfer point "$run"
"$mlx_python" -B -m scripts.writing_typed_decision_transfer held "$run" --snapshot "$candidate"
backend/.venv/bin/python -B -m scripts.writing_typed_decision_transfer primary "$run" --snapshot "$incumbent"
backend/.venv/bin/python -B -S -m scripts.writing_typed_decision_transfer publish "$run" --output "$receipt"
```
