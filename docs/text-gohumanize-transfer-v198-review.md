# Independent review of the article-localization readout

The reviewer uses Python’s standard library to recompute the v196 readout from its committed inputs. It reads the v193 preparation and v196 transfer freezes, checks the frozen code and operating-point hashes, and reconciles all 54 article pairs and 324 planned cases with the published preparation counts. It checks the original text hashes, replacement offsets and paired all-human formatting controls against the retained inputs. Every batch needs the exact planned case sequence, a successful resource profile, and matching worker, resource and response-journal hashes.

The reviewer builds both plain-text views from original Unicode code points using its own paragraph scanner and source index mapping. It projects each proposal into the original non-whitespace characters and applies the unchanged v7 research points. It recomputes the four v101 masks and their metrics by arm and generator. An unavailable view makes any union that requires it unassessed. A saved empty response is assessed with zero scored characters. Unavailable constructions stay in the planned denominator, and mean character overlap uses only assessed cases.

The published v196 JSON must match the reconstruction in every field and value, including the ordered profile hashes, coverage and errors. Key order and indentation can differ. The reviewer prints an aggregate receipt only. The receipt contains whole-artifact hashes and counts. Keep the roster, source text, individual offsets, responses and native proofs private.

Independence covers the source mapping, mask construction, metrics and artifact checks. The reviewer does not import the app, scorers or model libraries. The retained model responses and their native proof hashes remain inputs. Replaying those records does not repeat tokenizer calls, neural forwards or calibration inference, verify the underlying tensors, or establish accuracy on another dataset.

Run from the repository root after all six v196 batches have completed and the aggregate readout has been saved.

```sh
python3 -B backend/scripts/review_text_gohumanize_transfer_v198.py \
  --expected docs/evaluation-panels/text-gohumanize-transfer-complete-v196.json
```

Run the synthetic tests separately.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  backend/tests/test_review_text_gohumanize_transfer_v198.py
```
