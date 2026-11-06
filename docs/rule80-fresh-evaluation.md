# Fresh evaluation of the frozen 80% headline rule

Policy update on 2026-11-05: current customer reports present finding-level tendencies and scoped confidence estimates without a public screening-threshold claim. This study remains a historical 80% evaluation; its figures must not be presented as validation of the current presentation, later model changes, or another operating point. The exact prior production files are retained in the local trial's `frozen-production-source.tar.gz`. Reproduction of this historical study requires those archived sources; the current-code audit deliberately rejects the changed source tree.

Completed locally on 2026-11-04. The result does **not** establish reliable original-content conclusions across the tested edit types. Localized image edits and face swaps exposed false-original headlines; article headlines remained entirely inconclusive.

The scoring code, installed weights, calibration files, coverage safeguards, policy thresholds, and `80-percent-leaning/v2` customer rule stayed unchanged during valid inference. The full production pipelines ran through publication in isolated storage. No fitting or model replacement occurred.

## Sample and license audit

- 24 genuine and 24 AI-generated/edited source inputs per media type: 192 source inputs in total.
- Audio used three conditions on the same 48 recordings: clean, equal-power noise (0 dB SNR), and background music (5 dB SNR). This produced 288 completed analyses overall, with no pipeline failures.
- Images combined 12 genuine/12 fully generated examples with 12 genuine/edit pairs. Videos combined 12 genuine/generated pairs with 12 target/face-swap pairs. Articles used 24 human/generated topic pairs from two datasets.
- Source selection was fixed before each media's first model inference and excluded retained local source files/groups/topics. All 35 exclusion manifests were rehashed; no selected input or original audio hash repeated their recorded hashes. Training-set overlap remains unknown.
- All 18 enabled model entries and their dependencies declared commercial permission: 10 Apache-2.0, seven MIT, and one CC-BY-4.0. All 44 installed checkpoint/configuration files were verified against the frozen hashes. This records publisher-declared permissions rather than independently resolving training-data rights.
- Every published result was rehashed and its customer headline independently recomputed. All 409 production source/calibration/report files matched the freeze.

## Frozen statistical-rule results

Each row has 24 genuine and 24 AI inputs. False alarms and wrong-original rates use their respective 24-input class; inconclusive uses all 48. No-flag counts include inconclusive AI, wrong-original AI, provenance-led outcomes and failures. They preserve the frozen rule-only definition.

| Input | False alarms | AI called original | AI without an 80% statistical flag | Inconclusive | AI declarations in file credentials |
|---|---:|---:|---:|---:|---:|
| Images | 0/24 (0.0%) | 6/24 (25.0%) | 15/24 (62.5%) | 9/48 (18.8%) | 4/24 |
| Videos — visual content | 0/24 (0.0%) | 5/24 (20.8%) | 21/24 (87.5%) | 29/48 (60.4%) | 2/24 |
| Audio — clean | 0/24 (0.0%) | 0/24 (0.0%) | 6/24 (25.0%) | 17/48 (35.4%) | 0/24 |
| Audio — strong noise | 1/24 (4.2%) | 0/24 (0.0%) | 11/24 (45.8%) | 33/48 (68.8%) | 0/24 |
| Audio — music | 1/24 (4.2%) | 0/24 (0.0%) | 5/24 (20.8%) | 22/48 (45.8%) | 0/24 |
| Articles | 0/24 (0.0%) | 0/24 (0.0%) | 24/24 (100.0%) | 48/48 (100.0%) | 0/24 |

The four images and two videos with AI declarations received provenance-led AI conclusions. Counting these as service alerts, AI content with **no AI alert of any kind** was 11/24 images (45.8%) and 19/24 videos (79.2%). The audio/article counts remain the same. These supplementary service-level rates do not replace the frozen statistical-rule rates above.

## What the results establish

- Fully generated images received 9/12 statistical AI flags. None of the 12 localized image edits received a statistical AI headline; four were identified through file credentials, six were called original, and two were inconclusive.
- None of the 12 face swaps received a statistical AI headline: five were called original and seven were inconclusive. Four swap results had a completed face-specific check with zero effective weight in the combined estimate, including two false-original cases. This integration/coverage issue needs investigation; raw inspection scores alone cannot supply a calibrated replacement.
- Original-content tendencies were correct on 20/26 images (76.9%) and 9/14 videos (64.3%) in this balanced sample. One known face swap received 98.0% original-content confidence. The observed errors require review before extending broad original-content claims to these edit types.
- Audio issued no wrong-original verdicts. Strong noise reduced AI flags from 18/24 clean to 13/24 noisy recordings and increased inconclusive outcomes to 33/48. One genuine recording was falsely flagged in noise and another in music. Calibration was not changed in response.
- Article inspection checks completed, but the current whole-article confidence policy withheld every tendency. Their zero false-alarm count reflects abstention and does not demonstrate useful headline detection coverage.

## Scope and uncertainty

Audio used four synthesis families, six recordings each: GradTTS, MatchaTTS, PflowTTS and StyleTTS2. The first two families were absent from the earlier calibration. Recording/prompt IDs were fresh, but speaker identities overlapped earlier work. Inputs were prefixes of at most four seconds, normalized to mono 16 kHz PCM. Four genuine and four AI recordings were shorter than the supported 2.5-second minimum; none received an AI/original tendency in any condition. The fixed pool of four noise and two music backgrounds limits generalization.

All video headlines concerned visual content. No video supplied usable speech, and lip-sync checks completed on only two clips. These results do not establish general lip-sync performance or combined video/speech confidence.

Native image/video resolutions, formats and compression were retained; source/codec shortcuts remain possible. Labels follow the dataset publishers and genuine origin was not independently authenticated. The sample is small and balanced, so observed verdict correctness is not future customer-alert precision. JSON includes descriptive Wilson intervals, which assume independent cases and do not adjust for paired/shared-source dependencies. Zero false alarms in 24 genuine inputs still has an approximate 13.8% upper 95% case-level bound.

There were 23 excluded video setup attempts without a bound model manager, before learned-model inference. The runner was repaired and the same selected clips were rerun. Original setup logs/results and freezes are retained; the production scoring configuration and weights matched the original lock.

## Artifacts and reproduction

- Reviewable report: `output/evaluation/rule80-fresh/results.html`.
- Counts, intervals, cohorts and audit: `output/evaluation/rule80-fresh/summary.json`.
- All 288 case outcomes: `output/evaluation/rule80-fresh/outcomes.csv`.
- Frozen protocol, samples, raw outcomes and published evidence: `backend/data/independent-evaluation/rule80-v1/` (local ignored data).

From `backend/`, reproduce the summary without inference or fitting:

```sh
PYTHONPATH=. .venv/bin/python scripts/summarize_rule_validation.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_rule_validation_summary.py
```

Source datasets: [DiTFake](https://huggingface.co/datasets/Jouesmak/DiTFake), [TGIF](https://github.com/IDLabMedia/tgif-dataset), [DeepAction](https://huggingface.co/datasets/faridlab/deepaction_v1), [RTFS](https://huggingface.co/datasets/kontur-ai/rtfs-10k), [DFADD](https://github.com/isjwdu/DFADD), [LLMTrace](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification), and [Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset). Exact source revisions, URLs and hashes are retained in the local receipts.
