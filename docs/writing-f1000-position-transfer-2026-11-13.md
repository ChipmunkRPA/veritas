# Longer article localization on fresh historical revisions

The frozen writing position check localized two rewrites the existing context-window check missed. Combining the checks increased useful localization from 2 of 18 to 4 of 18 and flagged 2 of 9 genuine works, up from 1 of 9. One of the two added locations came from each generator. [Frozen measurement receipt](evaluation-panels/writing-f1000-position-transfer-2026-11-13.json)

| Outcome | Current check | Position check at the longer-document point | Combined at the longer-document point | Combined at the short-passage point |
| --- | ---: | ---: | ---: | ---: |
| Useful Granite rewrite locations | 1/9 | 1/9 | 2/9 | 2/9 |
| Useful Qwen rewrite locations | 1/9 | 1/9 | 2/9 | 2/9 |
| Useful rewrite locations overall | 2/18 | 2/18 | 4/18 | 4/18 |
| Historical versions flagged | 1/18 | 2/18 | 3/18 | 5/18 |
| Historical works flagged in either version | 1/9 | 1/9 | 2/9 | 3/9 |

At the longer-document point, the new exact ranges were `[125, 320)` in a Granite rewrite of F1000Research work 9026 and `[1274, 1400)` in a Qwen rewrite of work 13610. Both historical versions of work 12687 triggered the candidate at a different sentence from the chosen historical revision target. Both generated variants of that work also triggered the candidate on that genuine neighboring material. At the lower point, one further genuine work was flagged and two already-localized rewrites gained duplicate coverage. At the longer point the union missed 14 of 18 generated edits. The earlier Gates and Wellcome held panel had no added exact locations at the same longer point. Across both frozen trials, useful localization moved from 4 of 42 to 6 of 42 and historical work flags moved from 6 of 21 to 7 of 21. The three journals use the same publishing platform, and all sources are historical English scholarly prose. [Earlier held trial](writing-long-position-transfer-2026-11-12.md)

## Frozen comparison

We used all nine fresh same-byline publisher pairs and preserved both original historical versions as controls. Each earlier passage received one exact sentence substitution from each of two pinned Apache-2.0 generators. Three fixed-seed attempts per work and generator were allowed. All 18 substitutions passed admission before detector execution. The normalized passage retained its other sentences byte for byte and their character offsets shifted only by the replacement length. Each passage contained 11–22 native sentences and 316–570 words. The source reconstruction publishes no article prose. [Source and license ledger](writing-f1000-revision-spans-2026-11-13.md) · [Granite publisher card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [Pinned Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)

CPU inference evaluated 415 unique model inputs and covered all 292 eligible sentence centers with complete responses. The incumbent retained its first six qualifying three-sentence windows and the earlier overlap/recall scoring. The candidate retained the unique highest center response above the respective frozen point. The candidate only earns a useful location when the selected native sentence is the exact substitution. The fitted head and both operating points stayed fixed. This held trial met the prior research rule of two new distinct locations with at most one extra flagged genuine work at the longer-document point. A broader rollout still needs evidence beyond this small panel.

## Replay

From the repository root, Python 3.12 can recompute source geometry, score arithmetic, false alarms, and exact selected locations from the published commitments and saved responses. This command uses no third-party runtime or model weights.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.review_writing_f1000_position_transfer \
  docs/evaluation-panels/writing-f1000-position-transfer-2026-11-13.json
```

The independent publisher replay separately verifies the genuine text and frozen native inputs against the licensed XML. The model-free comparison replays saved scores and source commitments. Generated text and the original inference logs stay local. One archived version contains a standalone zero-word punctuation segment from the normal segmenter. The replay preserves its exact native position, and that segment cannot qualify as a candidate focus.

## Local app option

The prototype is off by default. With the existing candidate writing checkpoint installed, add these settings and restart the local worker:

```dotenv
VERITAS_TEXT_DETECTOR_MODE=candidate
VERITAS_TEXT_WINDOW_SENTENCES=3
VERITAS_TEXT_LONG_PASSAGE_POSITION_REVIEW=true
```

The option runs on fully mapped English plain text with three to five paragraphs, 9–25 sentences, 160–680 words and at most 5,200 characters. An eligible sentence must have both immediate neighbors in the same paragraph. A complete review evaluates every eligible center. The plain evidence view, Standard PDF, and JSON expose the selected source sentence and its original decoded character range alongside the neighbors. When complete readings are unavailable, the check records an unassessed result and preserves the primary analysis. The signed package binds the coverage record, and replay with the original verifies every recorded input and its source positions. The check carries zero weight in the overall assessment. Existing three-sentence behavior is controlled by its separate setting.

Five simulated prepaid runs each on CPU and Apple GPU reproduced both exact generated locations, one ordinary no-flag case, one known ordinary false alarm, and omission when disabled. The disabled and enabled copies of the same input preserved the existing assessment, findings and raw detector scores on each device. For the signed replay fixture, the two devices agreed on every source range, input hash and token count across six eligible centers, and the largest absolute raw response difference was below 0.000010. Chrome rendered the excerpt and all three neighboring source ranges, and its report and JSON downloads matched the worker’s output bytes. All jobs waited for simulated payment confirmation. [Local application receipt](evaluation-panels/writing-f1000-position-app-2026-11-13.json)
