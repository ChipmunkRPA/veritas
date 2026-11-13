# Localizing a rewritten sentence by removing it from two contexts

The sentence-removal check found **one of 21** known AI rewrite locations in eLife excerpts, up from zero for the two existing checks combined. It flagged none of the 24 ordinary eLife files. On a separate F1000Research panel it found one already-reported edit and added one ordinary work flag. Neither candidate met the frozen criteria for serving. [Text-free measurements, exact locations and operating points](evaluation-panels/writing-omission-attribution-2026-11-13.json)

## Frozen comparison

A contextual classifier can score a genuine sentence differently when adjacent wording changes. For every eligible interior sentence, compare the same classifier's raw log odds on the sentence with its left neighbor and with its right neighbor. Measure each drop when the focus is omitted. Divide by the square root of the number of omitted words relative to 24 words. The first arm uses the smaller of the two normalized drops. The second arm also requires the focus omission to reduce the response more than omitting the other sentence in each pair. It keeps the smallest of the four values. Each check can select one unique strongest exact sentence in the uploaded source. All measurements preserve the complete words in the source; there is no generated replacement, retraining, or reuse of another paper's code. [Sentence-level localization research](https://aclanthology.org/2024.findings-acl.495/) · [Token contextual dependency research](https://aclanthology.org/2026.emnlp-main.1708/) · [Installed checkpoint and reviewed terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Freeze both operating points on the second-largest ordinary maximum across 14 older F1000Research works, with both archived versions per work and a floor at zero. Both points were 1.7221808973863402. The two held panels opened after those points were saved. Nine F1000Research works supply 18 original-version controls and 18 exact rewrites. Twelve eLife works supply 12 untouched excerpts, 12 constructed controls using historical same-work sentences, and 21 exact rewrites. The held works and all outcomes had prior project exposure under other methods. The omission inputs and responses were new. The source and license ledgers provide exact publisher versions, author-group separation and generation conditions. [Point-selection sources](writing-long-position-transfer-2026-11-12.md) · [Held F1000Research sources](writing-f1000-revision-spans-2026-11-13.md) · [Held eLife source and rewrite qualification](writing-elife-long-position-transfer-2026-11-13.md#archived-source-and-comparison-method)

## Held results

| Outcome | F1000Research | eLife |
| --- | ---: | ---: |
| Known edits located by either existing check | 4/18 | 0/21 |
| Exact locations from two-context drop alone | 1/18 | 1/21 |
| Exact locations from stricter drop margin alone | 1/18 | 0/21 |
| Additional locations from two-context drop | 0/18 | 1/21 |
| Wrong-only locations from two-context drop | 2/18 | 0/21 |
| Ordinary work groups flagged by two-context drop | 1/9 | 0/12 |
| Ordinary work groups flagged by either existing check | 2/9 | 2/12 |
| Additional ordinary groups flagged by two-context drop | 1/9 | 0/12 |

The eLife hit is Qwen work 32660 at exact character range `[370, 458)`. The stricter arm suppressed this hit. On F1000Research both arms selected an unchanged sentence at `[626, 811)` in original work 12989 and both its generated variants. In those variants the actual edits start at character 1209. This is another case of a high background response unrelated to the substitution. The intended sentence ranked first before applying the point in 7 of 18 F1000Research and 5 of 21 eLife edits for the two-context drop. Only three eLife targets ranked first under the stricter arm. All 39 exact targets were assessed. The combined result located 5 of 39 edits and flagged one additional ordinary group. The strongest new evidence came from one generator and one work.

In a retrospective sensitivity readout, allowing up to two ordinary work flags would raise the combined F1000Research result to 8 of 18, with six generated files flagged at the wrong location by the new arm. On eLife it would still locate one edit, with three wrong locations. Those points use held labels and leave serving unchanged.

## Reproduction

The offline CPU run scored 1,113 unique inputs within their declared panel roles and measured 918 sentence centers, with no input over the native limit. CPU scoring took 66.653 seconds across the three roles. A separate local run reconstructed every input and exact native sentence range from the original retained source files. The portable reader replays each location and aggregate without the model, source prose, or third-party dependencies. The published receipt contains hashes and responses, without original or generated article prose.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_omission_attribution review \
  --receipt docs/evaluation-panels/writing-omission-attribution-2026-11-13.json
```

Use `native` with `--point`, `--f1000`, and `--elife` pointing to the retained panel directories to verify source geometry and input hashes. For a fresh local inference replay, use `freeze` with the same three inputs and a new `--run` directory, then run `measure --panel point`, `calibrate`, `measure --panel f1000`, `measure --panel elife`, and `publish` with the corresponding panel and run paths. The model runs require the pinned project checkpoint, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
