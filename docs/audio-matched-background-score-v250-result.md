# Matched clean and noisy Turkish speech comparison

The current check and the experimental second opinion both assessed 324 of 360 planned three-second variants. The panel had 355 prepared variants. One genuine recording was unavailable across all five conditions, and the shared speech gate skipped 31 other variants, all genuine. Each condition included 48 assessed synthetic variants from two generators. All ten local CPU batches completed, with 404 current classifier calls and 324 candidate encoder calls. [Frozen method](audio-matched-background-score-v250-method.md) · [Exact aggregate](evaluation-panels/audio-matched-background-score-complete-v250.json)

| Condition | Synthetic AI flags, current → second opinion | Synthetic called original, both | Synthetic inconclusive, current → second opinion | Genuine false AI, both |
| --- | ---: | ---: | ---: | ---: |
| Clean | 45 → 45 of 48 | 1 | 2 → 2 | 0 of 22 |
| Kitchen, 0 dB | 17 → 21 of 48 | 10 | 21 → 17 | 0 of 13 |
| Kitchen, +5 dB | 22 → 25 of 48 | 8 | 18 → 15 | 1 of 19 |
| Traffic, 0 dB | 10 → 18 of 48 | 2 | 36 → 28 | 0 of 12 |
| Traffic, +5 dB | 24 → 27 of 48 | 4 | 20 → 17 | 0 of 18 |
| All noisy variants | 73 → 91 of 192 | 24 | 95 → 77 | 1 of 62 |

The second opinion promoted 18 previously inconclusive noisy synthetic variants to AI, 12 from the designed-voice generator and six from the stock-voice generator. It added zero genuine false AI flags. At 0 dB traffic it recovered eight synthetic variants, the largest gain on this panel. The candidate alone produced 46 wrong-original calls on noisy synthetic variants. The second-opinion rule retained the current route’s 24 wrong-original calls.

The shared speech gate excluded 30 prepared genuine variants in the noisy conditions. Matching the saved gate decisions by speaker shows that 11 of the 22 genuine speakers admitted in clean audio lost coverage in at least one noisy condition, accounting for 26 of those exclusions. Two of those speakers lost coverage in all four noisy conditions. These exclusions limit what the false-alarm count can show.

The replay used frozen English diagnostic score points on Turkish speech. The same foregrounds recur across five conditions, one generator contributes a single stock voice, and the 48 background excerpts come from seven recordings with provisional machine screening. The serving route and customer confidence remain unchanged. These results support further testing of the second opinion with more distinct voices and recordings. [Panel construction](audio-matched-background-panel-v248-result.md)
