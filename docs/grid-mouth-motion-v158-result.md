# Measured mouth motion in the finished GRID clips

The local motion readout measured all 176 finished cases and all 96 prespecified
paired comparisons. It compared the visible mouth geometry with the genuine
recording on one common set of timestamps for every tested lag. The two redraw
families behaved differently: producing changed pixels did not consistently
produce the requested trajectory.

| Finished video condition | Compatible with the intended visible timing | Cases |
| --- | ---: | ---: |
| Genuine frame copy shifted 320 ms earlier | 16 | 16 |
| Original picture | 15 | 16 |
| Original picture with audio-only delay | 15 | 16 |
| Expression-driven redraw, source time | 15 | 16 |
| Expression-driven redraw, earlier driver | 10 | 16 |
| Speech-conditioned redraw, source time | 0 | 16 |
| Speech-conditioned redraw, earlier speech | 0 | 16 |

The original and audio-only rows had identical observed geometry and identical
motion outcomes, including one ambiguous result. All 16 fixed repeats and both
384-picture exact-copy controls matched. Neither a different audio track nor
the condition label entered the visual observer.

All 4,560 scheduled native frame observations yielded usable geometry. The
sixteen observer stages took 239.66 seconds in total, with 0.39 GiB maximum
sampled descendant RSS and no guard or sampling failures. The motion arithmetic
used only the saved geometry and verified clock records.

The earlier input shifted relative preference toward the earlier reference in
15 of 16 speech-conditioned pairs and all 16 expression-driven pairs. Across
the within-generator and branch-matched ordinary comparisons, 95 of 96 pairs
shifted by more than the one-pixel dead band and one stayed within it.
These are related comparisons from eight known source components. A relative
shift can be real without the earlier trajectory being a reliable absolute
match. The speech-conditioned cases all fell into ambiguous, search-boundary
or other/poor match states despite the relative shift.

The readout supports a limited visible-motion claim under the fixed observer.
It does not establish spoken phonemes, perceived synchronization, edit
realism or detector accuracy. The earlier [detector comparison](grid-native-mouth-scores-v155-result.md)
remains separate. These clips contain mouth redraws and ordinary edits, rather
than identity face swaps. Nothing was added to the app's confidence estimate.

[Observer receipt](evaluation-panels/grid-mouth-observer-complete-v157.json) ·
[Motion aggregate](evaluation-panels/grid-mouth-motion-readout-complete-v158.json) ·
[Frozen method](grid-mouth-motion-readout-v158-method.md)
