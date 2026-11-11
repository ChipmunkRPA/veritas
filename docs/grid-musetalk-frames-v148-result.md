# Speech-conditioned frame bank completes on all pilot clips

All **16 fixed pilot clips** completed the second generation branch. It produced
**640 planned condition frames**, 16 exact render repeats and 16 exact geometry
repeats. No planned frame was skipped or replaced. Each native composite changed
pixels only inside its branch-specific mouth support. The 320 paired source-time
and earlier-conditioned crops differed, with the same source pixels and sampled
latents within each pair.

The actual workers recorded 336 face API inputs and 2,624 top-level renderer
forwards. The renderer used the retained original whole-waveform states; no
speech encoder forward ran in this branch. The separately measured stage time
was 8.44 minutes summed, with workers taking 31.09–33.84 seconds per clip.
Maximum sampled descendant RSS was **1.13 GiB**, and maximum observed post-render
GPU driver allocation was **4.34 GiB**. They are different measurements, not
additive peak estimates. All workers exited zero with no guard violation or
monitor sampling error. See the [actual receipt](evaluation-panels/grid-musetalk-frames-complete-v148.json)
and [frozen method](grid-musetalk-frames-v148-method.md).

Both generation branches now supply verified native composites and changed-pixel
masks for the matched lossless clip construction. These results establish how
the benchmark edits were produced. Achieved lip motion and detector performance
still require separate measurement. The reserve remains untouched by these
model stages. Media, geometry, private clocks and individual fingerprints were
retained locally; only aggregate results and whole-artifact commitments are
public. The app's model selection, confidence, billing and deployment did not change.
