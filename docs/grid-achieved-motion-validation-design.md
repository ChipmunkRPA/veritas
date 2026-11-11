# Visible mouth motion on the finished GRID clips

This design measures whether a finished mouth redraw follows the visible
trajectory of its matched genuine recording at source time or 320 ms earlier.
The bank has 16 pilot recordings, 64 neural redraws, 16 remuxed originals and 96
ordinary controls. Saved planning metadata confirms 75 actual native pictures
per pilot, 40 ms between consecutive PTS, and eight known source components.
[v145 motion design](grid-transfer-serving-and-motion-review-v145-design.md) ·
[Completed v149 bank](grid-native-clips-v149-result.md)

Both redraw families retain the original soundtrack. Driver frame or speech
conditioning determines the nominal target. The observed lag comes from the
finished image trajectories.

## Independent observations

Admit the completed v143 pilot plan and each successful v149 case inventory,
profile, journal and input binding through `grid_native_clips_v149.load_completed`.
Hash each admitted case file and verify the actual decoded frame index, RGB bytes
and rational timestamp against its v149 proof. Use original native source times
and the matching decoded Matroska timestamps. The frame schedule avoids the
terminal picture duration. Each case uses the same lossless RGB and PCM container.

Use the isolated v132 CPU Face Landmarker task and the v144 `IMAGE` options.
Recheck the installed wheel manifest, model cards and task hash with the v144
runtime admission helper. Run the observer directly on the finished full native
RGB pictures. The task input contains the picture and fixed runtime settings.
Join condition labels after measuring the pictures. The original's larger
reference schedule is fixed in advance. The observer shares the landmark model
used for construction geometry, while all output coordinates come from fresh
observations of finished pixels. This measures agreement under one observer.

Measure original frames 15–58 inclusive and frames 25–48 inclusive for each of
the ten other cases. Add one exact original frame 27 repeat per pilot. This is
44 reference pictures plus 240 variant pictures plus one repeat, for 285 task
inputs per ready pilot and 4,560 across the bank. Independently measure the audio
and genuine-frame-copy controls even when their RGB matches a reference exactly.
Retain the original planned requests and each observed status. Check the repeat
and both exact-RGB controls by comparing the measured landmark values and statuses
at their bound correspondences. A repeat discrepancy leaves a failed observer
attempt for diagnosis. Missing geometry in both corresponding inputs preserves
unavailable motion support.

## Descriptor and clock

Use one finite 478-point face mesh per picture. Convert the needed coordinates
to native pixels as `p[k] = (width*x[k], height*y[k])`. Require those points
inside the native picture. Set `L = (p[33]+p[133])/2`,
`R = (p[362]+p[263])/2`, `s = ||R-L||`, `ex = (R-L)/s` and
`ey = (-ex_y, ex_x)`. Require `s >= 12` pixels and a nose tip (index 1) below
the eye axis in the `ey` direction. The pinned landmark topology pairs the
near-centre inner upper and lower lip vertices as `(82,87)`, `(13,14)` and
`(312,317)`. Define:

```
gap(u, l) = dot(p[l] - p[u], ey)
width = abs(dot(p[308] - p[78], ex))
q = [max(0, gap(82,87)), max(0, gap(13,14)), max(0, gap(312,317)), width] / s
```

Require at least 8 native pixels of projected width. A lip pair crossing by
more than one pixel makes that picture unavailable. A smaller crossing clips
to zero. For accepted geometry, the descriptor is invariant to 2D translation,
rotation and uniform scale. Record `s`, `atan2(ex_y, ex_x)` and the two
projections of `p[1] - (L+R)/2` onto `(ex, ey)`, divided by `s`, as pose
diagnostics. On redraws, check that the 7-by-7 RGB patches centered at the
rounded original eye vertices `(33,133,362,263)` match the same-time original.
When all four patches fit and match, report the RMS displacement of those four
landmark estimates in native pixels. This reveals observer drift on identical
eye pixels. Perspective, occlusion and stylized redraws can affect the measurement.

Let `Y(t)` be the measured variant descriptor and `R(t)` the genuine original
descriptor at the observed PTS `t`. Use exactly the twenty edit positions
`[27,47)`. For each `k` from −12 through +12, compare `Y(t)` with
`R(t + k/25 seconds)`. Find these by exact rational timestamp lookup. The earlier
target is `k=-8`, and the source-time target is `k=0`. Reference frames 15–58
cover every lag for every edit position. Use one common set `S` across all
25 lags: retain a variant time only when it has usable geometry and all 25
reference positions have usable geometry. A curve needs at least 16 of the
20 edit positions and at least 12 adjacent native pairs `P` in `S`. Use no
interpolation. A separate descriptive contrast of the two target lags can use
their exact shared support when the full curve is unavailable. Give that contrast
its own denominator and retain the full-curve result as unassessed.

## Distances and interpretation

Take RMS over four coordinates and the indicated set of times. Subtract each
trajectory's mean over `S` for the primary distance. This keeps motion amplitude
and removes a constant difference in mouth shape. Record the absolute shape
distance and adjacent-pair velocity error for every lag.

```
Rk(t) = R(t + k/25)
Yc = Y - mean_S(Y)
Rkc = Rk - mean_S(Rk)
C[k] = RMS_S(Yc - Rkc)
D[k] = RMS_S(Y - Rk)
V[k] = RMS_P(((Y(b)-Y(a)) - (Rk(b)-Rk(a))) / (b-a))
delta = 1 / median_S(original eye separation in native pixels)
```

`delta` is a one-pixel engineering dead band in normalized coordinates. It
defines the resolution of this comparison. Keep the distances and observed
motion magnitudes for every assessed curve. Apply these rules in order:

1. Mark insufficient common support unavailable. Mark weak reference motion
   when either reference target's centred RMS or `RMS_S(R0c - R-8c)` is at most
   `delta`. This keeps distinct-looking frames from implying distinct trajectories.
2. Mark the output static at this resolution when `RMS_S(Yc) <= delta`.
3. Let `near_best = {k: C[k] <= min(C) + delta}`. Call a trajectory compatible
   with source time or with the earlier target when all near-best lags fall
   within one frame of that target, the other target's `C` exceeds the selected
   target's `C` by more than `delta`, and `C[target] + delta` is below
   `RMS_S(Rtargetc)`. The last test compares the match with a static output.
4. Record a search-boundary result if a near-best lag reaches either end of the
   grid. Record ambiguous timing if the near-best set spans more than two frame
   intervals or the two target distances are within the dead band. Retain the
   remaining curves as other or poor matches.

The compatibility result has one native frame of lag tolerance. Keep the full
curves, absolute shape errors and velocity errors privately, including the curves
whose best lag is unclear. The comparison never stretches time or rescales the
motion to favor a target. Measure the entry jump from frame 26 to 27 and the
exit jump from 46 to 47 separately, using the actual adjacent PTS and the
descriptor velocity norm. Compare these with the same transitions in the
original and ordinary controls. A measured edit-boundary jump is its own result.

## Paired controls and outputs

- The genuine frame retime supplies an exact earlier RGB positive control at
  each edit time. Independently observed valid landmark values must agree with
  the corresponding earlier reference. The audio-only delay supplies the same
  check at source time and tests visual invariance to a changed soundtrack.
- Score each branch's resample and brightness controls against source time and
  the earlier target. An earlier-compatible ordinary control measures how often
  a visual treatment yields the same geometric reading. Keep its measured
  distortion and boundary jump in the matched comparison.
- Compare the earlier and source-time redraws within each generator. Recompute
  both curves on their intersection of common support, and keep the same
  requirements and dead band. Define earlier preference as `A = C[0] - C[-8]`.
  Report the paired `A[earlier] - A[source_time]` on that support. Use the same
  intersection rule for earlier redraw versus each matched ordinary control.
  Report every contrast that meets the shared support minimum, with its reference
  motion status. Count differences above `delta` as gains, below `-delta` as
  losses and the rest within the dead band. Each contrast has its own coverage
  count. Pairing preserves each case's standalone result.

Preserve all 176 planned case statuses. Publish counts by generator and condition
for support, weak reference, static output, ambiguity, target compatibility and
other matches. Summarize compatible lag ranges using the first and last near-best
lag. Report paired gains and
losses, median and interquartile range of the paired preference differences,
ordinary-control results and entry/exit jumps. Summarize paired differences within
known source components. Report the median and interquartile range of the
component medians and their contributing component count. These finite-panel
counts describe correlated variants. Private journals retain source links,
native times, the small set of used landmark coordinates, arrays and individual
RGB commitments. Public records contain aggregates and whole-artifact hashes.

Freeze the exact observer and arithmetic before measuring outcomes. Run the
observer under the existing v108 neural guard, one pilot at a time. The neural
limits are 900 seconds, 6 GiB sampled descendant RSS and 6 GiB initial RAM.
Replay the saved geometry arithmetic in a separate data-only stage with 900
seconds, 1.5 GiB sampled descendant RSS and 3 GiB initial RAM. Both use 3 GiB
initial project and archive disk and 1.5 GiB running reserves. Authored tests
cover similarity transforms, exact zero and eight-frame shifts, static outputs,
periodic ambiguity, support loss and paired control invariants. The arithmetic
admission depends on the construction and observer receipts. Detector scores and
reserve pictures stay outside it. This design measures visible geometry under
the specified observer. Phoneme alignment, speech intelligibility, perceived
realism and customer detector performance require separate evidence.
