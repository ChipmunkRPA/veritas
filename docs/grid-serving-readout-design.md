# Full-serving GRID pilot readout

Prospective contract. Use the frozen v143 pilot, matched v149 construction and
admitted corrected serving receipts. Measure the existing appearance review, audio/picture
timing and customer conclusion as three distinct endpoints. Keep the v152
commercial manifest, full profile and unchanged decision rules. No detector
result or candidate admission is reported here.

## Input and coverage

Join privately on the original pilot ordinal, numeric case number and condition.
Require matching source slot and connected component, the exact v149 input hash
for attempted cases, and the complete verified roster from the frozen plan.
Retain all eleven planned conditions per source. Keep unavailable inputs,
unstarted stages, preserved guard failures and missing case results in the
planned denominator. A successful worker receipt can still contain failed
analyses. Retain those failures with their actual stages.

For visual comparisons require `sampled_input.clock_exact` and
`sampled_input.sampled_rgb_exact` to be true. For timing also require true
`audio_samples_equal`, `audio_origin_equal` and `audio_pcm_equal`.
A missing or false input check makes the corresponding comparison unassessed.
Keep the engine's actual returned outcome separately. Count expected, attempted,
successful, skipped, failed, timed-out, not-applicable and unobserved checks by
the frozen routes and actual stage and detector records. Keep profile exclusions
and unavailable model coverage distinct from cases where a completed check
found no signal.

Report the proportion of planned legacy samples decoded, the proportion of those
samples with a detected face, and the proportion of native frames actually
tracked. Count unique frame indices within each case before aggregation. Use
the production `face_track` exhibits for tracked positions, the recorded detector
input indices for window measurements, and the actual before/during/after packet
anchors for displayed context. A sparse series or an enclosing window does not
establish coverage between its measured frames. The engine's sampling seed can
change with the case file hash. Keep each method's actual sampling and losses.

## Appearance review and fallback

Use the single `vid.review.face_appearance` entry from the actual detect stage.
With `appearance_check_status=ok`, require method
`face-appearance-change/1`. The existing review point is strictly greater than
0.11243900750879454. With `appearance_check_status=not_applicable` and the
returned sequence method `face-sequence-change/1`, use the existing inclusive
point 0.007326911389827726. This branch is the production fallback when the
appearance check has no eligible window. A failed, skipped or unsupported
appearance check is not itself a fallback result. Record whether fallback
actually ran and whether it had usable windows.

Use the actual selected `details.face_sequence` as the flag. Verify its method,
exact response, frame indices and pixel commitments against the source and the
production sequence validator. The other `face_sequence_windows` establish
measured support, including nonselected windows. They never create extra flags.
Report the returned appearance and fallback flag counts separately and as the
combined production route. An `ok` result with valid measured windows and no
selected range is measured with no flag. Unavailable windows stay unassessed.

The flag span is the first through last core center, using the exact source
PTS and time base as a half-open interval for duration calculations. The
outer-context bounds of a packet do not extend the flagged span. For delivery,
require the published video evidence packet with that same `face_sequence`
record and the three source-bound display anchors. The adapter chooses the
last before center, middle during center and first after center. Count whether
they truly bracket the known edit and whether the during frame contains a
verified pixel change. A face crop shows the measured area. It does not measure
which pixels changed or establish an identity swap.

## Audio/picture timing

Use `vid.lipsync.landmark_xcorr` from the actual detect stage. The flags are
offset-step and correlation-collapse segments in
`details.av_input.tracks[].segments`. Use the entire recorded set. Segment
ticks are on the bound audio clock at
`audio_origin_pts / 16000 + tick / 25` seconds, including fractional change
ticks. Use exact rational arithmetic for their recorded numeric values.
Cross-check native frame indices and PTS against the v149 source and match the
PCM commitment, sample count and origin. Check recorded segment counts against
the stage metrics and the raw timing exhibits. The derived temporal lane and
the maximum-four presentation packets cannot substitute for all raw segments.

Measure coverage from the union of verified windows and their unique sampled
frames, preserving each track's actual support. Silence or no speech can
return `ok` without measurements. No attributable speaker, missing or
discontinuous audio, and unbound inputs are unassessed. When measured windows
have no flags, keep no-stable-offset cases distinct from stable measurements
with no local discrepancy. A constant offset alone and a single low
correlation reading are contextual observations. Keep any raw detector flag
with missing source binding in a separate unavailable-evidence count.

For delivery, require a published `evidence.av` packet that corresponds to a
bound raw window and records an overlap with an actual offset-step or
collapse segment. Verify the window's frame anchors and exact PCM interval.
Report recorded, displayed and omitted window counts separately. The 1-second
window and ±600 ms search context do not define the flagged span. Counts of
ordinary retiming and soundtrack-delay flags describe response to the
constructed timing intervention. Neural driver offset and successful rendering
do not establish achieved alignment or its absence.

## Localization and customer direction

For observable visual edits, use the v149 authored half-open edit interval
`[PTS[27], PTS[47])` on its verified source clock. Count actual changed-picture
coverage as the fraction of frame indices with positive
`edit_proofs.changed_pixels` that occur among the endpoint's verified inputs.
For the deliberate audio delay use the output clock
range of the copied interval in `audio_value_mapping`, at 16 kHz and the
verified origin. The copied interval is the single piece whose input and
output sample positions differ. Require a positive changed-sample count for
observable audio edits. Keep zero-effect constructions in the planned cohort
and omit them from observable-effect rates. Genuine frame retiming is a
separate ordinary visual and timing control. Report temporal overlap with
neural redraws as descriptive for timing until achieved motion is measured.

Union all exact flagged spans per endpoint and case before measuring duration.
Report intersection, flagged duration outside the edit, and temporal
intersection over union. The prespecified research success point is IoU at
least 1/5. Intersect predicted spans with the observed range from first to
last video PTS and, for timing, the verified audio interval. Report spans
truncated by those bounds and the excluded duration. Use the full known edit
as the denominator where its clock is valid. The assigned terminal picture
duration supplies no extra observed support.
Measured cases without a flag have zero overlap. Cases without assessed
support have no IoU value. Report unassessed counts and all-planned hit rates
alongside assessed-only rates. For the original case, report total flagged
duration and review burden. There is no known edit interval for an IoU.

Use `customer_assessment.direction` from the published production projection.
Count `ai` and `original` only for a completed analysis with
`directional_conclusion` and `content_assessed` both true. Keep
`inconclusive`, `unavailable` and provenance-led `record` separate. Preserve
the stated scope, including separately assessed soundtrack and timing.
A directional AI finding on a verified non-neural control is false AI
attribution. A review cue on an ordinary edit can be useful and does not
constitute that error. For an observable neural edit report AI direction,
original direction, inconclusive and operationally unavailable separately.
These are case-level outcomes. They do not establish which stream or region
caused the conclusion, or a wrong claim about an excluded stream.
No check score, fused raw probability or public evidence packet upgrades an
abstention or alters the existing decision rule.

## Aggregation and publication

Report the eleven fixed conditions. For disjoint grouped totals use the
original, neural redraw, resample and brightness, genuine frame retime, and
audio-delay families. Report observable, zero-effect and unavailable status
within each applicable family. Keep exact
numerators and denominators for planned, constructed, completed, assessed and
delivered outcomes. Report overall and within-condition rates. Calculate a
component-weighted rate by taking the mean of the within-component rates,
with the connected components from the fixed plan carrying equal weight.
Use the same grouping for duration or IoU summaries. Report the number of
contributing components and keep omitted denominators explicit. Treat the
small pilot as descriptive.

For a prespecified matched research comparison, use the same clip and
condition. Record both-hit, gain, loss, both-miss and one- or both-unassessed
counts. Define gains and losses on the common endpoint, such as a delivered
localization or a matched detector-level inspection cue, and report the
common-support denominator plus the complete planned denominator. A
component-weighted paired difference averages each component's within-pair
differences, with gain +1, loss −1 and ties zero. A dense diagnostic or a
standalone candidate score does not supply the production headline endpoint.
Any claim about changing that headline needs a separate paired run of both
complete serving configurations.

Publish a versioned allowlist of aggregate counts, rates, duration summaries,
cohort roles, settings commitments and whole-artifact hashes. Keep cases,
component membership, IDs, per-media hashes, paths, excerpts, frames,
waveforms, scores, track details and timing lists private. Publish
cross-tabulated metrics only over groups of at least five source components.
If a smaller breakdown would expose an individual component, publish the
prespecified broader group and omit complements that reconstruct it. Keep
the report at the level of the measured pilot and its detector configuration.
