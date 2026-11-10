# Native AV audio and conditioning waveforms qualified locally

All **48 audiovisual sources** passed the
[frozen waveform procedure](av-waveforms-v105-method.md). Their native decoded
sample counts match the retained frame records; second source-decode PCM hashes
match exactly; and all mono-16-kHz sample counts and PCM hashes reproduce the
earlier canonical content. The 96 native/conditioning WAVs remain private on
the approved Seagate archive. No detector, encoder or generator was executed.

The code, tests, source/role/clock bindings and FFmpeg binary were frozen and
manually published as `2d74d7e`. Remote freeze blob
`f4f0b3913764932f1a91595697750f1638118886` was verified before source decoding.
The [aggregate result](evaluation-panels/av-waveform-result-v105.json) and
[resource profile](evaluation-panels/av-waveform-profile-v105.json) record the
completed stage and whole-artifact commitments.

## Verified inputs

- 5,446,656 native sample frames across 48 sources at 44.1 kHz.
- Forty-seven mono sources and one stereo source; the native cache preserves
  both channels of the stereo recording.
- 1,976,117 mono samples at 16 kHz across the conditioning cache.
- Exact earlier canonical content and repeated native-decode hashes on all
  48 sources, with no substituted file or fallback conversion.
- All 3,642 common picture centers have source-timed fractional sample
  coordinates and in-range floor/ceiling sample brackets. None required padding
  or clamping.

Every saved WAV was reopened in a separate read-only check. Its header fields,
sample-frame count, file commitment and full PCM payload commitment match the
source-bound journal. Sample zero retains the source audio origin, including
the 25–27 ms audio lead over the first picture. The video origin is not silently
substituted for the audio origin.

## Timing scope

Ten authored mono/stereo impulse checks passed the fixed one-output-sample
allowance. Their largest observed peak-position difference was **0.63719 of a
16-kHz sample**. This measures the pinned resampler on those authored inputs;
it is not a universal sample-accuracy or captured-synchronization claim.

The source container's 1 ms timestamp precision and recorded clock allowance
remain explicit. The previously identified 129 overlapping audio-frame clock
memberships remain in the source plan. Native sample counts and actual decoded
waveforms now support the conditional sample lattice, but they do not remove
uncertainty in the captured stream relationship or prove phonetic alignment.
Both decoding passes share the same FFmpeg codec library; the repeat is not an
independent decoder engine or outside review.

`conditioning_waveform_ready` is true for all 48 sources under this narrow
byte/count/time-bound definition. `generator_conditioning_qualified` is false
for every source. A waveform can pass while its clip is still too short for a
particular visual operation: the prior native 20-frame edit eligibility remains
47 available and one unavailable.

The next model stage must qualify feature stride, frame-center correspondence,
valid support and padding for the actual encoder and generator. Requested audio
conditioning does not establish achieved lip synchronization. No new detector
accuracy, authorship probability, learned calibration or serving admission is
claimed here.

## Execution

All 34 relevant local tests passed, including 16 new waveform cases and actual
authored FFmpeg mono/stereo phase and native-hash executions. Owned-child
monitoring permission inspection passed before the source stage launched outside
the sandbox. The source stage exited zero in **6.81 seconds**, with maximum
observed parent/active-command RSS **151,650,304 bytes**, no resource-bound failure
and no sampling error. Samples are not guaranteed peaks. The unchanged 900-second,
1.5-GiB data-only limits and initial/running RAM/disk reserves were retained;
neural-study budgets were not relaxed.

Source identifiers, per-source hashes, clocks and waveform caches remain private.
Publication contains code, aggregates and whole-artifact hashes. Original study
roles and audio rollback calibration remain unchanged. No customer analysis,
real payment, GitHub Actions, outreach or GCP action occurred. The separate
full face-swap input build continues locally.
