# Local soundtrack differences over matching video pictures

The local earlier-copy comparison can look for a short loss of matching
sound while the decoded pictures remain identical. A finding shows both
playable excerpts, waveforms on the same scale, the same before/during/after
picture positions from both supplied files, and which frequency bands lost
matching sound. The selected range is an acoustic lead. Ordinary editing,
including a local timing change or filtering, can cause it. This check carries
zero weight in the overall AI assessment. It does not transcribe words or
assess whether articulation matches the replacement speech. [Timing and
content distinction](https://arxiv.org/abs/2610.30872)

## Method and local results

The four-second, upright 720p25 H.264 profile requires 100 matching decoded
pictures and continuous correspondingly timed first audio tracks. A short
timing search first checks whether the same sound moved. The new measurement
works on 120 ms windows spaced 40 ms apart and compares weighted matching-sound
gain and normalized correlation in three voice-band ranges. It requires both
flanks to have matching audio and a localized reduction in two bands. No
model weights, transcription or third-party processing is used. The paired
design separates replacement from background-only and ordinary processing
controls. [Research on distinct speech and background interventions](https://arxiv.org/abs/2608.09593)

Five Wikitongues recordings supplied two new excerpts each to develop the
check. Four distinct publisher recordings supplied two excerpts each to test
the revised check. The project had previously used these source files at
other offsets. Both panels used real speech with staged 320–800 ms
substitutions from the same speaker or another speaker. The earlier tracks
were clean. Held submitted tracks included clean, rain, train, and competing
speech at about 5 dB above the original voice RMS. PCM and actual 64 kbps AAC
deliveries reuse the same edit plans. The receipt binds the license,
attribution, edit interval and both decoded input hashes for each case.
[Source licenses](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)
· [Background source terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

Useful location means temporal intersection over union of at least 0.50 with
the staged edit. The development version found 34 of 80 substitutions per
carrier, including 22 of 60 under strong background sound. The short timing
comparison located zero of those substitutions. The broader check also
flagged 13 of 320 controls per carrier: 11 local delays and two filtered
clips. It reported no priority-control flags on unrelated or constant-shift
clips. This version missed its frozen requirement of 36 useful AAC
locations and at most five added ordinary flags per carrier.

The revised version retains a finding only if at least two of its supporting
bands reach 0.84 measured correlation on both flanks. That requirement was
chosen from the development data before running the transfer. The transfer
required eight useful clean AAC locations across three sources, at most one
added ordinary flag per carrier, and zero added priority-control flags.

| Held condition | PCM useful | AAC useful | Added control flags per carrier |
| --- | ---: | ---: | ---: |
| Clean | 6/16 | 6/16 | 0/64 |
| Rain | 0/16 | 0/16 | 0/64 |
| Train | 0/16 | 0/16 | 0/64 |
| Competing speech | 0/16 | 0/16 | 0/64 |

The six locations came from three of four speakers. The result missed the
eight-location rollout requirement. The clean AAC analysis reached a
measurement on 14 of 16 substitutions and 37 of 64 controls. The remaining
cases stay in the reported denominators. Loud noise remains a gap.
[Development receipt](evaluation-panels/video-reference-sound-difference-development-2026-11-14.json)
· [Transfer receipt](evaluation-panels/video-reference-sound-difference-transfer-2026-11-14.json)

## Optional local trial and replay

On a development instance with the earlier-copy route and
`VERITAS_VIDEO_REFERENCE_SOUND_DIFFERENCE=true`, open **Compare with an earlier
video** and select **Include experimental soundtrack comparison**. The checkbox
starts off. This trial compares a supplied earlier copy in a completed
analysis and carries no weight in the saved confidence. The API requires
an explicit request and accepts it separately from a marked mouth event.
The comparison export binds the earlier and analyzed frames, local audio,
sample positions and band measurements. These are supplemental findings.
The PDF and JSON also retain a result when the trial cannot report a range.
The result identifies missing picture or audio coverage, a supported timing
alternative, insufficient matching sound, or a completed search with no
qualifying range. Only a measured difference includes matching picture
exhibits and audio excerpts.
Two isolated $9.99 test checkouts exercised the paid worker and authenticated
comparison. No analysis jobs ran before simulated payment. The held AAC
replacement showed **1.400–1.880 s**, with six matching source pictures and
both excerpts reproduced directly from the video files. Passing the same two
inputs without opting in left the trial out of the result, and the unmodified
control produced no difference finding.

Run the numeric replays from the repository root. To regenerate a receipt, use
the same module with `freeze` and `run`, passing `--transfer` for the second
panel. Supply the pinned video directory and the two pinned `.ogg` files.
Source media stays local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for panel in development transfer; do
  python3 -B -S -m scripts.video_reference_sound_difference_transfer review \
    "docs/evaluation-panels/video-reference-sound-difference-$panel-2026-11-14.json" || exit 1
done
backend/.venv/bin/python -B -m scripts.compare_video_references earlier.mp4 analyzed.mp4 export.json --verify
```
