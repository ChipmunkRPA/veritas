# Finding more short speech changes during loud room speech

These two tests ask whether the optional earlier-copy comparison misses short
changes because background speech obscures the foreground or because the
measurement windows begin too far apart. Both use the same 264 previously
committed native AAC comparisons. The two panels use distinct placements
and four different recorded room backgrounds. Each uses 12 ICSI foreground
groups, two synthesis generators, and AMI room speech mixed 5 dB louder
than the foreground. The stock-voice cases from the earlier study are
outside this replay. These are new measurements of retained source inputs.
[Original placements, terms, and local app](audio-stereo-room-transfer-2026-11-15.md)

## Stereo interference reduction

Fit the earlier recording's gain separately on both microphones from
three strong one-second matches in each half. Require comparable left
and right balance across both halves. Estimate residual room covariance
where the earlier recording is quieter, then take a fixed weighted
mix that preserves its estimated gain. Keep the 16 kHz measurement
clock and apply the unchanged waveform comparison to the result.
This uses the principle of distortionless spatial filtering without
external weights or detector code.
[Reference on room interference filtering](https://arxiv.org/abs/2507.13863)

Using the current result first and the projected mix as a fallback kept
12/24 synthetic and 6/12 real splice locations in the first panel.
It added three ordinary flags among 36 controls and misplaced ranges on
two synthetic edits and one real splice. The projected mix on its own
fell to 8/24 and 4/12. In the held panel all three comparisons located
14/24 synthetic edits and 7/12 real splices, with zero ordinary flags.
Use the current stereo corroboration.

## More overlapping windows

The current comparison runs six six-second windows starting 1.2 seconds
apart. Run the same six-second measurements from 16 starts at 400 ms
steps. Compare support from windows 1.2 seconds apart and 0.8 seconds
apart separately. Each pair needs two 720 ms ranges starting within
360 ms. Preserve the original reference admission and frequency-balance
gate per channel. An additional range still needs the existing native
channel corroboration. The tested fallbacks keep any current range first.

| Panel and rule | Correct synthetic edit ranges | Correct real splice ranges | Ordinary flags | Flags when the earlier copy was room speech |
| --- | ---: | ---: | ---: | ---: |
| First, current | 12/24 | 6/12 | 0/36 | 1/12 |
| First, denser starts with 1.2 s separation | 12/24 | 6/12 | 0/36 | 1/12 |
| First, denser starts with 0.8 s separation | 14/24 | 7/12 | 0/36 | 2/12 |
| Held, current | 14/24 | 7/12 | 0/36 | 0/12 |
| Held, denser starts with 1.2 s separation | 16/24 | 8/12 | 0/36 | 0/12 |
| Held, denser starts with 0.8 s separation | 16/24 | 8/12 | 0/36 | 0/12 |

Both denser rules retained every correct current range. Neither flagged a
half-matching reference, an unrelated reference, or either room-only
control. In the first panel only 14/24 synthetic edits intersected a
useful 720 ms range the two original windows could both measure at the
exact same start. The shorter spacing expanded this to 18/24. In the
held panel all 24 edits had a useful common grid position under every
schedule. This geometry diagnostic uses known edit times and explains
opportunities to measure a range. It is separate from the detector's
two-range agreement rule.

Keep the dense 1.2 s schedule as the candidate for a fresh room and
microphone comparison. It recovered three held locations without
additional control flags. Test the shorter spacing separately because
it recovered three more locations in the first panel and added one
wrong-reference alert. Both remain research checks. The comparison
requires a matching earlier recording and cannot attribute a change
to AI. Customer-facing confidence and the current detector remain as
previously released.

## Replay

The numeric reviewers independently check the prior panel commitments,
source group membership, output ranges and paired counts. The schedule
receipt retains the chosen window starts and ranges for each channel.
Rebuilding those measurements requires native sources. Both portable
reviewers run without weights, recordings or third-party libraries.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.audio_reference_spatial_transfer review \
  "$panels/audio-reference-spatial-transfer-2026-11-15.json.gz"
python3 -B -S -m scripts.audio_clean_speech_schedule_transfer review \
  "$panels/audio-clean-speech-schedule-transfer-2026-11-15.json.gz"
```

To recalculate the measurements, use the prepared ICSI PCM directory
and first and held two-meeting AMI directories from the earlier study.
Each capture verifies all 264 pairs against the frozen reference,
delivery and decoded channel hashes before measuring them. The public
receipts contain numeric measurements and hashes without audio.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for method in audio_reference_spatial_transfer audio_clean_speech_schedule_transfer; do
  backend/.venv/bin/python -B -m "scripts.$method" capture \
    "$icsi" "$first_rooms" "$held_rooms" "$run/$method.json.gz"
done
```
