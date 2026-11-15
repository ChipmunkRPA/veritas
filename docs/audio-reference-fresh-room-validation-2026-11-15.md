# Stereo speech comparison on three new room recordings

The denser window check found more edits on retained noisy speech. This
follow-up tests whether those gains survive a change in recorded room
speech and microphone positions. The comparison uses the same 24 ICSI
foreground groups, earlier copies and known placements. This isolates
background transfer. Foreground voices and synthesis methods were
already present in the earlier study.
[Earlier comparison](audio-reference-spatial-schedule-2026-11-15.md)

## Inputs and comparison

Use non-scenario AMI meetings EN2005a, IB4001 and IN1016, with microphone
channels 03 and 07 from the first array. These source recordings are
separate from the four meetings used in the prior schedule comparison.
The frozen input plan assigns eight foreground pairs per meeting. It
balances early and late edits and the quieter foreground channel.
Each room excerpt is twelve seconds long. It contains at least 2.4
seconds of annotated words in each half, including words during at
least half of each synthetic replacement. The room speech is 5 dB
louder than the foreground. The mix is delivered as 48 kHz stereo AAC.
Controls include unmodified foreground, resampled speech, real splices,
room-only sound, and references that match the background, match half
the foreground, or refer to unrelated speech.
[AMI recordings and channel specifications](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) ·
[CC BY 4.0 source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Compare the serving check with 400 ms window starts paired 1.2 seconds
or 0.8 seconds apart. Each fallback keeps the current supported range
when there is one. Both keep the original reference and frequency-balance
qualification and require stereo corroboration. A useful highlight covers
half the known change and at least 30% of the temporal union. The
published plan fixes all source assignments before the first measurement.

## Results

| Stereo check | Correct synthetic edit ranges | Correct real splice ranges | Ordinary flags | Flags with a matching room reference |
| --- | ---: | ---: | ---: | ---: |
| Current | 26/48 | 13/24 | 0/72 | 1/24 |
| Denser starts, original 1.2 s separation | 28/48 | 14/24 | 0/72 | 1/24 |
| Denser starts, 0.8 s separation | 30/48 | 15/24 | 0/72 | 1/24 |

Both variants preserved every current range. They added zero misplaced
ranges and zero flags on room-only, unrelated or half-matching controls.
The three additional ranges at the original separation came from IN1016.
The shorter separation found three more ranges with EN2005a. It also
added one wrong-background flag in the earlier two-panel comparison.
The original separation advanced to a local fallback. The results
leave new foreground voices, microphones outside this corpus, different
edit mechanisms and unrestricted durations for later evaluation.

## Local app

Use `VERITAS_AUDIO_REFERENCE_STEREO_DENSER_WINDOWS=true` alongside the
existing twelve-second stereo AAC comparison settings. The comparison
is for a supported uploaded recording and its clean earlier copy.
The browser, API, PDF and JSON explain that each highlight needs
matching sound around it. Changes near either end can be missed. A
recovered interval retains its source ticks, both waveform excerpts,
the corroborating channel and the two window starts that supported it.
The fallback adds zero weight to the overall AI estimate.

Two local USD 9.99 payment simulations queued zero unpaid analyses.
With the option enabled, the edited stereo recording gained a highlight
at **7.280–8.000 s**, confirmed by the left channel. The existing check
had no supported interval. The ordinary recording and the same recording
compared against room speech gained no highlight. The PDF and JSON
agree on the recovered range and its window starts. Offline verification
redecoded both files and matched both waveform excerpts, source ticks,
and the channel. Stored AI assessments retained their original hashes.
[Local app verification receipt](evaluation-panels/audio-clean-speech-denser-stereo-app-2026-11-15.json)

## Replay

The standard-library review checks the source roster, prior receipt
hashes, room byte-excerpt commitments, AAC clock, measured ranges,
window witnesses and matched controls. It rebuilds the comparison
counts from the numeric ledger without downloading the recordings
or loading any model.

```sh
export PYTHONPATH=backend
module=scripts.audio_clean_speech_schedule_fresh_rooms
receipt=docs/evaluation-panels/audio-clean-speech-schedule-fresh-rooms-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, use the exact [AMI manual annotations](https://groups.inf.ed.ac.uk/ami/download/)
and prepared ICSI foreground bank from the earlier comparison. The
capture reads the verified 44-byte PCM headers and 48 bounded public
microphone excerpts. IB4001 declares trailing data outside the PCM
chunk. The byte ranges use its independent PCM length. The runner
encodes one temporary AAC submission at a time. Match the room PCM
and comparison hashes as well as the counts. Different FFmpeg or
numerical-library builds may change delivery or measurement hashes.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m "$module" freeze "$annotations" "$run/fresh-rooms-plan.json"
backend/.venv/bin/python -B -m "$module" capture "$run/fresh-rooms-plan.json" "$annotations" \
  "$icsi" "$run/fresh-rooms.json.gz"
backend/.venv/bin/python -B -m scripts.audio_clean_speech_aac_app "$run/app" \
  --receipt "$receipt" --icsi "$icsi" --long --carrier aac48_stereo --denser-stereo
```
