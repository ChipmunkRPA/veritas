# Repeated background sound in lossless 48 kHz stereo

Chipmunk Verify can report two locations with similar sound after reducing the
dominant sound shared across stereo channels. This check uses short, continuous,
lossless 48 kHz stereo, including eligible video soundtracks. It preserves the
source sample positions, native stereo excerpts, aligned excerpts, and waveforms.
The finding carries zero weight in the AI assessment. Music, recurring alerts,
and ordinary edits can produce repeated sound.

## What the check measures

The check evaluates all three native alignments within each 16 kHz time step.
It selects an offset and gain supported by several active windows. It then
subtracts the shared signal using the original 48 kHz samples. A band-limited
search proposes repeat locations and checks agreement in the resulting waveform.
The evidence retains the original sample ranges needed to reproduce the
subtraction exactly. A stable timing and level relationship is necessary. This
method uses no trained weights. [Spatial transfer-function research](https://arxiv.org/abs/2503.09412)

The app verifies the start, continuity, sample rate and full sample count on the
native clock as well as the existing decoded clock. It admits 2.5–4 second stereo
PCM16 and 16-bit FLAC. The local video replay uses a PCM soundtrack starting
0.400 seconds after the pictures. Lossy sources continue through the other
configured audio checks. Set `VERITAS_AUDIO_STEREO_NATIVE_REPEAT_REVIEW=true`
in development to run the new comparison.

## Fresh comparison

Two DEMAND scenes supplied development examples. Four separate scenes supplied
32 test excerpts from a park, a hallway, a restaurant, and a station. The
48 kHz background comes from microphone channels 1 and 8. Different AMI
recordings supply the foreground voice in the two splits. A staged repeat is
240, 400 or 640 ms long. The test mixes the same voice with delays of 4–35
native samples and different channel gains. It records both edit ranges before
mixing. Each correct pair overlaps at least half the union of both labeled
ranges. [DEMAND audio and CC BY-SA 3.0 terms](https://zenodo.org/records/1227121)
· [AMI audio and CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Outcome | Existing mono checks | Align after conversion to 16 kHz | Align native 48 kHz |
| --- | ---: | ---: | ---: |
| Correct repeats with speech 10 dB louder | 0/32 | 5/32 | 20/32 |
| Correct repeats with speech 18 dB louder | 0/32 | 0/32 | 20/32 |
| Controls with a reported pair | 0/128 | 0/128 | 0/128 |
| Controls assessed | 122/128 | 72/128 | 81/128 |

The native search found correct ranges in all four scene groups. It returned
one misplaced pair and left three edited files unassessed at each loudness.
The two loudness sets share the same 32 background excerpts. Native stereo
clips with unsteady timing or level often have insufficient alignment. An
early development fixture used a replacement already present in the clip.
The corrected control uses a separate excerpt and was replayed before the
confirmation measurements. [Held results](evaluation-panels/audio-native-alignment-2026-11-17.json.gz)
· [Corrected development comparison](evaluation-panels/audio-native-alignment-development-2026-11-17.json.gz)

## App and independent replay

The app replay checks the same edit in WAV, FLAC and a video with
offset lossless audio. It verifies 48 kHz playback, the signed original
samples and the same waveform pixels in the browser and Standard PDF.
Each waveform uses its own peak scale so quiet patterns remain visible.
It checks an ordinary input, a lossy AAC input, and the edited WAV with
the option disabled. The same WAV receives the same AI assessment with
the option on or off. All six local checkouts use a Stripe test double.
[App replay receipt](evaluation-panels/audio-native-alignment-app-2026-11-17.json)

Use the [source roster](evaluation-panels/audio-native-alignment-sources-2026-11-17.json)
to extract the two recorded channels from each pinned publisher archive.
The output filenames are `DEMAND scene/ch01.wav` and
`DEMAND scene/ch08.wav` below the source directory. Save the roster in that
directory as `recorded-sources.json`. The measurement verifies the
uncompressed files by SHA-256. The four AMI filenames are
`<meeting>.Array1-05.wav` under the AMI directory. Their SHA-256 pins are in
the two result files. Raw audio and test renderings stay in local storage.
The portable input audit verified 288 WAV deliveries, 576 input and parent
file bindings, and both original channels from all six recorded scenes.
Use the same source renders for comparing other repeat-localization methods.
[Portable file bindings](evaluation-panels/audio-native-alignment-bindings-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panel=docs/evaluation-panels/audio-native-alignment-2026-11-17.json.gz
python3 -B -S -m scripts.audio_native_alignment_panel review "$panel"
"$runner" -B -m scripts.audio_native_alignment_panel measure "$recordings" "$ami" \
  confirmation "$run/confirmation"
npm --prefix frontend run build
"$runner" -B -m scripts.audio_native_alignment_app "$run/confirmation" "$run/app" --browser
```
