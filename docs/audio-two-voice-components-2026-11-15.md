# Testing which of two voices contains a short synthetic edit

The earlier attenuation check could flag the right time while favoring
the wrong estimated voice. This benchmark independently changes the
primary voice and the competing voice. It tests whether a flag locates
the edited interval and whether the favored estimate resembles the
voice that actually changed. Component-wise mixed-audio benchmarks
motivate the matched design. Here both components are speech and the
known targets are short replacements on separate sample clocks.
[Earlier attenuation comparison](audio-stream-attenuation-2026-11-15.md) ·
[Component-wise detector study](https://arxiv.org/abs/2610.17617) ·
[MixFake benchmark](https://arxiv.org/abs/2605.23201)

## Sources and exact controls

Use 18 declared readers in nine disjoint pairs. The six PhonemeDF
development readers supply 12 scenes. Four different PhonemeDF readers
supply eight held scenes. The eight separate MLS stock-source readers
supply eight more held scenes. Each PhonemeDF pair uses four different
utterance pairs. Each stock pair reuses its two reader recordings with
two placement schedules and reversed generator assignment. Four generator
families appear across the held sets. The source recordings were used in
earlier studies. Every stereo mixture in this benchmark is a new render.
[PhonemeDF release](https://zenodo.org/records/19522979) ·
[Frozen PhonemeDF inputs and terms](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles) ·
[Stock voices and genuine MLS controls](audio-random-clock-stock-2026-11-13.md#frozen-inputs)

Each scene yields five six-second deliveries at 16 kHz. Both voices
undergo a genuine edit, only the primary voice has a synthetic edit,
only the competing voice has a synthetic edit, or both edits are synthetic.
An additional untouched delivery has neither edit. All instances of each
voice state use the same rendered stem. In the PhonemeDF material, the
genuine control round-trips the source words through resampling. Three
of those control fragments were byte-identical, so the frozen same-reader
genuine substitution fills those cells. The stock material already has
matched same-reader genuine splices. The native validator verifies that
non-edit PCM is identical for all versions of a voice before rendering.

Half of the scenes place edit starts at 1.45 and 3.65 seconds; the other
half start them at 2.25 and 2.40 seconds so the edits overlap. The louder
voice uses the second clock. Whole-track gain is fixed using the two
untouched stems, with the competing voice 5 dB louder in the mono mean.
The stereo renders use the measured left and right room responses, with
directions alternated. Calibration uses the Surrey room. Both held
source sets use the BBC room. Truth ranges include the dominant measured
arrival delay of each impulse response. Reflections extend beyond those
ranges. These are stationary two-source renders with known geometry,
separate from recordings of conversations in a room.
[Surrey measurements and terms](audio-room-assets-v299-method.md) ·
[BBC measurements and terms](audio-bbc-room-assets-v303-method.md)

## Checks and measurements

The current two-second app scan runs on the exact mono mean. Its included
scope is the mono short-review path. The candidate classifier reads
one-second windows every half second. It runs the same weights on the
mono mean and on anonymous stereo views from the earlier lag and angular
mask method. Discard separated views when two usable directions are not
available. Compare the strongest separated response with a rule requiring
it to lead all other admitted estimates at that same time by at least log(2)
in log odds. Each rule has a separate cutoff set on the development
untouched and genuinely edited controls. The primary point uses the
largest control response. The wider point allows at most one development
speaker pair to flag and is also recorded before held inference.
[Directional mask method](audio-spatial-mask-voice-transfer-2026-11-13.md) ·
[Pinned classifier and license](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification/tree/f7050b586236dc910d1157f430def2d0647b02b4) ·
[Current voice asset terms](commercial-model-audit.md)

A useful window covers half of the known edit and reaches 0.30 temporal
intersection over union. On the both-synthetic files the replay counts
whether the single selected interval overlaps both known changes. Compare
that interval to the untouched and genuine-edit controls separately.
Attribution uses the two benchmark stems only after inference. A favored
stereo estimate is associated with the closer stem at the selected clock
if their scale-invariant signal-to-distortion ratios differ by at least
1 dB. This test treats a well-timed interval associated with the other
voice as an attribution error. A mono finding provides timing without a
speaker assignment. No benchmark stem or edit clock enters inference.

An independent recount caught an index mix-up between estimates and time
windows in the initial experiment runner. The native classifier readings
were collected on the full grid independently of this selector. We added
a regression check, recomputed the cutoffs using only the development
responses and reran the held inference before inspecting the candidate
outcomes. The original two-second app scan was separate from the affected code.
All held raw scores and stereo estimates reproduced exactly on the rerun.

## Held results

Counts in the table give usable times for primary-only, competing-only
and both-voice edits. Each cohort has eight files per edit condition
and 16 ordinary controls. "Right voice and time" counts at most one
estimate per delivery. An overlap in a both-voice file can identify one
of the edited voices while missing the other.

| Check | PhonemeDF useful times | PhonemeDF right voice and time | PhonemeDF ordinary flags | Stock useful times | Stock right voice and time | Stock ordinary flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current two-second app scan | 0 / 0 / 0 | — | 0 | 0 / 1 / 1 | — | 0 |
| One-second mono, strict | 0 / 0 / 0 | — | 0 | 0 / 0 / 0 | — | 0 |
| One-second mono, wider | 0 / 0 / 0 | — | 6 | 0 / 1 / 1 | — | 6 |
| Strongest directional estimate, strict or wider | 0 / 0 / 0 | 0 | 0 | 0 / 0 / 0 | 0 | 1 |
| Directional estimate with same-time lead, strict | 0 / 0 / 0 | 0 | 0 | 0 / 0 / 0 | 0 | 0 |
| Directional estimate with same-time lead, wider | 1 / 0 / 0 | 1 | 4 | 0 / 0 / 1 | 1 | 2 |

The wider lead rule added two intervals that the current scan missed.
In a primary-only PhonemeDF edit, it selected **2.000–3.000 s** for the
known **2.256–2.736 s** replacement and favored the primary voice. On a
stock file with staggered edits to both voices it selected **4.000–5.000 s**
for the louder voice's **3.656–4.616 s** replacement. Neither interval
recurred at the edit clock in its own untouched or genuine-edited control.
The same rule flagged six ordinary files out of 32 and selected
incorrect times on eight edited files. The stricter rule located no edits.
None of the tested checks located both edits in the same file.

The two-second app grid had an interval meeting the overlap rule on 36
of 48 edited deliveries. Its windows were too broad for the 12 shorter
PhonemeDF deliveries, even if centered on the edit. The one-second grid
could reach all 48. All 140 deliveries produced at least two estimated
directions, and two held stock deliveries had no usable same-time lead.
The test gives a reason to keep this branch in evaluation. Event timing
improved for two files at the wider point, while the false alarms and misses
are too frequent to use it for speaker attribution in customer analyses.

A second renderer rebuilt 140 unclipped mixtures and 308 distinct
waveforms including the source stems byte for byte. An independent
scalar recount reproduced 70 held outcome rows and all cutoffs. The
same readers occur in several matched conditions and placements. Counts
describe this controlled comparison and its particular room responses.

## Replay

The numeric package has only sample offsets, hashes and scalar readings.
Its review is runnable without weights or source recordings.

```sh
export PYTHONPATH=backend
module=scripts.audio_two_voice_components
receipt=docs/evaluation-panels/audio-two-voice-components-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native reproduction use the dry-source directories referenced above
and the root containing the two previously qualified room-asset folders.
The classifier and speech activity weights use the same snapshots as the
[two-voice attenuation experiment](audio-stream-attenuation-2026-11-15.md#replay).
Choose a fresh output path for each phase.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/input" "$phoneme" "$stock" "$rooms"
"$runner" -B -m "$module" score development "$run/input" "$run/dev" "$classifier" "$speech"
"$runner" -B -m "$module" score held "$run/input" "$run/held" "$classifier" "$speech" \
  --development "$run/dev/scores.json"
"$runner" -B -m "$module" current "$run/input" "$run/app.json" "$voice_weights"
python3 -B -S -m "$module" combine "$run/input/freeze.json" "$run/dev/scores.json" \
  "$run/held/scores.json" "$run/app.json" "$run/receipt.json"
```
