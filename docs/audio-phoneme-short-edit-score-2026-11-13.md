# Voice-window checks on short same-word edits under competing speech

The test measures whether the existing voice response can locate exact
720–890 ms word replacements in six-second audio. It compares the normal
four-second Full grid, the two-second short review, and a two-second
background-reduced view. Each method uses the same existing classifier
and speech admission logic. The four-second and two-second window scores
remain source-bound on the original 16 kHz clock. [Source panel and controls](audio-phoneme-short-edit-intake-2026-11-13.md)
· [Voice asset terms](commercial-model-audit.md)

## Frozen comparison

The original Full windows start at 0 and 2 seconds. Short windows start
every second from 0 through 4 seconds. An interval is assessed only if
at least half overlaps detected speech. For each route, select the
highest admitted raw response, with the earlier range breaking ties.
Freeze a separate point at the strict maximum over its admitted
ordinary development responses. These include untouched speech, local
gain, genuine resampling and isolated competing speech. The stock
short-review point is also reported diagnostically. Six-second recordings
fall outside the existing calibrated whole-file confidence scope.

The background-reduced route uses the previously published fixed
spectral subtraction. Estimate noise power per frequency bin from
the bottom 20th percentile of the submitted recording, subtract a
scaled share, preserve a 35% amplitude floor and apply local smoothing.
This route receives the submitted waveform alone. The dry and isolated
masker waveforms are separate benchmark conditions. [Fixed spectral subtraction method](audio-noise-window-transfer-2026-11-12.md)

Useful localization requires at least half the true edit covered and
at least 0.30 temporal intersection over union. Because the edits are
shorter than a second, a four-second window can never meet that
intersection requirement even with complete coverage. The comparison
therefore also reports how often a flagged range covers half the edit,
how often a flagged range misses that criterion, and the strongest
window's alignment before any operating point. The paired ranking
comparison compares each synthetic replacement with its matched
ordinary resampling control, both across all admitted windows and
across the jointly admitted windows covering at least half the edit.

The 54 held deliveries cover two reader groups, two synthetic
mechanisms from one provider and one recorded AMI background session.
The same clips repeat across dry, +5, 0 and −5 dB. One held reader
had historical calibration exposure through different source
identifiers. The other sits outside the recorded calibration reader
roles. These counts describe this source panel.

## Held results

Each two-second cell gives strict locations out of four synthetic deliveries,
then flags on ordinary and isolated-background controls. The four-second
column gives broad coverage and control flags.

| Target-to-background ratio | Full, broad coverage / control flags | Short, strict locations / control flags | Reduced short, strict locations / control flags |
| --- | ---: | ---: | ---: |
| Dry | 3/4 · 0/8 | 4/4 · 0/8 | 4/4 · 0/8 |
| +5 dB | 3/4 · 4/10 | 3/4 · 4/10 | 3/4 · 4/10 |
| 0 dB | 2/4 · 4/10 | 2/4 · 0/10 | 2/4 · 1/10 |
| −5 dB | 1/4 · 0/10 | 0/4 · 0/10 | 0/4 · 0/10 |

Every held delivery was assessed by each route. No route flagged an isolated
background control or selected a flagged synthetic range that missed half
the edit. The six-second primary coverage includes broad ranges only. Its
sole strong-noise flag was on a voice-converted edit from **2.570–3.430 s**,
inside the reported **2.000–6.000 s** window. It gives a review lead that
spans four seconds.

Both short routes recovered two text-to-speech and two voice-conversion
edits dry, one and two at +5 dB, one and one at 0 dB, and zero of either at
−5 dB. At 0 dB they reported **1.000–3.000 s** for both synthesis versions
of a shared **1.590–2.460 s** destination. At +5 dB the voice-converted
version of the other destination also appeared in the **2.000–4.000 s**
range. Ordinary false flags at +5 dB include one untouched, one gain and
two ordinary resampling variants from the same held reader. The standard
short-review point retains the same synthetic locations and adds one
gain-control flag at 0 dB compared with the fitted short point. Background
reduction likewise adds one gain-control flag at 0 dB and zero synthetic
locations.

At −5 dB, the strongest two-second range overlapped usefully on three
of four synthetic deliveries before applying either short operating point.
Native short scores put zero of four synthetic cases above their matched
ordinary controls, compared with one of four for the reduced and Full
routes. The weak ranking at that noise level limits what a threshold change
can recover. The existing short review has a narrow use as a source-linked
inspection lead for eligible inputs. This panel supplies no basis for
adding the reduced route or revising customer confidence.
[Anonymous score and location receipt](evaluation-panels/audio-phoneme-short-edit-score-2026-11-13.json)

## Replay

The native freeze binds both the original and processed source-window
bytes, model and speech-activity asset hashes, runtime versions and
comparison methods. The run completed 488 development and 528 assessment
classifier forwards. Offline reconstruction reproduced all 108 delivery
commitments and 1,296 scheduled window slots. The runner seals development scores and points
before starting held inference. After the initial source acquisition
documented in the intake guide, use the same source caches for native
replay. The separate standard-library reader verifies all recorded
window bindings, same-input score consistency, counts and point
arithmetic.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.audio_phoneme_short_edit_score "$phase" \
    --cache "$new_cache" --old-cache "$old_phonemedf" \
    --record docs/evaluation-panels/audio-phonemedf-release-19522979.json \
    --odss-cache "$odss_cache" --ami "$ami_inputs" \
    --intake docs/evaluation-panels/audio-phoneme-short-edit-intake-2026-11-13.json \
    --run "$run" || exit 1
done
python3 -B -IS backend/scripts/review_audio_phoneme_short_edit_score.py \
  "$receipt" docs/evaluation-panels/audio-phoneme-short-edit-intake-2026-11-13.json
```
