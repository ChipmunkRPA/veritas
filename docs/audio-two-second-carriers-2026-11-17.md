# Short speech evidence on the original 48 kHz clock

In a local Full analysis, **Two-second speech windows** can review continuous
mono recordings between nine and fourteen seconds long. The existing check
already supports native 16 kHz WAV and FLAC. The local extension supports 48 kHz
PCM16 WAV and FLAC, and 48 kHz mono AAC-LC in an
MP4-family audio container. It requires one stream and a verified zero-based
presentation clock. The AAC tests use a nominal 96 kb/s rate. The eligible
range is 80–112 kb/s measured by the container probe.

Enable `VERITAS_AUDIO_TWO_SECOND_LONG_REVIEW=true` on the development
backend. `VERITAS_AUDIO_TWO_SECOND_LONG_48K_REVIEW=true` enables lossless
uploads. `VERITAS_AUDIO_TWO_SECOND_LONG_AAC48_REVIEW=true` enables AAC.
Both carriers require a confirming window. The existing 16 kHz options
continue to govern 16 kHz uploads.

Two speech-admitted windows must return elevated responses and overlap by at
least one second. Overlapping excerpts can contain an identical window. Its
response is computed once at that sample position. Confirmation requires a
different start time on the same upload.
The report preserves both complete windows, the original 48 kHz
source ticks and the 16 kHz samples used by the detector. Codec priming and
final decoded samples beyond the presented extent are excluded from this
review. Files with gaps, shifted starts or uncertain sample endpoints stay
outside its scope. Findings carry zero weight in the overall AI estimate.
[Existing calibration and source verification](audio-two-second-long-review-2026-11-16.md) ·
[Commercial terms for the existing weights](../backend/app/models.yaml)

Recent work on speech-edit localization distinguishes boundary evidence from
the response measured within the segment. The public review preserves that
distinction by showing the entire windows and their overlap as a clue to inspect.
Short edits can occupy too little of a two-second window. The smallest label
that can pass the event counter's 30% overlap rule inside a two-second window
is 0.6 seconds. The method uses the previously reviewed serving weights and
per-carrier points. No code or new pretrained weights from the publication
are used. [Boundary and within-segment research](https://arxiv.org/abs/2609.25822)

## Held comparison in louder speech

The [earlier longer-input study](audio-two-second-long-review-2026-11-16.md#fresh-transfer-procedure)
and [separate six-to-ten-second source study](audio-six-ten-context-2026-11-16.md)
supply two cohorts. Each uses 12 original speech edits from six published
generator families, 12 unedited recordings, and the 24 exact room tracks from
those mixtures. The two cohorts share no declared reader, room identity or
input PCM. The source recordings were used in earlier project studies. The
48 kHz encodes and this comparison are new. Results cover the original 16 kHz
input bandwidth. The competing conversation is nominally 10 dB
louder over each full recording. [Publisher audio edits and labels](https://zenodo.org/records/14214149) ·
[Original speech recordings](https://www.openslr.org/12/) ·
[Competing speech terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Re-encode the original mixed PCM to each format with FFmpeg 7.0.2 and verify
length and time before scoring. The comparison uses the established
three-second selection on each decoded carrier and adds the independently
confirmed two-second windows. The test's three-second comparator is run on
both carriers, including files where the local app offers fewer optional
windows. Useful regions must cover at least half of the labeled edit and
achieve at least 0.30 temporal intersection over union. Both methods use the
same source interval labels and canonical persisted logit precision.

| Held inputs | Correct edit locations, before → with two windows | Ordinary clips flagged, before → after | Isolated room clips flagged, before → after |
| --- | ---: | ---: | ---: |
| WAV, first cohort | 10 → 12 of 76 | 2 → 2 of 12 | 0 → 0 of 24 |
| WAV, second cohort | 10 → 13 of 81 | 0 → 1 of 12 | 0 → 0 of 24 |
| AAC, first cohort | 12 → 13 of 76 | 2 → 2 of 12 | 0 → 0 of 24 |
| AAC, second cohort | 10 → 12 of 81 | 0 → 1 of 12 | 0 → 0 of 24 |

All 192 deliveries were assessed. The two carriers share the same 157
labels and control clips. The same previously clear ordinary recording
gained a flag in both. The extension preserved earlier useful locations.
One edited WAV and two edited AAC deliveries gained only misplaced windows.
Each combined check missed 132 of 157 labeled intervals. Fifty-six labels
are shorter than 600 ms and cannot meet the two-second overlap requirement.
The measured AAC bit rates ranged from 96,284 to 101,110 b/s. Both profiles
qualified for local source evidence with the existing confidence calibration
for the rest of the analysis. [WAV measurements](evaluation-panels/audio-two-second-carrier-wav-2026-11-17.json.gz) ·
[AAC measurements](evaluation-panels/audio-two-second-carrier-m4a-2026-11-17.json.gz) ·
[Independent event and clock checks](evaluation-panels/audio-two-second-carrier-audit-2026-11-17.json)

## Local application replay

Seven simulated USD 9.99 checkouts covered the edited WAV and AAC with the
extension off and on, a FLAC version of the same lossless audio, and one
ordinary recording in each measured carrier. The edited example under a
louder voice flagged **8.993375–10.993375 s** in all three formats. A second
window covered **9.993375–11.993375 s** and ended exactly at the presented
audio boundary. The AAC decoded 277 more samples at 16 kHz than it presented.
No extra samples entered the plotted regions or confirmation. An AAC remux
with a shifted start was rejected by the clock qualification.

The browser loaded every playable excerpt and waveform. The JSON and PDF
showed the same original ticks for both windows. The signed export replay
regenerated the clips and charts from the actual WAV, FLAC and AAC uploads.
The ordinary controls stayed clear, and the earlier findings and overall
assessment agreed for the paired on and off runs. The queue held zero new
analysis jobs before each simulated payment.
[Browser and source replay receipt](evaluation-panels/audio-two-second-carrier-app-2026-11-17.json)

An additional eight local checkouts reproduced the earlier 16 kHz results
with the old single-window and confirmed modes, including signed-source replay
and the browser. The additional source check did not require changes to those
evidence schemas.

## Reproduce the comparison

Keep downloaded media on the local scratch drive. The public receipts carry
input hashes, published labels, clock measurements and scalar responses.
The portable reader recounts the same comparisons with standard Python.
The separate review supplies exact carrier commitments and a general-format
manifest, reference intervals, and predictions for evaluating another
detector. The native readback re-encodes each carrier, measures its frame clock
with FFprobe, and checks that a separately decoded PCM stream matches the
recorded input hash. No classification model is needed for that readback.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_two_second_carrier_transfer
panels=docs/evaluation-panels
python3 -B -S -m "$module" verify "$panels/audio-two-second-carrier-wav-2026-11-17.json.gz"
python3 -B -S -m "$module" verify "$panels/audio-two-second-carrier-m4a-2026-11-17.json.gz"
"$runner" -B -m "$module" freeze "$first" "$second" "$run/plan.json"
"$runner" -B -m "$module" score "$run/plan.json" "$first" "$second" "$run/wav.json.gz" wav
"$runner" -B -m "$module" score "$run/plan.json" "$first" "$second" "$run/m4a.json.gz" m4a
python3 -B -S -m scripts.audio_two_second_carrier_audit "$run/wav.json.gz" "$run/wav-review" \
  --native "$first" "$second"
python3 -B -S -m scripts.audio_two_second_carrier_audit "$run/m4a.json.gz" "$run/m4a-review" \
  --native "$first" "$second"
```

Set `first` to the rebuilt input directory from the [longer-file transfer](audio-two-second-long-review-2026-11-16.md#reproduce).
Use the native prospective output from the [six-to-ten-second comparison](audio-six-ten-context-2026-11-16.md#replay)
for `second`. Both contain a `freeze.json` and the exact PCM files. Set `run`
to an empty writable work directory. The app replay uses a production frontend
build, Chromium, an isolated database, and simulated Stripe.

```sh
"$runner" -B -m scripts.audio_two_second_carrier_app "$first" "$second" "$run/app" \
  "$run/wav.json.gz" "$run/m4a.json.gz"
```

## Source coverage for the overlapping scan

New saved reviews list the parts of the recording covered by at least one
checked two-second window and the intervals where no covering window had
enough detected speech to run. The browser, API, PDF, JSON, and signed
evidence use the same source intervals. Overlapping windows contribute
once to covered duration. A window repeated across excerpts counts once
and qualifies as checked if any recorded occurrence was admitted. The
original excerpt records preserve each admission count.

The native 48 kHz view lists exact source ticks and stops at the verified
presentation boundary. Extra samples emitted by the AAC decoder do not
extend analysis coverage. The signed verifier reconstructs the map from
the saved admissions and original clock. Earlier signed reviews replay
under their original evidence version. A scan that did not complete
includes its skip reason.

Four local analyses completed with simulated checkout, including an
unassessed traffic recording, a mixed-coverage WAV, the legacy evidence
version of that WAV, and a mixed-coverage AAC file. The AAC input decoded
to 192,171 samples. The presented extent stopped at 192,000 samples on
the analysis clock. The browser verified seven displayed intervals.
The WAV's AI assessment matched with the earlier and newer coverage
formats. [Local application receipt](evaluation-panels/audio-two-second-source-coverage-app-2026-11-18.json)
