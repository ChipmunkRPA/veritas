# Speech and mouth timing on stereo AAC

The stereo trial preserves exact audio and picture times and can produce
playable evidence. The fresh test found one of 19 known replacements and
flagged five of 19 untouched clips with centered speech. A matched mono
replay on the same speakers found zero replacements and flagged six
untouched clips. The results support a development-only test option.
Normal stereo analyses keep the existing coverage.

## Input and scope

Use 34 retained Lombard Grid clips from 17 speakers and 19 separate
clips from 12 speakers outside the earlier articulation fit, calibration
and held panel. The new roster was fixed before scoring. Each target
supplies untouched speech, a replacement of the same spoken words and
a replacement of two different words from the same speaker. The audio
outside the labeled range stays sample-identical before encoding. The
picture track stays intact. These are soundtrack substitutions with
unchanged mouth movement. [Source recordings and terms](https://zenodo.org/records/3736465)
· [Earlier articulation study and pins](video-lip-articulation-2026-11-15.md)

Each of the three speech variants is delivered with centered speech, a
panned CC0 environmental sound, and a separately panned genuine meeting
excerpt. Center the foreground equally in both channels. Set the
environmental sound to −12 dB in its louder channel and the competing
speech to +6 dB. Use the same background and input gain across the
variants of a target. One further delivery reverses the polarity of the
right channel in the untouched clip. The study uses 48 kHz AAC-LC at
160 kbps. Environmental and meeting recordings come from previously
verified project panels. A native audio check excludes almost silent
environment excerpts before assignment. Backgrounds recur across some
target groups.
[Environmental source checks](environmental-component-transfer-2026-11-16.md)
· [AMI recordings and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The compared methods apply the existing speech-to-mouth check to the
same decoded mono mix. One assesses all stereo inputs with measured
clocks. The other checks that both channels support the mix across the
entire presented sample extent. It compares level difference,
correlation and attenuation in 400 ms sections, with coverage of the
final partial section. A stereo channel conflict makes the timing
evidence inconclusive. Both methods use the existing operating points
and weights. The measured first stream must be continuous at the source
clock and must be a two-channel 48 kHz AAC-LC soundtrack in MP4. A useful
range overlaps at least 0.40 of the temporal union of the result and the
known replacement. [Speech-content timing research](https://arxiv.org/html/2610.30872v1)
· [Checkpoint terms](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)

## Results

All 530 deliveries had verified clocks. Each channel aligned to its
input within one 16 kHz analysis sample. The table compares use of all
downmixes with the channel-agreement rule. The denominators include
inconclusive deliveries. Same-word substitutions are processing
controls and their alerts are shown independently from pure originals.

| Speakers and conditions | Useful locations, all → agreement | Untouched alerts, all → agreement | Same-word alerts, all → agreement |
| --- | ---: | ---: | ---: |
| Retained, centered | 5 → 5 of 34 | 1 → 1 of 34 | 3 → 3 of 34 |
| Retained, panned environment | 8 → 7 of 34 | 1 → 1 of 34 | 3 → 3 of 34 |
| Retained, louder competing speech | 2 → 0 of 34 | 1 → 0 of 34 | 2 → 0 of 34 |
| New speakers, centered | 1 → 1 of 19 | 5 → 5 of 19 | 4 → 4 of 19 |
| New speakers, panned environment | 0 → 0 of 19 | 1 → 1 of 19 | 2 → 2 of 19 |
| New speakers, louder competing speech | 1 → 0 of 19 | 1 → 0 of 19 | 1 → 0 of 19 |

The channel rule declined all 159 louder competing-speech deliveries
and all 53 opposing-channel controls. It admitted 45 of 53 environment
deliveries containing a replacement and 45 of 53 untouched environment
deliveries. Three of 19 centered edited files from new speakers were
inconclusive after timing measurement. The same-speaker result among
the retained clips cannot substitute for the fresh-speaker comparison.
[Stereo measurements](evaluation-panels/video-lip-stereo-transfer-2026-11-16.json.gz)

A secondary comparison used the exact same 19 new source clips and 57
centered stereo case inputs. It encoded the shared waveform as mono
48 kHz AAC and ran the same fixed timing rule. The mono results gave
0/19 useful locations, 6/19 alerts on untouched inputs, and 4/19
same-word alerts. Centered stereo gave 1/19, 5/19 and 4/19 respectively.
The untouched alerts in each carrier cover five distinct speakers.
This comparison suggests that the poor transfer involves the underlying
timing method across these speakers. The difference between formats is
small, with strong background interference still unresolved.
[Matched mono measurements](evaluation-panels/video-lip-stereo-mono-2026-11-16.json.gz)

## Local trial and replay

The development-only `VERITAS_VIDEO_LIP_STEREO_AAC_REVIEW` option
defaults off. It uses the existing short, continuous shot scope and
requires the local 48 kHz articulation option. The sound excerpt and
waveform identify the analyzed mix of both channels. Source verification
recomputes the channel measurements and checks the exact audio samples,
frame timestamps and mouth crops. Findings carry zero weight in overall
AI confidence.

Five simulated USD 9.99 checkouts exercised the full local worker. No
analysis job queued before payment. On the selected environment mix,
the trial highlighted audio **0.800–1.200 s** and corresponding picture
time **0.840–1.240 s**. Browser playback decoded the 0.400-second excerpt
and displayed the waveform and three original mouth crops. The PDF,
JSON and signed evidence used the same times and the signed package
verified against the submitted MP4. Toggling the trial preserved the
automated assessment. The unchanged and same-word controls produced no
timing packet. The opposing-channel case remained inconclusive. This
checkout verifies the evidence path on selected examples from the panel.
[Local replay record](evaluation-panels/video-lip-stereo-app-2026-11-16.json)

The public receipts store measured coordinates, channel levels,
correlations, outcomes and digests. The standard-library reviewer
recounts source overlap, admission and the mono comparison. The voice
recordings, facial crops and local staged files stay in the test folder.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
module=scripts.video_lip_stereo_transfer
stereo_result="$panels/video-lip-stereo-transfer-2026-11-16.json.gz"
python3 -B -S -m "$module" verify "$stereo_result"
python3 -B -S -m "$module" verify-mono "$panels/video-lip-stereo-mono-2026-11-16.json.gz" "$stereo_result"
```

Restore the earlier articulation plan and held picture capture using
the [pinned archive instructions](video-lip-articulation-2026-11-15.md#local-app-and-independent-replay).
Restore the verified environmental run and the AMI room receipts from
the linked source studies. The latter is the verified room directory for
the [fresh six-second audio panel](audio-six-ten-context-2026-11-16.md#replay).
The acoustic and landmark assets are the same licensed versions used by
the app's mono check.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$plan" "$sources" "$environment" "$rooms" "$run/stereo-plan.json"
"$runner" -B -m "$module" score "$run/stereo-plan.json" "$plan" "$pictures" \
  "$sources" "$environment" "$rooms" "$assets" "$run/stereo.json.gz"
"$runner" -B -m "$module" mono "$run/stereo.json.gz" "$run/stereo-plan.json" \
  "$sources" "$environment" "$rooms" "$assets" "$run/mono.json.gz"
npm --prefix frontend run build
"$runner" -B -m scripts.video_lip_articulation_app "$local/app" "$sources" "$run/stereo-plan.json" \
  "$run/stereo.json.gz" "$assets" --stereo-environment "$environment" --stereo-rooms "$rooms" \
  --standard-checks --browser
```
