# Locating a mismatch between speech sounds and mouth opening

The local video trial now compares visible mouth openings with groups of
speech sounds whose articulation tends to look different. The existing
timing checks retain priority. The new check can supply a range when
speech and picture agree better before and after that range under a
shared timing alignment. It uses the existing pinned acoustic and
face-landmark assets plus a small visual fit trained on ordinary
recordings. No third-party lip-sync weights are needed.
[Timing and articulation research](https://arxiv.org/html/2610.09223v1) ·
[Speech-content replacement research](https://arxiv.org/html/2610.30872v1) ·
[Pinned Apache-2.0 acoustic checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4) ·
[Landmark asset and runtime review](video-bilabial-mouth-transfer-2026-11-14.md)

## Method and held results

Fourteen fit speakers supplied 84 unchanged Lombard Grid recordings and
537 usable acoustic events. Those events train the visual fit to read
closure, open, rounded, spread and lower-lip articulation. One arm
uses measured opening, relative opening within the clip and change
between consecutive frames. The other adds lip width and the shape of
the opening. The detector aligns events to native frame timestamps.
Both sides of a possible mismatch must support a common timing, and
the comparison allows a one-step timing tolerance inside the range.

Eight other speakers supplied 16 recordings for calibration. The maximum
per-speaker score on untouched, gain-only, same-word, and uniform-delay
controls fixes the tighter operating point. The second highest fixes the
wider point. There are 17 further held speakers, with one plain and one
Lombard recording per speaker. The audio and picture members differ from
those in the prior same-speaker study. Each held recording has a
same-speaker two-word replacement and a duration-matched same-word
replacement. The groups speak short scripted utterances with the mouth
visible. Lombard speech changes speaking style. The delivered tracks have
no added background sound. [Lombard Grid corpus and CC BY 4.0 terms](https://zenodo.org/records/3736465) ·
[Earlier same-speaker comparison](video-lip-content-transfer-2026-11-15.md)

A correct location reaches at least 0.40 intersection over union with the
known edit interval. The second delivery encodes and decodes the same
held soundtracks as 48 kbps mono AAC at 16 kHz. It preserves the source
picture files and checks the decoded audio origin within one sample.
Each result below uses the same tight operating point across both
deliveries.

| Audio delivery and check | Correct ranges /34 | Misplaced ranges /34 | Ordinary alerts /68 | Same-word splice alerts /34 | Uniform-delay alerts /68 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCM, existing timing | 0 | 1 | 2 | 1 | 2 |
| PCM, add mouth-opening classes | 7 | 3 | 3 | 3 | 3 |
| PCM, add full landmark shape | 3 | 1 | 4 | 2 | 3 |
| AAC, existing timing | 1 | 1 | 2 | 1 | 2 |
| AAC, add mouth-opening classes | 10 | 4 | 3 | 3 | 3 |
| AAC, add full landmark shape | 5 | 1 | 4 | 3 | 2 |

The mouth-opening classes recovered seven PCM ranges and nine additional
AAC ranges. All useful locations avoided the matching same-word alert
at that location. Three PCM edits and one AAC edit remained unassessed.
The wider mouth-opening point raised PCM recovery to nine and ordinary
alerts to five. It added no correct AAC locations. The local trial uses
the tighter mouth-opening fallback. The clips establish coverage on
staged soundtrack substitutions. Accuracy on unrestricted conversation,
background noise, and regenerated mouth motion remains unmeasured here.
[PCM measurements](evaluation-panels/video-lip-articulation-pcm-2026-11-15.json.gz) ·
[AAC measurements](evaluation-panels/video-lip-articulation-aac-2026-11-15.json.gz)

## Local app and independent replay

Enable `VERITAS_VIDEO_LIP_ARTICULATION_REVIEW` alongside the existing
short video lip-timing settings. The trial applies to supported continuous
shots at roughly 24–25 fps with mono 16 kHz PCM or AAC and verified clocks.
For a match, the app presents the original before, during and after frames,
the mouth crops, the selected audio excerpt and its waveform. The during
frame depicts the strongest supported mismatch in the range. The PDF,
JSON and signed evidence preserve the cues and their source times.
The observation carries zero weight in the AI estimate.

The tested AAC case flagged audio **0.640–1.040 s**, corresponding
to picture time **0.680–1.080 s**. Frame 21 shows the mouth region
compared with a speech event at **0.853 s**. The unchanged and same-word
copies had no evidence packet in the checkout test. The four simulated
USD 9.99 purchases queued zero jobs before payment. The signed replay
verified the soundtrack, 45 native frame positions and their mouth crops.
Chromium loaded the three original mouth crops and waveform in the plain
results view and decoded the 0.400-second playback excerpt. An
additional checkout run exercised the full configured detector set and
returned the same customer assessment with the new evidence enabled.
[Checkout and evidence receipt](evaluation-panels/video-lip-articulation-app-2026-11-15.json) ·
[Full detector checkout receipt](evaluation-panels/video-lip-articulation-app-full-2026-11-15.json)

The portable reader checks partitioning and totals, and recomputes
reported frame times from the committed source clock. It uses the
standard library alone. Native replay needs the pinned publisher
sources and local timing assets from the linked instructions. The public
receipts contain measurements, class coefficients and file commitments.
Raw source files, participant labels and full landmark tracks stay local.
[Source archive preparation](video-neural-phone-clock-2026-11-14.md#independent-replay) ·
[Pinned audio and picture assets](video-lip-viseme-transfer-2026-11-15.md#independent-replay)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.video_lip_articulation_transfer
runner=backend/.venv/bin/python
receipt=docs/evaluation-panels/video-lip-articulation-aac-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$module" freeze "$sources" "$run/plan.json"
for phase in fit calibration held; do
  "$runner" -B -m "$module" capture "$phase" "$run/plan.json" "$sources" "$assets" "$run/$phase.json.gz"
done
"$runner" -B -m "$module" calibrate "$run/fit.json.gz" "$run/calibration.json.gz" "$run/points.json.gz"
"$runner" -B -m "$module" held "$run/points.json.gz" "$run/held.json.gz" "$run/pcm.json.gz"
"$runner" -B -m "$module" capture held "$run/plan.json" "$sources" "$assets" "$run/held-aac.json.gz" --aac
"$runner" -B -m "$module" held "$run/points.json.gz" "$run/held-aac.json.gz" "$run/aac.json.gz"
"$runner" -B -m scripts.video_lip_articulation_app "$run/app" "$sources" "$run/plan.json" "$run/aac.json.gz" "$assets"
```

Use `--standard-checks` to run the full configured worker. To replay the
plain browser view, build the frontend and add `--browser` with Chromium
installed for Playwright.
