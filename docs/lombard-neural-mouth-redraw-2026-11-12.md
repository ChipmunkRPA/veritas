# Testing lip-sync checks on neural redraws that preserve a speech event

This study compares a genuine mouth movement, an ordinary resample, a conventional timing edit and a neural redraw over the same native picture interval. The redraws use the same speaker and source-time expression. The study tests whether the mouth still opens at the corresponding acoustic event and whether existing checks detect the altered pictures. [Original source qualification](lombard-original-mouth-event-2026-11-12.md)

## Frozen comparisons

The 13 prequalified Lombard recordings include two development recordings from one speaker group and 11 held recordings from six other groups. All 13 scheduled sources stay in the denominator. Each source has four deliveries. The first preserves the original pixels. The next reconstructs the same native mouth region using ordinary bilinear resampling. The delayed control uses pixels from exactly five native pictures earlier. The neural delivery uses retained CPU float32 LivePortrait to reconstruct the region from the same-time frame. The three edits use the same source crop, inverse affine mapping, native alpha, and opacity schedule across the same 16 pictures. The retained human animation checkpoints and the geometry task have reviewed terms. The study uses the University of Sheffield's CC BY 4.0 media and retains the original files, edited videos, source identities and detailed geometry locally. [Publisher data](https://zenodo.org/records/3736465) · [Renderer paper](https://arxiv.org/abs/2407.03168) · [Publisher checkpoints](https://huggingface.co/KlingTeam/LivePortrait) · [Pinned MIT licence](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/LICENSE)

Deliveries use lossless RGB AVC in MOV with the exact picture time base and the original zero-origin 16 kHz float32 waveform. The independent replay reopens each file and checks every picture timestamp and PCM sample. It verifies native altered pixels and frame intervals against the original. The original and the edits share the same delivery profile, so the edit label cannot be inferred from the container or picture cadence.

A separate muted process receives opaque fixture IDs and the publisher's coarse forced-phone neighborhood. It measures mouth closure and reopening before the construction labels and acoustic measurements are joined. Both the original and post-render observations use a fixed landmark method. A redraw qualifies only if its unique source-neighborhood release is within one native picture of the original, remains inside the edit, and has a bracket midpoint within 80 ms of the independently measured acoustic burst. Conventional delayed controls must show the release five native pictures later, within one picture. This provides evidence of local event alignment. It does not establish the identity of each phoneme, perceived synchrony across the clip or performance on other generators.

The readout uses the released lip-timing adapter with directly observed single-face tracks, along with previously frozen research checks for lip timing and motion-aligned mouth texture. The mouth texture measurements ask whether a particular interval looks altered. The timing measurements ask whether the relationship between mouth and sound changed. A localized timing mismatch in an ordinary delayed control carries no AI inference. [Earlier timing test](video-lipsync-local-contrast-transfer-2026-11-12.md) · [Mouth texture test](video-mouth-temporal-detail-2026-11-12.md)

Recent work separately tests head-pose and lip-motion relationships and fine acoustic-articulatory correspondence. The latter reports a drop on a held retalking generator, which makes source and generator separation material here. Our same-time expression redraws retain the original utterance. They test the visual redraw gap and cannot test altered wording. Neither study’s checkpoint or dataset is part of this run. [Head-pose and lip motion](https://proceedings.mlr.press/v306/she26a.html) · [Fine audio-visual consistency](https://arxiv.org/abs/2610.30872)

## Development results

The muted observer confirmed the original acoustic-to-visible event in both neural redraws and both same-time resamples. Both delayed controls showed the intended five-picture delay. The renderer generated 32 neural pictures in 111.5 seconds at 2.56 GiB peak reported memory. The independent audit verified 508 picture timestamps, 334,080 unchanged float samples and 96 exact altered-picture masks across all eight deliveries. For one neural edit spanning **0.625–1.292 s**, the observed reopening bracket was **0.875–0.917 s**. Its midpoint was **13.3 ms** after the acoustic burst midpoint, and the reopening occurred on the same native picture as the original.

The released timing check measured one window on each of four deliveries. No development delivery had enough windows for the check to assess a local change, and it returned no ranges. The research timing check localized one of two observed delayed controls, but also flagged one genuine original and all four source-aligned edited clips. Each visual-detail arm localized one neural edit and one ordinary resample. The fine arm flagged one genuine original, and the scale-adjusted arm flagged both. The frozen held comparison uses the same measurements without fitting to these results.

The released check uses 1-second speech windows on half-second starts and requires 0.6 seconds of surrounding audio for its lag search. Four complete windows need at least 4.08 seconds at the published zero origins. A clock-only count allows at most one such window on every selected clip, even before speech and mouth quality checks. This tests the short-clip coverage limit. Performance on longer recordings requires a separate evaluation.

When an earlier copy was provided, the existing video comparison found the exact 16 altered pictures in both development redraws. Each pair yielded six linked before, during and after images, exact rational timestamps and zero soundtrack differences. This comparison describes what changed between the supplied copies and leaves automated confidence unchanged. The optional marked-event timing check, including its short-waveform fallback, returned an explicit profile limitation for both pairs. Its verified profile currently requires 75 pictures at 25 fps, 360 × 288 dimensions and 47,648 aligned audio samples.

## Held native results

The muted review confirmed the source-time event in **10 of 11 neural redraws**, covering all six held speaker groups and both speaking styles. One redraw had no unique reopening. All 11 originals and all 11 ordinary resamples retained their event. The observer independently verified the delayed event in eight of 11 ordinary timing controls and retained the remaining three as unqualified. For one held redraw, the exact edit spans frames **[16, 32)** at **0.667–1.335 s**, with the reopening in **0.918–0.959 s**. Its bracket midpoint was **5.9 ms** later than the matching acoustic burst midpoint and its reopening occurred on the same native picture as the genuine clip.

The held renderer completed 176 neural pictures in 607.3 seconds at 2.65 GiB peak reported memory. Independent readback of all 44 deliveries verified 2,756 native picture timestamps, 1,803,520 unchanged float PCM samples, and 528 exact altered-picture masks. The held originals retain four rational cadences from **24000/1001** to **145/6 fps**. Earlier-copy comparison found the exact changed range on all 11 neural pairs and generated all 66 linked frame exhibits. The marked-event timing route returned the profile limitation for all 11 pairs.

## Held detector results

The released single-upload timing check was **unassessed for local changes on all 44 held files**. It measured one speech window each on 20 files and none on 24. The frozen research timing candidate localized **0 of 8** independently observed timing changes. It flagged five genuine originals, six ordinary resamples, and four of the ten event-aligned neural redraws. A qualifying timing localization needed temporal overlap of at least 0.40 and a signed offset within two 25 Hz analysis ticks of the observed delay.

| Experimental visual check | Localized event-aligned neural redraws | Flagged untouched originals | Flagged ordinary resamples | Localized ordinary resamples |
| --- | ---: | ---: | ---: | ---: |
| Fine mouth detail | 2/10 | 6/11 | 5/11 | 1/11 |
| Fine detail adjusted for broad motion | 1/10 | 7/11 | 7/11 | 2/11 |

Both visual checks measured all 44 held files. The localization criterion was at least 0.50 temporal overlap. Neither candidate separated synchronized redraws from ordinary footage well enough to serve. When an earlier copy is available, the exact picture comparison has demonstrated value for locating the changes. For a short single upload, the current coverage result stays unassessed. [Sanitized replay and full counts](evaluation-panels/lombard-neural-mouth-redraw-2026-11-12.json)

## Local replay

Use the qualified private source directory and an unused private `run` directory. The macOS Face Landmarker runtime requires access to its local graphics context even with its CPU delegate. The retained LivePortrait weights run on the CPU with one Torch thread and unchanged bounded convolution. No new checkpoints are required.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.build_lombard_neural_mouth freeze "$sources" "$run"
backend/.venv/bin/python -B -m scripts.score_lombard_neural_mouth freeze "$run"
for split in development held; do
  for phase in controls neural; do
    backend/.venv/bin/python -B -m scripts.build_lombard_neural_mouth "$phase" "$sources" "$run" --split "$split" || exit 1
  done
  for phase in prepare observe audit; do
    backend/.venv/bin/python -B -m scripts.review_lombard_neural_mouth "$phase" "$sources" "$run" --split "$split" || exit 1
  done
  backend/.venv/bin/python -B -m scripts.score_lombard_neural_mouth score "$run" --split "$split" || exit 1
  backend/.venv/bin/python -B -m scripts.verify_lombard_neural_reference "$run" --split "$split" || exit 1
done
PYTHONPATH=backend python3 -B -S -m scripts.summarize_lombard_neural_mouth "$run"
PYTHONPATH=backend python3 -B -S -m scripts.summarize_lombard_neural_mouth docs/evaluation-panels/lombard-neural-mouth-redraw-2026-11-12.json --verify
```
