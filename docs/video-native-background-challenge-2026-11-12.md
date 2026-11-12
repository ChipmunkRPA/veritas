# Fresh challenge for the paired background cut check

The prior 64-clip confirmation found no new false face-review ranges on genuine cuts. Two earlier false ranges motivated the guard. This challenge looks for genuine cuts that trigger the current tracked face review before asking the frozen guard to explain their boundaries. Its denominator measures the guard's behavior on existing alerts. [Earlier evidence and frozen method](video-native-background-edges-2026-11-12.md)

## Frozen sources and delivery

The archive supplied 22 eligible groups after excluding the 65 previously used groups and their documented source and donor keys. Six mutually disjoint pairs supply the challenge. Their clips retain the archive's publisher labels and native picture commitments. Unknown shared identity and earlier unrelated project use remain possible. [Source and rights record](video-common-source-preparation-v89.md)

Pair selection uses seven fixed preview pictures from genuine recordings. It weights outer grayscale appearance at 35% and whole-frame HSV palette similarity at 65%. The six pairs have normalized grayscale distances of .133–.246 and palette distances of .333–.525. Each pair supplies three exact cut durations in both directions and two treatments. Direct cuts switch the complete picture to another genuine recording and return. The ordinary color-grade treatment adjusts the entire inserted recording with a fixed, bounded per-channel gain and offset fitted on the genuine preview pictures. Both cuts use the host recording before and after the authored interval. This gives 36 direct cuts and 36 graded cuts.

The paired cut spans are **2.360–2.760 s**, **2.600–3.080 s**, and **2.840–3.240 s**. The twelve longer publisher swap controls run **2.000–2.720 s** or **2.240–3.120 s**. Twelve genuine originals, six camera punch-ins and six lighting adjustments complete the 108 clips. Every output is 640 × 360, exact 25 fps, 120 native H.264 pictures with 4:2:0 chroma. Each output tick takes the nearest verified source picture, breaking ties toward the earlier picture. The decoder never interpolates a source picture.

The direct screen calls the production Full probe, decode, shot, track, face and appearance stages in process. It keeps in-memory detector receipts and uses a dedicated local work directory. Four separate parity cases reproduced the exact paid worker decisions, source timestamps and coarse face-box bindings. They included both known genuine cut alerts, one true swap and one genuine original. The largest difference in the recorded coarse response was 3.81 × 10⁻⁷.

The candidate SHA stays at `ab7908fbaafe35a5e879190ada426401d7b248e8d37069db733b330a5df2eca6`. The entire incumbent journal seals before candidate scoring. The candidate only sees the real selected coarse range and source-bound face boxes. A useful correction must find both authored cut edges within one output picture. The gate requires at least two such corrections across two distinct pairs, at least one useful coarse swap to assess collateral loss, and zero lost useful swaps or false cut explanations on a swap or unaffected control.

## Held results

| Held inputs | Current coarse face ranges | Inputs with a shot boundary | Candidate accepted cut pairs |
| --- | ---: | ---: | ---: |
| Direct genuine cuts | 0/36 | 36/36 | 0 |
| Color-graded genuine cuts | 0/36 | 33/36 | 0 |
| Long publisher swaps | 12/12 | 0/12 | 0 |
| Original, camera and lighting controls | 0/24 | 0/24 | 0 |

None of the six genuine source pairs yielded a false face alert, including the three graded cuts that passed without a shot boundary. The candidate therefore had zero new cut alerts to correct and failed the usefulness gate. All twelve useful swap ranges were preserved. The candidate remains experimental. The original two false alerts still come from one known source pair.

The direct screen completed 108 Full stage runs in **459 seconds**. The independent replay reconstructed all 108 MP4s byte for byte, verified 1,680 native source pictures and 96 authored intervals, and recomputed every held decision. [Public result and replay summary](evaluation-panels/video-native-background-challenge-2026-11-12.json)

## Reproduce

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_native_background_challenge freeze "$run"
backend/.venv/bin/python -B -m scripts.evaluate_video_native_background_challenge freeze "$run"
backend/.venv/bin/python -B -m scripts.video_native_background_challenge prepare "$run"
backend/.venv/bin/python -B -m scripts.screen_video_native_background_challenge "$run"
backend/.venv/bin/python -B -m scripts.evaluate_video_native_background_challenge run "$run"
backend/.venv/bin/python -B -m scripts.review_video_native_background_challenge authoring "$run"
backend/.venv/bin/python -B -m scripts.review_video_native_background_challenge results "$run"
```

The commands use the retained private source archive, historical paid parity receipts and qualified local model cache. Pass `--known-paid` to the screen if the prior exact paid receipt is stored elsewhere. The public summary omits media, per-recording source keys, box coordinates and private frame commitments.
