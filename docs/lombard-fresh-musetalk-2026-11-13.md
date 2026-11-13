# Fresh audio-driven mouth redraws on native Lombard clips

This source-qualification study uses six independently observed original speech events from six speaker groups absent from the earlier neural Lombard panel. The genuine inputs appeared in prior ordinary timing work, and their neural deliveries are new. The study uses retained commercially permitted MuseTalk V1.5 with its reviewed auxiliary speech and image weights. The archive copy was restored to Seagate without a new download, and the retained loader checked all seven publisher asset commitments before inference. [Source qualification](short-lip-release-timing-2026-11-12.md) · [Lombard terms and publisher](https://zenodo.org/records/3736465) · [Reviewed adaptation and asset permissions](mouth-multifamily-v7-evaluation.md) · [MuseTalk research](https://arxiv.org/abs/2410.10122) · [Publisher terms](https://github.com/TMElyralab/MuseTalk#disclaimerlicense)

## Frozen construction

Each source contributes the original, same-time ordinary resampling, an audio-conditioned neural redraw using its original timing, a neural redraw conditioned with audio five native pictures earlier, and an ordinary copy of mouth pixels from five native pictures earlier. The same 16-picture interval, mouth alpha and half-opacity edges apply to all four edits. Both neural conditions use the same seeds and original-time source crops. Their conditioning changes by five pictures. The shared native carrier is lossless RGB AVC with exact original float32 mono PCM in MOV. No picture or audio samples are interpolated for cadence conversion.

The renderer computes the original waveform's hidden states once. Each edited picture selects a fixed set of ten 50 Hz slots around the floor of its exact source time, or the exact time of the picture five positions earlier. All slots must be supported by real source PCM. The three original cadences retain their exact rational presentation timestamps. Each delivery preserves the native picture clock, source audio and pixels outside its committed edit geometry.

A separate muted pass receives opaque randomized file tokens and the coarse publisher phone neighborhood. It observes visual closure and reopening without the role, renderer timing, authored pixel masks or measured burst. A native readback independently reopens the delivered files and checks exact picture clocks, changed-picture bytes and every audio sample. A same-time redraw qualifies for a local event only if the original and same-time resampling retain the event, the ordinary earlier-pixel control moves it as expected, and the redraw's reopening is inside the edit, within one native picture of the original and within 80 ms of the independent acoustic burst. The earlier-sound neural delivery remains a separate specificity challenge. These observations establish a local timing relationship. They do not measure phone identity, full-clip synchronization, perceived realism or detection accuracy.

## Results

The muted pass found the source-time event in all **6 originals**, all **6 same-time resamples**, and **3 of 6 source-time neural redraws**. The ordinary delayed-pixel check confirmed a five-picture delay within tolerance in **4 of 6**. The intersection of the redraw and control requirements yielded **one newly qualified independent speaker group**. Two additional source-time redraws retained the visible event but did not pass the matched delay control. In the earlier-sound neural condition, two clips showed a delayed visual event within one picture of the intended five-picture delay. One was the fully qualified case. [Sanitized results and replay](evaluation-panels/lombard-fresh-musetalk-2026-11-13.json)

| Qualified case evidence | Exact source interval, seconds |
| --- | --- |
| Authored 16-picture change | 1.038062–1.702422 |
| Acoustic burst in the unchanged soundtrack | 1.285000–1.300000 |
| Original and ordinary same-time reopening | 1.287197–1.328720 |
| Source-time neural reopening | 1.328720–1.370242 |
| Earlier-pixel ordinary reopening | 1.494810–1.536332 |
| Earlier-sound neural reopening | 1.536332–1.577855 |

All **30 deliveries** passed native verification. It checked **1,885 pictures**, **1,232,800 unchanged float32 audio samples**, and **384 committed changed pictures**. In the 96 matched neural frame pairs, the source image inputs, sampling seed and generated latent before conditioning agreed exactly. Each pair produced different neural RGB output when given the different speech window. The standard-library replay recomputes the local timing qualifications. The initial sandbox could not initialize Apple MPS and rendered zero files. The separate local retry used the same inputs and rendered every scheduled file.

The qualified clip supplies a fresh, exact second-generator case for a future blinded detector comparison. It comes from a source group absent from earlier neural work. One new case is insufficient to estimate detector accuracy across generators. This study changes no serving method, checkout path, or confidence.

## Replay

Use the retained source bank, source-timing study, ordinary visual challenge, previous neural bank and restored reviewed runtime. The local geometry task and GPU renderer need access to the Mac graphics context. The source media, detailed masks, render receipts and individual geometry remain private.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze render observe audit; do
  backend/.venv/bin/python -B -m scripts.lombard_fresh_musetalk "$phase" "$run" "$sources" "$source_timing" "$ordinary" "$previous_neural" "$runtime" || exit 1
done
python3 -B -IS backend/scripts/review_lombard_fresh_musetalk.py "$receipt" --assemble "$run"
```
