# Phone identity and acoustic release timing in one video

The blind phoneme decoder recovered **0 of 5** expected `/b,p/` sequences and returned a `/b/` or `/p/` on **3 of 7** phrases with neither sound in the published transcript. The audio checks established two usable signal landmarks on nine publisher-timed word windows. The blind decoder and the prior raw transient each had one nearby event. The check stays experimental. [Aggregate and independent replay](evaluation-panels/video-phone-release-qualification-2026-11-12.json)

## Lexical identity

A native-video candidate can confuse a sound near a moved mouth gesture for the sound that belongs to it. The preceding timing study found this ambiguity on both delayed neural redraws. This follow-up tests whether a local audio-only phoneme decoder can supply a reliable `/b/` or `/p/` identity. It uses no outside reference recording or visual cue for the audio identity result. [Prior single-file timing study](grid-single-clip-release-timing-2026-11-12.md) · [Phoneme and viseme consistency research](https://arxiv.org/abs/2610.09223)

Use the 12 already-qualified CREMA-D confirmation actors, separate from the GRID material and with prior project exposure disclosed. The publisher's fixed phrases give five clips with six expected `/b,p/` events and seven phrases with neither phone. Freeze exact file, decoder, model, and dictionary checksums before scoring. Compare the bundled PocketSphinx all-phone search with its known-transcript alignment on the same decoded waveform. The known-transcript arm sets a ceiling because it receives the expected words. FLV audio is used in decoded sample order for lexical scoring. Packet timestamps have millisecond granularity, so this bank supplies no audio-to-picture release timing truth. [Publisher phrases](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md) · [Data terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Native source qualification](video-native-source-v92-result.md)

| CREMA-D outcome | Blind phoneme decoding | Known transcript |
| --- | ---: | ---: |
| Completed alignment or decode | 12/12 | 6/12 |
| Exact expected sequence on positive phrases | 0/5 | 3/5 |
| Phone reported on a negative phrase | 3/7 | 0/7 |

All three positive phrases completed by the known-transcript arm had the expected sequence. It also completed three of the seven negative phrases. Six clips failed that arm's native state alignment. The retained 5.0.4 wheel and bundled US English acoustic model carry permissive notices. The exact bundled all-phone model is SHA-256 `c57e0fa4191b096b1279cfe3a77927f52568fdecfc6624ddb5cec9527c763a54`. [Code licence](https://raw.githubusercontent.com/cmusphinx/pocketsphinx/v5.0.4/LICENSE) · [Acoustic model notice](https://raw.githubusercontent.com/cmusphinx/pocketsphinx/v5.0.4/model/en-us/en-us/README)

## Native acoustic coordinates

On the 12 held GRID originals, nine published word windows begin with a lexical `/b/` or `/p/`. The six groups are distinct from the two discovery groups in the preceding single-file study. Within each word's onset neighborhood, independently of decoder results, require both a distinct first-difference impulse and a distinct positive spectral flux peak. Both measures must agree within 15 ms. Only **two of nine** windows support a common point, with a fixed 25 ms uncertainty on either side. These are signal landmarks. The publisher supplies word intervals and this test does not independently establish phoneme release times.

Measure whether the earlier raw acoustic detector produced any event within 50 ms, and whether the known-transcript or blind decoder produced the expected phone with a model interval ending within 50 ms. The raw method produced **1/2**, the known-transcript arm **2/2**, and blind decoding **1/2**. Neither complete landmark slice lay inside the exact copied interval of the four previously authored audio-delay pairs. Those pairs verify native clocks and PCM construction. They supply zero eligible shifted landmarks, so no signed shift sensitivity can be estimated from this trial. [Publisher word and PCM qualification](grid-reserve-word-clocks-2026-11-12.md) · [Earlier source-bound shift construction](grid-paired-closure-timing-2026-11-12.md)

The first intake stopped on the requirement for a nonoverlapping sample clock on the CREMA-D FLV audio. The completed intake uses those files for lexical identity and retains the stricter clock requirement for GRID timing. The intake and advancement criteria were sealed before either decoder ran. Separate FFprobe and FFmpeg readback verified **2,132 native picture timestamps**, **1,267,683 decoded samples**, and all four paired media copies. No serving change followed. A larger assessment needs independently marked release locations and explicit examples of expected stops that have no clear burst.

## Reproduce

Use a new local run directory and the retained source and model files. Keep phrase hypotheses, individual timing records, images, and PCM local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in intake score report; do
  backend/.venv/bin/python -B -m scripts.video_phone_release_qualification "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_phone_release_qualification "$run"
```
