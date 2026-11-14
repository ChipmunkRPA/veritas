# Sound during a measured mouth-region hold

Full video analysis now checks whether the soundtrack contains sustained voice-like periodicity during an independently measured local mouth-region hold. The report shows the half-open source interval, three source frames and exact mouth crops, the waveform, and the number of 40 ms audio blocks with the acoustic pattern. The browser and evidence ZIP also include the exact decoded mono WAV excerpt. The observation carries zero weight in the overall AI confidence assessment. Ordinary dubbing, another speaker, music or other periodic sounds can also produce the pattern.

The check starts only after the qualified HD mouth-region detector has measured unusually stable pixels in the mouth region while nearby face regions change. It uses whole decoded 16 kHz samples strictly inside the exact video boundaries, with a continuous sound clock. It evaluates mean-centered, amplitude-gated autocorrelation in 40 ms blocks spaced 20 ms apart. At least six blocks are required. It reports sustained periodicity when at least two-thirds show a repeat consistent with vocal pitch. It uses no learned model or voice identity. A usable soundtrack can have no sustained periodicity, and a track that is entirely silent or otherwise unavailable shows that the soundtrack was unmeasured. [HD picture methodology and source rights](video-mouth-hd-hold-2026-11-12.md)

Recent research explicitly separates audio–picture offset from whether the mouth visibly articulates particular sounds. This check measures a narrow acoustic pattern alongside a measured hold. Broader experiments using a closed-mouth geometry estimate and general speech activity warned on unchanged scripted clips and ordinary global offsets during development. Those experiments stay outside the app. [Timing and articulation](https://arxiv.org/abs/2610.09223) · [Phonetic alignment for dubbing](https://arxiv.org/abs/2610.00825)

## Local comparison

The diagnostic reused nine held, rights-reviewed Wikitongues recordings from the earlier HD mouth-hold bank. Seven supported the paired comparison using the nominal 400 ms held-picture interval. Nine supplied natural 200 ms quiet intervals. Audio donors were selected for clear periodicity by searching within the same test partition across other recordings, so the positive results show measurement feasibility. The experiment supplies no independent estimate of general dubbing or deepfake accuracy. The noise variants add deterministic band-limited random sound at roughly −28 dBFS RMS. Recordings in the development and held groups are identified only by anonymous indices in the receipt. [Source and license ledger](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)

| Condition | Sustained periodicity |
| --- | ---: |
| Original soundtrack in the 400 ms tested slot | 2/7 |
| Clean speech inserted in the 400 ms slot | 7/7 |
| Same speech inserts with strong added noise | 0/7 |
| Brief natural quiet interval | 0/9 |
| Natural quiet interval with strong added noise | 0/9 |

An isolated local checkout compared two videos with identical pictures and audio outside **3.300–3.700 s**. The quiet version had speech elsewhere in the clip and measured **0/19** periodic blocks in the held interval. The inserted-speech version measured **19/19**. Chrome displayed the waveform, playable audio, and three matching source images and crops. The PDF and JSON retained the same interval and measurement. Signed replay verified both the WAV and waveform against the original video. [Local comparison counts](evaluation-panels/video-mouth-hold-sound-2026-11-13.json)

A second checkout shifted the entire audio stream origin by **500 ms** while preserving video timestamps. The same **3.300–3.700 s** source interval selected decoded audio samples **44,800–51,200**. The waveform showed the original source clock, the JSON kept both clocks, and the PDF, Chrome waveform and signed replay verified the resulting excerpt. Older signed packages still replay their original waveform bytes.
