# Background-reduced speech review at 48 kHz

The existing background-reduced review can recover AI-like speech responses in
short noisy clips. Extending it to 48 kHz lossless audio retained 18 of its 19
prior useful flags in two re-rendered tests. On native 48 kHz recordings with
loud added traffic, it flagged five of 12 genuine speakers. The extension
remains experimental. [Measurements and source commitments](evaluation-panels/audio-background-lossless-rate-transfer-2026-11-13.json)

## Fixed comparison

The two earlier held panels supply 166 three-second 16 kHz mixtures. The
original panel has 22 source IDs, including six synthetic voice generators.
The confirmation has 28 RADAR source files whose generator identities are
not specified. Each input is rendered at 48 kHz, then stored losslessly as
mono PCM16 WAV and FLAC. A carrier-independent decode verifies that both
represent the same native 48 kHz samples exactly. The app resamples both
carriers to 16 kHz and verifies the complete three-second source clock. Five
pairs had different resampled PCM commitments, so both decoded views were
scored. Standard FLAC files with a duration in the header reproduced all 166
separately scored FLAC analysis streams byte for byte. The prior panels and
their licenses, providers, and source limitations are documented in the
earlier comparison. [Original study and commercial asset terms](audio-nemotron-noise-transfer-2026-11-12.md)

The voice detector keeps its primary decision. The background check runs
only on an inconclusive or speech-unassessed primary result. It uses the
existing pinned enhancer, speech admission and operating point. No point
was refitted. The frozen transfer policy requires preserving at least 80%
of the prior noisy synthetic additions in each panel and permits at most
two added noisy genuine flags. It requires no new gain-only or isolated
flags. Research on codecs and realistic corruptions motivated testing the
delivered media through the existing decoder. This study does not measure
lossy codecs. [Audio detector robustness study](https://arxiv.org/abs/2503.17577)

| Cohort | Background check at 16 kHz | Background check at 48 kHz, each carrier |
| --- | ---: | ---: |
| First panel, noisy synthetic | 8/24 | 8/24 |
| First panel, noisy genuine | 1/20 | 1/20 |
| RADAR confirmation, noisy synthetic | 11/28 | 10/28 |
| RADAR confirmation, noisy genuine | 1/28 | 1/28 |
| Both panels, isolated backgrounds | 0/16 | 0/16 |
| Both panels, gain-only | 0/50 | 1/50 |

The additional gain-only flag was a genuine RADAR clip. Its enhanced
response crossed the unchanged point by 0.00120 logit. One prior noisy
synthetic response moved below the point. Across the 92 RADAR files the
primary changed direction four times. One was a newly correct primary
direction, one removed a genuine false direction, and the two others did
not change which files had an AI direction. All auxiliary decisions agreed
between the matched 48 kHz carrier files.

## Native-rate controls

The 48 kHz comparison also used 12 distinct speakers from the University of
Edinburgh's CC BY 4.0 VCTK corpus. Select source speakers and one microphone
file per speaker by SHA-256 ordering on the publisher's exact ZIP directory,
limited to utterance slots 30–99, FLAC members of 130,000–172,000 bytes and
microphone 1. Exclude four speakers from the preceding local matched VCTK
comparison. The corpus has prior exposure in other project studies. Each clip
contains its center three seconds without source resampling. Each clip and
its matched traffic mix share the same fixed gain. The target is 5 dB below
the added sound by whole-clip RMS. The roadside recording was already used
by the first held comparison. Native PCM, ZIP member CRC and byte ranges,
source dates and source clocks are checked or bound before scoring. These
are studio recordings with a controlled street-noise mix, not recordings
made in moving vehicles or in conversation. [VCTK source](https://doi.org/10.7488/ds/2645)
· [Publisher CC BY 4.0 license](https://datashare.ed.ac.uk/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/download)
· [Vienna roadside recording](https://commons.wikimedia.org/w/index.php?title=File:2023-Mai-20_14h34min_Wien_innerer_Lerchenfelder_G%C3%BCrtel_Thaliagasse_Nachmittagsverkehr.wav&oldid=982248618)

| Native 48 kHz VCTK | Existing primary AI signals | Additional background-reduced flags | Usable enhanced readings |
| --- | ---: | ---: | ---: |
| Clean or gain-only | 0/12 | 0/12 | 1/12 |
| Traffic at −5 dB | 0/12 | 5/12 | 10/12 |

Five ordinary flags exceed the maximum of one allowed for the separate
native-rate validation. Since that panel has genuine speech only and no
paired native-16 kHz run, it establishes a difficult negative condition for
the proposed expansion. It cannot attribute the false alerts to the sample
rate itself or establish sensitivity on native 48 kHz synthetic files. The
strong performance on re-rendered mixtures alone would miss this problem.
The [matched 16 kHz follow-up](audio-background-rate-diagnostic-2026-11-13.md)
reproduces the five flags on the same speakers at the app's 16 kHz rate.

## Local application and replay

In an isolated local development build, two flagged 48 kHz deliveries
passed checkout using a Stripe test double, preserved the primary result,
and reproduced the exact analyzed samples, source seconds, and original
waveform in JSON, PDF and two signed packages. An isolated background was
unassessed. The optional check remains limited to its existing 16 kHz scope
in the app. That distinguishes working delivery and reporting from the
observed detector errors. The test used no live payment.

Run the standard-library reviewer to reproduce arithmetic from the frozen
records. To replay actual native input bytes, download each VCTK archive
range beginning at the `offset` in `native_freeze.rows` and ending at
`offset + size + 255` inclusive. Preserve each response's `Content-Range`
header as `<offset>.h` and the body as `<offset>.range`. Use `native-replay`
with the pinned Vienna recording. No test media or weights are bundled.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/audio-background-lossless-rate-transfer-2026-11-13.json
python3 -B -S -m scripts.audio_background_lossless_rate_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_background_lossless_rate_transfer \
  native-replay "$receipt" --inputs "$vctk_ranges" --background "$vienna_wav"
```
