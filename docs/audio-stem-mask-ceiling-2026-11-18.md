# How much can selective masking help under competing speech?

We replayed 75 known edits from the 11 matched source groups in the local
Opus study. The published recurrence filter lowered the median
foreground-to-interference ratio by 0.25 dB. The new test measures how much
of that outcome comes from the filter's choice of frequencies and how much
room there is to improve with reliable source information.
[Parent audio study](audio-recurrent-spectrum-opus-2026-11-18.md)

The replay verified the encoded file, presented sample clock, decoded PCM
and filtered waveform for every submitted clip. Each benchmark pair has the
exact isolated interference used to construct the upload. Subtracting that
interference from the lossless source mixture recovers the foreground to
within source quantization. The original recurrence mask is estimated from
the decoded Opus mixture. A separate soft ratio mask receives both known
components and keeps the same 45% spectral amplitude floor and time
smoothing. We apply each mask to the known components and measure their
power within each labeled edit. [Ideal-ratio mask research](https://pmc.ncbi.nlm.nih.gov/articles/PMC4293540/)

For a fair comparison, attenuate each mask's effect until it loses at most
1 dB of foreground energy in that labeled interval. The chosen strengths
use the benchmark's isolated tracks and exact edit boundaries. This is an
acoustic reference for the headroom available with exact source information.

| Measurement | All 75 edit intervals | 57 intervals at or below −6 dB input ratio |
| --- | ---: | ---: |
| Median input foreground-to-interference ratio | −10.33 dB | −11.60 dB |
| Median ratio change, published recurrence mask | −0.25 dB | −0.65 dB |
| Median ratio change, recurrence with the 1 dB foreground cap | −0.18 dB | −0.33 dB |
| Median ratio change, exact-stem mask with the same cap | +4.29 dB | +3.65 dB |
| Median resulting ratio, exact-stem mask with the same cap | −6.24 dB | −7.53 dB |

The capped recurrence mask improved 25 of 75 intervals and two of 11
source-group medians. The exact-stem mask improved all 75 intervals and
all 11 source-group medians. Its gains also held in the 49 intervals where
both sources had measurable energy for at least three-quarters of the
interval, with a median improvement of 4.75 dB. This activity measure uses
20 ms component power against a separate level estimated within each clip.

Above −6 dB, the uncapped recurrence mask itself improved the median ratio
by 1.03 dB. Under heavier interference, simply limiting that mask's strength
still reduced the median ratio. Competing speech remained stronger across
all 57 severe intervals even with the exact-stem mask. Across the full set,
only 14 intervals had more total foreground than interference after that
mask. Selective source estimation has measurable acoustic headroom, with
substantial interference remaining under the fixed foreground cap. The
test uses previously evaluated audio from six generator families and
measures component energy without running an authenticity detector.

## Verify the comparison

The published capsule contains case slots, exact interval positions,
component power differences, and source commitments. A standard-library
reader binds the measurements to the parent study and recomputes the
per-interval and group summaries. The native rerun is recorded by digest.
[Numeric diagnostic receipt](evaluation-panels/audio-stem-mask-ceiling-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_stem_mask_ceiling \
  docs/evaluation-panels/audio-stem-mask-ceiling-2026-11-18.json.gz \
  docs/evaluation-panels/audio-recurrent-spectrum-opus-2026-11-18.json.gz
```
