# Matched river-excerpt crossover preparation

The consumed v208 comparison assigned the genuine and synthetic Turkish
foregrounds to different portions of the river recording. This data-only
preparation pairs each genuine and synthetic foreground with both excerpts
before any new scoring. It reuses the fixed v204 roster and publisher river
tape. [Earlier result](audio-demand-transfer-v208-result.md)

Pair genuine roster position `i` with synthetic position `i`, for all 24
positions in the original v204 order. Pairing uses neither an identity selected
by a score nor a model response. If either foreground was unavailable in v204,
both members of that planned pair remain unavailable in all three conditions.
There are five such pairs in the recorded preparation; preserve all 24 pairs
and 144 planned cases in denominators.

For each complete pair, take both foregrounds from sample zero through the
minimum of their original sample counts. This common length is 2.5–4 seconds;
the two sources and their different production histories are retained. For
position `i`, select river channel 1 at the exact two original v204 offsets:

- **Early:** `4 + 6i` seconds, the window allocated to the genuine slot.
- **Late:** `4 + 6(i + 24)` seconds, the window allocated to the synthetic slot.

Both classes receive the exact same source-noise excerpt at each offset and
common length. This crosses each pair over both original offset strata. Each
foreground has its own cropped, gain-matched clean control and 0 dB mixture
for both excerpts. A single attenuation is chosen per foreground across its
two excerpts using the existing v204 DSP; the clean control is identical for
the two noise conditions. The shared source excerpt is rescaled independently
for each foreground to achieve whole-prefix 0 dB SNR, so the quantized noise
residual can differ between classes. Verify the shared raw excerpt hash,
source prefixes, quantized residuals and measured SNR for every condition.

The river recording is from DEMAND 1.0 by Joachim Thiemann, Nobutaka Ito and
Emmanuel Vincent, [published under CC BY-SA 3.0](https://zenodo.org/records/1227121).
The script verifies the pinned v204 preparation, publisher archive SHA-256 and
MD5, and production PCM before building. It makes no neural calls. All source
audio, per-case references, cropped waveforms and hashes remain private. The
existing data-only v108 resource guard retains failed attempts; readout replays
the construction and reports aggregate coverage only.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_v210 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_v210 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_v210 readout
```

A later separately frozen scoring run can compare each noisy result with its
own cropped matched-clean result and examine class-by-window transitions.
Cropping can affect scores, so each comparison uses the new matched-clean
control. This diagnostic covers one previously tested river tape and one
synthetic source collection. The two source classes have different recording
and generation histories.
