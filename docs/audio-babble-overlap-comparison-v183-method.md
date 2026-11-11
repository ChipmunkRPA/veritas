# Competing-speech abstention on matched genuine and synthetic foregrounds

Use the 96 frozen development cases from
[v181](audio-matched-babble-v181.md). Six cases per pair share the
same genuine or synthetic foreground, gain recipe, source clock and
assigned babble pool. Within a pair and noise level, the two foregrounds
use the same quantized babble residual. Their measured foreground SNR can
differ. The four connected components share foreground ancestry and a
babble pool within each component. Treat the results as descriptive
development evidence.

Verify the v181 freeze and completed data-only v108 guard against the
published aggregate, private case journal and source receipt. Verify the
48,000-sample production waveform for each constructed case and the
sample-wise residual wherever the matched clean waveform is available.
Keep unavailable and failed cases in the planned denominators.

Score each constructed waveform through the pinned
[v174 voice route](audio-matched-edit-diagnostic-v174-method.md) and the
verified [v180 overlap detector](audio-overlap-context-v180-method.md).
The voice route uses its existing speech gate, calibration and lazy
original-side stability check. The overlap detector uses a CPU session.
It excludes padded receptive fields and marks at least 250 ms of
continuous source-supported competing speech. Its intervals describe
acoustic context and supply no judgment about speech origin.

The research comparator changes a baseline AI tendency to inconclusive
when the overlap marker is present. Use this condition-blind rule for
clean, nominal 0 dB and nominal +5 dB waveforms. Preserve the baseline
direction for every case and retain all clean outcomes in both policies.
No new probability, fitting or threshold enters the comparison.

Report four directions, assessed coverage, false AI flags on genuine
foregrounds, missed AI and wrong-original decisions on synthetic
foregrounds, and AI-to-inconclusive counts. Show the paired synthetic-only,
genuine-only, both and neither AI flag counts and the median raw-score
difference for pairs where both arms were scored. Report clean-to-noisy
direction transitions by arm and keep overlap-only evidence counts
separate. Show the same descriptive results for each of the four
components without revealing their source identifiers. No change to the
serving route is authorized by this development panel.

Use the already verified isolated v180 runtime root.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_babble_overlap_comparison_v183 freeze /path/to/v180-runtime
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_babble_overlap_comparison_v183 run /path/to/v180-runtime
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_babble_overlap_comparison_v183 readout /path/to/v180-runtime
```

The worker runs offline under the v108 owned-process guard. It does not
open holdout metadata or media. The public readout contains counts and
commitments. Waveform evidence and individual scores stay private.
