# Can reducing part of a noisy speech window locate an edit?

Adding a local attenuation check recovered two useful locations across
two previously recorded held panels. It also added ten or eleven
ordinary flags depending on the rule. The candidate stays outside
customer analyses. The new report improvement links existing short
speech findings to overlapping observations in the mixed soundtrack.
[Frozen comparison](evaluation-panels/audio-voice-attenuation-plan-2026-11-13.json)

## Method

Use the existing five two-second voice windows and keep the original
speech admission and operating point. Take the strongest two admitted
windows. For each one, lower the first and second halves separately to
a quarter of their original amplitude with 40 ms cosine transitions.
Test the drop in the voice response after reducing each half. A second
rule also requires the drop to be larger than in the opposite half.
The one-second range comes from the original clock. Lowering the mix in
one interval can change the contribution of any speaker or other sound
inside it. It cannot establish the cause of a result or who produced it.
[Time-domain attribution research](https://arxiv.org/abs/2501.13887) ·
[Speech-model asset terms](commercial-model-audit.md)

The fitting set uses 40 archived stock-voice deliveries at −5 dB
and four isolated backgrounds. Each experimental statistic uses
the second-highest ordinary group maximum among cases the existing
method did not flag, with an isolated-background ceiling. The two
points were set before the candidate ran on the two held banks.
A useful range covers at least half the known edit and reaches 0.30
temporal intersection-over-union. The report also counts wrong ranges,
ordinary source groups, and isolated backgrounds.
[Development sources](audio-stock-voice-transfer-2026-11-13.md) ·
[Random-placement sources](audio-random-clock-stock-2026-11-13.md) ·
[Three-generator sources](audio-edge-window-grouped-2026-11-12.md)

## Held results

| Panel and check | Useful edits | Wrong edit windows | Ordinary files flagged | Ordinary groups flagged | Background-only flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random placements at −5 dB, existing | 1/16 | 1 | 3/24 | 2/8 | 0/3 |
| Existing plus response drop | 2/16 | 3 | 6/24 | 3/8 | 0/3 |
| Existing plus comparison between halves | 2/16 | 4 | 6/24 | 3/8 | 1/3 |
| Three generators at −5 or −8 dB, existing | 2/12 | 0 | 0/36 | 0/12 | 0/2 |
| Existing plus response drop | 3/12 | 3 | 7/36 | 4/12 | 0/2 |
| Existing plus comparison between halves | 3/12 | 3 | 8/36 | 4/12 | 0/2 |

The additional ranges were **4.000–5.000 s** in both panels. One
was a Kitten edit and the other used a Kokoro donor. In the
random-placement panel, three controls and both synthetic deliveries
from another host all selected **3.000–4.000 s**, before the planted
edit. All five deliveries share the same competing recording. The
matched errors show that changing the level in this interval can
draw the model to common content outside the edit. Every file
had an available measurement. The banks had earlier project exposure,
and this is a new comparison on those fixed recordings. Neither rule
qualified for a product trial.
[Scalar results](evaluation-panels/audio-voice-attenuation-2026-11-13.json)

## Evidence in the app

When a flagged short speech window overlaps a measured acoustic
observation, the browser and newly generated PDF link to the matching
observation and its original waveform or frequency plot. The two
intervals can also be checked against the same sample clock in the
JSON. The link carries no confidence weight. It gives users a way to
listen for another sound or a recording change at that time.

Four local analyses ran after simulated payment, with zero unpaid jobs.
A fixture with an authored level drop linked the **3.000–5.000 s**
speech window to a **2.500–3.500 s** recording observation. The PDF
link reached that observation, and the source ticks matched the API
and JSON. The two flagged controls without overlapping observations
had no context link. All four signed packages verified against the
actual submitted recordings.
[Paid-worker verification](evaluation-panels/audio-short-recording-links-app-2026-11-13.json)

## Replay

The public receipt contains source hashes, scalar scores and sample
coordinates. It contains no audio, voice embeddings, names of speakers,
or transcripts. The standard-library reader reconstructs decisions,
misses and false alarms. It does not rerun neural inference.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/audio-voice-attenuation-2026-11-13.json
python3 -B -S -m scripts.audio_voice_attenuation_transfer review "$receipt"
```

Use the retained earlier stock source bank, random-placement bank, and
local independent-evaluation directory for the three native roles.
For each role, `audit-inputs` rechecks source hashes and reconstructs
the attenuated windows while preserving the untouched half exactly.
The local replay verified 137 original waveforms and 548 processed
model inputs across all three roles. To rerun inference with the exact
installed, pinned audio assets, use `measure` instead of `audit-inputs`
and provide a new JSONL output path in place of the receipt.

```sh
backend/.venv/bin/python -B -m scripts.audio_voice_attenuation_transfer \
  audit-inputs "$receipt" --role "$role" --folder "$source_bank"
```
