# Paired sound transients around short speech edits

Two single-upload checks looked for short bursts of energy at both ends of a
speech edit. One looked at the lowest and highest measured frequency bands.
The other required agreement across three bands. Neither located a known edit
under strong rain, train noise, or competing speech at its frozen operating
point. The checks remain offline.

## Measurement

The test uses six-second recordings at 16 kHz. A 40 ms Hann window advances
every 10 ms. Each band's level is compared with the louder of its two nearby
80th-percentile levels, with a 30 ms guard and a floor of −84 dBFS. The first
arm takes the larger excess from below 90 Hz or in the 6.8–7.9 kHz band.
The broadband arm takes the smallest excess across 2.2–3.6, 3.6–5.5 and
5.5–7.9 kHz. A candidate interval lasts 160–1,200 ms. The weaker endpoint
sets its score. Each check keeps its highest scoring pair.

Studies of splice artifacts show why this is an editing clue and why benign
speech cuts need their own controls. This implementation uses no paper code or
new weights. [Splice artifacts and dynamic range](https://arxiv.org/html/2408.13784v1)
· [Matched genuine and synthetic edits](https://arxiv.org/html/2610.04479v1)

## Sources and results

Development used 288 deliveries from 20 readers. The point for each check was
the maximum response on 192 untouched, gain-adjusted, globally filtered, or
same-word resynthesized development deliveries. The rule requires strict
excess above that point. The held panel used 15 different readers over 316
deliveries, including 16 background-only controls. Four deliveries of each
source represent clean audio, rain, train, and competing speech. Noisy mixtures
have background RMS 5 dB above the untouched speech RMS. Held original,
publisher-edited and isolated deliveries are checked against the exact decoded
sample hashes of the previous long-review test. The readers and backgrounds
have earlier project exposure. This test uses a single publisher's splicing
procedure and does not establish performance on seamless edits.
[SINE resynthesis and splice source bank](audio-sine-resynthesis-transfer-2026-11-14.md)
· [Held sample selection](audio-narrow-flag-transfer-2026-11-14.md#new-readers-from-the-publisher-archive)
· [Background pins and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

Each held edit has an untouched version, a same-clock ordinary cut with a
20 ms fade at both ends, a short level reduction, and a global low-pass
control. A useful range covers at least half the annotation with temporal
intersection over union of at least 0.30. A second metric requires each end
to be within 40 ms of its annotated position.

| Held outcome | Existing long voice review | Lowest or highest band | Three-band agreement |
| --- | ---: | ---: | ---: |
| Correct location on clean splices | 0/15 | 0/15 | 1/15 |
| Correct location on noisy splices | 0/45 | 0/45 | 0/45 |
| Both endpoints within 40 ms | 0/60 | 0/60 | 0/60 |
| Ordinary no-splice deliveries flagged | 0/60 untouched only | 4/180 | 0/180 |
| Same-clock ordinary cuts flagged | — | 4/60 | 0/60 |
| Background-only controls flagged | 0/16 | 0/16 | 0/16 |

The existing long voice review was measured on publisher edits, untouched
speech and isolated backgrounds. The stricter endpoint metric has zero useful
locations for the incumbent because it found no useful range. The lowest or
highest band arm flagged two distinct genuine readers across four untouched
or gain-adjusted deliveries. Three-band agreement located one clean edit by
overlap, and its two ends missed the tighter metric. No strongest candidate
on a held publisher splice met the 40 ms endpoint metric even before applying
either operating point. Neither arm met the frozen advancement criterion of
three noisy locations across at least two noise conditions. The measurements
support keeping this cue out of the product and out of the AI estimate.

## Replay

The receipt stores numeric measures, intervals and input commitments. The
standard-library reviewer recomputes both development points and every count.
Native replay uses the retained SINE development and held banks documented
above, plus the two pinned noise files. No source audio is included here.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/audio-transient-edges-2026-11-14.json
python3 -B -S -m scripts.audio_transient_edges_transfer review "$receipt"
python3 -B -S -m scripts.audio_transient_edges_transfer freeze "$fresh_freeze"
backend/.venv/bin/python -B -m scripts.audio_transient_edges_transfer run "$fresh_receipt" \
  --freeze "$fresh_freeze" --development "$sine_development" --held "$sine_held" --sounds "$sounds"
cmp "$receipt" "$fresh_receipt"
```
