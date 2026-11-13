# Short voice edits with known foreground isolation

A controlled diagnostic separates the detector's localization limit from the
quality of blind speech separation. It compares the original mixture, both
anonymous blind separator outputs, a benchmark-only ideal mask, and the
exact foreground stem. All views keep the original six-second sample clock.
A source can sound cleaner while yielding a less useful detector response.
The exact foreground is a diagnostic for this detector and benchmark, not
an upper bound on all detectors or a method for analyzing an ordinary upload.

## Frozen comparison

The development controls come from the previous context and variable-boundary
studies. Forty deliveries include 12 generated splices and 28 ordinary
controls, with 24 matched genuine splices and four isolated backgrounds.
The 52 held deliveries contain 12 generated splices spanning Chatterbox,
ElevenLabs and Kokoro, matched genuine same-reader and other-reader
splices, untouched hosts and isolated room controls. The held rooms and
recorded speaker and donor identities are disjoint from development. The
exact files had project exposure in earlier experiments. No held result
set an operating point. The native freeze verifies the published roster,
original and stem hashes, that foreground and background sum exactly to
each submitted mixture, and the earlier two-second measurement hashes.
[Development context sources](audio-remote-neighbor-transfer-2026-11-12.md) ·
[Variable-boundary sources](audio-variable-boundary-transfer-2026-11-12.md) ·
[Held recordings, labels and terms](audio-remote-neighbor-replication-2026-11-12.md)

- **Existing geometry.** Two native two-second windows start at one and
  three seconds. The original stock rule and its previous point are
  reproduced on held PCM. A development-matched point is also measured.
- **Dense geometry and readout.** A one-second interval starts every half
  second. Read the existing affine head on contextual frames from unchanged
  four-second inputs at zero and two seconds. Take the median in each one-second
  interval and the lower median when both contexts cover it. Whole-clip
  speech admission is unchanged across views. Context extends beyond the
  displayed range. This arm changes both geometry and readout.
- **Blind separation.** The pinned two-output separator processes two
  overlapping four-second views of the submission. Match output ordering
  and polarity on their shared two seconds by absolute centered correlation
  and crossfade when both correlations reach 0.25. Retain streams with at
  least 8% of submitted RMS. Keep the largest admitted response at each
  exact range, with no known speaker or edit information.
- **Isolated controls.** The ideal-ratio mask uses both known stems' squared
  STFT magnitudes on the submitted mixture phase. The exact-foreground view
  uses the verified pre-mix samples and preserves ordinary splices. On an
  isolated room input its foreground is deliberately silent. These two
  controls describe behavior with information the app will not have.

The experiment uses the existing MIT-declared voice checkpoint and speech
activity check. The separate blind separator retains conflicting publisher
statements of CC BY-SA 3.0 and 4.0. The evaluation uses the pinned local
checkpoint without redistributing weights or separation outputs. An app
release would require its own asset terms and transfer review. The study
reimplements the ideal mask and uses no model weights from the research
papers below. [Voice asset audit](commercial-model-audit.md) ·
[Separator audit](audio-separation-v14-preparation.md) ·
[Time-domain separation](https://arxiv.org/abs/1809.07454) ·
[Matched component study](https://arxiv.org/abs/2607.10345) ·
[Source-resolved temporal benchmarking](https://arxiv.org/abs/2609.10051)

Every candidate point is the maximum assessed genuine development response
for its own view and grid. Held flags must strictly exceed it. Useful means
at least 50% of the known edit covered and 0.30 temporal IoU. We report
geometric reachability, reachability after speech admission, the strongest
range before applying the point, fixed-point performance, and retrospective
matched false-group counts. The blind follow-up criterion requires at
least two new useful edits across two generator families, no lost prior
useful edit, at most one additional ordinary group and no isolated-room
flag. The diagnostic cannot promote either isolated control to serving.

## Held detection results

Each cell shows useful locations out of 12 known edits and ordinary source
groups flagged out of 16. The separately fitted points are fixed throughout
the assessment. The native stock comparison reproduced the earlier result
against the original fixed point, with three synthetic flags, one useful
location and two misplaced flags.

| Analysis input | Stock windows, useful · false groups | Dense contextual windows, useful · false groups |
| --- | ---: | ---: |
| Unchanged mix, earlier stock point | 1/12 · 3/16 | — |
| Unchanged mix, new development point | 0/12 · 1/16 | 0/12 · 0/16 |
| Blind separated streams | 0/12 · 1/16 | 0/12 · 1/16 |
| Ideal mixture-phase mask | 0/12 · 0/16 | 0/12 · 0/16 |
| Exact pre-mix foreground | 5/12 · 0/16 | 4/12 · 0/16 |

The exact foreground retained the incumbent's one useful edit with either
grid. With the stock grid it added four useful locations, and results
covered all three generator families. Stock geometry could reach eight of
12 edits. The dense grid admitted at least one useful interval for all 12
edits in each of the four views. Before applying the fixed points, its
highest interval was useful for 3/12 original mixes, 5/12 blind views,
11/12 ideal-mask views and 12/12 exact foregrounds. All synthetic inputs
were assessed. The four room-only cases had no foreground and remained
unassessed in both isolated controls. Neither blind arm flagged a room-only
case. [Frozen classifier responses](evaluation-panels/audio-source-isolation-diagnostic-2026-11-13.json)

At a retrospective budget of two ordinary groups, the dense blind view
located three edits and the mixture located one. At zero ordinary flags,
the dense ideal-mask view located one, and the exact foreground located
eight. These comparisons use held controls and establish no fixed point for
use. The acoustic improvement from blind separation and from the ideal
mask did not yield useful detections at any frozen point. The study supports
testing artifact-preserving separation or training on matched noisy speech.
It provides no basis for changing the current detector or customer confidence.

## Foreground recovery

A separate readout measures scale-invariant signal-to-distortion ratio
(SI-SDR) against the exact foreground at its original clock, without a
timing search. A recovery reaches at least 0 dB and improves the original
mixture by at least 3 dB. The blind comparison chooses the better anonymous
output with hindsight. That comparison measures availability of the
foreground in one output and says nothing about selecting it for a customer.

| Known foreground range | Blind recovery / measured | Blind median gain | Ideal recovery / measured | Ideal median gain |
| --- | ---: | ---: | ---: | ---: |
| All non-silent six-second foregrounds | 37/48 | +9.09 dB | 48/48 | +15.36 dB |
| Entire edited foreground recording | 8/12 | +8.83 dB | 12/12 | +15.24 dB |
| Exact edit interval | 8/12 | +8.15 dB | 12/12 | +14.27 dB |

The four isolated-room controls have no foreground to recover. The exact
edit check uses all 12 known intervals without conditioning on any classifier
response. A separate native run reproduced the acoustic receipt byte for
byte, and the readout also agreed on Python 3.11 and 3.12.
[Acoustic readout](evaluation-panels/audio-source-isolation-acoustics-2026-11-13.json)

## Replay

Use the exact retained source folders from the linked studies. The runner
checks their local PCM against the previously published commitments. It
checks the checkpoint and helper hashes and records the installed runtime.
Development scores and points are sealed before held inference. Only scalar
scores, source-relative intervals and hashes leave the local source folders.
The local CPU run made 651 development forwards and 803 held forwards.
The two stock window responses matched the previously archived native
scores at the frozen numeric tolerance. The standard-library readout matched
on Python 3.11 and 3.12. A separate recount verified all eight operating
points, false-group counts, useful ranges and the acoustic/classifier roster
correspondence.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development assessment; do
  backend/.venv/bin/python -B -m scripts.audio_source_isolation_diagnostic "$phase" "$run" || exit 1
done
python3 -B -IS backend/scripts/audio_source_isolation_diagnostic.py review \
  docs/evaluation-panels/audio-source-isolation-diagnostic-2026-11-13.json
```

To repeat only the acoustic assessment after freezing, run the first command
with the backend runtime. The second recounts results with standard-library
Python and does not load model files or require the original samples.

```sh
backend/.venv/bin/python -B -m scripts.audio_source_isolation_acoustics score \
  "$run/acoustics.json" --freeze "$run/freeze.json"
PYTHONPATH=backend python3 -B -S -m scripts.audio_source_isolation_acoustics review \
  docs/evaluation-panels/audio-source-isolation-acoustics-2026-11-13.json
```
