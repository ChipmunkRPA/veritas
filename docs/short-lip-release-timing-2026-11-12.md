# Unmarked mouth-release timing on short single uploads

This check tested whether a short video contains enough independently observed acoustic and visible events to locate a temporary lip-sync offset. It recovered **0 of 40** known local timing moves at the frozen review point. It did not add a finding to any of the 336 ordinary controls. The prototype remains outside the app. [Outcome ledger and replay](evaluation-panels/short-lip-release-timing-2026-11-12.json)

## Test design

The unmarked detector sees the entire native picture series and a 16 kHz mono waveform. It finds rapid mouth reopenings after near closure and distinct high-frequency acoustic bursts after reduced energy. It matches events in order, comparing a single shared lag with one local offset supported by events on both sides. Missing events and ambiguous assignments reduce support. The detector gets no earlier copy, phoneme labels, transcript, or selected time. Prior work demonstrates that preparation, closure and release have different natural timing, and recent work separates timing and articulation. This trial uses independent event matching with a retained face geometry asset and no newly trained weights. [Natural timing study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003743) · [Timing and articulation research](https://arxiv.org/abs/2610.09223)

Thirty-two previously measured Lombard clips from eight speaker groups established the internal review point. The held roster uses 48 previously unused publisher recordings from 12 separate speaker codes, with two plain and two Lombard recordings per code. The source list was frozen before download. Each official archive member matched the frozen ZIP length and CRC and was bound by SHA-256. Acquisition paused twice at the publisher rate limit and resumed from verified bytes without replacing sources. The test kept all 48 sources for the ordinary controls. A separate waveform and picture qualification used a coarse publisher phone neighborhood only to create ground truth. It found 20 eligible source events across nine held groups. Source membership in the same corpus limits generalization across recording conditions. [Corpus release and CC BY 4.0 terms](https://spandh.dcs.shef.ac.uk/avlombard/) · [Earlier source qualification](lombard-original-mouth-event-2026-11-12.md)

Each qualified source supplied one earlier and one later audio move inside a 640 ms interval. The signed moves were five or seven native picture periods, about **208–293 ms**. The same event waveform was recovered at its authored position and all samples outside the edit matched the source. Controls include unchanged speech, local gain, filtering and background noise, both whole-track offsets, and matched sound and mouth timing moves. The last control uses prepared aperture observations and the corresponding waveform, so it tests the timing algorithm and supplies no re-encoded pixels. This is a native-signal test with conventional timing changes. It measures neither AI generation nor single-upload detection of synchronized redraws.

Standalone FFprobe and FFmpeg independently verified **2,941** native picture timestamps and **1,936,480** audio samples across 44 float32 and four signed-16 originals. The audit also checked 40 source mouth crops and both signed witness intervals for each of the 20 qualified events. The supported clip lengths top out at 3.4 seconds, below the released four-window timing-change requirement. The new check needed three native mouth events and three acoustic events for a measured comparison. A correct proposal needed to identify each true event and the relative signed offset to within 80 ms. The frozen point of 1.9409 is an internal development decision, with a separate 0.12 ambiguity margin.

## Held results

| Outcome | Count |
| --- | ---: |
| Comparisons with enough mouth and acoustic events | 259/376 |
| Known moves with enough events | 25/40 |
| Correct proposals before the score and ambiguity gates | 4/40 |
| Ordinary controls with a proposal before those gates | 116/336 |
| Correct signed moves at the frozen point | 0/40 |
| Ordinary control flags at the frozen point | 0/336 |

The pre-gate useful proposals include three later and one earlier moves. One example pairs the mouth reopening at **1.022 s** with the acoustic onset at **1.293 s**, estimating a **+276 ms** local displacement against **+292 ms** authored. Its comparison score stayed below the ordinary-calibrated point. Raw proposals also occurred on 15 unchanged clips and 34 whole-track offset controls. Sparse, unlabeled events in this panel did not establish a reliable correspondence. The current release still reports insufficient local timing support on comparable short clips.

## Local replay

Use the retained earlier [source bank](lombard-original-mouth-event-2026-11-12.md) as `development` and new private directories `sources` and `run`. The geometry runtime uses the existing Mac graphics context with the CPU Face Landmarker task. Raw source media and detailed frame commitments stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze acquire; do
  backend/.venv/bin/python -B -m scripts.lombard_short_timing_intake "$phase" "$development" "$sources" || exit 1
done
for phase in observe score calibrate; do
  backend/.venv/bin/python -B -m scripts.evaluate_short_lip_release_timing "$phase" "$development" "$run" || exit 1
done
for phase in observe score; do
  backend/.venv/bin/python -B -m scripts.evaluate_short_lip_release_timing "$phase" "$sources" "$run" --split held || exit 1
done
backend/.venv/bin/python -B -m scripts.audit_short_lip_release_timing "$sources" "$run"
python3 -B -S -m scripts.review_short_lip_release_timing "$receipt" --assemble "$run"
python3 -B -S -m scripts.review_short_lip_release_timing docs/evaluation-panels/short-lip-release-timing-2026-11-12.json
```
