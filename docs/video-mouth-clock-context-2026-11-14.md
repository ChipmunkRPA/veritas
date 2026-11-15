# Distinguishing a repeated audio offset from a local change

New Chipmunk Verify video reports link repeated audio offsets to their
supporting windows. The observation requires the same offset on the same
face track in at least two displayed one-second windows with disjoint
source intervals. The existing track measurement must record a constant
offset. Each selected window must reach its recorded correlation minimum
and exceed the strongest alternative at least three ticks away by 0.05.
The displayed offset must be at least three ticks, or 120 ms, from zero.
The browser and Standard PDF link the source frames, mouth crops and audio
evidence. JSON records the linked packet IDs and the signed offset at
`/av/coverage/repeated_offsets`. The observation applies to those windows
and carries zero AI confidence weight. A repeated offset can come from
recording or export timing.

An encoded fixture with a known 200 ms delay measured five 40 ms ticks.
The browser and Standard PDF linked the first and third measured windows.
The JSON verifier rejected a replacement link to the intervening window,
which overlaps the first. The report regression also checked that flagged
local timing changes kept their existing source-context links.

## Experimental correction using other parts of the utterance

The benchmark keeps the six development and ten held Lombard Grid
speakers from the earlier missing-closure study. It reruns acoustic
inference to identify M, B and P and retains the committed native frame
geometries. In addition to earlier cases, the comparison adds four clean
whole-soundtrack shifts at ±80 and ±160 ms and two combinations of a
local dub and an overall shift. The 300 held deliveries reuse these
speakers and source pictures. The newly authored audio deliveries test
timing transfer within this panel. They do not test a new speaker cohort
or generated mouth motion.
[Lombard Grid and CC BY 4.0 terms](https://zenodo.org/records/3736465)
· [Parent closure benchmark](video-mouth-closure-transfer-2026-11-14.md)
· [Pinned model and release measurements](video-bilabial-mouth-transfer-2026-11-14.md)
· [Articulation and timing research](https://arxiv.org/abs/2610.09223)

For each audible M, the trial looks for at least two distinct B or P
events outside the proposed change's immediate neighborhood. It matches
those sounds to independently measured visible lip releases. The closest
plausible timing must be supported by both pairs, and a competing fit
can leave the offset ambiguous. The previous study's fixed thresholds
remain in force. The corrected arm falls back to the nominal timing
when it lacks anchors. A third arm assesses only cases with timing anchors.
The flagged frames and the audio interval retain their separate clocks.

| Held result at the broader point | Nominal 100 ms | Corrected 100 ms | Nominal 160 ms | Corrected 160 ms | Anchors required, 160 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful locations on 20 local dubs | 4 | 4 | 5 | 5 | 0 |
| Useful locations on 40 dubs with an overall shift | 9 | 9 | 9 | 10 | 1 |
| False local flags on 120 ordinary deliveries | 2 | 5 | 5 | 5 | 0 |
| False local flags on 120 constant-shift deliveries | 12 | 15 | 10 | 8 | 0 |

The clock estimate was within 60 ms of the known shift on 29 of the 30
constant-shift clips where the 160 ms corrected arm had enough anchors.
Only 4 of 60 edited deliveries had enough anchors, including the one
additional location. That example measured a 173.5 ms delay on a clip
with an authored 160 ms delay. The flagged sound spans 1.105–1.425 s.
The 0.932–1.252 s picture window contains frames 23–30. The correction preserved all
14 previously useful locations at the broader 160 ms point.
The 100 ms correction increased false alerts in both
control categories. The B/P correction remains experimental, pending
transfer to more speakers and clips with more usable context. The shipped
report observation above uses existing correlation evidence and is separate
from this trial.
[Held measurements](evaluation-panels/video-mouth-clock-context-held-2026-11-14.json)
· [Development measurements](evaluation-panels/video-mouth-clock-context-development-2026-11-14.json)

## Replay

Standard-library Python recounts both panels without models or audio.
Native replay reconstructs 480 waveforms, verifies the shifted samples
in 256 deliveries, and verifies 206 supporting source pictures and mouth
crops. It reuses the committed acoustic detections and frame geometry.
[Development source receipt](evaluation-panels/video-mouth-clock-context-development-native-2026-11-14.json)
· [Held source receipt](evaluation-panels/video-mouth-clock-context-held-native-2026-11-14.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for split in development held; do
  receipt="docs/evaluation-panels/video-mouth-clock-context-$split-2026-11-14.json"
  python3 -B -S -m scripts.video_mouth_clock_context review "$receipt"
  backend/.venv/bin/python -B -m scripts.video_mouth_clock_context replay "$receipt" \
    --sources "$sources" --output "$run/$split-native.json"
  backend/.venv/bin/python -B -m scripts.video_mouth_clock_context "$split" "$run/$split.json" \
    --sources "$sources" --runtime "$runtime" --model "$checkpoint"
done
```

Use the maintained archive and model preparation from the
[parent benchmark](video-mouth-closure-transfer-2026-11-14.md#local-replay).
The measured run used Apple MPS for acoustic inference. The application
regression is in `backend/tests/test_video_av_evidence.py` under
`test_av_windows_share_nonoverlapping_source_context`.
