# Find a briefly held mouth region in a single video

The mouth-region check measures a textured patch that stays nearly unchanged across consecutive pictures while surrounding face areas change in brightness. It reports the source range and the original pictures and mouth crops from before, during and after it. Ordinary editing and compression can produce the pattern. This finding carries zero weight in the AI confidence estimate. The check requires no earlier copy, soundtrack or manual mark.

## Fixed test

The fixed delivery is an upright H.264 4:2:0 clip with 50 verified pictures at exactly 25 fps in 360 × 288 or 480 × 360. For the held comparison, GRID provides six source groups and CREMA-D provides eight. Two separate GRID groups and four separate CREMA-D actors supplied development views. The same recordings appeared in prior project work. The study authors new matched picture edits on the exact retained native sources. The GRID frames retain their source spacing. CREMA-D pictures map to each output tick by nearest original presentation time, using the earlier picture on ties. Three CREMA-D clips lacked original coverage through the last required picture and were excluded before scoring. The published license terms cover the source recordings and the existing face geometry asset used in the experiment. [GRID recordings](https://zenodo.org/records/3625687) · [CREMA-D recordings](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Face geometry asset terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each group supplies two mouth holds of eight or ten pictures, authored from one earlier picture with a five-pixel feather around a fixed source rectangle. The author verifies changed native RGB pixels on every edited picture. Matched controls are the genuine original, a tandem H.264 re-encode, a full-picture hold, a cut to another genuine recording, a local mouth blur and a held patch above the mouth. The check sees only the submitted delivery. It uses fixed native-coordinate rectangles around the observed mouth and measures consecutive brightness differences, texture and nearby change on either side of the range. Three exact native frames provide the context.

An initial development arm measured an immediate ring outside each candidate box and localized two of twelve edits. The ring sometimes sat inside the larger held region. A second arm moved the context to several distinct face regions and required change in at least two on most transitions. It located ten development holds and flagged one tandem encode. The final method, source roster and acceptance criteria were sealed before held scoring. A useful result needs at least 0.50 temporal intersection over union and at least 75% of its selected rectangle inside the authored fixed core. The release gate requires eight useful GRID locations across four groups and ten CREMA-D locations across five groups, 80% coverage in each bank, at most one control alert per bank, and zero alerts on cuts or full-picture holds.

## Held outcomes

| Outcome | GRID | CREMA-D |
| --- | ---: | ---: |
| Local mouth holds correctly localized | 11/12 | 15/16 |
| Positive cases assessed | 12/12 | 16/16 |
| Alerts on assessed controls | 0/36 | 0/45 |
| Controls left unassessed | 0/36 | 3/48 |

Twenty-four of the 26 localized holds matched both authored endpoints. The lowest temporal overlap was 0.75. Both misses were ten-picture holds. The three unassessed CREMA-D controls were genuine whole-picture cuts where the check could not establish one consistently visible mouth. Independent replay reproduced 112 decisions, 5,600 exact native pictures, 78 source crop commitments, and 206 measured brightness transitions. [Held case ledger and independent replay](evaluation-panels/video-mouth-local-hold-2026-11-12.json)

The method covers fixed-coordinate holds in short, single-face clips. It may miss redraws that preserve motion, regions that move with the head, and timing changes in speech. Longer clips, common higher-resolution deliveries and independently rendered neural mouth edits remain outside this measured scope. The tandem alert in development demonstrates why compression remains an alternative explanation.

## Local app and replay

In the local Full worker, the held GRID example selected **0.520–0.840 s**, with the exclusive end at the first following picture. The before, during and after evidence came from frames **12, 17 and 21**. The browser, JSON and Standard PDF showed the same three original pictures and three exact mouth crops. Browser playback navigated to the recorded frame. The signed-package verifier read all 50 native source pictures and clocks and recomputed the selected region's pixel measurements against the original. An unchanged recording returned no match. Both analyses used simulated checkout and the unpaid events queued zero jobs.

Use a new local run directory and the retained source folders described by the earlier [GRID intake](grid-fresh-release-qualification-2026-11-12.md) and [CREMA-D intake](video-lipsync-local-delay-transfer-2026-11-11.md). The evidence reviewer also supports portable outcome arithmetic without the original media. Source recordings and exact crops stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_local_hold_panel "$phase" "$run" --workers 2 || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_local_hold_panel "$run/receipt.json" --run "$run" --workers 2
python3 -B -S -m scripts.review_video_mouth_local_hold_panel docs/evaluation-panels/video-mouth-local-hold-2026-11-12.json
```
