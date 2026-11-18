# Eye-region holds at 25 and 30 frames per second

The guarded shortlist located **16 of 36** staged eye holds at 25 fps and
**14 of 36** at 30 fps in a separate camera corpus. It raised two alerts among
84 ordinary files at 25 fps, including a whole-picture jump, and zero among
84 at 30 fps. The 30 fps results met all four predeclared checks for that
rate. The combined trial missed its zero whole-picture alert rule, so this
remains an offline comparison.

## Fixed comparison

Twelve CREMA-D actors supplied recordings with 60 or more verified camera
pictures. Three shorter originals were excluded before scoring. These
recordings were used in earlier mouth or timing work. The earlier eye tests
used RTFS-10K. For each actor, the edit intervals varied by a rule fixed
before scoring. A short and a long hold reused an earlier eye patch within
the same recording, with a feathered boundary. The longer edit also had a
twice-encoded delivery. The two frame rates use the same authored source
pictures. The 25 fps delivery selects the nearest real camera picture at
each target tick. The 30 fps delivery uses the first 60 source pictures in
order. Both are upright 480 × 360 H.264 with verified presentation times.
[Pinned CREMA-D media terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

The raw arm returns the first landmark-qualified texture proposal. The
top-only arm checks nearby motion around that proposal. The shortlist arm
applies the same check in rank order across the existing eight proposals.
The texture and nearby-motion points stay fixed. For 30 fps, the method
scales the search window, edge support and landmark sample positions to
the new cadence. A useful location selects the correct eye, overlaps at
least half of the union with the authored time range, and puts at least
75% of the reported crop inside the authored patch.
[Earlier method and comparison](video-eye-shortlist-context-2026-11-18.md)

The source roster, edited roles, ordinary roles and visual review of
blinks were locked before scoring. Controls include unchanged and
twice-encoded footage, eye blur and brightness changes, whole-picture
holds, same-source jumps and cuts to another actor. The gate required
eight useful holds across at least four actors per rate, at most two
ordinary alerts per rate, and zero whole-picture alerts at both rates.
It also allowed up to four individual 25 fps locations to be lost in the
matched 30 fps version.

## Results

| Measure | 25 fps | 30 fps |
| --- | ---: | ---: |
| Useful locations, raw → top-only → shortlist | 23 → 11 → 16 / 36 | 25 → 10 → 14 / 36 |
| Ordinary alerts, raw → top-only → shortlist | 34 → 1 → 2 / 84 | 33 → 0 → 0 / 84 |
| Whole-picture alerts, raw → top-only → shortlist | 10 → 1 → 1 / 36 | 10 → 0 → 0 / 36 |
| Actors with a useful shortlist location | 8 / 12 | 8 / 12 |
| Controls assessed | 80 / 84 | 80 / 84 |
| Verified source frame and crop pairs | 54 | 42 |

The 25 fps shortlist flagged an untouched clip at **0.320–0.640 s** and
an ordinary picture jump at **0.480–0.920 s**. The jump alert was also
present in the top-only arm. At each rate the shortlist assessed all 36
edited files and eight of the 12 cross-actor cuts. It flagged none of
those cuts. The raw arm flagged four of ten assessed original and tandem
deliveries containing independently observed natural blinks at each rate.
Both nearby-motion arms flagged zero of those controls. No shortlist
finding matched both authored endpoints exactly.

Thirteen matched edits had a useful location at both rates, three only
at 25 fps, one only at 30 fps, and 19 at neither rate. The paired loss
check passed. The promising 30 fps result is from one actor bank under
controlled recording conditions. A separate 30 fps confirmation is
needed before using it as an app observation. The test measures an
observable texture pattern on staged edits. AI attribution and
scientific reliability on general uploads are outside its scope.

## Replay

The numeric receipt contains anonymous groups, authored positions,
camera and delivery timestamps, numeric comparisons, and frame and crop
commitments. Source media, publisher identifiers and local paths stay
out of the package. The standard-library reader reconstructs the camera
frame selection, the overlap counts, the visible-blink controls, and all
frozen advancement checks. The native run independently verified the
96 frame-and-crop hashes against decoded pictures.
[Numeric receipt](evaluation-panels/video-eye-cadence-transfer-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_eye_cadence_transfer \
  docs/evaluation-panels/video-eye-cadence-transfer-2026-11-18.json.gz
```
