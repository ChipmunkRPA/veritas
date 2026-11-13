# Four-second H.264 face-change confirmation

The first four-second H.264 comparison found four additional useful
locations across two source groups, with no new ordinary flags. It
needed gains across at least three groups. This confirmation keeps the
same face proposal and fixed response point, and changes the sources and
the placement of the known brief edits. A separate study qualified the
existing four-second VP9 check. [Previous carrier comparison](video-brief-face-120-carrier-2026-11-13.md)

## Frozen comparison

The proposed check scans 120 upright 1280 × 720 H.264 pictures for local
changes that return toward their surrounding appearance. It shortlists
motion at the edges and directly compares faces at both boundaries. It
keeps one interval of five through seven pictures at the existing 0.10
operating point and uses at most 84 direct face observations. The
comparison adds it only when the existing Full appearance stages return
no supplemental short range. The evidence concerns a local appearance
change and carries zero overall AI-confidence weight.
[Short-interval localization research](https://arxiv.org/abs/2606.07090)
· [Face asset licenses](video-appearance-plan-v107-method.md)

Six retained RTFS groups have distinct known original links, donor keys
and decoded source components. The freeze excludes overlaps with the
prior four-second dual-swap bank and the two earlier eight-second
proposal banks. Some recordings appeared in other project experiments.
This is separation from those three comparisons and does not establish
unseen people or unknown pretraining ancestry. The panel uses the
publisher's existing edits from both generator families. No face-swap
generator ran. The source grant and incomplete portrait attribution are
documented in the acquisition record.
[Publisher release](https://huggingface.co/datasets/godmodes/rtfs-10k)
· [Source permissions](video-common-source-preparation-v89.md)

Each group supplies an unmodified delivery, its tandem encode, a
six-picture swap, its tandem encode, a brief exposure change, a jump
within the genuine recording and a whole-picture cut to another genuine
recording. Primary and tandem compression use CRF 23 and 35. The known
edit starts are frames 63, 72, 81, 90, 99 and 108. A fresh delivery
contains 120 native pictures at exact 30 fps. All interventions use
verified source pictures on that clock, with no frame interpolation.

An added location must overlap at least half of its temporal union with
the authored range. The fixed criteria require four useful additions
across at least three source groups, including two exact locations,
with at least 75% of routed clips measured. They allow one new ordinary
flag, none on genuine cuts or jumps and one misplaced additional range
on edited footage. The comparison uses existing Full face stages on
the exact same deliveries. The current tracking checkpoint's local
files also matched the existing pinned asset manifest.
[Model manifest](../backend/app/models.yaml)

The offline proposal run measures algorithm
coverage without imposing the serving worker's 12-second deadline.

## Held results

| Outcome | Existing Full | Full with fixed proposal |
| --- | ---: | ---: |
| Useful primary swap locations | 5/6 | 6/6 |
| Useful tandem swap locations | 5/6 | 6/6 |
| Ordinary deliveries flagged | 0/30 | 0/30 |
| Misplaced additional ranges | — | 0 |
| Routed files measured by the proposal | — | 32/32 |

The two gains come from the same source recording in two encodes.
Both ranges match the exact authored **3.600–3.800 s** interval.
Their representative original pictures are **107, 111 and 114**.
The proposal meets the exact-location and ordinary-control criteria,
but adds only two locations from one source group. It remains
experimental on four-second AVC MP4. The current Full review found
five of six independent source edits with zero ordinary flags in
this prepared bank. Only one missed source could benefit here.
[Case and timing ledger](evaluation-panels/video-brief-face-avc-confirmation-2026-11-13.json)

FFmpeg independently reopened all **5,040** native presentation times
and RGB frames, including six selected before, during and after
images. The second authoring run reproduced all 42 video files
byte for byte. The public result records source-clock and image
commitments without redistributing the clips.

## Local replay

The decision reader requires Python and the public receipt. It recounts
locations, ordinary flags and source groups and checks every published
before, during and after frame commitment against the exact native time.
The reader agreed on Python 3.11 and 3.12 and rejected changed truth
and evidence timestamps. The media, source identities and face crops
stay in the local archive.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_brief_face_avc_confirmation \
  docs/evaluation-panels/video-brief-face-avc-confirmation-2026-11-13.json
```

Native replay uses the retained RTFS source archive and the private
freeze directories from the eight-second HEVC and AVC/VP9 runs. Use a new
local output directory for each run. The independent FFmpeg pass verifies
every decoded picture, its presentation time and each selected anchor.
A second authoring run checks that each compressed delivery and all
recorded source bindings are reproducible. The first scoring attempt
stopped at the freeze guard before running inference after a test
configuration import was moved. The scoring run verified the original
delivery hashes against the corrected, otherwise identical freeze.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze prepare score; do
  backend/.venv/bin/python -B -m scripts.video_brief_face_avc_confirmation "$phase" "$run" \
    --prior-hevc "$hevc_run" --prior-avc-vp9 "$carrier_run" || exit 1
done
for phase in freeze prepare; do
  backend/.venv/bin/python -B -m scripts.video_brief_face_avc_confirmation "$phase" "$repeat_run" \
    --prior-hevc "$hevc_run" --prior-avc-vp9 "$carrier_run" || exit 1
done
python3 -B -S -m scripts.review_video_brief_face_avc_confirmation "$receipt" \
  --assemble "$run" --authoring-repeat "$repeat_run"
```
