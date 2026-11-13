# Brief face changes across eight seconds of HEVC video

The current short face-change proposal inspects an exact four-second, 120-picture HEVC delivery. This study applies the same measured rule to three overlapping four-second windows within a continuous eight-second clip. A later window can supply a source-bound interval when the first four seconds contain no relevant footage. The comparison concerns six-picture face changes and would carry zero weight in overall confidence. [Earlier four-second app evidence](video-face-unrouted-short-transfer-2026-11-12.md)

Recent work on sparse video manipulations evaluates consecutive-frame evidence and window-level localization. This study uses those ideas to test wider clock coverage. It reuses the project's existing small, commercially admitted face detection and appearance assets and the same 0.10 internal measurement point. [Temporal manipulation study](https://arxiv.org/abs/2606.07090) · [Appearance and residual research](https://arxiv.org/abs/2611.12508) · [Face asset terms](video-appearance-plan-v107-method.md)

## Frozen comparison

The test selected two development and six held RTFS source groups from a retained publisher archive. Known source links, donor keys and original source components do not overlap across the eight groups. They were previously used in other project studies. The roster used a source-only native picture qualification and excluded the eight groups in the preceding dual-edit HD test where possible. The study took the first eight continuous seconds of original recordings, with original picture times and no time interpolation. Each positive inserts six same-time pictures from a corresponding published face swap. The early and late edits use the two separate publisher manipulation versions. Late starts vary between frame 128 and frame 212. Tandem controls re-encode the late edit once at CRF 35 after its primary CRF 26 encode. [Publisher sources, rights and limitations](video-common-source-preparation-v89.md)

Each group supplies four positive deliveries, one with both edits. The five ordinary controls are the unchanged original, a whole-picture loop, a same-source jump, an exposure adjustment and a cut to a different genuine recording. The comparison evaluates the new 1280×720 HEVC deliveries at exact 30 fps, upright, with `hvc1` MP4 and 8-bit 4:2:0. It compares the actual Full face stages, a hypothetical first 120-picture window, and three independent windows starting at frames 0, 60 and 120. Every comparison uses the same media bytes. Three representative pictures preserve the original time and pixel commitments around each candidate. Overlapping results are compared on the full native clock and deduplicated against earlier ranges.

The frozen gate requires eight new useful late locations over the first window across four held groups, four added exact late ranges, at least 75% fully assessed coverage, no loss of a useful first-window result, at most one new ordinary flag and zero new genuine-cut flags. A useful location overlaps at least half the temporal union with its authored range.

## Held results

| Outcome | Current Full | First four seconds plus Full | Overlapping windows plus Full |
| --- | ---: | ---: | ---: |
| Known changes localized | 0/30 | 12/30 | 30/30 |
| Changes localized after four seconds | 0/18 | 0/18 | 18/18 |
| Late changes localized after tandem encoding | 0/6 | 0/6 | 6/6 |
| Ordinary deliveries with a flagged range | 0/30 | 0/30 | 0/30 |
| Deliveries with all planned windows measured | — | 54/54 | 54/54 |

Every selected range matched both authored boundaries. All six held groups supplied new late locations. The latest range is **7.067–7.267 s**, or native frames **[212,218)**. No useful first-window range was lost. The candidate requested at most 88 distinct native face inputs on the held files. The two development groups supplied five of ten exact locations and no new flags on ten controls, showing that an eligible clip can still miss edits. A separate FFmpeg and FFprobe pass checked 5,760 native timestamps and 90 source picture anchors on the 24 localized held deliveries. [Frozen case and source-commitment ledger](evaluation-panels/video-face-short-prefix-transfer-2026-11-12.json)

This is a controlled benchmark of two publisher face-edit methods on continuous eight-second sources. It leaves different lengths, codecs, varied cadences, multi-person clips and lip-sync timing outside the demonstrated scope.

## Local app evidence

Enable `VERITAS_VIDEO_FACE_SHORT_WINDOW_REVIEW=true` to run the eight-second check in Full or Deep analysis. The setting defaults to off. The ordinary view shows each measured range with original before, during and after pictures and their face crops. The window coverage lists which parts were measured and which were unassessed. If processing time runs out, the existing finding stays available and incomplete windows are identified as unassessed.

The isolated checkout simulation confirmed two ranges on the same held upload, **2.367–2.567 s** and **4.267–4.467 s**, with source frames 70, 74, 77, 127, 131 and 134. The browser, Standard PDF and JSON used the same source times and pixels. The signed replay remeasured each selected shortlist and both boundaries from the original upload. A changed original was rejected. The ordinary cut reported no ranges and kept measured window coverage. Disabling the setting preserved the same overall decision. The separate earlier 120-picture result kept its existing evidence version and replayed unchanged. [Local app and browser receipt](evaluation-panels/video-face-short-windows-app-2026-11-12.json)

## Reproduce

Use the retained publisher archive and a new private `$run` folder. The source matcher checks native clocks and RGB hashes before creating each view. The current Full comparator runs on the same encoded bytes.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_face_short_prefix_transfer "$phase" "$run" || exit 1
done
for phase in development held; do
  backend/.venv/bin/python -B -m scripts.replay_video_face_short_prefix_incumbent "$run" "$phase" || exit 1
done
python3 -B -S -m scripts.review_video_face_short_prefix_transfer "$run/public.json" --assemble "$run"
```
