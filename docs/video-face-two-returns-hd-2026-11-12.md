# Two brief face changes at 720p and 30 fps

The local app can show a second brief face change on supported 360p25 uploads. This comparison tests that selection on a separate set of source recordings and a different delivery profile. It preserves any previous reported range and carries zero weight in AI confidence. [Current two-range evidence](video-face-two-returns-2026-11-12.md) · [Current 720p30 single-range evidence](video-face-720p30-transfer-2026-11-12.md)

## Frozen comparison

The study uses eight retained confirmation groups with exact original 30 fps clocks. The documented original links and known donor keys are separate from each other, the eight 360p two-range groups, and the four earlier 720p single-range groups. The files had prior project exposure. The edit placements, lengths, combinations and controls in this comparison were prepared anew. Retained qualified native pictures are resized spatially if needed. All outputs are upright 1280 × 720 H.264 MP4 with 8-bit 4:2:0 and 120 pictures at exactly 30 fps. Each output time takes its corresponding native source picture, with no interpolated or duplicated frames for frame-rate conversion. [Source rights and limitations](video-common-source-preparation-v89.md) · [Retained face assets](video-appearance-plan-v107-method.md)

Each group contributes two six- or eight-picture swaps from the two publisher-rendered families. The four placement patterns vary the position relative to the native scan grid, including odd-picture starts. The two-swap delivery is encoded once and also tandem-recompressed at CRF 35. Controls include an unchanged original, a tandem-recompressed original, a swap paired with a same-source jump, a local illumination change paired with a swap, two ordinary same-recording picture jumps, and two whole-picture cuts to a different genuine recording. The genuine cut counterpart is the other held recording with the strongest centered border luminance correlation on one prespecified genuine picture. This source-only matching does not use detector outputs. Original and publisher frames must differ throughout each authored swap.

The actual Full probe, decode, shot, track, face and detect stages supply the comparator. The candidate only reuses the existing 15 Hz native measurements when the worker already found a brief face range. It retains the same strict appearance-separation point, half-second duration limit, 80 ms gap from every earlier range and .60 lower-quartile background agreement at both boundaries. Both stages share the original 12-second inspection budget. The original app’s multiple-face and other coverage rules remain in force.

A useful range has at least half temporal intersection over union with a previously unlocalized authored edit. The frozen gate requires four additional useful ranges across three groups, at least two newly complete dual-swap deliveries, and at least one useful addition under tandem recompression. It allows at most one additional ordinary range, one incorrect mixed-control addition and one incorrect dual-swap addition. It allows zero added genuine-cut ranges.

## Held results

| Delivery | Files | Useful ranges before | Useful ranges with second review | Additional wrong ranges |
| --- | ---: | ---: | ---: | ---: |
| Two swaps, first encode | 8 | 6/16 | 8/16 | 0 |
| Two swaps, tandem-recompressed | 8 | 6/16 | 9/16 | 0 |
| Swap with same-source jump | 8 | 8/8 | 8/8 | 0 |
| Illumination with swap | 8 | 6/8 | 6/8 | 0 |
| Original and tandem original | 16 | 0 alerts | 0 alerts | 0 |
| Genuine same- or different-recording cuts | 16 | 0 alerts | 0 alerts | 0 |

Five newly localized edits came from three source groups, with three of the five on tandem-recompressed deliveries. Five of sixteen dual-swap files showed both known ranges, up from zero. Four of the five added ranges matched both authored endpoints exactly. The fifth was one frame late at both ends. The useful ranges include both publisher families.

The second check was measured on 12/16 dual-swap deliveries and 14/16 mixed-control deliveries. Thirty-two other files had no qualified first range. Six files had a result from the sparse pass, outside the existing 720p publication scope. The ordinary-only and cut files never reached the second check, so the results show no new alerts under the tested routing. The matching ordinary cross-recording controls had source border correlations from .198 to .519, which leaves genuine subject changes in nearly identical scenes untested.

The Full comparison took 445.4 seconds and reported 4.90 GiB peak RSS. The encoded deliveries occupy 22.2 MiB. The added check ranks the measurements already produced by the guarded 15 Hz pass and adds no new model inference at this delivery profile. Independent replay checked 64 decisions, 7,680 native pictures, 15 exact source crops and 24 byte-exact reconstructions. It recomputed both background boundaries for each of the five added ranges. [Held results and source replay](evaluation-panels/video-face-two-returns-hd-2026-11-12.json)

## Local app verification

With `VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW=true` alongside the brief face-change review and appearance mode, the Full worker reports both source intervals on the tested 720p30 profile. The first encode and its tandem-recompressed copy both showed **2.133–2.333 s** and **2.933–3.200 s**. Before, during and after pictures were native frames **62, 66, 70** and **86, 92, 96** respectively. The ordinary jump paired with one swap kept only its first range. The genuine two-cut control showed no brief face-change range.

Five isolated Full analyses used simulated prepaid checkout. Both ranges preserved their timestamps and native crops across the API, JSON and Standard PDF. Signed original-source replay checked all four guarded background boundaries on both dual-swap deliveries. The four signed packages verified with networking and application runtime imports disabled. [App verification receipt](evaluation-panels/video-face-two-returns-hd-app-2026-11-12.json)

## Replay

Use the retained source archive and a new private output directory. The study froze the app sources at `92e8a1d`. Use those app sources with the current study runner and reviewer to reproduce that comparison. The readback checks actual codec and source clock, all native picture hashes, original frames and crops for any added range, and the two recorded background boundaries. It rebuilds the eight dual-swap, eight tandem dual-swap and eight cross-recording cut deliveries from the verified source pictures. The raw media and individual observations stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_two_returns_hd "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_two_returns_hd "$run"
```
