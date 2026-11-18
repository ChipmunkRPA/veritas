# Localize stabilized eye texture during ongoing facial motion

The prior eye-hold trial required nearly unchanged decoded brightness.
In its development clips, encoding changed brightness inside regions
that had been held before encoding. This follow-up measures the
stability of fine texture independently of overall brightness and
contrast. It compares the same region over several consecutive source
pictures and checks for independent movement nearby. The method uses
ordinary image processing and the existing rights-reviewed face
locator. [Earlier eye hold comparison](video-eye-hold-2026-11-18.md) ·
[Recent appearance and temporal residual research](https://arxiv.org/abs/2611.12508)

## Frozen comparison

Each proposed eye crop is placed on the native source pictures. The
comparison subtracts a small blurred copy and normalizes the remaining
texture. It requires sustained similarity between successive pictures
and against the window's midpoint. Texture changes at both ends
support the time range. A separate whole-picture continuity check
rejects abrupt wider changes and nearly identical runs in the
surrounding pictures. At least two nearby regions must show continuing
motion. The three exhibit images are selected before, within, and
after the observed range.

We evaluated three cutoff values on the 27 original development
deliveries. The selected value recovered five of nine edits, with
one misplaced range and zero flags across 18 controls. It and the
continuity rules were fixed before the new held media was analyzed.
The held roster contains ten capture groups with no known source-link
or file overlap with the earlier eye study. The delivery profile,
authored intervals, visible-eye geometry, controls and acceptance
gates match that study. An independent raw-picture check found
changed eye pixels in all 240 edited pictures before recompression.
[Held source, clock, and crop commitments](evaluation-panels/video-eye-texture-hold-2026-11-18.json.gz)

## Held outcome

| Delivery | Existing native checks | First eye candidate | Normalized texture |
| --- | ---: | ---: | ---: |
| All three eye-hold variants | 0/30 | 6/30 | 23/30 |
| Holds after a second lossy encode | 0/10 | 2/10 | 7/10 |
| Ordinary controls with flags | 0/60 | 0/60 | 3/60 |

The texture method found useful locations in all ten held groups.
It assessed all 90 deliveries. One candidate on a staged edit was
misplaced, and the useful ranges overlapped the authored intervals
without matching both boundaries exactly. Each finding's three
original frames and measured crops matched an independent decode.
The median added calculation took 0.124 seconds after decoding.

The three ordinary findings came from one recording. Its original,
blurred-eye and brightness-edited views each triggered a finding.
No whole-picture hold or picture jump did. This run exceeded the
frozen maximum of two ordinary flags, despite the improved sensitivity.

## Independent confirmation

The unchanged rule ran on ten more capture groups. The three
publisher rosters use distinct source keys and file hashes. The
existing picture and mouth checks located zero staged eye holds.
The first eye candidate located 5 of 30. The normalized texture
method located 25 of 30, including 7 of 10 after the second lossy
encode. One finding matched both authored endpoints exactly.

The new method flagged 15 of 60 controls across three capture groups.
Five of those flags came from unchanged or recompressed originals
and four came from whole-picture holds or picture jumps. The first
eye candidate flagged one ordinary delivery. The stronger sensitivity
therefore came with more ordinary alerts in both held comparisons.
The texture comparison remains evaluation-only. All 40 findings in
the confirmation retained independent hashes of their three frames
and crops. [Confirmation receipt](evaluation-panels/video-eye-texture-confirmation-2026-11-18.json.gz)

## Replay

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_eye_texture_hold_panel verify \
  docs/evaluation-panels/video-eye-texture-hold-2026-11-18.json.gz \
  docs/evaluation-panels/video-eye-hold-2026-11-18.json.gz
PYTHONPATH=backend python3 -B -S -m scripts.video_eye_texture_hold_panel verify \
  docs/evaluation-panels/video-eye-texture-confirmation-2026-11-18.json.gz \
  docs/evaluation-panels/video-eye-texture-hold-2026-11-18.json.gz
```

The replay checks the held outcomes, source hashes and source roster
separation from the earlier test. Native reproduction also needs
the public inputs and their pinned face locator. Raw media stays local.
