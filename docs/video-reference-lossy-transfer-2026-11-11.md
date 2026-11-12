# Local picture differences after H.264 re-encoding

The earlier-copy comparator detects every exact RGB difference on a shared native picture clock. Re-encoding alone often changes every frame and makes the exact result unsuitable for locating one short visual edit. This study evaluated a separate spatial measurement using both supplied recordings. It locates differences relative to the earlier copy. Ordinary edits may produce the same evidence. [Earlier-copy method](video-reference-comparison-2026-11-11.md) · [Full-reference video distortion research](https://ece.uwaterloo.ca/~z70wang/publications/vssim.html)

## Frozen test

The study used verified source-bound mouth redraws from eight GRID development recordings, eight separate GRID held recordings, and ten CREMA-D transfer recordings. The GRID groups were separated by known source components. Both banks had earlier project exposure. The held scoring made new H.264 deliveries of the archived native images without changing relative picture timestamps or dimensions. Original re-encode controls include CRF 28, CRF 33 and global tone adjustment. Different-source GRID controls use the matching picture clock and dimensions from a different source component. Local brightness, blur, resampling and picture-retiming controls contain real differences and are reported separately. [GRID and CREMA-D input verification](video-mouth-jaw-transfer-2026-11-11.md)

The detector smooths both luma images, normalizes their broad brightness and contrast using whole-picture quantiles, then measures local tile error above the rest of the picture. It searches for the strongest fixed tile whose error persists across a 0.48–0.96 second span and exceeds the same tile in both flanks. The full comparison requires exact native timestamps and enough shared decoded picture content. The development point is the larger of 0.8 and the greatest neutral development response plus 0.15. Correct localization requires at least 0.50 frame-interval intersection over union. The example crop must also overlap at least one eighth of the tile and one twelfth of the original authored difference in that picture. Both held banks must add useful ranges across three or more source groups, measure at least 80% of known redraws and neutral deliveries, and flag at most one neutral delivery each with zero wrong-earlier-source flags.

The first development run had insufficient coverage. Its 25th-percentile shared-content bound excluded every CRF 33 development control. The largest observed 25th-percentile background error was 2.214 in the neutral set and the smallest wrong-source result was 21.848. The revised freeze set that bound to 2.5 and retained the 65th-percentile bound at 3.0. The final held views were scored only after the new development point was fixed.

## Held results

| Held outcome | GRID | CREMA-D |
| --- | ---: | ---: |
| Redraws correctly localized by the exact comparison | 0/32 | 0/20 |
| Redraws correctly localized in time and pixels by the paired tile check | 24/32 | 20/20 |
| Neutral delivery-only views flagged | 0/24 | 1/30 |
| Different-source comparisons flagged | 0/8, all unassessed | — |
| Authored ordinary local edits correctly located | 16/40 | 20/30 |

Both panels met the frozen gate. One otherwise unchanged CREMA-D clip at CRF 33 yielded a local review range. The seven GRID expression-driven redraws with independently verified source-compatible visible motion all fell below the fixed point. The result measures visual differences when both copies are available. It provides no inference about whether an edit is AI-generated or synchronized to its soundtrack.

The independent reviewer verified all 184 held delivery clocks and identities, reopened 186 source or delivery pictures for mask counts and example crops, and separately recomputed the selected tile response on 61 native picture pairs. It produced 12 exact before, during and after crops. The GRID example selects the authored 1.080–1.880 s range exactly. The CREMA-D example selects 0.733–1.234 s within the authored 0.667–1.335 s edit. The crops and raw recordings remain private.

[Aggregate and independent replay](evaluation-panels/video-reference-lossy-transfer-2026-11-11.json)

## Reproduce

Use the source archive and the earlier paired motion study's frozen roster. Run the study from its exact code revision because the runner binds the picture comparison implementation used by its exact baseline. The independent reviewer reopens the native clocks and selected crop pixels. [Frozen study code](https://github.com/ChipmunkRPA/veritas/tree/25e840055fe7bddafc2bcefd054abee50655f118)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development calibrate held report; do
  backend/.venv/bin/python -B -m scripts.video_reference_lossy_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_reference_lossy_transfer "$run"
```
