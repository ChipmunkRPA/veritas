# Image-region proposals with and without an earlier copy

The frozen patch heads often responded to edited pixels when given their exact coordinates. Their highest-ranked boxes still missed most edits on the same files. This development replay asks whether an existing independent cue can supply a useful region for those misses. It also tests how often the app's existing earlier-copy comparison can locate the same changed pixels when the photographer or reviewer has a separate earlier file. [Fresh patch-head results](image-three-class-patch-fresh-2026-11-12.md) · [Current earlier-copy workflow](image-reference-comparison-2026-11-11.md)

The blind arm uses the existing two-band RGB residual proposal. It measures a consistent change in residual color correlations across two bands and compares each rectangle with four surrounding strips. Keep the previously recorded JPEG75 score point. Also measure the top rectangle before that point, and whether the union of its shortlist with the saved three-class top box contains a useful region. This two-box union is candidate coverage only. A user would still need a way to choose the right box. An exhaustive audit of the three-class geometry tests whether any allowed box could reach the overlap criterion. It uses the known edit mask and is unavailable during blind screening. [Independent color cue and operating point](image-residual-color-transfer-2026-11-11.md)

The reference arm runs the same inspection and evidence-mask routine used by the current app, using either a same-framing earlier PNG of the source or a quality-90 JPEG encoding of that source. The submitted JPEG75 bytes are identical to the blind study. Measure the actual difference mask in submitted-image coordinates, including ordinary blur, tone, same-photo copy and autoencoder controls. An ordinary local edit can be a true difference between files. Those highlights measure how often the reference comparison also responds to conventional changes. [Earlier-copy comparison and limits](image-reference-comparison-2026-11-11.md)

Both panels had been consumed by the time this audit was designed. The earlier panel has 20 photos across three generator families. The later panel has 12 photos across two of those families. Both compare a sharp and feathered edit plus six ordinary controls per photo. An overlapping result requires at least 0.30 intersection over union and 0.50 recall of the exact pre-JPEG changed pixels. No model was loaded or refitted. [Earlier source roles](image-three-class-patch-probe-2026-11-12.md) · [Later source roles and exact edit construction](image-three-class-patch-fresh-2026-11-12.md)

## Results

| Method | Useful sharp regions /32 | Useful feathered regions /32 |
|---|---:|---:|
| Saved three-class top box, before operating point | 7 | 6 |
| Two-band color top box, before operating point | 1 | 1 |
| Two-band color top box, with frozen operating point | 0 | 0 |
| Either saved top box, before operating points | 8 | 7 |
| Any allowed 14-pixel grid rectangle, with exact mask known | 32 | 32 |
| Current comparison with an earlier PNG | 22 | 22 |
| Current comparison with an earlier JPEG90 | 22 | 22 |

The color cue added the sharp and feathered variants of one Stable Diffusion 1.5 edit to the two-box shortlist before applying its score point. At its original point it highlighted 36 of 192 ordinary files across seven photo groups and located no useful edit. The exact geometry audit shows that the head's permitted box dimensions could cover every edit in the panel. The independent color shortlist made little progress on the ranking gap.

Both earlier-copy deliveries localized the same 22 sharp and 22 feathered edits. The feathered cases include 4/6 SDXL, 9/12 Kandinsky and 9/14 Stable Diffusion 1.5 edits. All other 20 edited views returned unresolved integer alignment. Across the 32 unchanged JPEG75 controls, 23 pairs were comparable with either earlier format. The earlier PNG produced zero unchanged highlights and the earlier JPEG90 produced five. They also highlighted 104/192 and 107/192 ordinary files respectively, including intentional conventional changes. The paired result can identify where aligned copies differ when the earlier file is available. The user still needs independent evidence to assess why those copies differ.

## Replay

The one-thread run rebuilt 256 exact JPEG and truth bindings, including 512 app comparison calls, in 125.91 seconds. A second execution repeated the entire local proposal and comparison calculation and matched all 256 private result records. Source assets and masks remain local. [Replay code](../backend/scripts/image_patch_proposal_audit.py) · [Aggregate and commitments](evaluation-panels/image-patch-proposal-audit-2026-11-12.json)

Use the exact earlier and later patch-head study directories, retained source archives and pinned encoder manifest. The runner verifies the checkpoint's file digests without loading it. No public network is needed.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_patch_proposal_audit "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --output "$run" --publication "$receipt" || exit 1
done
```
