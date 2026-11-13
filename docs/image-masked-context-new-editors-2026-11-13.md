# Frozen contextual patch check on new editor sources

The initial JPEG experiment added one useful retained soft edit while reducing ordinary file highlights. All useful results came from Stable Diffusion 1.5. This follow-up freezes those exact points and applies the same two masked-context readouts, the unchanged visible head and aligned JPEG luma to newly qualified photos. [Frozen method and prior result](image-masked-context-transfer-2026-11-13.md)

The pinned publisher index and strict overlap audit left 18 possible LaMa source groups. They come from the publisher's training split and have no recorded use by this project. Neither frozen head was fitted on LaMa. Eight groups were ordered by a fixed digest before opening media, with six desired admissions. LaMa is a neural object-removal editor. This transfer should be read separately from diffusion insertions. Only one additional photo remained eligible for both SDXL and Kandinsky, and the two variants share that same source. They form a paired challenge and count as one independent photograph. Source freshness refers to the retained project inventory and cannot establish what general encoder pretraining saw. [Original photo rights and edit-family records](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Every LaMa group supplies the unchanged source, a native removal and an inward-feathered version. Ordinary controls include matched local blur, tone and same-photo copy, a resize round trip and an intermediate JPEG encode. The pinned LaMa release supplies exact changed pixels inside its edit mask, with no published autoencoder-only control. The optional diffusion source uses each publisher edit and matched autoencoder control, with a fixed same-size local rectangle copied only from within the publisher's mask. Its positives and ordinary controls use the original benchmark recipe. Final delivery uses JPEG75 4:2:0, with exact pre-final-encode changed masks. Useful regions need at least 30% intersection over union and 50% edited-pixel recovery. No score points or model weights changed.

## Local results

| Check | Useful sharp LaMa regions /6 | Useful feathered LaMa regions /6 | Ordinary LaMa files flagged /36 | Ordinary LaMa groups flagged /6 | Wrong edited-file locations /12 |
|---|---:|---:|---:|---:|---:|
| Current JPEG luma | 0 | 0 | 3 | 3 | 0 |
| Frozen three-class head | 0 | 0 | 6 | 1 | 2 |
| Context disagreement alone | 0 | 0 | 1 | 1 | 0 |
| Joint context and three-class | 0 | 0 | 11 | 2 | 4 |

All six removal sources qualified on their first ordered attempt. The exact removals occupied 1.04–3.50% of their source pixels. The learned checks failed to put a useful LaMa region first even before applying their score points. The joint check also retained wrong locations on four edit deliveries.

Both editor variants of the separate diffusion source qualified. Each exact patch occupies 6.25% of its image. The frozen head and joint check both localized the sharp and feathered Kandinsky patches. Both missed the two SDXL patches. The joint check retained zero of the twelve matched ordinary files that the visible head highlighted. These observations share one photograph. The joint readout produced no newly useful region in this transfer and stays offline.

## Replay

The one-thread CPU run processed 64 deliveries in 108.67 seconds, covering 832 native tiles, 208 visible batches and 832 masked batches. The no-model readback reproduced 64 JPEG and exact mask bindings and 256 native overlap and selection checks. [Frozen aggregate](evaluation-panels/image-masked-context-new-editors-2026-11-13.json)

Use a new run directory, the original source archives and the frozen masked-context run. Only the acquisition stage needs public network access.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze acquire run review; do
  backend/.venv/bin/python -B -m scripts.image_masked_context_new_editors "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --masked "$masked_run" --output "$run" \
    --publication "$receipt" || exit 1
done
```
