# Reproduce JPEG copy evidence from public photos

This CPU-only replay runs the shipped same-orientation copy checks on a fixed public Wikimedia roster. It prepares the JPEG inputs, runs the existing translation check, then tries the exact brightness-block and between-block fallbacks in their serving order. It also reopens each reported crop through the original-source verifier. Reviewers can run the comparison from a clean checkout with the backend Python dependencies and the public source links. [Replay runner](../backend/scripts/image_jpeg_copy_public_replay.py)

## Sources and fixed preparation

Use eight credited contributors across architecture, cityscapes, forests, and lakes. The photos come from the earlier published source roster, which includes attribution, license terms, and the SHA-256 of each acquired delivery. A first attempt to obtain all publisher originals encountered HTTP 429. Before generating or scoring any inputs, the replay adopted the roster's already pinned deliveries of one publisher original and seven Wikimedia thumbnails. Each file passed its recorded SHA-256, and the publisher original also passed Wikimedia's original SHA-1. The acquisition was about 8.8 MiB. The photos have earlier project exposure. This replay authors a new set of edits from the public deliveries and leaves the photos and derivatives in scratch storage. Preserve attribution, the license and change notices, and ShareAlike terms where applicable if redistributing derivative images. [Public source and rights roster](evaluation-panels/image-affine-native-scale-2026-11-11.json) · [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

Orient each source according to its EXIF metadata and reduce its long side to 768 px. Each source supplies an unchanged view, a contrast change, a local blur, and three sharp copy-and-paste edits. Encode each view at JPEG75 and JPEG90. The copies include 64 × 64 px translations aligned to the 8-pixel grid and 40 × 48 and 64 × 64 px translations between blocks. Freeze all 96 JPEG hashes, truth boxes, and changed-pixel masks before scoring. A useful location requires both predicted crops to reach at least 0.50 intersection-over-union with the authored boxes and to recover at least half the changed pixels. The 96 inputs are paired variants of eight sources.

## Local reproduction result

| Delivery | Authored copy | Translation only | With JPEG fallbacks |
| --- | --- | ---: | ---: |
| JPEG75 | 64 × 64 aligned | 5/8 | 8/8 |
| JPEG75 | 40 × 48 between blocks | 0/8 | 2/8 |
| JPEG75 | 64 × 64 between blocks | 1/8 | 4/8 |
| JPEG90 | 64 × 64 aligned | 5/8 | 8/8 |
| JPEG90 | 40 × 48 between blocks | 0/8 | 2/8 |
| JPEG90 | 64 × 64 between blocks | 2/8 | 3/8 |

The JPEG fallbacks added 14 useful locations, from 13 to 27 of 48 edited views. The exact brightness-block comparison added six and the between-block comparison added eight. The four flagged edited views below the overlap criterion came from the initial translation check. No ordinary view flagged across 48 controls. These numbers describe controlled copy-and-paste edits and add zero weight to AI authorship confidence.

The native verifier checked 62 source crop assets. It rejected a one-pixel position change for each observed matching basis and an altered inside-color measurement for the between-block evidence. This replay uses the original-source verification step on a minimal evidence archive. PDF rendering, application payment, and package signing were verified separately in the local application tests. [Frozen public replay receipt](evaluation-panels/image-jpeg-public-replay-2026-11-12.json) · [Earlier app verification](image-offgrid-copy-transfer-2026-11-12.md)

## Run from a clean checkout

Run the commands from the repository root after installing the backend dependencies. The `acquire` stage downloads the pinned public deliveries into a new scratch folder. The remaining stages read those bytes locally and check the frozen commitments. If a publisher delivery is unavailable, rerun acquisition when available or put the exact matching delivery at the documented filename. A changed file or a substitute delivery fails the source hash check.

```sh
replay_project="$(pwd)"
replay_data="$(mktemp -d)"
export PYTHONPATH="$replay_project/backend" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in acquire freeze run review; do
  "$replay_project/backend/.venv/bin/python" -B -m scripts.image_jpeg_copy_public_replay \
    "$stage" "$replay_data" || exit
done
shasum -a 256 "$replay_data/freeze.json" "$replay_data/result.json"
```

Compare the freeze and result SHA-256 with the receipt when using the recorded dependency versions. The reviewer re-runs scoring and original-source verification, then compares every measurement and crop commitment byte for byte against its local result.
