# Rebuild the single-upload picture-change test from public recordings

An independent reviewer can now rebuild the shipped picture-return
check's HEVC validation panel from eight pinned CREMA-D recordings.
The source roster records the exact publisher object hashes, native
picture timestamps, and decoded picture commitments. The downloads
total **2,387,226 bytes**. [Publisher license and attribution](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
· [Source roster](evaluation-panels/video-scene-return-public-sources-2026-11-13.json)

The replay reconstructs all **56 deliveries**, including layout changes,
speaker-plus-layout changes and ordinary controls. It compares every
newly generated file hash and result against the original published
ledger. A separate FFmpeg decode verifies **6,720 source timestamps**
and **42 selected RGB frames**. The results include **14 exact change
ranges** and zero alerts on 40 ordinary controls. This verifies the same
previously published check under the same carrier and test conditions.
It measures picture changes and adds no AI-confidence weight.
[Original study](video-scene-return-2026-11-13.md)
· [Public replay receipt](evaluation-panels/video-scene-return-public-replay-2026-11-13.json)

## Run the replay

Use the backend environment with OpenCV 5.0.0 and FFmpeg 7.0.2 with
libx265. Exact encoder bytes can vary with its build and CPU. A mismatch
remains visible and the run retains the reconstructed files for diagnosis.
Run in an empty scratch directory. This replay requires no customer
uploads, paid account, or detector weights.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.replay_video_scene_return_public "$run" --receipt "$receipt"
```

With access to the preexisting local archive, the same command accepts
`--archive "$archive"` to obtain only the eight files by exact hash.
The generated media stay local. Share the receipt and the FFmpeg build
when reporting differences.
