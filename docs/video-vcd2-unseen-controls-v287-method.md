# Additional camera originals for video false-flag testing

Use the publisher's pinned VCD2 metadata and CDLA-Permissive 2.0 agreement. Keep the original-camera `TH` category and exclude the 12 previously qualified recording stems and camera types. Choose the first two eligible camera types in each spatial and temporal stratum by a fixed hash of the recording stem, also excluding camera types already selected in this study. The data agreement permits local use and modifications. Distinct recording stems and camera types do not establish distinct participants. [Publisher release](https://github.com/microsoft/VCD/tree/3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68) · [Camera-native study](https://arxiv.org/html/2603.26763v3) · [Earlier camera study](video-vcd2-genuine-v195-method.md)

Freeze the eight source groups before transfer. Download each official Azure object with the established byte and ETag checks. Verify the FFV1 stream and compare decoded picture times with native packet times. A separate readout rehashes every qualified file and independently repeats its clock check. Failed and unattempted groups remain visible. The original, conventional controls and any later detector output stay distinct. A useful next comparison includes identical export handling and separates face-region flags from whole-clip directions. Recent paired and partial-region studies inform that design. Their noncommercial datasets remain outside this intake. [SynthForensics](https://arxiv.org/abs/2602.04939) · [RLDF](https://arxiv.org/html/2610.21438v1)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.video_vcd2_unseen_controls_v287 freeze --metadata-dir /path/to/pinned/vcd2-metadata
backend/.venv/bin/python -B -m scripts.video_vcd2_unseen_controls_v287 qualify --metadata-dir /path/to/pinned/vcd2-metadata --slot 1
backend/.venv/bin/python -B -m scripts.video_vcd2_unseen_controls_v287 readout --metadata-dir /path/to/pinned/vcd2-metadata
```

Qualification is one recording at a time. Keep source names, videos, native clocks, and per-record receipts on the private archive. No model runs in this intake.
