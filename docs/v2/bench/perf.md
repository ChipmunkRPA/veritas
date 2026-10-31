# Performance benchmark (X-4)

Generated 2026-09-17T10:51:24Z by `backend/scripts/bench.py` (git 88274cef559d3ccc81fbfced76700e0a9123d851) against http://127.0.0.1:8160; device `mps`; macOS-26.6.2-arm64-arm-64bit; Python 3.12.13.

- **Host load.** 1-minute load average 6.15 before and 7.03 after the run (18 logical CPUs; 22.6 of 51.5 GB available at the start); bg.sh jobs of this repository running at the start: dev-api, j4-bench. The host is shared with other projects' jobs, so these figures bound an idle host from above.
- **Method.** Per profile, one untimed warm-up analysis per kind, then n timed analyses per workload through `POST /api/analyze/{kind}` with `profile=`; wall time from upload to `done` as the client saw it, polling every 0.5 s, so it bounds the true time from above by up to one poll interval; the server column is `created_at` → `finished_at` on the analysis row (queue plus run, no upload). p95 is nearest-rank (with n ≤ 20, the maximum). Budgets use the client figures. Peak RSS covers the API process tree, sampled every 0.25 s.

## Workloads

| Workload | Kind | File | Bytes | sha256 | Facts | Derivation |
|---|---|---|---|---|---|---|
| `image_256px_face` | image | `face_Real_1.jpg` | 10953 | `06b0f3809ae2…` | size 256x256 | demo sample data/samples/face_Real_1.jpg |
| `image_800px` | image | `hemg_AiArtData_3.jpg` | 78321 | `87e7991cecd5…` | size 800x800 | demo sample data/samples/hemg_AiArtData_3.jpg |
| `image_12mp` | image | `image_12mp.jpg` | 1271849 | `0b9e66fd83b6…` | size 4000x3000 | data/samples/hemg_AiArtData_1.jpg cropped to 4:3 and resized to 4000 × 3000 (Lanczos), JPEG quality 92 |
| `image_24mp_camera` | image | `image_24mp_camera.jpg` | 4509538 | `06ec53485c06…` | size 6000x4000 | vb-1.1 item `06ec53485c06…` (6000 × 4000, licence Attribution), an R1 Commons photograph of about 24 MP with EXIF, copied unmodified |
| `video_9s_1080p` | video | `vid_fake_1.mp4` | 1164071 | `2a0882fc4a09…` | duration_s 9.04, size 1920x1080 | demo sample data/samples/vid_fake_1.mp4 |
| `video_60s_1080p` | video | `video_60s_1080p.mp4` | 31630281 | `5331f026f527…` | duration_s 60.06, size 1920x1080, sample_rate 44100 | 60 s of core item `9f0bb3c025fc…` (622 s) from 60 s, scaled to 1920 × 1080 (Lanczos), H.264 CRF 20 + AAC 128 kb/s |
| `audio_5min` | audio | `audio_5min.flac` | 23362512 | `14c4fd075ae9…` | duration_s 300.0, sample_rate 44100 | 300 s of core item `9f0bb3c025fc…` (622 s)'s sound from 60 s, FLAC 44.1 kHz mono |
| `text_1000_words` | document | `text_1000_words.txt` | 4847 | `490fe0eec405…` | words 1000 | the first 1,000 words of HC3 human answers (data/eval_cache/hc3_all.jsonl), as scripts/bench.py v0.3 |

## Latency per profile

| Profile | Workload | n | p50 s | p95 s | max s | Server p50 s | Failed | Peak RSS GB | Load 1 min (before → after) | Recorded profile | Pipeline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quick | `image_256px_face` | 5 | 0.52 | 0.53 | 0.53 | 0.27 | 0 | 1.35 | 6.15 → 5.74 | quick | `img-2.0-p6` |
| quick | `image_800px` | 5 | 0.53 | 0.54 | 0.54 | 0.28 | 0 | 1.38 | 5.74 → 5.74 | quick | `img-2.0-p6` |
| quick | `image_12mp` | 3 | 0.54 | 0.54 | 0.54 | 0.44 | 0 | 1.92 | 5.74 → 5.60 | quick | `img-2.0-p6` |
| quick | `image_24mp_camera` | 3 | 2.09 | 2.15 | 2.15 | 1.61 | 0 | 2.80 | 5.60 → 5.39 | quick | `img-2.0-p6` |
| quick | `video_9s_1080p` | 2 | 2.60 | 2.62 | 2.62 | 2.11 | 0 | 3.06 | 5.39 → 5.52 | quick | `vid-2.0-p7` |
| quick | `video_60s_1080p` | 1 | 8.32 | 8.32 | 8.32 | 8.02 | 0 | 3.36 | 5.52 → 6.18 | quick | `vid-2.0-p7` |
| quick | `audio_5min` | 1 | 0.62 | 0.62 | 0.62 | 0.31 | 0 | 3.21 | 6.18 → 6.18 | quick | `aud-2.0-a2` |
| quick | `text_1000_words` | 3 | 2.06 | 2.07 | 2.07 | 1.57 | 0 | 1.54 | 6.08 → 6.08 | quick | `doc-2.0-p7` |
| full | `image_256px_face` | 5 | 0.53 | 0.55 | 0.55 | 0.33 | 0 | 2.37 | 6.07 → 5.99 | full | `img-2.0-p6` |
| full | `image_800px` | 5 | 1.03 | 1.06 | 1.06 | 0.92 | 0 | 2.39 | 5.99 → 6.95 | full | `img-2.0-p6` |
| full | `image_12mp` | 3 | 2.58 | 3.08 | 3.08 | 2.56 | 0 | 4.11 | 6.95 → 6.63 | full | `img-2.0-p6` |
| full | `image_24mp_camera` | 3 | 4.11 | 4.12 | 4.12 | 3.63 | 0 | 3.47 | 6.63 → 5.83 | full | `img-2.0-p6` |
| full | `video_9s_1080p` | 2 | 9.19 | 9.22 | 9.22 | 9.09 | 0 | 4.04 | 5.25 → 6.55 | full | `vid-2.0-p7` |
| full | `video_60s_1080p` | 1 | 50.79 | 50.79 | 50.79 | 50.17 | 0 | 4.34 | 6.55 → 5.18 | full | `vid-2.0-p7` |
| full | `audio_5min` | 1 | 5.18 | 5.18 | 5.18 | 4.66 | 0 | 4.18 | 4.85 → 4.62 | full | `aud-2.0-a2` |
| full | `text_1000_words` | 3 | 2.58 | 3.11 | 3.11 | 2.35 | 0 | 0.58 | 4.81 → 4.30 | full | `doc-2.0-p7` |
| deep | `image_256px_face` | 5 | 0.54 | 0.56 | 0.56 | 0.30 | 0 | 0.68 | 4.30 → 4.19 | deep | `img-2.0-p6` |
| deep | `image_800px` | 5 | 1.04 | 1.07 | 1.07 | 0.89 | 0 | 0.95 | 4.19 → 4.18 | deep | `img-2.0-p6` |
| deep | `image_12mp` | 3 | 3.11 | 3.12 | 3.12 | 2.67 | 0 | 2.29 | 4.18 → 4.22 | deep | `img-2.0-p6` |
| deep | `image_24mp_camera` | 3 | 4.12 | 4.14 | 4.14 | 3.77 | 0 | 2.85 | 4.22 → 4.17 | deep | `img-2.0-p6` |
| deep | `video_9s_1080p` | 2 | 9.74 | 9.75 | 9.75 | 9.44 | 0 | 2.26 | 5.53 → 5.81 | deep | `vid-2.0-p7` |
| deep | `video_60s_1080p` | 1 | 60.86 | 60.86 | 60.86 | 60.57 | 0 | 4.18 | 5.81 → 7.89 | deep | `vid-2.0-p7` |
| deep | `audio_5min` | 1 | 5.21 | 5.21 | 5.21 | 4.67 | 0 | 4.12 | 7.89 → 7.66 | deep | `aud-2.0-a2` |
| deep | `text_1000_words` | 3 | 2.60 | 2.60 | 2.60 | 2.37 | 0 | 4.06 | 7.66 → 7.03 | deep | `doc-2.0-p7` |

Peak RSS per profile (warm-ups included): quick 3.32 GB; full 7.40 GB; deep 4.18 GB.

## Budgets (ARCHITECTURE §8.2, PRD PL-02)

| Workload | Profile | Measured | Target | Gated | Status | Detail |
|---|---|---|---|---|---|---|
| `image_12mp` | quick | 0.54 s | ≤ 3 s | no | PASS |  |
| `image_12mp` | full | 3.08 s | ≤ 8 s | yes | PASS |  |
| `image_12mp` | deep | 3.12 s | ≤ 60 s | no | PASS |  |
| `video_60s_1080p` | quick | 8.32 s | ≤ 40 s | no | PASS |  |
| `video_60s_1080p` | full | 50.79 s | ≤ 120 s | yes | PASS |  |
| `video_60s_1080p` | deep | 60.86 s | ≤ 360 s | no | PASS |  |
| `audio_5min` | quick | 0.62 s | ≤ 20 s | no | PASS |  |
| `audio_5min` | full | 5.18 s | ≤ 60 s | yes | PASS |  |
| `text_1000_words` | quick | 2.07 s | ≤ 10 s | no | PASS |  |
| `text_1000_words` | full | 3.11 s | ≤ 15 s | yes | PASS |  |
| `text_1000_words` | deep | 2.60 s | ≤ 20 s | no | PASS |  |
| `peak_rss` | full | 7.40 GB | ≤ 36 GB | yes | PASS |  |
| `peak_rss` | deep | 4.18 GB | ≤ 44 GB | no | PASS |  |

G13 (`goals.yaml`) reads the `full` figures of `image_256px_face`, `image_800px`, `image_24mp_camera` (J-4, joins the G13a maximum when measured) and `video_9s_1080p` from this file through `vbench_report.py --latency ../docs/v2/bench/perf.json`.

## Stage breakdown

Slowest 5 stages per workload (p50 / max seconds over the timed runs).

| Profile | Workload | Stages |
|---|---|---|
| quick | `image_256px_face` | `provenance` 0.20 / 0.20; `detect` 0.08 / 0.09; `condition` 0.02 / 0.03; `render` 0.02 / 0.02; `decode` 0.01 / 0.01 |
| quick | `image_800px` | `provenance` 0.20 / 0.21; `detect` 0.11 / 0.13; `faces` 0.04 / 0.04; `render` 0.02 / 0.02; `decode` 0.01 / 0.02 |
| quick | `image_12mp` | `provenance` 0.37 / 0.37; `detect` 0.06 / 0.07; `decode` 0.06 / 0.06; `faces` 0.04 / 0.04; `probe` 0.01 / 0.02 |
| quick | `image_24mp_camera` | `render` 0.78 / 0.82; `provenance` 0.75 / 0.75; `decode` 0.29 / 0.30; `detect` 0.20 / 0.32; `faces` 0.05 / 0.05 |
| quick | `video_9s_1080p` | `detect` 1.03 / 1.03; `probe` 0.47 / 0.47; `faces` 0.23 / 0.23; `decode` 0.17 / 0.18; `render` 0.14 / 0.14 |
| quick | `video_60s_1080p` | `probe` 4.82 / 4.82; `detect` 1.88 / 1.88; `decode` 0.83 / 0.83; `faces` 0.21 / 0.21; `render` 0.15 / 0.15 |
| quick | `audio_5min` | `decode` 0.10 / 0.10; `detect` 0.06 / 0.06; `probe` 0.04 / 0.04; `condition` 0.03 / 0.03; `provenance` 0.02 / 0.02 |
| quick | `text_1000_words` | `detect` 1.47 / 1.47; `render` 0.04 / 0.04; `decode` 0.02 / 0.02; `publish` 0.01 / 0.01; `ingest` 0.00 / 0.01 |
| full | `image_256px_face` | `provenance` 0.22 / 0.25; `detect` 0.19 / 0.20; `faces` 0.04 / 0.05; `render` 0.02 / 0.02; `decode` 0.01 / 0.02 |
| full | `image_800px` | `detect` 0.71 / 0.74; `provenance` 0.25 / 0.25; `faces` 0.05 / 0.06; `render` 0.04 / 0.04; `localise` 0.02 / 0.02 |
| full | `image_12mp` | `detect` 1.77 / 1.77; `localise` 0.58 / 0.64; `provenance` 0.42 / 0.42; `decode` 0.07 / 0.08; `faces` 0.04 / 0.04 |
| full | `image_24mp_camera` | `detect` 2.17 / 2.20; `render` 0.79 / 0.84; `provenance` 0.79 / 0.79; `decode` 0.32 / 0.32; `localise` 0.19 / 0.19 |
| full | `video_9s_1080p` | `detect` 5.89 / 5.92; `tracks` 1.90 / 1.90; `faces` 0.62 / 0.65; `probe` 0.49 / 0.49; `shots` 0.47 / 0.48 |
| full | `video_60s_1080p` | `detect` 25.84 / 25.84; `tracks` 15.75 / 15.75; `probe` 4.84 / 4.84; `shots` 3.13 / 3.13; `decode` 1.01 / 1.01 |
| full | `audio_5min` | `detect` 4.38 / 4.38; `decode` 0.12 / 0.12; `probe` 0.05 / 0.05; `condition` 0.04 / 0.04; `provenance` 0.02 / 0.02 |
| full | `text_1000_words` | `detect` 2.22 / 2.76; `render` 0.03 / 0.04; `decode` 0.02 / 0.02; `publish` 0.01 / 0.04; `condition` 0.01 / 0.01 |
| deep | `image_256px_face` | `provenance` 0.19 / 0.25; `detect` 0.17 / 0.21; `faces` 0.03 / 0.03; `render` 0.02 / 0.02; `publish` 0.01 / 0.04 |
| deep | `image_800px` | `detect` 0.71 / 0.81; `provenance` 0.20 / 0.22; `render` 0.04 / 0.04; `faces` 0.03 / 0.04; `localise` 0.02 / 0.03 |
| deep | `image_12mp` | `detect` 1.83 / 1.85; `localise` 0.63 / 0.66; `provenance` 0.39 / 0.41; `decode` 0.06 / 0.08; `faces` 0.04 / 0.05 |
| deep | `image_24mp_camera` | `detect` 2.30 / 2.40; `render` 0.83 / 0.95; `provenance` 0.78 / 0.78; `decode` 0.31 / 0.32; `localise` 0.20 / 0.20 |
| deep | `video_9s_1080p` | `detect` 5.99 / 6.00; `tracks` 2.02 / 2.04; `faces` 0.64 / 0.65; `probe` 0.57 / 0.64; `shots` 0.50 / 0.52 |
| deep | `video_60s_1080p` | `detect` 26.43 / 26.43; `tracks` 20.68 / 20.68; `shots` 7.16 / 7.16; `probe` 5.18 / 5.18; `decode` 2.46 / 2.46 |
| deep | `audio_5min` | `detect` 4.39 / 4.39; `decode` 0.11 / 0.11; `probe` 0.05 / 0.05; `condition` 0.03 / 0.03; `provenance` 0.03 / 0.03 |
| deep | `text_1000_words` | `detect` 2.26 / 2.29; `render` 0.03 / 0.04; `decode` 0.02 / 0.02; `publish` 0.01 / 0.03; `fuse` 0.01 / 0.01 |

## Detector breakdown

Slowest 6 detectors per workload by p50 duration.

| Profile | Workload | Detector | Runs | Statuses | Devices | p50 ms | max ms | Peak MB |
|---|---|---|---|---|---|---|---|---|
| quick | `image_256px_face` | `img.meta.consistency` | 5 | ok 5 | cpu | 92 | 92 | – |
| quick | `image_256px_face` | `img.meta.qtables` | 5 | ok 5 | cpu | 67 | 68 | – |
| quick | `image_256px_face` | `img.dfdc_b7` | 5 | ok 5 | mps | 56 | 57 | – |
| quick | `image_256px_face` | `img.faces.yunet` | 5 | ok 5 | cpu | 3 | 3 | – |
| quick | `image_256px_face` | `img.meta.exif` | 5 | ok 5 | cpu | 2 | 2 | – |
| quick | `image_256px_face` | `img.watermark.dwtdct` | 5 | ok 5 | cpu | 2 | 2 | – |
| quick | `image_800px` | `img.meta.consistency` | 5 | ok 5 | cpu | 96 | 100 | – |
| quick | `image_800px` | `img.meta.qtables` | 5 | ok 5 | cpu | 69 | 74 | – |
| quick | `image_800px` | `img.dfdc_b7` | 5 | ok 5 | mps | 61 | 72 | – |
| quick | `image_800px` | `img.commfor384` | 5 | ok 5 | mps | 37 | 47 | – |
| quick | `image_800px` | `img.faces.yunet` | 5 | ok 5 | cpu | 12 | 26 | – |
| quick | `image_800px` | `img.watermark.dwtdct` | 5 | ok 5 | cpu | 8 | 9 | – |
| quick | `image_12mp` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 158 | 162 | – |
| quick | `image_12mp` | `img.meta.consistency` | 3 | ok 3 | cpu | 109 | 110 | – |
| quick | `image_12mp` | `img.meta.qtables` | 3 | ok 3 | cpu | 76 | 80 | – |
| quick | `image_12mp` | `img.commfor384` | 3 | ok 3 | mps | 49 | 53 | – |
| quick | `image_12mp` | `img.faces.yunet` | 3 | ok 3 | cpu | 30 | 30 | – |
| quick | `image_12mp` | `img.meta.exif` | 3 | ok 3 | cpu | 2 | 2 | – |
| quick | `image_24mp_camera` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 326 | 336 | – |
| quick | `image_24mp_camera` | `img.meta.consistency` | 3 | ok 3 | cpu | 291 | 294 | – |
| quick | `image_24mp_camera` | `img.dfdc_b7` | 3 | ok 3 | mps | 123 | 258 | – |
| quick | `image_24mp_camera` | `img.meta.qtables` | 3 | ok 3 | cpu | 102 | 105 | – |
| quick | `image_24mp_camera` | `img.commfor384` | 3 | ok 3 | mps | 57 | 58 | – |
| quick | `image_24mp_camera` | `img.faces.yunet` | 3 | ok 3 | cpu | 27 | 46 | – |
| quick | `video_9s_1080p` | `vid.frames.dfdc_b7` | 2 | ok 2 | mps | 834 | 840 | – |
| quick | `video_9s_1080p` | `vid.faces.yunet` | 2 | ok 2 | cpu | 219 | 225 | – |
| quick | `video_9s_1080p` | `vid.forensic.face_noise` | 2 | ok 2 | cpu | 173 | 176 | – |
| quick | `video_9s_1080p` | `prov.c2pa.video` | 2 | not_applicable 2 | cpu | 9 | 9 | – |
| quick | `video_9s_1080p` | `vid.temporal.face_flicker` | 2 | ok 2 | cpu | 4 | 4 | – |
| quick | `video_9s_1080p` | `aud.gustking_xlsr` | 2 | not_applicable 2 | – | 0 | 0 | – |
| quick | `video_60s_1080p` | `aud.gustking_xlsr` | 1 | ok 1 | mps | 861 | 861 | – |
| quick | `video_60s_1080p` | `vid.frames.dfdc_b7` | 1 | ok 1 | mps | 824 | 824 | – |
| quick | `video_60s_1080p` | `vid.faces.yunet` | 1 | ok 1 | cpu | 202 | 202 | – |
| quick | `video_60s_1080p` | `vid.forensic.face_noise` | 1 | ok 1 | cpu | 175 | 175 | – |
| quick | `video_60s_1080p` | `prov.c2pa.video` | 1 | not_applicable 1 | cpu | 9 | 9 | – |
| quick | `video_60s_1080p` | `vid.temporal.face_flicker` | 1 | ok 1 | cpu | 3 | 3 | – |
| quick | `audio_5min` | `aud.gustking_xlsr` | 1 | ok 1 | mps | 57 | 57 | – |
| quick | `text_1000_words` | `txt.desklib_ft_v2` | 3 | ok 3 | mps | 1449 | 1451 | – |
| quick | `text_1000_words` | `txt.stylometry` | 3 | ok 3 | cpu | 6 | 7 | – |
| full | `image_256px_face` | `img.meta.consistency` | 5 | ok 5 | cpu | 110 | 125 | – |
| full | `image_256px_face` | `img.meta.qtables` | 5 | ok 5 | cpu | 83 | 84 | – |
| full | `image_256px_face` | `img.dfdc_b7` | 5 | ok 5 | mps | 62 | 83 | – |
| full | `image_256px_face` | `img.forensic.jpeg_dq` | 5 | ok 5 | cpu | 48 | 74 | – |
| full | `image_256px_face` | `img.forensic.ela` | 5 | ok 5 | cpu | 35 | 37 | – |
| full | `image_256px_face` | `img.forensic.copymove` | 5 | ok 5 | cpu | 7 | 7 | – |
| full | `image_800px` | `img.dda` | 5 | ok 5 | mps | 386 | 412 | – |
| full | `image_800px` | `img.meta.consistency` | 5 | ok 5 | cpu | 114 | 117 | – |
| full | `image_800px` | `img.meta.qtables` | 5 | ok 5 | cpu | 93 | 98 | – |
| full | `image_800px` | `img.forensic.jpeg_dq` | 5 | ok 5 | cpu | 69 | 72 | – |
| full | `image_800px` | `img.dfdc_b7` | 5 | ok 5 | mps | 57 | 57 | – |
| full | `image_800px` | `img.forensic.ela` | 5 | ok 5 | cpu | 47 | 48 | – |
| full | `image_12mp` | `img.forensic.ela` | 3 | ok 3 | cpu | 477 | 498 | – |
| full | `image_12mp` | `img.forensic.noise` | 3 | ok 3 | cpu | 443 | 448 | – |
| full | `image_12mp` | `img.dda` | 3 | ok 3 | mps | 441 | 443 | – |
| full | `image_12mp` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 177 | 181 | – |
| full | `image_12mp` | `img.forensic.jpeg_dq` | 3 | ok 3 | cpu | 169 | 171 | – |
| full | `image_12mp` | `img.forensic.copymove` | 3 | ok 3 | cpu | 144 | 147 | – |
| full | `image_24mp_camera` | `img.forensic.ela` | 3 | ok 3 | cpu | 535 | 570 | – |
| full | `image_24mp_camera` | `img.forensic.noise` | 3 | ok 3 | cpu | 462 | 469 | – |
| full | `image_24mp_camera` | `img.forensic.jpeg_dq` | 3 | ok 3 | cpu | 458 | 459 | – |
| full | `image_24mp_camera` | `img.dda` | 3 | ok 3 | mps | 433 | 434 | – |
| full | `image_24mp_camera` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 335 | 339 | – |
| full | `image_24mp_camera` | `img.meta.consistency` | 3 | ok 3 | cpu | 315 | 318 | – |
| full | `video_9s_1080p` | `vid.synth.frames.dda` | 2 | ok 2 | mps | 3100 | 3103 | – |
| full | `video_9s_1080p` | `vid.tracks.yunet_dinov2` | 2 | ok 2 | mps | 1892 | 1893 | – |
| full | `video_9s_1080p` | `vid.frames.dfdc_b7` | 2 | ok 2 | mps | 1846 | 1879 | – |
| full | `video_9s_1080p` | `vid.faces.yunet` | 2 | ok 2 | cpu | 608 | 635 | – |
| full | `video_9s_1080p` | `vid.synth.frames.commfor384` | 2 | ok 2 | mps | 406 | 414 | – |
| full | `video_9s_1080p` | `vid.forensic.face_noise` | 2 | ok 2 | cpu | 385 | 386 | – |
| full | `video_60s_1080p` | `vid.synth.frames.dda` | 1 | ok 1 | mps | 18565 | 18565 | – |
| full | `video_60s_1080p` | `vid.tracks.yunet_dinov2` | 1 | ok 1 | mps | 15495 | 15495 | – |
| full | `video_60s_1080p` | `vid.frames.dfdc_b7` | 1 | ok 1 | mps | 2074 | 2074 | – |
| full | `video_60s_1080p` | `vid.synth.frames.commfor384` | 1 | ok 1 | mps | 1876 | 1876 | – |
| full | `video_60s_1080p` | `vid.lipsync.landmark_xcorr` | 1 | ok 1 | cpu | 1763 | 1763 | – |
| full | `video_60s_1080p` | `aud.gustking_xlsr` | 1 | ok 1 | mps | 946 | 946 | – |
| full | `audio_5min` | `aud.gustking_xlsr` | 1 | ok 1 | mps | 4372 | 4372 | – |
| full | `text_1000_words` | `txt.desklib_ft_v2` | 3 | ok 3 | mps | 1513 | 2040 | – |
| full | `text_1000_words` | `txt.binoculars` | 3 | ok 3 | mps | 684 | 697 | – |
| full | `text_1000_words` | `txt.stylometry` | 3 | ok 3 | cpu | 7 | 7 | – |
| deep | `image_256px_face` | `img.meta.consistency` | 5 | ok 5 | cpu | 93 | 101 | – |
| deep | `image_256px_face` | `img.meta.qtables` | 5 | ok 5 | cpu | 66 | 72 | – |
| deep | `image_256px_face` | `img.dfdc_b7` | 5 | ok 5 | mps | 56 | 95 | – |
| deep | `image_256px_face` | `img.forensic.jpeg_dq` | 5 | ok 5 | cpu | 48 | 59 | – |
| deep | `image_256px_face` | `img.forensic.ela` | 5 | ok 5 | cpu | 8 | 18 | – |
| deep | `image_256px_face` | `img.forensic.copymove` | 5 | ok 5 | cpu | 7 | 8 | – |
| deep | `image_800px` | `img.dda` | 5 | ok 5 | mps | 390 | 448 | – |
| deep | `image_800px` | `img.meta.consistency` | 5 | ok 5 | cpu | 95 | 98 | – |
| deep | `image_800px` | `img.meta.qtables` | 5 | ok 5 | cpu | 71 | 76 | – |
| deep | `image_800px` | `img.forensic.jpeg_dq` | 5 | ok 5 | cpu | 69 | 69 | – |
| deep | `image_800px` | `img.dfdc_b7` | 5 | ok 5 | mps | 56 | 66 | – |
| deep | `image_800px` | `img.forensic.noise` | 5 | ok 5 | cpu | 47 | 50 | – |
| deep | `image_12mp` | `img.forensic.ela` | 3 | ok 3 | cpu | 484 | 491 | – |
| deep | `image_12mp` | `img.dda` | 3 | ok 3 | mps | 473 | 476 | – |
| deep | `image_12mp` | `img.forensic.noise` | 3 | ok 3 | cpu | 460 | 481 | – |
| deep | `image_12mp` | `img.forensic.jpeg_dq` | 3 | ok 3 | cpu | 177 | 180 | – |
| deep | `image_12mp` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 172 | 175 | – |
| deep | `image_12mp` | `img.forensic.copymove` | 3 | ok 3 | cpu | 144 | 153 | – |
| deep | `image_24mp_camera` | `img.forensic.ela` | 3 | ok 3 | cpu | 541 | 558 | – |
| deep | `image_24mp_camera` | `img.forensic.noise` | 3 | ok 3 | cpu | 487 | 488 | – |
| deep | `image_24mp_camera` | `img.dda` | 3 | ok 3 | mps | 475 | 495 | – |
| deep | `image_24mp_camera` | `img.forensic.jpeg_dq` | 3 | ok 3 | cpu | 465 | 491 | – |
| deep | `image_24mp_camera` | `img.watermark.dwtdct` | 3 | ok 3 | cpu | 335 | 354 | – |
| deep | `image_24mp_camera` | `img.meta.consistency` | 3 | ok 3 | cpu | 302 | 315 | – |
| deep | `video_9s_1080p` | `vid.synth.frames.dda` | 2 | ok 2 | mps | 3178 | 3192 | – |
| deep | `video_9s_1080p` | `vid.tracks.yunet_dinov2` | 2 | ok 2 | mps | 2002 | 2033 | – |
| deep | `video_9s_1080p` | `vid.frames.dfdc_b7` | 2 | ok 2 | mps | 1847 | 1852 | – |
| deep | `video_9s_1080p` | `vid.faces.yunet` | 2 | ok 2 | cpu | 628 | 634 | – |
| deep | `video_9s_1080p` | `vid.synth.frames.commfor384` | 2 | ok 2 | mps | 412 | 423 | – |
| deep | `video_9s_1080p` | `vid.forensic.face_noise` | 2 | ok 2 | cpu | 400 | 412 | – |
| deep | `video_60s_1080p` | `vid.tracks.yunet_dinov2` | 1 | ok 1 | mps | 20458 | 20458 | – |
| deep | `video_60s_1080p` | `vid.synth.frames.dda` | 1 | ok 1 | mps | 19022 | 19022 | – |
| deep | `video_60s_1080p` | `vid.frames.dfdc_b7` | 1 | ok 1 | mps | 2140 | 2140 | – |
| deep | `video_60s_1080p` | `vid.synth.frames.commfor384` | 1 | ok 1 | mps | 1925 | 1925 | – |
| deep | `video_60s_1080p` | `vid.lipsync.landmark_xcorr` | 1 | ok 1 | cpu | 1762 | 1762 | – |
| deep | `video_60s_1080p` | `vid.faces.yunet` | 1 | ok 1 | cpu | 1012 | 1012 | – |
| deep | `audio_5min` | `aud.gustking_xlsr` | 1 | ok 1 | mps | 4385 | 4385 | – |
| deep | `text_1000_words` | `txt.desklib_ft_v2` | 3 | ok 3 | mps | 1562 | 1587 | – |
| deep | `text_1000_words` | `txt.binoculars` | 3 | ok 3 | mps | 680 | 682 | – |
| deep | `text_1000_words` | `txt.stylometry` | 3 | ok 3 | cpu | 7 | 8 | – |

## Model memory (measured footprints)

| Model | Device | GB | Load s | Measured at |
|---|---|---|---|---|
| `clip.vit_l14_openai` | mps | 2.215 | 2.47 | 2026-09-13T13:27:25+00:00 |
| `commfor.vit_s16_384` | mps | 0.109 | 0.22 | 2026-09-17T10:46:08+00:00 |
| `dda.dinov2_vitl14_lora` | mps | 1.091 | 2.90 | 2026-09-17T10:46:54+00:00 |
| `dfdc.effnet_b7_555` | mps | 0.260 | 0.83 | 2026-09-17T10:46:09+00:00 |
| `dfdc.effnet_b7_999` | mps | 0.294 | 0.71 | 2026-09-17T10:46:10+00:00 |
| `dinov2.small` | mps | 0.025 | 0.20 | 2026-09-17T10:47:24+00:00 |
| `gend.clip_l14` | mps | 2.215 | 4.47 | 2026-09-13T00:59:21+00:00 |
| `hf.ateeqq.ai_vs_human_image_detector` | mps | 0.411 | 0.32 | 2026-09-14T05:50:14+00:00 |
| `hf.desklib.ai_text_detector_v1_01` | mps | 1.074 | 0.85 | 2026-09-12T13:15:09+00:00 |
| `hf.dima806.deepfake_vs_real_image_detection` | mps | 0.336 | 0.29 | 2026-09-14T05:50:14+00:00 |
| `hf.qwen.qwen2_5_1_5b` | mps | 1.078 | 3.93 | 2026-09-17T10:48:53+00:00 |
| `hf.qwen.qwen2_5_1_5b_instruct` | mps | 3.272 | 2.44 | 2026-09-17T10:48:55+00:00 |
| `local.desklib_ft_v2` | mps | 1.082 | 1.06 | 2026-09-17T10:46:43+00:00 |
| `syncnet.latentsync_stable` | mps | 2.231 | 0.85 | 2026-09-13T04:44:56+00:00 |
| `v1:audio:Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | mps | 2.189 | 2.12 | 2026-09-14T00:48:09+00:00 |
| `v1:audio:mo-thecreator/Deepfake-audio-detection` | mps | 0.336 | 1.10 | 2026-09-14T18:53:41+00:00 |
| `v1:dfdc_effnet_b7` | mps | 0.109 | 1.55 | 2026-09-14T18:53:37+00:00 |
| `v1:imgclf:Wvolf/ViT_Deepfake_Detection` | mps | 0.302 | 1.09 | 2026-09-14T18:53:33+00:00 |
| `v1:imgclf:haywoodsloan/ai-image-detector-deploy` | mps | 1.258 | 1.35 | 2026-09-14T18:53:29+00:00 |
| `v1:text:binoculars` | mps | 5.444 | 6.89 | 2026-09-12T10:22:47+00:00 |
| `v1:text:desklib` | mps | 1.074 | 1.84 | 2026-09-12T10:22:38+00:00 |
| `v1:univfd` | mps | 2.189 | 3.13 | 2026-09-14T18:53:28+00:00 |
| `yunet.2023mar` | cpu | 0.005 | 0.02 | 2026-09-17T10:46:08+00:00 |

## Limits

- Few runs per workload (n in the latency table): p95 is the slowest run, not a population estimate.
- The 12 MP image, 60 s video and 5 min audio are derived workloads (table above), not field media.
- `quick` scores one audio chunk of a recording (`engine/profiles.py`, `audio_windows=1`); `full` and `deep` score the whole track.
- `deep` ran as many detectors as `full` on `image_256px_face`, `image_800px`, `image_12mp`, `image_24mp_camera`, `video_9s_1080p`, `video_60s_1080p`, `audio_5min`, `text_1000_words`, so these figures do not measure deep-only work.
- On Apple silicon, Metal allocations of MPS models may not all show in process RSS; model memory is the manager's measured footprint.
- The host was not idle (load above); figures are not a clean-host measurement.
