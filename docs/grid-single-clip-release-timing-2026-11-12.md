# Single-clip speech release timing on native GRID video

The candidate localized **0 of 48** known audio shifts. It assessed **17 of 120** views and returned zero timing ranges. The current one-second calculation measured **112 of 120** views and also returned zero ranges. This candidate did not qualify for the app. [Aggregate and source replay](evaluation-panels/grid-single-clip-release-timing-2026-11-12.json)

## Method and discovery

Locate a closed-lip release between the last closed native picture and the following picture. Search the same clip's mono soundtrack for a short high-frequency transient after low energy and before sustained speech energy. The visual measurement uses three eye-normalized inner-lip gaps. The sound measurement uses five-millisecond power bins and a 1.6 kHz high-pass. Measure ordinary timing from unambiguous picture–sound pairings on either side of a proposed change. A reported middle event requires a unique alternative and a change of 160–400 ms. This is an independent event measurement. A burst can come from several types of speech sound, and event proximity by itself gives no phoneme identity. [Acoustic and visual release timing](https://doi.org/10.1371/journal.pcbi.1003743) · [Frame-level audiovisual consistency study](https://arxiv.org/abs/2610.30872)

Discovery used two existing originals, two synchronized neural redraws, two delayed neural redraws, and two ordinary mouth retimes. The method measured all eight and returned no ranges. The measured visible release in the two delayed redraws moved 320 ms. A different acoustic transient from the one near each original closure was within 60 ms and 20 ms, respectively, of the newly positioned picture anchor. A proximity rule treated both as plausible local matches. The raw candidate needs a way to resolve that ambiguity before it can reliably interpret timing. [Independently observed redraws](grid-reserve-neural-mouth-2026-11-12.md)

## Frozen held panel

Twelve other qualified originals came from six source groups outside discovery. These retained recordings have prior project exposure. The 108 delivered controls and edited copies were built for this test. All views contain 75 native pictures at 25 fps with upright 360 × 288 display dimensions and a continuous zero-origin 16 kHz mono soundtrack. The fixed held contract binds source files, code, signed edits, controls, and the promotion gate before any held detector outcome. [GRID source terms](https://zenodo.org/records/3625687) · [Earlier source qualification](grid-reserve-word-clocks-2026-11-12.md)

Four local edits per recording move sound 320 ms earlier or later in **0.720–1.520 s**, or 240 ms earlier or later in **1.080–1.960 s**. Each keeps the native pictures and samples outside the interval unchanged. A 10 ms crossfade applies at each audio edit boundary. Controls include untouched originals, exact picture and audio remuxes, local gain and low-pass filtering, and ±320 ms whole-soundtrack offsets. The latter have a real global offset and test whether the candidate incorrectly calls it a local change. A useful event must put the acoustic anchor inside the unblended authored interval and estimate the signed change within 80 ms. The current check uses its usual one-second timing arithmetic with directly observed single-face native mouth patches. Its range criterion requires 0.40 interval overlap and the same signed tolerance.

| Held condition | Candidate assessed | Candidate correct | Current measured |
| --- | ---: | ---: | ---: |
| Local sound 320 ms later | 2/12 | 0/12 | 11/12 |
| Local sound 320 ms earlier | 1/12 | 0/12 | 11/12 |
| Local sound 240 ms later | 1/12 | 0/12 | 12/12 |
| Local sound 240 ms earlier | 2/12 | 0/12 | 11/12 |
| Untouched, remuxed, gain and filtered controls | 9/48 | 0 alerts | 44/48 |
| Constant whole-track offsets | 2/24 | 0 local alerts | 23/24 |

The gate required eight useful results across at least three groups, at least 60% assessed coverage, and at most one control flag. Low flag rates accompanied widespread abstention. The evidence here concerns ordinary timing shifts. The held panel contains no neural positives, and the two neural delayed cases were discovery only. A timing shift can come from ordinary editing and carries no AI confidence.

## Reproduce and review

An independent FFprobe and FFmpeg pass verified all **9,000 native pictures**, **5,717,760 decoded samples**, and the exact authored waveform of each delivery. The replay independently reconstructed the controls and edits, including the fixed blends. It also verified the published counts. No reported event range was available for crop readback.

Use the retained reserve sources, the reviewed local landmark runtime and the existing native video timing tools. Keep source media, per-recording events, speech samples and native crops local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in discovery freeze author score report; do
  backend/.venv/bin/python -B -m scripts.grid_single_clip_release_timing "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_grid_single_clip_release_timing "$run"
```
