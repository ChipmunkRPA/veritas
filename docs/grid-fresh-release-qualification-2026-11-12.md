# Local speech timing around independently observed releases

The two single-upload comparisons localized **0 of 26** known ±240 ms shifts. The reference-assisted comparison located all 26 with the correct direction and exact sample offset, using the original waveform and a known mouth event. One single-upload candidate also alerted on an ordinary gain control. These results validate the shifted examples and their source evidence. The reference-assisted result does not establish blind detection performance. [Public counts and verification](evaluation-panels/grid-fresh-release-qualification-2026-11-12.json)

## Qualifying source events

The study selected 34 previously unplanned original GRID utterances in deterministic order. Two separate speaker groups supplied four development utterances. Six other groups supplied 30 held utterances, including 24 lexical targets beginning with /b/ or /p/ and six lexical negatives. The speakers had prior project exposure. The exact original archives, MPEG members and publisher word annotations were committed before observation. The video keeps 75 decoded pictures at 25 frames per second with native presentation timestamps. The mono soundtrack keeps 47,648 contiguous signed PCM samples at 16 kHz. [GRID publisher data and word alignments](https://zenodo.org/records/3625687) · [Prior native source qualification](grid-reserve-word-clocks-2026-11-12.md)

The waveform-only rule looked for a high-band burst following reduced energy and leading into sustained speech. It recorded a 15 ms sample bracket, kept other nearby transients and rejected similarly strong alternatives. The muted-picture rule looked for consecutive near-closed mouth shapes followed by a sustained reopening. It recorded the last closed and first reopened pictures and their native crop hashes. The two passes shared only a coarse word neighborhood from the publisher's annotations. They yielded 15 waveform brackets and 22 visible reopenings across the 24 lexical targets. Thirteen had both observations and met the clock-agreement and interval requirements. The observations came from separate signal rules. They are approximate landmarks and have no independent human phoneme annotation.

## Frozen deliveries and results

The label ledger was sealed before constructing the controls. Each of the 13 supported originals supplied an unchanged copy, a local gain-only control and two copies with the sound advanced or delayed 240 ms around the observed event. The shifted interval contains the source and moved waveform for both directions. Native pictures and samples outside the edited interval remain identical. Six lexical-negative originals brought the scored set to 58 deliveries.

The current comparison uses the existing correlation, speaker admission and lag logic. The experimental single-upload comparison searches for local changes between naturally aligned visual and acoustic releases. A useful shift needs the correct sign and an estimate within two pictures of the known offset. The current timing range also needs at least 40% intersection over union with the edit. For the reference-assisted comparison, the verified earlier waveform supplies a 60 ms acoustic template and the corresponding original mouth event. It searches for the same waveform within ±400 ms. The shifted template must fall wholly inside the authored interval. [Prior single-upload method](grid-single-clip-release-timing-2026-11-12.md) · [Earlier-copy timing method](grid-paired-closure-timing-2026-11-12.md)

| Held result | Current correlation | Experimental single upload | Supplied reference |
| --- | ---: | ---: | ---: |
| Correct locations across 26 shifted deliveries | 0 | 0 | 26 |
| Assessed shifted deliveries | 23 | 6 | 26 |
| Alerts across 19 unchanged originals | 0 | 0 | 0 |
| Alerts across 13 gain-only controls | 0 | 1 | 0 |

The reference comparison covered the 13 unchanged originals with a supported event and all 13 gain-only views. The six lexical-negative originals had no supplied reference event. Every supported held event had one to four nearby acoustic distractors. The waveforms in the shifted intervals came from the same original recording. The reference result therefore measures exact waveform correspondence. It does not estimate sensitivity to synthesized speech, unrelated audio or audio that has been re-encoded.

Independent readback verified 2,550 original pictures, 4,350 scored-view pictures, all 26 copied intervals, and each supplied reference match using a separate direct correlation calculation. Two initial scoring attempts stopped on transient FFmpeg input-open errors. The completed run reused the sealed code, inputs and binary, with an execution wrapper that could retry only the interrupted input open before it returned any bytes. The wrapper was local to the run.

## Reproduce

Use a new local run directory and the retained GRID archives, word annotations and qualified landmark runtime. Keep the original clips, voice samples, crop coordinates and per-clip outcomes local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in intake audio visual seal author score; do
  backend/.venv/bin/python -B -m scripts.grid_fresh_release_qualification "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_grid_fresh_release_qualification "$run"
backend/.venv/bin/python -B -m scripts.review_grid_fresh_release_qualification "$run" --aggregate "$run/public-aggregate.json"
```
