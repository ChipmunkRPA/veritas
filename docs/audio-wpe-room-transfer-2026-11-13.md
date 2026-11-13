# Blind dereverberation for localized voice evidence

The paired room experiment found five of seven edits with the early room response and three with the full room response. A blind weighted prediction error (WPE) filter estimates late reflections from the stereo waveform itself. This trial tests whether that view adds a correctly localized voice finding to the existing mono result. It keeps the same classifier and speech admission. [Paired room stress experiment](audio-room-stress-transfer-2026-11-13.md) · [Published weighted prediction error method](https://www.rd.ntt/cs/team_project/media/signal/wpe/) · [Commercial model audit](commercial-model-audit.md)

## Method and source roles

Transform stereo at the native 16 kHz clock using 32 ms square-root Hann frames at 8 ms hops. Predict each current stereo frame from ten earlier stereo frames starting 24 ms back. Run three variance-weighted least-squares updates with a local power floor and diagonal loading. Reconstruct the stereo residual as mono, retaining the original sample count and sample indices. The implementation follows the published method and uses no added pretrained weights or copied implementation. It receives the submitted waveforms alone. Evaluation references enter only a separate acoustic measurement.

The assessment uses the 111 prior Aachen deliveries, containing seven publisher edit groups, eight genuine reader groups, and six isolated competing-speech controls. Each foreground appears in dry, early-room, full-room, and four noisy variants. The four foreground-to-background ratios are +5, 0, −5, and −10 dB in the stereo mean. The source clips and room setups have prior project exposure. A separate eight-reader genuine group sets the WPE point at the maximum across full-room and all four noisy levels. These readers share no reader identity with assessment. No points are fitted on the assessment set. The earlier mono scores are reused after verifying exact source roster, room, native time and label hashes. [Publisher speech and labels, CC BY 4.0](https://doi.org/10.5281/zenodo.14214149) · [Measured room responses, MIT](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) · [AMI competing speech, CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The strongest admitted response selects either 1.000–3.000 s or 3.000–5.000 s. The fixed point comparison is strict. A location must cover half a labeled edit and achieve 0.30 temporal intersection over union. The additive candidate must contribute at least two useful edit groups from two synthesis families across full-room and noisy conditions. It may add at most one ordinary group, zero background-only groups, and zero dry or early-room ordinary groups. Repeated variants of a source count once. Report a second retrospective readout at the unchanged mono point. Acoustic results use the known early-room foreground reference and measure changes in scale-invariant signal-to-distortion ratio (SI-SDR). No reference guides the model or chooses a window.

## Acoustic recovery

All 151 inputs produced a usable WPE view. The alternate NumPy FFT agreed on every availability decision and stayed within one signed PCM unit of SciPy on all candidates. Thirty-two candidates matched byte for byte. The mono early-room waveform also serves as the acoustic reference, so comparing either route against that reference on early-room input is degenerate. The table covers the full-room and competing-speech conditions only.

| Paired condition | Median full-clip SI-SDR change | Median edit SI-SDR change |
| --- | ---: | ---: |
| Full room response | +4.16 dB | +4.20 dB |
| +5 dB | +2.39 dB | +2.25 dB |
| 0 dB | +1.71 dB | +1.72 dB |
| −5 dB | +1.54 dB | +2.40 dB |
| −10 dB | +1.81 dB | +3.00 dB |

Each source contributes its median across publisher-labeled edits before the seven sources are pooled.

A separate deterministic fixture confirmed the 96,000-sample output clock and overlap-add reconstruction to within 0.000000000001 units before subtraction. The synthetic wideband fixture with delayed echoes improved by 10.31 dB. A sparse tonal fixture degraded by 12.54 dB. The filter can remove desired predictable energy, which is one reason the detector comparison retains the submitted mono view.

## Localized edits and false alarms

Each cell gives correctly localized edits and false alarms among genuine readers, except the last row, which gives false alarms among isolated background excerpts. This table applies the earlier mono point to both views for diagnostic comparison.

| Paired condition | Mono | Mono plus WPE |
| --- | ---: | ---: |
| Dry | 5/7 edits, 0/8 false | 5/7 edits, 0/8 false |
| Early room response | 5/7, 0/8 | 5/7, 0/8 |
| Full room response | 3/7, 1/8 | 3/7, 2/8 |
| +5 dB | 1/7, 1/8 | 3/7, 4/8 |
| 0 dB | 2/7, 2/8 | 5/7, 5/8 |
| −5 dB | 0/7, 1/8 | 2/7, 3/8 |
| −10 dB | 0/7, 0/8 | 1/7, 1/8 |
| Background only | 2/6 false | 2/6 false |

WPE localized five distinct edit groups in at least one condition where mono missed them. Four synthesis families contributed. At −5 dB the **1.000–3.000 s** window covers the **1.138–2.103 s** labeled edit and the **3.000–5.000 s** window covers the **3.267–4.292 s** labeled edit. The first also survives at −10 dB. Four genuine reader groups were newly flagged in at least one condition. No background-only, dry, or early-room group was newly flagged.

Across the full-room and noisy deliveries, counting each source once regardless of condition, mono found **4/7 edits** and flagged **2/8 genuine readers**. Mono plus WPE found **5/7** and flagged **6/8**. The earlier blind separator rescued three distinct edit groups in specific noise conditions and added two genuine groups. Counting each source once, that earlier combination found **4/7** and flagged **4/8**. The additional locations in individual conditions matter for a user with only one recording. The pooled source totals show how much of that advantage repeats across acoustic variants. [Earlier blind separator comparison](audio-room-stress-transfer-2026-11-13.md)

The development-only WPE point was **7.353628**, compared with the earlier mono point of **5.356590**. At the development-only point the candidate added zero correct locations and zero false flags. Both points failed the frozen advancement rule. For an exploratory tradeoff using assessment scores, allowing 0, 1, 2, 3, or 4 added genuine reader groups would have rescued 0, 0, 2, 4, or 5 distinct edits in at least one acoustic condition. That retrospective tradeoff cannot establish a deployable point.

A prospective retest can compare an evidence-only WPE view against mono and the earlier separator. Use fresh reader groups with matched genuine, conventional edit, and partial-synthesis versions. Render every version in every new measured room with new competing speakers at the same four levels. Choose points on distinct calibration readers before scoring holdout. Compare localized gains at both one and two extra genuine-group flags, alongside false flags on isolated background and ordinary splices. Keep combined AI confidence outside this experiment.

## Replay

The standard-library reader validates the frozen windows and point, recomputes both profiles, and verifies the reused mono responses against the prior public scalar receipt. The local run made 75 development and 219 assessment model calls. Native replay also requires the earlier local source and model commitments. It rerenders the candidate and checks its exact frozen hashes before inference. The receipt contains scalar results and no speech samples. [Public scalar receipt](evaluation-panels/audio-wpe-room-transfer-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_audio_wpe_room_transfer.py \
  docs/evaluation-panels/audio-wpe-room-transfer-2026-11-13.json \
  --ancestor docs/evaluation-panels/audio-room-stress-transfer-2026-11-13.json

export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score; do
  backend/.venv/bin/python -B -m scripts.audio_wpe_room_transfer "$phase" "$run" \
    --stress-run "$paired_room_run" --input "$aachen_workdir" --publisher "$publisher_workdir" \
    --prior-directional "$prior_directional_run" --ami "$ami_recordings" || exit 1
done
```
