# Directional voice localization in two measured rooms

The prior directional-mask experiment found two additional partial edits in one reused room layout. This test repeats the same stereo separation, speech admission, voice weights, two-second analysis windows, and operating points on two newly used measured impulse responses and 19 foreground reader identities held out from the prior directional and partial-publisher panels. The tested route adds directional findings to the existing mono result. [Prior directional method and result](audio-spatial-mask-voice-transfer-2026-11-13.md)

## Sources and frozen comparison

- **Measured rooms.** RWTH Aachen AIR 1.4 supplies binaural dummy-head impulse responses from a stairway and Aula Carolina. The archive includes an MIT license. Each fixed rig uses the publisher's angle label 0 for the lateral target and 90 for the nearly balanced competing speech. At 16 kHz the two lateral responses have 10- and 9-sample differences between their left and right peaks, compared with two and zero samples for the balanced responses. The experiment resamples the full measured coefficients from 48 kHz and retains the first six seconds of the rendered mixture. Direct-path arrival adds roughly 6–9 ms. Evaluation retains the publisher's original sample clock, while reverberation extends the acoustically affected interval. [Aachen database and release](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
- **Foreground speech.** Luong et al.'s LlamaPartialSpoof v1.0.b publisher audio and interval labels use CC BY 4.0. The tested cases contain seven partial edits from seven reader groups and all six synthesis families, plus 12 genuine reader groups. This test used existing published clips and generated no new voices. Exact selected reader IDs, WAV and PCM hashes were checked against 84 prior reader IDs and 208 prior source commitments from the prior publisher and directional source rosters. Underlying reader IDs come from the publisher's source labels. Model-training exposure and undocumented cross-dataset identity overlap are unknown. [Publisher dataset, labels, and license](https://doi.org/10.5281/zenodo.14214149) · [Dataset methodology](https://arxiv.org/abs/2409.14743)
- **Competing voices.** The AMI Consortium's CC BY 4.0 recordings and manual word clocks supply two disjoint sets of four participants, each outside the earlier directional roster. Both AMI sessions were used in other Chipmunk Verify experiments. No two selected six-second excerpts overlap in this test, and none intersects the latest gain-replay intervals. Each edit has at least 400 ms of overlapping annotated words during its principal label and at least 1.2 seconds of background overlap overall. Genuine and background-only controls have at least 600 ms of background overlap overall. The foreground-to-background ratio is −5 dB in the stereo mean. [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [AMI participant consent form](https://groups.inf.ed.ac.uk/ami/corpus/ami_consent_form_081004.pdf) · [Earlier gain replay](audio-gain-public-replay-2026-11-13.md)

Source selection used file hashes and publisher clocks before opening any model scores. The first 192 MiB of the publisher tar admitted six eligible positive reader groups. The first 288 MiB admitted seven, drawn from 852 complete matching native WAV members. The bounded subset contained no same-utterance genuine matches for those seven edits. The control cases therefore compare unrelated genuine readers and six background-only excerpts. The two room cohorts contain four and three edits, respectively. They have distinct readers, and the test does not replay an identical edit in both rooms.

The trial compares rounded stereo mono, the highest response across the prior blind-separation outputs, and the highest response across eligible directional-mask outputs. Each uses the same two windows, 1.000–3.000 s and 3.000–5.000 s. The three prior genuine-reference maxima are 5.356590, 6.760728, and 7.068066 in native model log-odds. A located edit requires the strongest admitted window to exceed its route's frozen point, cover at least half of a publisher-labeled interval, and overlap at least 30% of the union of the window and that interval. The advance rule requires two additions in two synthesis families, no added background-only group, and at most one added ordinary group per room. [Commercial model audit](commercial-model-audit.md)

## Results

Each cell gives localized edits and flagged ordinary groups. Ordinary groups include the background-only controls.

| Measured room | Mono | Prior blind separator | Directional mask | Mono plus directional mask |
| --- | ---: | ---: | ---: | ---: |
| Stairway, 4 edits and 9 controls | 0 / 2 | 1 / 2 | 0 / 0 | 0 / 2 |
| Aula Carolina, 3 edits and 9 controls | 0 / 2 | 0 / 0 | 0 / 0 | 0 / 2 |
| Pooled, 7 edits and 18 controls | 0 / 4 | 1 / 2 | 0 / 0 | 0 / 4 |

The prior separator selected **3.000–5.000 s** in the stairway for a labeled insert at **3.267–4.292 s**. The window covers the full insert and has interval IoU of **0.5125**. The directional mask recovered no above-point edit location in either room and added no ordinary or background-only flags. Mono flagged two of six background-only excerpts and the prior blind separator flagged zero. Mono and the directional arm each left one ordinary case unassessed, on different inputs. There were no above-point flags at the wrong edit location in any arm. The frozen advancement gate failed. The tested directional route remains a research check. The single prior-separator gain does not establish its transfer across rooms or generator families.

The test uses new impulse responses, held-out readers and overlapping real background voices in a controlled render. All positive edits come from one publisher and one type of boundary mixing. The positive cases use a single foreground-to-background ratio and independent ordinary controls. These results cannot establish performance on uploaded room recordings or on routine edits to the same utterance. Repeating each matched utterance and ordinary edit across rooms and several background levels would distinguish room transfer from source variability.

## Replay

The published ledger contains anonymous scalar responses, native truth ranges, and commitments. Public verification recomputes location counts and the gate with Python's standard library. Native replay additionally requires the earlier directional run and publisher roster, the first 288 MiB of the part A archive, the complete AIR 1.4 archive, the two original AMI microphone WAVs, and the pinned local weights. After 236 local model forwards, the native audit reproduced the exact window PCM and speech admission across all 25 cases using NumPy's FFT after the SciPy run. The audit reuses the locked scalar scores. The private roster stays local. [Scalar ledger](evaluation-panels/audio-spatial-mask-aachen-transfer-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_audio_spatial_mask_aachen_transfer.py \
  docs/evaluation-panels/audio-spatial-mask-aachen-transfer-2026-11-13.json

export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score audit; do
  backend/.venv/bin/python -B -m scripts.audio_spatial_mask_aachen_transfer "$phase" "$run" \
    --prior "$prior_directional_run" --publisher "$prior_publisher_run" \
    --input "$publisher_tar_and_air_archive" --ami "$ami_recordings" || exit 1
done
```
