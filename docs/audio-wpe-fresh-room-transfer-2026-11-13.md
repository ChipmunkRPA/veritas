# Test dereverberation on matched edits in new measured rooms

The earlier Aachen test found that the dereverberated view could recover additional voice edits at the previous mono point, with four added genuine reader flags. This test freezes a new matched panel before inference and compares the same WPE implementation at two earlier points and three points fitted on separate genuine controls. [Earlier dereverberation test](audio-wpe-room-transfer-2026-11-13.md) · [Published weighted prediction error method](https://www.rd.ntt/cs/team_project/media/signal/wpe/)

## Matched audio and independent roles

The bounded LlamaPartialSpoof v1.0.b publisher archive supplies two held genuine host readers outside the earlier directional, Aachen and WPE point rosters. Each host has an untouched original, two ordinary same-reader splices, and two synthetic-splice variants, one for each of two synthesis families. The synthetic splices use segments already labeled by the publisher. One host uses segments linked to the same declared underlying reader. The second uses a separate publisher reader's edited segments, so its synthetic variant also changes speaker. Both ordinary splices use a different genuine segment from the host. The 960 ms edits use identical 20 ms fades. Before room rendering, matched waveforms are byte-identical outside the authored splice. The selected publisher prefix did not supply untouched counterparts for the exact synthetic utterances. [Publisher speech and labels, CC BY 4.0](https://doi.org/10.5281/zenodo.14214149) · [Publisher methodology](https://arxiv.org/abs/2409.14743)

The two held measured head-rig geometries are the Aachen meeting and lecture rooms. The earlier WPE panel used Aachen's stairway and Aula Carolina. Each host and ordinary control repeats in both new rooms and at 0 and −5 dB in the stereo mean. The matched versions share the same background excerpt and gain. Two separate AMI sessions provide recorded overlapping words, with zero declared participants shared with the previous directional or WPE panels. Their source recordings had prior project use in other tests. Each selected edit interval contains 560–960 ms of overlapping annotated competing words. Six separate background-only excerpts test false flags without a target reader. [Aachen database, MIT](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) · [AMI recordings and manual annotations, CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Three additional genuine readers form the point-fitting group. Each contributes an untouched and an ordinary-splice control in each room at both levels. Two further genuine readers stay in the assessment group. The point procedure takes each development reader's highest WPE response on deliveries where mono did not flag, then fixes one point at each of the three ranks. A strict comparison above those points admits at most zero, one or two additional genuine groups in development. The frozen points were **6.937976**, **5.525381**, and **−2.862373**, respectively. The earlier WPE point was **7.353628** and the earlier mono point **5.356590**. With three development readers the fitted points describe this sample. The low two-reader point is included to show sensitivity to this small calibration set.

Use the unchanged mono score and add the strongest admitted WPE response from either **1.000–3.000 s** or **3.000–5.000 s**. Count a location when the chosen window covers at least half of the authored interval with at least 0.30 intersection over union. The two authored edits are **1.450–2.410 s** and **3.450–4.410 s**. A genuine reader counts once if any untouched or ordinary-splice version flags. Compare localized deliveries and unique edit pairs so repeated room renders do not inflate source totals. [Commercial model audit](commercial-model-audit.md)

## Results

| Route added to mono | Correct at 0 dB | Correct at −5 dB | Correct deliveries | Distinct edits | Flagged genuine readers | Flagged background excerpts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mono alone | 2/8 | 1/8 | 3/16 | 2/4 | 1/4 | 1/6 |
| WPE at its prior point | 2/8 | 1/8 | 3/16 | 2/4 | 1/4 | 1/6 |
| WPE at zero added development readers | 3/8 | 1/8 | 4/16 | 2/4 | 1/4 | 1/6 |
| WPE at one added development reader | 3/8 | 2/8 | 5/16 | 2/4 | 2/4 | 1/6 |
| WPE at two added development readers | 3/8 | 3/8 | 6/16 | 2/4 | 4/4 | 2/6 |
| WPE at the prior mono point | 3/8 | 2/8 | 5/16 | 2/4 | 2/4 | 1/6 |

All additional useful deliveries came from the same cross-reader splice. At the zero-add development point the **1.000–3.000 s** window in the lecture room at 0 dB fully covered its **1.450–2.410 s** edit, with interval IoU of 0.48. The same false reader and background excerpt flagged as in mono. At the one-add point, WPE also found this splice at −5 dB in the lecture room. It added one false reader and produced three above-point windows at the wrong edit location. The two-add point also found the same splice in the meeting room at −5 dB, while flagging every genuine reader group. Mono produced two wrong edit windows. Every held delivery had at least one speech-admitted response on each route.

The zero-add point shows a modest gain at unchanged observed false alarms, but none of the added deliveries came from a second edit. The new score point is based on three development readers. The same-reader synthetic splices showed no additional localization. The WPE view remains experimental. A larger follow-up needs exact publisher genuine/edited counterparts and more independently assigned point and assessment readers to test whether the zero-add operating point transfers. The held score evidence does not support incorporating WPE into combined AI confidence.

## Replay

The public reader independently verifies group separation, edit clocks, point commitments and scalar decisions. The local run made 96 development and 232 assessment model calls. Native replay reproduced 86 waveform commitments, verified no shared host or donor reader across roles, and found at most a one-unit PCM difference between the NumPy and SciPy WPE reconstructions. Native replay uses the pinned first 416 MiB of the publisher's part A tar delivery, the earlier local source rosters, MIT room archive, AMI recordings, and local model assets. The locked receipt publishes anonymous scalar responses and private-source commitments. [Public scalar receipt](evaluation-panels/audio-wpe-fresh-room-transfer-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_audio_wpe_fresh_room_transfer.py \
  docs/evaluation-panels/audio-wpe-fresh-room-transfer-2026-11-13.json

export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score; do
  backend/.venv/bin/python -B -m scripts.audio_wpe_fresh_room_transfer "$phase" "$run" \
    --prefix "$part_a_416m_tar" --publisher "$publisher_workdir" --aachen "$aachen_workdir" \
    --directional "$prior_directional_run" --stress-run "$room_stress_run" --ami "$ami_recordings" || exit 1
done
```
