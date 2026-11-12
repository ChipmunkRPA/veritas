# Distinct voice support after three-stream separation

The compound stability check supported 37 of 52 controlled three-voice recordings and flagged none of the 26 controlled two-voice recordings. It supported none of 18 naturally overlapping three-voice recordings from three AMI meetings. The earlier energy-only rule flagged every two-voice control in both panels. [Frozen methods, exact aggregate and independent replay](evaluation-panels/audio-stream-presence-transfer-2026-11-11.json)

## Fixed comparison

Run the [previously verified Apache-licensed separator](audio-separation-third-voice-pilot-2026-11-11.md) on the same three-second waveform three times. Use the exact signal once, then prepend 40 ms or 90 ms of silence with fixed short trailing padding. Read back the same original three-second interval each time. Match the anonymous outputs across views by maximizing total absolute sample-aligned correlation. The weakest matched output sets the stability value. Apply the pinned speech-activity check on the original three outputs and require 160 ms of three-way activity and each output to carry at least 4% of the input RMS. An independent signal measure estimates the least energy left in one output after projecting it onto the other two. The compound response takes the smaller of alignment stability and this unexplained fraction. It measures signal consistency and local redundancy. A different voice can be unstable, and a repeated voice can contain portions with limited waveform correlation. [Related work on speaker tracking and activity](https://arxiv.org/abs/2610.05693)

The eight previously scored separator triplets set three private decision points using the second-highest maximum over the two-voice controls, with one maximum per source group. The exact schedule and source commitments were sealed before the development pass. The points were sealed before the held pass. The held constructed cases use the remaining 13 disjoint VoxPopuli triplets and include dry and reflected deliveries with a quieter and fainter third voice. These sources had earlier speaker-count evaluation and no earlier separator-pilot exposure. The criterion required the compound check to detect at least half of the three-voice inputs and flag at most one sixth of the two-voice controls in both held panels. [Parent controlled-source study](audio-hard-overlap-transfer-2026-11-11.md)

The independent natural panel uses three first-party AMI microphone recordings acquired for this comparison. The annotation-only selection excludes participants from six earlier AMI rooms and requires distinct participants across its three selected meetings. It takes six three-second two-voice and six three-second three-voice windows per meeting. Each reference voice is annotated for at least 350 ms, no additional annotated participant exceeds 80 ms, and the required simultaneous word activity lasts at least 450 ms. A separate sample-grid audit found actual minima of 460 ms and 490 ms. Nonverbal sounds, unannotated speech, and possible model pretraining exposure remain outside the reference. The analysis uses original, unmodified mono microphone samples. The publisher releases the corpus and annotation under CC BY 4.0. The evaluation keeps raw recordings, individual source metadata, activity measures and separated outputs local. [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [AMI ethics and consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)

## Held results

| Method | Controlled three-voice support /52 | Controlled two-voice flags /26 | Natural three-voice support /18 | Natural two-voice flags /18 |
| --- | ---: | ---: | ---: | ---: |
| Raw 10% RMS per output | 52 | 26 | 18 | 18 |
| Three-way speech activity | 23 | 3 | 0 | 0 |
| Aligned output stability | 37 | 1 | 1 | 0 |
| Stability plus unexplained energy | 37 | 0 | 0 | 0 |

The activity condition was available in 52/52 controlled three-voice views and 24/26 controlled two-voice views, compared with 12/18 and 5/18 in the natural recordings. The controlled known-reference audit recovered the third source in 40 of 52 cases. Of these, 34 also passed the compound cue. On the 26 controlled two-voice inputs the unmatched channel largely matched one known voice in three cases and a linear mixture of both known voices in one. In the other 22 cases, neither reference explanation accounted for half its energy. This last category includes nonlinear distortion or fragmented speech and supplies no evidence for an additional talker. The natural result failed the frozen criterion. The speaker overlap supplement remains experimental.

The CPU run completed 144 development and 342 held separator forwards. A separate reader verified all 162 PCM bindings, the 36 natural source clocks, the exact development and confirmation source roles, the point fit and nine condition summaries. Rebuilding the annotation plan with the published selector reproduced the frozen plan byte for byte.

## Comparing two- and three-stream models

A separate fixed comparison uses two independently trained versions of the same separator architecture. The new two-stream version is the Apache-licensed Libri2Mix checkpoint pinned to `eb43c5bfbb2aa654630adbf849373bcec0a20ed4`. Run both on the exact same native-aligned input. Match its two outputs to the best distinct pair of three outputs, then measure how much of the unmatched third output remains after fitting both of the two-stream outputs. Express the residual in units of input RMS. The models can generate their own signal scales, so this ratio can exceed one. The qualified variant additionally requires both matched correlations to reach 0.60 and the unmatched output to contain at least 480 ms of independently measured speech activity. The source labels enter only point selection and evaluation. [Two-stream checkpoint and license](https://huggingface.co/speechbrain/sepformer-libri2mix/tree/eb43c5bfbb2aa654630adbf849373bcec0a20ed4) · [Frozen paired comparison and verified counts](evaluation-panels/audio-two-three-separator-transfer-2026-11-11.json)

The 16 previously used two-voice development controls set the second-highest maximum across eight groups separately for each arm. The follow-up protocol was frozen before reading the first experiment's held results. It replays the same 114 held inputs as a different measurement, with the same advancement criterion.

| Paired model method | Controlled three-voice support /52 | Controlled two-voice flags /26 | Natural three-voice support /18 | Natural two-voice flags /18 |
| --- | ---: | ---: | ---: | ---: |
| Novel residual alone | 20 | 3 | 5 | 3 |
| Novel residual with speech and matching | 19 | 3 | 3 | 1 |

The qualified comparison left five three-voice and seven two-voice natural windows unassessed. It recovered three natural three-voice cases, with one additional two-voice flag. Both methods fell below the required three-voice coverage. The study completed 32 development and 228 held model forwards. A separate readout checked all 130 input bindings, both point fits and the decision arithmetic. These checks remain experimental.

## Reproduce locally

Use the exact [parent overlap panel](audio-hard-overlap-transfer-2026-11-11.md), the [pinned separator assets](audio-separation-third-voice-pilot-2026-11-11.md) and the pinned speech activity model. Generate the natural plan from the official AMI manual annotation archive. Its file commitment is in the selector. Obtain the declared native recordings from their publisher URLs and place them in `RUN_DIR/natural-raw` under the plan's meeting IDs with `.wav` appended. Compare their hashes with the published freeze before inference. Follow the same isolate and CPU version setup as the separation pilot.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -m scripts.prepare_natural_overlap_transfer "$annotation_zip" "$run/natural-source-plan.json"
python -m scripts.audio_stream_presence_transfer freeze "$run" "$parent" "$separator_assets" "$vad"
python -m scripts.audio_stream_presence_transfer fit "$run" "$parent" "$separator_assets" "$vad"
python -m scripts.audio_stream_presence_transfer confirm "$run" "$parent" "$separator_assets" "$vad"
```

Run the [model-free verifier](../backend/scripts/review_stream_presence_transfer.py) in isolated Python to compare the public summary with the local receipts. It validates the frozen points and decision arithmetic without loading weights or running inference.

```sh
python3 -I backend/scripts/review_stream_presence_transfer.py "$run" "$parent" \
  --published docs/evaluation-panels/audio-stream-presence-transfer-2026-11-11.json
```

For the paired test, add the pinned two-voice checkpoint to a local directory with the filenames and digests in the [paired runner](../backend/scripts/audio_two_three_separator_transfer.py). Pass a new directory for its outputs.

```sh
for phase in freeze fit confirm; do
  python -m scripts.audio_two_three_separator_transfer "$phase" "$paired_run" \
    "$run" "$parent" "$separator_assets" "$two_voice_assets" "$vad" || break
done
```
