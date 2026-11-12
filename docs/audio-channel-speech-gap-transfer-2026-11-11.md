# Speech coverage in individual stereo channels

Chipmunk Verify tested whether the original left or right channel can reveal speech activity in an interval the mono synthetic-voice check left unscored. A supported interval helps a reviewer decide where to listen. The measurement estimates speech presence on a recorded channel. It cannot determine who spoke, whether a voice was in the background, or whether any voice was synthetic. Neither trial met the frozen coverage requirement for advancement.

## Method

Each input was three seconds long at 16 kHz. The experiment used the app’s exact canonical mono downmix and the full-profile voice window admission rule. No authenticity classifier ran. When the mono check admitted no window, the test compared the pinned speech-activity model’s ranges on the left, right and mono signals. It removed a 50 ms guard around mono speech, retained individual-channel spans of at least 320 ms and required a measured channel level of at least −48 dBFS. The comparison cue was the existing strong downmix attenuation rule applied to one-second windows. No denoising, speaker separation or signal gain was applied.

Natural reference labels came from the AMI manual word timestamps on a 10 ms grid. A flagged range counted as supported when at least 240 ms and half the range intersected an annotated spoken word. Natural speech opportunities were mono-unscored windows with at least 240 ms of annotated speech. For controlled trials the same-clock dry foreground energy was a proxy reference. It cannot establish which speaker triggered a flag or rule out unlabelled speech from another source. Every retained range was bound to the WAV, mono PCM, individual-channel PCM and exact source sample range. Multiple ranges in a view count once toward the view-level results.

The initial trial used 60 windows with exact cut times disjoint from the earlier stereo and spatial studies. Its four meetings had previous project exposure. The confirmation used a preacquisition roster based on annotations, with four previously unused meetings and disjoint participant sets relative to nine earlier AMI meetings and each other. Both trials fixed 24 little-overlap, 20 overlap and 16 annotation-quiet views, and used the same check, labels and advancement gate. The 200 previously qualified synthetic and genuine controls were replayed for equivalence and count only once across the two trials.

## Results

| Natural panel | Views | Mono assessed | Mono-unscored speech opportunities | Supported additions | Unsupported flagged views | Quiet flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial, little overlap | 24 | 13 | 11 | 1 | 0 | — |
| Initial, overlap | 20 | 13 | 7 | 3 | 0 | — |
| Initial, annotation-quiet | 16 | 0 | 0 | 0 | 0 | 0 |
| Confirmation, little overlap | 24 | 7 | 17 | 1 | 0 | — |
| Confirmation, overlap | 20 | 11 | 9 | 4 | 0 | — |
| Confirmation, annotation-quiet | 16 | 0 | 0 | 0 | 0 | 0 |
| Total | 120 | 44 | 44 | 9 | 0 | 0 |

The incumbent attenuation cue produced zero flags on both natural panels and the controlled pool. The new measurement found four supported native windows in three initial meetings and five in all four confirmation meetings. Across both trials it found 9 of 44 mono-unscored speech opportunities, with no quiet warnings in 32 views. Each trial required six supported additions on its own. Pooling the two results does not change either decision.

In 200 distinct controlled views, the mono check admitted 173 windows and left 27 labelled opportunities unscored. The candidate flagged six views, of which four had adequate foreground reference support. The supported additions covered three source groups and two synthesis families. Two expressive-family flags had inadequate foreground support under forced kitchen and traffic conditions. There were zero flags on the 40 matched-clean views. The two controlled exceptions and the 35 missed natural opportunities matter when using the cue to direct listening. The trial did not measure any change to deepfake detection accuracy or estimate a population false-alarm rate.

## Local replay

- The frozen runner records exact source ranges, checkpoints, gate outcomes and output commitments. [Trial runner](../backend/scripts/audio_channel_speech_gap_transfer.py)
- The confirmation preparation tool regenerates the annotation-only participant and cut selection, verifies the first-party microphone acquisitions and binds the exact repeated control bytes. [Confirmation preparation](../backend/scripts/prepare_audio_channel_speech_gap_confirmation.py)
- With retained trial folders, run `python backend/scripts/review_audio_channel_speech_gap_transfer.py /path/to/trial` for each. The standard-library replay independently checked 260 views and 10 exact source intervals for the initial trial and 260 views and 13 intervals for confirmation. It recomputes admitted windows from saved activity, PCM levels, time subtraction, waveform bytes, reference overlap, counts and both decisions. It relies on frozen annotation labels and activity spans and does not rerun model inference. [Independent replay](../backend/scripts/review_audio_channel_speech_gap_transfer.py)
- The machine-readable aggregates include both unchanged decisions and SHA-256 commitments. Original audio, review clips, word annotations and individual case records are kept out of the public artifacts. [Aggregate results](evaluation-panels/audio-channel-speech-gap-transfer-2026-11-11.json)

The study used AMI meeting audio and annotations from the AMI Consortium under its published CC BY 4.0 terms. The pinned speech-activity weights are from Silero VAD under its MIT release. Spatial cues in multichannel speech activity are established in research. This experiment independently tests an app-specific coverage comparison on the recorded channels. [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [AMI signal documentation](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) · [Silero VAD license](https://github.com/snakers4/silero-vad/blob/master/LICENSE) · [Dual Microphone Voice Activity Detection Based on Reliable Spatial Cues](https://pmc.ncbi.nlm.nih.gov/articles/PMC6678508/)
