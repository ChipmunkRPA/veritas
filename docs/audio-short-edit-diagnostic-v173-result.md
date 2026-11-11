# Ordinary speech edits reproduce false AI flags on fresh development audio

The guarded local CPU run completed all 79 cases from 37 development recordings
in 96.81 seconds. It made 156 speech-model forwards, with 2.73 GiB maximum
sampled process-tree RSS and no guard or sampling errors. Seventy-eight cases
returned a speech score. One expressive synthetic recording was unavailable.
[Frozen method](audio-short-edit-diagnostic-v173-method.md) · [Verified aggregate](evaluation-panels/audio-short-edit-diagnostic-result-v173.json)

| Development arm | Cases | Existing AI flags | AI flags after band check | AI flags after repetition abstention |
| --- | ---: | ---: | ---: | ---: |
| Untouched genuine | 14 | 0 | 0 | 0 |
| Genuine hard copy | 14 | 7 | 5 | 0 |
| Genuine copy, 20 ms fade | 14 | 7 | 7 | 0 |
| Genuine copy, 50 ms fade | 14 | 7 | 7 | 0 |
| Untouched cloned speech | 5 | 5 | 5 | 5 |
| Untouched expressive synthetic speech | 12 | 9 | 7 | 9 |
| Untouched stock synthetic speech | 6 | 6 | 6 | 6 |

The same 14 genuine recordings, from seven groups, supply all four genuine
arms. Across the three copy variants, ten recordings were falsely flagged
at least once and five were falsely flagged in every variant. The exact
repetition cue located the known copy in all 42 edited versions. The
approximate cue located it in 33 versions and reported one span on an
untouched expressive recording. All 21 false AI decisions in the edited
genuine arms became inconclusive under the repetition comparator.

The band check removed two hard-copy false flags and two correct flags
on expressive synthetic speech. The repetition comparator preserved all
20 existing AI flags across the 23 fully synthetic controls. Two synthetic
controls remained inconclusive and one remained unavailable. None leaned
original. These are paired development observations on clean 3-second WAVs.
The test leaves mixed-origin speech, strong background noise and codec
transfer of the comparator unmeasured. No holdout audio was opened for this
diagnostic and no customer-facing rule or confidence changed.
[Source qualification and limits](audio-short-edit-materialization-v169c-result.md) · [Codec evidence study](grid-audio-repetition-codecs-v170-result.md)
