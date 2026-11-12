# Stereo voice rescue with matched sequential and overlapping speech

The fixed physical-channel rule selected no channel in 101 held views. The current mono detector flagged 16 of 32 genuine speech mixtures under strong recorded interference. Seven false flags came from sequential background speech and nine from overlapping background speech. The controlled stereo rescue had recovered six synthetic flags under deliberate imbalance. The new natural-room comparison found no additional synthetic flags. [Earlier held result](audio-native-stereo-holdout-v284-result.md) · [New audited aggregate](evaluation-panels/audio-paired-stereo-overlap-2026-11-11.json)

## Matched inputs

Select two AMI meetings with documented participants disjoint from the earlier known meeting groups. Each room supplies four paired three-second cuts using the same two main annotated participants in each pair. Both participants have at least 350 ms of annotated words and at least one participant is speaking in at least 2.45 seconds. The sequential cut contains at most 100 ms of simultaneous annotated words. Its counterpart contains at least 900 ms. The cut positions and pair assignments are based on annotations and fixed hashes. The native audio comes from two aligned microphone files. The first room's publisher files store the same physical channel twice in each WAV. The entire file was checked for exact sample equality before using one column. [AMI audio sources](https://groups.inf.ed.ac.uk/ami/download/) · [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Use eight genuine readers and nine synthetic voices, with three groups each from clone, expressive, and stock-voice sources. Each foreground appears clean and in one paired sequential and overlapping mixture from each room. The same exact centered speech samples recur in the two members of a pair. Scale each recorded interference track to −5 dB relative to the foreground in the channel average. Preserve the relative balance of the physical microphone samples and apply one clipping-safe gain to all views of a foreground. Include the 16 unchanged room cuts as genuine controls. The voices had prior project exposure, and the earlier forced-imbalance comparison used the same source bank. The new room groups and exact mixtures are separate. [Foreground and license review](audio-short-edit-source-plan-v169.md)

Use the fixed 512-sample speech-activity blocks and the unchanged acoustic channel selection. A channel requires support from at least eight active and eight quiet blocks and a 3 dB contrast advantage. Add an AI flag only if the selected physical channel is speech-admitted and leans AI while the original mono is inconclusive or unassessed. The gate and source replay check exact mono and physical sample hashes. Advancement requires gains on at least three distinct synthetic groups and two families, gains under both interference conditions, at most one added genuine false flag overall, zero added clean or native false flags, and at least five assessed genuine mixtures per room-condition cell. [Fixed gate](audio-native-stereo-gate-v279-method.md) · [Fixed decision policy](audio-ai-only-rescue-v283-method.md)

## Local results

| Condition | Genuine false AI / assessed | Correct synthetic AI / assessed | Synthetic inconclusive |
| --- | ---: | ---: | ---: |
| Clean | 0/8 | 9/9 | 0 |
| Room 1, sequential | 2/8 | 4/9 | 5 |
| Room 1, simultaneous | 3/8 | 7/9 | 2 |
| Room 2, sequential | 5/8 | 8/9 | 1 |
| Room 2, simultaneous | 6/8 | 6/9 | 3 |
| Untouched microphone recordings | 0/12 | — | — |

Four of the 16 untouched native cuts remained unassessed. All mixed and clean views were assessed, and no synthetic clip leaned original. The channel rule found too few supported active or quiet blocks in 64 views. The remaining 37 had less than 3 dB contrast between channels. The fixed rule added zero AI flags and did not meet the advancement criteria.

The local study made 28,482 speech-activity frame forwards and 112 classifier forwards, including existing stability checks. A separate standard-library audit verified 101 stereo and mono sample commitments, re-read all 16 microphone cuts, regenerated all 68 mixtures, and recalculated the original and alternative directions. Only aggregates and whole-artifact commitments are published.

## Source verification in the app

For new and previously frozen current-format evidence, `verify --source-audio` recomputes measured stereo channel levels, correlation and mono-downmix attenuation against the supplied original recording or video. It checks the same bounded source windows shown in the report. A standard signed ZIP review can still check the committed evidence without the original. The source replay covers acoustic evidence and carries no AI confidence weight. [Verification instructions](independent-testing.md#verify-audio-clips-and-timestamps-from-a-signed-export)

## Reproduce

Supply the four publisher microphone WAVs named in the aggregate receipt and the pinned AMI v1.6.2 annotation ZIP. The runner verifies the retained foreground bank and uses the installed voice and speech-activity assets.

```sh
export PYTHONPATH=backend
python -m scripts.audio_paired_stereo_overlap freeze "$run" --ami "$ami" --annotations "$annotations"
python -m scripts.audio_paired_stereo_overlap gate "$run"
python -m scripts.audio_paired_stereo_overlap classify "$run"
python -m scripts.audio_paired_stereo_overlap readout "$run" --ami "$ami"
python backend/scripts/review_audio_paired_stereo_overlap.py "$run" "$ami" backend/app/detectors/audio/calibrations/local-speech-v3.json
```
