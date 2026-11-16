# Learning time-local speech-edit cues from clean recordings

This study tests whether a speech model can learn to keep a useful
response at the same clock when loud competing voices are added. The
earlier genuine-only features and clean-feature reconstruction recovered
few locations under interference. This comparison transfers the clean
model's content and boundary responses during training. Each evaluated
model receives only the submitted mixture. [Prior genuine-only results](audio-local-speech-normality-2026-11-16.md)
· [Earlier feature reconstruction](audio-short-edit-noise-projection-2026-11-15.md)
· [Clean-to-noisy distillation research, October revision](https://arxiv.org/abs/2605.23954v4)

## Fixed training and calibration

- Reuse the six-second PhonemeDF source bank and the frozen speech
  encoder. Fit one model using MeloTTS edits and one using Chatterbox
  edits. Hold the other generator out in each direction. Genuine
  splices, original speech and same-clock resampling supply ordinary
  comparisons. [Source roles and native sample preparation](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles)
- Split the six fitting readers into three pairs. Score each pair's
  clean waveforms with a teacher fitted on the other four readers.
  The clean edit's score must exceed all three matched genuine
  scores at the authored clock to contribute a training target.
  This admits 44 of 48 fitting prompts for MeloTTS and 45 of 48 for
  Chatterbox.
- Compare the earlier example labels and paired loss alone, those same
  objectives plus the clean responses at the matching clock, and a
  shifted control that moves the clean targets by 1.10 seconds. The
  two teacher arms learn deviations from the per-clip mean for both
  content and boundary scores. They give four times the weight to
  the labeled interior and joins respectively.
- Set each arm's narrow and wide operating points using separate
  genuine noisy files, with the highest isolated conversation score
  as a lower bound. The three arms use the same speech admission,
  500–1,000 ms window lengths and exact 16 kHz sample clock. A useful
  range covers half of a known edit and reaches 30% temporal
  intersection over union. Report mistakes at other times and
  look for alerts at the same clock in matched genuine controls.

PhonemeDF's repository record declares CC BY 4.0. The pinned encoder's
model card points to CC BY-SA 3.0. The newly selected SINE inputs have
an Apache-2.0 publisher declaration and the conversation excerpts use
the AMI corpus terms. The release contains the project source, scalar
measurements and source commitments. Fitted weights, voice
representations, and waveforms stay local. [PhonemeDF metadata](https://zenodo.org/api/records/19522979)
· [Pinned encoder terms and digests](audio-wavlm-v13-protocol.md)
· [SINE license](https://huggingface.co/datasets/PeacefulData/SINE#license)
· [AMI corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held and fresh outcomes

The first evaluation reuses 346 six-second deliveries from the
earlier noisy speech and stereo panels. These include edited
Chatterbox, MeloTTS, SINE, Piper and Kitten speech and matched ordinary
controls. The stereo rows use the exact mono mix of two microphone
channels, with edited words in either or both voices.
[Stereo source preparation](audio-stereo-channel-replication-2026-11-15.md)

The table selects the already fixed wider point for each fallback.
The current check has priority when it supplies a range. Each cell
gives useful edited locations followed by ordinary foreground alerts.
The stereo and SINE entries use the Chatterbox-trained fit because
this was the arm with the first room gain. Both fits and both
operating points are in the receipts.

| Held panel | Current | With paired labels | With aligned clean teacher | With shifted teacher |
| --- | ---: | ---: | ---: | ---: |
| Held PhonemeDF generators, both directions summed | 4/48 · 64/144 | 5/48 · 68/144 | 5/48 · 67/144 | 5/48 · 70/144 |
| Aachen and Huddersfield simulated rooms | 3/36 · 1/24 | 5/36 · 4/24 | 7/36 · 2/24 | 10/36 · 8/24 |
| Prior SINE reader pairs | 1/53 · 16/53 | 1/53 · 16/53 | 1/53 · 17/53 | 1/53 · 18/53 |
| Separate BBC room replay | 2/36 · 0/24 | 3/36 · 0/24 | 2/36 · 0/24 | 2/36 · 0/24 |

The extra PhonemeDF location from the aligned teacher also flags a
matched ordinary edit at the same clock. The aligned fallback adds two
isolated-conversation alerts out of 48 PhonemeDF room decisions. The
four useful Aachen and Huddersfield detections come from three scenes,
with a clear matched control at each edit clock. In one Huddersfield
delivery the selected **1.800–2.300 s** lies inside the planted
**1.476–2.436 s** edit. The larger gain from the shifted target came
with seven additional ordinary alerts, so increased detections alone
give little evidence that matching the teacher clock caused the gain.
The BBC follow-up excludes four scenes whose declared readers appeared
in fitting or calibration. It uses 12 other previously recorded
reader scenes and the same frozen checkpoints. The aligned method
adds zero useful locations there. [First held receipt](evaluation-panels/audio-clean-teacher-transfer-2026-11-16.json.gz)
· [Separate room receipt](evaluation-panels/audio-clean-teacher-room-2026-11-16.json.gz)

An additional 12 SINE readers supply 54 newly prepared deliveries.
The reader set excludes the declared readers from the used PhonemeDF
and earlier SINE rosters. Six retained AMI excerpts provide the same
background samples for each matched pair at each level. The room
conversation has five or ten dB more total RMS level than the
unmodified foreground. Cross-corpus reader aliases and model
pretraining exposure remain unknown.

The app and all six candidate heads miss every new labeled edit at
both noise levels. The app flags 3/12 originals at five dB and 2/12
at ten dB. The aligned heads produce zero ordinary or isolated-room
alerts, with zero edit locations. Their best proposals overlap 3/12
and 1/12 edits at the five and ten dB levels for the Chatterbox fit,
and 1/12 and 2/12 for the MeloTTS fit, before applying the frozen
operating points. The strongest candidate remains in evaluation.
[Fresh reader receipt](evaluation-panels/audio-clean-teacher-fresh-2026-11-16.json.gz)

## Replay

The portable numeric review runs without the speech model and uses each
receipt's exact PCM commitments, sample clocks, calibration responses,
and paired controls. Both Python 3.12 and 3.14 produced the same counts.
A separate sample-set recount agreed on all 240 model, panel and
decision rows. The native refit reproduced all six weight files and
calibration logs byte for byte, and the second fresh build reproduced
its roster and all 54 PCM inputs. The study covers 460 deliveries and
445 distinct waveforms.

```sh
export PYTHONPATH=backend
module=scripts.audio_clean_teacher_transfer
panels=docs/evaluation-panels
python3 -B -S -m "$module" review "$panels/audio-clean-teacher-transfer-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/audio-clean-teacher-fresh-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/audio-clean-teacher-room-2026-11-16.json.gz"
```

Rebuild the `phoneme`, `sine`, `additional` and `stereo` trees using the linked
[earlier source procedures](audio-local-speech-normality-2026-11-16.md#independent-replay).
Rebuild `prefixes` from the six pinned SINE range files in the
[input-view transfer](audio-short-edit-input-views-2026-11-15.md).
Use the verified stereo source directory for the
[two-voice BBC study](audio-two-voice-components-2026-11-15.md) as `bbc`.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=2
runner=backend/.venv/bin/python
encoder=models/research/wavlm-v13
weights=models/weights/audio-upgrade
current="$panels/audio-local-speech-normality-2026-11-16.json.gz"
"$runner" -B -m scripts.audio_short_edit_ssl_transfer cache "$phoneme" "$run/cache" "$encoder"
"$runner" -B -m "$module" fit "$phoneme" "$run/cache" "$weights/silero_vad.jit" "$current" \
  "$sine/freeze.json" "$additional/freeze.json" "$stereo/freeze.json" "$run/fit"
"$runner" -B -m "$module" score "$phoneme" "$run/cache" "$weights/silero_vad.jit" "$current" \
  "$sine" "$additional" "$stereo" "$encoder" "$run/fit" "$run/held.json.gz"
"$runner" -B -m scripts.audio_clean_teacher_fresh build "$phoneme" "$additional" "$sine" "$prefixes" "$run/fit" "$run/fresh"
"$runner" -B -m scripts.audio_clean_teacher_fresh score "$run/fresh" "$run/fit" "$weights" "$encoder" "$run/fresh.json.gz"
"$runner" -B -m scripts.audio_clean_teacher_rooms "$phoneme" "$bbc" "$run/fit" "$run/bbc"
"$runner" -B -m scripts.audio_clean_teacher_fresh score "$run/bbc" "$run/fit" "$weights" "$encoder" "$run/bbc.json.gz"
```
