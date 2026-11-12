# Overlapping speech windows on publisher-labeled partial edits

The long speech-window review currently checks two or three disjoint
three-second portions. A short edit can fall in a gap, or span too little
of either window to produce two AI-like responses. This comparison
asks whether denser windows can recover useful locations, especially
under loud background sound. The experiment uses the existing voice
calibration. A narrower shared interval is an inspection location and
carries zero weight in the overall AI confidence score.
[Current long-window evidence](audio-radar-transfer-2026-11-12.md#local-evidence) ·
[Multi-crop and condition-specific research](https://arxiv.org/abs/2608.23437) ·
[Out-of-domain partial speech localization](https://arxiv.org/abs/2507.03468)

## Frozen input and decision rules

The CC BY 4.0 LlamaPartialSpoof release includes native speech, complete
synthetic speech and publisher-authored partial speech edits with segment
labels. The test reads a pinned 96 MiB prefix of the first published tar
and checks all 474 complete member names against 76,228 publisher labels.
The retained 28 sources use distinct publisher source-speaker IDs.
They consist of 12 partially edited clips (two per generation method),
six entirely synthetic clips (one per method) and ten genuine clips.
The genuine set contains English read speech. Source IDs identify
underlying source speakers and some generation methods reuse a voice.
Prior training exposure is unknown. The prefix is a convenience sample.
[Publisher release and terms](https://zenodo.org/records/14214149) ·
[Source construction and generator details](https://arxiv.org/html/2409.14743)

Selection uses only publisher metadata and exact WAV input bytes, with
SHA-256 ordering within each source class. Partial clips contain 10–60%
labeled synthetic speech and at least one edited interval of 0.6–2.4 s.
The original labels can contain more than one edited interval. Convert
publisher times to the nearest 16 kHz sample and reconcile the last
boundary to the exact native length only within 1 ms. Each voice supplies
a gain-only view and two mixtures at −5 dB foreground-to-added-background
RMS. Add the same peak-safe gain across its three views. The traffic and
café excerpts come from two recorded scenes used in previous project
experiments. Eight isolated excerpts are negative controls. Excerpts from
one scene can overlap and do not count as separate noise recordings.
A sample-by-sample check independently measured the rendered ratio within
0.000011 dB of the target on all 56 mixtures.
[Traffic source and credit](https://commons.wikimedia.org/wiki/File:Audio_sound_of_a_busy_traffic_in_Porthacourt.ogg) ·
[Café source and credit](https://commons.wikimedia.org/wiki/File:Cafe_ambiance.ogg)

The sparse comparison runs the existing long-review sampling and rule on
the exact decoded PCM. It supplies evidence only with at least two
AI-leaning disjoint responses and no original-leaning response. At the
time of the comparison, the released long-review route accepted qualifying
FLAC. This experiment measures the rule on the publisher WAV samples and
their authored mixtures.
The candidate checks three-second windows every 1.5 seconds and reports the
1.5-second intersection when both adjacent responses lean AI. These
overlapping responses share waveform samples. A final tail shorter than
the hop can lie outside the last dense window. Test both the dense rule
alone and its additive use with the sparse rule.

A useful partial location covers at least half of one labeled synthetic
interval and reaches at least 0.30 intersection over union with that same
interval. Any flag on a fully synthetic control counts as detection. The
frozen rule requires at least four new useful partial source groups under
noise across at least three generator families, at most one additional
noisy genuine group and zero additional clean genuine or isolated flags.
It also preserves existing useful clean and fully synthetic detections.
The study uses the pinned, commercially reviewed local voice assets.
[Checkpoint terms](commercial-model-audit.md)

## Local results

| Delivery | Useful partial locations, sparse / dense / combined | Fully synthetic flagged, sparse / dense / combined | Genuine flagged, sparse / dense / combined |
| --- | ---: | ---: | ---: |
| Gain-only | 5 / 10 / 11 of 12 | 6 / 6 / 6 of 6 | 0 / 0 / 0 of 10 |
| Traffic, −5 dB | 2 / 5 / 5 of 12 | 1 / 4 / 4 of 6 | 2 / 3 / 4 of 10 |
| Café, −5 dB | 0 / 0 / 0 of 12 | 0 / 0 / 0 of 6 | 0 / 0 / 0 of 10 |

The additive rule recovered three additional partial source groups under
traffic across three generators and flagged two additional genuine groups.
The dense-only rule also lost one useful gain-only location. Both fell
short of the frozen advancement rule. Eight isolated-background excerpts
produced no flags. Under café sound, only three partial clips, two fully
synthetic clips and zero genuine clips had two adjacent assessed windows.
The zero café flags therefore include many recordings without enough usable
speech for the candidate rule. Under traffic, the corresponding counts
were ten, six and four.

An interval-level readback found that the dense rule reported 28 intervals
on gain-only partial clips and 11 under traffic. Fourteen and six,
respectively, did not satisfy the location criterion for any labeled edit.
The sparse rule reported 17 and eight, with 11 and six below the location
criterion. Even with every candidate window positive, the available time
geometry could reach only 18 of 54 labeled edits with dense intersections
and nine with sparse windows. The gain-only result supports studying a
separate clean-speech procedure. Both rules stay experimental for local
edit timing under loud sound.

A second preparation reproduced the exact frozen roster and all 92
input hashes. An independent standard-library reader verified 556
native window hashes against those PCM inputs and recomputed the paired
results. It replays public outcomes without the audio or model.
[Public decision receipt](evaluation-panels/audio-partial-publisher-transfer-2026-11-12.json)

## Replay

From the repository root, use the publisher metadata and the two pinned
ambient recordings. Use a fresh scratch path for `run`. Download only
the bounded archive prefix and retain all media in scratch storage. The
speech assets are the current installed local candidate weights.

```sh
publisher=https://zenodo.org/records/14214149/files
source_dir=$(mktemp -d)
curl -fLsS --range 0-100663295 --max-filesize 100663296 \
  "$publisher/R01TTS.0.a.tgz?download=1" -o "$source_dir/part-a-96m.prefix.tar"
curl -fLsS --max-filesize 12000000 "$publisher/label_R01TTS.0.a.txt?download=1" \
  -o "$source_dir/part-a.labels.txt"
curl -fLsS --max-filesize 25000 "$publisher/LICENSE.txt?download=1" -o "$source_dir/LICENSE.txt"
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VERITAS_BILLING_MODE=disabled VERITAS_DATA_DIR="$source_dir/app-state"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_partial_publisher_transfer prepare "$run" \
  --source "$source_dir" --background "$backgrounds"
backend/.venv/bin/python -B -m scripts.audio_partial_publisher_transfer score "$run"
python3 -I -S backend/scripts/review_audio_partial_publisher_transfer.py export "$run" "$receipt"
```

```sh
python3 -I -S backend/scripts/review_audio_partial_publisher_transfer.py review \
  docs/evaluation-panels/audio-partial-publisher-transfer-2026-11-12.json
```
