# Short speech checks on the original recording channels

The original-channel 750 ms check added zero useful findings in a controlled two-voice comparison. All three routes located three of 36 edited files and three of 48 annotated interventions, while flagging one of 24 ordinary controls. The source-channel extension stays in development.

## Source and frozen procedure

The retained panel contains 60 six-second PCM16 stereo recordings at 16 kHz. Twenty-four declared MLS readers form twelve disjoint pairs. Six pairs use measured Aachen paths and six use measured Huddersfield paths. Each scene contributes an untouched mixture, one mixture containing two genuine same-reader splices, a synthetic edit in the quieter voice, a synthetic edit in the louder voice, and synthetic edits in both voices. These states produce five deliveries per scene. The second voice is five dB louder in the untouched mono mixture. The interventions last 960 ms with matched boundary treatment. Labels include the measured direct arrival and leave reverberant tails unresolved.

The 24 readers and native source files are disjoint from the latest 16 MLS host/donor readers. These exact recordings appeared in an earlier two-second channel comparison. They are new to this shorter method. Source exposure, repeated generator voices and unresolved training overlap limit generalization. [Prior source preparation and rights](audio-stereo-channel-replication-2026-11-15.md)

The baseline reads five two-second windows on the production decoder's canonical mono input. The shorter route reads 750 ms windows at 500 ms hops, first on canonical mono and then on each untouched original channel. Each view keeps the same speech-activity model and half-window admission rule. A short window is highlighted only when its immediate scheduled neighbor also exceeds the unchanged historical point. The final 250 ms is outside this shorter schedule. No model, head, operating point or admission threshold was changed.

The three compared routes preserve the original two-second range, add canonical mono shorter windows, then add shorter windows from both original channels. They do not separate speakers or name which speaker changed. Exact duplicate time intervals are counted once. A useful time covers half an annotated edit with at least 0.30 temporal intersection-over-union. Distinct edit clocks use one-to-one matching with the reported time intervals.

## Closed results

Each cell gives useful edited files out of 18, ordinary alerts out of 12, and edited files flagged only at an incorrect time.

| Route | Aachen | Huddersfield |
| --- | ---: | ---: |
| Canonical two-second | 1 / 0 / 0 | 2 / 1 / 2 |
| Add canonical mono 750 ms | 1 / 0 / 0 | 2 / 1 / 2 |
| Add original-channel 750 ms | 1 / 0 / 0 | 2 / 1 / 2 |

All routes miss 33 of 36 edited files at the frozen useful-location rule. The shorter routes add zero falsely flagged ordinary files and zero new useful files. The frozen continuation gate fails its detection, source-pair, quieter-voice, room and family requirements. The source-channel extension supplies no qualified change to product behavior or overall confidence.

The genuine controls contain two real voices and ordinary same-reader edits. They do not cover different-reader ordinary replacements within one voice, or isolated natural-room stereo recordings. Those negatives remain necessary before considering a new source-channel route for serving. Six-second unequal channels also remain outside the current twelve-second brief-review serving scope.

## Native and numeric verification

All 60 original stereo payloads matched the retained source bytes. An independent FFmpeg command reproduced every production mono input. The old arithmetic-mono cache differed from the current decoder by at most one PCM unit, with no clipping or gain change, so the two-second comparator was scored again.

The trial made 1,478 classifier calls, including one deterministic repeat, against at most 2,280 scheduled window requests. The independent standard-library reviewer checked all 180 view records, 2,280 source/admission/window bindings, 1,477 distinct authenticity input windows, and 2,304,000 exact sample identities linking the individual interventions to the simultaneous-edit mixture. Seven authored source/selection controls passed. It independently reproduced every count and the frozen gate.

The public capsule contains grouped decisions, counts and whole-study commitments. It omits recordings, identities, exact edit positions, individual source hashes, responses, features and file paths. The standalone checker verifies the named capsule and rejects eight authored corruptions. It recounts aggregate arithmetic and does not rerun native audio or model inference.

```sh
python3 -B -S backend/scripts/review_audio_original_channel_short_grid.py docs/evaluation-panels/audio-original-channel-short-grid-2026-11-19.json --self-test
```

Exact native/model replay needs the retained private source manifest, original recordings, room measurements and pinned local model/runtime assets. [MLS, CC BY 4.0](https://www.openslr.org/94/) · [Aachen measurements, MIT](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) · [Huddersfield measurements, CC BY 4.0](https://zenodo.org/records/2639677) · [Current classifier publisher](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector)
