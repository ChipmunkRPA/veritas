# Measuring speaker count before using a noisy-voice supplement

A local diarization gate preserved the gain of the unpaired layer-relation score on the separate room test. It mistook six of fourteen controlled three-voice mixtures for two voices. That error prevents confidence routing. [Frozen method and complete counts](evaluation-panels/audio-measured-speaker-routing-2026-11-11.json)

## Pinned local comparison

NVIDIA’s speaker activity model supplies time ranges for concurrent anonymous tracks. The test used the official quantized checkpoint and Apple Silicon CPU release, both hash-checked before inference. The checkpoint’s OpenMDW 1.1 terms permit commercial use and require notices when distributing the model materials. The repository contains the runner and aggregate results. The exact model stayed local. [Checkpoint and model card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/52a4326/Nemotron-3-Diarization.q8_0.gguf) · [License](https://openmdw.ai/license/1-1/) · [Runtime release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.3.0) · [Research paper](https://arxiv.org/abs/2611.12246)

On each three-second mono input, independently measure two diarization operating points in full-attention mode. The single-interferer route requires two substantial tracks and at least 240 ms of two-voice activity, with at least 180 ms continuously. Any three-voice activity lasting 160 ms in total marks overlap. The corroborated route admits the single-interferer case only when both operating points support it. Any overlap observation closes that route. The paired score test reuses the two previously frozen raw and unpaired-relation room points, allowing only an incumbent inconclusive result to become an AI flag. Reference acoustic labels stay separate from measured routes. [Prior room results and source terms](audio-cross-layer-room-transfer-2026-11-11.md)

Replay all 118 exact prior room views, with 59 from each source role. The publisher trained the diarization model on AMI training and development audio, and the natural-room checks use AMI recordings. They therefore measure task behavior on retained audio, with possible diarizer training exposure. The separate controlled panel makes 28 new waveforms from seven nonoverlapping triplets of genuine readers already retained in the room studies. Select triplets with at least 400 ms of joint source speech on separate original channels under the existing activity check. Each triplet yields one-, two- and three-speaker mixes, including a three-speaker mix with a weaker third voice. No role was refitted or selected from diarization outputs.

## Local results

| Reference condition | Views | Single-interferer route with one point | With both points |
| --- | ---: | ---: | ---: |
| Separate-room, mostly one interfering voice with foreground | 19 | 19 | 18 |
| Separate-room, overlapping interferers with foreground | 19 | 2 | 0 |
| Room point-selection, mostly one interfering voice with foreground | 19 | 18 | 16 |
| Room point-selection, overlapping interferers with foreground | 19 | 3 | 0 |
| Controlled two-source mix | 7 | 5 | 5 |
| Controlled three sources, equal-power background voices | 7 | 5 | 3 |
| Controlled three sources, quieter third voice | 7 | 4 | 3 |
| Controlled one-source input | 7 | 0 | 0 |

The separate-room corroborated route let the unpaired relation score add seven true AI flags with zero added genuine flags. The same route let the ordinary raw score add five true flags with zero added genuine flags. This changes the mostly-single-interferer scene from one to eight of eight detected synthetic clips. The one genuine flag already present in the overlapping scene remained. The single-point route added one genuine flag in each noisy scene. One overlapping background-only case from point selection still entered the corroborated route, without an added flag.

The corroborated gate missed the third voice in six of fourteen constructed three-source views. This failed the frozen controlled check even though the separate-room replay passed its detection and false-flag requirements. More sensitive segments reduced some misses. The remaining misses matter because a genuinely overlapping scene admitted by mistake could receive the unreliable supplement.

The CPU run completed 292 diarization forwards on 146 exact inputs. A separate sample-grid calculation reproduced all 292 interval measurements and all 146 PCM hashes. The portable reviewer reproduced the seven gained synthetic flags and zero added genuine flags on the separate-room panel. Its authored example also demonstrates how to detect a routing mistake without downloading weights. [Portable reviewer](independent-testing.md#review-an-audio-supplement-by-acoustic-condition)

The [local runner](../backend/scripts/audio_measured_speaker_routing.py) takes a new directory containing `manifest.json`, `pcm/<three-digit-id>.pcm`, the pinned `runtime.tar.gz` and `model.gguf`, and the extracted runtime binary as a positional argument. It checks exact sample lengths and source hashes before execution. The manifest records panel, acoustic reference condition, source group, truth, incumbent direction, previously frozen score-point outcomes and PCM checksum for every case. The three commands are `freeze RUN_DIR BINARY`, `run RUN_DIR BINARY`, and `readout RUN_DIR BINARY`. Audio and individual speaker timings remain with the evaluator.

A second run verified the executable against the corresponding pinned archive member and repeated all 292 forwards. It reproduced the individual speaker measurements and all aggregate counts. The published receipt retains both freeze hashes.
