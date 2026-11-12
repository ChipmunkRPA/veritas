# Comparing retained and removed stereo speech

The new stereo check recovered two synthetic flags under simultaneous competing speech and added one genuine false flag. All three changes came from the same held room. It also lost a correct flag under sequential interference and cleared none of the incumbent false flags. The method stays experimental. [Exact aggregate and independent replay](evaluation-panels/audio-removed-source-transfer-2026-11-12.json)

## Fixed comparison

Earlier stereo interference cancellation scored the submitted mono audio and a view intended to retain speech near the stereo midpoint. This test also scores the exact sound subtracted by that filter. It asks whether an AI response is stronger in the retained view or the removed view before changing a result. A high response on the removed view can reflect competing speech, processing artifacts, or target leakage. It supplies a relative diagnostic cue. It does not identify a speaker or establish which voice was generated. [Earlier stereo results and source lineage](audio-spatial-interference-2026-11-11.md) · [Environment-aware audio detection benchmark](https://arxiv.org/abs/2606.10791v2)

For a source window with distinct physical channels and existing assessed original and projected views, subtract the unrounded projected samples from their exact stereo mean. Require the removed signal's RMS to reach 10% of the original mono RMS before normalization. Normalize toward the original RMS with a gain cap of six and a peak cap of 31,129 in signed 16-bit PCM. Apply the current full-profile speech gate on this new three-second view. Score the removed waveform only if it passes. The pinned current model and its calibration are reused. [Checkpoint terms](commercial-model-audit.md)

The rescue rule requires an incumbent inconclusive result and a projected AI direction. It promotes the result when the projected raw log odds sufficiently exceed those of the removed waveform. The withholding rule requires incumbent and removed AI directions and a sufficiently stronger removed response. Its result is inconclusive. Each rule uses the higher of ln(2) and the maximum unsafe transition margin in the separate selection role, with strict excess. The selected points protect genuine recordings from an added AI flag and synthetic recordings from a lost AI flag in that role. Only two genuine foreground groups contributed eligible rescue controls and four synthetic groups contributed withholding controls. The projection-only comparison applies both original filter decision rules to the exact same files. Original and unassessed directions remain in place for every arm.

## Independent source roles

Reuse the exact inputs from the prior two-channel cancellation benchmark. The 101-view selection role has 17 distinct foreground groups, 68 centered mixtures, and 16 unchanged room recordings. The held role uses two different AMI meetings, seven genuine groups and nine synthetic groups from clone, expressive and stock families. The roles share no recorded foreground group or exact clean PCM. Both roles had earlier project exposure. The new removed-signal scores are measured here. The 116 held views include 64 centered and 32 displaced mixtures, 16 clean voices and four unchanged native room controls. Displaced speech has a four-sample delay and 3 dB attenuation in one microphone. The prior filter was designed for centered speech. These inputs test source attribution under a controlled capture geometry. [Foreground qualification and rights](audio-short-edit-source-plan-v169.md) · [AMI corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Local results

| Held input | Current true / false AI flags | Projection only | Removed-source comparison |
| --- | ---: | ---: | ---: |
| Centered, 36 synthetic and 28 genuine | 18 / 6 | 18 / 10 | 19 / 7 |
| Displaced, 18 synthetic and 14 genuine | 3 / 1 | 4 / 2 | 2 / 1 |
| Clean, 9 synthetic and 7 genuine | 9 / 0 | 9 / 0 | 9 / 0 |
| Untouched native, 4 genuine | — / 0 | — / 0 | — / 0 |

On centered inputs, the new rule recovered one clone and one stock synthetic flag and added one false flag on a genuine recording with simultaneous interference. The one lost centered clone flag and one lost displaced clone flag occurred under sequential interference. All five changes occurred in the same held room. The removed view was assessed for 46 of 64 centered and 31 of 32 displaced mixtures. No clean or native held control had an assessed removed view, so those rows test fallback. The method passed four of five frozen criteria. It missed the requirement to clear at least two incumbent false flags.

An independent acoustic replay applied the filter fitted on each mixture to its known foreground and interference separately. The removed waveform preserved numerical zero foreground leakage for the centered construction. The median removed-to-original foreground RMS fraction was 27.1% for the 31 checked displaced views, reaching 42.1% in one case. The removed waveform can contain appreciable target speech when capture geometry differs. This limits any use as a general stereo attribution check. The centered simultaneous result supports a larger test with known geometry and separate rooms. It does not establish a serving rule for arbitrary stereo files or mono uploads.

The local CPU stages made 126 new removed-waveform classifier forwards and 14,570 speech-activity frame forwards. They took 83.1 seconds and peaked at approximately 2.98 GB process RSS. The no-inference replay rehashed all 217 exact source views, rebuilt 155 normalized removed waveforms and reproduced 21 decision cells and the five advancement checks. One previously documented half-integer projected sample required the pinned SciPy rounding convention. Raw recordings and per-window responses stay local.

## Reproduce locally

Use the completed spatial-interference comparison and its retained paired-stereo source run. The local voice and activity checkpoints must match the prior receipts.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
args=("$run" "$spatial" "$paired_stereo" "$weights")
backend/.venv/bin/python -B -m scripts.audio_removed_source_transfer freeze "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_removed_source_transfer score "${args[@]}" --panel selection
backend/.venv/bin/python -B -m scripts.audio_removed_source_transfer fit "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_removed_source_transfer score "${args[@]}" --panel confirmation
backend/.venv/bin/python -B -m scripts.audio_removed_source_transfer report "${args[@]}"
backend/.venv/bin/python -B -m scripts.review_audio_removed_source_transfer "$run" "$spatial" "$paired_stereo"
```
