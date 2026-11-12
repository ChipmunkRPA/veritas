# Waveform fallback on new meetings, speakers, and backgrounds

The waveform fallback added **5 correct localizations across 2 source groups**. The exact and two-band checks localized **629 of 720** known-copy views. With the fallback the total reached **634 of 720**, with **zero added flags across 720 ordinary controls**. The result fell below the frozen detection and source-group criteria. [Audited aggregate](evaluation-panels/audio-waveform-fallback-transfer-2026-11-11.json)

## Held source panel

Three separately qualified AMI meeting recordings supplied 24 native three-second mono windows. The fixed selection takes the earliest four available two-voice and four three-voice windows from each meeting's existing annotation roster. The 12 VoxPopuli native stems cover five languages, distinct publisher speaker IDs, and distinct parliamentary session dates. The AMI meetings and parliamentary sources are disjoint from the AMI and RUEG foregrounds in the earlier repetition study. These new repetition inputs had prior exposure in speech overlap experiments. The AMI windows within each meeting share that recording and the number of independent foreground groups is 15. [AMI preparation and rights](audio-stream-presence-transfer-2026-11-11.md) · [VoxPopuli preparation and CC0 source](audio-hard-overlap-transfer-2026-11-11.md) · [Earlier repetition study](audio-repeat-robustness-transfer-2026-11-11.md)

Each source supplied a paired authored copy and a control with no authored copy. Select the highest-energy 650 ms among six early windows. Copy that span into 1.750–2.400 seconds and apply 20 ms boundary blends. Keep the exact sample origin of both ranges. Process the two arms through PCM, AAC at 64 kb/s, MP3 at 64 kb/s, and Opus at 48 kb/s. The app decoder verifies a continuous native sample clock. The 128 actual AAC padding samples are retained. Both arms receive identical background excerpts and common gain at each condition. The noise excerpts start at 100 seconds and step by 5 seconds for each source. The washroom and river tapes used for the earlier repetition study are distinct from these kitchen and traffic recordings. Kitchen and traffic were used in unrelated project experiments, and the windows here are separate from the earlier documented preparation at 4–94 seconds. [Pinned noise channel and CC BY-SA 3.0](evaluation-panels/audio-demand-kitchen-traffic-assets-v212.json) · [Earlier kitchen and traffic preparation](audio-fresh-shared-noise-v217-method.md)

## Frozen route and results

Run the existing exact and two-band methods. When both return empty pair lists, run the unchanged pre-emphasized waveform check. A correct localization requires 2,400 or more overlapping samples on each side, outside the fades. The exact check requires exact lag. Approximate cues allow a 32-sample lag error. Count any reported pair in a control as an ordinary flag, including possible naturally repeated sound. The advancement criteria require at least 24 added correct views across six source groups, including at least two from each family. Allow at most three added control flags from at most two groups and at most three added wrong-only copy views.

| Authored-copy condition | Views | Existing exact or two-band | With waveform fallback | Added |
| --- | ---: | ---: | ---: | ---: |
| Clean | 144 | 143 | 144 | 1 |
| Kitchen +5 dB | 144 | 140 | 140 | 0 |
| Kitchen −5 dB | 144 | 127 | 127 | 0 |
| Traffic +5 dB | 144 | 140 | 144 | 4 |
| Traffic −5 dB | 144 | 79 | 79 | 0 |
| **Total** | **720** | **629** | **634** | **5** |

Every added correct pair had the exact authored lag. The median of the smaller side intersection-over-union was 0.381. The clean gain was one Opus delivery from one meeting. All four traffic gains came from four encodings of one parliamentary speaker's recording. The baseline produced eight wrong-only copy views and the fallback added zero. All 1,440 inputs decoded, including 36 unchanged clean PCM controls that exactly match their source windows. The fallback ran on 803 views with no existing pairs. It returned a measured result for all 803 and left every existing pair in place. The hardest condition was traffic at −5 dB. It left 65 of 144 known repeats unlocalized in both paths.

The separate standard-library reviewer checks the complete planned roster, source group binding, clock geometry, fixed route and every reported pair against the authored ranges. The private receipt covers each source and each encoded and decoded payload hash. A second decode and score run reproduced the complete journal byte for byte, including codec and decoded PCM digests. This trial measures localization of within-recording copies in short continuous audio. It leaves natural different-source edits, longer recordings, and AI origin outside scope.

## Local app trial

Development instances can opt in with `VERITAS_AUDIO_WAVEFORM_REPEAT_REVIEW=true`. The setting defaults to false and applies to new evidence sets in development. For standalone audio or a video soundtrack, the check runs on continuous 2.5–4 second mono input only when neither the exact nor the frequency-band comparison produced a pair. It examines at most 12 candidate lags and can return up to eight pairs. Each reported pair includes the same two exact playback clips, source timestamp ranges and sample commitments in the analysis and its JSON, PDF and evidence package. The comparison contributes zero weight to the file's AI direction or confidence. ZIP-only verification checks the signed metadata and playback assets without importing the signal processing runtime. Verification with the original supplied via `--source-audio` also reconstructs the waveform measurement. Both legacy and new evidence remain verifiable with the setting off.

## Reproduce

Pass the original [AMI annotation-plan receipt](audio-stream-presence-transfer-2026-11-11.md), [VoxPopuli manifest and stems](audio-hard-overlap-transfer-2026-11-11.md), and pinned DEMAND kitchen and traffic publisher ZIPs. The worker requires the exact source commitments and uses the installed FFmpeg and ffprobe binaries. Its private freeze records both binary hashes. No model weights or neural inference are needed.

Use the historical checkout for the published run's byte-for-byte freeze. The app now calls the same waveform calculation from the evidence module.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run; do
  python -m scripts.audio_waveform_fallback_transfer "$phase" "$run_dir" \
    --ami "$ami_dir" --vox "$vox_dir" --noise "$demand_dir" || break
done
python -I backend/scripts/review_audio_waveform_fallback.py "$run_dir" \
  --published docs/evaluation-panels/audio-waveform-fallback-transfer-2026-11-11.json
```

The study publishes aggregate counts and whole-artifact commitments. Participant identifiers, individual audio, individual coordinates and all media remain in the private run.
