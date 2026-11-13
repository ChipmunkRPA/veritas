# Separate competing stereo voices using their recorded direction

Two microphones record timing and level differences that the mono mix loses. This trial uses those differences to extract anonymous audio views before applying the same voice and speech-activity checks. An earlier blind separation route estimated two outputs and could choose the wrong speaker. The new method can produce a third view from the uploaded stereo alone. It has no access to speaker identity, the edited interval, known directions, clean speech, or the room impulse response. [Earlier blind voice comparison](audio-unassigned-stereo-voice-2026-11-13.md)

The method estimates up to three distinct broadband lag peaks from the stereo recording. It uses each peak and its estimated channel-level difference to initialize a complex angular Gaussian distribution for the corresponding band of interchannel directions. Eight regularized updates produce a soft frequency mask for each anonymous output. Before quantization the masks sum to one, which preserves the original mix when the outputs are added together. Each reconstructed waveform keeps the input sample clock. The strongest admitted voice response is chosen within each two-second analysis interval. The directional calculation uses independently written code and no new pretrained weights. Its assumptions follow the published angular mixture method and later work on directional initialization. A recent localization study measures direct-path evidence using speech onsets, a separate technique from the lag initialization tested here. [Angular mixture method](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256519.pdf) · [Directional initialization](https://link.springer.com/article/10.1186/s13636-022-00246-7) · [Recent onset localization study](https://arxiv.org/abs/2604.01524) · [Local model license audit](commercial-model-audit.md)

## Frozen comparison

Surrey supplies six synthetic edits and twenty ordinary views. Only admitted ordinary views in Surrey set the operating point for each method. BBC supplies six held edits and twenty ordinary views. The Huddersfield balanced-target, lateral-target and two-masker layouts each supply eight held edits and twenty-eight ordinary views. The three Huddersfield layouts repeat the same eight foreground groups. The local roster audit found zero shared foreground and donor speaker identifiers, source or fragment hashes, or exact background capture hashes across the three rooms. Huddersfield uses different intervals from recurring AMI meeting sessions for its competing speech. All exact inputs had earlier project exposure under different methods. Each foreground has matched untouched, same-reader and other-reader versions. Room-only controls are separate. Rights and disjoint source memberships are pinned by the two ancestor panel receipts. [Surrey and BBC sources](audio-measured-spatial-transfer-2026-11-12.md#sources-and-fixed-comparison) · [Huddersfield sources](audio-blind-huddersfield-transfer-2026-11-12.md#inputs-and-scope)

Compare three arms at their frozen genuine maxima. The first uses the submitted stereo mean. The second takes the larger of the earlier two blind-separation responses. The third takes the largest response across the new eligible outputs. Each chooses its stronger admitted window from 1–3 and 3–5 seconds. A useful range must overlap at least half the actual edit and achieve temporal intersection over union of at least 0.30. The additive comparison retains the original mono findings. Equal ordinary-group rankings use held scores and are retrospective. Reference foreground stems are used after separation to measure waveform quality and do not determine the method's output. A source that passes audio reconstruction can still yield poor AI discrimination or the wrong interval.

## Acoustic recovery

Every held synthetic case yielded at least one unclipped reconstructed waveform. For each case, selecting by best full-clip reconstruction with the true stem gave the following median gains over mono. The same waveform is used for both measurements.

| Held room or layout | Full-clip SNR gain | Edit SNR gain | Edit SI-SDR gain |
| --- | ---: | ---: | ---: |
| BBC | +7.75 dB | +7.87 dB | +5.89 dB |
| Huddersfield balanced target | +6.54 dB | +6.89 dB | +3.11 dB |
| Huddersfield lateral target | +9.21 dB | +8.28 dB | +4.97 dB |
| Huddersfield two competing feeds | +6.49 dB | +6.88 dB | +2.82 dB |

Part of the raw SNR improvement comes from lowering the mixture level. Against mono scaled to minimize error against the reference, the median edit SNR changes were +2.09, −0.28, +1.04, and −0.42 dB. The positive scale-invariant signal-to-distortion ratio (SI-SDR) gains indicate an improvement after discounting a constant gain. These extra acoustic measurements were computed separately after the experiment was frozen and did not change the cutoffs or the advancement decision.

The earlier two-output separator's corresponding best-output gains inside the edit were +2.66, +4.89, and +3.21 dB in the three Huddersfield layouts. This comparison measures whether one recovered track is closer to the known foreground. An uploaded file provides no ground truth for choosing that track. [Earlier acoustic reconstruction](audio-blind-huddersfield-transfer-2026-11-12.md#acoustic-outcomes)

## Voice findings

Each result below gives useful locations and flagged ordinary source groups. An ordinary group can have more than one delivered variant. The maximum response on an ordinary Surrey file was higher for the directional method than for mono or the prior blind separator.

| Held room or layout | Mono | Prior blind outputs | Directional outputs | Mono plus directional |
| --- | ---: | ---: | ---: | ---: |
| BBC, 6 edits | 0 / 2 | 2 / 2 | 0 / 2 | 0 / 3 |
| Huddersfield balanced target, 8 edits | 0 / 4 | 0 / 2 | 0 / 0 | 0 / 4 |
| Huddersfield lateral target, 8 edits | 0 / 3 | 1 / 2 | 2 / 1 | 2 / 4 |
| Huddersfield two competing feeds, 8 edits | 3 / 3 | 0 / 3 | 0 / 0 | 3 / 3 |

Two useful lateral detections select **1.000–3.000 s**. The authored stock insert spans **1.840–2.640 s** and the clone spans **1.680–2.640 s**. The overlaps cover the entire edits at 0.40 and 0.48 interval IoU. On both clips the highest voice response also came from the output with the best full-clip acoustic reconstruction. Across all lateral edited cases, the strongest admitted response came from that best-reconstructed output on seven of eight clips. It did so on five of eight in the two-competitor layout, where the directional method alone lost all three mono locations. Three reconstructed views can include reverberation or secondary modes from the same voice.

The directional comparison measured 130 of 134 held files. Four lateral controls with only background had insufficient evidence for distinct lag modes. Its sole misplaced detection occurred in BBC. Combining with mono added two useful edits in distinct synthesis families and two flagged genuine groups, one in BBC and one in the lateral layout. It added zero background-only flags. At a retrospective budget of one ordinary group, the lateral arm located two edits, versus zero for the mono and prior blind outputs. At the same budget it found one balanced edit, versus zero for both baselines. All arms found zero at that budget for BBC and the two-feed layout. Results at matched held false-group budgets do not set a release point.

This meets the predeclared gate for new-source replication as an additive check. The gain comes from one room layout already used elsewhere in the project. The method stays offline and contributes no AI confidence while a new independent-room and speaker test is pending. [Complete frozen result and replay](evaluation-panels/audio-spatial-mask-voice-transfer-2026-11-13.json)

## Replay

Use the retained measured-room panel, third-room panel and previous blind voice run. The input clock and prior mono hashes must match their published commitments. The scoring stages use pinned local assets with offline mode. The independent native replay repeats waveform construction with NumPy rather than SciPy's FFT and checks the exact PCM hashes and speech admission. The public reader needs only the standard library. It checks case roles and fixed calibration points, recomputes location overlap and ordinary-group counts, and contains no speech, identities, or voice embeddings. It does not rerun model inference.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development held audit; do
  backend/.venv/bin/python -B -m scripts.audio_spatial_mask_voice_transfer "$phase" \
    "$measured_room" "$third_room" "$previous_voice_run" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_audio_spatial_mask_voice_transfer.py "$receipt" --from-run "$run"
python3 -B -IS backend/scripts/review_audio_spatial_mask_voice_transfer.py \
  docs/evaluation-panels/audio-spatial-mask-voice-transfer-2026-11-13.json
```
