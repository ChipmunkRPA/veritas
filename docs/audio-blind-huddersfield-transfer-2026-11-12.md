# Blind stereo separation in a separate measured room

The earlier blind stereo study used two independent measured rooms with a balanced target and one lateral playback feed. This follow-up keeps the separator and its channel-balance routing fixed. It measures whether the same blind route preserves the intended target when its position changes or when an additional competing feed is present. The outcome is waveform reconstruction, with zero voice model calls. [Prior measured-room design](audio-measured-spatial-transfer-2026-11-12.md)

## Inputs and scope

The University of Huddersfield MAIR release provides six pinned 44.1 kHz WAV files from its KU100 measurement at St Paul’s Hall. Recheck publisher blob identities and all six decoded channel commitments against the existing source intake. S01 is an approximately balanced path, with a +0.125 dB early left-minus-right energy difference and zero samples of correlation delay. S03 and S07 have opposite early imbalances of −10.680 and +10.808 dB and opposite delays of 33 and −35 samples. The measurement used a fixed interval around the two channel peaks. Those roles describe observed interaural differences. The publisher’s source ID to physical azimuth mapping remains unresolved. Keep the complete three-second responses and their stored sample zero. The room assets are CC BY 4.0. [Publisher release](https://zenodo.org/records/2639677) · [Exact local room qualification](audio-room-assets-v299-method.md)

Use eight previously evaluated six-second interior-edit groups, with three expressive, three stock and two clone inserts. Each group contains an unchanged target, same-reader splice, other-reader splice and generated edit. They are separate from both earlier blind stereo roles by exact source file, donor fragment, speaker, and source group. The four AMI meeting sessions recur, with 12 nonoverlapping retained background intervals excluded from all 16 previous blind-panel intervals. Neither the underlying speech banks nor the selected clips are new to the project. The selected foreground and generated source terms are CC BY 4.0, CC BY-SA 4.0 and the publisher’s PLN MIT declaration. Raw and rendered waveforms remain local. [Earlier speech sources and exact qualification](audio-remote-neighbor-replication-2026-11-12.md) · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Render each source under three layouts. The first places the target on S01 and one masker on S03 or S07. The second switches target and masker paths. The third keeps the target on S01 and plays two independently sourced AMI excerpts on S03 and S07, with equal mono power per competing feed before common scaling. A feed can already contain several people. Set combined background power to −5 dB input SNR for the untouched target’s mono mean. All variants of one source share the exact quantized masker and one gain per layout. Include four background-only controls per layout. Each layout provides 36 clips. Reproduce native PCM sums, scope the exact edit to its original sample positions and verify the authored change does not leave its causal room support.

The unchanged two-input separator receives only the two-channel mixture. Its routing prefers the candidate with lower side-to-mid speech-band energy and abstains if the candidates’ ratios are too close. Both blind outputs are also compared against the exact room-rendered target to measure the best achievable output with known assignment. That comparison uses the true foreground only after separation. Measure median improvement over the integer-rounded stereo mean, both across the full clip and within the authored edit. The frozen condition gate requires all eight edited groups routed, at least +3 dB median gain in both spans, and no selected clipping. A route on background-only audio is a routed waveform, with no authenticity decision.

## Acoustic outcomes

| Layout | Edited groups routed /8 | Blind full / edit gain, dB | Best-candidate full / edit gain, dB | Routed choices matching best full-clip candidate |
| --- | ---: | ---: | ---: | ---: |
| Balanced target, one masker | 8 | +2.92 / +2.78 | +2.92 / +2.66 | 7/8 |
| Lateral target, balanced masker | 8 | +0.24 / +0.60 | +5.50 / +4.89 | 0/8 |
| Balanced target, two maskers | 6 | +2.31 / +2.14 | +2.38 / +3.21 | 3/6 |

None of the three layouts passed the frozen acoustic gate. No selected edit waveform clipped. With the lateral target, the best separated candidate beat the native mono mean by a median 5.50 dB across the clip and 4.89 dB over the edit, but the fixed route picked the other candidate in all eight cases. The balanced single-masker layout fell narrowly below both gain requirements even with best-candidate assignment. The three-feed layout lost two routes and its best-candidate whole-clip improvement remained below 3 dB. The best-candidate row always chooses using the full clip, and its edit median is measured on those same candidates. No downstream classifier was evaluated here. These results keep the tested route offline and identify target identity as a separate problem from separating a waveform.

## Replay

The local review rebuilt all 432 PCM bindings and repeated separation on all 108 mixtures with NumPy FFT instead of SciPy FFT. Every route, score, and projected PCM digest agreed. The retained source bank, previous blind pilot’s source roster, and publisher room intake supply exact inputs. A result serialization issue was corrected without changing the byte-identical roster or measured rows. [Verified receipt](evaluation-panels/audio-blind-huddersfield-transfer-2026-11-12.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.audio_blind_huddersfield_transfer "$phase" \
    --archive "$archive" --parent "$measured_room_run" --sources "$remote_neighbor_run" \
    --output "$new_run" --publication "$receipt" || exit 1
done
```
