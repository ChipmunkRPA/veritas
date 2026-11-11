# Local qualification of a three-stream speech separator

After the speaker-count gate failed on the harder overlap panel, test source separation directly. The separator recovered 14 of 16 quieter third voices and 12 of 16 fainter third voices under the fixed acoustic criterion. All 16 two-voice controls had a substantial unmatched output. This supports testing separated speech as an inspection input, with a separate way to establish whether a voice is present. [Exact aggregate and freeze commitments](evaluation-panels/audio-separation-third-voice-pilot-2026-11-11.json)

## Frozen method

Use the SpeechBrain SepFormer checkpoint trained on Libri3Mix, released under Apache 2.0 and pinned to `1f2d824e2333aec9e60d0030fb144885c1afc426`. The checkpoint produces three outputs and expects 8 kHz mono input. The runner constructs the published architecture from reviewed settings and loads the exact tensor state with `weights_only=True`. Only the encoder, decoder and mask weights are used. [Checkpoint and terms](https://huggingface.co/speechbrain/sepformer-libri3mix/tree/1f2d824e2333aec9e60d0030fb144885c1afc426) · [Separation architecture paper](https://arxiv.org/abs/2010.13154)

Select eight disjoint triplets by a fixed salted order and language quotas from the [already consumed hard-overlap panel](audio-hard-overlap-transfer-2026-11-11.md). The new pilot uses one rotation per triplet, both deliveries, and the exact two-voice, −9 dB third-voice and −15 dB third-voice mixes. The exact 48 inputs and three reference stems per source are committed before this separator runs. Use the same resampling for input and reference, and assign predicted outputs by maximizing summed scale-invariant signal-to-distortion ratio, or SI-SDR, with no timing search. A third-voice recovery requires SI-SDR of at least 0 dB and improvement of at least 3 dB over the original mixture.

The exploratory advancement rule requires at least 12 recoveries out of 16 at each level. It also allows at most two of the 16 two-voice controls to have an unmatched output at or above 10% of mixture RMS. The unmatched measure checks output energy. It does not identify speech content or whether the output is a duplicate.

## Results

| Known third voice | Dry recovered | Reflections recovered | Total |
| --- | ---: | ---: | ---: |
| −9 dB relative to the other interferer | 8/8 | 6/8 | 14/16 |
| −15 dB relative to the other interferer | 7/8 | 5/8 | 12/16 |

All 16 two-voice controls exceeded the unmatched-output energy bound. A separated output can therefore carry significant energy even when the known mix contains only two voices. The pilot meets the acoustic recovery requirement on the consumed panel and fails the extra-output requirement. It does not evaluate downstream AI detection or a new voice confidence value. Test an independent presence or duplication check on new sources before treating separator channels as separately established voices.

The local CPU run completed 48 forwards in 38.0 seconds. A separate readout verified the exact parent and model commitments, all 48 PCM hashes, eight distinct groups and all 48 saved score decisions. No individual audio, estimates or source identifiers are included in the public receipt.

## Reproduce

With the exact source panel still available locally, place the checkpoint and `hyperparams.yaml` in the assets directory with names and digests from the [runner](../backend/scripts/audio_separation_third_voice_pilot.py). The local pilot used an isolated environment with PyTorch 2.8.0, TorchAudio 2.8.0 and SpeechBrain 1.0.3. The main app uses its own environment.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m scripts.audio_separation_third_voice_pilot freeze "$run" "$assets" "$overlap_run"
python -m scripts.audio_separation_third_voice_pilot run "$run" "$assets" "$overlap_run"
```
