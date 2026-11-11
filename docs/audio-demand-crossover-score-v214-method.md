# Score the matched river-excerpt crossover

Use the complete published and guarded [v210 preparation](audio-demand-crossover-v210-method.md).
Retain all 24 planned genuine/synthetic pairs in each of three conditions:
cropped matched clean, the shared early river excerpt and the shared late
river excerpt. Each complete pair uses the same source-noise samples across
classes at a common prefix length; noise is separately scaled to each
foreground. Incomplete prepared pairs remain unavailable in all conditions.

Before inference, bind the v210 freeze, published preparation and data-only
guard receipts, its complete ordered private journal and every prepared WAV
and PCM hash. Verify the frozen serving route and local assets used in v208;
do not change calibration, thresholds, or detector defaults. The diagnostic
uses the same offline CPU speech gate, voice check and original-side stability
test as [v208](audio-demand-transfer-v208-method.md). A speech-gate rejection
is unavailable and remains in the planned denominator. No cropped clean
score is assumed to equal the longer v208 source-clean score.

Run three **serial** guarded batches, one per condition. A later batch is
admitted only after all earlier batches have successful, verified guard and
observation receipts. Each case's source coordinates, score and model record
remain private. Report only aggregate commitments and counts:

- Per origin and condition, planned and scored counts, speech-gate rejection,
  preparation unavailable, AI, original and inconclusive results.
- For synthetic inputs, wrong-original calls separately from all missed AI
  calls. A zero AI count does not mean every case was labeled original:
  inconclusive and unavailable outcomes are distinct.
- Within each foreground, matched-clean→early, matched-clean→late and
  early→late direction and gate-status transitions. Keep all planned pairs
  and report how many have both conditions scored.
- Within each shared-excerpt pair, genuine-versus-synthetic joint directions
  and the number of pairs in which both were scored.

No public readout includes row identifiers, timestamps, individual model
responses or model identity. Use the unchanged v108 guard for each batch;
failed or partial attempts are retained, not retried or overwritten.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_score_v214 freeze
for batch in 0 1 2; do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_score_v214 run "$batch"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_demand_crossover_score_v214 readout
```

The paired window comparison is a consumed-data diagnostic on a single river
tape and one synthetic collection. Pairing removes the class-by-offset
assignment from v208 but does not remove differences in recording, source,
speaker and generator. Paired windows from one tape are not independent noise
recordings. Cropping also changes the original inputs. The result cannot
establish population error rates, a validated calibration or a production
improvement; no new policy or serving behavior follows from this test.
