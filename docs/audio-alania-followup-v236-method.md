# Additional designed voices for the noisy speech comparison

Use the existing pinned, attributed Alania shard and the v213 exact preparation method. Exclude all 40 planned rows from the v178 and v213 cohorts, including their publisher voice IDs, text documents and row locations. Take 32 distinct remaining voice and document pairs in the existing deterministic order. Keep duration, rights, postfilter and designed-voice metadata requirements fixed. Verify the published v213 preparation before freezing the roster. [Pinned publisher release](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr/tree/277dfc8a2322014099ff72688bcaee12480c4605) · [Prior preparation method](audio-alania-fresh-v213-method.md)

Use the guarded production decoder to retain the complete source extent as 16 kHz mono audio. Keep every planned slot and its outcome. Compare exact embedded bytes, decoded PCM and prepared WAV fingerprints against both prior cohorts and against the new batch. This verifies known exposure by identity and exact bytes. The new voices share one synthesis family with prior experiments. Later evaluation needs separate generators, genuine sources and background recording groups. The preparation does not fit or call a detector. Publish the aggregate and whole-artifact commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_alania_followup_v236 preflight
backend/.venv/bin/python -B -m scripts.audio_alania_followup_v236 freeze
backend/.venv/bin/python -B -m scripts.audio_alania_followup_v236 prepare
backend/.venv/bin/python -B -m scripts.audio_alania_followup_v236 readout
```
