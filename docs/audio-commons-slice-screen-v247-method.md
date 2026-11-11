# Machine screening for recorded background excerpts

Verify the exact v234 acquisition and local PCM receipts. Make a three-second nonoverlapping grid on each decoded recording. Retain at most 12 windows per recording at fixed positions across the grid. Keep the publisher's declared kitchen or traffic category and retain the exact sample offset and PCM hash in the private packet. [Verified background preparation](audio-commons-background-v234-result.md)

Run the existing CPU speech activity route and local PANNs sound-tag model on every frozen window. The screening point is fixed before inference. Set a window aside if the speech route returns any span or if the maximum of the listed speech or music tags reaches 0.30. Set constant windows aside as well. Record every direction privately. This is an exploratory selection rule. Class scores provide machine evidence, and human listening review remains pending. Preserve the provisional scope in any resulting mixtures and detector comparisons. [Speech activity implementation](../backend/app/detectors/audio/upgraded.py) · [PANNs publisher release](https://zenodo.org/records/3987831)

Use the existing one-worker resource guard. Run with local pinned assets and keep both source recordings and reviewed windows private. The public result includes aggregate coverage and exact whole-packet commitments. This stage makes no authenticity classifier calls. Report tape reuse separately from foreground speaker counts.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_commons_slice_screen_v247 freeze
backend/.venv/bin/python -B -m scripts.audio_commons_slice_screen_v247 run
backend/.venv/bin/python -B -m scripts.audio_commons_slice_screen_v247 readout
```
