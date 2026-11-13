# Confirm short reflected speech inputs on a separate recording set

The short reflected-input check added no useful edit locations on 60 other deliveries. It added one genuine source group with a flag, in both its gain-only copy and its traffic mix. It stays outside the app. [Held results and window commitments](evaluation-panels/audio-reflected-local-confirmation-2026-11-12.json)

| Condition | Current focused review | With shorter review | Genuine files flagged, current / combined |
| --- | ---: | ---: | ---: |
| Gain-only | 15/18 | 15/18 | 0/12 · 1/12 |
| Traffic, foreground +8 dB | 12/18 | 12/18 | 1/12 · 2/12 |

The result uses all 60 deliveries of the existing dense-evidence panel. The 30 exact foreground recordings are disjoint from the reflection comparison. Ten speaker groups were also outside its selection and held panels, including four edited speakers. The other 20 speaker groups had earlier exposure. Each foreground has a gain-only copy and a matched mix with the same recorded traffic source. This traffic was quieter than the foreground. The original rain and train mixtures had louder backgrounds. No reflection parameters, ranking rule or operating point changed. The study uses the original activity gate, voice weights and stored dense comparator. [Source and licensing](audio-dense-evidence-confirmation-2026-11-12.md) · [Frozen reflected-window point](audio-reflected-local-transfer-2026-11-12.md)

The distinct speaker subset retained all four useful edit locations in both conditions. The new false alarm came from that subset and cost 1.2 seconds in each delivery. One highlighted range on a previously exposed traffic speaker fell outside a useful location. Across edited files the candidate marked about 2.34 additional seconds outside known edits. These results give no additional support for adding this response to the customer confidence estimate.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_reflected_local_confirmation review \
  docs/evaluation-panels/audio-reflected-local-confirmation-2026-11-12.json
```

For native review, add `--native "$dense_panel"` using the exact local WAV-derived PCM from the [dense panel](audio-dense-evidence-confirmation-2026-11-12.md#reproduce). Native replay rebuilds each reflected input hash and the standard-library review recalculates the decisions. To repeat inference, use the pinned local candidate models and a fresh `run` directory with the same `--native` argument.
