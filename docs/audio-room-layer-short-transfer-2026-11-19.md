# Short speech calibration across measured rooms

A separately calibrated point on the existing 750 ms responses added two
useful deliveries and one ordinary false alarm. Both additions share one
edited voice and reader pair. A fitted combination of frozen layer
responses added five useful deliveries and four ordinary false alarms.
The original fitted-arm gate failed. Both arms stay in development.

## Ordered source roles

The frozen trial fits 25 nonnegative layer weights and a bias using the
existing classifier head on short windows. The encoder, head, preparation,
speech gate and native recordings remain unchanged. This is a narrower
room-transfer follow-up to the older
[full-vector layer experiment](audio-layer-v12-development-evaluation.md).

Fitting uses 132 inputs from six PhonemeDF readers in three Surrey room
pairs. Point selection uses only 48 untouched or genuinely edited inputs
from eight different MLS readers in four BBC room pairs. Each arm's point
is the second-highest of four genuine source-pair adjacent-window maxima.
The single fit and both points were sealed before transfer inference.

Transfer uses all sixty canonical mono deliveries from the
[original-channel short-grid study](audio-original-channel-short-grid-2026-11-19.md).
Twenty-four later MLS readers form twelve pairs across Aachen and
Huddersfield. The parent selection pins exclusions for the older stock
host and donor rosters. No native delivery hash crosses the three roles.
Cross-corpus reader aliases and model-training overlap remain unresolved.
All recordings appeared in earlier project studies. The hidden-layer
short-window measurements are new to this frozen method.

Each input retains the original six-second, 16 kHz PCM16 room recording
and source clock. Point-only and learned arms preserve every incumbent
interval and add their own adjacent 750 ms intervals. The test measures
timing without assigning a response to a person. Six-second files are
outside the current twelve-second brief-review product admission.

## Closed transfer outcomes

| Arm | Useful edited deliveries | Missed edited deliveries | Ordinary files flagged | Edited files with only wrong times |
| --- | ---: | ---: | ---: | ---: |
| Existing methods at their frozen points | 3/36 | 33/36 | 1/24 | 2/36 |
| Existing responses with additional calibrated point | 5/36 | 31/36 | 2/24 | 3/36 |
| Fitted layer combination with its calibrated point | 8/36 | 28/36 | 5/24 | 7/36 |

The point-only paired review found two additions and zero lost useful
deliveries. Both are the same louder synthetic edit in two delivery
conditions, within one Huddersfield reader pair and synthesis family.
Their matched ordinary controls had no useful-time collision. The added
ordinary alarm belongs to one genuine-splice Aachen file and highlights
1.25 additional seconds.

Three Aachen edited deliveries from that same false-alarm pair acquire
wrong-only findings. In Huddersfield, two previously wrong-only files gain
a useful interval. Their old incorrect intervals remain present, so the
change to their category represents an added useful time. It represents
no removal of a previous incorrect interval.

The layer fit gains five useful deliveries in three Huddersfield pairs
and two synthesis families. It gains no quieter-only delivery and no
useful Aachen delivery. Ordinary flags increase by four and highlighted
ordinary time by five seconds. The original gate requires broader gains
and a smaller ordinary cost, and remains failed.

The point-only tradeoff is a reasonable candidate for a separately frozen
twelve-second confirmation. Its two added files represent one physical
edited event. A meaningful next test needs additional source pairs,
ordinary edits and rooms while retaining this point. This is a research
judgment after reviewing a predeclared arm. It changes no frozen gate,
product setting, customer confidence or model weight.

## Independent verification and public replay

A model-free native reviewer checked all 240 inputs against retained
original stereo bytes and independently replayed their canonical mono
decoding. It compared 46,080,000 native stereo sample identities. The
separate outcome reviewer checked 2,640 source/window/admission bindings,
reproduced the optimizer and points, and passed nine authored temporal
and assignment controls. All 660 transfer short-window comparisons
exactly match the independently closed original model readings.

The scorer made 1,073 development and 414 transfer calls, including an
exact initial repeat in each phase. Three earlier authored calls verified
that the per-layer view reproduces the original forward and probability.
The largest source head-reconstruction difference was below 0.000002 in
raw response units. Inference took about eight minutes on two CPU threads.
There were no downloads, new recordings or generation calls.

The companion aggregate omits identities, exact edit positions, individual
waveform hashes, hidden responses, private paths and audio. Its portable
standard-library reviewer checks pinned aggregate arithmetic and eight
authored alterations. Exact source/model reproduction uses the retained
private custody and source procedures in the
[two-voice benchmark](audio-two-voice-components-2026-11-15.md) and
[stereo replication](audio-stereo-channel-replication-2026-11-15.md).

```sh
python3 -B -S backend/scripts/review_audio_room_layer_short_transfer.py \
  docs/evaluation-panels/audio-room-layer-short-transfer-2026-11-19.json \
  --self-test
```

Upstream sources are the [PhonemeDF release](https://zenodo.org/records/19522979),
[MLS release](https://www.openslr.org/94/) and the
[voice-model publisher](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector).
The existing commercial-use review and frozen parent provenance apply.
Controlled room renders describe this study's conditions. They establish
no population accuracy for natural multi-speaker recordings.
