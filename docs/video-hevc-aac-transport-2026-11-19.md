# Source-linked face evidence in HEVC clips with a soundtrack

Adding a supported AAC track to each of seven held HEVC clips preserved
the exact face comparison in all 21 deliveries. Every one of the three
tested soundtrack layouts retained both known swap ranges and the one
previous ordinary blur highlight. The layout repetitions share the same
picture sources and count as seven paired roles for interpretation.
[Earlier video-only comparison](video-short-face-hevc-25-ordinary-2026-11-19.md)

## Test design

The 14 archived source recordings used to build the earlier trials supplied
no original audio tracks. This test added an authored non-speech timing
signal while copying the original HEVC packets losslessly. The seven picture
roles include two swaps with blur, one swap without added blur, an ordinary
blur previously highlighted, a measured ordinary blur without a highlight,
an unchanged camera excerpt, and an authored straight cut between cameras.

All videos have 120 exact 25 fps, 1280 × 720 `hvc1` frames at a zero-origin
90 kHz clock. Each of the three tested layouts presents one zero-origin AAC-LC
stream for 4.8 seconds with 1,024 samples of encoder priming. The layouts are
mono 44.1 kHz, mono 48 kHz, and stereo 48 kHz. The media edit list and the
first packet's decoded skip agree on the playable start. Three additional
controls carry an offset, a shortened soundtrack, or two audio tracks.
[AAC edit-list timing](https://developer.apple.com/documentation/quicktime-file-format/using_track_structures_to_represent_encode_delay_explictly)

## Results and limits

| AAC layout | Same picture and face record | Both known ranges retained | Ordinary blur highlight retained |
| --- | ---: | ---: | ---: |
| 44.1 kHz mono | 7/7 | 2/2 | 1/1 |
| 48 kHz mono | 7/7 | 2/2 | 1/1 |
| 48 kHz stereo | 7/7 | 2/2 | 1/1 |

The exact known ranges remain **3.080–3.240 s** and **3.880–4.000 s**,
and the ordinary highlight remains **2.920–3.040 s**. The published receipt
preserves their exact source frame positions. The separate native audit
checked 2,880 encoded pictures, decoded RGB frames and timestamps across all
24 deliveries. Each qualified mux decoded to the same audio as its standalone
AAC track. The largest measured marker displacement was 19 samples at 48 kHz,
about 0.40 ms. The audit withheld all three out-of-scope cases.

The project team ran the study. It measures container transport on reused
picture sources with authored tones. Real soundtrack diversity, voice forgery,
lip-sync accuracy and additional encoder layouts need separate evidence.

## Recount

The receipt preserves the anonymous roles, ranges, clocks and measured marker
offsets. Native media, speech, face measurements and identifying file paths
stay in local custody. The offline reviewer checks the public arithmetic.

```sh
python3 -B -I -S backend/scripts/review_hevc_aac_transfer.py \
  docs/evaluation-panels/video-hevc-aac-transport-2026-11-19.json
```
