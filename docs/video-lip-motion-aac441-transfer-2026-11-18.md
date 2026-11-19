# Local speech and mouth timing on 44.1 kHz AAC

The 44.1 kHz video route verifies continuous mono AAC-LC source clocks for
soundtrack dropout checks. This paired local experiment applied the existing
48 kHz speech-to-mouth comparison to the exact 16 kHz presentation decoded
from 44.1 kHz AAC. It used the same trained comparison and fixed operating
point. The observation compares local speech and mouth movement and carries
zero weight in AI confidence.
[Current 48 kHz comparison](video-mouth-and-sound-2026-11-17.md) ·
[44.1 kHz dropout check](video-sound-dropouts-aac441-2026-11-17.md)

The test reuses 18 held recordings of nine Lombard Grid speakers. These
recordings were outside fitting and point selection and appeared in the
previous PCM and 48 kHz confirmation. The new 44.1 kHz deliveries use a
112 kb/s AAC-LC encode and keep the original picture streams and presentation
timestamps. Each recording supplies one staged replacement with different
speech, a same-word replacement, two whole-track offsets, an unchanged
control and a gain control. We used the native audio clock to exclude
encoder padding and checked audio alignment to one 16 kHz analysis sample.
[Publisher recordings and terms](https://zenodo.org/records/3736465)

| Carrier | Useful edit locations / 18 | Misplaced locations / 18 | Unchanged or gain flags / 36 | Same-word flags / 18 | Whole-track offset flags / 36 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 48 kHz AAC | 2 | 1 | 2 | 2 | 0 |
| 44.1 kHz AAC | 2 | 1 | 2 | 2 | 0 |

A useful location overlaps at least 40% of the union with the known edit
range. Both useful cases carried over and involved two different speakers.
Both carriers flagged the same ordinary controls and produced one misplaced
highlight. The earlier speech-event comparison correctly localized zero of
18 on the retained 48 kHz uploads. The two carriers share all target
recordings and edits. Their results measure transfer between encodings on
controlled speech replacements. Unscripted speech, regenerated mouths and
AI authorship accuracy remain outside this evidence.

Local replay checked 12,384 decoded picture timestamps across the 216
deliveries and all 42 proposed before, during and after RGB anchors against
the original files. In one correct case both carriers flagged **1.149–1.358 s**
of sound and supported it with native frames **26, 28 and 32**. The 44.1 kHz
lane was measured offline against the serving point. Existing serving
admission left the 44.1 kHz speech-to-mouth comparison unassessed.

## Recount the paired result

The numeric receipt contains anonymous recording and speaker groups, fixed
comparison readings, the authored intervals and source clock bounds.
The standard-library reader recomputes the selection, temporal overlap and
paired result. Commitments refer to the closed local inputs and original
RGB verification.

```sh
python3 -I -S backend/scripts/review_video_lip_motion_aac441.py \
  docs/evaluation-panels/video-lip-motion-aac441-transfer-2026-11-18.json.gz
```

[Anonymized carrier transfer](evaluation-panels/video-lip-motion-aac441-transfer-2026-11-18.json.gz)
