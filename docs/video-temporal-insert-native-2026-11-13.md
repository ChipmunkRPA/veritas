# Native evidence for inserted footage

The existing earlier-video comparison withheld all four inserted-footage
pairs, including after both clips were delivered at the same dimensions
and codec. An experimental alignment check located all four generated
inserts and three of four ordinary inserts. This is evidence of footage
added relative to an earlier copy. Ordinary edits can produce the same
finding. No model weights are used and the check carries zero AI confidence.
[Publisher dataset and license](https://doi.org/10.5281/zenodo.20548572) ·
[Publisher methods](https://arxiv.org/html/2606.07090v1#S3.SS2)

## Native source binding

The four triplets were the smallest equal-cadence examples chosen in the
[metadata-only intake](video-temporal-insert-source-intake-2026-11-13.md).
The runner downloads only 12 bounded ZIP member ranges from the 14.39 GB
publisher archive. It verifies the response range, local ZIP member name,
body size and publisher CRC. The media payload totals 202,539,032 bytes.
The independent FFmpeg frame-hash pass verified 2,640 decoded grayscale
thumbnails and every timestamp from the application frame table. All
native clips start at zero and advance 512 ticks at time base 1/12288,
exactly 24 fps. The released authentic clips are 4K H.264, the merged
clips are 4K MPEG-4 Part 2, and the generated inserts are 832 × 464 or
464 × 832 H.264.

All four released labels say **100–225** and specify **125 inserted
frames**. The paper describes inserting after the first 99 authentic
frames, implying zero-based **[99, 224)** for a 125-frame insert. Reading
the CSV fields directly as zero-based half-open values gives **[100, 225)**.
Comparing the retained authentic prefix, generated insert and resumed
authentic suffix at
nine possible splits places all four releases at **[101, 226)**,
or **4.208333–9.416667 s**. Each triplet has 125 matching generated
frames at the latter position. For example 38, the mean thumbnail
absolute luma error against the generated segment is 1.892 at native
frame 101 versus 6.848 at frame 99 on the 0–255 scale. The public
ledger retains both the original publisher interpretation and the
observed split. This small fixed-duration subset does not establish
the same displacement for the other 96 releases.

## Same-delivery comparison

Deliver the earlier and submitted clips as H.264 at 24 fps and the same
orientation and dimensions, with a 480-pixel long side and no soundtrack.
The ordinary insertion repeats 125 authentic pictures from the same
recording at the same boundary. The unchanged control reuses the
same delivered source. The unrelated control uses a different earlier
source on the same geometry and frame count, looping if necessary. These
controls separate physical insertion from an inference about origin.

For each possible split, fit the one length increase against both copies.
Use median-centered 96-square luma thumbnails scaled by the 20th-to-80th
percentile span. Score the retained prefix and suffix at the two possible
clock offsets. Require at least 24 retained frames on each side, enough
matching pictures on both flanks, and no near-tied split more than three
frames away. The point was set as a development diagnostic. All four
cases have the same insert position and length, so validation needs
additional sources, timings and durations.

| Delivery | Pairs | Existing comparison assessed | Exact paired insertion |
| --- | ---: | ---: | ---: |
| Published generated insertion | 4 | 0 | 4 |
| Matched ordinary insertion | 4 | 0 | 3 |
| Unchanged delivery | 4 | 4 | 0 |
| Unrelated earlier recording | 4 | 0 | 0 |

The existing comparison declined all four original 4K pairs at its pixel
limit. After controlled delivery it withheld two pairs because their
picture timelines differ and six at its duration or frame-count limit.
The experimental method withheld the
fourth ordinary insertion because the same-scene repetition supported
several nearly tied boundaries. Every accepted generated interval uses
submitted frames **100, 163, 226** as before, during and after witnesses.
The public ledger retains their verified native ticks and thumbnail
SHA-256 values without redistributing pixels.

The application's standalone shot-boundary detector found none of the
eight generated endpoints on either the publisher clips or the matched
H.264 deliveries. A fixed robust luma-step test found **one of eight**
on both carriers. It also found **three of eight** ordinary insert
endpoints, and neither method made a report on unchanged controls. These
standalone observations do not locate a generated interval on this bank.
The paired candidate remains experimental.
[Native clocks and outcomes](evaluation-panels/video-temporal-insert-native-2026-11-13.json)

## Separate confirmation plan

The 34 equal-cadence labels all list a 125-frame insertion with the same
start. Select the eight smallest remaining original and generated pairs
from the frozen index, excluding all four preflight IDs. The source plan
uses examples 2, 81, 97, 91, 82, 71, 70 and 72 and binds their publisher
member offsets, sizes and CRCs. Their source bytes total 345,702,960.
Use native decode and content fingerprints to check for overlapping
source pictures before authoring. Keep any failed or duplicated group
in the planned denominator. Separate IDs by themselves only establish
disjoint publisher examples.

Generate two deliveries per source from the released generated
segments, with 30-, 54-, 72- and 102-frame inserts across early and
late verified boundaries. Use the same render geometry and compression
for the unchanged controls, authentic inserts, equal-length
replacements and unrelated references. Measure whether a result
binds both physical edges, retains the two earlier-copy flanks, and
preserves unchanged and wrong-source controls. Report findings by
independent source as well as by render. Keep the current paired check's
settings fixed through confirmation. The confirmation roster has been
frozen from publisher metadata and remains unscored.
[Confirmation source plan](evaluation-panels/video-temporal-insert-confirmation-plan-2026-11-13.json)

## Replay

The first command acquires and verifies the 12 exact publisher members.
The next recomputes the native triplets and 16 controlled comparison
results using the local application decoder, FFmpeg and NumPy.
The third independently recounts the public ledger without the media
or scientific packages. Its results agree on Python 3.11 and 3.12. It
rejects altered source ticks, insertion boundaries and evidence times.
The second full native run reproduced the published receipt byte for byte.
Use a new receipt path for each native run.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.video_temporal_insert_native acquire "$media_cache"
backend/.venv/bin/python -B -m scripts.video_temporal_insert_native audit "$media_cache" --write "$new_receipt"
python3 -B -S -m scripts.video_temporal_insert_native review "$new_receipt"
cmp "$new_receipt" docs/evaluation-panels/video-temporal-insert-native-2026-11-13.json
python3 -B -S -m scripts.video_temporal_insert_native confirmation-plan "$intake" > "$new_plan"
cmp "$new_plan" docs/evaluation-panels/video-temporal-insert-confirmation-plan-2026-11-13.json
```
