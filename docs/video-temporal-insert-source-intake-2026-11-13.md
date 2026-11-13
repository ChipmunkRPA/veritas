# Public source intake for inserted video intervals

Umur, Güşlü and Delibasoglu published a general-video benchmark with
100 authentic videos, 100 generated segments and 100 edited copies.
They compared consecutive-frame feature changes against both inserted
regions and untouched videos. Their reported 53.64% frame recall on
the merged set motivates keeping exact location and file-level detection
as separate measures. The repository declares MIT for its code and the
Zenodo deposit declares CC BY 4.0 for the dataset. The analysis below
uses the deposit's metadata and annotations.
[Paper](https://arxiv.org/abs/2606.07090)
· [Authors' repository](https://github.com/OkanUmur/temporally-localized-video-manipulation-detection)
· [Zenodo release](https://doi.org/10.5281/zenodo.20548572)

The intake verified three bounded ranges totaling **152,090 bytes**
from the **14,390,339,732-byte** ZIP. Its 409-member central directory
matches all 400 manifest rows by exact filename and size. The two
annotation CSVs and the readme matched their local ZIP headers and CRCs.
Each of the 100 insertion rows starts at publisher frame 100. Every
end minus start equals the stated inserted length, and each merged
length equals the sum of the original and inserted lengths.

**34 of 100** original and generated pairs are both labeled 24 fps.
The other 66 use mixed cadences. A comparison across those 66 could
respond to resampling or frame-rate changes, so the four smallest
equal-cadence triplets form an initial, deterministic preflight
roster. Together they occupy **202,539,032 bytes**. The receipt binds
their ZIP member offsets, sizes and CRCs. The published insertions
last **62–315 frames** across the full archive. The benchmark tests
general picture insertion, and its positives are much longer than
the app's six-picture face-swap fixtures. Its metadata includes no
per-case speech event or source identity fields.
[Verified source receipt](evaluation-panels/video-temporal-insert-source-intake-2026-11-13.json)

The next native check can fetch only those selected media members,
then verify their packet cadence, exact decoded frame counts, insert
origin and whether the authentic frames and generated insert actually
match the merged clip near both boundaries. Compare the released
encodes with matched decode-and-encode controls before attributing a
hit to edit content. This intake ran zero media decodes or model
calls. It establishes no detection rate.

## Replay

Run with a new local `$intake` directory. The verifier checks each
requested range and its published ZIP entries before counting the
annotations. Python 3.11 and 3.12 produced identical receipts. Flipping
one annotation byte caused the verifier to reject the source.

```sh
set -e
mkdir "$intake"
curl -fLsS --max-filesize 262144 'https://zenodo.org/api/records/20548572' -o "$intake/record.json"
base='https://zenodo.org/records/20548572/files/temporally-localized-video-manipulation-detection-dataset.zip'
fetch_range() {
  part=$1
  curl -fLsS --max-filesize "$4" -H 'Accept-Encoding: identity' --range "$2-$3" \
    -D "$intake/$part.headers" "$base?download=1&chipmunk-part=$part" -o "$intake/$part.bin"
}
fetch_range tail 14390208660 14390339731 131072
fetch_range metadata 14390263862 14390283564 19703
fetch_range readme 14390283565 14390284879 1315
PYTHONPATH=backend python3 -B -S -m scripts.review_video_temporal_insert_sources "$intake" \
  --compare docs/evaluation-panels/video-temporal-insert-source-intake-2026-11-13.json
```
