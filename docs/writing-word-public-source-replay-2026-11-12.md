# Reproduce Word draft evidence from the publisher originals

This replay starts with a clean checkout and ten public PLOS Biology JATS files. It reconstructs the earlier excerpt and the constructed ordinary revision for each article. The ordinary variant replaces one sentence with another genuine sentence from the same article. All other excerpt text stays the same. The replay preserves the article bylines, license, and exact publisher URLs and records the transformations. [Word study](writing-word-boundary-transfer-2026-11-12.md) · [Publisher source and donor receipt](evaluation-panels/writing-semantic-continuity-transfer-2026-11-12.json) · [PLOS license and attribution terms](https://journals.plos.org/plosbiology/s/licenses-and-copyright)

## Run from a clean checkout

Use Python 3.10 or newer. This step requires the Python standard library and roughly 2 MiB for the public XMLs and generated local drafts.

```sh
word_replay="$(mktemp -d)"
python3 -B -I -S backend/scripts/rebuild_writing_word_boundary_sources.py \
  "$word_replay/publisher" --download --native-dir "$word_replay/drafts" \
  > "$word_replay/source-replay.json"

python3 -B -I -S backend/scripts/rebuild_writing_word_boundary_sources.py \
  "$word_replay/publisher" --native-dir "$word_replay/drafts" \
  > "$word_replay/offline-replay.json"

cmp "$word_replay/source-replay.json" "$word_replay/offline-replay.json"
```

The first run downloads the missing XMLs from the official PLOS Biology URLs and checks each SHA-256 before parsing. It also verifies each source's DOI, attribution, CC BY 4.0 license, date, extracted prose, and donor positions against the published receipts. The second run uses the verified local XMLs with no network access. If a publisher delivery changes or goes offline, put the exact recorded XML at the corresponding seven-digit filename in the cache. The tool reports hash changes and leaves existing files intact.

The replay checks 27 earlier case sides and 10 ordinary later case sides. The 37 source sides provide 111 sentence locations and 74 exact sentence or context input hashes. The 17 generated later sides remain private. The public study's arithmetic replay still checks their recorded response directions. [Word response receipt](evaluation-panels/writing-word-boundary-transfer-2026-11-12.json)

## Open a rebuilt pair in the app

The optional `--native-dir` step creates 20 Word files. The Word packages use a deterministic minimal layout, so their package hashes differ from the historical study files. The decoded text, changed words, native paragraph positions, and exact model inputs match. The local verification reopened all ten pairs through the app's comparison code. For an example, analyze `1002474-ordinary.docx` in an enabled local instance, then upload `1002474-earlier.docx` from **Compare with an earlier draft** and select the paired writing check. This ordinary revision triggered the frozen writing response and illustrates a known false alarm.

The replay's JSON lists each DOI, byline, license, transformation, and new Word file hash. Keep it with the derived files to preserve attribution. The publisher files and reconstructed text stay in the reviewer's scratch directory. This replay reproduces source assembly, wording changes and model inputs. Running the model and checking the resulting verdict are separate steps.

```sh
python3 -B -I -S backend/scripts/review_writing_word_boundary_transfer.py \
  docs/evaluation-panels/writing-word-boundary-transfer-2026-11-12.json
```
