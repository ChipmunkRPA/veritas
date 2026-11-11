# Fresh writing evaluation with ordinary editorial controls

Use every Insight in the January through April 2022 eLife Magazine archives.
The 26 fixed works are dated before the public launch of ChatGPT. The split
hashes each DOI with a fixed salt, puts the first 13 in development and the
other 13 in confirmation, and assigns each article and all its variants to
that role before retrieving the body. Exclusions stay in their original role
without replacement. Each role needs at least seven admitted groups and at
least four groups with an ordinary edit to reach the next stage. [January](https://elifesciences.org/archive/2022/january) · [February](https://elifesciences.org/archive/2022/february) · [March](https://elifesciences.org/archive/2022/march) · [April](https://elifesciences.org/archive/2022/april)

Use the same immutable publisher XML commit and version-1 files as the earlier
intakes. Verify each work's DOI, type, date, ordered authors and own explicit
CC BY 4.0 notice before admitting prose. Retain attribution and identify
adaptations. Reuse the earlier body and excerpt rules, including the fixed
interior paragraph location and Unicode code point offsets. Hold sources
sharing an author or linked research DOI with an earlier or another eligible
work. Screen exact normalized 12-token fragments against the retained exposure
inventory, the v272 and v286 publisher bodies, the v273 and v289 rewrites,
and across the new candidate groups. The v292 evaluation used those same v289
inputs. This screens known exact overlap and recorded author and work IDs.
Pretraining exposure and paraphrase similarity remain unmeasured. [Publisher terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Body selection and historical inputs](text-elife-source-intake-v272-method.md) · [Prior confirmation](text-elife-relative-paragraph-v292-result.md)

Each admitted group has the unchanged excerpt and two planned ordinary-edit
controls. Apply up to four fixed copyedits in text order within the target
paragraph. For the second control, substitute the closest-length eligible
paragraph from the same publisher work outside the excerpt. A donor must have
24–120 words and 75–125% of the original paragraph's word count. Use the
earliest body position to break a tie. Preserve both flanks exactly, record the
actual ordinary edit intervals and record zero generated spans. A case with
no allowed copyedit or donor is unavailable. The splice is a constructed
human-prose negative that can create a topic or style boundary. These controls
do not estimate false positives on naturally revised articles.

The next experiment should compare the current passage check against an
independent local transition cue, then compare the combined result with both
components. Measure whether a paragraph's length-matched lexical repetition
and surface style differ from the adjacent context. A transition describes a
change in the writing and can occur in ordinary editing. Use the unchanged
and ordinary-edited controls to measure false highlights separately. Freeze
the cue, both generator families and the confirmation criteria before scoring.
Choose any operating point using development only, then lock it before the
first confirmation response. Report useful interval overlap, incorrect
highlights, unchanged characters highlighted and coverage by role, edit type
and generator. The v301 intake computes no cue or classifier scores and makes
no generator calls. [Mixed-authorship boundaries](https://aclanthology.org/2026.findings-eacl.326/) · [Context dependence of linguistic features](https://aclanthology.org/2026.findings-aacl.10/)

Save the canonical roster before retrieving the 26 bounded version-1 XML
files. Preserve incomplete acquisitions as missing. The public aggregate
contains source metadata and hashes. The article passages and controls stay
under ignored local evaluation data.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
v301() { backend/.venv/bin/python -B -m scripts.text_elife_editorial_controls_v301 "$@"; }
v301_dir=$(mktemp -d /private/tmp/chipmunk-elife-v301.XXXXXX)
v301 roster > "$v301_dir/roster.json"
v301 urls "$v301_dir/roster.json" > "$v301_dir/urls.tsv"
while IFS=$'\t' read -r name url; do
  if curl --fail --silent --show-error --proto '=https' --connect-timeout 8 --max-time 20 \
    --max-filesize 256000 "$url" -o "$v301_dir/$name.partial"; then
    mv "$v301_dir/$name.partial" "$v301_dir/$name"
  fi
done < "$v301_dir/urls.tsv"
v301 intake "$v301_dir/roster.json" "$v301_dir" /path/to/exact/july-xml-cache \
  /path/to/exact/transfer-xml-cache /path/to/pinned/gohumanize/test.jsonl --retain
```
