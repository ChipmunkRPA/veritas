# Fresh eLife science-explainer intake

Use every item labeled Insight in the eLife Magazine archives for May, June, August, September and October 2022. This defines 35 publisher works before article body inspection. The July 2022 archive is already exposed through v272. Fix the complete ordered list of DOI suffixes, dates and author names against a single immutable eLife JATS commit before acquiring these works. Use version 1 of each work. A missing or rejected item stays in the denominator. The target is at least 15 eligible article groups. [May](https://elifesciences.org/archive/2022/may) · [June](https://elifesciences.org/archive/2022/june) · [August](https://elifesciences.org/archive/2022/august) · [September](https://elifesciences.org/archive/2022/september) · [October](https://elifesciences.org/archive/2022/october) · [Pinned publisher XML](https://github.com/elifesciences/elife-article-xml/tree/bdfb477443c50c8c7ca015ad19fb7b7874c672f5/articles)

Verify each work's type, DOI, exact date, ordered author names and explicit CC BY 4.0 license in its publisher record. Apply the sealed v272 body extraction and excerpt rules. Retain the first eligible continuous body window and the predetermined interior rewrite location with half-open Unicode code point offsets. Record raw XML, license notice, normalized body and excerpt commitments. The publisher license permits commercial adaptation with attribution and identification of edits. [Source extraction](text-elife-source-intake-v272-method.md) · [Publisher terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Repeat the existing 23 writing exposure checks and add the normalized bodies of all nine July publisher records and the six prepared v273 rewrites. Reject exact normalized 12-token overlap with known inputs or between new eligible articles. Exclude author names and linked research DOIs shared with July or with another eligible article in this roster. The resulting groups are distinct within the retained source material and recorded metadata. This procedure leaves paraphrase overlap and model pretraining exposure unmeasured. Raw passages stay under ignored local evaluation data. The public receipt contains source attribution, eligibility, coverage and hashes. This step makes no generator or detector calls. [July admissions](evaluation-panels/text-elife-source-intake-complete-v272.json) · [Prior rewrite exposure](evaluation-panels/text-elife-rewrites-complete-v273.json)

Save the roster before downloading any version-1 files. The retrieval list has at most 35 URLs and a 256 kB cap per file. The local intake can then replay against the exact July XML cache and GoHumanize packet.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
v286_dir=$(mktemp -d /private/tmp/chipmunk-elife-v286.XXXXXX)
backend/.venv/bin/python -B -m scripts.text_elife_transfer_intake_v286 roster > "$v286_dir/roster.json"
backend/.venv/bin/python -B -m scripts.text_elife_transfer_intake_v286 urls "$v286_dir/roster.json" > "$v286_dir/urls.tsv"
while IFS=$'\t' read -r name url; do
  if curl --fail --silent --show-error --proto '=https' --connect-timeout 8 --max-time 20 \
    --max-filesize 256000 "$url" -o "$v286_dir/$name.partial"; then
    mv "$v286_dir/$name.partial" "$v286_dir/$name"
  fi
done < "$v286_dir/urls.tsv"
backend/.venv/bin/python -B -m scripts.text_elife_transfer_intake_v286 intake \
  "$v286_dir/roster.json" "$v286_dir" /path/to/exact/july-xml-cache /path/to/pinned/gohumanize/test.jsonl --retain
```
