# Mixed-writing source and position audit

6 November 2026. Before attempting another writing detector or confidence
mapping, audited the complete previously acquired AITDNA span archive and all
historical mixed samples. No new classifier inference or fitting occurred.
Whole-article confidence remains withheld; the current email scope is unchanged.

## Actual preparation result

All 362 source records passed their declared character-provenance and extraction
coordinate checks. Every non-whitespace extracted Unicode codepoint has source
provenance; normalization that cannot preserve it is refused. Human-only metadata
cannot silently contain retained Bot-labeled characters. Historical mixed sources
are rebound to their original text and SHA-256 before exclusions are counted.
Failures would remain visible and block the overall preflight pass.

The earlier evaluation used 40 different authors. The unused-author pool has
202 records across 59 other authors, with no normalized exact-text reuse against
the original 680-file manifest. This includes 138 mixed-authorship records and
53 records declared human-only. The remaining eleven include five Bot-only
records and six with only retained User characters despite assistance metadata;
the latter must not be mislabeled as retained AI writing. Five generator labels
are represented in the pool.

This is an inventory, not a selected or frozen confirmation set. Shared prompts,
near duplicates, author/topic grouping across other historical text panels,
per-generator representation and pretrained-corpus overlap still need review.
The original 40 documents and 674 sentence-window results remain consumed data.
The earlier context-level AUC is not a calibrated probability for each focus
sentence or evidence that individual highlighted words were AI-written.

The [aggregate receipt](evaluation-panels/writing-v2-source-preflight.json)
omits source text, author names, per-record span labels and queries. The complete
private receipt preserves every row's status. Nine focused regression tests
passed, including Unicode/paragraph offsets, historical binding, reused authors,
exact-text exclusions, conflicting labels and failure denominators.
The separate evidence/export/independent-review selection passed 110 tests; one
OCR fixture test was skipped because its verified local source was not provisioned
in the isolated checkout. This does not establish a new OCR or detector result.

## Publisher terms verified at the pinned revisions

Fetched the exact dataset card at
[`21d77f7d44833507750c0da309295c4147c1bda5`](https://huggingface.co/datasets/UKPLab/AITDNA/blob/21d77f7d44833507750c0da309295c4147c1bda5/README.md),
SHA-256 `23c9f9af9dde11accd92ec3d94c87f190ceafda209d9749c3995c44a98010cb0`.
Its metadata declares `cc-by-sa-4.0` while the prose declares `gpl-2.0`.
The older study recorded the metadata declaration; that did not resolve this
conflict. Record both rather than assigning an unambiguous source license.
Do not use this audit to authorize new training or redistribution of source text
or annotations. The public receipt contains aggregate checks, not that content.

The exact installed detector's card at
[`f1795c86806e6838d4afa33d0b1427f8430c9615`](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/blob/f1795c86806e6838d4afa33d0b1427f8430c9615/README.md)
consistently declares Apache-2.0 and commercial use. Its card SHA-256 is
`b511193473ce1507ca578c465561bd8ea06f37df8549d972beaad8b9143350a7`.
No additional pretrained checkpoint is downloaded, selected or executed here.

## Institutional archive follow-up, 6 November

Also checked the author's [TU Darmstadt repository item](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5170)
and its linked 8,730,233-byte dataset archive. The download matches the
repository's published MD5, and every ZIP member passes its CRC check. Item
metadata explicitly declares CC-BY-SA-4.0, but the bundled README still says
`gpl-2.0`. This additional owner-hosted release therefore retains the same terms
conflict; do not silently choose the more convenient declaration.

The institutional span Parquet has different file bytes from the acquired
Hugging Face copy, but its Arrow schema and all 362 decoded records match
exactly. This supplies independent transfer/provenance evidence, not independent
human review or detection validation. No source was replaced and no classifier
forward or training ran. The [aggregate receipt](evaluation-panels/writing-v2-institutional-source-review.json)
publishes archive/card/checksum commitments and the equality result, without
text, queries, author identifiers or per-record annotations. A new commercially
permitted training-data decision remains necessary.

## Reproduce the audit

Acquire the pinned span archive and original manifest separately. Retrieve the
two publisher cards at the revisions above as `aitdna-card.md` and `tropa-card.md`.
With the existing local Python environment:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/audit_writing_sources_v2.py \
  /path/to/ait-span.parquet /path/to/original-manifest.json \
  /path/to/pinned-cards /path/to/new-source-audit.json
```

The script refuses changed source/card bytes and output overwrite. Its tests are
`backend/tests/test_writing_source_preflight_v2.py`. This local source audit is
neither a detector improvement nor participation by an independent reviewer.

Next, complete the connected-source and terms review, then freeze a comparison
of actual scored contexts and focus sentences on consumed development cases.
Any proposed localization change needs measured overlap/false-highlight gains,
source-disjoint confirmation and evidence/export/paid-flow checks before adoption.
Do not restore whole-document confidence from raw passage responses.
