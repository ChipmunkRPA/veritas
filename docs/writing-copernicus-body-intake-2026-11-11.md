# Copernicus article-body intake

The paired writing study found a useful location in a small publisher-abstract pilot. This intake looked for a separate body-text panel with actual professional revisions. It froze 20 Copernicus preprint-to-final pairs from the existing official DOI linkage, excluding the scored abstract work and the source-only scout. Both dated versions needed independently recorded CC BY terms and the same authors. The original public discussion manuscript and final publisher XML were the source witnesses. [Earlier paired-writing study](writing-paired-revision-replication-2026-11-11.md) · [Copernicus archive and license policy](https://publications.copernicus.org/for_authors/publication_policy.html)

The intake capped discussion PDFs at 2 MB and final XML at 600 KB per file. It checked each complete pair for one changed 12–40-word body sentence with two exact surrounding sentences on each side. It excluded citation-only and typographic changes. The PDF extractor preserves page and line boxes for any qualifying center. The final XML preserves body block coordinates. Known project source overlap was checked against the same retained comparison packets and the previous paired study.

The final XML was available for all 20 works. Seventeen discussion PDFs exceeded the size cap. The three complete pairs passed the official article, date, and license checks and contained no qualifying edit with the exact geometry. A local replay reproduced the source receipt. No writing generation or detector inference ran. This is a source admission result for the bounded intake. The three complete works cannot estimate how often Copernicus article bodies contain matched edits or how any detector would perform. [Audited intake](evaluation-panels/writing-copernicus-body-intake-2026-11-11.json)

## Replay

Use the retained Copernicus abstract source folder as `base`, the completed body intake directory as `run`, and the prior F1000Research originals file as `f1000`. Supply the six history arguments listed in the public receipt. The source replay reads the saved official files and performs no network access or model inference.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_copernicus_body_holdout verify "$run" "$base" --f1000 "$f1000" "${history_args[@]}"
```
