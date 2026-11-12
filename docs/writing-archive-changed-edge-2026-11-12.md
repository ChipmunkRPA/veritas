# Locate a changed word at a lost paragraph break

A plain-text copy can lose its paragraph breaks. The existing archive comparison can recover them when the opening and closing tokens of each paragraph match. It skipped every case in this panel with one changed word near an internal break. A second alignment located the exact changed word in all 24 cases, with the expected archive citation and the original word on the archive side. [Independent result](evaluation-panels/writing-archive-changed-edge-2026-11-12.json)

The second alignment considers consecutive indexed paragraphs that account for every submitted token. It accepts one unique chain with exactly one substituted token within eight tokens of an internal paragraph boundary. The candidate must also pass the existing per-paragraph fingerprint check. The recovered positions count Unicode code points in the submitted extracted text. The result measures a difference from a specific archive version and carries zero AI-authorship weight. Recent work on source-aligned writing evaluation likewise distinguishes locating a change from assigning its origin. [Archive method and publisher terms](writing-archive-source-evidence-2026-11-12.md) · [MixDetect research](https://arxiv.org/abs/2609.32625)

The 12 held documents came from four pinned U.S. State Department volumes. All documents have distinct recorded senders within the held panel. Sender identities recur in the historical panels. No selected work or exact 12-word sequence overlapped the earlier indexed evaluation sources. Each positive replaces one word in otherwise unchanged publisher prose. The experiment tested authored source differences and their coordinates.

| Input | Cases | Previous exact recovery | With one-edge alignment |
| --- | ---: | ---: | ---: |
| Collapsed unchanged source | 12 | 12 matched, zero changes | 12 matched, zero changes |
| Changed word before an internal break | 12 | 0 located | 12 exact locations |
| Changed word after an internal break | 12 | 0 located | 12 exact locations |
| Multiple edits, insertions, reordering, mixed sources, added introductions, quotes, or duplicate references | 84 | 0 source claims | 0 source claims |

The separate standard-library verifier reread all 120 frozen decisions and the bundled public archive. It checked 216 exact paragraph spans and both sets of coordinates for all 24 changed words. The follow-up comparison is intentionally bounded to a single substitution in the tested layout. Source material and private case records remain local. [Frozen comparison runner](../backend/scripts/writing_archive_changed_edge.py) · [Independent verifier](../backend/scripts/review_writing_archive_changed_edge.py)

## Local app

The existing public archive comparison now uses this alignment if exact paragraph-edge matching supplies no candidate. The normal app function reproduced all 120 frozen decisions and their submitted and archive coordinates. Three isolated analyses exercised checkout with simulated payment, the normal document worker, and authenticated JSON and PDF. Both changed-word cases returned the expected exact spans. The two-substitution control returned no match. Each unpaid event queued zero analyses. In Chrome, the plain result highlighted the original and changed words, and the downloaded JSON matched the API. [Local app receipt](evaluation-panels/writing-archive-changed-edge-app-2026-11-12.json)

## Replay

Use the study commit, the retained source directory from the [archive study](writing-archive-source-evidence-2026-11-12.md), and the [previous reflow study](writing-archive-joined-layout-2026-11-12.md). Set `run` to a new local directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
for stage in freeze measure replay; do
  backend/.venv/bin/python -B -m scripts.writing_archive_changed_edge "$stage" "$parent" "$previous" "$run" || exit 1
done
python3 -I -S backend/scripts/review_writing_archive_changed_edge.py \
  "$run" backend/app/evidence/data/frus-archive.json "$run/independent.json"
```
