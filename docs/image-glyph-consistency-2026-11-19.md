# Repeated-digit shape cues in screenshots

This pilot looks for one glyph whose shape differs from repeated
instances of the same recognized digit on the same line. It measures a
visible inconsistency in a submitted screenshot. The authored examples
test localization of conventional edits, with legitimate formatting and
processing controls. No AI-generated document or AI-authorship labels enter
this pilot. Recent work on AI-inpainted documents shows why a separate test
on real AI edits matters. The publisher's noncommercial material was
excluded. [AIForge-Doc v2 paper](https://arxiv.org/abs/2604.25213) ·
[Publisher dataset terms](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2)

The submitted image supplies the glyph boxes through the pinned local OCR
reader. The measurement compares a native crop with a consensus of at least
three peers. It normalizes ink contrast and shape, removes small alignment
differences and, in the second arm, adds an aspect-ratio term. It then selects
the strongest eligible glyph across the file. Truth masks come from exact
changed pixels in the paired PNG before delivering either PNG or JPEG.
Useful localization requires at least 50% changed-pixel recall and 0.30
intersection over union.

Six authored inventory layouts supply development points. Twelve other layouts
use other base font files and text placement, including half rendered at
double size before downsampling. Each source receives two single-glyph
substitutions and five ordinary variants. The controls include unchanged
content, a different font in a whole column, changed color, legitimate
single-glyph emphasis, and local blur. Within each format the operating
point is the second-highest ordinary development group maximum. The rule
and material were sealed before scores and the points before held evaluation.

| Held format | Reading | Useful single-glyph substitutions | Ordinary deliveries flagged | Ordinary layout groups flagged | Misplaced highlights on edited files |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Ink shape | 2/24 | 0/60 | 0/12 | 0/24 |
| PNG | Ink shape and aspect | 7/24 | 2/60 | 1/12 | 0/24 |
| JPEG | Ink shape | 0/24 | 5/60 | 2/12 | 2/24 |
| JPEG | Ink shape and aspect | 0/24 | 13/60 | 3/12 | 6/24 |

The second PNG reading met the pilot's criterion for a separate follow-up
on actual AI edits. It found six font substitutions and one shape
substitution, across seven held layouts. The two ordinary alerts came from
the same layout with legitimate column and single-glyph emphasis. JPEG
induced unrelated highlights even though every delivery produced an eligible
glyph measurement. The method stays experimental. Repeated digits on an
inventory page are a deliberately favorable setup, and the result says
little about handwriting, sparse text, invoices, or PDF page photographs.

A follow-up checked all 2,506 metadata rows of the English-infographic
preparation split at the pinned MaskEdit-10k revision. To qualify, the
opening instruction needed a numeric-field term after an editing verb,
along with a written digit. The pinned Parquet column reads found zero
qualifying instructions. The source and target infographics in that split
are both generated. The check used 2.03 MB of range reads for metadata
and loaded no candidate image, so these counts establish the intake
coverage of that selector. [Pinned publisher card and terms](https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k/tree/3e0c54f7a6aeca5fb3e12758cb93d0be14310f55)

The native audit rechecked all 252 input files and masks, independently
reconstructed each changed-pixel mask, and verified all 504 ranked region
areas and overlaps. The scalar receipt gives the group, role, carrier, score,
region area and intersection without images, text, or native positions.
[Scalar results](evaluation-panels/image-glyph-consistency-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_image_glyph_localization.py \
  docs/evaluation-panels/image-glyph-consistency-2026-11-19.json
```
