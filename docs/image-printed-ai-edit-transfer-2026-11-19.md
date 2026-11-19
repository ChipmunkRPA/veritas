# Locating AI-edited wording in infographics

Neither the current regional image check nor a text-line ranking found a
useful edit location in 12 fresh infographic pairs. The text-line ranking
pointed to much the same place before and after the edit in nine pairs.
This is a separate test from the exact printed identifier comparison.

The examples come from the publisher's English MaskEdit-10k test split.
The pinned media and masks are available under MIT terms. Intake selected
12 distinct source IDs using the publisher's edit description and mask
area. All 12 edit pairs showed a visible wording change inside the mask,
including a removed subtitle. Both sides were AI-generated. Changes also
appeared outside the declared mask, so the pair serves to test the target
intervention. It supplies no camera-original false-alarm rate. The
publisher's masks approximate the selected edit area.
[Dataset and terms](https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k#licensing-and-rights) ·
[Publisher paper](https://arxiv.org/html/2608.06929v1)

Each file was delivered at the current check's supported JPEG quality 90
and 768-pixel longest side, with a matched quality-75 repeat. The current
regional check applied its existing region selection and repeat rule. The
new ranking used rectangles around recognized words on the delivered pixel grid and
ranked the line with the largest local contrast between its regional
response and its surrounding ring in both delivery views. This second
reading tested ranking before choosing an operating point. A useful box
or mask needed to cover half the publisher mask with at least 0.30 pixel
intersection over union.

| Reading | Measurable edited files | Useful edited locations |
| --- | ---: | ---: |
| Current JPEG region selection and repeat | 12 / 12 | 0 / 12 |
| Text-line ranking | 12 / 12 | 0 / 12 |

The current check highlighted a region in five edited deliveries and
three pre-edit deliveries. One source file's top-ranked text
rectangle already coincided with the future edit area. The 24 normalized
delivery files and all 24 candidate rectangle overlaps were independently
recounted against the masks. The result leaves the text-line ranking out
of the app and identifies unchanged headings as important controls for
training a future text-centric localizer.
[Pinned source indices and numerical receipt](evaluation-panels/image-printed-ai-edit-transfer-2026-11-19.json)
