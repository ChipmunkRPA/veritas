# Image localizer v6: source and commercial-asset review

Checked 6 November 2026. These are acquisition and licensing observations,
not model evaluations. No new candidate checkpoint was loaded or scored during
this review. Existing inpainting weights remain available for a separate crop
experiment. No serving setting was changed.

| Candidate | Primary evidence | Decision before inference |
| --- | --- | --- |
| SAFIRE | [Maintainer notice at ec2af6a](https://github.com/mjkwon2021/SAFIRE/blob/ec2af6af8c789fb6359493a3e352f89c6fd8107f/README.md) grants Apache-2.0 code and CC-BY-4.0 weights, with attribution and retained third-party obligations. | The linked checkpoint returned an owner/editor-only download refusal. Preserve that restriction; no weights acquired. This repeats a download obstacle already recorded in the earlier Mesorch architecture decision. |
| CAT-Net v2 | [Maintainer notice at 5eb8590](https://github.com/mjkwon2021/CAT-Net/blob/5eb85900b0e72423c95a8d41bc835b86c8671f8a/README.md) explicitly permits commercial use of CC-BY-4.0 weights; code is Apache-2.0. HRNet retains MIT and bundled SyncBN retains BSD-3-Clause. | The current linked v2 checkpoint also returned an owner/editor-only refusal. No weights acquired or alternate endpoint attempted after the refusal. |
| DetectiveSAM | [GitHub code license](https://github.com/Gertlek/DetectiveSAM/blob/2ca86f12f6bab76ed693ddf105f7baa23f10b991/LICENSE) is MIT. Its separately hosted [weight card](https://huggingface.co/Gertlek/DetectiveSAM/blob/a90e147e985ef8f532d3b38d9319e494275201a9/README.md) still declares `other` and unresolved release terms. | Defer commercial admission: do not apply the code grant to the separately distributed weights despite the more permissive GitHub README. No weights acquired. |

The anonymous HTML refusal SHA-256s are
`77cacfd00216dd8a30003e624b34af904516816090de6a2dcc0a5b0480a44bd0`
for SAFIRE and the CAT-Net refusal commitment recorded in the JSON receipt.
These hashes identify the retained local observations; they are not publisher
signatures. The public record includes URLs and digests, not cookies or account
credentials. Anonymous clients supplied no credentials. No download restriction
was bypassed, and no message was sent to a maintainer.

The existing [FUSED weight card](https://huggingface.co/aonuzhdin/FUSED/blob/b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7/README.md)
continues to declare MIT (card SHA-256
`27f2530bf24ab7ef2dced14b83b483a7a51eee9d31a9d8ec445561ef337635c1`).
Its separate upstream architecture source is not treated as licensed by that
card; the application uses its existing independently implemented runtime and
the Apache-2.0 semantic backbone. See [the original compatibility review](fused-local-research.md).
The [crop development protocol](fused-crop-v6-protocol.md) tests a different
method using those already pinned assets. It establishes no accuracy gain,
commercial clearance for unrelated datasets, or completed outside review.
