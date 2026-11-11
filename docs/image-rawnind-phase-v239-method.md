# Native camera phase feasibility replay

The study fixes four genuinely photographed RAW captures, two Sony ILCE-7C scenes and two Canon EOS D60 scenes. It measures support for the local Bayer-phase cue on developed originals and five ordinary processing variants. The four captures are an engineering feasibility sample. This study has no generated edits, operating-point fitting, AI-origin verdict, or customer route. [Cue implementation](../backend/scripts/image_bayer_phase_v237.py)

The first-party RawNIND V1.0 catalog and author scene sidecar fix the complete clean/noisy capture grouping, source IDs, lengths, and publisher checksums. Select known-sensor scenes outside the authors’ reserve with both clean and noisy memberships. Take one clean file per scene, then two scenes in each capture format using the fixed hash order. The roster total is 111,284,720 bytes. Keep the exact IDs, names, per-source hashes, and outputs in an owner-only folder and publish whole-roster commitments and aggregate observations. The dataset license permits commercial use under CC BY-SA 4.0, with attribution and ShareAlike obligations for redistributed adaptations. [Institutional release](https://doi.org/10.14428/DVN/DEQCIM) · [RawNIND paper](https://arxiv.org/html/2501.08924v1)

After review, run each stage separately from the repository root with the existing CPython 3.12 environment on macOS ARM64. Use a fresh absolute owner-only study path under private storage, with permission enforcement enabled.

```sh
PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/python -B \
  -m scripts.image_rawnind_phase_v239 freeze /absolute/private/rawnind-v239
PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/python -B \
  -m scripts.image_rawnind_phase_v239 acquire /absolute/private/rawnind-v239
PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/python -B \
  -m scripts.image_rawnind_phase_v239 replay /absolute/private/rawnind-v239
```

- `freeze` uses the versioned Dataverse catalog and single checksum-bound metadata sidecar. Both bodies and the reconstructed roster must match their exact SHA-256 commitments.
- `acquire` makes up to four direct original-file requests and one official wheel request. It verifies each RAW size, MD5 and SHA-1 before writing. Existing bytes are revalidated before resuming. Transfers have no redirects or automatic retries.
- `replay` rehashes the inputs, checks the wheel platform and license notices, then unpacks it into an isolated temporary directory for that process. The project environment is left unchanged. The exact PyPI rawpy 0.28.0 CPython 3.12 macOS ARM64 wheel is 2,060,964 bytes, with SHA-256 `60e4c82d2dbce6ae20f95d0894bcacf355d32b783cd017c374c800dfc4a0d9b3`. Maik Riechert’s rawpy wrapper is MIT. The bundled LibRaw library is offered under LGPL 2.1 or CDDL 1.0. Retain both packaged license notices with any binary redistribution. [Official wheel](https://pypi.org/project/rawpy/0.28.0/#files) · [LibRaw terms](https://www.libraw.org/about)

Require a native 2×2 Bayer CFA and use linear demosaicing, as-shot white balance, no automatic brightness, no noise reduction, no rotation or resize, and fixed sRGB 8-bit development. The ordinary variants use JPEG 90 and 75 with 4:2:0 chroma, an odd-offset crop, a 0.75× area resize, and mild unsharp masking. Each variant is independently derived from the developed source. Publish counts and distributions for phase support, abstention, and residual mismatch. The four captures reuse the same two camera models and each set of variants shares one capture. [Decoder settings](https://letmaik.github.io/rawpy/api/rawpy.Params.html)
