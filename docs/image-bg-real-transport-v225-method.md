# BG-REAL publisher transport successor

The v221 attempt reached the pinned Hugging Face resolver for every
planned file. All 108 requests were rejected at the redirect before the
asset body was opened. The original 36 attempts, zero qualified sources
and five exact private receipts remain bound to the published v221
freeze. [v221 freeze](evaluation-panels/image-bg-real-acquisition-freeze-v221.json)

Header-only checks on the original, edited and semantic-mask assets
confirmed the frozen repository revision, a linked asset size and a
64-character linked SHA-256. The resolver returned a signed HTTPS
URL at `us.aws.cdn.hf.co` with path
`/xet-bridge-us/<24 lowercase hex>/<64 lowercase hex>`.

The successor admits that route directly from the exact pinned resolver.
The host and path must match, the signed query must be present, and the
resolver must supply its 64-character linked SHA-256 and bounded linked
size before the AWS request. The AWS route must be terminal. The
unchanged v221 reader verifies body length, SHA-256 and the publisher image
hash. Offline readback checks the saved route path and digest. Signed
query strings stay out of journals.

The ordered v218 roster, 24-primary and 12-backup limits, exact three
asset paths, request limits, timeout, pixel qualification, source exposure
checks and backup order carry through the unchanged v221 implementation.
Only the additional transport and the new private output paths differ.
[v221 method](image-bg-real-acquisition-v221-method.md)

## Commands

From the project root, choose new private trial and cache destinations
separate from the earlier failed study. The local v221 artifacts must
still match their recorded hashes. Preparation opens no media and
performs no network request.

```sh
trial="$PWD/backend/data/independent-evaluation/image-bg-real-acquisition-v225"
cache="/absolute/private/path/bg-real-v225-media"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_transport_v225 prepare "$trial" "$cache"
```

Use the reported SHA-256 as `plan_sha`. Only the acquisition command uses
the network. The seal produces the original-format private qualification
records within the new trial and a separate `summary-v225.json` for
publication. Keep the private journal, paths and images local.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_transport_v225 acquire "$trial" "$plan_sha"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_transport_v225 seal "$trial" "$plan_sha"
```

The result is a pixel-qualification receipt. JPEG localization scoring
and app evidence require a separate comparison on qualified inputs.
