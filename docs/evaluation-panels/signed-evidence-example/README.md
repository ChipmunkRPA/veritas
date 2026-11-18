# Verify a signed evidence package offline

This example gives you a signed [evidence package](evidence.zip), its separate
[public key](example-public.pem), and the [original image](authored-gradient.png).
The project authored a 160 × 120 gradient and assigned one example inspection
region. The archive contains a signed example PDF, JSON evidence and an
unannotated crop. It was built with the app's ordinary package builder, using
the [authored image regression fixture](../../../backend/tests/test_image_packets.py).
No detector ran. The example tests whether the package and a region agree with
the original file. It provides no evidence about detection accuracy.

Run these commands from the repository root with Python 3.11 or later. Use an
existing Python environment with `cryptography` and `Pillow` by creating a scratch
directory and setting `review_python` to its interpreter. The command below
sets up an environment if you need one. Network access may be needed to install
the packages. Verification runs offline, with no account, service, model files
or database.

```sh
review_dir=$(mktemp -d)
python3 -m venv "$review_dir/venv"
review_python="$review_dir/venv/bin/python"
"$review_python" -m pip install cryptography Pillow
```

Pin the example public key and verify the original and the crop.

```sh
example=docs/evaluation-panels/signed-evidence-example
PYTHONPATH=backend "$review_python" -B -m app.cli verify "$example/evidence.zip" \
  --public-key "$example/example-public.pem" --source-image "$example/authored-gradient.png"
```

Expect `VERIFIED` and a passing `original image crops` check. It should report
one matching unannotated crop. The crop starts at `(16, 6)` and covers 88 × 88
pixels. Removing the key argument should produce `SELF-CONSISTENT` with exit
status 3 because the included key has not been independently pinned. Real
reviews should obtain the instance public key separately and confirm who
supplied it. This public key is for this example only.

Change a scratch copy to confirm that valid package signatures alone cannot
authenticate a different original. This appends a byte to the example PNG.
The original-file check should fail, with exit status 1.

```sh
cp "$example/authored-gradient.png" "$review_dir/changed.png"
printf 'X' >> "$review_dir/changed.png"
PYTHONPATH=backend "$review_python" -B -m app.cli verify "$example/evidence.zip" \
  --public-key "$example/example-public.pem" --source-image "$review_dir/changed.png"
```

To report a reproduction or discrepancy, append `--json` for a machine-readable
receipt. Include the checkout revision, Python version, exact command and the
check you reproduced or could not reproduce. The example receipts can be
shared. For your own reviews, check your receipt and keep the underlying media
private unless you have permission to share it. [Independent evaluation issue form](https://github.com/ChipmunkRPA/veritas/issues/new?template=independent-evaluation.yml)

| File | SHA-256 |
| --- | --- |
| `evidence.zip` | `c2bd1a771f17b0406e25469641d0ee95da23d1f4c06b0223e35c44ee98382de3` |
| `example-public.pem` | `74b55a4237c5c6745b4ba20701657ac84ec2c2a1ecfa3d0d9d9e919fc97f0b6f` |
| `authored-gradient.png` | `94d88f978d7bff90a5e60b5f34ed52191faa51b98164c487743f214600c44e92` |
