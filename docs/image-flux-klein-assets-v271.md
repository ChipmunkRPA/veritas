# Verified reference-edit assets for image testing

The pinned FLUX.2 Klein 4B 4-bit bundle now has 12 byte-verified files, totaling 4,619,700,724 bytes. That covers the five weight shards, three shard indexes, three tokenizer files and the publisher README. Acquisition checked each publisher LFS digest or Git blob before placing the file. The receipt contains file names and digests only. [Byte receipt](evaluation-panels/image-flux-klein-assets-v271.json) · [Pinned model snapshot](https://huggingface.co/mlx-community/flux2-klein-4b-4bit/tree/860e87183ceb29e39627c0612ebd66d8ea66e68c) · [Upstream Apache 2.0 license](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md)

Use the CLI to verify an existing snapshot locally. Add `--download` to retrieve missing files into a caller-owned directory with sufficient space. Both commands check the same pinned digest and byte count for every file. Asset verification requires no private photos, model runtime or account credentials.

```sh
PYTHONPATH=backend python -m scripts.fetch_image_flux_klein_assets_v271 /path/to/snapshot
```

The model uses a reference image to generate a new image. A regional localization evaluation must verify the generated output and apply a predetermined source-coordinate composition. The separate geometry check will determine whether that procedure can produce useful benchmark inputs. No detector score or real-source edit follows from asset verification. [Runtime and license review](generator-components-and-runtime-v40-review.md)
