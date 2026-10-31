"""The `dwtDct` codec of invisible-watermark 0.2.0 (https://github.com/ShieldMnt/invisible-watermark, MIT; `LICENSE`).

- **Vendored:** `imwatermark/maxDct.py` as `max_dct.py`, the class `EmbedMaxDct` unchanged, taken from the PyPI wheel
  `invisible_watermark-0.2.0-py3-none-any.whl` on 2026-09-12. A `ruff: noqa` line was added at the top.
- **Not vendored:** `watermark.py` (the encoder/decoder wrappers; its import of `rivaGan` needs torch and ONNX models),
  `dwtDctSvd.py`, `rivaGan.py` and the ONNX files. The package is not a dependency because it requires `opencv-python`,
  which conflicts with this project's `opencv-python-headless`.
- **Use:** `app/provenance/watermarks.py` decodes with a vectorised equivalent of `EmbedMaxDct.decode`; the tests hold the
  two equal and generate positives with `EmbedMaxDct.encode`.
"""
