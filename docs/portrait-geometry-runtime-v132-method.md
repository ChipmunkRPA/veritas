# Isolated face geometry runtime for a second generator

Install four exact publisher wheels in a new research-only directory:
MediaPipe 0.10.32, absl-py 2.3.1 and FlatBuffers 25.9.23 (Apache-2.0), and
sounddevice 0.5.3 (MIT). Bind the retained official PyPI metadata, platform wheel
filenames, sizes, SHA-256 and MD5 before acquisition. These are the same previously
reviewed face-task API version, with explicitly pinned supporting packages.
No source build or automatic dependency resolution is allowed.

Download sequentially, verify full expected bytes/MD5/SHA-256 and retain failures.
Install using the application interpreter's pip with `--no-deps --no-index
--no-compile --target` pointing only at the exclusive research directory. Preserve
original wheel license/metadata files. Verify every original installed wheel
member, including native libraries and Python, against the verified wheel;
exclude only pip's regenerated RECORD. Reject unsafe paths or unplanned `.data`
layouts. Record the complete installed manifest privately and publish its hash.

Do not install MediaPipe's declared OpenCV-contrib dependency. Reuse existing
OpenCV-headless, NumPy, CFFI, protobuf and matplotlib, binding their versions and
METADATA bytes before and after. This is an explicit API compatibility probe,
not a claim that all declared dependencies or `pip check` are satisfied. Insert
the research site only in the worker process. Verify exact versions/import paths,
FaceLandmarker/CPU API availability and reuse of existing OpenCV. Do not instantiate
a face task, render a frame, load neural weights, query audio devices, record audio
or import InsightFace. The audio package is an import dependency only.

Freeze and manually publish the tested implementation and lock first. Run the
single data-only stage outside the sandbox after other guarded stages finish:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.portrait_geometry_runtime_v132 run
```

Use the qualified unchanged owned-descendant monitor: 900 seconds, 1.5 GiB sampled
descendant RSS, 3 GiB initial RAM/project/archive disk and 1.5 GiB running reserves.
No retries, overwrites or relaxed limits. A later neural qualification has its
own 6 GiB initial RAM/RSS limits and must test actual CPU geometry, repeatability,
strict checkpoint loading and native-coordinate composites before generator use.
No customer model/default, payment, Actions, outreach or deployment changes.
