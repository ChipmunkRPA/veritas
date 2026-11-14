# Animated PNG opening comparison

An animated PNG can carry a static fallback that is separate from the
animation. A still-image decoder can show that fallback while an animation
player opens with a different picture. Chipmunk Verify compares the fallback
and first animation frame for a bounded set of PNGs and shows the changed
pixel count, source bounds and both previews in the analysis and PDF. The JSON
export and signed report package bind the measurement and preview hashes to
the uploaded source. [PNG specification](https://www.w3.org/TR/png-3/#apng-frame-based-animation)

The check accepts 8-bit RGB or RGBA PNGs with a separate fallback and a
full-canvas first frame using replacement blending. It measures the two views
on white with the same EXIF orientation. It accepts up to 24 MiB, 4,096
pixels per side, and 4,194,304 pixels in total. Previews fit within 320 pixels
per side. The pixel count and coordinates are taken before preview resizing.
The frame count comes from the header. Later frames, partial opening frames
and opening frames using over-blending are outside the comparison.

Separate fallback art can be legitimate and carries zero AI-confidence
weight. New still-image analyses with multiple stored images omit the
combined AI confidence and explain that the statistical checks cover the
first stored image. Signed provenance can still describe the file when a
trusted source record is available.

## Local verification

Three authored images passed the simulated prepaid analysis flow, summary
and standard PDFs, JSON export, and signed-package verification from the
uploaded PNG. The different fallback case located all 4,800 changed
pixels at x 110, y 70, with bounds 80 × 60. The matching fallback case
reported zero changes. A normal animation without a separate fallback
produced no fallback finding. The customer's combined confidence was
unavailable for all three. These are format and coverage tests.

The tests also check EXIF rotation, differences hidden by full transparency,
unsupported blending, truncated input, and swapped preview identities.
They require no downloadable models or private footage.

```sh
cd backend
.venv/bin/python -m pytest tests/test_image_animation.py
```
