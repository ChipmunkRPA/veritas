# Active PNG region checks on a separate blended-edit panel

The sensitive PNG check found 2 of 24 known edit locations and highlighted 62
of 72 controls, including 10 of 12 untouched originals. The older recompression
check found zero useful locations and highlighted 12 controls. This result
supports keeping sensitive screening opt-in on the local app. [Audited counts](evaluation-panels/image-active-png-transfer-2026-11-11.json)

This follow-up measures both localizers that ran when the more sensitive option
was enabled. The earlier same-family confirmation found 45/59 useful locations
for the sensitive check versus 10/59 for recompression, with 13/59 and 3/59
untouched-photo flags respectively. This panel asks whether that sensitivity
transfers to small sharp and feathered inserts surrounded by exact original
pixels. [Earlier confirmation](image-native-confirmation-v46-result.md)

The twelve retained photo groups include six SDXL and six Kandinsky groups.
Each contributes two generated insertions and six ordinary controls. Controls
include the untouched photo, local blur and contrast, and local or global
autoencoder processing. These sources had prior project evaluation exposure.
The earlier wavelet study checked this panel against its own selection and
comparison rosters. This run did not establish overlap with the regional
checkpoint's complete training data. This comparison sealed exact PNGs, changed
pixel masks, checkpoint digests, code, and operating procedure before running
the localizers at their existing points. Useful localization requires at least
0.30 intersection over union and 0.50 changed-pixel recall. [Intake and rights](image-wavelet-region-transfer-2026-11-11.md)

| Check | Sharp edits /12 | Feathered edits /12 | Controls flagged /72 | Untouched originals flagged /12 |
| --- | ---: | ---: | ---: | ---: |
| Sensitive regional | 1 | 1 | 62 | 10 |
| Recompression | 0 | 0 | 12 | 1 |
| Wider retained area | 1 | 1 | 63 | 10 |
| Shared retained area | 0 | 0 | 11 | 1 |

Both useful locations came from the sensitive check. Eleven of 12 photo groups
produced a sensitive control flag. The two union and intersection arms use the
actual source-aligned pixels retained by each check, including empty masks.
They measure the spatial trade-off separately from presentation eligibility.
An independent readback matched all 96 image and truth commitments, checked
192 set arithmetic identities, and recomputed all aggregate counts.

All 192 actual CPU forwards completed in 136 seconds. The runner used the
existing model loaders, geometry, confidence-independent area rules and mask
readout. It exercised the detectors offline without running checkout, fusion
or the browser. These results concern this panel and do not supply an
AI-origin probability. The older recompression check and other existing image
checks remain available to ordinary uploads. The sensitive check remains
available through `VERITAS_IMAGE_REGION_MODE=candidate`. [Runner](../backend/scripts/image_active_png_transfer.py)
