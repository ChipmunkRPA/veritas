# JPEG compression diagnostic progress

The guarded data preparation froze 59 source photos in 48 known
source groups, each with five views at both JPEG qualities. The first
paired batch at each quality completed on the Mac. Each model scored
25 views per quality and reproduced its first input. An independent
readout verified all 50 matched views, production mask selection,
and source-coordinate measurements.
[Protocol](image-jpeg-diagnostic-v189-protocol.md) · [Exact freeze](evaluation-panels/image-jpeg-diagnostic-freeze-v189.json)

The verified batches account for 100 of 1,180 planned unique model
forwards and four repeats. Eleven paired batches remain per quality.
Accuracy comparisons and any serving decision require the complete
frozen run. Raw maps and per-photo measurements remain local.

## Continued JPEG75 coverage

Four additional JPEG75 batch pairs and their independent mask readouts
passed. Each model has scored 125 of 295 JPEG75 views and 25 of 295
JPEG90 views. The combined coverage is 300 of 1,180 planned unique
forwards and four repeats. Seven JPEG75 and eleven JPEG90 batch pairs
remain under the original freeze.

## JPEG75 coverage complete

All 295 JPEG75 views have now passed both model runs and independent mask
readout. Each model has also scored 25 JPEG90 views. The combined coverage
is 640 of 1,180 planned unique forwards and four repeats. Eleven JPEG90
batch pairs remain before the full aggregate can be sealed.

The JPEG75 batch 6 candidate initially stopped at the 6 GiB RAM admission
check, before the worker launched. The original receipt was retained. The
same frozen batch passed after available memory recovered.

## Both qualities sealed

The remaining JPEG90 phases and terminal seal passed. Both checks scored
295 views per quality. The seal verified 1,180 unique forwards, four exact
repeats and 73 preparation, scoring and mask-readout profiles.
[Complete aggregate](evaluation-panels/image-jpeg-diagnostic-complete-v189.json)
