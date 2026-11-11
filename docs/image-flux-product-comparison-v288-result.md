# Product image localization on six additional edited photos

The current PNG route produced **zero useful regions on six edits**. The sensitive check produced one useful region and highlighted 18 of 24 controls, compared with six for the current route. On JPEG75, neither check produced a useful region. The shared JPEG mask reduced control highlights from 13 to six, while touching one known edit instead of four. These tradeoffs keep the sensitive check and shared mask experimental. [Frozen method](image-flux-product-comparison-v288-method.md) · [Exact aggregate](evaluation-panels/image-flux-product-comparison-complete-v288.json)

| Format | Check | Edits with any overlap | Useful regions | Controls highlighted |
| --- | --- | ---: | ---: | ---: |
| PNG | Current route | 1/6 | 0/6 | 6/24 |
| PNG | Sensitive | 3/6 | 1/6 | 18/24 |
| PNG | Shared mask | 1/6 | 0/6 | 4/24 |
| JPEG75 | Current route | 4/6 | 0/6 | 13/24 |
| JPEG75 | Sensitive | 1/6 | 0/6 | 7/24 |
| JPEG75 | Shared mask | 1/6 | 0/6 | 6/24 |

All 60 views completed with 120 unique forwards and four exact repeats. Both formats use the same exact native RGB truth. JPEG75 added 143 to 254 changes outside the native edit footprint on the positive views. The current product uses different routes for PNG and JPEG. The sensitive check ran on JPEG as an offline transfer probe. Six source groups give a descriptive comparison across both edit placements. [Source construction](image-flux-photo-expansion-v285-result.md)
