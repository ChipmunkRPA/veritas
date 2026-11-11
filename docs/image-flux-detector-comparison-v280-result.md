# Image localization on the two licensed photo edits

The sensitive check found **one useful location out of two** on lossless PNG, compared with zero for the fused baseline. It also highlighted five of eight ordinary or untouched controls, compared with one for the baseline. After JPEG75 encoding, neither check nor their shared mask produced a useful location. The sensitive check highlighted every JPEG control. [Frozen method](image-flux-detector-comparison-v280-method.md) · [Exact aggregate](evaluation-panels/image-flux-detector-comparison-complete-v280.json)

| Format | Check | Useful edits | Controls highlighted |
| --- | --- | ---: | ---: |
| PNG | Fused baseline | 0/2 | 1/8 |
| PNG | Sensitive | 1/2 | 5/8 |
| PNG | Shared mask | 0/2 | 1/8 |
| JPEG75 | Fused baseline | 0/2 | 4/8 |
| JPEG75 | Sensitive | 0/2 | 8/8 |
| JPEG75 | Shared mask | 0/2 | 4/8 |

All 20 views completed with 40 unique forwards and four exact repeats. Both formats use the same native edit footprints. The check reports additional JPEG changes separately. The PNG arm used the same fused baseline as the JPEG arm to isolate format effects. The normal PNG product route can use a different baseline. Two source groups provide a descriptive pilot. These results keep the sensitive check experimental and identify transport robustness as a remaining issue. [Source geometry](image-flux-photo-preparation-v276-result.md)
