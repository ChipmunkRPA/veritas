# Review recorded writing responses beyond the passage cards

New document analyses show up to 12 complete, source-mapped classifier responses that have no selected passage card. The ordinary results view opens the exact input text for each response. The Standard PDF records its direction and source ranges, with a short excerpt. JSON contains the full input, SHA-256, raw response, and exact page, paragraph or MIME positions. A range may be part of a longer input. The display adds zero weight to the overall assessment.

The list covers fully recorded classifier inputs of up to 1,800 extracted characters. The page coverage table continues to show where text was mapped, read as context or lacked a local record. Ratio-only responses, incomplete inputs and unsupported source positions are outside the new list. Earlier reports and signed packages retain their frozen evidence and PDF bytes.

## Local verification

The isolated Full worker analyzed the three [earlier PDF coverage fixtures](writing-pdf-page-coverage-2026-11-12.md#local-verification) using the installed writing classifier after simulated prepaid checkout. No unpaid event queued an analysis. The new section exposed the existing responses without changing the previous overall decision or adding passage cards.

| Fixture | Additional exact responses | Mapped source pages | Overall decision |
| --- | ---: | --- | --- |
| Two known generated rewrites | 7, all leaned away from AI | 1, 3, 4 | Not assessable |
| Matched genuine revisions | 7, all leaned away from AI | 1, 3, 4 | Not assessable |
| Native text plus blank page | 3, all leaned away from AI | 1 | Not assessable |

The recorded contexts cover both known rewrites. The writing check still missed them. Page 2 stayed unmapped, and the unchecked image area on page 3 stayed outside the local responses. Chrome opened all 17 records and matched the displayed text and source ranges to the downloaded JSON. The API, JSON and Standard PDF agreed on every response range. The Newsroom PDF withheld the new excerpts. The original-file verifier reopened all three signed packages and checked their added locations and inputs. The older signed package verified against its original and contained the same saved PDF bytes. [Verification receipt](evaluation-panels/writing-unselected-responses-app-2026-11-12.json)

Original-file replay verifies exact extracted text, source positions and input hashes. The server also binds each direction to the recorded classifier response when loading the evidence. Neither replay runs a classifier or establishes who wrote the text.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.pdf
```
