# Verifying Content Credentials timestamps

Chipmunk Verify now uses the bundled C2PA Timestamp Authority List for
signature timestamps. Claim-signing certificates use the separate C2PA
Trust List. The two checks are visible in the evidence. A displayed
verified timestamp attests to the existence of the claim signature by
that time. File binding and the signed AI declaration are evaluated
separately. [C2PA trust lists](https://opensource.contentauthenticity.org/docs/conformance/trust-lists/) ·
[Signature timestamp validation](https://spec.c2pa.org/specifications/specifications/2.5/specs/ContentCredentials.html)

The verifier stays offline. It may use certificate status evidence
embedded in the upload. It performs zero live revocation queries and
zero remote manifest requests. A valid, trusted, asset-bound declaration
with zero validation failures still qualifies to skip statistical checks
when a signature timestamp is unavailable. A verified timestamp alone
carries zero weight in the AI determination.

## Results on exact published files

The same pinned SDK 0.92.0 read each file with and without the dedicated
timestamp authorities. Adding the list verified four previously
unverified timestamps. The replay of the [earlier 21-file panel](public-content-credentials-2026-11-17.md)
reproduced all file-binding and pipeline decisions.

| Sample | Signature timestamp with the authority list | Credential outcome |
| --- | --- | --- |
| Two Google Gemini JPEGs | Verified | Both declare AI involvement and pass the gate. The SDK accepted embedded certificate status on the first JPEG. |
| One OpenAI PNG | Unverified | Its trusted, file-bound AI declaration still passes the gate. |
| OpenAI PDF | No verified timestamp recorded | Its whole-document AI declaration passes when PDF credentials are enabled. |
| Older Sora MP4 | Unverified | The signer remains expired and untrusted for this verification. Further analysis continues. |
| Published AIWatermark PNG and MP3 | Verified | The signer is trusted and the asset hashes match. The SDK reports `assertion.action.softBindingMissing` for the watermark action and finds no explicit signed AI source declaration. Both continue to further analysis. |
| Three Rapidata MP4s labeled Veo 3.1 | No timestamp recorded | These exact downloaded copies contain no embedded C2PA manifest. Further analysis continues. |

The AIWatermark samples are published by the signing service. The tests
checked their original byte hashes and embedded C2PA records. The separate
neural watermark was outside the test. The Rapidata measurements describe
its distributed copies. [Publisher sample hashes](https://aiwatermark.com/samples/index.html) ·
[Rapidata dataset](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo3.1/tree/9b5ffd5a1ce186b746990e554dba2a03ee74434d/videos)

A local modification of the first Gemini JPEG left the signature timestamp
verifiable while breaking its signed file hash. The modified file continued
to further analysis. The published panel contains only the original file
hashes and measurements.

The local app replay covered a verified Gemini timestamp, an unverified
OpenAI PNG timestamp, and an OpenAI PDF with no recorded timestamp. The
ordinary browser view, API exhibit, PDF and customer JSON agreed on the
saved finding in all three cases. All jobs followed simulated USD 9.99
payment and used zero statistical model leases.

## Independent replay

The [receipt](evaluation-panels/public-content-timestamps-2026-11-18.json)
records ten download URLs, original byte hashes, the two signing
trust bundles, and the timestamp authority list. It also records the
decision to run further analysis or skip it. Download a file by its
sample ID and compare it to the receipt from the project root.

```sh
scratch=$(mktemp -d)
sample=aiwatermark_audio
receipt=docs/evaluation-panels/public-content-timestamps-2026-11-18.json
url=$(jq -er --arg sample "$sample" '.samples[] | select(.id == $sample) | .url' "$receipt")
curl -fsSL --max-filesize 20000000 "$url" -o "$scratch/$sample.asset"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_public_content_credentials \
  "$receipt" "$scratch" --sample "$sample"
```

Use `--file path/to/original.mp4` in place of the receipt and sample
arguments to check another file locally. The outputs record certificate
and trust-list results at verification time.
