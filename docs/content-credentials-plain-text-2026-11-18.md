# Offline verification of plain-text Content Credentials

The experimental native plain-text reader in C2PA SDK 0.92.2 verified both
locally signed test paragraphs and rejected changed text. It could identify
the container of Encypher's published signed paragraph, then stopped at an
assertion decoding error. The bundled application reader uses SDK 0.92.0 and
reports plain text as unsupported for credential validation. In the app, TXT
files continue through the writing checks. [Native SDK handler](https://github.com/contentauth/c2pa-rs/pull/2494)

The published TXT has 240 visible code points and 5,332 trailing format code
points. The wrapper has a byte order mark, the variation-selector encoding of
`C2PATXT` followed by its zero terminator, version 1, and a 5,138-byte JUMBF
store. The decoded store length agrees with the enclosing JUMBF box length.
The publisher used 180 padding bytes. These measurements describe the format;
cryptographic trust requires a verified signature and content binding.
[C2PA unstructured-text carrier](https://spec.c2pa.org/specifications/specifications/2.5/specs/ContentCredentials.html#_embedding_manifests_into_unstructured_text)

## Three publisher downloads

The exact files match Encypher's published SHA-256 values. Both versions used
the application's pinned public C2PA trust and timestamp lists. The native
0.92.2 reader was compiled without HTTP and remote manifest features.
[Encypher examples and published checksums](https://encypher.com/campaign-examples?source=campaign-article50)

| Download | SHA-256 | Bundled 0.92.0 | Native 0.92.2 |
| --- | --- | --- | --- |
| TXT, 20,853 bytes | `42e328461b2181b63c0f2e891d4c8af753351cbfd71899e9460920a1ebef90ab` | Plain text unsupported | Could not decode `c2pa.metadata` |
| MP4, 44,029 bytes | `5d03b045585d824df04b5fcd34bf1c5418a04000ddce2beaa4c4bbc4adc88f49` | Trusted signature and matching file binding | Same trust and file binding |
| WAV, 1,419,620 bytes | `a18c5d8f8b647eb098e606cd9a5c832791f28e0663ab1f9d81f8e4a84a1f7343` | Invalid file binding | Same `assertion.dataHash.mismatch` |

The exact text decoder error is an unexpected string value for the
`https://schema.org` context where the SDK expects a map. The MP4 claim
records opening the video without an AI declaration. The WAV has a verified
timestamp and its active data hash still mismatches. All three cases continue
ordinary detection under the app's existing trust and disclosure rules.

## Authored controls

The native SDK signed two short UTF-8 controls with a disposable local root.
One declared AI generation and one used a non-AI source declaration. Both
were trusted and matched the file hash with that local root. The same AI
control was merely valid under the public trust list and its test signer was
untrusted. Changing a visible word, moving the intact wrapper, or converting
the signed visible text to Unicode NFD yielded a hash mismatch. Removing or
duplicating the wrapper yielded no manifest. The unmerged NFC-aware verifier
proposal documents that compatibility limit. [NFC verification proposal](https://github.com/contentauth/c2pa-rs/pull/2732)

## Repeat locally

Download the public samples from the publisher and use the existing
`scripts.review_public_content_credentials --file ... --expect-sha256 ...`
command to reproduce the bundled WAV and MP4 results. For the native format
comparison, build SDK `=0.92.2` with default features disabled and enable
`rust_native_crypto` and `unstable_plain_text`. Build an isolated reader with
`Reader::from_context` and `with_stream` and pass `text/plain`, `video/mp4`,
or `audio/wav` explicitly. The verifier settings are available from
`app.provenance.c2pa.settings(app.provenance.c2pa.load_bundle("c2pa"))`.
Inspect `validation_state`, the active manifest's validation codes, source
declarations and exact file binding. [Native reader API](https://docs.rs/c2pa/latest/c2pa/struct.Reader.html)
