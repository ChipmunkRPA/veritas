# Bundled trust data: sources, licences and attribution

Snapshots taken by `backend/scripts/v2/c2pa_trust_snapshot.py`. The revisions and SHA-256 values are in `snapshot.json`,
and `app/provenance/c2pa.py` refuses a file that does not match them. The files are unmodified copies.

| directory / file | content | source | licence |
|---|---|---|---|
| `c2pa/C2PA-TRUST-LIST.pem`, `c2pa/C2PA-TSA-TRUST-LIST.pem` | C2PA Trust List and TSA trust list (conformance program) | c2pa-org/conformance-public, `trust-list/` at 99927caef670ca4ad9da5e5542dca39e42fad6f3 | CC BY 4.0, © Coalition for Content Provenance and Authenticity (https://creativecommons.org/licenses/by/4.0/) |
| `cai-legacy/anchors.pem`, `allowed.sha256.txt`, `store.cfg` | Legacy CAI interim trust list (frozen 2026-01-01) | https://contentcredentials.org/trust/ | the publisher states no licence (public certificates and certificate hashes) |
| `iptc_digital_source_type.json` | IPTC NewsCodes DigitalSourceType vocabulary (release 2024-10-23), with Veritas tier labels added (IF §4.4) | https://cv.iptc.org/newscodes/digitalsourcetype/ | CC BY 4.0, © IPTC, International Press Telecommunications Council (https://iptc.org) |

`iptc_digital_source_type.json` is not a verbatim copy: it keeps the term labels and retired flags and adds a `tier`
field. The SHA-256 of the source document is recorded in it.
