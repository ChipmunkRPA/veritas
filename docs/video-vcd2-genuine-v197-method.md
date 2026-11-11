# VCD2 qualification with Azure object tags

The v195 qualification stopped before the first transfer because the public
Azure endpoint returned its object tag without quotation marks. The original
slot 1 attempt directory is empty. v197 verifies the private roster and its
public commitment, rechecks the pinned source metadata, and creates a
separate private run directory for qualification. It inherits all 12 camera
recordings, the `[3,6)` ordinary-edit interval, and the v195 packet and
picture-clock checks. [VCD2 roster and paired control method](video-vcd2-genuine-v195-method.md)

Accept an Azure `0x` hexadecimal tag with or without one matching pair of
double quotes. Preserve the exact spelling of its hexadecimal digits and
send the quoted form with `If-Match`. Require the full GET's normalized tag
and exact length to match the HEAD response. Validate Content-MD5 if the
publisher supplies it, and calculate a local SHA-256. Record the exact
response tags and conditional request value privately with the source
receipts. [Azure conditional requests](https://learn.microsoft.com/en-us/rest/api/storageservices/specifying-conditional-headers-for-blob-service-operations) ·
[Azure GET response headers](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob)

An in-memory probe of the already selected first source found a 2009-09-19
anonymous API response with an unquoted tag. Both matching conditional HEAD
forms returned HTTP 200. A wrong tag returned HTTP 412. Explicit API version
2023-11-03 returned the same tag in quotes. One quoted conditional range GET
returned HTTP 206 with bytes 0–15 of a 195,636,298-byte object and a valid
Matroska header. The probe retained no source bytes.

## Commands

Run from the repository root. Preflight reads only the pinned text metadata
and existing v195 receipts. Freeze writes to the new v197 folder in the
private Seagate archive. Qualify downloads only the named slot into a new
attempt directory and then verifies it locally.

```sh
metadata=/private/tmp/chipmunk-vcd2.014eoQ
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v197 preflight --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v197 freeze --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v197 qualify --metadata-dir "$metadata" --slot 1
```

Use the next unused `--attempt` number for a retry. The acquisition record
and verified picture clocks belong to qualification only.
