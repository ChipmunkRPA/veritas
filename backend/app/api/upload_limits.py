"""Upload caps counted while the body streams (D-48; v2.1 code review P2-10; BUILD_SPEC J-7).

`BodyLimitMiddleware` bounds a whole request body by its path. For a multipart upload this observer also bounds each file part
while its bytes arrive, before Starlette's form parser spools them:

- **Declared kind:** the path kind (`/api/analyze/{kind}`), or a `kind` form field that precedes the file part (the UI sends it
  first) → that kind's cap.
- **Otherwise:** the kind sniffed from the part's first 8 KB (the batch detector), else the largest cap. Intake still refuses a
  declared kind the bytes do not match (415) and counts the size again.
- **Named fields:** `field_caps` (the batch `archive`) → that cap.

The observer parses a copy of each chunk. The chunk that takes a part past its cap is refused before the form parser receives
it, so at most the cap is spooled. A body the observer cannot parse is left to the form parser, which answers it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from python_multipart.multipart import MultipartParser, parse_options_header

HEAD_BYTES = 8192
MB = 1024 * 1024


class PartTooLarge(Exception):
    """A file part passed its cap; the message is the 413 detail."""


class PartLimiter:
    def __init__(self, boundary: bytes, kind_caps: Mapping[str, int], *, declared_kind: str | None = None,
                 field_caps: Mapping[str, int] | None = None, detect: Callable[[bytes, str], str | None] | None = None):
        self.kind_caps = kind_caps
        self.field_caps = dict(field_caps or {})
        self.declared_kind = declared_kind if declared_kind in kind_caps else None
        self.detect = detect
        self.broken = False
        self._kind_field: str | None = None
        self._header_name = self._header_value = b""
        self._begin()
        self._parser = MultipartParser(boundary, {
            "on_part_begin": self._begin, "on_header_field": self._on_header_field, "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end, "on_headers_finished": self._on_headers_finished,
            "on_part_data": self._on_part_data, "on_part_end": self._on_part_end,
        })

    def write(self, chunk: bytes) -> None:
        """Feed the next body chunk. Raises PartTooLarge."""
        if self.broken or not chunk:
            return
        try:
            self._parser.write(chunk)
        except PartTooLarge:
            raise
        except Exception:  # noqa: BLE001 - a malformed body: the form parser answers it
            self.broken = True

    # --- parser callbacks ------------------------------------------------------------------------------------------------------

    def _begin(self) -> None:
        self._disposition, self._name, self._filename = b"", None, None
        self._size, self._head, self._field, self._cap, self._label = 0, bytearray(), bytearray(), None, ""

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._disposition = self._header_value
        self._header_name = self._header_value = b""

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        self._name = options.get(b"name", b"").decode("latin-1")
        self._filename = options[b"filename"].decode("latin-1") if b"filename" in options else None
        if self._filename is not None and (self._name in self.field_caps or self.declared_kind or self._kind_field):
            self._resolve()

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._filename is None:
            if self._name == "kind" and len(self._field) < 64:
                self._field += data[start:end]
            return
        self._size += end - start
        if len(self._head) < HEAD_BYTES:
            self._head += data[start:min(end, start + HEAD_BYTES - len(self._head))]
        if self._cap is None and len(self._head) >= HEAD_BYTES:
            self._resolve()
        self._check()

    def _on_part_end(self) -> None:
        if self._filename is None:
            if self._name == "kind":
                value = self._field.decode("latin-1").strip()
                self._kind_field = value if value in self.kind_caps else None
            return
        if self._cap is None:  # a part shorter than the sniffed head
            self._resolve()
        self._check()

    # --- caps ------------------------------------------------------------------------------------------------------------------

    def _resolve(self) -> None:
        if self._name in self.field_caps:
            self._cap, self._label = self.field_caps[self._name], self._name or "upload"
            return
        kind = self.declared_kind or self._kind_field
        if kind is None and self.detect is not None:
            kind = self.detect(bytes(self._head), self._filename or "")
        if kind in self.kind_caps:
            self._cap, self._label = self.kind_caps[kind], kind
        else:
            self._cap, self._label = max(self.kind_caps.values()), "upload"

    def _check(self) -> None:
        if self._cap is not None and self._size > self._cap:
            raise PartTooLarge(f"The {self._label} exceeds the {max(self._cap // MB, 1)} MB limit")


def for_body(content_type: bytes, kind_caps: Mapping[str, int], *, declared_kind: str | None = None,
             field_caps: Mapping[str, int] | None = None, detect: Callable[[bytes, str], str | None] | None = None) -> PartLimiter | None:
    """A limiter for a multipart body with a boundary, else None."""
    media_type, params = parse_options_header(content_type)
    if media_type != b"multipart/form-data" or not params.get(b"boundary"):
        return None
    return PartLimiter(params[b"boundary"], kind_caps, declared_kind=declared_kind, field_caps=field_caps, detect=detect)
