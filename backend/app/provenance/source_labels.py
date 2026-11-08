"""Bounded, local reads of unverified structured AIGC metadata.

These records are deliberately outside the signal/fusion system. Recognizing
the field format authenticates neither its author nor the uploaded content.
No provider-private watermark is decoded, and no network request is made.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

VERSION = 'source-labels/1'
MAX_PACKET = 65536
MAX_JSON = 8192
MAX_RECORDS = 8
MAX_STEPS = 4096
NAMESPACES = ('http://www.tc260.org.cn/ns/AIGC/1.0/', 'http://www.tc260.org.cn/ns/AIGC/1.0')
NOTICE = ('These are unverified metadata claims. Labels can be copied, changed or removed. '
    'They do not authenticate a provider or locate edits, and do not enter confidence scoring. '
    'Missing labels do not establish originality. Private provider watermarks were not checked.')
PROVIDERS = {
    'OpenAI': ('openai', 'chatgpt', 'sora', 'dall-e'),
    'Anthropic / Claude': ('anthropic', 'claude'),
    'Google / Gemini': ('google', 'gemini', 'imagen', 'veo', 'lyria'),
    'Meta / Muse': ('meta', 'muse'), 'xAI / Grok': ('xai', 'grok'),
    'MiniMax / Hailuo': ('minimax', 'hailuo', '海螺'), 'DeepSeek': ('deepseek', '深度求索'),
    'Alibaba / Qwen / Wan': ('alibaba', 'qwen', 'wan', '通义千问', '通义万相'),
    'ByteDance / Doubao / Seedance': ('bytedance', 'doubao', 'seedance', '豆包', '即梦'),
    'Tencent / Hunyuan': ('tencent', 'hunyuan', '混元'),
    'Baidu / ERNIE': ('baidu', 'ernie', '文心'), 'Kuaishou / Kling': ('kuaishou', 'kling', '可灵'),
}
ALIASES = {alias: name for name, aliases in PROVIDERS.items() for alias in aliases}
DECLARATIONS = {'1': 'Declares AI generation or synthesis', '2': 'Labels possible AI involvement',
    '3': 'Labels suspected AI involvement'}


class InvalidMetadata(ValueError):
    pass


def _pairs(items):
    out = {}
    for k, v in items:
        if k in out:
            raise InvalidMetadata('Duplicate JSON field')
        out[k] = v
    return out


def _xml(data):
    # UTF-8 only: reject NUL/UTF-16 and DTDs before ElementTree can expand an
    # entity. This reader needs no external entities or schema resolution.
    if len(data) > MAX_PACKET or b'\0' in data or any(v in data.upper() for v in (b'<!DOCTYPE', b'<!ENTITY')):
        raise InvalidMetadata('Unsupported or oversized XML metadata')
    root = ET.fromstring(data.decode('utf-8-sig'))
    if sum(1 for _ in root.iter()) > 1024:
        raise InvalidMetadata('Too many metadata elements')
    return root


def _inflate(data):
    inflater = zlib.decompressobj()
    result = inflater.decompress(data, MAX_PACKET + 1)
    if len(result) > MAX_PACKET or not inflater.eof or inflater.unused_data:
        raise InvalidMetadata('Compressed metadata exceeds the supported bound')
    return result


class Reader:
    def __init__(self, stream, size, check=None):
        self.stream, self.size, self.check = stream, size, check
        self.records, self.checked, self.issues = [], [], []
        self.steps = self.read_bytes = 0
        self.deadline = time.monotonic() + 3.

    def at(self, offset, size):
        self.steps += 1
        self.read_bytes += size
        if self.check:
            self.check('source-label metadata')
        if (self.steps > MAX_STEPS or self.read_bytes > 1024 * 1024 or time.monotonic() > self.deadline
                or size > MAX_PACKET or min(offset, size) < 0 or offset + size > self.size):
            raise InvalidMetadata('Metadata reader bound or file boundary reached')
        self.stream.seek(offset)
        data = self.stream.read(size)
        if len(data) != size:
            raise InvalidMetadata('Truncated metadata')
        return data

    def add(self, value, location, offset=None):
        if len(self.records) >= MAX_RECORDS:
            self.issues.append('Additional metadata claims were omitted at the record limit.')
            return
        try:
            if not isinstance(value, (str, bytes, bytearray, memoryview)) or len(value) > MAX_JSON:
                raise InvalidMetadata('Unsupported or oversized label field')
            raw = value.encode('utf-8') if isinstance(value, str) else bytes(value)
            if len(raw) > MAX_JSON:
                raise InvalidMetadata('Oversized label field')
            text = raw.decode('utf-8-sig').strip('\0 \t\r\n')
            obj = json.loads(text, object_pairs_hook=_pairs)
            if isinstance(obj, dict) and set(obj) == {'AIGC'}:
                obj = obj['AIGC']
            if not isinstance(obj, dict) or type(obj.get('Label')) is not str or obj['Label'] not in DECLARATIONS:
                raise InvalidMetadata('Unsupported label declaration')
            if any(not isinstance(v, str) or len(v) > 4096 for v in obj.values()):
                raise InvalidMetadata('Unsupported metadata field value')
            producer = obj.get('ContentProducer', '').strip()
            claimed = ALIASES.get(producer.casefold())
            self.records.append({'id': f'SL{len(self.records) + 1:02d}', 'location': location, 'byte_offset': offset,
                'payload_sha256': hashlib.sha256(raw).hexdigest(), 'declaration': DECLARATIONS[obj['Label']],
                'payload_hash_scope': 'Extracted label field before JSON parsing; decoded text is encoded as UTF-8, not the entire metadata container.',
                'label_value': obj['Label'], 'claimed_provider': claimed, 'provider_authenticated': False,
                'provider_field_present': bool(producer), 'provider_value_redacted': bool(producer and not claimed),
                'content_identifier_present': bool(obj.get('ProduceID') or obj.get('PropagateID')),
                'reserved_fields_present': bool(obj.get('ReservedCode1') or obj.get('ReservedCode2')),
                'confidence_pct': None, 'used_in_confidence': False})
        except (ValueError, UnicodeError, RecursionError):
            self.issues.append('A structured label field was malformed, unsupported or oversized; its contents are withheld.')

    def xmp(self, data, location, offset=None):
        root = _xml(data)
        for node in root.iter():
            for ns in NAMESPACES:
                key = '{' + ns + '}AIGC'
                if node.tag == key and not list(node):
                    self.add(node.text or '', location, offset)
                if key in node.attrib:
                    self.add(node.attrib[key], location, offset)

    def exif(self, data, location, offset):
        from PIL import Image

        exif = Image.Exif()
        exif.load(data)
        value = exif.get(37510)
        if value is None:
            value = exif.get_ifd(34665).get(37510)
        if value is not None:
            if isinstance(value, bytes):
                if value.startswith(b'ASCII\0\0\0'):
                    value = value[8:].decode('utf-8')
                elif value.startswith(b'UNICODE\0'):
                    value = value[8:].decode('utf-16')
                else:
                    value = value.decode('utf-8')
            # UserComment is also ordinary prose. Only the documented wrapped
            # AIGC object is a candidate; provider mentions are not labels.
            if isinstance(value, str) and value.lstrip().startswith('{'):
                obj = json.loads(value, object_pairs_hook=_pairs)
                if isinstance(obj, dict) and set(obj) == {'AIGC'}:
                    self.add(value, location, offset)

    def png(self):
        self.checked.append('PNG AIGC text and XMP metadata')
        at = 8
        while at + 12 <= self.size:
            length, kind = struct.unpack('>I4s', self.at(at, 8))
            if at + length + 12 > self.size:
                raise InvalidMetadata('Truncated PNG chunk')
            if kind in (b'tEXt', b'zTXt', b'iTXt'):
                prefix = self.at(at + 8, min(length, 80))
                key = prefix.split(b'\0', 1)[0]
                if key in (b'AIGC', b'XML:com.adobe.xmp'):
                    payload = self.at(at + 8, length)
                    crc = int.from_bytes(self.at(at + 8 + length, 4), 'big')
                    if zlib.crc32(kind + payload) != crc:
                        raise InvalidMetadata('Invalid PNG metadata CRC')
                    body = payload.split(b'\0', 1)[1]
                    if kind == b'zTXt':
                        if body[:1] != b'\0':
                            raise InvalidMetadata('Unsupported PNG compression')
                        body = _inflate(body[1:])
                    elif kind == b'iTXt':
                        flag, method = body[:2]
                        body = body[2:].split(b'\0', 2)[2]
                        if method != 0 or flag not in (0, 1):
                            raise InvalidMetadata('Unsupported PNG text encoding')
                        if flag:
                            body = _inflate(body)
                    if key == b'AIGC':
                        self.add(body, 'PNG AIGC text', at)
                    else:
                        self.xmp(body, 'PNG XMP AIGC property', at)
            at += length + 12
            if kind == b'IEND':
                return
        raise InvalidMetadata('Missing PNG end chunk')

    def jpeg(self):
        self.checked.append('JPEG APP1 XMP and EXIF UserComment')
        at = 2
        while at + 2 <= self.size:
            marker = self.at(at, 2)
            if marker[0] != 255:
                raise InvalidMetadata('Invalid JPEG metadata marker')
            if marker[1] in (0xDA, 0xD9):
                return
            if marker[1] == 255:
                at += 1
                continue
            if marker[1] in (*range(0xD0, 0xD9), 1):
                at += 2
                continue
            length = int.from_bytes(self.at(at + 2, 2), 'big')
            if length < 2 or at + 2 + length > self.size:
                raise InvalidMetadata('Truncated JPEG metadata segment')
            if marker[1] == 0xE1:
                data = self.at(at + 4, length - 2)
                prefix = b'http://ns.adobe.com/xap/1.0/\0'
                if data.startswith(prefix):
                    self.xmp(data[len(prefix):], 'JPEG XMP AIGC property', at)
                elif data.startswith(b'Exif\0\0'):
                    self.exif(data, 'JPEG EXIF UserComment', at)
            at += length + 2

    def riff(self, kind):
        self.checked.append('RIFF AIGC and XMP metadata')
        end = int.from_bytes(self.at(4, 4), 'little') + 8
        if not 12 <= end <= self.size:
            raise InvalidMetadata('Invalid RIFF size')
        at = 12
        while at + 8 <= end:
            tag, size = struct.unpack('<4sI', self.at(at, 8))
            if at + 8 + size > end:
                raise InvalidMetadata('Truncated RIFF metadata')
            if tag == b'AIGC':
                self.add(self.at(at + 8, size), 'RIFF AIGC chunk', at)
            elif tag in (b'XMP ', b'_PMX'):
                self.xmp(self.at(at + 8, size), 'RIFF XMP AIGC property', at)
            elif tag == b'EXIF' and kind == 'image':
                self.exif(self.at(at + 8, size), 'WebP EXIF UserComment', at)
            at += size + 8 + size % 2

    def atoms(self, start, end):
        at = start
        while at + 8 <= end:
            size, tag = struct.unpack('>I4s', self.at(at, 8))
            header = 8
            if size == 1:
                size = int.from_bytes(self.at(at + 8, 8), 'big')
                header = 16
            if size == 0:
                size = end - at
            if size < header or at + size > end:
                raise InvalidMetadata('Invalid media atom size')
            yield tag, at + header, at + size
            at += size

    def bmff(self):
        self.checked.append('MP4/MOV/M4A keyed AIGC metadata')
        for tag, start, end in self.atoms(0, self.size):
            if tag != b'moov':
                continue
            for sub, first, last in self.atoms(start, end):
                if sub == b'udta':
                    for name, a, b in self.atoms(first, last):
                        if name == b'meta':
                            self.meta(a, b)

    def meta(self, start, end):
        if self.at(start, 4) != b'\0' * 4:
            raise InvalidMetadata('Unsupported media metadata version')
        children = list(self.atoms(start + 4, end))
        keys = {}
        for tag, a, b in children:
            if tag == b'keys':
                data = self.at(a, b - a)
                if data[:4] != b'\0' * 4 or len(data) < 8:
                    raise InvalidMetadata('Invalid metadata keys')
                count, pos = int.from_bytes(data[4:8], 'big'), 8
                if count > 256:
                    raise InvalidMetadata('Too many metadata keys')
                for index in range(1, count + 1):
                    size = int.from_bytes(data[pos:pos + 4], 'big')
                    if size < 8 or pos + size > len(data):
                        raise InvalidMetadata('Invalid metadata key')
                    if data[pos + 4:pos + 8] == b'mdta' and data[pos + 8:pos + size] == b'AIGC':
                        keys[index] = True
                    pos += size
        for tag, a, b in children:
            if tag == b'ilst':
                for index, first, last in self.atoms(a, b):
                    if int.from_bytes(index, 'big') in keys:
                        for name, p, q in self.atoms(first, last):
                            if name == b'data' and q - p >= 8 and self.at(p, 4) == b'\0\0\0\1':
                                self.add(self.at(p + 8, q - p - 8), 'Media keyed AIGC metadata', p)

    def id3(self):
        self.checked.append('ID3v2 TXXX AIGC metadata')
        header = self.at(0, 10)
        version, flags = header[3], header[5]
        if version not in (3, 4) or flags & 0xC0 or any(v > 127 for v in header[6:10]):
            raise InvalidMetadata('Unsupported ID3 header')
        end = 10 + sum(v << (7 * (3 - i)) for i, v in enumerate(header[6:10]))
        if end > self.size or end > 256 * 1024:
            raise InvalidMetadata('Oversized ID3 metadata')
        at = 10
        while at + 10 <= end:
            frame = self.at(at, 10)
            if frame[:4] == b'\0' * 4:
                break
            size = int.from_bytes(frame[4:8], 'big') if version == 3 else sum(v << (7 * (3 - i)) for i, v in enumerate(frame[4:8]))
            if size == 0 or at + 10 + size > end:
                raise InvalidMetadata('Invalid ID3 frame')
            if frame[:4] == b'TXXX':
                if frame[8:] != b'\0\0':
                    raise InvalidMetadata('Unsupported ID3 frame flags')
                data = self.at(at + 10, size)
                encoding = {0: 'latin1', 1: 'utf-16', 2: 'utf-16-be', 3: 'utf-8'}[data[0]]
                descriptor, value = data[1:].decode(encoding).split('\0', 1)
                if descriptor == 'AIGC':
                    self.add(value, 'ID3v2 TXXX AIGC frame', at)
            at += size + 10

    def comments(self, data, location, offset):
        if len(data) < 8:
            raise InvalidMetadata('Truncated sound metadata')
        pos = int.from_bytes(data[:4], 'little') + 4
        count = int.from_bytes(data[pos:pos + 4], 'little')
        pos += 4
        if pos > len(data) or count > 256:
            raise InvalidMetadata('Unsupported sound comments')
        for _ in range(count):
            size = int.from_bytes(data[pos:pos + 4], 'little')
            pos += 4
            if pos + size > len(data):
                raise InvalidMetadata('Truncated sound comment')
            field = data[pos:pos + size]
            key, _, value = field.partition(b'=')
            if key.upper() == b'AIGC':
                self.add(value, location, offset)
            pos += size

    def flac(self):
        self.checked.append('FLAC Vorbis AIGC comments')
        at = 4
        while at + 4 <= self.size:
            header = self.at(at, 4)
            size = int.from_bytes(header[1:], 'big')
            if header[0] & 127 == 4:
                self.comments(self.at(at + 4, size), 'FLAC AIGC comment', at)
            at += size + 4
            if header[0] & 128:
                return

    def ogg(self):
        self.checked.append('Initial Ogg Vorbis/Opus comment packets')
        at, packet, packets, serial = 0, b'', 0, None
        for _ in range(8):
            head = self.at(at, 27)
            if head[:5] != b'OggS\0' or serial not in (None, head[14:18]):
                raise InvalidMetadata('Unsupported Ogg stream layout')
            serial = head[14:18]
            lacing = self.at(at + 27, head[26])
            pos = at + 27 + len(lacing)
            for size in lacing:
                packet += self.at(pos, size)
                pos += size
                if len(packet) > MAX_PACKET:
                    raise InvalidMetadata('Oversized Ogg metadata packet')
                if size < 255:
                    if packet.startswith(b'OpusTags'):
                        self.comments(packet[8:], 'Ogg Opus AIGC comment', at)
                        return
                    if packet.startswith(b'\x03vorbis'):
                        self.comments(packet[7:], 'Ogg Vorbis AIGC comment', at)
                        return
                    packet, packets = b'', packets + 1
                    if packets > 3:
                        raise InvalidMetadata('No supported initial sound-comment packet')
            at = pos
        raise InvalidMetadata('Initial Ogg metadata scan limit reached')

    def docx(self):
        self.checked.append('OOXML custom AIGC property')
        tail = self.at(max(0, self.size - 65536), min(65536, self.size))
        at = tail.rfind(b'PK\x05\x06')
        if at < 0 or len(tail) - at < 22:
            raise InvalidMetadata('Unsupported document directory')
        _, disk, directory_disk, disk_entries, entries, size, offset, comment = struct.unpack('<4s4H2IH', tail[at:at + 22])
        end_record = self.size - len(tail) + at
        # ZIP64 can override the legacy bounds, even when those legacy fields
        # are small. Refuse it before ZipFile reads its directory into memory.
        locator = self.at(end_record - 20, 4) if end_record >= 20 else b''
        if (disk or directory_disk or disk_entries != entries or entries > 2048 or size > 1024 * 1024
                or offset + size != end_record or at + 22 + comment != len(tail) or locator == b'PK\x06\x07'):
            raise InvalidMetadata('Unsupported or oversized document directory')
        cursor, counted = offset, 0
        while cursor < end_record:
            if end_record - cursor < 46 or counted >= 2048:
                raise InvalidMetadata('Unsupported document directory entries')
            header = self.at(cursor, 46)
            name, extra, annotation, start_disk = struct.unpack('<4H', header[28:36])
            length = 46 + name + extra + annotation
            if header[:4] != b'PK\x01\x02' or start_disk or cursor + length > end_record:
                raise InvalidMetadata('Invalid document directory entry')
            cursor, counted = cursor + length, counted + 1
        if counted != entries:
            raise InvalidMetadata('Inconsistent document directory count')
        # Reuse the verified source descriptor, rather than reopening its path.
        with zipfile.ZipFile(self.stream) as archive:
            members = [i for i in archive.infolist() if i.filename == 'docProps/custom.xml']
            if len(members) > 1:
                raise InvalidMetadata('Duplicate document metadata member')
            if not members:
                return
            member = members[0]
            if (member.file_size > MAX_PACKET or member.compress_size > MAX_PACKET or member.flag_bits & 1
                    or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise InvalidMetadata('Unsupported document metadata member')
            with archive.open(member) as stream:
                root = _xml(stream.read(MAX_PACKET + 1))
            ns = '{http://schemas.openxmlformats.org/officeDocument/2006/custom-properties}'
            vt = '{http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes}lpwstr'
            if root.tag != ns + 'Properties':
                raise InvalidMetadata('Unsupported custom-property namespace')
            for prop in root:
                if prop.tag == ns + 'property' and prop.get('name') == 'AIGC':
                    children = list(prop)
                    if len(children) != 1 or children[0].tag != vt:
                        raise InvalidMetadata('Unsupported custom-property value')
                    self.add(children[0].text or '', 'Document custom AIGC property', member.header_offset)

    def pdf(self, path):
        import pymupdf

        self.checked.append('PDF Information AIGC entry and bounded catalog XMP')
        if self.size > 16 * 1024 * 1024:
            raise InvalidMetadata('PDF metadata reader supports files up to 16 MiB')
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise InvalidMetadata('Encrypted PDF metadata is not assessed')
            kind, value = doc.xref_get_key(-1, 'Info')
            if kind == 'xref':
                field, text = doc.xref_get_key(int(value.split()[0]), 'AIGC')
                if field == 'string':
                    self.add(text, 'PDF Information AIGC entry')
            kind, value = doc.xref_get_key(doc.pdf_catalog(), 'Metadata')
            if kind == 'xref':
                xref = int(value.split()[0])
                raw = doc.xref_stream_raw(xref)
                if len(raw) > MAX_PACKET:
                    raise InvalidMetadata('Oversized PDF metadata stream')
                filter_kind, filter_name = doc.xref_get_key(xref, 'Filter')
                if filter_kind == 'null':
                    data = raw
                elif filter_kind == 'name' and filter_name == '/FlateDecode':
                    data = _inflate(raw)
                else:
                    raise InvalidMetadata('Unsupported PDF metadata stream filter')
                self.xmp(data, 'PDF catalog XMP AIGC property')


def inspect_file(path: Path, kind: str, expected_sha256: str, check=None) -> dict:
    """Read only the verified source; never trust the filename or article body."""
    result = {'version': VERSION, 'source_sha256': expected_sha256, 'status': 'not_assessed',
        'records': [], 'checked_metadata': [], 'issues': [], 'notice': NOTICE,
        'provider_private_watermarks_checked': False, 'used_in_confidence': False}
    with path.open('rb') as stream:
        sha = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            if check:
                check('source-label binding')
            sha.update(block)
        if sha.hexdigest() != expected_sha256:
            raise ValueError('Source-label input differs from the verified upload')
        reader = Reader(stream, stream.tell(), check)
        try:
            magic = reader.at(0, min(16, reader.size))
            if kind == 'image' and magic.startswith(b'\x89PNG\r\n\x1a\n'):
                reader.png()
            elif kind == 'image' and magic.startswith(b'\xff\xd8'):
                reader.jpeg()
            elif kind in ('audio', 'image') and magic.startswith(b'RIFF') and magic[8:12] == {'audio': b'WAVE', 'image': b'WEBP'}[kind]:
                reader.riff(kind)
            elif kind in ('audio', 'video') and magic[4:8] == b'ftyp':
                reader.bmff()
            elif kind == 'audio' and magic.startswith(b'ID3'):
                reader.id3()
            elif kind == 'audio' and magic.startswith(b'fLaC'):
                reader.flac()
            elif kind == 'audio' and magic.startswith(b'OggS'):
                reader.ogg()
            elif kind == 'document' and magic.startswith(b'PK\x03\x04'):
                reader.docx()
            elif kind == 'document' and magic.startswith(b'%PDF-'):
                reader.pdf(path)
            else:
                reader.issues.append('This format has no supported structured-label reader; text spacing and body mentions are not provider verification.')
        except (InvalidMetadata, ValueError, UnicodeError, KeyError, IndexError, TypeError, AttributeError,
                RecursionError, struct.error, ET.ParseError, zipfile.BadZipFile, zlib.error, RuntimeError):
            reader.issues.append('Some metadata was malformed, unsupported or beyond the reader limits. Coverage is incomplete.')
        result.update(records=reader.records, checked_metadata=reader.checked, issues=list(dict.fromkeys(reader.issues)))
        result['status'] = 'partial' if reader.checked and reader.issues else 'checked' if reader.checked else 'not_assessed'
    return result


def stage(ctx):
    from app.engine.stages import common

    digest = (ctx.outputs.get('ingest') or {}).get('sha256')
    if ctx.state.get('refusion'):
        return {'version': VERSION, 'source_sha256': digest, 'status': 'not_assessed', 'records': [],
            'checked_metadata': [], 'issues': ['Historical re-fusion does not add a new source-label examination.'],
            'notice': NOTICE, 'provider_private_watermarks_checked': False, 'used_in_confidence': False}
    return inspect_file(common.media_path(ctx), ctx.state.get('kind', ''), digest, ctx.check)
