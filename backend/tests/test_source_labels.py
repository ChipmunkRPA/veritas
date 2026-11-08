"""Actual authored containers plus hostile metadata and source-binding controls."""
import hashlib
import html
import io
import json
import struct
import subprocess
import zipfile
import zlib

import pytest
from PIL import Image, PngImagePlugin

from app.provenance import source_labels as labels


def claim(producer='OpenAI', label='1', **kwargs):
    return json.dumps({'Label': label, 'ContentProducer': producer, **kwargs})


def xmp(value, namespace=labels.NAMESPACES[0]):
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:t="{namespace}">'
        f'<t:AIGC>{html.escape(value)}</t:AIGC></x:xmpmeta>').encode()


def read(tmp_path, data, kind):
    path = tmp_path / 'source.bin'
    path.write_bytes(data)
    return labels.inspect_file(path, kind, hashlib.sha256(data).hexdigest())


def png(value=None, xml=None):
    info = PngImagePlugin.PngInfo()
    if value is not None:
        info.add_text('AIGC', value)
    if xml is not None:
        info.add_itxt('XML:com.adobe.xmp', xml.decode(), zip=True)
    target = io.BytesIO()
    Image.new('RGB', (12, 12), '#349988').save(target, format='PNG', pnginfo=info)
    return target.getvalue()


@pytest.mark.parametrize('provider', ['OpenAI', 'Claude', 'Gemini', 'Muse', 'Grok', 'MiniMax', 'DeepSeek',
    'Qwen', 'Seedance', 'Hunyuan', 'ERNIE', 'Kling'])
def test_named_provider_claims_remain_unverified_and_outside_confidence(tmp_path, provider):
    got = read(tmp_path, png(claim(provider)), 'image')
    assert got['status'] == 'checked' and len(got['records']) == 1
    row = got['records'][0]
    assert provider.casefold() in labels.ALIASES
    assert row['claimed_provider'] == labels.ALIASES[provider.casefold()]
    assert row['provider_authenticated'] is row['used_in_confidence'] is False
    assert row['confidence_pct'] is None and got['provider_private_watermarks_checked'] is False


@pytest.mark.parametrize('label', ['1', '2', '3'])
def test_declaration_levels_are_not_interchanged(tmp_path, label):
    got = read(tmp_path, png(claim(label=label)), 'image')
    assert got['records'][0]['declaration'] == labels.DECLARATIONS[label]


def test_private_identifiers_and_reserved_signatures_are_never_exposed(tmp_path):
    secret = '0022' + '17777777777' + '0' * 12
    raw = claim(secret, ProduceID='private-document-id', ReservedCode1='unverified-signature-material')
    got = read(tmp_path, png(raw), 'image')
    text = json.dumps(got)
    assert all(v not in text for v in (secret, 'private-document-id', 'unverified-signature-material'))
    row = got['records'][0]
    assert row['claimed_provider'] is None and row['provider_value_redacted'] is True
    assert row['reserved_fields_present'] and row['content_identifier_present']


@pytest.mark.parametrize('value', ['{"Label":"1","Label":"2"}', '{"Label":true}', '{"Label":1}',
    '{"Label":"0"}', '[]', claim('x' * 9000), '{"Label":"1","ContentProducer":{"x":"OpenAI"}}'])
def test_malformed_or_ambiguous_fields_cannot_supply_a_claim(tmp_path, value):
    got = read(tmp_path, png(value), 'image')
    assert got['status'] == 'partial' and got['records'] == []


def test_body_text_spacing_and_provider_mentions_are_not_metadata(tmp_path):
    got = read(tmp_path, (claim('Grok') + '\u200b  Gemini OpenAI').encode(), 'document')
    assert got['status'] == 'not_assessed' and got['records'] == []
    assert 'spacing' in ' '.join(got['issues'])


@pytest.mark.parametrize('namespace', list(labels.NAMESPACES))
def test_compressed_png_xmp_uses_namespace_not_literal_prefix(tmp_path, namespace):
    got = read(tmp_path, png(xml=xmp(claim('DeepSeek'), namespace)), 'image')
    assert got['records'][0]['claimed_provider'] == 'DeepSeek'
    assert 'XMP' in got['records'][0]['location']


def test_wrong_namespace_does_not_become_a_label(tmp_path):
    got = read(tmp_path, png(xml=xmp(claim(), 'https://example.invalid/untrusted')), 'image')
    assert got['records'] == []


def test_xml_entities_and_compression_bombs_are_refused(tmp_path):
    attack = b'<!DOCTYPE x [<!ENTITY bad "OpenAI">]><x/>'
    got = read(tmp_path, png(xml=attack), 'image')
    assert got['status'] == 'partial' and got['records'] == []
    with pytest.raises(labels.InvalidMetadata):
        labels._inflate(zlib.compress(b' ' * (labels.MAX_PACKET + 1)))


@pytest.mark.parametrize('kind', ['JPEG', 'WEBP'])
def test_actual_exif_user_comment_is_read_without_decoding_pixels(tmp_path, kind):
    exif = Image.Exif()
    exif[37510] = b'ASCII\0\0\0' + json.dumps({'AIGC': json.loads(claim('MiniMax'))}).encode()
    output = io.BytesIO()
    Image.new('RGB', (12, 12), '#deccab').save(output, format=kind, exif=exif)
    got = read(tmp_path, output.getvalue(), 'image')
    assert got['status'] == 'checked' and got['records'][0]['claimed_provider'] == 'MiniMax / Hailuo'


def test_source_hash_mismatch_cannot_attach_a_label_to_another_upload(tmp_path):
    path = tmp_path / 'source.png'
    path.write_bytes(png(claim()))
    with pytest.raises(ValueError, match='verified upload'):
        labels.inspect_file(path, 'image', '0' * 64)


def test_crc_damage_and_metadata_stripping_have_no_originality_verdict(tmp_path):
    data = png(claim())
    damaged = data.replace(b'OpenAI', b'OpenAJ')
    got = read(tmp_path, damaged, 'image')
    assert got['status'] == 'partial' and got['records'] == []
    stripped = read(tmp_path, png(), 'image')
    assert stripped['status'] == 'checked' and stripped['records'] == []
    assert 'do not establish originality' in stripped['notice']


@pytest.mark.parametrize('extension,codec', [('mp3', 'libmp3lame'), ('flac', 'flac'), ('opus', 'libopus'), ('m4a', 'aac')])
def test_real_encoded_audio_metadata(tmp_path, extension, codec):
    path = tmp_path / ('audio.' + extension)
    args = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.1',
        '-c:a', codec, '-threads', '1', '-metadata', 'AIGC=' + claim('Grok')]
    if extension == 'm4a':
        args += ['-movflags', 'use_metadata_tags']
    subprocess.run([*args, str(path)], check=True, timeout=30)
    got = read(tmp_path, path.read_bytes(), 'audio')
    assert got['status'] == 'checked' and got['records'][0]['claimed_provider'] == 'xAI / Grok', got


def test_wav_aigc_chunk_and_mp4_keyed_metadata(tmp_path):
    import wave

    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0' * 3200)
    base, raw = output.getvalue(), claim('Gemini').encode()
    chunk = b'AIGC' + struct.pack('<I', len(raw)) + raw + b'\0' * (len(raw) % 2)
    data = base[:4] + struct.pack('<I', len(base) + len(chunk) - 8) + base[8:] + chunk
    got = read(tmp_path, data, 'audio')
    assert got['records'][0]['claimed_provider'] == 'Google / Gemini'
    path = tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'color=s=64x64:r=1:d=1',
        '-c:v', 'libx264', '-threads', '1', '-movflags', 'use_metadata_tags', '-metadata', 'AIGC=' + claim('Muse'),
        str(path)], check=True, timeout=30)
    got = read(tmp_path, path.read_bytes(), 'video')
    assert got['records'][0]['claimed_provider'] == 'Meta / Muse'


def test_actual_docx_custom_property_and_pdf_information_dictionary(tmp_path):
    import pymupdf
    from docx import Document

    path = tmp_path / 'text.docx'
    doc = Document()
    doc.add_paragraph('Authored test content.')
    doc.save(path)
    xml = ('<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<property name="AIGC"><vt:lpwstr>' + html.escape(claim('Claude')) + '</vt:lpwstr></property></Properties>')
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('docProps/custom.xml', xml)
    got = read(tmp_path, path.read_bytes(), 'document')
    assert got['records'][0]['claimed_provider'] == 'Anthropic / Claude'
    path = tmp_path / 'document.pdf'
    with pymupdf.open() as doc:
        doc.new_page()
        doc.set_metadata({'producer': 'Authored local test'})
        _, info = doc.xref_get_key(-1, 'Info')
        doc.xref_set_key(int(info.split()[0]), 'AIGC', pymupdf.get_pdf_str(claim('DeepSeek')))
        doc.save(path)
    got = read(tmp_path, path.read_bytes(), 'document')
    assert got['records'][0]['claimed_provider'] == 'DeepSeek'


@pytest.mark.parametrize('comment', ['OpenAI edited this family picture', '{"Label":"1","ContentProducer":"OpenAI"}'])
def test_ordinary_exif_comments_are_not_structured_label_declarations(tmp_path, comment):
    exif = Image.Exif()
    exif[37510] = b'ASCII\0\0\0' + comment.encode()
    output = io.BytesIO()
    Image.new('RGB', (12, 12)).save(output, format='JPEG', exif=exif)
    got = read(tmp_path, output.getvalue(), 'image')
    assert got['status'] == 'checked' and got['records'] == []


def test_duplicate_document_members_and_record_caps_remain_incomplete(tmp_path):
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w') as archive:
        archive.writestr('docProps/custom.xml', '<Properties/>')
        with pytest.warns(UserWarning, match='Duplicate name'):
            archive.writestr('docProps/custom.xml', '<Properties/>')
    got = read(tmp_path, target.getvalue(), 'document')
    assert got['status'] == 'partial' and got['records'] == []
    info = PngImagePlugin.PngInfo()
    for _ in range(9):
        info.add_text('AIGC', claim())
    out = io.BytesIO()
    Image.new('RGB', (12, 12)).save(out, format='PNG', pnginfo=info)
    got = read(tmp_path, out.getvalue(), 'image')
    assert got['status'] == 'partial' and len(got['records']) == 8


def test_pdf_xmp_is_bounded_and_source_linked(tmp_path):
    import pymupdf

    path = tmp_path / 'xmp.pdf'
    with pymupdf.open() as doc:
        doc.new_page()
        doc.set_xml_metadata(xmp(claim('Muse')).decode())
        doc.save(path, deflate=True)
    got = read(tmp_path, path.read_bytes(), 'document')
    assert got['status'] == 'checked' and got['records'][0]['claimed_provider'] == 'Meta / Muse'


def test_library_metadata_failure_is_partial_not_a_pipeline_failure(tmp_path, monkeypatch):
    def malformed(reader):
        reader.checked.append('PNG AIGC text and XMP metadata')
        raise TypeError('malformed test-only third-party metadata value')

    monkeypatch.setattr(labels.Reader, 'png', malformed)
    got = read(tmp_path, png(), 'image')
    assert got['status'] == 'partial' and got['records'] == []
    assert 'test-only' not in json.dumps(got)


def test_cancellation_during_metadata_parsing_is_never_swallowed(tmp_path):
    from app.engine.boundary import Cancelled

    path = tmp_path / 'cancel.png'
    data = png(claim())
    path.write_bytes(data)
    checks = []

    def check(label):
        checks.append(label)
        if len(checks) > 1:
            raise Cancelled('cancel authored test')

    with pytest.raises(Cancelled):
        labels.inspect_file(path, 'image', hashlib.sha256(data).hexdigest(), check)


def document_archive(compression=zipfile.ZIP_DEFLATED):
    target = io.BytesIO()
    xml = ('<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<property name="AIGC"><vt:lpwstr>' + html.escape(claim('Gemini')) + '</vt:lpwstr></property></Properties>')
    with zipfile.ZipFile(target, 'w', compression=compression) as archive:
        archive.writestr('docProps/custom.xml', xml)
    return target.getvalue()


@pytest.mark.parametrize('damage', ['zip64_override', 'multi_disk', 'wrong_count', 'wrong_offset', 'wrong_comment'])
def test_document_directory_bounds_are_checked_before_zipfile(tmp_path, monkeypatch, damage):
    data = document_archive()
    end = data.rfind(b'PK\x05\x06')
    trailer = bytearray(data[end:])
    if damage == 'zip64_override':
        # A real ZIP64 locator can override apparently small legacy sizes.
        directory_size, offset = struct.unpack('<2I', trailer[12:20])
        extended = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, 1, 1,
            2**32 + directory_size, offset)
        locator = struct.pack('<4sIQI', b'PK\x06\x07', 0, end, 1)
        data = data[:end] + extended + locator + trailer
    else:
        if damage == 'multi_disk':
            trailer[4:6] = struct.pack('<H', 1)
        elif damage == 'wrong_count':
            trailer[8:12] = b'\0' * 4
        elif damage == 'wrong_offset':
            trailer[16:20] = struct.pack('<I', 1)
        else:
            trailer[20:22] = struct.pack('<H', 10)
        data = data[:end] + trailer

    def forbidden(*args, **kwargs):
        raise AssertionError('An unbounded directory must not reach ZipFile')

    monkeypatch.setattr(labels.zipfile, 'ZipFile', forbidden)
    got = read(tmp_path, data, 'document')
    assert got['status'] == 'partial' and got['records'] == []


@pytest.mark.parametrize('compression', [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_unbounded_document_compression_is_refused_before_open(tmp_path, monkeypatch, compression):
    data = document_archive(compression)

    def forbidden(*args, **kwargs):
        raise AssertionError('Only bounded stored/deflate metadata may be opened')

    monkeypatch.setattr(zipfile.ZipFile, 'open', forbidden)
    got = read(tmp_path, data, 'document')
    assert got['status'] == 'partial' and got['records'] == []


def test_document_reader_reuses_the_verified_file_descriptor(tmp_path, monkeypatch):
    data = document_archive()
    path = tmp_path / 'source.bin'
    path.write_bytes(data)
    original = zipfile.ZipFile
    opened = []

    def descriptor_only(file, *args, **kwargs):
        assert hasattr(file, 'read') and hasattr(file, 'seek'), 'Do not reopen a potentially replaced path'
        opened.append(file)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(labels.zipfile, 'ZipFile', descriptor_only)
    got = labels.inspect_file(path, 'document', hashlib.sha256(data).hexdigest())
    assert opened and got['status'] == 'checked'
    assert got['records'][0]['claimed_provider'] == 'Google / Gemini'
