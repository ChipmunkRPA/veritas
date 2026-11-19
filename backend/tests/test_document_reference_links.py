import hashlib
from copy import deepcopy

import pytest

from app.core.canonical import canonical_hash
from app.evidence import document_reference_links as reader
from app.evidence import document_reference_links_binding as binding
from app.evidence import document_scope


def _html(label=4, target=3):
    entries = "".join(
        f'<li><span>{number}. </span><a id="entry-{number}"></a>Source note number {number}.</li>'
        for number in range(1, 9)
    )
    return (
        '<!doctype html><html><head><title>Source record</title></head><body><main><article>'
        '<h1>Independent source notes</h1><p>Our field team recorded a bright day 😀 with steady winds and clear roads. '
        f'The observations were checked against a saved reference [<a href="#entry-{target}">{label}</a>] in the summary. '
        'The report records the time of each observation and a description of the site where the team first took measurements.</p>'
        f'<section><h2>References</h2><ol>{entries}</ol></section></article></main></body></html>'
    )


def _document(record):
    return {"source": {"method": "html", binding.COMMITMENT: canonical_hash(record)},
            "coverage": {document_scope.HTML_STATIC_KEY: document_scope.HTML_STATIC_POLICY}, binding.KEY: record}


@pytest.mark.parametrize("encoding", ("utf-8", "utf-8-sig", "utf-16"))
def test_exact_unicode_positions_and_saved_fields(tmp_path, encoding):
    raw = _html()
    path = tmp_path / "saved.html"
    data = raw.encode(encoding)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    record = reader.inspect(path, ".html", source_sha256=digest)
    assert record["decoded_characters"] == len(raw)
    assert record["checked_links"] == 1
    assert record["references"] == 8
    assert record["authorship_confidence_contribution"] == record["destinations_contacted"] == 0
    first = raw.index(">4</a>") + 1
    second = raw.index("<span>3. </span>") + len("<span>")
    assert record["findings"] == [{"label_number": 4, "target_number": 3,
                                    "label_range": [first, first + 1], "target_range": [second, second + 1]}]
    document = _document(record)
    binding.verify_record(document, record, commitment=document["source"][binding.COMMITMENT])
    with pytest.raises(ValueError, match="Submitted HTML differs"):
        reader.inspect(path, ".html", source_sha256="0" * 64)


def test_only_standalone_default_html_is_eligible(tmp_path):
    source = tmp_path / "source.xhtml"
    source.write_text(_html(), encoding="utf-8")
    assert reader.inspect(source, ".xhtml", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()) is None
    record = reader.capture(_html())
    assert record
    document = _document(record)
    document["coverage"][document_scope.HTML_STATIC_KEY] = document_scope.XHTML_STATIC_POLICY
    with pytest.raises(ValueError, match="committed HTML inspection"):
        binding.validate(document)


def test_exact_label_can_be_split_from_following_space_by_markup():
    raw = _html().replace('>3. </span>', '>3.</span> ')
    record = reader.capture(raw)
    assert record and raw[slice(*record["findings"][0]["target_range"])] == "3"


@pytest.mark.parametrize("raw", [
    _html(label=3),
    _html().replace('<a href="#entry-3">4</a>', '40–42'),
    _html().replace('id="entry-3"', 'id="renamed"'),
    _html().replace('>3. </span>', '>Reference three </span>'),
    _html().replace('<p>Our', '<p hidden>Our'),
    _html().replace('<p>Our', '<iframe><p>Our').replace('</p><section>', '</p></iframe><section>'),
    _html().replace('<section>', '<section style="display:none">'),
    _html().replace('</head>', '<base href="https://example.invalid/"></head>'),
    _html().replace('</body>', '<p hidden id="entry-3">Other label</p></body>'),
    _html().replace('href="#entry-3"', 'href="#entry-3" href="#entry-4"'),
    _html().replace('href="#entry-3"', 'href="other.html#entry-3"'),
    _html().replace('href="#entry-3"', 'href="#entry-3-ordinary"').replace(
        '</body>', '<div id="entry-3-ordinary">3. A nonbibliographic note.</div></body>'),
])
def test_controls_do_not_publish_discrepancy(raw):
    assert reader.capture(raw) is None


def test_tampered_anchor_or_downgrade_rejected_and_legacy_can_omit():
    record = reader.capture(_html())
    assert record
    original = _document(record)
    modified = deepcopy(original)
    modified[binding.KEY]["findings"][0]["label_range"] = [3, 4]
    with pytest.raises(ValueError, match="committed HTML inspection"):
        binding.validate(modified)
    modified["source"][binding.COMMITMENT] = canonical_hash(modified[binding.KEY])
    with pytest.raises(ValueError, match="frozen original"):
        binding.verify_record(modified, record, commitment=modified["source"][binding.COMMITMENT])
    del modified[binding.KEY]
    with pytest.raises(ValueError, match="committed HTML inspection"):
        binding.validate(modified)
    del modified["source"][binding.COMMITMENT]
    binding.verify_record(modified, None, commitment=None)
