import json
from xml.etree import ElementTree as ET

import pytest

from scripts import text_elife_source_intake_v272 as intake

REVISION = "1" * 40


def publisher_fixture(*, license_url=intake.LICENSE, day=28, body=""):
    return f"""
    <article article-type="article-commentary" xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="doi">10.7554/eLife.81306</article-id>
        <article-categories><subj-group subj-group-type="display-channel"><subject>Insight</subject>
        </subj-group></article-categories>
        <title-group><article-title>Mock source title</article-title></title-group>
        <contrib-group><contrib contrib-type="author"><name><surname>Carmichael</surname>
        <given-names>S Thomas</given-names></name></contrib></contrib-group>
        <pub-date><year>2022</year><month>7</month><day>{day}</day></pub-date>
        <volume>11</volume><elocation-id>e81306</elocation-id>
        <permissions><license xlink:href="{license_url}">
        <license-p>Attribution required</license-p></license></permissions>
      </article-meta></front>
      <body>{body}</body>
    </article>
    """.encode()


def mock_paragraph(number):
    return " ".join(f"passage{number}-tökén{i}" for i in range(50))


def test_verify_work_identity_and_per_work_commercial_license():
    work = intake.WORKS[0]
    result = intake.metadata(ET.fromstring(publisher_fixture()), work)
    assert result["authors"] == ["S Thomas Carmichael"]
    assert result["publication_date"] == "2022-07-28"
    assert result["license_url"] == intake.LICENSE
    with pytest.raises(intake.Rejected, match="license_not_cc_by_4"):
        intake.metadata(
            ET.fromstring(publisher_fixture(license_url=intake.LICENSE.replace("by/", "by-nc/"))), work
        )
    with pytest.raises(intake.Rejected, match="publication_date_mismatch"):
        intake.metadata(ET.fromstring(publisher_fixture(day=27)), work)


def test_normalize_author_prose_and_leave_gaps_at_excluded_text():
    xml = ET.fromstring("""<body>
        <p>Café <italic>résumé</italic> (<xref ref-type="bibr">Another author, 2022</xref>).</p>
        <fig><caption><p>Third-party caption</p></caption></fig>
        <p>A continuing paragraph.</p>
        <p>He said “quoted text”.</p>
        <disp-quote><p>Original quote</p></disp-quote>
        <p>A <ext-link>linked source</ext-link> in prose.</p>
        <p>Independent section.</p>
    </body>""")
    assert list(intake.body_paragraphs(xml)) == [
        "Café résumé.",
        "A continuing paragraph.",
        None,
        None,
        None,
        "Independent section.",
    ]


def test_fixed_denominator_private_text_and_exact_unicode_target(tmp_path):
    parts = [mock_paragraph(i) for i in range(5)]
    body = "".join(f"<p>{part}</p>" for part in parts)
    (tmp_path / "elife-81306-v1.xml").write_bytes(publisher_fixture(body=body))
    journal, receipt = intake.run(tmp_path, REVISION, [("prior", "f" * 64, 1, ["Distinct source passage."])])
    result = json.loads(journal)
    assert receipt["dispositions"] == {"admitted": 1, "missing_direct_xml": 8}
    assert receipt["private_originals_sha256"] == intake.sha(journal)
    assert all(part not in json.dumps(receipt) for part in parts)
    assert [result["text"][a:b] for a, b in result["paragraph_codepoints"]] == parts
    start, stop = result["edit_target_codepoints"]
    assert result["text"][start:stop] in parts[1:-1]

    journal, receipt = intake.run(tmp_path, REVISION, [("prior", "f" * 64, 1, [parts[-1]])])
    assert not journal
    assert receipt["status"] == "no_sources_admitted"
    assert receipt["works"][0]["status"] == "historical_fragment_overlap"


def test_overlap_rejects_every_work_sharing_a_fragment():
    receipts, historical, internal = intake.overlap(
        [("prior", "f" * 64, 1, [mock_paragraph(1)])],
        {"a": {"shared"}, "b": {"shared"}, "c": intake.prior.text.fingerprints(mock_paragraph(1))},
    )
    assert historical == {"c"}
    assert internal == {"a", "b"}
    assert receipts[0]["matched_sources"] == 1
