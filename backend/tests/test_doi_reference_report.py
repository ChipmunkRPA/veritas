from __future__ import annotations

from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen


def test_standard_pdf_shows_doi_reference_positions_and_registry_coverage(monkeypatch):
    snapshot = specimen("document", "standard")
    evidence = {
        "schema": "chipmunk.document-evidence/1",
        "id": "synthetic-doi-report",
        "version": "document-evidence-1",
        "sha256": "a" * 64,
        "original_result_hash": "b" * 64,
        "source": {"sha256": snapshot.media.sha256, "method": "python-docx", "chars": 250},
        "packets": [],
        "coverage": {
            "writing_bibliography": {
                "schema": "chipmunk.writing-bibliography/1", "heading": "References", "entry_count": 3,
                "section_start": 40, "section_end": 220,
                "first_location": {"kind": "docx", "body_index": 5},
                "last_location": {"kind": "docx", "body_index": 7},
            },
        },
        "doi_reference_consistency": {
            "schema": "document-doi-reference/1", "affects_overall_confidence": False,
            "scope": "Compares one visible DOI per reference against a frozen local registry.",
            "limitation": "This reference observation carries zero AI confidence weight.",
            "snapshot": {"provider": "Crossref", "as_of": "2026-11-18", "sha256": "c" * 64, "records": 364},
            "references": 3, "consistent": 1, "different_work": 1, "unassessed": 1,
            "locations": [{
                "start": 190, "end": 207, "title_start": 104, "title_end": 138,
                "text": "10.1234/example.1", "cited_title": "A long title in the written citation",
                "anchor": {"kind": "docx", "body_path": [6]},
                "location": {"ranges": [[94, 111]]}, "title_location": {"ranges": [[8, 42]]},
                "printed": {"doi": "10.1234/example.1", "title": "Different work found by the printed DOI on steady‐state growth"},
                "cited": {"doi": "10.1234/example.2", "title": "Publication matched by title and author"},
            }],
        },
    }
    monkeypatch.setattr(multimedia_evidence, "verified_set", lambda _: evidence)
    monkeypatch.setattr(multimedia_evidence, "verify_assets", lambda *_args, **_kwargs: {})
    rendered = engine.render_pdf(snapshot)
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "DOI reference check" in pdf_text
    assert "Local Crossref snapshot dated 2026-11-18 with 364 registry records" in pdf_text
    assert "1 matched, 1 pointed to a different work, and 1 was unassessed" in pdf_text
    assert "This reference observation carries zero AI confidence weight" in pdf_text
    assert "Different work found by the printed DOI" in pdf_text
    assert "on steady-state growth" in pdf_text and "\ufffd" not in pdf_text
    assert evidence["doi_reference_consistency"]["locations"][0]["printed"]["title"].endswith("steady‐state growth")
    assert "Publication matched by title and author · 10.1234/example.2" in pdf_text
    assert "Word body path 6" in pdf_text
    assert "/doi_reference_consistency/locations/0" in pdf_text.replace(" ", "")
    assert "Printed DOI · Extracted characters [190, 207) · Original characters [94, 111)" in pdf_text
    assert "Cited title · Extracted characters [104, 138) · Original characters [8, 42)" in pdf_text
    assert "See the DOI reference check." in pdf_text and "Citation accuracy was not assessed." not in pdf_text
    assert "No usable local candidates were recorded" not in pdf_text
    for kind, label in (("odt_paragraph", "Decoded OpenDocument paragraph 7"), ("rtf_paragraph", "Decoded Rich Text block 7")):
        evidence["doi_reference_consistency"]["locations"][0]["anchor"] = {"kind": kind, "paragraph_index": 6}
        assert label in engine.render_html(snapshot)

    redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
    html = engine.render_html(redacted)
    assert "DOI reference check" in html and "1 pointed to a different work" in html
    assert "10.1234/example.1" not in html and "Different work found by the printed DOI" not in html
