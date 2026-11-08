"""Risk disclosures remain readable without suppressing unrelated overclaims."""
import pytest

from app.engine.service_policy import standard_service_policy
from app.reports import engine
from app.reports.specimens import specimen


def test_exact_standard_disclosures_are_allowed_but_authenticity_overclaims_are_flagged(monkeypatch):
    policy = standard_service_policy()
    fixed_copy = '\n'.join(policy[k] for k in ('confidence_notice', 'court_notice', 'private_engagement_notice'))
    snapshot = specimen('video', 'summary')
    monkeypatch.setattr(engine, 'pdf_text', lambda _: fixed_copy)
    assert engine.report_lint(snapshot, b'unused', {})['banned_terms'] == []
    monkeypatch.setattr(engine, 'pdf_text', lambda _: fixed_copy + '\nThis proves the file is real.')
    assert engine.report_lint(snapshot, b'unused', {})['banned_terms'] == ['proves', 'real']


@pytest.mark.parametrize('kind', ['image', 'video', 'audio', 'document'])
def test_unavailable_evidence_retains_warning_without_a_forced_empty_chapter(kind):
    html = engine.render_html(specimen(kind, 'summary'), embed_fonts=False)
    assert 'evidence-unavailable' in html
    assert '<section class="chapter image-evidence' not in html
    assert '<section class="chapter multimedia-evidence' not in html
    assert html.count('For decisions with legal, financial or reputational consequences') == 1
