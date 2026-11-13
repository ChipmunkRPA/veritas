from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from starlette.responses import Response

from app.api import v2_reports
from app.api.v2 import ProblemError


@contextmanager
def _session():
    yield object()


@pytest.mark.parametrize('template', ['veritas-report-2.0.r47', 'veritas-report-2.0.r50', 'veritas-report-2.0.r51', 'veritas-report-2.0.r63',
                                     'veritas-report-2.0.r64', 'veritas-report-2.0.r65', 'veritas-report-2.0.r66',
                                     'veritas-report-2.0.r67', 'veritas-report-2.0.r68'])
def test_old_reports_are_preserved_but_not_served(monkeypatch, template):
    row = SimpleNamespace(status='done', artefact_id='old', template_set=template, analysis_id='analysis')
    monkeypatch.setattr(v2_reports, 'get_session', _session)
    monkeypatch.setattr(v2_reports, '_report', lambda db, ident: row)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: SimpleNamespace(kind='image'))
    for endpoint in (lambda: v2_reports.report_file('old', None), lambda: v2_reports.report_preview('old')):
        with pytest.raises(ProblemError) as error:
            endpoint()
        assert error.value.status == 409
        assert 'new report' in error.value.detail.lower()
        assert 'remains stored' in error.value.detail
    assert row.artefact_id == 'old'


@pytest.mark.parametrize('template', [v2_reports.engine.TEMPLATE_SET, 'veritas-report-2.0.r87', 'veritas-report-2.0.r88', 'veritas-report-2.0.r89',
                                     'veritas-report-2.0.r90', 'veritas-report-2.0.r91', 'veritas-report-2.0.r92', 'veritas-report-2.0.r94',
                                     'veritas-report-2.0.r95', 'veritas-report-2.0.r96', 'veritas-report-2.0.r97', 'veritas-report-2.0.r98',
                                     'veritas-report-2.0.r99', 'veritas-report-2.0.r100', 'veritas-report-2.0.r101',
                                     'veritas-report-2.0.r102', 'veritas-report-2.0.r103', 'veritas-report-2.0.r104',
                                     'veritas-report-2.0.r105', 'veritas-report-2.0.r106', 'veritas-report-2.0.r107',
                                     'veritas-report-2.0.r108'])
def test_public_safe_reports_remain_downloadable(monkeypatch, template):
    from app.engine.fusion.runtime import FACE_SCOPE_CAVEAT
    from app.reports.public_policy import package_template

    row = SimpleNamespace(id='report', status='done', artefact_id='new', template_set=template,
                          analysis_id='analysis', display_id='report')
    monkeypatch.setattr(v2_reports, 'get_session', _session)
    monkeypatch.setattr(v2_reports, '_report', lambda db, ident: row)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: SimpleNamespace(kind='image'))
    caveat = (FACE_SCOPE_CAVEAT.format(name='Previously named face check') if template == 'veritas-report-2.0.r101'
              else 'Source metadata was retained.')
    monkeypatch.setattr(v2_reports, '_public_snapshot', lambda db, row: SimpleNamespace(
        report=SimpleNamespace(id=row.id, template_set=row.template_set),
        analysis=SimpleNamespace(id=row.analysis_id, kind='image'), result={'caveats': [caveat]}))
    monkeypatch.setattr(v2_reports, '_artefact', lambda *args: object())
    monkeypatch.setattr(v2_reports, '_binary', lambda *args: Response(content=b'pdf', media_type='application/pdf'))
    response = v2_reports.report_file('new', None)
    assert response.status_code == 200
    assert response.body == b'pdf'
    assert package_template(template, 'image')
    if template != v2_reports.engine.TEMPLATE_SET:
        with pytest.raises(ProblemError) as error:
            v2_reports.report_preview('new')
        assert error.value.status == 409
        assert 'saved PDF is still available' in error.value.detail


def test_only_named_legacy_face_scope_requires_report_and_package_regeneration(monkeypatch):
    from app.engine.fusion.runtime import FACE_SCOPE_CAVEAT

    row = SimpleNamespace(id='report', status='done', artefact_id='original', template_set='veritas-report-2.0.r100',
                          analysis_id='analysis', display_id='report')
    package = SimpleNamespace(status='done', artefact_id='original-package', schema_id=v2_reports.package.PUBLIC_SCHEMA,
                              analysis_id='analysis', report_id=row.id)
    db = SimpleNamespace(get=lambda model, ident: row)

    @contextmanager
    def session():
        yield db

    frozen = SimpleNamespace(
        report=SimpleNamespace(id=row.id, template_set=row.template_set),
        analysis=SimpleNamespace(id=row.analysis_id, kind='image'),
        result={'caveats': [FACE_SCOPE_CAVEAT.format(name='Previously named face check')]})
    monkeypatch.setattr(v2_reports, 'get_session', session)
    monkeypatch.setattr(v2_reports, '_report', lambda db, ident: row)
    monkeypatch.setattr(v2_reports, '_package', lambda db, ident: package)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: SimpleNamespace(kind='image'))
    monkeypatch.setattr(v2_reports, '_public_snapshot', lambda db, row: frozen)

    for endpoint in (lambda: v2_reports.report_file(row.id, None), lambda: v2_reports.report_preview(row.id),
                     lambda: v2_reports.package_file('package', None)):
        with pytest.raises(ProblemError) as error:
            endpoint()
        assert error.value.status == 409
        assert 'new report' in error.value.detail.lower()
    assert row.artefact_id == 'original'
    assert package.artefact_id == 'original-package'


def test_internal_package_download_stays_blocked(monkeypatch):
    monkeypatch.setattr(v2_reports, 'get_session', _session)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: object())
    monkeypatch.setattr(v2_reports, '_package', lambda db, ident: SimpleNamespace(schema_id=v2_reports.package.SCHEMA))
    with pytest.raises(ProblemError) as error:
        v2_reports.package_file('old-package', None)
    assert error.value.status == 409
    assert 'new public evidence package' in error.value.detail
    assert 'stored' in error.value.detail


@pytest.mark.parametrize('kind', ['image', 'audio', 'video', 'document'])
def test_reports_using_the_old_headline_rule_require_a_new_copy(kind):
    from app.reports.public_policy import downloadable_template, package_template

    for check in (downloadable_template, package_template):
        for revision in range(18, 72):
            assert not check(f'veritas-report-2.0.r{revision}', kind)
        assert check(v2_reports.engine.TEMPLATE_SET, kind)


@pytest.mark.parametrize('template', ['veritas-report-2.0.r83', 'veritas-report-2.0.r84', 'veritas-report-2.0.r85', 'veritas-report-2.0.r87'])
def test_previous_public_safe_report_stays_downloadable_after_clock_evidence_update(template):
    from app.reports.public_policy import downloadable_template, package_template

    assert downloadable_template(template, 'audio')
    assert package_template(template, 'audio')


def test_r60_report_with_private_contribution_prose_is_preserved_but_withheld(monkeypatch):
    from app.reports.public_policy import downloadable_template, package_template

    row = SimpleNamespace(status='done', artefact_id='signed-r60-original',
                          template_set='veritas-report-2.0.r60', analysis_id='analysis')
    monkeypatch.setattr(v2_reports, 'get_session', _session)
    monkeypatch.setattr(v2_reports, '_report', lambda db, ident: row)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: SimpleNamespace(kind='video'))
    assert not downloadable_template(row.template_set, 'video')
    assert not package_template(row.template_set, 'video')
    for endpoint in (lambda: v2_reports.report_file('old', None), lambda: v2_reports.report_preview('old')):
        with pytest.raises(ProblemError) as error:
            endpoint()
        assert error.value.status == 409
    assert row.artefact_id == 'signed-r60-original'


@pytest.mark.parametrize('kind', ['document', 'audio', 'video'])
@pytest.mark.parametrize('template', ['veritas-report-2.0.r48', 'veritas-report-2.0.r49',
                                     'veritas-report-2.0.r52', 'veritas-report-2.0.r53', 'veritas-report-2.0.r54'])
def test_unverified_historical_nonimage_reports_stay_stored_but_unavailable(monkeypatch, kind, template):
    from app.reports.public_policy import downloadable_template, package_template

    row = SimpleNamespace(status='done', artefact_id='signed-original', template_set=template, analysis_id='analysis')
    monkeypatch.setattr(v2_reports, 'get_session', _session)
    monkeypatch.setattr(v2_reports, '_report', lambda db, ident: row)
    monkeypatch.setattr(v2_reports, '_analysis', lambda db, ident: SimpleNamespace(kind=kind))
    assert not downloadable_template(template, kind)
    assert not package_template(template, kind)
    with pytest.raises(ProblemError) as error:
        v2_reports.report_file('old-report', None)
    assert error.value.status == 409
    assert row.artefact_id == 'signed-original'
    assert downloadable_template(v2_reports.engine.TEMPLATE_SET, kind)
