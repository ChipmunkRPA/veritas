"""Error responses must not disclose detector identities."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api import public_projection, v2_refusion, v2_rerun
from app.api.v2 import install_problem_handlers
from app.engine import refusion, rerun


@pytest.fixture
def client(monkeypatch):
    # Avoid the real registry and database while exercising the HTTP handlers.
    monkeypatch.setattr(public_projection, '_identities', lambda: ({'img.dda': 'check.test'}, ('DDA',)))
    app = FastAPI()
    install_problem_handlers(app)
    app.add_api_route('/api/v2/analyses/{aid}/rerun', v2_rerun.request_rerun, methods=['POST'])
    app.add_api_route('/api/v2/analyses/{aid}/refusion', v2_refusion.request_refusion, methods=['POST'])

    @app.middleware('http')
    async def set_principal(request: Request, call_next):
        request.state.principal = SimpleNamespace(id='test-user')
        return await call_next(request)

    with TestClient(app) as test_client:
        yield test_client


def assert_public_refusal(response):
    assert response.status_code == 409
    assert response.headers['content-type'].startswith('application/problem+json')
    assert 'unavailable' in response.json()['detail']
    assert 'img.dda' not in response.text.lower()
    assert 'dda' not in response.text.lower()


def test_rerun_refusal_hides_detector_identity(client, monkeypatch):
    source = SimpleNamespace(id='sample', status='done')
    monkeypatch.setattr(v2_rerun, 'get_session', lambda: nullcontext(object()))
    monkeypatch.setattr(v2_rerun, '_analysis', lambda db, aid: source)
    monkeypatch.setattr(v2_rerun.review, 'revision_base', lambda db, requested: source)
    monkeypatch.setattr(v2_rerun.review, 'superseded', lambda db, aid: False)
    monkeypatch.setattr(v2_rerun, 'signed_rerun_reason', lambda *args: None)
    monkeypatch.setattr(v2_rerun, '_published', lambda analysis: False)
    monkeypatch.setattr(v2_rerun, 'get_registry', lambda: object())

    def refuse(*args):
        raise rerun.RerunRefused('img.dda: DDA model file unavailable')

    monkeypatch.setattr(rerun, 'check', refuse)
    assert_public_refusal(client.post('/api/v2/analyses/sample/rerun', json={}))


def test_refusion_refusal_hides_detector_identity(client, monkeypatch):
    monkeypatch.setattr(v2_refusion.audit, 'writing', lambda: nullcontext(object()))
    monkeypatch.setattr(v2_refusion.audit, 'repeated', lambda unit: unit())
    monkeypatch.setattr(v2_refusion, 'get_registry', lambda: object())
    monkeypatch.setattr(refusion, 'check_source', lambda db, aid: object())

    def refuse(*args):
        raise refusion.RefusionError('img.dda: DDA model file unavailable')

    monkeypatch.setattr(refusion, 'compatible_selection', refuse)
    assert_public_refusal(client.post('/api/v2/analyses/sample/refusion', json={}))
