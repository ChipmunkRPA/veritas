"""Declared route access (ARCHITECTURE §5.2 minimum roles, §5.3 RBAC; PRD CM-02; BUILD_SPEC C-1).

Every authenticated route has one entry `(method, path template) → minimum role`. `security.authorize` looks the matched
route up and fails closed (403) on a route without an entry; `tests/test_auth.py` checks that the table and the app's
routes agree. Minimum roles:
- `viewer`: any principal (reads, resource tokens);
- `examiner`: administrator, examiner or api_client (submits analyses);
- `reviewer`: administrator or reviewer;
- `administrator`: administrator only (principals, refusion, deletion, warm-up, metrics);
- `any`: an authenticated principal of any role (`/me`, sign-out).
"""
from __future__ import annotations

from collections.abc import Iterable

ROLES = ("administrator", "examiner", "reviewer", "viewer", "api_client")
SATISFIES: dict[str, frozenset[str]] = {
    "any": frozenset(ROLES),
    "viewer": frozenset(ROLES),
    "examiner": frozenset({"administrator", "examiner", "api_client"}),
    "reviewer": frozenset({"administrator", "reviewer"}),
    "administrator": frozenset({"administrator"}),
}
PUBLIC: frozenset[tuple[str, str]] = frozenset({("GET", "/api/health"), ("GET", "/api/ready"), ("POST", "/api/v2/auth/login"),
                                                ("POST", "/api/v2/auth/logout"), ("GET", "/api/v2/billing/config"),
                                                ("POST", "/api/v2/billing/webhook")})
ROUTES: dict[tuple[str, str], str] = {
    ("POST", "/api/v2/billing/checkout"): "examiner",
    ("POST", "/api/v2/billing/orders/{order_id}/checkout"): "examiner",
    ("GET", "/api/v2/billing/orders/{order_id}"): "viewer",
    # v1 (/api): kept through v2.1 (API-03)
    ("POST", "/api/analyze/{kind}"): "examiner",
    ("POST", "/api/analyze-text"): "examiner",
    ("GET", "/api/analyses"): "viewer",
    ("GET", "/api/analyses/{aid}"): "viewer",
    ("DELETE", "/api/analyses/{aid}"): "administrator",
    ("GET", "/api/analyses/{aid}/artifacts/{name}"): "viewer",
    ("GET", "/api/analyses/{aid}/source"): "viewer",
    ("GET", "/api/analyses/{aid}/download-token"): "viewer",
    ("GET", "/api/analyses/{aid}/report.pdf"): "viewer",
    ("GET", "/api/analyses/{aid}/export.json"): "viewer",
    ("GET", "/api/metrics"): "administrator",
    ("GET", "/api/models"): "viewer",
    ("GET", "/api/demo"): "viewer",
    ("POST", "/api/demo/start"): "examiner",
    # v2 read API, progress, tokens, models, refusion
    ("GET", "/api/v2/analyses"): "viewer",
    ("GET", "/api/v2/analyses/{aid}"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/cancel"): "examiner",
    ("GET", "/api/v2/analyses/{aid}/stages"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/detector-runs"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/signals"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/findings"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/assessments"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/exhibits"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/evidence-packets"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/evidence-packets/{set_id}/images/{sha256}.png"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/evidence-packets/{set_id}/assets/{sha256}.{extension}"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/image-reference-comparison"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/image-reference-comparison"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/document-revision-comparison"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/document-revision-comparison"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/stills/{pts}.jpg"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/frames"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/progress"): "viewer",
    ("GET", "/api/v2/analyses/{aid}/events"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/refusion"): "administrator",
    ("GET", "/api/v2/exhibits/{exhibit_id}"): "viewer",
    ("GET", "/api/v2/exhibits/{exhibit_id}/payload"): "viewer",
    ("GET", "/api/v2/exhibits/{exhibit_id}/mask"): "viewer",
    ("GET", "/api/v2/exhibits/{exhibit_id}/preview"): "viewer",
    ("GET", "/api/v2/exhibits/{exhibit_id}/tiles/{z}/{x}_{y}.png"): "viewer",
    ("POST", "/api/v2/tokens"): "viewer",
    ("GET", "/api/v2/models"): "viewer",
    ("GET", "/api/v2/models/{detector_id}"): "viewer",
    ("GET", "/api/v2/models/{detector_id}/records/{record_id}"): "administrator",  # D-71: an engineering record is not product content
    # principals and sessions (C-1)
    ("GET", "/api/v2/me"): "any",
    ("POST", "/api/v2/me/password"): "any",
    ("GET", "/api/v2/principals"): "administrator",
    ("POST", "/api/v2/principals"): "administrator",
    ("PATCH", "/api/v2/principals/{principal_id}"): "administrator",
    ("POST", "/api/v2/principals/{principal_id}/api-keys"): "administrator",
    # cases, members, intake, media (C-2); case resources also need membership (app.cases.access)
    ("POST", "/api/v2/cases"): "examiner",
    ("GET", "/api/v2/cases"): "viewer",
    ("GET", "/api/v2/cases/{case_id}"): "viewer",
    ("PATCH", "/api/v2/cases/{case_id}"): "examiner",
    ("GET", "/api/v2/cases/{case_id}/members"): "viewer",
    ("GET", "/api/v2/cases/{case_id}/members/candidates"): "examiner",  # the owner, checked in the handler (D-67)
    ("PUT", "/api/v2/cases/{case_id}/members/{principal_id}"): "examiner",
    ("DELETE", "/api/v2/cases/{case_id}/members/{principal_id}"): "examiner",
    ("POST", "/api/v2/cases/{case_id}/items"): "examiner",
    ("GET", "/api/v2/cases/{case_id}/items"): "viewer",
    ("GET", "/api/v2/media/{media_id}"): "viewer",
    ("GET", "/api/v2/media/{media_id}/original"): "viewer",
    # audit log and custody (C-3)
    ("GET", "/api/v2/audit"): "reviewer",
    ("GET", "/api/v2/audit/verify"): "reviewer",
    ("GET", "/api/v2/audit/export"): "reviewer",
    ("GET", "/api/v2/cases/{case_id}/custody"): "viewer",  # the owner, a reviewer of the case or an administrator (D-67)
    ("GET", "/api/v2/cases/{case_id}/custody/verify"): "viewer",
    # examiner layer (C-4): the author or an administrator changes an annotation (checked in the handler)
    ("GET", "/api/v2/analyses/{aid}/annotations"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/annotations"): "examiner",
    ("PATCH", "/api/v2/annotations/{annotation_id}"): "examiner",
    ("DELETE", "/api/v2/annotations/{annotation_id}"): "examiner",
    # the machine layer refuses writes with 403 for every role (RW-01)
    **{(method, "/api/v2/analyses/{aid}/signals"): "viewer" for method in ("POST", "PUT", "PATCH", "DELETE")},
    **{(method, "/api/v2/analyses/{aid}/signals/{signal_id}"): "viewer" for method in ("PUT", "PATCH", "DELETE")},
    # review and sign-off (C-5): the handler checks the role and case role each sign-off kind needs
    ("GET", "/api/v2/analyses/{aid}/conclusion"): "viewer",
    ("PUT", "/api/v2/analyses/{aid}/conclusion"): "examiner",
    ("GET", "/api/v2/analyses/{aid}/signoffs"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/signoffs"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/withdraw"): "examiner",
    ("POST", "/api/v2/analyses/{aid}/reanalyse"): "examiner",
    # reports (R-1): a report belongs to its analysis's case
    ("POST", "/api/v2/analyses/{aid}/reports"): "examiner",
    ("GET", "/api/v2/analyses/{aid}/reports"): "viewer",
    ("GET", "/api/v2/reports/{report_id}"): "viewer",
    ("GET", "/api/v2/reports/{report_id}/file"): "viewer",
    ("GET", "/api/v2/reports/{report_id}/preview.html"): "viewer",
    # signatures and evidence packages (R-4): a package belongs to its analysis's case; exporting one is examiner work (§5.2)
    ("GET", "/api/v2/instance/public-key"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/packages"): "examiner",
    ("GET", "/api/v2/packages/{package_id}"): "examiner",
    ("GET", "/api/v2/packages/{package_id}/file"): "examiner",
    # batches and webhooks (X-1, §5.2): a batch belongs to its case
    ("POST", "/api/v2/batches"): "examiner",
    ("GET", "/api/v2/batches/{batch_id}"): "examiner",
    ("GET", "/api/v2/batches/{batch_id}/summary.json"): "examiner",
    ("GET", "/api/v2/batches/{batch_id}/summary.csv"): "examiner",
    ("GET", "/api/v2/webhooks"): "administrator",
    ("POST", "/api/v2/webhooks"): "administrator",
    ("PATCH", "/api/v2/webhooks/{webhook_id}"): "administrator",
    ("DELETE", "/api/v2/webhooks/{webhook_id}"): "administrator",
    ("GET", "/api/v2/webhooks/{webhook_id}/deliveries"): "administrator",
    # policy thresholds and re-run (X-2, §5.2): analyses name the policy version they used; a revision's deltas belong to its case
    ("GET", "/api/v2/policy/thresholds"): "viewer",
    ("PUT", "/api/v2/policy/thresholds"): "administrator",
    ("GET", "/api/v2/policy/thresholds/versions"): "viewer",
    ("GET", "/api/v2/policy/thresholds/{policy_id}"): "viewer",
    ("POST", "/api/v2/analyses/{aid}/rerun"): "examiner",
    ("GET", "/api/v2/analyses/{aid}/deltas"): "viewer",
}


def satisfies(roles: Iterable[str], minimum: str) -> bool:
    return bool(set(roles) & SATISFIES[minimum])


def minimum_role(method: str, path: str) -> str | None:
    """The declared minimum role of a route template, or None when the route has no rule (fail closed)."""
    return ROUTES.get((method.upper(), path))
