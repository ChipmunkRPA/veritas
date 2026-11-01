"""Public-safe report generations; older signed bytes are never rewritten."""
from app.reports import engine


def downloadable_template(template: str, kind: str | None = None) -> bool:
    # Pre-r64 generated methods, calibration rows and implementation coverage
    # can expose architecture/fusion identities for any media kind. A clean
    # individual fixture cannot establish that an entire old template is safe.
    return template in {engine.TEMPLATE_SET, 'veritas-report-2.0.r64', 'veritas-report-2.0.r65', 'veritas-report-2.0.r66'}


def package_template(template: str, kind: str | None = None) -> bool:
    return downloadable_template(template, kind)
