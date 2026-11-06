"""Public-safe report generations; older signed bytes are never rewritten."""
from app.reports import engine


def downloadable_template(template: str, kind: str | None = None) -> bool:
    # Earlier headlines could promote inconclusive results or overstate video
    # coverage. Generate a new public copy using the corrected presentation.
    return template == engine.TEMPLATE_SET


def package_template(template: str, kind: str | None = None) -> bool:
    return downloadable_template(template, kind)
