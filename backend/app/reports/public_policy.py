"""Public-safe report generations; older signed bytes are never rewritten."""
from app.reports import engine

# These generations already used the public-safe headline and check labels.
_PREVIOUS_PUBLIC_TEMPLATES = {
    "veritas-report-2.0.r83", "veritas-report-2.0.r84", "veritas-report-2.0.r85", "veritas-report-2.0.r87", "veritas-report-2.0.r88",
    "veritas-report-2.0.r89", "veritas-report-2.0.r90",
}


def downloadable_template(template: str, kind: str | None = None) -> bool:
    # Earlier headlines could promote inconclusive results or overstate video
    # coverage. Generate a new public copy using the corrected presentation.
    return template == engine.TEMPLATE_SET or template in _PREVIOUS_PUBLIC_TEMPLATES


def package_template(template: str, kind: str | None = None) -> bool:
    return downloadable_template(template, kind)
