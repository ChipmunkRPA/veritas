from copy import deepcopy
from email import policy
from email.message import EmailMessage
from html import escape, unescape

import pytest

from app.core.canonical import canonical_hash
from app.evidence import email_alternatives as alternatives

OPENING = "The garden committee will meet after the morning class. Please bring the shared copies of the attendance list."
CLOSING = "The coordinator will arrange the tables before guests arrive. Everyone should use the side entrance during setup."
DIFFERENCE = "Please bring the decorated cahier to the volunteers waiting beside the front entrance."


def message(plain, html, *, mixed=False):
    item = EmailMessage(policy=policy.SMTP)
    item.set_content(plain, cte="quoted-printable")
    item.add_alternative(html, subtype="html", cte="base64")
    if mixed:
        item.add_attachment(b"Passive local fixture", maintype="application", subtype="octet-stream", filename="local.bin")
    return item.as_bytes()


def test_alternative_difference_binds_passive_words_to_decoded_mime_parts():
    visible = DIFFERENCE.replace("cahier", "ca<b>hi</b>er")
    raw = message(f"{OPENING}\n\n{CLOSING}", f'<p>{OPENING}</p><p>{visible}</p><p>{CLOSING}</p>', mixed=True)
    record = alternatives.review(raw)
    document = {"source": {"method": "email"}, "coverage": {"email": {"body_part": "1.1.1", "body_content_type": "text/plain"},
                alternatives.KEY: record}}
    assert alternatives.validate_set(document, canonical_hash(record)) == record
    assert record["status"] == "difference" and record["authorship_confidence_contribution"] == 0
    assert record["parts"]["html"]["id"] == "1.1.2"
    [difference] = record["differences"]
    assert set(difference) == {"id", "html"}
    assert difference["html"]["text"] == DIFFERENCE[:-1]
    html_body = alternatives.body_pair(raw)["html"][1]
    assert "cahier" in ''.join(unescape(html_body[a:b]) for a, b in difference["html"]["source_ranges"])
    assert alternatives.verify_source(document, raw) == record
    altered = deepcopy(document)
    altered["coverage"][alternatives.KEY]["differences"][0]["html"]["source_ranges"][0][0] += 1
    with pytest.raises(ValueError, match="coordinates differ"):
        alternatives.verify_source(altered, raw)


@pytest.mark.parametrize("plain,rich", [
    (f"{OPENING}\n\n> An unrelated quoted line.\n\n{CLOSING}", f"<p>{OPENING}</p><p>{CLOSING}</p>"),
    (f"{OPENING}\n\n{CLOSING}", f"<blockquote><p>{OPENING}</p></blockquote><p>{CLOSING}</p>"),
    (f"{OPENING}\n\n{CLOSING}", f'<p style="font-weight:bold">{OPENING}</p><p>{CLOSING}</p>'),
])
def test_ambiguous_reply_structure_abstains(plain, rich):
    assert alternatives.review(message(plain, rich))["status"] == "unsupported"


def test_only_anchored_service_footer_is_ignored():
    plain = f"{OPENING}\n\n{CLOSING}"
    base = f"<p>{OPENING}</p><p>{CLOSING}</p>"
    service = '<p>To unsubscribe from these notifications, visit <a href="https://example.invalid/preferences">preferences</a>.</p>'
    substantive = "To unsubscribe from these notifications, visit the public help page after you finish reading the instructions."
    assert alternatives.review(message(plain, base + service))["status"] == "equivalent"
    assert alternatives.review(message(plain, base + f"<p>{escape(substantive)}</p>"))["status"] == "difference"
