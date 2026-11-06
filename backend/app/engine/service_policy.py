"""Versioned scope of the standard automated service, shared by all exports."""

RULE = "evidence-led-assessment/v4"
THRESHOLD = 0.6
CONTACT = "sales@chipmunkrpa.com"


def standard_service_policy() -> dict:
    return {
        "version": "standard-saas-evidence/v2",
        "service_tier": "standard_saas",
        "title": "Preliminary screening - not for court use",
        "confidence_notice": (
            "Findings link observations to their supporting evidence and distinguish AI-leaning, original-leaning and inconclusive results. "
            "Displayed percentages are system estimates, not verified accuracy or proof. "
            "Unsupported and inconclusive checks remain unresolved."
        ),
        "court_notice": (
            "This automated SaaS report has limited scientific reliability and is not prepared for court filing, "
            "proof of authenticity or expert testimony. Do not rely on it as court evidence. "
            "Automated checks can miss AI content and flag genuine content."
        ),
        "private_engagement_notice": (
            "A privately researched report, expert opinion or litigation support requires a separate signed "
            "engagement with the Chipmunk Verify expert. Scope, methods, fees and deliverables are agreed "
            "privately. A private engagement does not guarantee court admissibility."
        ),
        "expert_contact": CONTACT,
        "court_use": "not_prepared_for_court_use",
        "separate_expert_engagement_required": True,
    }
