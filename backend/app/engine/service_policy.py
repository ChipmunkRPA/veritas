"""Versioned scope of the standard automated service, shared by all exports."""

RULE = "evidence-led-assessment/v4"
THRESHOLD = 0.6
CONTACT = "sales@chipmunkrpa.com"


def standard_service_policy() -> dict:
    return {
        "version": "standard-saas-evidence/v3",
        "service_tier": "standard_saas",
        "title": "Preliminary screening - not for court use",
        "confidence_notice": (
            "Automated deepfake screening is uncertain. Performance can vary substantially with generators, editing, "
            "compression, background noise and writing context. Use these results to guide further checks, not as proof. "
            "Displayed percentages are system estimates, not verified accuracy. "
            "Misses, false alarms and inconclusive results remain possible."
        ),
        "court_notice": (
            "This automated SaaS report has limited scientific reliability and is not prepared for court filing, "
            "proof of authenticity or expert testimony. Do not rely on it as court evidence. "
            "Automated checks can miss AI content and flag genuine content."
        ),
        "private_engagement_notice": (
            "For decisions with legal, financial or reputational consequences, request a separately scoped private "
            "investigation. An expert can examine source files, provenance, surrounding context and corroborating "
            "evidence. A private expert report or litigation support requires a separate signed engagement with the "
            "Chipmunk Verify expert, with methods, fees and deliverables agreed privately. "
            "Expert review does not guarantee a conclusion or court admissibility."
        ),
        "expert_contact": CONTACT,
        "court_use": "not_prepared_for_court_use",
        "separate_expert_engagement_required": True,
    }
