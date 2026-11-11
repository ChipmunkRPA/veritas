"""Version groups shared by storage, presentation and the offline verifier."""

ALIGNMENT_CONTEXT_VERSION = "video-evidence-8"
CONTINUITY_VERSIONS = frozenset({"video-evidence-5", "video-evidence-6", "video-evidence-7", ALIGNMENT_CONTEXT_VERSION})
ACOUSTIC_VERSIONS = CONTINUITY_VERSIONS | {"video-evidence-4"}
NESTED_VERSIONS = ACOUSTIC_VERSIONS | {"video-evidence-3"}
VERSIONS = NESTED_VERSIONS | {"video-evidence-1", "video-evidence-2"}
