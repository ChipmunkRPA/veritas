import pytest

from scripts.prepare_audio_multisource_v8 import ROLES
from scripts.prepare_audio_multisource_variants_v8 import background_pools


def backgrounds():
    return [
        {"role": role, "kind": kind, "id": f"{role}:{kind}:{n}", "sha256": f"{role}:{kind}:{n}"}
        for role in ROLES
        for kind, count in (("noise", 8), ("music", 4), ("speech", 6 if role == "confirmation" else 4))
        for n in range(count)
    ]


def test_backgrounds_stay_in_their_own_roles():
    pools = background_pools(backgrounds())
    assert len(pools) == 12
    assert all(r["role"] == role and r["kind"] == kind for (role, kind), rows in pools.items() for r in rows)


def test_missing_background_is_not_silently_replaced():
    with pytest.raises(ValueError, match="complete fixed"):
        background_pools(backgrounds()[:-1])


def test_duplicate_recording_across_roles_is_refused():
    rows = backgrounds()
    rows[-1]["sha256"] = rows[0]["sha256"]
    with pytest.raises(ValueError, match="repeated across"):
        background_pools(rows)
