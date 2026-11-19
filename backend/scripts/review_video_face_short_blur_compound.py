"""Recount the independent matched local blur and swap follow-up."""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("swap", "gaussian", "swap_plus_blur")
SIGMAS = (4.0, 5.5, 7.0)
CLOCK = {"1/30000": 1001, "1/90000": 3003}
STATUSES = {
    "localized",
    "no_range",
    "unassessed",
    "preexisting_short",
    "held_preexisting_face_return",
    "held_separate_shot_boundary",
}
FIELDS = {
    "treatment",
    "cluster",
    "role",
    "sigma",
    "time_base",
    "status",
    "authored",
    "both_authored_edges_supported",
    "candidate",
    "withheld",
    "bound_frame_pts",
    "scalar_hex",
}
THRESHOLDS = {
    "min_reduction": 0.32,
    "min_loss": 0.50,
    "min_input_detail": 2.5,
    "min_fit_sigma": 1.25,
    "max_blur_error": 7.5,
    "max_coarse_error": 7.0,
    "flank_coarse_floor": 4.0,
    "flank_multiplier": 1.9,
    "flank_offset": 1.0,
}
OBSERVED = {
    "min_reduction",
    "min_loss",
    "min_input_detail",
    "min_fit_sigma",
    "max_blur_error",
    "max_coarse_error",
    "flank_coarse_error",
}
GATE = {
    "min_exact_compounds": 3,
    "min_compound_clusters": 2,
    "min_exact_clean": 3,
    "max_lost_compound": 0,
    "max_lost_clean": 0,
    "max_unrelated_suppressed": 0,
}


def qualifies(raw):
    assert type(raw) is dict and set(raw) == OBSERVED
    x = {key: float.fromhex(value) for key, value in raw.items()}
    assert all(math.isfinite(value) for value in x.values())
    t = THRESHOLDS
    if not all(
        x[key] >= t[key] for key in ("min_reduction", "min_loss", "min_input_detail", "min_fit_sigma")
    ):
        return False
    limit = max(t["flank_coarse_floor"], t["flank_multiplier"] * x["flank_coarse_error"] + t["flank_offset"])
    return x["max_blur_error"] <= t["max_blur_error"] and x["max_coarse_error"] <= min(
        limit, t["max_coarse_error"]
    )


def recount(receipt):
    assert receipt["schema"] == "chipmunk.short-face-softening-compound-result/1"
    assert receipt["thresholds"] == THRESHOLDS and receipt["frozen_gate"] == GATE
    assert receipt["prior_held_pass"] is False
    expected = {(i, role) for i in range(1, 10) for role in ROLES}
    triples = {}
    by_role = defaultdict(Counter)
    by_sigma = defaultdict(lambda: defaultdict(Counter))
    by_cluster = defaultdict(lambda: defaultdict(Counter))
    local_clusters = set()
    native_checks = 0
    for row in receipt["observations"]:
        assert set(row) == FIELDS
        assert type(row["treatment"]) is int and type(row["cluster"]) is int
        number, cluster, role = row["treatment"], row["cluster"], row["role"]
        clock, sigma = row["time_base"], row["sigma"]
        expected.remove((number, role))
        assert cluster == (number - 1) // 3 + 1 and sigma in SIGMAS and clock in CLOCK
        interval = row["authored"]
        a, b = interval
        assert 60 <= a < b <= 114 and b - a in (3, 4)
        binding = (cluster, sigma, interval, clock)
        assert triples.setdefault(number, binding) == binding
        state = row["status"]
        assert state in STATUSES
        assert type(row["both_authored_edges_supported"]) is bool and type(row["withheld"]) is bool
        proposal = row["candidate"]
        withheld = False
        if state == "localized":
            a, b = proposal
            assert 6 <= a < b <= 114 and b - a in (3, 4)
            witness = [a - 1, a + (b - a) // 2, b]
            assert row["bound_frame_pts"] == [[idx, idx * CLOCK[clock]] for idx in witness]
            native_checks += 3
            withheld = qualifies(row["scalar_hex"])
        else:
            assert proposal is None and row["scalar_hex"] is None and row["bound_frame_pts"] == []
        assert row["withheld"] is withheld
        exact = proposal == interval
        if exact and role == "swap_plus_blur":
            local_clusters.add(cluster)
        for results in (by_role, by_sigma[str(sigma)], by_cluster[cluster]):
            counts = results[role]
            counts["total"] += 1
            counts[state] += 1
            counts["exact"] += exact
            counts["exact_retained"] += exact and not withheld
            counts["withheld"] += withheld
            counts["unrelated_withheld"] += withheld and not exact
            counts["both_authored_edges_supported"] += row["both_authored_edges_supported"]
    assert not expected and len(triples) == 9 and receipt["fresh_cases"] == 27
    assert all({t[1] for t in triples.values() if t[0] == group} == set(SIGMAS) for group in range(1, 4))
    assert by_role == receipt["by_role"] and by_sigma == receipt["by_sigma"]
    assert len(local_clusters) == receipt["compound_localized_source_groups"]
    assert native_checks == receipt["native_crop_replays"]
    clean, compound = by_role["swap"], by_role["swap_plus_blur"]
    passed = (
        compound["exact"] >= GATE["min_exact_compounds"]
        and len(local_clusters) >= GATE["min_compound_clusters"]
        and clean["exact"] >= GATE["min_exact_clean"]
        and compound["withheld"] <= GATE["max_lost_compound"]
        and clean["withheld"] <= GATE["max_lost_clean"]
        and sum(c["unrelated_withheld"] for c in by_role.values()) <= GATE["max_unrelated_suppressed"]
    )
    assert passed == receipt["compound_only_gate_pass"]
    return {
        "compound_trial_pass": passed,
        "prior_held_pass": receipt["prior_held_pass"],
        "compound_source_groups": len(local_clusters),
        "by_role": by_role,
        "by_cluster": by_cluster,
        "by_sigma": by_sigma,
        "native_crop_replays": native_checks,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: review_video_face_short_blur_compound.py RECEIPT.json")
    print(json.dumps(recount(json.loads(Path(sys.argv[1]).read_bytes())), indent=2, sort_keys=True))
