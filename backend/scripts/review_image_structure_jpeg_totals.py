"""Check the published JPEG structure comparison's aggregate arithmetic."""

import argparse
import json
from pathlib import Path

EDITS = ("sharp", "feathered")
ORDINARY = ("global_blur", "no_tone", "ordinary_graded", "ordinary_roundtrip", "resize_lanczos", "resize_roundtrip")
INCOMPATIBLE = ("fractional", "translated", "unrelated")
LOCAL = ("local_blur", "local_copy", "local_tone")
ROLES = EDITS + ORDINARY + INCOMPATIBLE + LOCAL
MEASURES = ("useful", "flagged", "compared", "views")
ARMS = ("current", "monotone_fallback", "sharpness_fallback", "structure_fallback")


def review(document):
    sources = document["sources"]
    if (document["schema"] != "chipmunk.image-reference-structure-jpeg-transfer/1"
            or type(sources) is not int or sources < 1
            or document["comparisons"] != document["exact_replays"] or document["comparisons"] != sources * len(ROLES)
            or set(document["strata"]) != set(document["paired"]) or set(document["strata"]) != set(ARMS)):
        raise ValueError("Invalid panel size or comparison roster")
    summaries = {}
    for arm in ARMS:
        summary = document["strata"][arm]
        if set(summary) != {f"{role}/{measure}" for role in ROLES for measure in MEASURES}:
            raise ValueError("Missing role or measure")
        for role in ROLES:
            counts = [summary[f"{role}/{measure}"] for measure in MEASURES]
            if any(type(value) is not int or value < 0 for value in counts) or counts != sorted(counts) or counts[-1] != sources:
                raise ValueError("Inconsistent counts for " + role)
        pair = document["paired"][arm]
        control_flags = sum(summary[f"{role}/flagged"] for role in ORDINARY)
        incompatible = sum(summary[f"{role}/compared"] for role in INCOMPATIBLE)
        if incompatible != pair["incompatible_compared"]:
            raise ValueError("Incompatible-pair totals differ")
        summaries[arm] = {"sharp": summary["sharp/useful"], "feathered": summary["feathered/useful"],
                          "ordinary_highlights": control_flags, "incompatible_compared": incompatible,
                          "ordinary_groups_highlighted": pair["ordinary_groups_flagged"]}
        if arm != "current":
            incumbent = summaries["current"]
            actual = summaries[arm]
            for key in EDITS:
                if pair[f"new_{key}"] != actual[key] - incumbent[key]:
                    raise ValueError("Added edit totals differ")
            if (pair["new_nuisance_files"] != control_flags - incumbent["ordinary_highlights"]
                    or pair["new_nuisance_groups"] != pair["ordinary_groups_flagged"] - incumbent["ordinary_groups_highlighted"]
                    or pair["passes"] != (min(pair["new_sharp"], pair["new_feathered"]) >= 2
                                          and pair["new_nuisance_files"] <= 2 and pair["new_nuisance_groups"] <= 1
                                          and incompatible == 0)):
                raise ValueError("Paired control totals or admission differ")
    return {"source_groups": sources, "ordinary_files": sources * len(ORDINARY), "arms": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.ledger.read_bytes())), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
