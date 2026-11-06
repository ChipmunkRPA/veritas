import copy
import hashlib
import struct

import pytest

from scripts.mouth_probe_capsule import validate_vectors


@pytest.mark.parametrize("corruption", ["vector", "actor", "duplicate"])
def test_public_fitting_capsule_refuses_changed_vectors_or_source_separation(corruption):
    vector = [0.125] * 768
    rows = [
        {
            "id": role,
            "group": role,
            "role": role,
            "label": 0,
            "condition": "original",
            "features": list(vector),
            "features_float32_sha256": hashlib.sha256(struct.pack("<768f", *vector)).hexdigest(),
        }
        for role in ("fit", "point")
    ]
    head = {"fit_groups": ["fit"], "point_groups": ["point"]}
    validate_vectors(rows, head)
    altered = copy.deepcopy(rows)
    if corruption == "vector":
        altered[0]["features"][0] += 0.25
    elif corruption == "actor":
        altered[1]["group"] = "fit"
    else:
        altered.append(copy.deepcopy(altered[0]))
    with pytest.raises(ValueError):
        validate_vectors(altered, head)
