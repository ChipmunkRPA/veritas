import numpy as np
import pytest

from scripts import image_flux_photo_expansion_v285 as expansion


def test_selection_excludes_entire_pilot_groups_and_is_deterministic():
    keys = ["pilot-one", "pilot-alias", "pilot-two", *(f"new-{n}" for n in range(6))]
    candidates = [
        {"key": key, "source_jpeg_sha256": key + "-hash", "size": [400, 400], "license_id": 4} for key in keys
    ]
    groups = [["pilot-one", "pilot-alias"], ["pilot-two"], *[[key] for key in keys[3:]]]
    pilot = [
        {"group": ["pilot-one"], "source_jpeg_sha256": "pilot-one-hash"},
        {"group": ["old-alias"], "source_jpeg_sha256": "pilot-two-hash"},
    ]
    selected, eligible = expansion.choose(candidates, groups, pilot)
    assert eligible == 6 and {row["key"] for row, _ in selected} == set(keys[3:])
    assert expansion.choose(list(reversed(candidates)), list(reversed(groups)), pilot) == (selected, eligible)
    with pytest.raises(ValueError, match="Both consumed pilot"):
        expansion.choose(candidates, groups[1:], pilot)


def test_batch_schedule_preserves_both_mask_placements():
    assert [[case[0] for case in batch] for batch in expansion.BATCHES] == [
        ["interior-1", "edge-1"],
        ["interior-2", "edge-2"],
        ["interior-3", "edge-3"],
    ]
    for batch in expansion.BATCHES:
        assert [(bounds, prompt) for _, bounds, prompt in batch] == [
            (bounds, prompt) for _, bounds, prompt in expansion.prior.CASES
        ]
    with pytest.raises(ValueError, match="frozen batches"):
        expansion.paths(0)


def test_readback_recovers_exact_composition_and_rejects_tamper(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(expansion, "TRIAL", root)
    slot = "edge-2"
    prepared, output = root / slot, root / "outputs"
    prepared.mkdir()
    output.mkdir()
    source, mask = expansion.flux.fixture("edge")
    raw = source.copy()
    raw[mask == 255] ^= 7
    raw[0, 1] ^= 9
    composite, truth, proof = expansion.flux.compose(source, raw, mask)
    before = {
        "slot": slot,
        "png_sha256": {
            name: expansion.flux.save_png(prepared / f"{name}.png", values)
            for name, values in (("source", source), ("mask", mask))
        },
    }
    row = {
        "slot": slot,
        "status": "qualified_geometry",
        **proof,
        **{
            name + "_png_sha256": expansion.flux.save_png(output / f"{name}.png", values)
            for name, values in (("raw", raw), ("composite", composite), ("changed", truth))
        },
    }
    result = expansion.verify_case(before, row, output)
    assert result["raw_changes_outside"] == 1 and result["composite_outside_changes"] == 0
    assert result["composite_changed_pixels"] == int(np.count_nonzero(mask))
    with pytest.raises(ValueError, match="Raw RGB footprint differs"):
        expansion.verify_case(before, {**row, "raw_changes_outside": 0}, output)
