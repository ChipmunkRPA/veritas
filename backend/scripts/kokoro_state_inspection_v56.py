"""Read-only checkpoint/key inspection after the preserved failed strict-load check."""
from __future__ import annotations

import json
from pathlib import Path


def inspect():
    import torch
    from kokoro import KModel

    root = Path(__file__).resolve().parents[2] / "backend/data/independent-evaluation/kokoro-state-inspection-v56-a03"
    receipt = root.parent / "audio-generator-runtime-v54/asset-receipt.json"
    assets = Path(json.loads(receipt.read_text())["private_root"])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(assets / "config.json"),
                   model=str(assets / "kokoro-v1_0.pth")).cpu().float().eval()
    saved = torch.load(assets / "kokoro-v1_0.pth", weights_only=True, map_location="cpu")
    rows = []
    for name, module in model.named_children():
        a, b = module.state_dict(), saved[name]
        exact = set(a) & set(b)
        rows.append({"module": name, "expected_keys": len(a), "saved_keys": len(b),
                     "missing": sorted(set(a) - set(b)), "unexpected": sorted(set(b) - set(a)),
                     "matching_keys": len(exact), "exact_loaded_matching_keys": sum(
                         a[key].shape == b[key].shape and a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])
                         for key in exact)})
    result = {"status": "complete", "scope": "Read-only constructor and tensor-key inspection; no G2P or generator/detector forward.",
              "generator_forwards": 0, "detector_forwards": 0, "modules": rows}
    folder = root / "smoke"
    folder.mkdir(exist_ok=False)
    with (folder / "completed.json").open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    inspect()
