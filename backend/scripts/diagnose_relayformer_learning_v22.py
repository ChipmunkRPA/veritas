"""Explicit authored-only gradient diagnostic after the retained first probe failure."""

import argparse
import json
from pathlib import Path

import torch

from scripts.relayformer_learning_v22 import Native, authored_image, configure, logits, prepare, regional_loss


def diagnose(folder):
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261108)
    model = Native().model
    parameters = configure(model)
    image, target = authored_image()
    tensor, geometry = prepare(image)
    values, _binding = logits(model, tensor, geometry)
    with torch.no_grad():
        canonical = model(image=tensor, origin_shape=torch.tensor([[1024, 1024]]), clip_len=torch.tensor([1]))
        parity = torch.equal(values.sigmoid(), canonical)
    loss = regional_loss(values, target)
    loss.backward()
    missing = [name for name, parameter in parameters.items() if parameter.grad is None]
    nonfinite = [name for name, parameter in parameters.items() if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    result = {"schema": "chipmunk.authored-regional-gradient-diagnostic/22", "selected_tensors": len(parameters), "missing_gradient_tensors": missing, "nonfinite_gradient_tensors": nonfinite, "native_inference_parity_exact": parity, "authored_forwards": 2, "authored_backward_passes": 1, "optimizer_updates": 0, "project_source_forwards": 0, "project_fitting_steps": 0, "scope": "Explicit new authored-only diagnostic; original probe remains failed. Parameter names describe implementation use, not detector quality. No source study, update or serving admission."}
    with (folder / "diagnostic.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    diagnose(args.folder)
