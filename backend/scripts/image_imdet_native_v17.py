"""Research-only exact ONNX graph and source-crop contract; no pixel attribution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_image_imdet_v17 import MODEL_SHA, REVISION, ROOT, digest

ASSETS = ROOT / "models/research/image-v17-imdet"
RUNTIME = ROOT / "backend/data/independent-evaluation/image-imdet-v17-runtime/packages"
LOGIT_BOUND, RESPONSE_BOUND = 0.01, 0.001


def require(value, reason):
    if not value:
        raise ValueError(reason)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def preprocess(image):
    require(isinstance(image, Image.Image), "Explicit already-oriented source image required")
    source = image.convert("RGB")
    w, h = source.size
    require(min(w, h) > 0, "Nonempty source image required")
    ow, oh = (256, int(256 * h / w)) if w <= h else (int(256 * w / h), 256)
    left, top = round((ow - 256) / 2), round((oh - 256) / 2)
    resized = source.resize((ow, oh), Image.Resampling.BILINEAR)
    crop = resized.crop((left, top, left + 256, top + 256))
    tensor = np.ascontiguousarray(np.asarray(crop).transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(
        255
    )
    geometry = {
        "source_size": [w, h],
        "resized_size": [ow, oh],
        "resized_crop_xyxy": [left, top, left + 256, top + 256],
        "inspected_source_xyxy": [left * w / ow, top * h / oh, (left + 256) * w / ow, (top + 256) * h / oh],
        "conservative_resampling_support_xyxy": [
            max(0, math.floor(left * w / ow - max(1, w / ow))),
            max(0, math.floor(top * h / oh - max(1, h / oh))),
            min(w, math.ceil((left + 256) * w / ow + max(1, w / ow))),
            min(h, math.ceil((top + 256) * h / oh + max(1, h / oh))),
        ],
        "source_rgb_sha256": array_sha(np.asarray(source)),
        "input_float32_sha256": array_sha(tensor),
        "source_area_fraction": 256 * 256 / (ow * oh),
        "coordinates_mean": "Nominal center-crop bounds; bilinear sampling uses a conservative neighbor-pixel support box. Neither is an edit mask.",
    }
    require(
        tensor.shape == (1, 3, 256, 256)
        and tensor.dtype == np.float32
        and np.isfinite(tensor).all()
        and 0 <= float(tensor.min()) <= float(tensor.max()) <= 1,
        "Complete published finite input tensor required",
    )
    return tensor, geometry


def libraries():
    require(RUNTIME.is_dir(), "Isolated runtime must exist; no implicit installation")
    sys.path.insert(0, str(RUNTIME))
    import onnx
    import onnxruntime as ort

    require(
        Path(onnx.__file__).is_relative_to(RUNTIME) and Path(ort.__file__).is_relative_to(RUNTIME),
        "Only the separate verified research runtime may load",
    )
    require(onnx.__version__ == "1.19.1" and ort.__version__ == "1.23.2", "Pinned runtime versions required")
    return onnx, ort


def inspect():
    onnx, ort = libraries()
    record = json.loads((ROOT / "docs/evaluation-panels/image-imdet-v17-acquisition.json").read_text())
    require(
        record["revision"] == REVISION and record["new_model_forwards"] == 0, "Exact acquisition required"
    )
    for name, asset in record["assets"].items():
        require(digest(ASSETS / name) == asset["sha256"], "Pinned checkpoint/license/config asset changed")
    path = ASSETS / "pe_tiny/model.onnx"
    require(digest(path) == MODEL_SHA, "Exact released ONNX graph required")
    graph = onnx.load_model(path, load_external_data=False)
    require(
        not graph.functions and not graph.graph.sparse_initializer,
        "No unreviewed function or sparse-weight path",
    )
    require(all(n.domain in ("", "ai.onnx") for n in graph.graph.node), "Standard ONNX operators only")
    for tensor in graph.graph.initializer:
        require(
            not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL,
            "No external tensor references",
        )
        values = onnx.numpy_helper.to_array(tensor)
        require(np.isfinite(values).all(), "Non-finite released initializer")
    for node in graph.graph.node:
        for attribute in node.attribute:
            require(
                attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS),
                "No unreviewed nested execution graph",
            )
            for tensor in [attribute.t] if attribute.HasField("t") else list(attribute.tensors):
                require(not tensor.external_data, "External constant tensor forbidden")
    onnx.checker.check_model(graph, full_check=True)
    files = [
        Path(onnx.__file__),
        Path(ort.__file__),
        *Path(ort.__file__).parent.glob("capi/*.so"),
        *Path(ort.__file__).parent.glob("capi/*.dylib"),
    ]
    return {
        "revision": REVISION,
        "model_sha256": MODEL_SHA,
        "ir_version": graph.ir_version,
        "opsets": {op.domain: op.version for op in graph.opset_import},
        "nodes": len(graph.graph.node),
        "operators": dict(sorted(Counter(n.op_type for n in graph.graph.node).items())),
        "initializers": len(graph.graph.initializer),
        "initializer_elements": sum(int(np.prod(t.dims, dtype=np.int64)) for t in graph.graph.initializer),
        "initializer_types": dict(
            Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in graph.graph.initializer)
        ),
        "external_tensors": 0,
        "custom_operator_domains": 0,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in (
                "onnx",
                "onnxruntime",
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "ml_dtypes",
                "protobuf",
            )
        },
        "isolated_native_code_sha256": {str(p.relative_to(RUNTIME)): digest(p) for p in files},
        "new_model_forwards": 0,
        "serving_admitted": False,
    }


class Native:
    def __init__(self):
        self.inspection = inspect()
        _, ort = libraries()
        self.sessions = {}
        self.attempted_forwards = self.completed_forwards = 0
        for name, optimization in (
            ("canonical", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
            ("optimized", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        ):
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = optimization
            session = ort.InferenceSession(
                str(ASSETS / "pe_tiny/model.onnx"), options, providers=["CPUExecutionProvider"]
            )
            require(
                session.get_providers() == ["CPUExecutionProvider"]
                and len(session.get_inputs()) == 1
                and len(session.get_outputs()) == 1,
                "Only the exact CPU graph input/output path is supported",
            )
            inp, out = session.get_inputs()[0], session.get_outputs()[0]
            require(
                inp.type == "tensor(float)"
                and list(inp.shape[1:]) == [3, 256, 256]
                and out.type == "tensor(float)"
                and out.shape[-1] == 2,
                "Published float32 RGB tensor/two-logit output contract changed",
            )
            self.sessions[name] = session

    def predict(self, tensor, mode="canonical"):
        require(
            mode in self.sessions
            and tensor.shape == (1, 3, 256, 256)
            and tensor.dtype == np.float32
            and np.isfinite(tensor).all()
            and 0 <= float(tensor.min()) <= float(tensor.max()) <= 1,
            "Exact finite model input required",
        )
        session = self.sessions[mode]
        before = array_sha(tensor)
        self.attempted_forwards += 1
        output = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0]
        self.completed_forwards += 1
        require(
            before == array_sha(tensor) and output.shape == (1, 2) and np.isfinite(output).all(),
            "Actual graph must preserve input and return both finite logits",
        )
        logits = output[0].astype(np.float64)
        values = np.exp(logits - logits.max())
        return logits.tolist(), float(values[0] / values.sum())


def fixture(w, h):
    y, x = np.indices((h, w), dtype=np.uint32)
    return Image.fromarray(
        np.stack([(x * 17 + y * 3) % 256, (y * 11 + x) % 256, (x * 7 + y * 5) % 256], -1).astype(np.uint8)
    )


def smoke():
    from torchvision import transforms

    native = Native()
    reference = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    rows = []
    for w, h in ((31, 77), (88, 47), (256, 256), (515, 389), (389, 515), (1025, 129)):
        image = fixture(w, h)
        tensor, geometry = preprocess(image)
        require(
            np.array_equal(tensor, reference(image).unsqueeze(0).numpy()),
            "Independent source resize/crop/tensor path differs from published reference",
        )
        a, qa = native.predict(tensor)
        repeat, qr = native.predict(tensor)
        b, qb = native.predict(tensor, "optimized")
        delta = max(abs(x - y) for x, y in zip(a, b, strict=True))
        require(
            a == repeat and qa == qr and delta <= LOGIT_BOUND and abs(qa - qb) <= RESPONSE_BOUND,
            "Exact canonical repeat or prospective optimization bound failed",
        )
        rows.append(
            {
                "geometry": geometry,
                "canonical_logits": a,
                "optimized_logits": b,
                "raw_response": qa,
                "maximum_logit_difference": delta,
                "maximum_raw_response_difference": abs(qa - qb),
                "canonical_repeat_exact": True,
            }
        )
    return {
        "schema": "chipmunk.image-imdet-authored-native/17",
        "inspection": native.inspection,
        "authored_inputs": len(rows),
        "records": rows,
        "attempted_model_forwards": native.attempted_forwards,
        "completed_model_forwards": native.completed_forwards,
        "project_media_forwards": 0,
        "detection_quality_evaluated": False,
        "logit_bound": LOGIT_BOUND,
        "response_bound": RESPONSE_BOUND,
        "implementation_sha256": digest(Path(__file__)),
        "serving_admitted": False,
        "customer_confidence_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "smoke"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "New receipt required; no overwrite")
    value = inspect() if args.mode == "inspect" else smoke()
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mode": args.mode, "project_media_forwards": 0, "serving_admitted": False}), flush=True)
