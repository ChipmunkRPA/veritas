"""Export executed preparation evidence without images, maps or checkpoint values."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def export(output):
    tiny = read("docs/evaluation-panels/image-imdet-v17-authored-native.json")
    relay = read("docs/evaluation-panels/relayformer-v18-authored-native.json")
    structural = read("docs/evaluation-panels/relayformer-v18-checkpoint.json")
    assert tiny["implementation_sha256"] == sha("backend/scripts/image_imdet_native_v17.py")
    assert relay["implementation_sha256"] == sha("backend/scripts/relayformer_native_v18.py")
    assert tiny["completed_model_forwards"] == tiny["attempted_model_forwards"] == 18
    assert relay["completed_model_forwards"] == relay["attempted_model_forwards"] == 6
    assert relay["complete_checkpoint_tensors"] == structural["tensors"] == 410
    assert all(r["canonical_repeat_exact"] for r in tiny["records"] + relay["records"])
    for record in (tiny, relay):
        assert record["project_media_forwards"] == 0 and record["serving_admitted"] is False
        assert record["detection_quality_evaluated"] is False
    profiles = {
        name: read(path)
        for name, path in {
            "tiny_graph": "output/evaluation/image-imdet-v17-inspect-profile/profile.json",
            "tiny_authored": "output/evaluation/image-imdet-v17-smoke-profile/profile.json",
            "relay_tensors": "output/evaluation/relayformer-v18-checkpoint-profile/profile.json",
            "relay_authored": "output/evaluation/relayformer-v18-authored-profile/profile.json",
        }.items()
    }
    assert all(p["child_exit_code"] == 0 and p["guard"] is None for p in profiles.values())
    files = (
        "docs/evaluation-panels/image-imdet-v17-acquisition.json",
        "docs/evaluation-panels/image-imdet-v17-graph.json",
        "docs/evaluation-panels/image-imdet-v17-authored-native.json",
        "docs/evaluation-panels/relayformer-v18-acquisition-v2.json",
        "docs/evaluation-panels/relayformer-v18-checkpoint.json",
        "docs/evaluation-panels/relayformer-v18-authored-native.json",
    )
    code = (
        "backend/scripts/prepare_image_imdet_v17.py",
        "backend/scripts/image_imdet_native_v17.py",
        "backend/scripts/image_imdet_runtime_v17.py",
        "backend/scripts/prepare_relayformer_v18.py",
        "backend/scripts/inspect_relayformer_v18.py",
        "backend/scripts/relayformer_native_v18.py",
        "backend/scripts/relayformer_runtime_v18.py",
        "backend/scripts/export_localizer_preparation_v18.py",
        "backend/tests/test_image_imdet_v17.py",
        "backend/tests/test_relayformer_v18.py",
    )
    install = read("backend/data/independent-evaluation/image-imdet-v17-runtime/install-attempt-1.json")
    retry = read("backend/data/independent-evaluation/image-imdet-v17-runtime/install-attempt-2.json")
    acquisition_failure = read("models/research/image-v18-relayformer/failed-acquisition.json")
    assert install["exit_code"] == 1 and retry["exit_code"] == 0 and acquisition_failure["exit_code"] == 1
    assert acquisition_failure["model_file_created"] is False
    voice = "backend/app/detectors/audio/calibrations/local-speech-v3.json"
    assert sha(voice) == "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
    body = {
        "schema": "chipmunk.new-localizers-preparation/18",
        "artifacts_sha256": {name: sha(name) for name in files},
        "implementation_sha256": {name: sha(name) for name in code},
        "resource_profiles": profiles,
        "retained_preparation_failures": {
            "isolated_runtime": install,
            "separate_runtime_completion": retry,
            "relay_card_format_refusal": acquisition_failure,
        },
        "actual_authored_forwards": {"tiny_whole_image_classifier": 18, "relay_localizer": 6},
        "project_media_forwards": 0,
        "detection_quality_evaluated": False,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "voice_calibration_sha256": sha(voice),
        "local_authored_tests": 32,
        "github_workflows_executed": 0,
        "outside_human_reviewers": 0,
        "source_images_or_maps_or_weights_published": False,
        "scope": "Acquisition, permission correspondence, structural inspection, source geometry and actual authored execution only. Matched real-media comparison, fresh confirmation and integration remain required.",
    }
    text = json.dumps(body, indent=2, allow_nan=False)
    assert "/Users/" not in text and '"values"' not in text
    with output.open("x") as stream:
        stream.write(text + "\n")
    print(
        json.dumps(
            {
                "published_metadata_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "actual_authored_model_forwards": 24,
                "project_media_forwards": 0,
                "serving_admitted": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)
