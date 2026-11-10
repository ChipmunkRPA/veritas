import copy

import pytest

from scripts import acquire_grid_transfer_v130 as original
from scripts import grid_transfer_context_v131 as acquisition


def publisher_record():
    return {
        "id": 3625687,
        "metadata": {"license": {"id": "cc-by-4.0"}, "access_right": "open"},
        "files": [
            {
                "key": name,
                "size": 10000,
                "checksum": "md5:" + "a" * 32,
                "links": {"self": f"https://zenodo.org/api/records/3625687/files/{name}/content"},
            }
            for name in [f"s{i}.zip" for i in range(1, 35) if i != 21] + ["jasagrid.pdf"]
        ],
    }


def test_new_namespace_retains_original_selection_and_exact_asset_identities():
    record = publisher_record()
    plan = acquisition.select(record)
    legacy = copy.deepcopy(record)
    for item in legacy["files"]:
        item["links"]["self"] = f"https://zenodo.org/api/files/example/{item['key']}"
    old = original.select(legacy)
    assert [r["publisher_name"] for r in plan["selected"]] == [r["publisher_name"] for r in old["selected"]]
    record["files"].reverse()
    assert acquisition.select(record) == plan
    assert len(plan["selected"]) == 16


@pytest.mark.parametrize("asset_index", [0, -1])
@pytest.mark.parametrize(
    "change",
    ["foreign_host", "other_record", "other_file", "credentials", "query", "fragment", "legacy", "unbounded"],
)
def test_every_archive_and_document_url_is_bound_to_its_exact_record_and_file(asset_index, change):
    record = publisher_record()
    item = record["files"][asset_index]
    url = item["links"]["self"]
    if change == "foreign_host":
        url = url.replace("zenodo.org", "example.com")
    elif change == "other_record":
        url = url.replace("3625687", "158963")
    elif change == "other_file":
        url = url.replace(item["key"], "replacement.zip")
    elif change == "credentials":
        url = url.replace("zenodo.org", "someone@zenodo.org")
    elif change == "query":
        url += "?download=1"
    elif change == "fragment":
        url += "#replacement"
    elif change == "legacy":
        url = f"https://zenodo.org/api/files/example/{item['key']}"
    else:
        item["size"] = 700 * 1024**2
    item["links"]["self"] = url
    with pytest.raises(ValueError):
        acquisition.select(record)


@pytest.mark.parametrize("problem", ["noncommercial", "closed", "duplicate", "missing"])
def test_license_and_complete_publisher_inventory_still_required(problem):
    record = publisher_record()
    if problem == "noncommercial":
        record["metadata"]["license"]["id"] = "cc-by-nc-nd-4.0"
    elif problem == "closed":
        record["metadata"]["access_right"] = "restricted"
    elif problem == "duplicate":
        record["files"][0] = copy.deepcopy(record["files"][1])
    else:
        record["files"].pop(0)
    with pytest.raises(ValueError):
        acquisition.select(record)


def test_context_routes_only_data_stage_and_preserves_old_code_objects(monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    scope = acquisition.context()
    scope["routed_popen"](["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "out", "0"])
    assert calls[0][3] == "scripts.grid_transfer_context_v131"
    with pytest.raises(ValueError):
        scope["routed_popen"](["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
    assert scope["acquire"].__code__ is original.acquire.__code__
    assert scope["run"].__code__ is original.run.__code__
    assert scope["TRIAL"] != original.TRIAL and scope["CACHE"] != original.CACHE
