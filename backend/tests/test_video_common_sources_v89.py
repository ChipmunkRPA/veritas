"""Source-history, alias and role checks; no network, media or model calls."""
import csv

import pytest

from scripts import video_common_sources_v89 as source


@pytest.mark.parametrize("url", ["https://youtu.be/abcdefghijk?t=3", "http://www.youtube.com/watch?v=abcdefghijk",
    "https://m.youtube.com/shorts/abcdefghijk", "https://youtube.com/embed/abcdefghijk", "https://youtube.com/live/abcdefghijk"])
def test_youtube_aliases_cannot_cross_history_exclusion(url):
    assert source.source_key(url) == "youtube:abcdefghijk"


def test_conservative_photo_url_aliases_and_malformed_sources():
    assert source.source_key("https://www.example.org/photo.jpg?width=400") == source.source_key("http://example.org/photo.jpg?width=100")
    with pytest.raises(ValueError, match="missing or malformed"):
        source.source_key("javascript:alert(1)")
    with pytest.raises(ValueError, match="Unrecognized YouTube"):
        source.source_key("https://youtube.com/watch?v=short")


def put_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(base, destination):
    permissions = destination / "permissions"
    permissions.mkdir(parents=True)
    originals = [{"id": letter * 11, "filename": letter + ".mp4", "original_link": "https://youtu.be/" + letter * 11,
                  "tags": "adult"} for letter in "abcde"]
    put_table(permissions / "rtfs-videos.csv", originals)
    photos = [{"filename": "old.jpg", "tags": "adult"}]
    for family in ("inswapper", "uniface"):
        rows = []
        for letter in "bcde":
            donor = family + letter + ".jpg"
            photos.append({"filename": donor, "tags": "adult"})
            rows.append({"video": letter + ".mp4", "photo": donor, "generator": family,
                "filename": family + letter + ".mp4", "success": "True", "additional_features": "",
                "photo_tags": "adult", "video_tags": "adult"})
        put_table(permissions / ("rtfs-" + family + ".csv"), rows)
    put_table(permissions / "rtfs-photos.csv", photos)
    put_table(permissions / "crema-SentenceFilenames.csv", [{"Filename": f"{actor}_DFA_NEU_XX"} for actor in range(1001, 1092)])
    put_table(permissions / "crema-VideoDemographics.csv", [{"ActorID": str(actor)} for actor in range(1001, 1092)])
    files = {path.name: {"sha256": source.sha(path)} for path in permissions.iterdir()}
    source.write(permissions / "receipt.json", {"status": "complete", "files": files,
        "rtfs_revision": source.RTFS_REV, "crema_revision": source.CREMA_REV})
    for name in source.PARENTS:
        folder = base / name
        folder.mkdir(parents=True)
        source.write(folder / "manifest.json", {"revision": source.RTFS_REV,
            "metadata_sha256": {key: source.sha(permissions / ("rtfs-" + key)) for key in ("videos.csv", "inswapper.csv", "uniface.csv")},
            "items": [{"group": "RTFS:" + "a" * 11, "source_video_url": "https://www.youtube.com/watch?v=" + "a" * 11,
                       "donor_filename": "old.jpg"}]})
    history = base / "lipsync-old/inputs"
    history.mkdir(parents=True)
    source.write(history / "manifest.json", {"items": [{"group": "CREMA-D:1001"}]})


def test_actual_plan_keeps_paired_variants_in_one_role_and_source_identities_private(tmp_path, capsys):
    base, destination = tmp_path / "history", tmp_path / "new"
    fixture(base, destination)
    source.plan(base, destination, rtfs_per_role=1, crema_per_role=1)
    private, public = source.read(destination / "plan.json"), source.read(destination / "summary.json")
    assert public["source_groups"] == 8 and public["source_assets_planned"] == 16
    assert public["portrait_source_urls_supplied"] == 0
    assert public["role_assignment_provisional"] is True
    assert public["new_media_downloads"] == public["neural_calls"] == 0
    for role in source.ROLES:
        assert public["roles"][role] == {"paired_face_swap": 1, "recorded_audiovisual_speech": 1}
    assert all(row["group"] not in {"CREMA-D:1001", "RTFS:" + "a" * 11} for row in private["groups"])
    assert all(len(row["assets"]) == 3 for row in private["groups"] if row["source_family"] == "paired_face_swap")
    assert all(row["assets"][0]["source_path"].endswith(".flv") for row in private["groups"] if row["source_family"] == "recorded_audiovisual_speech")
    assert "groups" not in public and "CREMA-D:1001" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="Never overwrite"):
        source.plan(base, destination, rtfs_per_role=1, crema_per_role=1)


def test_tampered_metadata_is_not_accepted_as_a_source_plan(tmp_path):
    base, destination = tmp_path / "history", tmp_path / "new"
    fixture(base, destination)
    with (destination / "permissions/rtfs-photos.csv").open("a") as stream:
        stream.write("another.jpg,adult\n")
    with pytest.raises(ValueError, match="Metadata bytes changed"):
        source.plan(base, destination, rtfs_per_role=1, crema_per_role=1)
    assert not (destination / "plan.json").exists()


def test_url_linked_donor_alias_is_excluded_even_under_a_different_filename():
    old = [{"group": "RTFS:old", "source_video_url": "https://youtu.be/aaaaaaaaaaa", "donor_filename": "old.jpg"}]
    photos = [{"filename": name, "link": url} for name, url in (("old.jpg", "https://example.org/p.jpg?a=1"),
        ("alias.jpg", "https://example.org/p.jpg?a=2"), ("other.jpg", "https://example.org/other.jpg"))]
    originals = [{"id": "bbbbbbbbbbb", "filename": "b.mp4", "original_link": "https://youtu.be/bbbbbbbbbbb", "tags": "adult"}]
    generated = {family: [{"video": "b.mp4", "photo": donor, "filename": family + ".mp4", "generator": family,
        "success": "True", "additional_features": "", "photo_tags": "adult", "video_tags": "adult"}]
        for family, donor in (("inswapper", "alias.jpg"), ("uniface", "other.jpg"))}
    assert source.paired_targets(originals, generated, photos, old) == []


def test_source_counts_and_path_traversal_are_rejected_before_plan_write(tmp_path):
    base, destination = tmp_path / "history", tmp_path / "new"
    fixture(base, destination)
    with pytest.raises(ValueError, match="Positive explicit"):
        source.plan(base, destination, rtfs_per_role=True, crema_per_role=1)
    with pytest.raises(ValueError, match="Unsafe source filename"):
        source.filename("../../outside.mp4", ".mp4")
    with pytest.raises(ValueError, match="Insufficient"):
        source.plan(base, destination, rtfs_per_role=2, crema_per_role=1)
    assert not (destination / "plan.json").exists()
