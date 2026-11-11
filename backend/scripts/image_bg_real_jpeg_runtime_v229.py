"""Monitor one BG-REAL JPEG comparison phase with the v189 worker limits."""

from pathlib import Path

from scripts import image_jpeg_diagnostic_runtime_v189 as original
from scripts.image_bg_real_jpeg_v229 import parse_args, require, write
from scripts.video_appearance_corrected_v112 import scoped

MODULE = "scripts.image_bg_real_jpeg_v229"
SCHEMA = "chipmunk.image-bg-real-jpeg-profile/229"
LIMITS = original.LIMITS
IDENTITY = {"MODULE": MODULE, "__file__": __file__}
original_assert = scoped(original.assert_profile, IDENTITY)


def assert_profile(folder, command, arguments):
    profile = original_assert(folder, command, arguments)
    require(profile["schema"] == SCHEMA, "Exact v229 resource profile required")
    return profile


def profile_write(path, data):
    require(path.name == "profile.json", "Only one resource profile per phase")
    write(path, {**data, "schema": SCHEMA})


run = scoped(original.run, {**IDENTITY, "assert_profile": assert_profile, "write": profile_write})


if __name__ == "__main__":
    command, arguments = parse_args(monitor=True)
    run(command, [str(arg.resolve()) if isinstance(arg, Path) else str(arg) for arg in arguments])
