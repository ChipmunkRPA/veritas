# SPDX-License-Identifier: Apache-2.0
"""Challenge the standard small-crop footprint on new JPEG95 source groups."""

from unittest.mock import patch

from scripts import image_jpeg_offgrid_quality95_transfer as transfer

SCHEMA = "chipmunk.image-jpeg-offgrid-quality95-small-transfer/1"
MIN_SIDE = 32
HARD_SLOTS = (2, 9, 12, 13, 19, 25, 28, 30, 33, 37)


if __name__ == "__main__":
    # Use the same sharp and feathered recipes with a separate source commitment.
    with (patch.object(transfer, "SCHEMA", SCHEMA), patch.object(transfer, "MIN_SIDE", MIN_SIDE),
          patch.object(transfer, "HARD_SLOTS", HARD_SLOTS),
          patch.object(transfer, "__file__", __file__)):
        transfer.main()
