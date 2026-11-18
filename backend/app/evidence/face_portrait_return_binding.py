"""Native source scope for the temporary portrait face-appearance comparison."""

from app.evidence.face_local_return_binding import PORTRAIT

METHOD = "face-short-return/24"
POINTS = {
    (360, 640): float.fromhex("0x1.81f19cb4084c6p-5"),
    (720, 1280): float.fromhex("-0x1.e04ca08bfce55p-5"),
}


def supported(table, encoding):
    dimensions = table.width, table.height
    return (tuple(encoding[:2]) == ("h264", "yuv420p") and dimensions in POINTS
            and dimensions == (table.coded_width, table.coded_height)
            and not (table.rotation or table.truncated or table.interpolated)
            and len(table) == PORTRAIT.frame_count
            and all((pts - table.pts[0]) * table.tb == index * PORTRAIT.step for index, pts in enumerate(table.pts)))
