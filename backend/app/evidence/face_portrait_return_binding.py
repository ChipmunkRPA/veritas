"""Native portrait and square scopes for the brief face-appearance comparison."""

from app.evidence.face_local_return_binding import PORTRAIT

METHOD = "face-short-return/24"
HEVC_METHOD = "face-short-return/25"
SQUARE_METHOD = "face-short-return/26"
METHODS = frozenset({METHOD, HEVC_METHOD, SQUARE_METHOD})
HEVC_DIMENSIONS = (360, 640)
HEVC_ENCODING = ("hevc", "yuv420p", "hvc1", "Main")
SQUARE_DIMENSIONS = (480, 480)
SQUARE_ENCODING = ("h264", "yuv420p", "avc1")
POINTS = {
    (360, 640): float.fromhex("0x1.81f19cb4084c6p-5"),
    (720, 1280): float.fromhex("-0x1.e04ca08bfce55p-5"),
}
SQUARE_POINT = POINTS[(360, 640)]


def _upright_grid(table):
    dimensions = table.width, table.height
    return (dimensions == (table.coded_width, table.coded_height)
            and not (table.rotation or table.truncated or table.interpolated)
            and len(table) == PORTRAIT.frame_count
            and all((pts - table.pts[0]) * table.tb == index * PORTRAIT.step for index, pts in enumerate(table.pts)))


def supported(table, encoding):
    return (tuple(encoding[:2]) == ("h264", "yuv420p") and (table.width, table.height) in POINTS
            and _upright_grid(table))


def hevc_supported(table, encoding):
    return (tuple(encoding) == HEVC_ENCODING and (table.width, table.height) == HEVC_DIMENSIONS
            and _upright_grid(table) and table.pts[0] == 0)


def square_supported(table, encoding):
    return (tuple(encoding) == SQUARE_ENCODING and (table.width, table.height) == SQUARE_DIMENSIONS
            and _upright_grid(table) and table.pts[0] == 0)
