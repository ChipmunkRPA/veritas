from itertools import chain

import pytest

from app.evidence.h264_quantization import QuantizationUnavailable, parse_trace


def _number(name, value):
    return f"8   {name}        010   = {value}"


def _pps(identifier, initial):
    return ["Picture Parameter Set", _number("pic_parameter_set_id", identifier),
            _number("pic_init_qp_minus26", initial)]


def _picture(pts, kind, identifier, delta):
    return [f"Packet: 24 bytes, pts {pts}, dts 0, duration 1000.", "Slice Header",
            _number("first_mb_in_slice", 0), _number("slice_type", kind),
            _number("pic_parameter_set_id", identifier), _number("slice_qp_delta", delta)]


def _trace(*elements):
    return [f"[trace_headers @ 0xABC] {line}" for line in chain.from_iterable(elements)]


def test_h264_quantization_uses_each_picture_parameter_set_and_presentation_order():
    traces = _trace(_pps(0, -3), _pps(1, 4), _picture(0, 7, 0, -9), _picture(3000, 5, 1, 3),
                    _picture(1000, 6, 0, -1), _picture(2000, 6, 0, 2))
    result = parse_trace(traces, (0, 1000, 2000, 3000))
    assert result["frames"] == [[0, "I", 14], [1000, "B", 22], [2000, "B", 25], [3000, "P", 33]]
    assert result["inter_picture_median"] == 25


@pytest.mark.parametrize(("extra", "ticks"), [
    ([], (0, 1500)),  # A packet clock that differs from the decoded picture clock.
    (_picture(1000, 5, 0, -1), (0, 1000)),  # A repeated picture packet.
    (["Sequence Parameter Set", _number("bit_depth_luma_minus8", 2)], (0, 1000)),
])
def test_h264_quantization_withholds_incomplete_or_unsupported_traces(extra, ticks):
    source = _trace(_pps(0, -3), _picture(0, 7, 0, -9), _picture(1000, 5, 0, -1), extra)
    with pytest.raises(QuantizationUnavailable):
        parse_trace(source, ticks)
