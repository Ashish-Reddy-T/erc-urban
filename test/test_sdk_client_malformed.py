"""Malformed /data and /v2/screenshot payloads must degrade to the existing
fallbacks (lat/lon NaN, yaw/hw_ts/yaw_rate None, frame None), never raise and
never leak NaN/inf where a fallback exists. Mocked HTTP, no live SDK."""
from __future__ import annotations
import base64
import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from erc_urban.sdk_client import RoverClient, Telemetry

MOD = "erc_urban.sdk_client"
NAN = float("nan")
INF = float("inf")


def _png_b64(color=(255, 0, 0), size=(4, 4)) -> str:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resp(body):
    r = MagicMock(status_code=200)
    r.raise_for_status = lambda: None
    r.json = lambda: body
    return r


def _get_data(body) -> Telemetry:
    with patch(f"{MOD}.requests.get", return_value=_resp(body)):
        return RoverClient("http://localhost:8000", timeout=1.0).get_data()


def _get_frames(body):
    with patch(f"{MOD}.requests.get", return_value=_resp(body)):
        return RoverClient("http://localhost:8000", timeout=1.0).get_frames()


def _check_contract(tel: Telemetry) -> None:
    assert isinstance(tel, Telemetry)
    for v in (tel.lat, tel.lon):
        assert isinstance(v, float)
        assert math.isnan(v) or math.isfinite(v)
    for v in (tel.yaw, tel.hw_ts, tel.age_s, tel.yaw_rate):
        assert v is None or (isinstance(v, float) and math.isfinite(v))
    assert tel.wheels_turning in (None, True, False)
    assert isinstance(tel.raw, dict)


GOOD = {
    "latitude": 40.0,
    "longitude": -74.0,
    "orientation": 90,
    "gyros": [[0.1, 0.2, 5.0, 1_700_000_000.0]],
    "accels": [[0.0, 0.0, 9.8, 1_700_000_000.0]],
    "rpms": [[0, 0, 0, 0, 1_700_000_000.0]],
}

BAD_SCALARS = [None, "abc", "", NAN, INF, -INF, "nan", "inf", [], {}, [1.0], {"v": 1}, True]
BAD_SERIES = [None, "abc", [], [[]], [None], [[None]], ["abc"], [[NAN, NAN, NAN, NAN]],
              [[INF, INF, INF, INF]], [["a", "b", "c", "d"]], {"x": 1}, [{"x": 1}],
              [[1.0, 2.0]], [[1.0, 2.0, None, "abc"]], 5, "[[1,2,3,4]]"]
BAD_BODIES = [None, [], [GOOD], "abc", "", 0, 1.5, True]


@pytest.mark.parametrize("body", BAD_BODIES, ids=repr)
def test_non_dict_body_is_empty_telemetry(body):
    tel = _get_data(body)
    _check_contract(tel)
    assert math.isnan(tel.lat) and math.isnan(tel.lon)
    assert tel.yaw is None
    assert tel.raw == {}


@pytest.mark.parametrize("key", ["latitude", "longitude"])
@pytest.mark.parametrize("val", BAD_SCALARS, ids=repr)
def test_bad_gps_falls_back_to_nan(key, val):
    body = dict(GOOD, **{key: val})
    tel = _get_data(body)
    _check_contract(tel)
    assert math.isnan(getattr(tel, key[:3]))
    other = "lon" if key == "latitude" else "lat"
    assert math.isfinite(getattr(tel, other))
    assert tel.yaw == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("val", BAD_SCALARS, ids=repr)
def test_bad_orientation_falls_back_to_none(val):
    tel = _get_data(dict(GOOD, orientation=val))
    _check_contract(tel)
    assert tel.yaw is None
    assert tel.lat == pytest.approx(40.0)


@pytest.mark.parametrize("key", ["gyros", "accels", "rpms"])
@pytest.mark.parametrize("val", BAD_SERIES, ids=repr)
def test_bad_imu_series_never_raises(key, val):
    tel = _get_data(dict(GOOD, **{key: val}))
    _check_contract(tel)
    assert tel.lat == pytest.approx(40.0)
    assert tel.yaw == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("val", BAD_SERIES, ids=repr)
def test_all_imu_series_bad(val):
    tel = _get_data(dict(GOOD, gyros=val, accels=val, rpms=val))
    _check_contract(tel)


@pytest.mark.parametrize("ts", [NAN, INF, -INF, "nan"], ids=repr)
def test_non_finite_imu_timestamp_is_none(ts):
    tel = _get_data(dict(GOOD, gyros=[[0.1, 0.2, 5.0, ts]], accels=[[0.0, 0.0, 9.8, ts]]))
    _check_contract(tel)
    assert tel.hw_ts is None and tel.age_s is None
    assert tel.yaw_rate == pytest.approx(5.0)


@pytest.mark.parametrize("gz", [NAN, INF, "nan", None, "abc"], ids=repr)
def test_bad_yaw_rate_is_none(gz):
    tel = _get_data(dict(GOOD, gyros=[[0.1, 0.2, gz, 1_700_000_000.0]]))
    _check_contract(tel)
    assert tel.yaw_rate is None
    assert tel.hw_ts == pytest.approx(1_700_000_000.0)


@pytest.mark.parametrize("key", list(GOOD))
def test_nested_null_each_key(key):
    tel = _get_data(dict(GOOD, **{key: None}))
    _check_contract(tel)


def test_all_keys_null():
    tel = _get_data({k: None for k in GOOD})
    _check_contract(tel)
    assert math.isnan(tel.lat) and math.isnan(tel.lon) and tel.yaw is None


def test_well_formed_unchanged():
    tel = _get_data(dict(GOOD, latitude="40.5", orientation="180"))
    assert tel.lat == pytest.approx(40.5)
    assert tel.lon == pytest.approx(-74.0)
    assert tel.yaw == pytest.approx(math.pi)
    assert tel.hw_ts == pytest.approx(1_700_000_000.0)
    assert tel.yaw_rate == pytest.approx(5.0)
    assert tel.wheels_turning is False
    tel = _get_data(dict(GOOD, rpms=[[0, 12.0, 0, 0, 1.0]]))
    assert tel.wheels_turning is True


def test_json_decode_error_is_empty_telemetry():
    r = MagicMock(status_code=200)
    r.raise_for_status = lambda: None
    r.json = MagicMock(side_effect=ValueError("no json"))
    with patch(f"{MOD}.requests.get", return_value=r):
        tel = RoverClient("http://localhost:8000", timeout=1.0).get_data()
    _check_contract(tel)
    assert math.isnan(tel.lat) and tel.raw == {}


# ---- frames ----------------------------------------------------------------

BAD_FRAMES = [
    None,
    "",
    "!!!not base64!!!",
    "abc",
    "data:image/jpeg;base64,",
    "data:image/png;base64,@@@",
    base64.b64encode(b"").decode(),
    base64.b64encode(b"not an image").decode(),
    _png_b64()[:24],
    123,
    1.5,
    [],
    ["abc"],
    {},
    {"a": 1},
    True,
]


@pytest.mark.parametrize("b64", BAD_FRAMES, ids=repr)
def test_decode_bad_frame_is_none(b64):
    assert RoverClient._decode(b64) is None


def test_decode_good_frame_and_data_uri():
    a = RoverClient._decode(_png_b64((0, 255, 0)))
    b = RoverClient._decode("data:image/png;base64," + _png_b64((0, 255, 0)))
    assert a is not None and a.shape == (4, 4, 3) and a.dtype == np.uint8
    assert b is not None and np.array_equal(a, b)


@pytest.mark.parametrize("bad", BAD_FRAMES, ids=repr)
def test_bad_rear_keeps_good_front(bad):
    front, rear = _get_frames({"front_frame": _png_b64((0, 255, 0)), "rear_frame": bad})
    assert front is not None and front.shape == (4, 4, 3)
    assert rear is None


@pytest.mark.parametrize("body", BAD_BODIES + [{}], ids=repr)
def test_frames_non_dict_body_is_none_pair(body):
    assert _get_frames(body) == (None, None)
