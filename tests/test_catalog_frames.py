"""Catalog frames (namgpu/outputs.py, NAMGPU_CATALOG_EVERY): which off-hour times carry the run's own selection,
uncompressed, and that a raw write holds exactly the values a compressed one does."""
from datetime import datetime

import numpy as np
import pytest

from namgpu import outputs


def _times(s):
    return [(t["tag"], t["light"], t["catalog"], t["grib"]) for t in s.doc(datetime(2026, 11, 15, 18))["times"]]


def test_no_env_no_catalog_frames():
    s = outputs.Schedule(25.0, 1.0, write="15m", light_fields=outputs.MAP_FIELDS)
    assert [t[1:3] for t in _times(s)] == [(False, False), (True, False), (True, False), (True, False), (False, False)]
    assert "catalog" not in s.counts()


def test_every_quarter_hour_is_a_catalog_frame_at_15m():
    s = outputs.Schedule(25.0, 2.0, write="15m", light_fields=outputs.MAP_FIELDS, catalog_h=0.25)
    rows = _times(s)
    assert all(not light for _, light, _, _ in rows)
    assert [tag for tag, _, cat, _ in rows if cat] == ["f000m15", "f000m30", "f000m45", "f001m15", "f001m30", "f001m45"]
    assert all(not grib for _, _, cat, grib in rows if cat) and s.counts()["catalog"] == 6
    assert s.catalog(36) and not s.light(36) and not s.catalog(144)


def test_five_minute_run_keeps_light_frames_between_catalog_frames():
    s = outputs.Schedule(25.0, 1.0, write="5m", light_fields=outputs.MAP_FIELDS, catalog_h=0.25)
    cats = [tag for tag, _, cat, _ in _times(s) if cat]
    lights = [tag for tag, light, _, _ in _times(s) if light]
    assert cats == ["f000m15", "f000m30", "f000m45"] and len(lights) == 8


def test_named_fields_have_no_light_or_catalog_frames():
    s = outputs.Schedule(25.0, 1.0, write="15m", light_fields=None, catalog_h=0.25)
    assert all(not light and not cat for _, light, cat, _ in _times(s))


def test_env_parse():
    assert outputs.catalog_every_h({}) is None
    assert outputs.catalog_every_h({"NAMGPU_CATALOG_EVERY": "15m"}) == 0.25
    assert outputs.catalog_every_h({"NAMGPU_CATALOG_EVERY": "off"}) is None
    with pytest.raises(ValueError):
        outputs.catalog_every_h({"NAMGPU_CATALOG_EVERY": "soon"})


def test_raw_write_holds_the_compressed_values(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from namgpu.io.history import write_payload
    rng = np.random.default_rng(1)
    t = rng.normal(280, 10, (3, 4, 5)).astype(np.float32)
    ps = rng.normal(1e5, 500, (4, 5)).astype(np.float32)
    base = {"valid": datetime(2026, 11, 15, 18, 15), "start": datetime(2026, 11, 15, 18), "xtime_min": 15.0,
            "xlat": np.zeros((4, 5), np.float32), "xlon": np.ones((4, 5), np.float32),
            "dims": {"west_east": 5, "south_north": 4, "bottom_top": 3}, "cen_lat": 1.0, "cen_lon": 2.0,
            "pole_lat": 3.0, "pole_lon": 4.0, "dx": 5.0, "dy": 12000.0,
            "fields": [("T", t, ("bottom_top", "south_north", "west_east")), ("PSFC", ps, ("south_north", "west_east"))]}
    a, b = write_payload(tmp_path / "zip", base), write_payload(tmp_path / "raw", {**base, "zlib": False})
    with netCDF4.Dataset(a) as x, netCDF4.Dataset(b) as y:
        x.set_auto_chartostring(False)
        y.set_auto_chartostring(False)
        assert x["T"].filters()["zlib"] and not y["T"].filters()["zlib"] and not y["XLAT"].filters()["zlib"]
        for n in x.variables:
            assert np.array_equal(np.asarray(x[n][:]), np.asarray(y[n][:])), n
        assert {k: x.getncattr(k) for k in x.ncattrs()} == {k: y.getncattr(k) for k in y.ncattrs()}
