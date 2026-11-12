"""woof-nam products: the accumulated snowfall water record (WEASD, accumulation; awphys row 334) has NAM's 12 h
window and is rebuilt from the model's 3 h snowfall bucket.

NCEP's archived awphys files carry WEASD 0.1.13 with statistic accumulation in the 12 h window only (0-1 ... 0-12 h,
then 12-13 ... 12-24 h, the same window as APCP's first record). The model keeps snowfall in ACSNOW, a 3 h bucket
(kg m-2) emptied after the history write at every multiple of 3 h (drv-bucket-reset, PRECIPBUCKET = 3 h). So the window's
total is ACSNOW at each 3 h mark inside the window plus the current file's ACSNOW. This test builds small synthetic
history files from a known hourly snowfall and checks the carrier against the true window totals, the initial time,
a missing bucket file (left out, never filled) and a history without ACSNOW (left out). CPU only."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io import products as P  # noqa: E402
from namgpu.products import spec  # noqa: E402

JM, IM = 7, 9
START = "2024-11-05_12:00:00"


def _rate(h):
    """Snowfall water in hour h (h-1 to h), kg m-2, different at every point and hour."""
    j, i = np.mgrid[0:JM, 0:IM]
    return (0.1 * (1 + (h * 7 + j + 2 * i) % 5)).astype(np.float64)


def _acsnow(h):
    """The model's 3 h bucket at hour h: snowfall since the last multiple of 3 h before h (emptied after each write
    at a multiple of 3 h, so the 3 h file still holds hours 1-3)."""
    if h == 0:
        return np.zeros((JM, IM))
    b3 = ((h - 1) // 3) * 3
    return sum(_rate(k) for k in range(b3 + 1, h + 1))


def _total(h):
    return sum((_rate(k) for k in range(1, h + 1)), np.zeros((JM, IM)))


def _histories(run: Path, hours, with_snow=True):
    import netCDF4
    run.mkdir(parents=True, exist_ok=True)
    for h in hours:
        p = run / f"wrfout_d01_2024-11-{5 + (12 + h) // 24:02d}_{(12 + h) % 24:02d}:00:00"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("Time", None)
            d.createDimension("south_north", JM)
            d.createDimension("west_east", IM)
            d.START_DATE = START
            d.createVariable("XTIME", "f4", ("Time",))[0] = 60.0 * h
            z = np.zeros((JM, IM), np.float32)
            for name, a in (("RAINNC", z + h), ("RAINC", z)):
                d.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = a
            if with_snow:
                d.createVariable("ACSNOW", "f4", ("Time", "south_north", "west_east"))[0] = _acsnow(h).astype(np.float32)
    return run


def _tables():
    jj, ii = np.mgrid[0:JM - 1, 0:IM - 1]
    z = np.zeros((JM - 1, IM - 1), np.float32)
    return {"j0": jj, "i0": ii, "wx": z, "wy": z}


def _buckets(run: Path, h):
    import netCDF4
    t = _tables()
    f = sorted(run.glob("wrfout_d01_*"))[0].with_name(
        f"wrfout_d01_2024-11-{5 + (12 + h) // 24:02d}_{(12 + h) % 24:02d}:00:00")
    with netCDF4.Dataset(f) as s:
        def get(n):
            return P._regrid(t, np.asarray(s[n][0], dtype=np.float32))
        return dict(P._buckets(f, s, get, t, P.PSETS.AWPHYS))


def test_snowfall_water_has_nams_12h_window(tmp_path):
    run = _histories(tmp_path / "run", range(0, 17))
    rec0 = _buckets(run, 0)
    assert "WEASD_S12" in rec0 and np.all(rec0["WEASD_S12"] == 0.0)
    for h in range(1, 17):
        b12 = ((h - 1) // 12) * 12
        want = (_total(h) - _total(b12))[:JM - 1, :IM - 1]
        got = _buckets(run, h).get("WEASD_S12")
        assert got is not None, h
        assert np.allclose(got, want, rtol=1e-5, atol=1e-5), (h, float(np.abs(got - want).max()))
    # the window restarts at 12 h: at +13 h it is the 12-13 h snowfall only
    assert np.allclose(_buckets(run, 13)["WEASD_S12"], _rate(13)[:JM - 1, :IM - 1], atol=1e-5)


def test_missing_bucket_file_leaves_the_record_out(tmp_path):
    run = _histories(tmp_path / "run", [h for h in range(0, 9) if h != 6])
    assert "WEASD_S12" in _buckets(run, 5)          # needs the 3 h file only
    assert "WEASD_S12" not in _buckets(run, 8)      # needs the 3 h and 6 h files: never filled
    assert "APCP_S12" in _buckets(run, 8)


def test_history_without_acsnow_leaves_the_record_out(tmp_path):
    run = _histories(tmp_path / "run", range(0, 5), with_snow=False)
    assert "WEASD_S12" not in _buckets(run, 0)
    assert "WEASD_S12" not in _buckets(run, 4)


def test_table_row_334_is_the_12h_carrier():
    by = {r["rec"]: r for r in spec.read()}
    r = by["334"]
    assert (r["name"], r["dcp"], r["level"], r["stat"]) == ("WEASD", "0.1.13", "1:0", "acc")
    assert r["source"] == "carrier:WEASD_S12", r["source"]
    assert r["window"] == "since:12h", r["window"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))


def test_every_acsnow_writer_and_reader_uses_kg_m2():
    """ACSNOW is kg m-2 everywhere: Noah adds FFROZP PRCP DT (mm); the RUC driver and the column suites used to add
    metres (x 1e-3) while the nest WEASD carrier multiplied by 1000, so a Noah nest file read 1000 times the snowfall."""
    root = Path(__file__).resolve().parents[1] / "namgpu"
    from namgpu.products import hiresf_carriers as HC
    a = np.full((1, 3, 4), 2.5, np.float32)

    class S:
        variables = {"ACSNOW": None}

        def __getitem__(self, k):
            return a
    fn, _interp = HC.CARRIERS["NAMH_WEASD_ACC"]
    out = fn(S(), {})
    assert out is not None and np.allclose(out, 2.5), out
    ruc = (root / "physics/ruc/phys_ruc_driver.py").read_text()
    col = (root / "physics/suite_column.py").read_text()
    assert "rain * sr * f32(1.0e-3)" not in ruc
    assert 'st["ACSNOW"][...] += (snow_mm * np.float32(1.0e-3))' not in col
