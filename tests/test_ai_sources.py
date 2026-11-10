"""CPU tests of the AI-forecast init rows (aifs, aigfs): table rows, 6-hourly boundary plan, column derivations."""
from datetime import datetime
from pathlib import Path

import numpy as np

from namgpu.domains import bc_hours
from namgpu.init import fetch_case, source_records as srec, sources


def test_rows_are_table_rows():
    for name, start, rec in (("aifs", "ifs-0p25", "aifs"), ("aigfs", "gfs", "aigfs")):
        row = sources.SOURCES[name]
        assert row["start_from"] == start and row["start_from"] in sources.SOURCES
        assert row["boundary"]["interval_h"] == 6 and row["boundary"]["records"] == rec
        assert srec.boundary_table(row) is srec.TABLES[rec]
        assert row["licence"] and row["attribution"]
    # every other row reads its boundaries with its own table, as before
    for name, row in sources.SOURCES.items():
        if "boundary" in row and not row["boundary"].get("records"):
            assert srec.boundary_table(row) is srec.table(row)


def test_aifs_table_is_ifs_with_geopotential_levels():
    a, i = srec.AIFS, srec.IFS
    assert a["pl"]["HGT"] == (0, 3, 4) and i["pl"]["HGT"] == (0, 3, 5)
    assert list(a["pl_derive"]) == ["HGT", "RH"]               # z / g0 before RH from q
    assert a["sfc_columns"] == i["sfc_columns"] and a["sfc"] == i["sfc"]


def test_bc_hours_interval():
    assert [bc_hours(h) for h in (1, 6, 7, 12, 24)] == [6, 6, 9, 12, 24]      # unchanged for 3 h
    assert [bc_hours(h, 6) for h in (1, 6, 7, 12, 24)] == [6, 6, 12, 12, 24]


def test_plan_aifs_12h(tmp_path):
    jobs = fetch_case.plan_source("aifs", datetime(2026, 1, 23, 12), 12, tmp_path / "raw")
    dests = sorted(str(next(x for x in j if isinstance(x, Path)).relative_to(tmp_path)) for j in jobs)
    assert "raw/aifs/aifs.t12z.0p25.f000.grib2" in dests
    assert "raw/aifs/aifs.t12z.0p25.f006.grib2" in dests and "raw/aifs/aifs.t12z.0p25.f012.grib2" in dests
    assert not any("f003" in d or "f009" in d for d in dests if "aifs/" in d)
    assert "raw/ifs/ifs.t12z.0p25.f000.grib2" in dests                        # the start: the IFS analysis
    assert any("icec" in d for d in dests)                                     # its GFS sea-ice donor


def test_plan_aigfs_whole_files(tmp_path):
    jobs = fetch_case.plan_source("aigfs", datetime(2026, 11, 8, 0), 12, tmp_path / "raw")
    leads = sorted(j[3] for j in jobs if j[0] is fetch_case._whole_lead)
    assert leads == [0, 6, 12]
    assert any(j[0] is not fetch_case._whole_lead for j in jobs)              # the gfs analysis start file


def _column(zs, t0=288.0, lapse=0.0065):
    """A standard-atmosphere column on 13 levels: heights and temperatures that the hydrostatic formula inverts."""
    plev = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 5000.0])
    p0 = 101325.0
    g_r = srec.G0 / srec.RD_
    z = (t0 / lapse) * (1.0 - (plev / p0) ** (lapse / g_r))
    t = t0 - lapse * z
    ny, nx = zs.shape
    hgt = np.broadcast_to(z[:, None, None], (13, ny, nx)).astype(np.float32).copy()
    tmp = np.broadcast_to(t[:, None, None], (13, ny, nx)).astype(np.float32).copy()
    return plev, hgt, tmp, p0, t0, lapse, g_r


def test_psfc_from_isobaric_standard_atmosphere():
    zs = np.array([[-50.0, 0.0, 300.0], [1500.0, 3000.0, 4500.0]], np.float32)
    plev, hgt, tmp, p0, t0, lapse, g_r = _column(zs)
    ps = srec.psfc_from_isobaric(hgt, tmp, list(plev), zs, xp=np)
    want = p0 * (1.0 - lapse * zs / t0) ** (g_r / lapse)
    assert np.allclose(ps, want, rtol=2e-4), (ps, want)


def test_psfc_skips_undefined_low_levels():
    zs = np.array([[1500.0]], np.float32)
    plev, hgt, tmp, p0, t0, lapse, g_r = _column(zs)
    hgt[:2] = np.nan                                    # 1000 and 925 hPa undefined (below ground in the source)
    ps = srec.psfc_from_isobaric(hgt, tmp, list(plev), zs, xp=np)
    assert np.allclose(ps, p0 * (1.0 - lapse * zs / t0) ** (g_r / lapse), rtol=2e-4)


def test_isobaric_at_psfc_log_linear():
    plev = [100000.0, 85000.0, 70000.0]
    f = np.array([10.0, 20.0, 40.0], np.float32)[:, None, None] * np.ones((3, 1, 3), np.float32)
    ps = np.array([[100000.0, np.sqrt(100000.0 * 85000.0), 102000.0]], np.float32)
    out = srec.isobaric_at_psfc(f, plev, ps, xp=np)
    assert np.allclose(out[0], [10.0, 15.0, 10.0], atol=1e-4)


def test_column_derive_aigfs():
    zs = np.array([[200.0, 800.0]], np.float32)
    plev, hgt, tmp, *_ = _column(zs)
    order = ["HGT", "TMP", "RH"]
    rh = np.linspace(90.0, 10.0, 13, dtype=np.float32)[:, None, None] * np.ones((13, 1, 2), np.float32)
    pl = np.stack([hgt, tmp, rh])
    sfc = {"HGT": zs}
    srec.derive_columns_sfc(srec.AIGFS, pl, list(plev), order, sfc)
    assert set(sfc) == {"HGT", "PRES", "RH2"}
    assert np.all((sfc["PRES"] > 90000) & (sfc["PRES"] < 101325)) and np.all(sfc["RH2"] > 80.0)


def test_aifs_start_takes_gfs_above_50hpa(tmp_path):
    from namgpu.init import source_grids as sgr
    row = sources.SOURCES["aifs"]
    assert row["start_fallback"]["source"] == "gfs" and row["start_fallback"]["top_blend_pa"] > 0
    assert "fallback" not in sources.SOURCES["ifs-0p25"]                  # the ifs rows themselves are unchanged
    g = sgr.GlobalLatLon(1440, 721)
    fy, fx = g.place(np.zeros((3, 4)), np.zeros((3, 4)))
    assert np.all(g.edge_weight(fy, fx, 8) == 1.0)
    jobs = fetch_case.plan_source("aifs", datetime(2026, 1, 23, 12), 12, tmp_path / "raw")
    dests = [str(next(x for x in j if isinstance(x, Path)).relative_to(tmp_path)) for j in jobs]
    assert "raw/gfs/gfs.t12z.pgrb2.0p25.f000" in dests
