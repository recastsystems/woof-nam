"""CPU checks of the HRRR start source row (no GPU, no network).

Run: python tests/test_hrrr_source.py   (or pytest tests/test_hrrr_source.py). CI runs it.
  - hrrr is a SOURCES row and a record table on the shared isobaric path (the rap row's path, no new code path): every
    field the start and the land surface read is mapped to a wrfprsf00 record or derived, canopy water is read (HRRR
    has it, RAP does not), and there is no isobaric TKE record to ask for (HRRR has none);
  - the fetch pattern keeps every record the table selects, cloud water under HRRR's .idx name CLMR included (RAP's
    .idx says CLWMR: without the CLMR term the start had no cloud water records);
  - the Lambert grid (template 3.30): a section is parsed, the first point lands on (0, 0), the far corner on HRRR's
    published corner (47.842 N, 60.917 W), placement round-trips, and the wind turn equals the cone's analytic angle
    -n (lon - LoV);
  - rrfs (RRFS v1): the same path with the surface read from the 2dfld file beside prslev, cloud ice under GFS's
    code 0-1-23, both files fetched from noaa-rrfs-ops-pds (from 2026-08-12);
  - the fetch plan: an hrrr start needs the HRRR analysis, the GFS f000 (its fallback) and the GFS boundaries 6 h
    older; a gfs start no longer downloads the HRRR or RAP analysis.
"""
from __future__ import annotations

import os
import struct
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from namgpu.init import fetch_case, source_grids as sgr, source_records as srec, sources  # noqa: E402

STAGE_SFC = ("PRES", "HGT", "TSK", "T2", "RH2", "Q2", "U10", "V10", "TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4",
             "SOILW1", "SOILW2", "SOILW3", "SOILW4", "CNWAT", "WEASD", "SNOD", "ICEC", "LANDSRC")
# names in HRRR's wrfprsf00 .idx (2026-01-23 12Z), by (d, c, p)
HRRR_IDX_NAME = {(0, 3, 5): "HGT", (0, 0, 0): "TMP", (0, 1, 0): "SPFH", (0, 2, 2): "UGRD", (0, 2, 3): "VGRD",
                 (0, 1, 22): "CLMR", (0, 1, 82): "CIMIXR", (0, 1, 24): "RWMR", (0, 1, 25): "SNMR", (0, 1, 32): "GRLE",
                 (0, 3, 0): "PRES", (2, 0, 2): "TSOIL", (2, 0, 192): "SOILW", (0, 1, 13): "WEASD", (0, 1, 11): "SNOD",
                 (10, 2, 0): "ICEC", (2, 0, 0): "LAND", (2, 0, 13): "CNWAT"}   # WMO 2-0-13 as the file carries it (grib2_inventory, 2026-11-09)


def _derived(rec):
    out = set()
    for o, _, _ in rec["sfc_derive"]:
        out |= set(o if isinstance(o, tuple) else (o,))
    return out


def test_hrrr_row_on_the_shared_path():
    row = sources.SOURCES["hrrr"]
    assert row["kind"] == "isobaric" and row["gdt"] == 30 and row["records"] == "hrrr"
    assert "boundary" not in row and row["fallback"]["source"] == "gfs"
    assert set(row["soil_from"]) == {"source", "gfs", "gfs-moisture"} and row["licence"].startswith("public-domain")
    assert srec.table(row) is srec.HRRR and not srec.HRRR["donors"]


def test_hrrr_record_table_complete():
    rec = srec.HRRR
    have = set(rec["sfc"]) | _derived(rec)
    assert not [k for k in STAGE_SFC if k not in have]
    assert "CNWAT" in rec["sfc"] and "CNWAT" not in _derived(rec)        # read, not set to 0
    assert not rec["pl_extra"] and rec["warm"]                            # no isobaric TKE in HRRR; condensate kept
    assert rec["pl"]["ICMR"] == (0, 1, 82) and rec["pl_add"]["SNMR"] == ((0, 1, 32),)
    assert srec.RAP["pl_extra"] == {"TKE": (0, 19, 11)}                    # RAP's table is untouched


def test_fetch_pattern_keeps_every_record():
    pat = fetch_case.PATTERNS["hrrr_init"]
    rec = srec.HRRR
    lines = []
    for dcp in list(rec["pl"].values()) + [d for v in rec["pl_add"].values() for d in v]:
        for lev in ("1013.2 mb", "500 mb", "50 mb"):
            lines.append(f":{HRRR_IDX_NAME[dcp]}:{lev}:")
    for name, spec in rec["sfc"].items():
        d, c, p, lt, lv, lv2 = spec
        lev = {"1": "surface", "103": f"{lv} m above ground"}.get(lt, f"{lv}-{lv2} m below ground")
        lines.append(f":{HRRR_IDX_NAME[(d, c, p)]}:{lev}:")
    miss = [ln for ln in lines if not pat.search(ln)]
    assert not miss, miss
    assert pat.search(":CLMR:925 mb:") and not pat.search(":ABSV:500 mb:") and not pat.search(":REFC:entire atmosphere:")


def _section(nx=1799, ny=1059, la1=21.138123, lo1=237.280472, lov=262.5, latin=38.5, dx_mm=3000000, scan=0x40,
             resflags=0x08, shape=6, pcf=0):
    def s4(v):
        return struct.pack(">I", (abs(int(v)) | 0x80000000) if v < 0 else int(v))
    b = bytearray(81)
    b[0:4] = struct.pack(">I", 81)
    b[4] = 3
    b[12:14] = struct.pack(">H", 30)
    b[14] = shape
    b[30:34], b[34:38] = struct.pack(">I", nx), struct.pack(">I", ny)
    b[38:42], b[42:46] = s4(round(la1 * 1e6)), s4(round(lo1 * 1e6))
    b[46] = resflags
    b[47:51], b[51:55] = s4(round(latin * 1e6)), s4(round(lov * 1e6))
    b[55:59], b[59:63] = struct.pack(">I", dx_mm), struct.pack(">I", dx_mm)
    b[63], b[64] = pcf, scan
    b[65:69], b[69:73] = s4(round(latin * 1e6)), s4(round(latin * 1e6))
    return bytes(b)


def test_lambert_geometry():
    g = sgr.LambertGrid.from_section(sgr.parse_grid_section(_section()))
    assert g.grid_relative_winds and (g.nx, g.ny) == (1799, 1059) and g.radius == 6371229.0
    fy, fx = g.place(np.radians([21.138123]), np.radians([237.280472]))
    assert np.allclose(fy, 0.0, atol=1e-6) and np.allclose(fx, 0.0, atol=1e-6)
    la, lo = g.latlon(1058, 1798)                    # HRRR's published NE corner
    assert abs(np.degrees(la) - 47.8422) < 1e-3 and abs(np.degrees(lo) - 360.0 + 60.9172) < 1e-3
    la, lo = g.latlon(np.array([321.7, 1000.0]), np.array([1234.5, 10.0]))
    fy, fx = g.place(la, lo)
    assert np.allclose(fy, [321.7, 1000.0], atol=1e-6) and np.allclose(fx, [1234.5, 10.0], atol=1e-6)
    c, s = g.wind_turn()
    assert np.allclose(c * c + s * s, 1.0, atol=1e-6)
    for j, i in ((500, 1500), (100, 50), (1000, 1700)):
        _, lo = g.latlon(j, i)
        want = -g.n * (np.degrees(lo) - 262.5)
        assert abs(np.degrees(np.arctan2(s[j, i], c[j, i])) - want) < 1e-3


def test_lambert_refusals():
    for kw, word in ((dict(scan=0x00), "flipped"), (dict(pcf=0x80), "south pole"), (dict(shape=4), "ellipsoid")):
        try:
            sgr.LambertGrid.from_section(sgr.parse_grid_section(_section(**kw)))
            raise AssertionError(f"{kw} was accepted")
        except ValueError as e:
            assert word in str(e), (kw, e)


def test_hrrr_fetch_plan(tmp=None):
    import tempfile
    tmp = Path(tmp or tempfile.mkdtemp())
    jobs, _ = fetch_case.plan(datetime(2026, 1, 23, 12), 6, tmp / "raw", source="hrrr", age_h=6)
    files = [(Path(j[2]).name, j[1], j[-1]) for j in jobs if j[0] is not fetch_case._mrms_hour]
    need = {n for n, _, r in files if r}
    assert {"hrrr.t12z.wrfprsf00.grib2", "gfs.t12z.pgrb2.0p25.f000"} <= need, need
    assert {"gfs.t06z.f006.grib2", "gfs.t06z.f009.grib2", "gfs.t06z.f012.grib2"} <= need, need
    url = next(u for n, u, _ in files if n.startswith("hrrr."))
    assert url == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260123/conus/hrrr.t12z.wrfprsf00.grib2", url
    assert not any(n.startswith("rap.") for n, _, _ in files)            # another regional analysis is not fetched
    jobs, skipped = fetch_case.plan(datetime(2026, 1, 23, 12), 6, tmp / "raw", source="gfs", age_h=6)
    names = [Path(j[2]).name for j in jobs if j[0] is not fetch_case._mrms_hour]
    assert not any(n.startswith(("hrrr.", "rap.")) for n in names), names
    assert any(n == "hrrr-wrfprs" and "start source is gfs" in why for n, why in skipped)


def test_rrfs_row_and_table():
    row = sources.SOURCES["rrfs"]
    assert row["gdt"] == 30 and row["records"] == "rrfs" and row["fallback"]["source"] == "gfs"
    rec = srec.table(row)
    assert rec is srec.RRFS and rec["pl"]["ICMR"] == (0, 1, 23)        # GFS's cloud-ice code in RRFS prslev
    assert srec.HRRR["pl"]["ICMR"] == (0, 1, 82)                         # HRRR's table is untouched
    assert rec["sfc_file_from"] == ("prslev", "2dfld") and "prslev" in row["file"]
    assert not [k for k in STAGE_SFC if k not in set(rec["sfc"]) | _derived(rec)]


def test_rrfs_fetch_patterns_keep_every_record():
    names = {**HRRR_IDX_NAME, (0, 1, 23): "ICMR"}
    rec = srec.RRFS
    prs, sfc = fetch_case.PATTERNS["rrfs_prs"], fetch_case.PATTERNS["rrfs_sfc"]
    for dcp in list(rec["pl"].values()) + [d for v in rec["pl_add"].values() for d in v]:
        for lev in ("1000 mb", "500 mb", "2 mb"):
            assert prs.search(f":{names[dcp]}:{lev}:"), (dcp, lev)
    for name, spec in rec["sfc"].items():
        d, c, p, lt, lv, lv2 = spec
        lev = {"1": "surface", "103": f"{lv} m above ground"}.get(lt, f"{lv}-{lv2} m below ground")
        assert sfc.search(f":{names[(d, c, p)]}:{lev}:"), name
    assert not prs.search(":TMP:surface:") and not sfc.search(":TMP:500 mb:")


def test_rrfs_fetch_plan(tmp=None):
    import tempfile
    tmp = Path(tmp or tempfile.mkdtemp())
    jobs, _ = fetch_case.plan(datetime(2026, 10, 14, 0), 6, tmp / "raw", source="rrfs", age_h=6)
    files = {Path(j[2]).name: (j[1], j[-1]) for j in jobs if j[0] is not fetch_case._mrms_hour}
    for n in ("rrfs.t00z.prslev.3km.f000.conus.grib2", "rrfs.t00z.2dfld.3km.f000.conus.grib2"):
        assert files[n][1], n
    assert files["rrfs.t00z.2dfld.3km.f000.conus.grib2"][0] == (
        "https://noaa-rrfs-ops-pds.s3.amazonaws.com/rrfs.20261014/00/rrfs.t00z.2dfld.3km.f000.conus.grib2")
    _, sk = fetch_case.plan(datetime(2026, 1, 23, 12), 6, tmp / "raw", source="rrfs", age_h=6)
    assert any(n == "rrfs-prslev" and "archive starts" in why for n, why in sk)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")


def test_hrrr_soil_default_is_gfs_moisture():
    from types import SimpleNamespace
    from namgpu.init import gfs_start
    ctx = SimpleNamespace(cfg=SimpleNamespace(raw={}))
    import os
    os.environ.pop(gfs_start.SOIL_FROM_ENV, None)
    assert gfs_start.soil_from(ctx, sources.SOURCES["hrrr"]) == "gfs-moisture"
    assert gfs_start.soil_from(ctx, sources.SOURCES["rap"]) == "source"           # unmeasured rows keep theirs
    ctx.soil_from = "source"
    assert gfs_start.soil_from(ctx, sources.SOURCES["hrrr"]) == "source"
