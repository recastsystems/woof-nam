"""CPU checks of the RAP start source row (no GPU, no network).

Run: python tests/test_rap_source.py   (or pytest tests/test_rap_source.py). CI runs it.
  - rap is a SOURCES row and a record table on the shared isobaric path (no new code path): every field the start,
    the land surface and the boundaries read is mapped to a RAP wrfprsf00 record or derived; the condensate species
    and TKE the warm start reads are mapped; the fetch pattern keeps every record the table selects;
  - the RUC soil remap: Noah's four layer midpoints read linearly between the bracketing RUC nodes (gpuwm's
    "linear_node_samples"), values checked by hand, and a linear profile is reproduced exactly;
  - the rotated grid: a 3.32769 section is parsed, the first point and the centre land on the right indices, the
    grid-relative wind turn is the identity on the centre meridian and is a rotation everywhere;
  - the fetch plan: a rap start needs the RAP analysis, the GFS f000 (its fallback) and the GFS boundaries 6 h older.
"""
from __future__ import annotations

import os
import re
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
# idx names of the RAP wrfprsf00 records (NCEP GRIB2 tables as wgrib2 prints them), by (d, c, p)
RAP_IDX_NAME = {(0, 3, 5): "HGT", (0, 0, 0): "TMP", (0, 1, 0): "SPFH", (0, 2, 2): "UGRD", (0, 2, 3): "VGRD",
                (0, 1, 22): "CLWMR", (0, 1, 82): "CIMIXR", (0, 1, 24): "RWMR", (0, 1, 25): "SNMR", (0, 1, 32): "GRLE",
                (0, 19, 11): "TKE", (0, 3, 0): "PRES", (2, 0, 2): "TSOIL", (2, 0, 192): "SOILW", (0, 1, 13): "WEASD",
                (0, 1, 11): "SNOD", (10, 2, 0): "ICEC", (2, 0, 0): "LAND"}


def _derived(rec):
    out = set()
    for o, _, _ in rec["sfc_derive"]:
        out |= set(o if isinstance(o, tuple) else (o,))
    return out


def test_rap_row_on_the_shared_path():
    row = sources.SOURCES["rap"]
    assert row["kind"] == "isobaric" and row["gdt"] == 32769 and row["records"] == "rap"
    assert "boundary" not in row                    # GFS boundaries (gfs_bocos.boundary_row), as the native start
    assert row["fallback"]["source"] == "gfs" and "gfs" in row["soil_from"] and "source" in row["soil_from"]
    assert row["licence"].startswith("public-domain")
    assert srec.table(row) is srec.RAP and not srec.RAP["donors"]


def test_rap_record_table_complete():
    rec = srec.RAP
    have = set(rec["sfc"]) | _derived(rec)
    missing = [k for k in STAGE_SFC if k not in have]
    assert not missing, missing
    assert "TDEEP" in have                          # Noah's TG from RAP's own 3 m node
    seen = set(rec["sfc"])
    for out, op, args in rec["sfc_derive"]:
        srcs = args[0] if op in ("layer_remap", "node_interp") else [a for a in args if isinstance(a, str)]
        assert set(srcs) <= seen, (out, srcs)
        seen |= set(out if isinstance(out, tuple) else (out,))
    for k in rec["sfc_columns"]:
        assert k in rec["sfc"]
    # every condensate slot the warm start reads, cloud ice on NCEP 0-1-82 (not GFS's 0-1-23), graupel into snow
    assert {"CLMR", "ICMR", "RWMR", "SNMR"} <= set(rec["pl"]) and rec["pl"]["ICMR"] == (0, 1, 82)
    assert rec["pl_add"]["SNMR"] == ((0, 1, 32),) and rec["pl_extra"]["TKE"] == (0, 19, 11) and rec["warm"]
    assert "RH" in rec["pl_derive"] and rec["pl_derive"]["RH"][0] == "rh_from_q"


def test_fetch_pattern_keeps_every_record():
    pat = fetch_case.PATTERNS["rap_init"]
    rec = srec.RAP
    lines = []
    for dcp in list(rec["pl"].values()) + [d for v in rec["pl_add"].values() for d in v] + list(rec["pl_extra"].values()):
        for lev in ("1013.2 mb", "500 mb", "50 mb"):
            lines.append(f":{RAP_IDX_NAME[dcp]}:{lev}:")
    for name, spec in rec["sfc"].items():
        d, c, p, lt, lv, lv2 = spec
        n = RAP_IDX_NAME[(d, c, p)]
        lev = {"1": "surface", "103": f"{lv} m above ground"}.get(lt, f"{lv}-{lv2} m below ground")
        lines.append(f":{n}:{lev}:")
    miss = [ln for ln in lines if not pat.search(ln)]
    assert not miss, miss
    assert not pat.search(":TMP:1 hybrid level:") and not pat.search(":ABSV:500 mb:")


def test_ruc_node_weights():
    w = srec.node_interp_weights(srec.RUC_SOIL_NODES, srec.NOAH_SOIL_LAYERS)
    assert w.shape == (4, 9) and np.allclose(w.sum(axis=1), 1.0)
    # 0.05 m between the 0.04 and 0.1 nodes, 0.25 between 0.1 and 0.3, 0.7 between 0.6 and 1.0, 1.5 between 1.0 and 1.6
    want = {0: {2: 5 / 6, 3: 1 / 6}, 1: {3: 0.25, 4: 0.75}, 2: {5: 0.75, 6: 0.25}, 3: {6: 1 / 6, 7: 5 / 6}}
    for t, row in want.items():
        for k, v in row.items():
            assert abs(w[t, k] - v) < 1e-12, (t, k, w[t, k], v)
        assert np.count_nonzero(w[t]) == len(row)


def test_ruc_soil_remap_through_the_table():
    rec = srec.RAP
    z = np.asarray(srec.RUC_SOIL_NODES)
    shape = (3, 4)
    sfc = {}
    for k, d in enumerate(z):                       # a linear profile is read back exactly at every midpoint
        sfc[f"STN{k}"] = np.full(shape, 290.0 - 5.0 * d, np.float32)
        sfc[f"SMN{k}"] = np.full(shape, 0.20 + 0.05 * d, np.float32)
    srec.derive_sfc(rec, sfc, only=("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4", "SOILW1", "SOILW2", "SOILW3", "SOILW4",
                                    "TDEEP", "CNWAT"))
    for i, (t0, t1) in enumerate(srec.NOAH_SOIL_LAYERS):
        m = 0.5 * (t0 + t1)
        assert np.allclose(sfc[f"TSOIL{i + 1}"], 290.0 - 5.0 * m, atol=1e-4)
        assert np.allclose(sfc[f"SOILW{i + 1}"], 0.20 + 0.05 * m, atol=1e-6)
        assert sfc[f"TSOIL{i + 1}"].dtype == np.float32
    assert np.allclose(sfc["TDEEP"], 290.0 - 15.0) and np.all(sfc["CNWAT"] == 0.0)


def _section(la1, lo1, lac, loc, di_nano, nx, ny, resflags=0x38, scan=0x40):
    def s4(v):
        return struct.pack(">I", (abs(int(v)) | 0x80000000) if v < 0 else int(v))
    b = bytearray(80)
    b[0:4] = struct.pack(">I", 80)
    b[4] = 3
    b[12:14] = struct.pack(">H", 32769)
    b[14] = 6
    b[30:34], b[34:38] = struct.pack(">I", nx), struct.pack(">I", ny)
    b[46:50], b[50:54] = s4(round(la1 * 1e6)), s4(round(lo1 * 1e6))
    b[54] = resflags
    b[55:59], b[59:63] = s4(round(lac * 1e6)), s4(round(loc * 1e6))
    b[63:67], b[67:71] = struct.pack(">I", di_nano), struct.pack(">I", di_nano)
    b[71] = scan
    return bytes(b)


# rap.t00z.wrfprsf00.grib2 of 2023-07-29, section 3 as read from the file
RAP_SEC = dict(la1=-10.590603, lo1=220.914154, lac=54.0, loc=254.0, di_nano=121813000, nx=953, ny=834)


def test_rotated_grid_geometry():
    g = sgr.RotatedGrid.from_section(sgr.parse_grid_section(_section(**RAP_SEC)))
    assert g.grid_relative_winds and (g.nx, g.ny) == (953, 834)
    assert abs(g.dlmd - 0.121833) < 2e-6 and abs(g.dphd - 0.121833) < 2e-6
    fy, fx = g.place(np.radians([RAP_SEC["la1"], 54.0]), np.radians([RAP_SEC["lo1"], -106.0]))
    assert np.allclose(fy, [0.0, 416.5], atol=1e-6) and np.allclose(fx, [0.0, 476.0], atol=1e-6)
    la, lo = g.latlon(fy, fx)                       # round trip
    assert np.allclose(np.degrees(la), [RAP_SEC["la1"], 54.0], atol=1e-9)
    assert np.allclose(np.degrees(lo) % 360.0, [RAP_SEC["lo1"], 254.0], atol=1e-9)
    w = g.edge_weight(np.array([416.5, 2.0, -3.0]), np.array([476.0, 476.0, 476.0]), 8)
    assert np.allclose(w, [1.0, 0.25, 0.0])


def test_rotated_grid_refusals():
    bad = dict(RAP_SEC, di_nano=125000000)          # not centred on the rotation centre at this spacing
    try:
        sgr.RotatedGrid.from_section(sgr.parse_grid_section(_section(**bad)))
        raise AssertionError("an off-centre spacing was accepted")
    except ValueError as e:
        assert "far edge" in str(e)
    try:
        sgr.RotatedGrid.from_section(sgr.parse_grid_section(_section(**RAP_SEC, scan=0x00)))
        raise AssertionError("scan 0x00 was accepted")
    except ValueError as e:
        assert "flipped" in str(e)


def test_wind_turn():
    small = dict(RAP_SEC, nx=953, ny=834)
    g = sgr.RotatedGrid.from_section(sgr.parse_grid_section(_section(**small)))
    c, s = g.wind_turn()
    assert np.allclose(c * c + s * s, 1.0, atol=1e-6)
    # the centre meridian (column 476) south of the pole crossing: the grid x axis points east
    j = np.arange(0, 700)
    assert np.allclose(c[j, 476], 1.0, atol=1e-6) and np.allclose(s[j, 476], 0.0, atol=1e-6)
    # west of the centre the grid's x axis tilts north of east (a rotated pole at 36 N over 106 W), east of it south
    assert s[416, 100] > 0.2 and s[416, 850] < -0.2
    u, v = np.ones((2, 834, 953), np.float32), np.zeros((2, 834, 953), np.float32)
    sgr.to_earth(u, v, (c, s))
    assert np.allclose(np.hypot(u, v), 1.0, atol=1e-6)


def test_rap_fetch_plan(tmp=None):
    import tempfile
    tmp = Path(tmp or tempfile.mkdtemp())
    jobs, skipped = fetch_case.plan(datetime(2023, 7, 29, 0), 6, tmp / "raw", source="rap", age_h=6)
    req = [(Path(j[2]).name if j[0] is not fetch_case._mrms_hour else "mrms", j[-1]) for j in jobs]
    need = {n for n, r in req if r}
    assert "rap.t00z.wrfprsf00.grib2" in need, need
    assert "gfs.t00z.pgrb2.0p25.f000" in need, need                        # the fallback start
    assert {"gfs.t18z.f006.grib2", "gfs.t18z.f009.grib2", "gfs.t18z.f012.grib2"} <= need, need   # 6 h older cycle
    rap_url = next(j[1] for j in jobs if j[0] is not fetch_case._mrms_hour and "wrfprs" in str(j[2]))
    assert rap_url == "https://noaa-rap-pds.s3.amazonaws.com/rap.20230729/rap.t00z.wrfprsf00.grib2", rap_url
    # before the archive: the row is skipped with its reason
    _, sk = fetch_case.plan(datetime(2021, 2, 1, 0), 6, tmp / "raw", source="rap", age_h=6)
    assert any(n == "rap-wrfprs" and "archive starts" in why for n, why in sk)
    # a gfs start still does not need RAP
    jobs, _ = fetch_case.plan(datetime(2023, 7, 29, 0), 6, tmp / "raw", source="gfs", age_h=6)
    assert not any(j[-1] and "wrfprs" in str(j[2]) for j in jobs if j[0] is not fetch_case._mrms_hour)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
