"""The maps leave a grid's boundary rows off (maps BOUNDARY ROWS): the history's LNSH rows plus EDGE_ROWS_EXTRA on
each side, LNSH 5 for a history written before it carried LNSH; the frame is the drawn interior's; a grid too small
to trim is drawn whole; rain only in the boundary rows is a dry window, not an all-blank field the renderer refuses.

Defect: a WOOF-NAM box's maps showed a band of stronger 10 m wind along its edges and a warm strip along its west edge
(3 km box over the southern Plains from GFS 2026-11-13 00Z; east edge at +3 h 9.3-10.0 kt in the outer five rows
against 4.0-5.0 kt from the seventh in): those rows hold the start's GFS boundary values, not the forecast. CPU only (numpy, netCDF4). Run: python tests/test_maps_edge_rows.py
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from namgpu.io import maps  # noqa: E402

START = datetime(2026, 11, 13, 0)
NY, NX = 60, 70


def history(path: Path, hour: int, lnsh=None, band=False, rain_edge_only=0.0):
    lat, lon = np.meshgrid(np.linspace(31.5, 40.5, NY), np.linspace(-103.6, -92.4, NX), indexing="ij")
    t2 = np.full((NY, NX), 290.0)
    if band:
        t2[:5, :] = t2[-5:, :] = t2[:, :5] = t2[:, -5:] = 300.0    # the boundary rows' own values
    rain = np.zeros((NY, NX))
    if rain_edge_only:
        rain[:, :3] = rain_edge_only                               # rain in the west boundary rows only
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("south_north", NY)
        nc.createDimension("west_east", NX)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = netCDF4.stringtochar(np.array([(START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")], "S19"))
        for name, a in (("XLAT", lat), ("XLONG", lon), ("T2", t2), ("RAINNC", rain), ("RAINC", np.zeros_like(rain))):
            nc.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = a
        nc.START_DATE = START.strftime("%Y-%m-%d_%H:%M:%S")
        if lnsh is not None:
            nc.LNSH = lnsh
            nc.LNSV = lnsh
    return path


def cf_shape(cf):
    with netCDF4.Dataset(cf) as nc:
        return nc["lat"].shape, np.asarray(nc["t2m"][0]) if "t2m" in nc.variables else None


def main() -> int:
    work = Path(tempfile.mkdtemp())
    need = sorted({"XLAT", "XLONG", "T2", "RAINNC", "RAINC"})
    x = maps.EDGE_ROWS_EXTRA

    # a history with LNSH 5 and a band in its boundary rows: the CF frame is the grid less 5 + extra rows a side,
    # and nothing of the band is left in it
    h1 = history(work / "wrfout_d01_2026-11-13_01:00:00", 1, lnsh=5, band=True)
    assert maps.edge_rows(h1) == 5 + x, maps.edge_rows(h1)
    cf, valid, start, have, sk, table, frame = maps._prep_frame((str(h1), {}, need, ["t2m"], str(work), "WOOF-NAM"))
    shape, t2 = cf_shape(cf)
    n = 5 + x
    assert shape == (NY - 2 * n, NX - 2 * n), shape
    assert float(np.nanmax(t2)) < 290.0 * 9 / 5 - 459.67 + 0.01, "the boundary rows' band is still drawn"
    # the frame is the interior's: inside the whole grid's frame on every side
    whole = maps.frame_bounds(*np.meshgrid(np.linspace(31.5, 40.5, NY), np.linspace(-103.6, -92.4, NX), indexing="ij"))
    assert frame[0] > whole[0] and frame[1] < whole[1] and frame[2] > whole[2] and frame[3] < whole[3], (frame, whole)

    # a history from before LNSH was written: the configfile's 5
    h0 = history(work / "wrfout_d01_2026-11-13_00:00:00", 0)
    assert maps.edge_rows(h0) == maps.EDGE_ROWS_LNSH + x == 5 + x
    # a fire-weather nest's LNSH 3
    (work / "fw").mkdir()
    h3 = history(work / "fw" / "wrfout_d01_2026-11-13_00:00:00", 0, lnsh=3)
    assert maps.edge_rows(h3) == 3 + x

    # the 0.1.0 call trims too
    r = maps._prep_frame((str(h1), None, need, ["t2m"], str(work), "WOOF-NAM"))
    assert cf_shape(r[0])[0] == (NY - 2 * n, NX - 2 * n)

    # rain only inside the boundary rows: a dry window (an expected skip), not an all-blank field
    h2 = history(work / "wrfout_d01_2026-11-13_02:00:00", 2, lnsh=5, rain_edge_only=5.0)
    cf, valid, start, have, sk, table, frame = maps._prep_frame(
        (str(h2), {1: str(h1)}, need, ["t2m", "qpf_1h"], str(work), "WOOF-NAM"))
    assert have == ["t2m"] and [s[0] for s in sk] == ["qpf_1h"] and "dry window" in sk[0][1], (have, sk)

    # a grid too small to trim is drawn whole
    small = np.zeros((10, 12))
    assert maps.trim_edges(small, 5).shape == (10, 12)
    assert maps.trim_edges(np.zeros((3, 40, 50)), 5).shape == (3, 30, 40)
    print("test_maps_edge_rows: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
