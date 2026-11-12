"""Lambert conformal and polar stereographic grids (products.sets.OutGrid, sets._ij, ncep_firewx.master_projection)
against NCEP's published grids: Office Note 388 Table B corner points and pole points, and the grid sections of
NOAA's fire-weather nest files (nam.20261103 t00z / t06z firewxnest.hiresf01). WMO-No. 306 flag table 3.3 for the
resolution and component flags NCEP's files carry.

    python tests/test_projections_ncep.py        (or pytest)
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.products import sets as PS  # noqa: E402
from namgpu.nest import ncep_firewx as FW, ncep_nests  # noqa: E402


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def _corners_agree(g, corners, tol_deg):
    for i, j, la, lo in corners:
        gla, glo = g.latlon_ij(i - 1.0, j - 1.0)
        assert abs(float(gla) - la) < tol_deg, (i, j, float(gla), la)
        assert abs(_wrap(float(glo) - lo)) * math.cos(math.radians(la)) < tol_deg, (i, j, float(glo), lo)


def test_grid_218_is_office_note_388():
    # ON 388 Table B grid 218: 614 x 428, (1,1) 12.190N 133.459W, LoV 95W, Dx 12.19058 km at 25N (tangent)
    g = PS.OutGrid("lambert", 614, 428, 12.190, 226.541, 12190.58, 12190.58, lov=265.0, latin1=25.0, latin2=25.0)
    _corners_agree(g, [(1, 428, 54.564, -152.878), (614, 428, 57.328, -49.420), (614, 1, 14.342, -65.127)], 0.002)
    pi, pj = PS._ij(g, 90.0, -95.0)
    assert abs(pi - 347.668) < 0.01 and abs(pj - 1190.097) < 0.01, (pi, pj)
    i, j = PS._ij(g, 35.0, -95.0)                          # "35N & 95W falls on point (347.668, 160.999)"
    assert abs(i - 347.668) < 0.005 and abs(j - 160.999) < 0.005, (i, j)


def test_polar_grids_are_office_note_388():
    # Table B grid 202 (LoV 105W, 190.5 km) and grid 217 (Alaska, LoV 135W, 22.5 km), both true at 60N
    g = PS.OutGrid("polar", 65, 43, 7.838, 218.972, 190500.0, 190500.0, lov=255.0, lad=60.0)
    _corners_agree(g, [(1, 43, 35.616, 168.577), (65, 43, 35.617, -18.576), (65, 1, 7.838, -68.973)], 0.002)
    pi, pj = PS._ij(g, 90.0, 0.0)
    assert abs(pi - 33.0) < 0.002 and abs(pj - 45.0) < 0.002, (pi, pj)
    g = PS.OutGrid("polar", 277, 213, 30.0, 187.0, 22500.0, 22500.0, lov=225.0, lad=60.0)
    _corners_agree(g, [(1, 213, 50.454, 143.597), (277, 213, 70.111, -62.850), (277, 1, 38.290, -114.856)], 0.002)
    pi, pj = PS._ij(g, 90.0, 0.0)
    assert abs(pi - 188.818) < 0.002 and abs(pj - 241.397) < 0.002, (pi, pj)


def test_ij_scalar_and_array():
    g = PS.FIREWX_CONUS
    assert PS._ij(g, g.la1, g.lo1) == (1.0, 1.0)
    i, j = PS._ij(g, np.array([g.la1, 36.487]), np.array([g.lo1, 244.377]))
    assert i.shape == (2,) and abs(i[1] - 700.0) < 0.05 and abs(j[1] - 1334.0) < 0.05


def test_master_projection_round_trip_and_noaa_first_points():
    fwd, inv = FW.master_projection(FW.FIREWX_MASTER["conus"])
    # NOAA's firewx files (nam.20261103): each grid's first point is a point of the 1.27 km master grid
    for (la1, lo1), ij in (((36.487, 244.377), (700, 1334)), ((39.182, 233.507), (-11, 1719))):
        i, j = fwd(la1, lo1)
        assert abs(i - ij[0]) < 0.05 and abs(j - ij[1]) < 0.05, (la1, lo1, i, j)
        la, lo = inv(float(ij[0]), float(ij[1]))
        assert abs(la - la1) < 0.0006 and abs((lo % 360.0) - lo1) < 0.0006, (la, lo)
    assert max(abs(a - b) for a, b in zip(inv(1.0, 1.0), (20.192, -121.554))) < 1e-9
    fwd, inv = FW.master_projection(FW.FIREWX_MASTER["alaska"])
    for lat, lon in ((61.2, -149.9), (64.8, -147.7), (71.3, -156.8), (55.0, 160.0)):
        la, lo = inv(*fwd(lat, lon))
        assert abs(la - lat) < 1e-9 and abs(_wrap(lo - lon)) < 1e-9 and -180.0 <= lo <= 180.0


def test_firewx_output_grids_are_noaas():
    # both of NCEP's last production points: the computed box equals the grid section of NOAA's file
    for hh, spec in ((0, "lambert:265:25:25 244.377:543:1270 36.487:379:1270"),
                     (6, "lambert:265:25:25 233.507:646:1270 39.182:389:1270")):
        g = ncep_nests.nest_grid("firewxnest", cycle_hour=hh)
        assert g["firewx"]["output_grid"]["new_grid"] == spec, (hh, g["firewx"]["output_grid"])


def test_resolution_flags_56_are_grid_relative():
    # WMO-No. 306 flag table 3.3 (bit 1 the most significant): bit 3 i increments given (32), bit 4 j increments
    # given (16), bit 5 u/v relative to the grid (8). NCEP's grid 218, fire-weather and native analysis files carry 56.
    from namgpu.init import init_grib2_decode, source_grids
    assert 56 == 32 | 16 | 8
    assert init_grib2_decode.GRID_RELATIVE_UV_BIT == source_grids.GRID_RELATIVE_UV_BIT == 8
    assert 56 & init_grib2_decode.GRID_RELATIVE_UV_BIT and not (0x30 & init_grib2_decode.GRID_RELATIVE_UV_BIT)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("test_projections_ncep: PASS")
