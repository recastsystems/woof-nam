"""CPU checks: ECMWF's pressure levels under the ground no longer warm the start in arctic air (no GPU, no network).

Run: python tests/test_below_ground_t.py   (or pytest). Breakage it prevents: an ERA5 or IFS start was up to 20 K warm
on its lowest model levels under a surface inversion, and its 0 h 2 m temperature 6.9 K warm on the coldest tenth of
the land (ERA5 2015-01-26 12Z on the 12 km grid; 4 K at IFS's coldest points, 6 C under ERA5's arctic air on a box).
The NAM's interpolation ignores the surface level for temperature and brackets a model level between the isobaric levels
around it, under-ground ones included; ECMWF writes an under-ground level from above the inversion, NCEP from the air
near the ground. The column below is ERA5's own at 69.77N 75.86W on 2015-01-26 12Z.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from namgpu.init import gfs_start, source_records as srec  # noqa: E402

# level 0 = the surface (psfc, terrain, 2 m T), then 1000, 975, 950, 925, 900 hPa (ERA5 T and Z there)
PRES = np.array([97280.0, 100000.0, 97500.0, 95000.0, 92500.0, 90000.0], np.float32)
GHT = np.array([225.0, 25.0, 207.0, 386.0, 572.0, 765.0], np.float32)
TT = np.array([-57.6, -34.4, -36.2, -35.0, -34.0, -33.5], np.float32) + np.float32(273.15)


def interp_lin_T(pres, tt, desired):
    """init_gfs_vinterp_lin's interior bracket for temperature (ignore_lowest: the search starts at level 1)."""
    for kk in range(1, len(pres) - 1):
        if pres[kk] > desired > pres[kk + 1]:
            dlnp = math.log(pres[kk]) - math.log(pres[kk + 1])
            return tt[kk + 1] + (tt[kk] - tt[kk + 1]) / dlnp * (math.log(desired) - math.log(pres[kk + 1]))
    raise AssertionError("no bracket")


def test_under_ground_levels_take_the_surface_air():
    out = gfs_start.below_ground_from_2m(PRES[:, None], GHT[:, None], TT[:, None])[:, 0]
    # 1000 and 975 hPa are under the ground (p > 972.8 hPa): 2 m T carried down 200 m and 18 m at 6.5 K/km
    assert abs(out[1] - (TT[0] + 0.0065 * 200.0)) < 1e-3 and abs(out[2] - (TT[0] + 0.0065 * 18.0)) < 1e-3
    # the surface and every level above the ground keep the source's values
    assert np.array_equal(out[[0, 3, 4, 5]], TT[[0, 3, 4, 5]])
    assert np.array_equal(TT, TT.copy())                     # the input is not changed in place


def test_lowest_model_level_in_the_inversion():
    lowest = 97000.0                                          # a model level about 25 m above the ground
    before = interp_lin_T(PRES, TT, lowest) - 273.15
    after = interp_lin_T(PRES, gfs_start.below_ground_from_2m(PRES[:, None], GHT[:, None], TT[:, None])[:, 0],
                         lowest) - 273.15
    assert before > -36.5                                     # what the start had: the inversion top's air
    assert -57.6 < after < -50.0, after                       # now in the inversion, near the 2 m air
    # GFS-like column (its under-ground level near the 2 m air): the change moves it little
    gfs = TT.copy()
    gfs[1], gfs[2] = TT[0] + 1.8, TT[0] + 1.6
    moved = interp_lin_T(PRES, gfs_start.below_ground_from_2m(PRES[:, None], GHT[:, None], gfs[:, None])[:, 0], lowest)
    assert abs(moved - interp_lin_T(PRES, gfs, lowest)) < 1.5


def test_rows():
    assert srec.ERA5.get("below_ground") == srec.BELOW_GROUND_FROM_2M
    assert srec.IFS.get("below_ground") == srec.BELOW_GROUND_FROM_2M      # ifs and ifs-0p25
    assert srec.AIFS.get("below_ground") == srec.BELOW_GROUND_FROM_2M     # starts from ifs-0p25, edges AIFS
    for name in ("GFS", "AIGFS", "RAP", "HRRR", "RRFS"):
        tab = getattr(srec, name, None)
        if tab is not None:
            assert tab.get("below_ground") is None, name                 # NCEP's own under-ground values


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
