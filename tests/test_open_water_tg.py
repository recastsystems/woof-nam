"""GPU: an isobaric start's open water takes the NAM's own TG (280 K), not land soil filled out from the coasts.

The start fills the source's soil over its water from the nearest source land, and TG is the bottom layer. Noah never
runs on open water, so nothing resets that TG: the history's TG and the 3 m soil temperature product kept the fill.
After sea ice took its own column (sea_ice_column), the 88.9N and 84N 30E GFS boxes of 2026-11-13 still showed the
fill's rays and wedges in TG over the open Barents and Kara Seas. The NAM's own analysis has 280.0 K over every
open-water cell (and 271.2 K over its sea ice).
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)

from namgpu.init import gfs_start as gs, init_gfs_hinterp as hin  # noqa: E402

MISSING = -9999.0


def _cap(dlat=0.5, lat0=60.0):
    """A polar cap of a global 0.5 degree source: land on 70-72N whose soil temperature alternates 250 / 270 K every
    15 degrees of longitude, open water from 72N to 84N, pack ice (ICEC 0.9) from 84N to the pole."""
    lat = np.arange(lat0, 90.0 + 1e-9, dlat)
    nx = int(round(360.0 / dlat))
    lon = np.arange(nx) * dlat
    land = np.zeros((lat.size, nx), bool)
    land[(lat >= 70.0) & (lat <= 72.0)] = True
    t = np.where(land, 260.0 + 10.0 * np.sign(np.sin(np.radians(lon) * 12.0))[None, :], MISSING).astype(np.float32)
    icec = np.where((lat >= 84.0)[:, None], 0.9, 0.0) * np.ones((1, nx))
    grid = hin.Grid(lat0=lat0, lon0=0.0, dlat=dlat, dlon=dlat, lon_periodic=True)
    return lat, land, cp.asarray(t), icec.astype(np.float32), grid


def _start_surface(lat, land, t, icec, grid):
    """The start's soil path on the source grid itself: the fill, then the four layers and TG as raw_sfc holds them."""
    (a, m), = gs._ring_fill_many([t], [cp.asarray(~land)], lon_periodic=True, grid=grid)
    assert not bool(m.any())
    raw = {
        "TSOIL": cp.ascontiguousarray(cp.stack([a] * 4)),
        "TSOIL_3M": a.copy(),
        "TMP_SFC": cp.full(a.shape, 250.0, cp.float32),
        "ICEC": cp.asarray(icec),
        "LAND": cp.asarray(land.astype(np.float32)),
    }
    gs.sea_ice_column(raw)
    return raw


def test_open_water_tg_has_no_rays_and_is_the_nams():
    lat, land, t, icec, grid = _cap()
    raw = _start_surface(lat, land, t, icec, grid)
    water = (~land) & (icec <= 0.5)
    rows = np.flatnonzero(water.all(axis=1))
    before = cp.asnumpy(raw["TSOIL_3M"])
    # the defect: the fill carries the coast's 15-degree pattern over the open water (rays and wedges round the circle)
    assert max(float(np.ptp(before[j])) for j in rows) > 5.0
    n = gs.open_water_tg(raw)
    tg = cp.asnumpy(raw["TSOIL_3M"])
    assert n == int(water.sum())
    assert (tg[water] == np.float32(gs.OPEN_WATER_TG)).all()
    for j in rows:
        assert float(np.ptp(tg[j])) == 0.0, f"{lat[j]}N: open-water TG spread {np.ptp(tg[j]):.2f} K"
    # sea ice keeps its own TG, and land its own soil
    ice = icec > 0.5
    assert (tg[ice] == np.float32(gs.SEA_ICE_TBOT)).all()
    assert np.array_equal(tg[land], cp.asnumpy(t)[land])


def test_open_water_tg_touches_only_open_water():
    rng = np.random.default_rng(7)
    jm, im = 6, 8
    land = np.zeros((jm, im), np.float32)
    land[:, :2] = 1.0
    icec = np.zeros((jm, im), np.float32)
    icec[:, 5:] = 0.9            # pack ice
    icec[0, 0] = 0.9             # an ice flag on land: land stays land
    icec[1, 3] = 0.4             # thin ice: open water in Noah
    tsoil = rng.uniform(245.0, 275.0, (4, jm, im)).astype(np.float32)
    raw = {"TSOIL": cp.asarray(tsoil), "TSOIL_3M": cp.asarray(tsoil[3].copy()),
           "TMP_SFC": cp.full((jm, im), 250.0, cp.float32), "ICEC": cp.asarray(icec), "LAND": cp.asarray(land)}
    gs.sea_ice_column(raw)
    stc = cp.asnumpy(raw["TSOIL"]).copy()
    ice_tg = cp.asnumpy(raw["TSOIL_3M"])[icec > 0.5].copy()
    n = gs.open_water_tg(raw)
    tg = cp.asnumpy(raw["TSOIL_3M"])
    water = (land < 0.5) & (icec <= 0.5)
    assert n == int(water.sum()) == 6 * 3
    assert (tg[water] == np.float32(280.0)).all() and tg[1, 3] == np.float32(280.0)
    assert np.array_equal(tg[land > 0.5], tsoil[3][land > 0.5])
    assert np.array_equal(tg[icec > 0.5], ice_tg)
    assert np.array_equal(cp.asnumpy(raw["TSOIL"]), stc)      # the four layers are not touched
