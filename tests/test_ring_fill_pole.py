"""GPU: the start's soil fill over source water respects the sphere near a pole (the pole starburst, 2026-11-13).

The start fills the source's soil layers over its water from the nearest source land, ring by ring, and sea-ice
cells start Noah from those layers; TG is the bottom one. On a latitude-longitude source a ring stepped one column
and one row: a column at 89N is 1/57 of a row long, so each meridian carried its own coast's value to the pole.
The Arctic Ocean's soil (and the ice's starting temperature) became rays from the pole, plain in 2 m temperature
on the 88.9N IFS 0.1 box, with a pole row that held a different value in every column.
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
    """A polar cap of a global 0.5 degree source (the rows a box near the pole decodes, pole row included): land on
    70-72N whose soil temperature alternates 250 / 270 K every 15 degrees of longitude, open water to the pole."""
    lat = np.arange(lat0, 90.0 + 1e-9, dlat)
    nx = int(round(360.0 / dlat))
    lon = np.arange(nx) * dlat
    land = np.zeros((lat.size, nx), bool)
    land[(lat >= 70.0) & (lat <= 72.0)] = True
    t = np.where(land, 260.0 + 10.0 * np.sign(np.sin(np.radians(lon) * 12.0))[None, :], MISSING).astype(np.float32)
    grid = hin.Grid(lat0=lat0, lon0=0.0, dlat=dlat, dlon=dlat, lon_periodic=True)
    return lat, cp.asarray(t), cp.asarray(~land), grid


def test_soil_fill_near_the_pole_has_no_rays():
    lat, t, miss, grid = _cap()
    (a, m), = gs._ring_fill_many([t], [miss], lon_periodic=True, grid=grid)
    a = cp.asnumpy(a)
    assert not bool(m.any())
    pole = a[-1]
    # the pole row is one point: one value
    assert float(pole.max() - pole.min()) < 1e-3, f"pole row spread {pole.max() - pole.min():.2f} K"
    # within 2 degrees of the pole (222 km) the coast's 15-degree pattern, 2,000 km away, is mixed out; the old fill
    # carried it to the pole along each meridian at its full 20 K
    for j in np.flatnonzero(lat >= 88.0):
        spread = float(a[j].max() - a[j].min())
        assert spread < 1.0, f"{lat[j]}N: spread round the circle {spread:.2f} K"
    # the land itself is untouched, and nothing leaves the coast's range
    land = ~cp.asnumpy(miss)
    assert np.array_equal(a[land], cp.asnumpy(t)[land])
    assert a.min() >= 250.0 - 1e-3 and a.max() <= 270.0 + 1e-3


def test_soil_fill_south_pole_cap_is_the_mirror():
    lat, t, miss, grid = _cap()
    tn = cp.ascontiguousarray(t[::-1])
    mn = cp.ascontiguousarray(miss[::-1])
    gsouth = hin.Grid(lat0=-90.0, lon0=0.0, dlat=grid.dlat, dlon=grid.dlon, lon_periodic=True)
    (a, m), = gs._ring_fill_many([tn], [mn], lon_periodic=True, grid=gsouth)
    a = cp.asnumpy(a)
    assert not bool(m.any())
    assert float(a[0].max() - a[0].min()) < 1e-3
    assert float(a[4].max() - a[4].min()) < 1.0          # 88S


def test_soil_fill_equatorward_of_60_is_the_8_neighbour_ring_bitwise():
    """A box away from the poles gets the fill it had: every row of a source band within 60 degrees reaches one
    column, and the result is bitwise the fill without a grid."""
    rng = np.random.default_rng(3)
    dlat = 0.25
    lat0 = 30.0
    ny, nx = int(round((59.75 - lat0) / dlat)) + 1, 1440
    grid = hin.Grid(lat0=lat0, lon0=0.0, dlat=dlat, dlon=dlat, lon_periodic=True)
    assert int(gs.ring_reach(grid, ny, nx).max()) == 1
    land = np.zeros((ny, nx), bool)
    land[20:60, 100:400] = True
    land[90:94, 900:905] = True
    a = np.where(land, rng.normal(280.0, 10.0, (ny, nx)), MISSING).astype(np.float32)
    (g, gm), = gs._ring_fill_many([cp.asarray(a)], [cp.asarray(~land)], lon_periodic=True, grid=grid)
    (n, nm), = gs._ring_fill_many([cp.asarray(a)], [cp.asarray(~land)], lon_periodic=True)
    assert cp.array_equal(gm, nm)
    assert cp.array_equal(g.view(cp.uint32), n.view(cp.uint32))


def test_ring_reach():
    g = hin.Grid(lat0=-90.0, lon0=0.0, dlat=0.25, dlon=0.25, lon_periodic=True)
    k = gs.ring_reach(g, 721, 1440)
    lat = -90.0 + 0.25 * np.arange(721)
    assert (k[np.abs(lat) <= 60.0] == 1).all()
    assert k[0] == k[-1] == 1440                         # the pole rows: the whole circle
    assert k[lat == 80.0][0] == 5 and k[lat == 89.75][0] == 229
    proj = hin.Grid(lat0=0.0, lon0=0.0, dlat=1.0, dlon=1.0, lon_periodic=False, proj=object())
    assert (gs.ring_reach(proj, 50, 60, False) == 1).all()
