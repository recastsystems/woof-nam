"""GPU: an isobaric start's sea-ice cells take Noah's sea-ice column, not soil filled in from distant land.

On the 88.9N IFS 0.1 box (2026-11-13) the sea ice started from the source's land soil carried over the Arctic Ocean
from the coasts, as rays from the pole, and 2 m temperature over the ice showed them. The ice layers now follow the
source's skin temperature down to 271.16 K (Noah's sea-ice bottom) under 1.5 m of ice, so ice cells under the same skin
start alike, whatever coast is nearest.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)

from namgpu.init import gfs_start as gs  # noqa: E402


def _raw(jm=6, im=8):
    rng = np.random.default_rng(5)
    tsoil = rng.uniform(245.0, 275.0, (4, jm, im)).astype(np.float32)   # land soil filled out over the sea: rays
    land = np.zeros((jm, im), np.float32)
    land[:, :2] = 1.0                                                     # a coast
    icec = np.zeros((jm, im), np.float32)
    icec[:, 3:] = 0.9                                                     # pack ice
    icec[0, 0] = 0.9                                                      # ice flag on a land cell: land stays land
    icec[1, 2] = 0.4                                                      # thin ice: open water in Noah
    tsk = np.full((jm, im), 250.0, np.float32)
    tsk[:, -1] = 274.0                                                    # a skin above melting: the ice is at 273.15
    return {"TSOIL": cp.asarray(tsoil), "TSOIL_3M": cp.asarray(tsoil[3].copy()), "TMP_SFC": cp.asarray(tsk),
            "ICEC": cp.asarray(icec), "LAND": cp.asarray(land)}


def test_sea_ice_cells_take_the_ice_column():
    raw = _raw()
    before = {k: cp.asnumpy(v).copy() for k, v in raw.items()}
    n = gs.sea_ice_column(raw)
    ice = (before["ICEC"] > 0.5) & (before["LAND"] < 0.5)
    assert n == int(ice.sum()) == 6 * 5
    ts = np.minimum(before["TMP_SFC"], 273.15)
    tsoil = cp.asnumpy(raw["TSOIL"])
    for k, f in enumerate((0.25, 0.75, 1.0, 1.0)):      # layer centres 0.375 ... 2.625 m; 1.5 m of ice, water below
        want = ts + (271.16 - ts) * f
        assert np.allclose(tsoil[k][ice], want[ice], atol=1e-4), k
        assert np.array_equal(tsoil[k][~ice], before["TSOIL"][k][~ice])
    tg = cp.asnumpy(raw["TSOIL_3M"])
    assert np.allclose(tg[ice], 271.16) and np.array_equal(tg[~ice], before["TSOIL_3M"][~ice])
    # one skin, one column: the ice no longer carries the land soil's spread
    assert float(np.ptp(tsoil[0][:, 3:-1])) < 1e-4


def test_no_sea_ice_changes_nothing():
    raw = _raw()
    raw["ICEC"][...] = 0.0
    before = {k: v.copy() for k, v in raw.items()}
    assert gs.sea_ice_column(raw) == 0
    for k in raw:
        assert cp.array_equal(raw[k], before[k])
