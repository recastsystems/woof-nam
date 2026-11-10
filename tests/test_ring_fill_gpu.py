"""GPU: the batched ring fill is bitwise the per-field ring fill (global/periodic and window/regional grids)."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)

from namgpu.init import gfs_start as gs  # noqa: E402


@pytest.mark.parametrize("periodic", [True, False])
def test_ring_fill_many_is_the_loop(periodic):
    rng = np.random.default_rng(7)
    ny, nx = 211, 307
    land = np.zeros((ny, nx), bool)
    land[60:90, 40:120] = True                     # a continent far from the corners (hundreds of rings)
    land[150:153, 250:254] = True                  # an island
    arrs, miss = [], []
    for k in range(5):
        a = rng.normal(280.0, 15.0, (ny, nx)).astype(np.float32)
        m = ~land | (rng.random((ny, nx)) < 0.05 * k)
        a[m] = -9999.0
        arrs.append(cp.asarray(a))
        miss.append(cp.asarray(m))
    miss[4] = cp.ones((ny, nx), bool)              # nothing present: stays missing
    many = gs._ring_fill_many(arrs, miss, lon_periodic=periodic)
    for k in range(5):
        a0, m0 = gs._ring_fill(arrs[k], miss[k], lon_periodic=periodic)
        assert cp.array_equal(m0, many[k][1])
        assert cp.array_equal(a0.view(cp.uint32), many[k][0].view(cp.uint32)), k
