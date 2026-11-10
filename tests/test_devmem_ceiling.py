"""devmem: init runs under the card's free memory (CUPY_GPU_MEMORY_LIMIT), the step loop under the pre-flight's cap
(NAMGPU_FORECAST_POOL_LIMIT) from forecast_start on. A nest from a parent's history builds its boundaries in init,
past the forecast's estimate (CONUS 3 km on one card, init-gfs-bocos at 66.4 of 66.6 GiB, suite proof 2026-11-06)."""
import os
import subprocess
import sys

import pytest

CODE = r"""
import cupy as cp
from namgpu import devmem
devmem.setup()
MB = 1 << 20
a = cp.empty(150 * MB, dtype=cp.uint8)          # init: past the forecast's 100 MB, inside the 200 MB init ceiling
try:
    cp.empty(100 * MB, dtype=cp.uint8)          # 250 MB > 200 MB
    print("init-over-ok")
except cp.cuda.memory.OutOfMemoryError:
    print("init-over-refused")
del a
devmem.forecast_start()
b = cp.empty(60 * MB, dtype=cp.uint8)
try:
    c = cp.empty(60 * MB, dtype=cp.uint8)       # 120 MB > the forecast's 100 MB
    print("forecast-over-ok")
except cp.cuda.memory.OutOfMemoryError:
    print("forecast-over-refused")
"""


def test_init_ceiling_then_forecast_cap():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no card")
    except Exception:  # noqa: BLE001
        pytest.skip("no card")
    env = dict(os.environ, CUPY_GPU_MEMORY_LIMIT=str(200 << 20), NAMGPU_FORECAST_POOL_LIMIT=str(100 << 20),
               NAMGPU_POOL="cupy", NAMGPU_POOL_CAP="all")
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["init-over-refused", "forecast-over-refused"], out.stdout
