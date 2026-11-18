"""The wind-rotation coefficients reach the device fresh on every call; no device copy lives from one stage to the next.

Breakage it prevents: the HRRR 15Z start of 2026-11-12 on the 12 km Kansas box (woof-nam nest --center 38.5,-97.5
--size-km 1500 --res-km 12 --source hrrr --at 2026111215) wrote V boundary strips that are exactly
u_e cn_e + v_e x 20000 at every strip point of all three boundary times (U right; V to -1.07e6 m/s): the one device
copy of cn_n cached on the grid held 20000.0 for the rest of that process, and every later start of that case read
the files and went to NaN at step 2. CPU only: numpy stands in for cupy."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import grid as gridmod  # noqa: E402
from namgpu.init import init_gfs_wind_rotate as wr  # noqa: E402


def _grid():
    cfg = SimpleNamespace(im=12, jm=10, lm=2, dlmd=0.108, dphd=0.108, ref_wbd=-0.594, ref_sbd=-0.486,
                          tph0d=38.5, tlm0d=-97.5, win_i0=0, win_j0=0)
    try:
        return gridmod.build(cfg)
    except (AttributeError, TypeError):
        return SimpleNamespace(im=12, jm=10, wb=np.radians(-0.594), sb=np.radians(-0.486), dlm=np.radians(0.108),
                               dph=np.radians(0.108), tlm0=np.radians(-97.5), tph0=np.radians(38.5), win_i0=0,
                               win_j0=0, joff=0)


def test_a_value_written_into_one_calls_coefficients_does_not_reach_the_next(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", np)
    ctx = SimpleNamespace(grid=_grid())
    first = wr._coefficients(ctx)
    want = [a.copy() for a in first]
    first[3][...] = 20000.0                       # what the 12 Nov process's cached cn_n held
    again = wr._coefficients(ctx)
    for a, b in zip(again, want):
        assert np.array_equal(a, b), "a device coefficient array carried a value from one call into the next"
    assert np.allclose(again[2] ** 2 + again[3] ** 2, 1.0, atol=1e-5)
