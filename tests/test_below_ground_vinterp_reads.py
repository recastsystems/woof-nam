"""CPU check: init-gfs-vinterp interpolates temperature and humidity from TT_LEV_IN when the start built one (no GPU).

Run: python tests/test_below_ground_vinterp_reads.py   (or pytest). Breakage it prevents: tests/test_below_ground_t.py
checks gfs_start.below_ground_from_2m and the record rows, not that the interpolation reads the result. If
init_gfs_vinterp went back to reading TT_IN for temperature or RH-to-q, that file would still pass and an ERA5 or IFS
start would again be up to 20 K warm on its lowest levels under an arctic inversion; only a real run would show it.
This runs init_gfs_vinterp.run on a 3 x 2 grid with numpy standing in for cupy and every kernel launch recorded, and
checks which arrays the launches were given: T and RH-to-q read TT_LEV_IN, the surface-pressure reduction reads TT_IN,
TT_LEV_IN is reversed with the other inputs for a top-down source, and a source with no TT_LEV_IN (NCEP) reads TT_IN.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from namgpu import kernels  # noqa: E402
from namgpu.init import gfs_start, init_gfs_vinterp as vi  # noqa: E402

IM, JM, LM = 3, 2, 4
# level 0 = the surface (972.8 hPa, 225 m, 2 m T -57.6 C: ERA5's Baffin column, test_below_ground_t.py), then 1000 hPa
# up to 750 hPa, ground up; 1000 and 975 hPa are under the ground
PRES = np.array([97280.0, 100000.0, 97500.0, 95000.0, 92500.0, 90000.0, 87500.0, 85000.0, 82500.0, 80000.0, 77500.0,
                 75000.0], np.float32)
GHT = np.array([225.0, 25.0, 207.0, 386.0, 572.0, 765.0, 965.0, 1170.0, 1385.0, 1605.0, 1830.0, 2065.0], np.float32)
TT = np.array([-57.6, -34.4, -36.2, -35.0, -34.0, -33.5, -33.8, -34.6, -35.9, -37.4, -39.0, -40.8], np.float32) + 273.15


class _NumpyAsCupy(types.ModuleType):
    """`import cupy as cp` inside run() gets numpy: the wrapper's array calls run on the host."""

    def __getattr__(self, name):
        return getattr(np, name)


class _State:
    def __init__(self):
        self.fields = {}

    def __getitem__(self, name):
        return self.fields[name]

    def add(self, name, shape):
        if name not in self.fields:
            self.fields[name] = np.zeros(shape, np.float32)
        return self.fields[name]


def _columns(n):
    return np.ascontiguousarray(np.broadcast_to(n[:, None, None], (len(n), JM, IM)), dtype=np.float32)


def _run(top_down=False, with_lev=True):
    """init_gfs_vinterp.run on PRES/GHT/TT in every column; returns (launches, gi after the run, TT_LEV_IN given)."""
    pres, ght, tt = _columns(PRES), _columns(GHT), _columns(TT)
    lev = gfs_start.below_ground_from_2m(pres, ght, tt)
    assert not np.array_equal(lev, tt)                         # the column has levels under the ground
    gi = {"PRES_IN": pres, "GHT_IN": ght, "TT_IN": tt}
    for k in ("RH_IN", "UU_IN", "VV_IN", "SNMR_IN", "CLWMR_IN", "RWMR_IN", "CICE_IN", "FRIMEF_IN"):
        gi[k] = np.full((len(PRES), JM, IM), 0.5, np.float32)
    if with_lev:
        gi["TT_LEV_IN"] = lev
    if top_down:                                               # a source written top down: run() reverses its input
        for k in vi.REVERSED_KEYS + (("TT_LEV_IN",) if with_lev else ()):
            gi[k] = np.ascontiguousarray(gi[k][::-1])
    given = gi.get("TT_LEV_IN")

    st = _State()
    st.add("HGT_M", (JM, IM))[...] = 225.0
    st.add("SOILHGT", (JM, IM))[...] = 225.0
    for k, shape in (("PD", (JM, IM)), ("PINT", (LM + 1, JM, IM)), ("T", (LM, JM, IM)), ("Q", (LM, JM, IM)),
                     ("CW", (LM, JM, IM)), ("U", (LM, JM, IM)), ("V", (LM, JM, IM))):
        st.add(k, shape)
    lm1 = np.linspace(0.0, 1.0, LM + 1, dtype=np.float32)
    grid = types.SimpleNamespace(im=IM, jm=JM, lm=LM, pdtop=np.float32(30000.0), pt=np.float32(5000.0),
                                 sg1=lm1, sg2=lm1[::-1].copy(), sgml1=lm1[:LM], sgml2=lm1[1:])
    cfg = types.SimpleNamespace(raw={}, do_clouds=False, lnsh=1, lnsv=1)
    ctx = types.SimpleNamespace(grid=grid, cfg=cfg, st=st, tables={"gfs_hinterp": gi}, first_step=True, stream=None)

    launches = []

    def get(unit, name):
        assert unit == vi.UNIT
        return lambda grid_, block, args: launches.append((name, args))

    saved_cp, saved_get = sys.modules.get("cupy"), kernels.get
    sys.modules["cupy"] = _NumpyAsCupy("cupy")
    kernels.get = get
    try:
        vi.run(ctx)
    finally:
        kernels.get = saved_get
        if saved_cp is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = saved_cp
    return launches, gi, given, st


def _temperature_args(launches, st):
    """The arrays given to the temperature interpolation (the lin launch that writes st T) and to RH-to-q."""
    t = [a for n, a in launches if n == "init_gfs_vinterp_lin" and a[3] is st["T"]]
    q = [a for n, a in launches if n == "init_gfs_vinterp_rhq"]
    s = [a for n, a in launches if n == "init_gfs_vinterp_surfp"]
    assert len(t) == 1 and len(q) == 1 and len(s) == 1, [n for n, _ in launches]
    return t[0][2], q[0][1], s[0][3]


def test_interpolation_reads_tt_lev_in():
    launches, gi, given, st = _run()
    t_in, q_in, surf_in = _temperature_args(launches, st)
    lev = gfs_start.below_ground_from_2m(_columns(PRES), _columns(GHT), _columns(TT))
    # temperature and RH-to-q read the under-ground levels the start built, not the source column
    assert t_in is given and np.array_equal(t_in, lev), "init-gfs-vinterp T does not read TT_LEV_IN"
    assert q_in is given, "init-gfs-vinterp RH-to-q does not read TT_LEV_IN"
    assert not np.array_equal(t_in, gi["TT_IN"])
    # the surface-pressure reduction keeps the source column
    assert surf_in is gi["TT_IN"] and np.array_equal(surf_in, _columns(TT))


def test_top_down_source_reverses_tt_lev_in_too():
    launches, gi, given, st = _run(top_down=True)
    t_in, q_in, surf_in = _temperature_args(launches, st)
    lev = gfs_start.below_ground_from_2m(_columns(PRES), _columns(GHT), _columns(TT))
    # after run()'s reversal, TT_LEV_IN is in the same (ground-up) order as TT_IN, PRES_IN and GHT_IN
    assert np.array_equal(gi["PRES_IN"], _columns(PRES)) and np.array_equal(gi["TT_IN"], _columns(TT))
    assert t_in is gi["TT_LEV_IN"] and q_in is gi["TT_LEV_IN"], "init-gfs-vinterp does not read TT_LEV_IN"
    assert np.array_equal(t_in, lev), "TT_LEV_IN was not reversed with the other inputs"
    assert surf_in is gi["TT_IN"]


def test_source_without_tt_lev_in_reads_tt_in():
    launches, gi, given, st = _run(with_lev=False)              # NCEP rows: no TT_LEV_IN, as before
    t_in, q_in, surf_in = _temperature_args(launches, st)
    assert given is None and "TT_LEV_IN" not in gi
    assert t_in is gi["TT_IN"] and q_in is gi["TT_IN"] and surf_in is gi["TT_IN"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
