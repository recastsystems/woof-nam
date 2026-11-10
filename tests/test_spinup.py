"""CPU checks of the spin-up start (namgpu/spinup.py): the IAU weight schedule, the nudging window, the increment and
its application on small shapes, the plan's refusals and the spin-up window's input files. No GPU, no network.

Run: python tests/test_spinup.py   (or pytest). The model run itself is a real smoke (README "Spin-up start").
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import spinup  # noqa: E402

T0 = datetime(2024, 8, 25, 18)
DT = 25.0


def _plan(**kw):
    args = dict(t0=T0, hours=6, update="iau", iau_hours=None, nudge_hours=1.0, nudge_tau_s=3600.0, source="gfs",
                main_age_h=6)
    args.update(kw)
    return spinup.make_plan(**args)


def test_iau_weights_sum_to_one_over_centred_window():
    n = spinup.window_steps(6.0, DT)
    assert n == 864
    ks = range(-2000, 2000)
    w = np.array([spinup.iau_weight(k, n) for k in ks])
    on = [k for k, x in zip(ks, w) if x]
    assert len(on) == n
    assert on[0] == -431 and on[-1] == 432           # steps ending in (T0-3h, T0+3h]
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.allclose(w[w > 0], 1.0 / n)
    # half of it is in by T0 (the forecast's 0 h), the rest over the first 3 h
    assert abs(sum(spinup.iau_weight(k, n) for k in range(-431, 1)) - 0.5) < 1e-12


def test_iau_odd_and_short_windows():
    for hours in (1.0, 2.5, 3.0):
        n = spinup.window_steps(hours, DT)
        s = sum(spinup.iau_weight(k, n) for k in range(-n, n + 1))
        assert abs(s - 1.0) < 1e-12, (hours, s)


def test_step_k_across_the_rebase():
    s0 = spinup.make_plan(T0, 6, "iau", None, 1.0, 3600.0, "gfs", 6).spin_start
    assert spinup.step_k(s0, 0, T0, DT) == -864             # spin-up start
    assert spinup.step_k(s0, 432, T0, DT) == -432           # the rewind point T0-3 h
    assert spinup.step_k(s0, 864, T0, DT) == 0              # T0 before the re-base ...
    assert spinup.step_k(T0, 0, T0, DT) == 0                # ... and after it: the same step
    assert spinup.step_k(T0, 432, T0, DT) == 432            # T0+3 h, the last IAU step


def test_nudge_window_and_relaxation():
    n = spinup.window_steps(1.0, DT)
    assert n == 144
    on = [k for k in range(-500, 10) if spinup.nudge_coef(k, n, DT, 3600.0)]
    assert on[0] == -143 and on[-1] == 0 and len(on) == n
    # the departure left after the hour: (1 - dt/tau)^n ~ e^-1
    x, a = np.float64(0.0), np.float64(1.0)
    for k in on:
        x += spinup.nudge_coef(k, n, DT, 3600.0) * (a - x)
    assert abs((a - x) - (1 - DT / 3600.0) ** n) < 1e-12 and 0.36 < a - x < 0.37


def _state(rng, lm=4, jm=5, im=6):
    st = {n: rng.random((lm, jm, im), dtype=np.float32) for n in ("T", "Q", "U", "V", "TP", "UP", "VP", "CW", "W")}
    st["Q"] *= np.float32(0.01)
    st["PD"] = (90000.0 + 1000.0 * rng.random((jm, im))).astype(np.float32)
    sg2 = np.linspace(0.0, 1.0, lm + 1).astype(np.float32)
    st["PINT"] = (200.0 + sg2[:, None, None] * st["PD"][None]).astype(np.float32)
    return st, sg2


def test_increment_and_full_application():
    rng = np.random.default_rng(1)
    st, sg2 = _state(rng)
    an = {n: (st[n] + rng.standard_normal(st[n].shape).astype(np.float32)) for n in spinup.FIELDS}
    an["Q"] = np.abs(an["Q"]) * np.float32(0.01)
    an["PD"] = st["PD"] + np.float32(300.0) * rng.standard_normal(st["PD"].shape).astype(np.float32)
    inc = spinup.increment(an, st)
    assert set(inc) == set(spinup.FIELDS) and all(v.dtype == np.float32 for v in inc.values())
    keep = {n: st[n].copy() for n in ("CW", "W")}
    pint0, tp0, pd0 = st["PINT"].copy(), st["TP"].copy(), st["PD"].copy()
    n = 864
    for k in range(-431, 433):
        spinup.apply_increment(st, inc, spinup.iau_weight(k, n), sg2)
    for f in ("T", "U", "V", "PD"):
        assert np.allclose(st[f], an[f], rtol=1e-4, atol=1e-3), f     # float32 sum of 864 parts
    assert np.allclose(st["Q"], np.maximum(an["Q"], spinup.EPSQ), atol=1e-6)
    for f in ("CW", "W"):                                   # hydrometeors and W are never touched
        assert np.array_equal(st[f], keep[f]), f
    assert np.allclose(st["TP"] - tp0, inc["T"], atol=1e-3)     # AB history moves with T
    # PINT moved by its hydrostatic part only: sg2 * dPD at every interface; the top (sg2 = 0) stays
    assert np.allclose(st["PINT"] - pint0, sg2[:, None, None] * (st["PD"] - pd0)[None], atol=10.0)
    assert np.array_equal(st["PINT"][0], pint0[0])


def test_q_floor():
    rng = np.random.default_rng(2)
    st, sg2 = _state(rng)
    inc = {n: np.zeros_like(st[n]) for n in spinup.FIELDS}
    inc["Q"] = -np.ones_like(st["Q"])
    spinup.apply_increment(st, inc, 1.0, sg2)
    assert float(st["Q"].min()) == float(spinup.EPSQ)


def test_nudge_moves_toward_the_analysis():
    rng = np.random.default_rng(3)
    st, sg2 = _state(rng)
    an = {n: st[n] + np.float32(1.0) for n in spinup.FIELDS}
    d0 = {n: float(np.abs(an[n] - st[n]).mean()) for n in spinup.FIELDS}
    spinup.apply_nudge(st, an, 0.5, sg2)
    for n in ("T", "U", "V", "PD"):
        assert abs(float(np.abs(an[n] - st[n]).mean()) - 0.5 * d0[n]) < 1e-3, n


def test_plan_and_refusals():
    p = _plan()
    assert p.spin_start == datetime(2024, 8, 25, 12) and p.iau_hours == 6.0 and p.spin_age_h == 0
    assert spinup.Plan.from_meta(p.meta()) == p
    assert _plan(main_age_h=12).spin_age_h == 6 and _plan(main_age_h=0).spin_age_h == 0
    for bad in (dict(hours=3), dict(hours=0), dict(update="3dvar"), dict(iau_hours=13.0), dict(nudge_hours=7.0),
                dict(nudge_tau_s=0.0)):
        try:
            _plan(**bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad} must be refused")


def test_window_inputs_are_the_forecast_boundary_cycle():
    from namgpu.init import sources
    row = sources.SOURCES["gfs"]
    p = _plan()
    files = spinup.input_files(Path("case"), p, row)
    names = [f.name for f in files]
    assert names[:2] == ["gfs.t12z.pgrb2.0p25.f000", "gfs.t18z.pgrb2.0p25.f000"]
    assert names[2:] == ["gfs.t12z.f000.grib2", "gfs.t12z.f003.grib2", "gfs.t12z.f006.grib2"]
    # a 00Z cycle spins up from the previous day's 18Z
    p0 = _plan(t0=datetime(2023, 7, 29, 0))
    assert [f.name for f in spinup.input_files(Path("c"), p0, row)][0] == "gfs.t18z.pgrb2.0p25.f000"


def test_fetch_plan_matches_the_window_inputs():
    """fetch_case's spin-up rows plan exactly the files the spin-up reads (gfs: T0-6 analysis, leads 0/3/6)."""
    from namgpu import config
    from namgpu.init import fetch_case, sources
    p = _plan()
    raw = Path("case") / "raw"
    rows = tuple(r for r in fetch_case.FETCH_ROWS if r["source"] == "gfs" and r["role"] in ("init", "boco"))
    jobs, _ = fetch_case.plan(p.spin_start, p.hours, raw, config.load(), "gfs", p.spin_age_h, rows)
    planned = sorted(str(j[2]) for j in jobs)
    want = sorted(str(f) for f in spinup.input_files(Path("case"), p, sources.SOURCES["gfs"])
                  if "t18z" not in f.name)                  # the cycle's own analysis comes with the cycle's fetch
    assert planned == want, (planned, want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
