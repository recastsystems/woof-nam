"""Poisson's equation for the history T2, the start's 2 m state and the frame's skin temperature (CPU).

namgpu/io/poisson.py: T = theta (p / 1e5) ** (R_d / c_p), R_d = 287.04, c_p = 1004. The NCEP points below are read
from NOAA's native-grid NAM surface file of 2024-11-05 12Z, +0 h (nam.t12z.bgrdsf00.tm00.grib2): surface potential
temperature (POT 0.0.2, level 1), surface pressure (PRES 0.3.0, level 1) and surface temperature (TMP 0.0.0,
level 1), with (row, column) on the 954 x 835 grid. The file packs TMP to 0.01 K and POT to 0.002 K.
"""
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from namgpu.init import sources
from namgpu.io import poisson
from namgpu.io import products

NCEP_SFC = [  # (row, column, POT K, PRES Pa, TMP K)
    (74, 824, 321.57223, 64205.263, 283.31),
    (691, 627, 267.73423, 69904.463, 241.68),
    (673, 565, 254.90223, 75866.863, 235.55),
    (102, 873, 311.18223, 82001.263, 294.02),
    (584, 361, 261.86223, 91799.663, 255.53),
    (704, 115, 265.98823, 99542.062, 265.64),
    (214, 875, 301.56223, 101334.86, 302.71),
    (737, 393, 245.03223, 103450.06, 247.42),
]
PACKING = 0.0065   # half the TMP step plus half the POT step, with float32 rounding


def test_kappa_is_r_over_cp():
    assert poisson.KAPPA == pytest.approx(287.04 / 1004.0, rel=1e-12)
    assert poisson.P0 == 1.0e5


def test_temperature_matches_ncep_surface_records():
    th = np.array([r[2] for r in NCEP_SFC], np.float32)
    p = np.array([r[3] for r in NCEP_SFC], np.float32)
    t = np.array([r[4] for r in NCEP_SFC], np.float32)
    got = poisson.temperature(th, p)
    assert got.dtype == np.float32
    assert np.max(np.abs(got - t)) <= PACKING
    # the 1004.6 exponent misses the high-terrain points by more than the packing
    k_alt = 287.04 / 1004.6
    alt = th * (p / 1.0e5) ** k_alt
    assert np.max(np.abs(alt - t)) > 0.015


def test_theta_inverts_temperature():
    rng = np.random.default_rng(1)
    t = rng.uniform(220.0, 320.0, 1000).astype(np.float32)
    p = rng.uniform(5.0e4, 1.05e5, 1000).astype(np.float32)
    back = poisson.temperature(poisson.theta(t, p), p)
    assert np.max(np.abs(back - t)) < 1.0e-4


def test_pressure_2m_is_hypsometric():
    ps = np.array([1.0e5, 7.0e4], np.float32)
    t2 = np.array([288.15, 260.0], np.float32)
    p2 = poisson.pressure_2m(ps, t2)
    want = ps * np.exp(-2.0 * 9.8060226 / (287.04 * t2.astype(np.float64)))
    assert np.allclose(p2, want, rtol=1e-6)
    assert 23.0 < float(ps[0] - p2[0]) < 24.5


class _State:
    def __init__(self):
        self.fields = {}

    def __getitem__(self, k):
        return self.fields[k]

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = np.zeros(shape, dtype=dtype)
        return self.fields[name]


def _ctx(jm=3, im=4, psfc=9.0e4):
    st = _State()
    st.add("PSFC", (jm, im))[...] = psfc
    return SimpleNamespace(st=st, grid=SimpleNamespace(im=im, jm=jm))


def test_shelter_state_round_trips_t2_with_analysis_pressure():
    ctx = _ctx()
    t2 = np.full((3, 4), 280.0, np.float32)
    q2 = np.full((3, 4), 0.004, np.float32)
    u = np.full((3, 4), 3.0, np.float32)
    v = np.full((3, 4), -2.0, np.float32)
    p2 = np.full((3, 4), 8.9975e4, np.float32)
    p2[0, 0] = 0.0      # a missing value in the file
    th10 = np.full((3, 4), 282.0, np.float32)
    q10 = np.full((3, 4), 0.0039, np.float32)
    sources.shelter_state(ctx, t2, q2, u, v, p2=p2, th10=th10, q10=q10)
    f = ctx.st.fields
    assert f["PSHLTR"][1, 1] == np.float32(8.9975e4)
    assert f["PSHLTR"][0, 0] == poisson.pressure_2m(np.float32(9.0e4), np.float32(280.0))
    assert np.allclose(f["TSHLTR"], 280.0 * (1.0e5 / f["PSHLTR"]) ** poisson.KAPPA, rtol=1e-6)
    # the history T2 of the start is the analysis t2
    assert np.max(np.abs(poisson.temperature(f["TSHLTR"], f["PSHLTR"]) - t2)) < 1.0e-4
    assert np.all(f["QSHLTR"] == q2) and np.all(f["U10"] == u) and np.all(f["V10"] == v)
    assert np.all(f["TH10"] == th10) and np.all(f["Q10"] == q10)
    assert all(f[k].dtype == np.float32 for k in ("PSHLTR", "TSHLTR", "QSHLTR", "U10", "V10", "TH10", "Q10"))


def test_shelter_state_without_2m_pressure_or_10m_records():
    ctx = _ctx(psfc=1.0e5)
    t2 = np.full((3, 4), 288.15, np.float32)
    q2 = np.full((3, 4), 0.008, np.float32)
    z = np.zeros((3, 4), np.float32)
    sources.shelter_state(ctx, t2, q2, z, z)
    f = ctx.st.fields
    assert np.allclose(f["PSHLTR"], poisson.pressure_2m(np.float32(1.0e5), np.float32(288.15)))
    assert np.all(f["TH10"] == f["TSHLTR"]) and np.all(f["Q10"] == q2)


def test_shelter_state_needs_psfc():
    ctx = SimpleNamespace(st=_State(), grid=SimpleNamespace(im=2, jm=2))
    a = np.ones((2, 2), np.float32)
    with pytest.raises(RuntimeError, match="PSFC"):
        sources.shelter_state(ctx, 280 * a, 0 * a, a, a)


def _history(path, names):
    with netCDF4.Dataset(path, "w") as s:
        s.createDimension("Time", 1)
        s.createDimension("south_north", 2)
        s.createDimension("west_east", len(NCEP_SFC) // 2)
        for n, val in names.items():
            v = s.createVariable(n, "f4", ("Time", "south_north", "west_east"))
            v[0] = val


def test_frame_tsk_from_ths_and_psfc(tmp_path):
    th = np.array([r[2] for r in NCEP_SFC], np.float32).reshape(2, -1)
    p = np.array([r[3] for r in NCEP_SFC], np.float32).reshape(2, -1)
    t = np.array([r[4] for r in NCEP_SFC], np.float32).reshape(2, -1)
    path = tmp_path / "wrfout_d01"
    _history(path, {"THS": th, "PSFC": p})
    with netCDF4.Dataset(path) as s:
        tsk = products.DERIVED["TSK"](s)
    assert tsk.shape == (2, 4) and tsk.dtype == np.float32
    assert np.max(np.abs(tsk - t)) <= PACKING


def test_frame_tsk_none_without_inputs(tmp_path):
    path = tmp_path / "wrfout_d01"
    _history(path, {"THS": np.full((2, 4), 290.0, np.float32)})
    with netCDF4.Dataset(path) as s:
        assert products.DERIVED["TSK"](s) is None
