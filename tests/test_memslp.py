"""MSLET (namgpu.products.memslp) on small synthetic histories, on the CPU: the inputs it needs, sea-level points,
a standard-lapse-rate atmosphere over a plateau (closed form), the relaxation on a harmonic field, and the
hydrostatic step."""
import numpy as np
import pytest

from namgpu.products import memslp as M

netCDF4 = pytest.importorskip("netCDF4")

JM, IM, LM = 14, 18, 12


def _lapse_column(ps, t_sfc, lm=LM, top=20000.0):
    """Interface and mid-layer pressures from ps up to `top`, and a dry atmosphere with the standard lapse rate:
    T(p) = t_sfc (p / ps)^(Rd gamma / g)."""
    pint = np.linspace(ps, top, lm + 1)
    pmid = np.sqrt(pint[:-1] * pint[1:])
    t = t_sfc * (pmid / ps) ** (M.RD * M.GAMMA / M.G)
    return pint, pmid, t


def _write(path, hgt, ps, t_sfc, drop=None):
    """A history with the variables memslp reads, the standard-lapse dry atmosphere in every column."""
    pint = np.empty((LM + 1, JM, IM), np.float32)
    pmid = np.empty((LM, JM, IM), np.float32)
    t = np.empty((LM, JM, IM), np.float32)
    for j in range(JM):
        for i in range(IM):
            a, b, c = _lapse_column(float(ps[j, i]), float(t_sfc[j, i]))
            pint[:, j, i], pmid[:, j, i], t[:, j, i] = a, b, c
    theta = t * (M.P00 / pmid) ** M.CAPPA - 300.0
    with netCDF4.Dataset(path, "w") as s:
        s.createDimension("Time", None)
        s.createDimension("bottom_top", LM)
        s.createDimension("bottom_top_stag", LM + 1)
        s.createDimension("south_north", JM)
        s.createDimension("west_east", IM)
        d3, d3s, d2 = ("Time", "bottom_top", "south_north", "west_east"), \
            ("Time", "bottom_top_stag", "south_north", "west_east"), ("Time", "south_north", "west_east")
        fields = {"T": (theta, d3), "QVAPOR": (np.zeros_like(theta), d3), "PRES": (pmid, d3),
                  "PINT": (pint, d3s), "HGT": (hgt.astype(np.float32), d2)}
        for name, (a, dims) in fields.items():
            if name == drop:
                continue
            s.createVariable(name, "f4", dims)[0] = a
    return path


def _slp_lapse(ps, t_sfc, h):
    """Closed form for the standard-lapse atmosphere: the ground at height h, T at sea level t_sfc + gamma h."""
    return ps * ((t_sfc + M.GAMMA * h) / t_sfc) ** (M.G / (M.RD * M.GAMMA))


@pytest.mark.parametrize("name", M.NEEDS)
def test_a_history_without_an_input_gives_none(tmp_path, name):
    hgt = np.zeros((JM, IM))
    p = _write(tmp_path / "h.nc", hgt, np.full((JM, IM), 101000.0), np.full((JM, IM), 288.0), drop=name)
    with netCDF4.Dataset(p) as s:
        assert M.from_history(s, xp=np) is None


def test_sea_level_ground_gives_the_surface_pressure(tmp_path):
    ps = 100000.0 + 50.0 * np.arange(IM)[None, :] + np.zeros((JM, 1))
    p = _write(tmp_path / "h.nc", np.zeros((JM, IM)), ps, np.full((JM, IM), 285.0))
    with netCDF4.Dataset(p) as s:
        a = M.from_history(s, xp=np)
    assert a.dtype == np.float32 and a.shape == (JM, IM)
    np.testing.assert_allclose(a, ps, rtol=0, atol=0.5)


def test_plateau_in_a_standard_lapse_atmosphere_matches_the_closed_form(tmp_path):
    # a 1500 m plateau in the middle, sea level around it; the same standard-lapse air everywhere, so the membrane
    # temperatures under the plateau are the surrounding air's and the reduction is the closed form
    t0, p0 = 288.0, 101325.0
    hgt = np.zeros((JM, IM))
    hgt[4:10, 5:13] = 1500.0
    ps = p0 * (1.0 - M.GAMMA * hgt / t0) ** (M.G / (M.RD * M.GAMMA))
    t_sfc = t0 - M.GAMMA * hgt
    p = _write(tmp_path / "h.nc", hgt, ps, t_sfc)
    with netCDF4.Dataset(p) as s:
        a = M.from_history(s, xp=np).astype(np.float64)
    expect = _slp_lapse(ps, t_sfc, hgt)
    np.testing.assert_allclose(expect, p0, rtol=1e-9)
    assert np.abs(a - expect).max() < 15.0           # Pa: 25 hPa layers, trapezoidal layer means
    assert np.abs(a - expect)[hgt == 0].max() < 0.5


def test_relaxation_keeps_a_harmonic_field_and_only_changes_below_ground():
    j, i = np.mgrid[0:JM, 0:IM].astype(np.float64)
    truth = 250.0 + 0.3 * i - 0.2 * j                 # linear: harmonic for the nine-point stencil
    under = np.zeros((1, JM, IM), bool)
    under[0, 5:9, 6:11] = True
    field = truth[None].copy()
    field[under] = 200.0                               # a poor first guess
    out = M.relax(np, field, under, 400)
    np.testing.assert_allclose(out[0], truth, atol=1e-6)
    keep = truth[None].copy()
    keep[under] = 200.0
    np.testing.assert_array_equal(M.relax(np, keep, under, 0), keep)


def test_relaxation_at_the_domain_edge_uses_the_neighbours_that_exist():
    under = np.zeros((1, 6, 6), bool)
    under[0, 0, 0] = True
    field = np.full((1, 6, 6), 280.0)
    field[0, 0, 0] = 0.0
    out = M.relax(np, field, under, 1)
    assert out[0, 0, 0] == pytest.approx(280.0)


def test_integration_through_constant_tv_is_the_hypsometric_equation():
    tv = 270.0
    levels = [80000.0, 85000.0, 90000.0, 95000.0, 100000.0]
    z0 = np.array([[1200.0, 200.0, 0.0, -20.0]])
    ps = np.array([[78000.0, 97000.0, 101000.0, 101300.0]])
    under = np.stack([lv > ps for lv in levels])
    tvl = np.full((len(levels),) + ps.shape, tv)
    lnp = M.integrate(np, tvl, under, levels, np.full(ps.shape, tv), np.log(ps), z0)
    expect = ps * np.exp(M.G * z0 / (M.RD * tv))
    np.testing.assert_allclose(np.exp(lnp), expect, rtol=1e-9)


def test_level_tv_is_linear_in_ln_p_between_layers_and_the_lowest_layer_below_it():
    pmid = np.array([95000.0, 90000.0, 85000.0, 80000.0])[:, None, None] * np.ones((1, 2, 2))
    tv = 300.0 + 10.0 * np.log(pmid / 1e5)
    ps = np.full((2, 2), 97000.0)
    ps[1, 1] = 86000.0
    levels = [87500.0, 96000.0]
    out, under = M.level_tv(np, tv.astype(np.float32), np.log(pmid).astype(np.float32), np.log(ps), levels)
    np.testing.assert_allclose(out[0, 0, 0], 300.0 + 10.0 * np.log(0.875), atol=1e-4)
    np.testing.assert_allclose(out[1, 0, 0], tv[0, 0, 0], atol=1e-4)   # between the lowest mid-layer and the ground
    assert under[0, 1, 1] and under[1, 1, 1] and not under[0, 0, 0] and np.isnan(out[0, 1, 1])
