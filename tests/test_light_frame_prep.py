"""Light frame preparation: the 850 hPa wind from the levels up to 850 hPa equals the one from
the whole column, and a history that carries the forecast's own MSLET is drawn from it, with no relaxation here."""
import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from namgpu.io import maps  # noqa: E402


def _frame(path, mslet=None):
    rng = np.random.default_rng(3)
    L, J, I = 12, 9, 11
    with netCDF4.Dataset(path, "w") as nc:
        nc.START_DATE = "2026-11-12_18:00:00"
        for k, n in (("Time", None), ("DateStrLen", 19), ("bottom_top", L), ("south_north", J), ("west_east", I)):
            nc.createDimension(k, n)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.frombuffer(b"2026-11-12_21:00:00", "S1")
        ps = 101000.0 - 30000.0 * rng.random((J, I))                    # some ground above 850 hPa
        frac = np.linspace(0.0, 0.9, L)[:, None, None]
        pres = ps[None] * (1.0 - frac)                                     # bottom-up
        for n, a in (("PRES", pres), ("U", rng.normal(0, 10, (L, J, I))), ("V", rng.normal(0, 10, (L, J, I)))):
            v = nc.createVariable(n, "f4", ("Time", "bottom_top", "south_north", "west_east"), zlib=True)
            v[0] = a
        for n in ("XLAT", "XLONG", "T2"):
            v = nc.createVariable(n, "f4", ("Time", "south_north", "west_east"))
            v[0] = rng.random((J, I))
        if mslet is not None:
            v = nc.createVariable("MSLET", "f4", ("Time", "south_north", "west_east"))
            v[0] = mslet


def test_850_wind_from_trimmed_levels_equals_whole_column(tmp_path):
    p = tmp_path / "wrfout_d01_2026-11-12_21:00:00"
    _frame(p)
    _, _, whole = maps._read(p, {"U", "V", "PRES"})
    _, _, trim = maps._read(p, {"U", "V", "PRES"}, 85000.0)
    assert trim["U"].shape[0] < whole["U"].shape[0]
    for n in ("U", "V"):
        a = maps._at_pressure(whole[n], whole["PRES"], 85000.0)
        b = maps._at_pressure(trim[n], trim["PRES"], 85000.0)
        assert np.array_equal(a, b, equal_nan=True)
        assert np.isnan(a).any() and np.isfinite(a).any()


def test_mslet_written_by_the_forecast_is_used(tmp_path, monkeypatch):
    p = tmp_path / "wrfout_d01_2026-11-12_21:00:00"
    want = np.full((9, 11), 101234.5, np.float32)
    _frame(p, want)
    from namgpu.products import memslp
    monkeypatch.setattr(memslp, "from_history", lambda nc: pytest.fail("relaxed again by the map drawer"))
    assert np.array_equal(maps._mslet(p, tmp_path), want)
