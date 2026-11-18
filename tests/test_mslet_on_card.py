"""MSLET on the card: the forecast writes MSLET as a 2-D field (io_history_fields.map_2d), light frames carry it,
and the GRIB2 MSLET carrier takes the history MSLET with no second relaxation. The device test proves map_2d on the
top-down state equals the map drawer's own computation from the written bottom-up file (memslp.from_history); it needs
a card."""
import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from namgpu import outputs  # noqa: E402
from namgpu.io import history_select, maps  # noqa: E402

L, J, I = 12, 9, 11


def _arrays(seed=5):
    rng = np.random.default_rng(seed)
    ps = (101000.0 - 30000.0 * rng.random((J, I))).astype(np.float32)
    frac = np.linspace(0.0, 0.9, L + 1)[:, None, None]
    pint = (ps[None] * (1.0 - frac)).astype(np.float32)
    pres = (0.5 * (pint[:-1] + pint[1:])).astype(np.float32)
    return {"PINT": pint, "PRES": pres, "HGT": (3000.0 * rng.random((J, I))).astype(np.float32),
            "T": (5.0 + 3.0 * rng.normal(size=(L, J, I))).astype(np.float32),
            "QVAPOR": (0.005 * rng.random((L, J, I))).astype(np.float32),
            "U": rng.normal(0, 10, (L, J, I)).astype(np.float32), "V": rng.normal(0, 10, (L, J, I)).astype(np.float32)}


def _frame(path, a, extra=None):
    with netCDF4.Dataset(path, "w") as nc:
        nc.START_DATE = "2026-11-12_18:00:00"
        for k, n in (("Time", None), ("DateStrLen", 19), ("bottom_top", L), ("bottom_top_stag", L + 1),
                     ("south_north", J), ("west_east", I)):
            nc.createDimension(k, n)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.frombuffer(b"2026-11-12_21:00:00", "S1")
        for n, x in list(a.items()) + list((extra or {}).items()):
            z = None if x.ndim == 2 else ("bottom_top_stag" if n == "PINT" else "bottom_top")
            dims = ("Time",) + ((z,) if z else ()) + ("south_north", "west_east")
            v = nc.createVariable(n, "f4", dims, zlib=True)
            v[0] = x


def test_light_frames_and_products_preset_carry_the_2d_fields():
    for n in ("MSLET",):
        assert n in outputs.MAP_FIELDS
        assert n in history_select.PRESETS["products"]
        assert n in history_select.KNOWN


def test_grib2_carrier_takes_the_history_mslet(tmp_path, monkeypatch):
    from namgpu.io import products
    from namgpu.products import memslp
    p = tmp_path / "wrfout_d01_2026-11-12_21:00:00"
    want = np.full((J, I), 101234.5, np.float32)
    _frame(p, _arrays(), {"MSLET": want})
    monkeypatch.setattr(memslp, "from_history", lambda nc: pytest.fail("relaxed again in the GRIB2 carrier"))
    got = {}
    monkeypatch.setattr(products, "_interp_for", lambda ps, n: (lambda t, x: got.setdefault("a", x)))
    products._mslet_carrier(p, None, None)
    assert np.array_equal(got["a"], want)


def test_map_2d_on_the_card_equals_the_map_lane(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception:  # noqa: BLE001 - no device
        pytest.skip("no card")
    from namgpu.io import io_history_fields as H
    from namgpu.products import memslp
    a = _arrays()
    p = tmp_path / "wrfout_d01_2026-11-12_21:00:00"
    _frame(p, a)
    out = {n: (cp.asarray(x[::-1] if x.ndim == 3 else x), None) for n, x in a.items()}    # the state is top-down
    f = H.map_2d(out)
    with netCDF4.Dataset(p) as nc:
        ref = memslp.from_history(nc, xp=cp)
    assert np.array_equal(cp.asnumpy(f["MSLET"][0]), ref)
