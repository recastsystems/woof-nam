"""namgpu.products.ptype on textbook soundings (CPU, numpy): each algorithm types the classic profiles as its paper
says, the vote and its tie order, and the masks."""
import numpy as np
import pytest

from namgpu.products import ptype as P

LEV = np.arange(100000.0, 39000.0, -2500.0)          # 1000 -> 400 hPa, bottom-up


def sounding(tc_of_p, psfc=100500.0, t2c=None, rh=0.97):
    T = np.array([tc_of_p(p / 100.0) for p in LEV]) + P.T0
    es = 611.2 * np.exp(17.67 * (T - P.T0) / (T - 29.65))
    w = rh * P.EPS * es / (LEV - es)
    t2 = (tc_of_p(psfc / 100.0) if t2c is None else t2c) + P.T0
    es2 = 611.2 * np.exp(17.67 * (t2 - P.T0) / (t2 - 29.65))
    return {"T": T[:, None], "P": LEV[:, None], "QV": w[:, None], "T2": np.array([t2]), "PSFC": np.array([psfc]),
            "Q2": np.array([rh * P.EPS * es2 / (psfc - es2)]), "PREC": np.array([1e-5])}


def all_types(f):
    return {k: int(v[0]) for k, v in P.classify_all(f, xp=np).items()}


def test_cold_column_is_snow():
    r = all_types(sounding(lambda p: -8.0 - (1000 - p) * 0.06))
    for a in ("ncep", "ncep-revised", "ramer", "bourgouin", "dominant"):
        assert r[a] == P.SN, (a, r)


def test_warm_surface_is_rain():
    r = all_types(sounding(lambda p: 8.0 - (1000 - p) * 0.05))
    for a in ("ncep", "ncep-revised", "ramer", "bourgouin", "dominant"):
        assert r[a] == P.RA, (a, r)


def nose(cold_c, depth_hpa, warm_c=4.0):
    def t(p):
        if p > 1000 - depth_hpa:
            return cold_c                                   # surface cold layer
        if p > 1000 - depth_hpa - 150:
            return warm_c                                   # warm nose
        return -6.0 - (1000 - depth_hpa - 150 - p) * 0.07
    return t


def test_shallow_cold_layer_under_a_warm_nose_is_freezing_rain():
    r = all_types(sounding(nose(-1.0, 40)))
    for a in ("ncep", "ncep-revised", "ramer", "bourgouin", "dominant"):
        assert r[a] == P.ZR, (a, r)


def test_deep_cold_layer_under_a_warm_nose_is_sleet():
    r = all_types(sounding(nose(-9.0, 200, warm_c=2.0)))
    for a in ("ncep", "bourgouin", "dominant"):
        assert r[a] == P.IP, (a, r)


def test_vote_and_tie_order():
    v = [np.array([P.SN, P.SN, P.RA, -1], np.int8), np.array([P.SN, P.ZR, P.RA, -1], np.int8),
         np.array([P.IP, P.ZR, P.SN, -1], np.int8), np.array([P.IP, P.SN, P.SN, -1], np.int8),
         np.array([-1, P.IP, -1, -1], np.int8)]
    d = P.dominant(np, v)
    # SN 2 / IP 2 -> IP (hazard first); SN 2 / ZR 2 / IP 1 -> ZR; RA 2 / SN 2 -> SN; nobody -> -1
    assert d.tolist() == [P.IP, P.ZR, P.SN, -1]


def test_explicit_abstains_without_hydrometeors_and_explicit_sr_is_the_old_rule():
    f = sounding(lambda p: -5.0 - (1000 - p) * 0.06)
    f.update(SR=np.array([0.0]), QRAIN0=np.array([0.0]), QSNOW0=np.array([0.0]), QGRAUP0=np.array([0.0]))
    r = all_types(f)
    assert r["explicit"] == P.NONE              # no vote
    assert r["explicit-sr"] == P.ZR             # the old rule: SR 0 and T2 < 0 C -> freezing rain (the defect)
    assert r["dominant"] == P.SN


def test_masks():
    f = {"PREC": np.array([0.0, 1e-9, 5e-6])}
    assert P.mask(f, "prec").tolist() == [False, True, True]
    assert P.mask(f, "rate", dtq2=30.0).tolist() == [False, False, True]   # 1e-9 m / 30 s = 1.2e-4 mm/h: below a trace
    with pytest.raises(ValueError):
        P.mask(f, "rate")
    assert P.mask(f, "prec4e-6").tolist() == [False, False, True]
    # the default rate is NCEP's 0.01 mm/h: 0.02 mm/h of light snow is typed, 0.005 mm/h is not. At the 0.05 mm/h used
    # before, a run from the NAM's own analysis typed 15.6 % of grid 218 where NCEP's file types 21.3 %.
    assert P.MASK_RATE_MM_H == 0.01
    light = {"PREC": np.array([0.005, 0.02]) * 1.0e-3 / 3600.0 * 50.0}
    assert P.mask(light, "rate", dtq2=50.0).tolist() == [False, True]
    with pytest.raises(ValueError):
        P.mask(f, "nope")


def test_history_product_and_flags(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    f = sounding(nose(-1.0, 40))
    L = f["T"].shape[0]
    p = tmp_path / "wrfout_d01_test.nc"
    with netCDF4.Dataset(p, "w") as d:
        for n, s in (("Time", None), ("bottom_top", L), ("south_north", 2), ("west_east", 3)):
            d.createDimension(n, s)
        th = f["T"][:, 0] * (1.0e5 / f["P"][:, 0]) ** 0.28571 - 300.0
        for n, v in (("T", th), ("PRES", f["P"][:, 0]), ("QVAPOR", f["QV"][:, 0])):
            d.createVariable(n, "f4", ("Time", "bottom_top", "south_north", "west_east"))[0] = v[:, None, None] * np.ones((1, 2, 3))
        for n, v in (("T2", f["T2"][0]), ("Q2", f["Q2"][0]), ("PSFC", f["PSFC"][0]), ("SR", 0.0)):
            d.createVariable(n, "f4", ("Time", "south_north", "west_east"))[0] = np.full((2, 3), v)
        prec = np.full((2, 3), 2e-6)
        prec[0, 0] = 0.0
        d.createVariable("PREC", "f4", ("Time", "south_north", "west_east"))[0] = prec
    assert P.history_ptype(p) is None                    # rate mask: no configfile beside the history -> left out
    (tmp_path / "nam_configfile").write_text("dt_int: 15" + chr(10) + "dt_num: 0" + chr(10) + "dt_den: 3" + chr(10) + "nphs: 2" + chr(10))
    c = P.history_ptype(p)                               # 2e-6 m per 30 s = 0.24 mm/h >= the trace
    assert c.shape == (2, 3) and c[0, 0] == P.NONE and (c.ravel()[1:] == P.ZR).all()
    assert (P.history_ptype(p, mask_name="prec") == c).all()
    assert P.history_ptype(p, method="explicit-sr", mask_name="prec4e-6").max() == P.NONE   # old mask: too light
    with netCDF4.Dataset(p) as d:
        assert P.flag(d, P.ZR).sum() == 5 and P.flag(d, P.SN).sum() == 0


def test_run_step_file_feeds_the_products_rate(tmp_path):
    """woof-nam run writes nam_configfile_step; woof-nam products (_run_ctx) reads DT*NPHS back from it."""
    from types import SimpleNamespace
    from namgpu.io.products import _run_ctx
    from namgpu.run_forecast import write_step_file
    write_step_file(tmp_path, SimpleNamespace(dt=15.0, nphs=2))
    assert _run_ctx(tmp_path)["dtq2"] == 30.0
    write_step_file(tmp_path, SimpleNamespace(dt=26.666666, nphs=2))
    assert abs(_run_ctx(tmp_path)["dtq2"] - 53.334) < 1e-3


@pytest.mark.parametrize('product_set', ['conusnest', 'alaskanest', 'hawaiinest', 'priconest', 'firewxnest'])
def test_folded_nest_types_keep_the_nest_grid(monkeypatch, tmp_path, product_set):
    """The parent-only helper must not regrid a nest's categorical flags onto grid 218."""
    from namgpu.io import products
    from namgpu.products import sets, spec
    gkey = (product_set, 'nest-window')
    tab = {'_near': np.array([5, 0]), 'wx': np.zeros((1, 2))}
    monkeypatch.setattr(products, 'tables_for', lambda key: (tab, None) if key == gkey else pytest.fail(str(key)))
    monkeypatch.setattr(products, '_tables', lambda: pytest.fail('parent table used for a nest'))
    field = np.arange(6, dtype=np.float32).reshape(2, 3)
    monkeypatch.setattr(products.LC, 'compute', lambda history, name: (field, 'nearest'))
    got = products._ptype_carriers(tmp_path / 'history', gkey, {'dtq2': 50.0})
    for name in products.PTYPE_CARRIERS:
        assert got[name][0].tolist() == [[5.0, 0.0]]
    rows = spec.read(sets.table_path(sets.get(product_set)))
    for name in products.PTYPE_CARRIERS:
        assert any(row['source'] == 'carrier:' + name for row in rows)


@pytest.mark.parametrize('product_set', ['awphys', 'conusnest', 'alaskanest', 'hawaiinest', 'priconest', 'firewxnest'])
def test_the_frame_holds_the_type_carriers(product_set):
    """The frame's carrier list is what the frame writer loops over: the four type carriers must be on it for the
    parent and every nest set (they are computed apart from the field pool, and the 0.2.0 fold then left them out of
    the frame: no door wrote CSNOW/CICEP/CFRZR/CRAIN)."""
    import inspect
    from namgpu.io import products
    from namgpu.products import sets
    names = products.frame_carriers(sets.get(product_set))
    for name in products.PTYPE_CARRIERS:
        assert names.count(name) == 1
    src = inspect.getsource(products.wrf_frame) + inspect.getsource(products._wrf_frame_write)   # the writer half
    assert "for name in frame_carriers(ps):" in src            # the write loop
    assert "if n not in PTYPE_CARRIERS" in src                  # the pool still leaves them to the main process
