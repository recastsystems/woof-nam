"""A start from the NAM's own analysis carries the analysis' 0 h diagnostics into the state before the time loop.

Before this, a native start (no digital filter) wrote its 0 h history with total cloud cover, boundary-layer height,
the surface fluxes, the radiation fluxes, moisture availability and the vertical velocity all zero, total soil
moisture 0.1 mm over land and friction velocity 0.1 (94 all-zero records of the +0 h awphys file of the
2024-11-05 12Z cycle, where NCEP's were not zero). The analysis file carries these fields; init-native decodes them
(namgpu/init/sources.py NATIVE_DIAG) and drv-native-diag-restore puts them in the state (fill), with the radiation
time factors set to 1 so the 0 h products are the analysis' own fluxes. The products' total soil moisture is
rebuilt from the soil state when the history's SMSTOT is all zero over land. CPU only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.init import sources  # noqa: E402
from namgpu.products import land_cloud as LC  # noqa: E402
from namgpu.steps import drv_native_diag_restore as R  # noqa: E402

JM, IM = 5, 6


def _inv_row(d, c, p, lt, lv, lv2=None, pdt="0"):
    return {"dcp": (d, c, p), "level_type": lt, "level_value": lv, "second_level_value": lv2 or "0", "pdt": pdt}


def test_native_diag_table_matches_the_analysis_records():
    # the instantaneous records of nam.t12z.bgrd3d00 (2024-11-05), plus their 4.8 averages, which must not match
    inv = [_inv_row(0, 3, 196, "1", "0"), _inv_row(0, 2, 197, "1", "0"),
           _inv_row(0, 0, 11, "1", "0", pdt="8"), _inv_row(0, 0, 11, "1", "0"),
           _inv_row(0, 0, 10, "1", "0", pdt="8"), _inv_row(0, 0, 10, "1", "0"),
           _inv_row(0, 4, 192, "1", "0"), _inv_row(0, 4, 193, "1", "0"),
           _inv_row(0, 5, 192, "1", "0"), _inv_row(0, 5, 193, "1", "0"),
           _inv_row(2, 0, 3, "106", "0", "2"), _inv_row(2, 0, 194, "106", "0", "1"),
           _inv_row(0, 1, 196, "1", "0"), _inv_row(0, 3, 0, "242", "0"), _inv_row(0, 3, 0, "243", "0"),
           _inv_row(0, 3, 0, "248", "0"), _inv_row(0, 3, 0, "206", "0")]     # shallow and grid-scale: not read
    assert sorted(sources.native_diag_keys(inv)) == sorted(sources.NATIVE_DIAG)
    assert sources.native_diag_keys([_inv_row(0, 0, 11, "1", "0", pdt="8")]) == []
    assert sources.NATIVE_3D["CFRAC"] == (0, 6, 1)


def test_fill_puts_the_analysis_diagnostics_in_the_state():
    names = ("PBLH", "USTAR", "TWBS", "QWBS", "RSWIN", "RSWOUT", "RLWIN", "RADOT", "CZEN", "CZMEAN", "SIGT4",
             "SMSTOT", "SMSTAV")
    st = {n: np.zeros((JM, IM), np.float32) for n in names}
    st["USTAR"][...] = 0.1                                   # JSFC_INIT's cold start
    r = np.arange(JM * IM, dtype=np.float32).reshape(JM, IM)
    diag = {"HPBL": 500 + r, "FRICV": 0.2 + 0.01 * r, "SHTFL": 10 + r, "LHTFL": 50 + r, "DSWRF": 300 + r,
            "USWRF": 40 + r, "DLWRF": 280 + r, "ULWRF": 350 + r, "SOILM": 400 + r, "MSTAV": 30 + r}
    t = 270.0 + r
    done = R.fill(st, diag, t, np)
    assert sorted(done) == sorted(names)
    assert np.allclose(st["PBLH"], 500 + r) and np.allclose(st["USTAR"], 0.2 + 0.01 * r)
    assert np.allclose(st["TWBS"], -(10 + r)) and np.allclose(st["QWBS"], -(50 + r))
    assert np.allclose(st["SMSTOT"], 400 + r) and np.allclose(st["SMSTAV"], np.clip((30 + r) / 100, 0, 1))

    # the products read the analysis' own fluxes back from these state fields at 0 h
    hist = {k: v[None] for k, v in st.items()}
    th = (t / (1.0e5 * np.float32(1e-5)) ** LC.CAPPA - 300.0)       # theta - 300 at p = 1000 hPa
    hist["T"] = th[None, None].astype(np.float32)
    hist["PRES"] = np.full((1, 1, JM, IM), 1.0e5, np.float32)
    s = _Hist(hist)
    assert np.allclose(LC.dswrf(s), 300 + r, rtol=1e-6)
    assert np.allclose(LC.uswrf(s), 40 + r, rtol=1e-6)
    assert np.allclose(LC.dlwrf(s), 280 + r, rtol=1e-4)
    assert np.allclose(LC.ulwrf(s), 350 + r)


class _Hist:
    """A history file as land_cloud reads one: .variables and s[name][0]."""
    def __init__(self, d):
        self.variables = {k: _Var(v) for k, v in d.items()}

    def __getitem__(self, k):
        return self.variables[k]


class _Var:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, k):
        return self.a[k]

    @property
    def shape(self):
        return self.a.shape

    def ncattrs(self):
        return []


def test_fill_skips_what_the_analysis_lacks():
    st = {"PBLH": np.zeros((JM, IM), np.float32), "USTAR": np.full((JM, IM), 0.1, np.float32)}
    done = R.fill(st, {"HPBL": np.full((JM, IM), 700.0, np.float32)}, np.full((JM, IM), 280.0, np.float32), np)
    assert done == ["PBLH"] and np.all(st["USTAR"] == np.float32(0.1))


def test_fill_convective_carries_the_analysis_convection_into_the_0h_state():
    """The 0 h convective rain and cloud of a native start: before this the 0 h composite reflectivity had no
    convective echo and the 0 h cloud base no convective bottom (2024-11-05 12Z: the hurricane's echo missing,
    cloud-base MAE 1,123 m against NCEP)."""
    lm = 4
    # mid-layer pressures, index 0 the top layer
    p_mid = np.empty((lm, JM, IM), np.float32)
    for k, p in enumerate((30000.0, 60000.0, 85000.0, 98000.0)):
        p_mid[k] = p
    st = {"CPRATE": np.zeros((JM, IM), np.float32), "CNVBOT": np.zeros((JM, IM), np.float32),
          "CNVTOP": np.zeros((JM, IM), np.float32)}
    cpr = np.full((JM, IM), 2.0e-4, np.float32)               # kg m-2 s-1 = 0.72 mm h-1
    bot = np.full((JM, IM), -50000.0, np.float32)             # the record's no-cloud value
    top = np.full((JM, IM), -50000.0, np.float32)
    bot[2, 3], top[2, 3] = 85001.5, 30002.0                  # within the record's packing of two mid levels
    done, worst = R.fill_convective(st, {"CPRAT": cpr, "PCNVBOT": bot, "PCNVTOP": top}, p_mid, 50.0, np)
    assert sorted(done) == ["CNVBOT", "CNVTOP", "CPRATE"]
    assert worst == 2.0
    # m of water over one convection call of 50 s, interior only (the glue never writes the outer ring)
    assert np.isclose(st["CPRATE"][2, 3], 2.0e-4 * 50.0 * 1e-3)
    assert np.all(st["CPRATE"][0] == 0) and np.all(st["CPRATE"][:, -1] == 0)
    # layers from the ground: 98000 is layer 1, 85000 layer 2, 30000 layer 4; empty columns get the glue's resets
    assert st["CNVBOT"][2, 3] == 2 and st["CNVTOP"][2, 3] == 4
    assert st["CNVBOT"][0, 0] == lm + 1 and st["CNVTOP"][0, 0] == 0

    # the cloud base reads that convective bottom: the middle of layer 2 from the ground (history levels bottom first)
    hist = _Hist({"PH": (np.array([0.0, 500.0, 2000.0, 6000.0, 16000.0], np.float32)[:, None, None]
                         * np.ones((1, JM, IM), np.float32) * np.float32(LC.G))[None],
                  "QCLOUD": np.zeros((1, lm, JM, IM), np.float32), "CNVBOT": st["CNVBOT"][None]})
    cb = LC.cldbase(hist)
    assert np.isclose(cb[2, 3], 1250.0) and cb[0, 0] == LC.NO_CLOUD


def test_fill_convective_skips_what_the_analysis_lacks():
    st = {"CPRATE": np.ones((JM, IM), np.float32)}
    done, worst = R.fill_convective(st, {}, np.ones((2, JM, IM), np.float32), 50.0, np)
    assert done == [] and worst == 0.0 and np.all(st["CPRATE"] == 1)


def test_soilm_rebuilds_an_all_zero_history_total():
    smc = np.full((4, JM, IM), 0.3, np.float32)
    sm = np.zeros((JM, IM), np.float32)
    sm[0, 0] = 1.0                                           # one water point
    vars_ = {"SMSTOT": np.zeros((1, JM, IM), np.float32), "SMOIS": smc[None], "SM": sm[None],
             "SICE": np.zeros((1, JM, IM), np.float32), "XLAND": (1.0 + sm)[None]}
    s = _Hist(vars_)
    out = LC.soilm(s)
    want = 1000.0 * float(np.sum(LC.NSOIL_DZ[:4]) * 0.3)
    assert out is not None
    assert abs(float(out[2, 3]) - want) < 1e-2, (float(out[2, 3]), want)
    assert float(out[0, 0]) == 1000.0                         # open water


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
