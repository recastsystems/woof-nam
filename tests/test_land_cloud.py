"""The awphys land, soil, snow, flux, radiation and cloud carriers (namgpu.products.land_cloud) on small histories.

Every awphys `carrier:NAM_*` row has a carrier; the soil-table records are the shipped tables' values for each
point's class; the masks, units and scalings are the ones the module states. CPU only."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.products import land_cloud as LC  # noqa: E402

JM, IM = 2, 3


class _Var:
    def __init__(self, a):
        self.a = np.asarray(a, np.float32)

    def __getitem__(self, k):
        return self.a[k]


class _Hist:
    def __init__(self, d):
        self.variables = {k: _Var(np.asarray(v)[None]) for k, v in d.items()}

    def __getitem__(self, k):
        return self.variables[k]


def _plane(*vals):
    return np.array(vals, np.float32).reshape(JM, IM)


# points: 0 sand, 1 clay, 2 loam, 3 open water, 4 sea ice, 5 land ice
SM = _plane(0, 0, 0, 1, 0, 0)
SICE = _plane(0, 0, 0, 0, 1, 0)
ISLTYP = _plane(1, 12, 6, 14, 16, 16)
IVGTYP = _plane(1, 12, 10, 17, 15, 15)


def _base(**more):
    d = {"SM": SM, "SICE": SICE, "ISLTYP": ISLTYP, "IVGTYP": IVGTYP}
    d.update(more)
    return _Hist(d)


def test_every_awphys_carrier_row_has_a_carrier():
    names = set()
    for line in (ROOT / "namgpu" / "products" / "awphys.tsv").read_text().splitlines():
        f = line.split("\t")
        if len(f) > 7 and f[7].startswith("carrier:NAM_"):
            names.add(f[7].split(":", 1)[1].split("[")[0])
    assert names == set(LC.CARRIERS)
    assert len(LC.CARRIERS) == 36
    assert set(LC.SOIL) <= set(LC.CARRIERS)


def test_soil_table_records_are_the_table_values_on_land_and_zero_elsewhere():
    s = _base()
    soil, veg = LC.soil_table(), LC.veg_table()
    for name, key in (("NAM_POROS", "MAXSMC"), ("NAM_SMREF", "REFSMC"), ("NAM_WILT", "WLTSMC"),
                      ("NAM_SMDRY", "DRYSMC")):
        a = LC.CARRIERS[name][0](s).ravel()
        want = [soil[key][0], soil[key][11], soil[key][5], 0.0, 0.0, soil[key][15]]
        assert np.allclose(a, want), (name, a, want)
    assert abs(float(soil["MAXSMC"][0]) - 0.395) < 1e-6 and abs(float(soil["REFSMC"][11]) - 0.403) < 1e-6
    r = LC.rlyrs(s).ravel()
    assert np.allclose(r, [veg["NROOT"][0], veg["NROOT"][11], veg["NROOT"][9], 0, 0, veg["NROOT"][14]])
    assert list(r[:3]) == [4.0, 3.0, 3.0]


def test_masks_classes_and_land_state():
    s = _base(SMSTAV=_plane(0.5, 1.2, 0.0, 0.0, 0.0, 0.3), SMSTOT=_plane(400, 500, 600, 0, 0, 700),
              SNO=_plane(0, 0.2, 0, 0, 5, 30), SI=_plane(0, 1, 0, 0, 20, 100), CMC=_plane(5e-4, 0, 1e-4, 0, 0, 0))
    assert np.array_equal(LC.land(s).ravel(), [1, 1, 1, 0, 0, 1])
    assert np.array_equal(LC.icec(s).ravel(), [0, 0, 0, 0, 1, 0])
    assert np.array_equal(LC.CARRIERS["NAM_SOTYP"][0](s).ravel(), [1, 12, 6, 0, 0, 16])
    assert np.array_equal(LC.CARRIERS["NAM_VGTYP"][0](s).ravel(), [1, 12, 10, 0, 0, 15])
    assert np.allclose(LC.mstav(s).ravel(), [50, 100, 0, 0, 0, 30])
    assert np.allclose(LC.soilm(s).ravel(), [400, 500, 600, LC.WATER_SOILM, 0, 700])
    assert np.array_equal(LC.snowc(s).ravel(), [0, 100, 0, 0, 100, 100])
    assert np.allclose(LC.CARRIERS["NAM_SNOD"][0](s).ravel(), [0, 1e-3, 0, 0, 0.02, 0.1])
    assert np.allclose(LC.CARRIERS["NAM_CNWAT"][0](s).ravel(), [0.5, 0, 0.1, 0, 0, 0])


def test_soilm_rebuilt_from_the_layers_when_the_history_total_is_zero():
    smc = np.full((4, JM, IM), 0.25, np.float32)
    s = _base(SMSTOT=np.zeros((JM, IM), np.float32), SMOIS=smc)
    out = LC.soilm(s).ravel()
    assert np.allclose(out[[0, 1, 2, 5]], 1000.0 * 2.0 * 0.25)
    assert out[3] == LC.WATER_SOILM and out[4] == 0.0


def test_fluxes_drag_and_radiation():
    t_lm = 280.0
    p = np.full((1, JM, IM), 9.0e4, np.float32)
    th = np.full((1, JM, IM), t_lm / (9.0e4 / 1.0e5) ** LC.CAPPA - 300.0, np.float32)
    sig = np.full((JM, IM), LC.STBOL * 270.0 ** 4, np.float32)
    s = _base(TWBS=_plane(-10, 5, 0, 0, 0, 0), QWBS=_plane(-100, 0, 0, 0, 0, 0), USTAR=np.full((JM, IM), 0.3),
              U=np.full((1, JM, IM), 3.0), V=np.zeros((1, JM, IM)), RSWIN=np.full((JM, IM), 600.0),
              RSWOUT=np.full((JM, IM), 120.0), CZEN=_plane(0.4, 0.4, 0.0, 0.4, 0.4, 0.4),
              CZMEAN=_plane(0.8, 0.0, 0.5, 0.8, 0.8, 0.8), RLWIN=np.full((JM, IM), 300.0), SIGT4=sig, T=th, PRES=p,
              RADOT=np.full((JM, IM), 390.0))
    assert np.allclose(LC.CARRIERS["NAM_SHTFL"][0](s).ravel(), [10, -5, 0, 0, 0, 0])
    assert np.allclose(LC.CARRIERS["NAM_LHTFL"][0](s).ravel()[0], 100)
    assert np.allclose(LC.cd(s), 0.01)
    assert np.allclose(LC.dswrf(s).ravel(), [300, 0, 0, 300, 300, 300])
    assert np.allclose(LC.uswrf(s).ravel(), [60, 0, 0, 60, 60, 60])
    assert np.allclose(LC.dlwrf(s), 300.0 * (280.0 / 270.0) ** 4, rtol=1e-5)
    assert np.allclose(LC.ulwrf(s), 390.0)
    s.variables["SIGT4"] = _Var(np.zeros((1, JM, IM)))
    assert np.allclose(LC.dlwrf(s), 300.0)                   # no longwave call yet: the flux as it stands


def test_calm_point_drag_is_finite():
    s = _base(USTAR=np.full((JM, IM), 0.05), U=np.zeros((1, JM, IM)), V=np.zeros((1, JM, IM)))
    assert np.all(np.isfinite(LC.cd(s)))


def test_total_cloud_is_the_largest_block_mean_layer_cover():
    nz, jm, im = 3, 4, 5
    cf = np.zeros((nz, jm, im), np.float32)
    cf[0, 1, 1] = 0.9                                        # one cloudy point in the lowest layer
    cf[2] = 0.3                                              # an overcast thin layer aloft
    cf[2, 3, 4] = 1.0
    out = LC.tcdc(_Hist({"CLDFRA": cf}))
    assert out.shape == (jm, im) and out.dtype == np.float32
    # the 3 x 3 block around (1, 1) and its neighbours holds 0.9 once: 10 %, below the 30 % layer aloft
    assert np.allclose(out[1, 1], 30.0) and np.allclose(out[0, 0], 30.0)
    # the corner block repeats the edge: (3, 4) counts 4 times in its own block, (2, 3) and (3, 3), (2, 4) once
    assert np.allclose(out[3, 4], 100.0 * (4 * 1.0 + 5 * 0.3) / 9.0, atol=1e-4)
    assert np.allclose(out[2, 3], 100.0 * (1.0 + 8 * 0.3) / 9.0, atol=1e-4)
    cf[2] = 0.0
    cf[2, 3, 4] = 0.0
    out = LC.tcdc(_Hist({"CLDFRA": cf}))
    assert np.allclose(out[1, 1], 10.0, atol=1e-4) and np.allclose(out[3, 4], 0.0)
    assert np.allclose(out[0, 0], 100.0 * 0.9 / 9.0, atol=1e-4)


def test_cloud_base_is_the_lowest_condensate_or_convective_layer():
    nz = 4
    ph = np.zeros((nz + 1, JM, IM), np.float32)
    for k in range(nz + 1):
        ph[k] = LC.G * (100.0 + 500.0 * k)                   # interfaces 100, 600, 1100, 1600, 2100 m
    qc = np.zeros((nz, JM, IM), np.float32)
    qs = np.zeros((nz, JM, IM), np.float32)
    qc[2, 0, 0] = 2.0e-5                                     # cloud water in layer 3
    qc[1, 0, 1] = 0.6e-5
    qs[1, 0, 1] = 0.6e-5                                     # water + snow pass CLOUD_Q together in layer 2
    qc[0, 0, 2] = 0.9e-5                                     # below CLOUD_Q: not cloud
    cnv = np.full((JM, IM), nz + 1, np.float32)              # no convective cloud
    cnv[0, 0] = 2                                            # convective bottom in layer 2, below the water
    cnv[1, 0] = 4
    s = _Hist({"PH": ph, "QCLOUD": qc, "QSNOW": qs, "CNVBOT": cnv})
    base = LC.cldbase(s).ravel()
    assert np.allclose(base[:4], [850.0, 850.0, LC.NO_CLOUD, 1850.0], atol=1e-3)
    assert np.all(base[4:] == LC.NO_CLOUD)
    s = _Hist({"PH": ph, "QCLOUD": qc})                     # without CNVBOT or snow: cloud water alone
    assert np.allclose(LC.cldbase(s).ravel()[:3], [1350.0, LC.NO_CLOUD, LC.NO_CLOUD], atol=1e-3)


def test_a_missing_input_leaves_the_record_out():
    s = _Hist({"SM": SM})
    for name in ("NAM_POROS", "NAM_CD", "NAM_DSWRF", "NAM_TCDC", "NAM_CLDBASE", "NAM_TSOIL", "NAM_SNOWC"):
        assert LC.CARRIERS[name][0](s) is None, name


def test_compute_reads_a_history_file(tmp_path):
    import netCDF4
    h = tmp_path / "wrfout_d01_2024-11-05_12:00:00"
    with netCDF4.Dataset(h, "w") as d:
        d.createDimension("Time", None)
        d.createDimension("soil_layers_stag", 4)
        d.createDimension("south_north", JM)
        d.createDimension("west_east", IM)
        for name, a in (("SM", SM), ("SICE", SICE), ("ISLTYP", ISLTYP), ("ALBEDO", np.full((JM, IM), 0.2))):
            d.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = a
        d.createVariable("TSLB", "f4", ("Time", "soil_layers_stag", "south_north", "west_east"))[0] = 280.0
    a, mode = LC.compute(str(h), "NAM_POROS")
    assert mode == "nearest" and a.shape == (JM, IM) and abs(float(a[0, 0]) - 0.395) < 1e-6
    a, mode = LC.compute(str(h), "NAM_TSOIL")
    assert a.shape == (4, JM, IM) and np.all(a == 280.0)
    a, mode = LC.compute(str(h), "NAM_ALBDO")
    assert np.allclose(a, 20.0)
    assert LC.compute(str(h), "NAM_TCDC")[0] is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
