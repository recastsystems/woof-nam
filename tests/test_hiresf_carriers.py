"""The nest (hiresf) carriers on a small synthetic history: averages, accumulations, rates, CPOFP, albedos, WTMP."""
import numpy as np
import pytest

from namgpu.products import hiresf_carriers as HC


class S:
    """A history Dataset stand-in: variables[name] -> [1, j, i]."""

    def __init__(self, **fields):
        self.variables = {k: np.asarray(v, np.float32)[None] for k, v in fields.items()}

    def __getitem__(self, k):
        return self.variables[k]


def run(name, s, ctx=None):
    fn, mode = HC.CARRIERS[name]
    assert fn is not None, name
    return fn(s, ctx if ctx is not None else {})


def test_every_carrier_has_a_function_and_a_regrid_mode():
    for name, (fn, mode) in HC.CARRIERS.items():
        assert callable(fn), name
        assert mode in ("nearest", "budget"), name
    assert HC.CARRIERS["NAMH_WEASD_ACC"][1] == "budget"


def test_bucket_averages_divide_by_the_count_with_the_record_sign():
    s = S(ASWIN=[[600.0, 0.0]], ASWOUT=[[-120.0, 0.0]], ARDSW=[[3.0, 0.0]],
          ALWIN=[[900.0, 0.0]], ALWOUT=[[-1200.0, 0.0]], ALWTOA=[[720.0, 0.0]], ARDLW=[[3.0, 0.0]],
          ASWTOA=[[300.0, 0.0]],
          SFCSHX=[[-60.0, 0.0]], SFCLHX=[[-300.0, 0.0]], SUBSHX=[[30.0, 0.0]], SNOPCX=[[-9.0, 0.0]],
          ASRFC=[[3.0, 0.0]])
    want = {"NAMH_DSWRF_AVE": 200.0, "NAMH_USWRF_AVE": 40.0, "NAMH_DLWRF_AVE": 300.0, "NAMH_ULWRF_AVE": 400.0,
            "NAMH_USWRF_TOA_AVE": 100.0, "NAMH_ULWRF_TOA_AVE": 240.0, "NAMH_SHTFL_AVE": 20.0,
            "NAMH_LHTFL_AVE": 100.0, "NAMH_GFLUX_AVE": 10.0, "NAMH_SNOHF_AVE": -3.0}
    for name, v in want.items():
        out = run(name, s)
        assert out.dtype == np.float32 and out.shape == (1, 2)
        assert out[0, 0] == pytest.approx(v), name
        assert out[0, 1] == 0.0, name          # an empty bucket (count 0) reads 0


def test_accumulations_in_kg_m2():
    s = S(SFCEVP=[[0.0004]], POTEVP=[[0.0011]], ACSNOM=[[1.5]], SSROFF=[[2.0]], BGROFF=[[0.25]], ACSNOW=[[3.0]])
    assert run("NAMH_EVP_ACC", s)[0, 0] == pytest.approx(0.4)
    assert run("NAMH_PEVAP_ACC", s)[0, 0] == pytest.approx(1.1)
    for name, v in (("NAMH_SNOM_ACC", 1.5), ("NAMH_SSRUN_ACC", 2.0), ("NAMH_BGRUN_ACC", 0.25),
                    ("NAMH_WEASD_ACC", 3.0)):
        assert run(name, s)[0, 0] == pytest.approx(v), name


def test_prate_and_cpofp_from_the_step_precipitation():
    s = S(PREC=[[0.0005, 0.0, 0.0002]], SR=[[0.25, 0.9, 1.0]])
    out = run("NAMH_PRATE", s, {"dtq2": 50.0})
    np.testing.assert_allclose(out, [[0.01, 0.0, 0.004]], rtol=1e-6)
    out = run("NAMH_PRATE", s, {"dt": 25.0, "nphs": 2})
    np.testing.assert_allclose(out, [[0.01, 0.0, 0.004]], rtol=1e-6)
    assert run("NAMH_PRATE", s, {}) is None    # no physics step known: the record is left out
    np.testing.assert_allclose(run("NAMH_CPOFP", s), [[25.0, HC.NO_PRECIP, 100.0]])


def test_hourly_maxima_to_kg_m2_s1():
    s = S(PRATEMAX=[[36.0]], FPRATEMAX=[[7.2]], REFDMAX263K=[[42.0]])
    assert run("NAMH_PRATE_MAX", s)[0, 0] == pytest.approx(0.01)
    assert run("NAMH_SRWEQ_MAX", s)[0, 0] == pytest.approx(0.002)
    assert run("NAMH_REFD263K_MAX", s)[0, 0] == pytest.approx(42.0)


def test_albedos_and_wtmp_by_surface():
    # land, open water, sea ice
    s = S(SM=[[0.0, 1.0, 0.0]], SICE=[[0.0, 0.0, 1.0]], ALBASE=[[0.18, 0.38, 0.38]], MXSNAL=[[0.55, 0.75, 0.75]],
          SST=[[280.0, 290.0, 271.0]], THS=[[300.0, 300.0, 260.0]], PSFC=[[90000.0, 101000.0, 100000.0]])
    np.testing.assert_allclose(run("NAMH_SNFALB", s), [[18.0, HC.WATER_ALB, HC.ICE_SNFALB]], rtol=1e-6)
    np.testing.assert_allclose(run("NAMH_MXSALB", s), [[55.0, HC.WATER_ALB, HC.ICE_MXSALB]], rtol=1e-6)
    t = run("NAMH_WTMP", s)
    assert t[0, 0] == pytest.approx(300.0 * 0.9 ** (287.04 / 1004.6), rel=1e-6)
    assert t[0, 1] == pytest.approx(290.0)
    assert t[0, 2] == pytest.approx(260.0)


def test_missing_input_leaves_the_record_out():
    s = S(ASWIN=[[1.0]])
    for name in HC.CARRIERS:
        assert run(name, s, {"dtq2": 50.0}) is None, name


def test_compute_reads_a_history_file(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    p = tmp_path / "wrfout_d01_test"
    with netCDF4.Dataset(p, "w") as d:
        d.createDimension("Time", None)
        d.createDimension("south_north", 2)
        d.createDimension("west_east", 3)
        for n, v in (("QSH", 0.01), ("PREC", 0.001), ("SR", 0.5)):
            x = d.createVariable(n, "f4", ("Time", "south_north", "west_east"))
            x[0] = np.full((2, 3), v, np.float32)
    a, mode = HC.compute(str(p), "NAMH_SPFH_SFC")
    assert mode == "nearest" and a.shape == (2, 3) and np.allclose(a, 0.01)
    a, _ = HC.compute(str(p), "NAMH_PRATE", {"dtq2": 100.0})
    assert np.allclose(a, 0.01)
    a, _ = HC.compute(str(p), "NAMH_PRATE")
    assert a is None
    a, _ = HC.compute(str(p), "NAMH_CPOFP")
    assert np.allclose(a, 50.0)
