"""History reflectivity with the convective rain (kernel hist_refl_conv, namgpu/kernels/io_history_fields.cu).

CPU: the method's numbers on hand-made columns (NumPy reference below, written from the formulas in the kernel
header: Z = 300 R^1.4, Fulton et al. 1998; above the 0 C level -3.5 dB and -4 dB per km, reaching the dry-ice
-6.5 dB of Smith 1984 0.75 km above it), and the kernel source compiled with NVRTC
when CuPy is installed (no card is touched). With a card, the kernel against the reference on random columns.

    python tests/test_refl_conv.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DBZMIN, ZMIN = np.float32(-20.0), np.float32(0.01)
ICE_D0, ICE_SLOPE = 3.5, 4.0e-3                              # dB at the 0 C level, dB per m above it


def zero_c_height(t, z):
    """Lowest 0 C crossing per column, linear in height between layer midpoints (l = 0 at the top)."""
    lm = t.shape[0]
    z0 = np.full(t.shape[1:], np.float32(3.0e38), np.float32)
    for j, i in np.ndindex(*t.shape[1:]):
        if t[lm - 1, j, i] < 273.15:
            z0[j, i] = z[lm - 1, j, i]
            continue
        for l in range(lm - 2, -1, -1):
            if t[l, j, i] < 273.15:
                w = (t[l + 1, j, i] - 273.15) / (t[l + 1, j, i] - t[l, j, i])
                z0[j, i] = z[l + 1, j, i] + w * (z[l, j, i] - z[l + 1, j, i])
                break
    return z0


def heights(lm, dz=500.0):
    """Layer midpoint heights (m), l = 0 at the top, the bottom layer at dz / 2."""
    return (dz * (lm - 1 - np.arange(lm)) + dz / 2).astype(np.float32)[:, None, None]


def reference(refl, t, cprate, cnvtop, rate_mmh, z=None):
    """refl, t, z [LM, JM, IM] (l = 0 at the top); cprate, cnvtop [JM, IM]. Returns (refl3, composite)."""
    lm = refl.shape[0]
    if z is None:
        z = np.broadcast_to(heights(lm), refl.shape)
    r = np.where(cprate > 0, cprate * np.float32(rate_mmh), 0).astype(np.float32)
    zconv = np.where(cprate > 0, np.float32(300.0) * r ** np.float32(1.4), 0).astype(np.float32)
    ktop = np.rint(cnvtop).astype(int)
    ktop = np.where((ktop < 1) | (ktop > lm), lm, ktop)
    from_bottom = lm - np.arange(lm)[:, None, None]          # layer number counted from the bottom
    inside = (from_bottom <= ktop[None]) & (zconv[None] > 0)
    above = np.maximum(z - zero_c_height(t, z)[None], 0)
    ice = np.float32(10.0) ** (np.float32(-0.1) * (ICE_D0 + ICE_SLOPE * above)).astype(np.float32)
    zc = np.where(t < np.float32(273.15), ice, np.float32(1.0)) * zconv[None]
    z = np.where(refl > DBZMIN, np.float32(10.0) ** (np.float32(0.1) * refl), 0) + zc
    with np.errstate(divide="ignore"):
        added = np.where(z > ZMIN, 10.0 * np.log10(np.maximum(z, 1e-30)), DBZMIN)
    out = np.where(inside, added, refl).astype(np.float32)
    return out, out.max(axis=0)


def column(lm=10, t_top=230.0, t_bot=295.0):
    t = np.linspace(t_top, t_bot, lm, dtype=np.float32)[:, None, None]
    return t


def test_zr_values():
    # 10 mm/h of convective rain into a column with no microphysics echo: 10 log10(300 * 10^1.4) = 38.77 dBZ
    lm = 10
    t = np.full((lm, 1, 1), 290.0, np.float32)
    refl = np.full((lm, 1, 1), DBZMIN, np.float32)
    out, comp = reference(refl, t, np.array([[10.0]], np.float32), np.array([[0.0]], np.float32), 1.0)
    want = 10.0 * np.log10(300.0 * 10.0 ** 1.4)
    assert abs(want - 38.771) < 1e-3
    assert np.allclose(out, want, atol=1e-4) and abs(comp[0, 0] - want) < 1e-4
    # 1 mm/h: 24.77 dBZ; 100 mm/h: 52.77 dBZ (Fulton et al. 1998: 104 mm/h is 53 dBZ)
    for rr, dbz in ((1.0, 24.771), (100.0, 52.771), (104.0, 53.01)):
        o, _ = reference(refl, t, np.array([[rr]], np.float32), np.array([[0.0]], np.float32), 1.0)
        assert abs(float(o[0, 0, 0]) - dbz) < 0.01, (rr, float(o[0, 0, 0]))


def test_sum_in_linear_units():
    # 30 dBZ of microphysics echo plus 30 dBZ of convective rain is 33.01 dBZ
    lm = 4
    t = np.full((lm, 1, 1), 290.0, np.float32)
    refl = np.full((lm, 1, 1), 30.0, np.float32)
    r30 = (1000.0 / 300.0) ** (1 / 1.4)                    # R giving Z = 1000 (30 dBZ)
    out, _ = reference(refl, t, np.array([[r30]], np.float32), np.array([[0.0]], np.float32), 1.0)
    assert np.allclose(out, 10 * np.log10(2000.0), atol=2e-3)


def test_fall_off_above_0c_and_cloud_top():
    lm = 10
    t = column(lm)                                           # cold at the top, warm at the bottom
    z = heights(lm)                                          # 250 m, 750 m, ... 4750 m
    refl = np.full((lm, 1, 1), DBZMIN, np.float32)
    top = 8                                                  # cloud top: 8th layer from the bottom (l = 2)
    out, comp = reference(refl, t, np.array([[10.0]], np.float32), np.array([[float(top)]], np.float32), 1.0, z)
    warm = 10 * np.log10(300 * 10 ** 1.4)
    z0 = float(zero_c_height(t, z)[0, 0])
    tk, zk = t[::-1, 0, 0], z[::-1, 0, 0]                    # bottom first
    k = int(np.argmax(tk < 273.15))
    assert abs(z0 - (zk[k - 1] + (tk[k - 1] - 273.15) / (tk[k - 1] - tk[k]) * (zk[k] - zk[k - 1]))) < 1e-2
    for l in range(lm):
        from_bottom = lm - l
        if from_bottom > top:
            assert out[l, 0, 0] == DBZMIN                    # above the cloud top: unchanged
        elif t[l, 0, 0] < 273.15:
            want = warm - 3.5 - 4.0e-3 * (z[l, 0, 0] - z0)
            assert abs(out[l, 0, 0] - want) < 2e-3, (l, float(out[l, 0, 0]), want)
        else:
            assert abs(out[l, 0, 0] - warm) < 1e-3
    assert abs(comp[0, 0] - warm) < 1e-3
    # the dry-ice value (Smith 1984: 0.224, -6.50 dB) is reached 0.75 km above the 0 C level
    assert abs(10 * np.log10(0.224) + 6.498) < 1e-3 and abs(-3.5 - 4.0e-3 * 750.0 - 10 * np.log10(0.224)) < 1e-2


def test_fall_off_against_ncep_profile_points():
    # The fitted profile at the heights of the fit (km above 0 C, dB relative to below 0 C)
    lm = 40
    z = heights(lm, 250.0)
    t = (np.float32(273.15 + 6.5e-3 * 2000.0) - np.float32(6.5e-3) * z).astype(np.float32)   # 0 C at 2000 m
    refl = np.full((lm, 1, 1), DBZMIN, np.float32)
    out, _ = reference(refl, t, np.array([[30.0]], np.float32), np.array([[0.0]], np.float32), 1.0, z)
    warm = 10 * np.log10(300 * 30.0 ** 1.4)
    for h_km, drop in ((1.0, -7.5), (2.0, -11.5), (3.0, -15.5)):
        l = int(np.argmin(np.abs(z[:, 0, 0] - (2000.0 + 1000.0 * h_km))))
        hh = (z[l, 0, 0] - 2000.0) / 1000.0
        assert abs((out[l, 0, 0] - warm) - (-3.5 - 4.0 * hh)) < 2e-3
        assert abs(hh - h_km) < 0.13 and abs((-3.5 - 4.0 * h_km) - drop) < 1e-9


def test_surface_below_0c():
    # A column already below 0 C at its lowest layer: the 0 C level is that layer, and the term starts at -3.5 dB
    lm = 6
    z = heights(lm)
    t = np.linspace(240.0, 270.0, lm, dtype=np.float32)[:, None, None]
    refl = np.full((lm, 1, 1), DBZMIN, np.float32)
    out, _ = reference(refl, t, np.array([[5.0]], np.float32), np.array([[0.0]], np.float32), 1.0, z)
    warm = 10 * np.log10(300 * 5.0 ** 1.4)
    assert abs(out[lm - 1, 0, 0] - (warm - 3.5)) < 2e-3
    assert abs(out[0, 0, 0] - (warm - 3.5 - 4.0e-3 * (z[0, 0, 0] - z[lm - 1, 0, 0]))) < 2e-3


def test_no_convective_rain_is_unchanged():
    rng = np.random.default_rng(3)
    lm, jm, im = 8, 5, 7
    refl = rng.uniform(-20, 55, (lm, jm, im)).astype(np.float32)
    t = rng.uniform(220, 300, (lm, jm, im)).astype(np.float32)
    cp_ = np.zeros((jm, im), np.float32)
    cp_[0, 0] = -1e-9                                       # a negative rounding residue is no rain
    out, comp = reference(refl, t, cp_, np.full((jm, im), 5.0, np.float32), 72000.0)
    assert np.array_equal(out, refl) and np.array_equal(comp, refl.max(axis=0))


def test_rate_scale():
    from types import SimpleNamespace
    try:
        from namgpu.io.io_history_fields import convective_rate_scale
    except ModuleNotFoundError as e:                         # the module imports CuPy
        print(f"  ({e.name} not installed: rate check skipped)")
        return
    ctx = SimpleNamespace(cfg=SimpleNamespace(dt=25.0, nprecip=2))
    # CPRATE is metres per call of 2 x 25 s: 1e-4 m in 50 s is 7.2 mm/h
    assert abs(float(convective_rate_scale(ctx)) * 1e-4 - 7.2) < 1e-5


def test_kernel_compiles():
    kdir = ROOT / "namgpu" / "kernels"
    src = (kdir / "io_history_fields.cu").read_text()
    assert 'extern "C" __global__ void hist_refl_conv(' in src
    try:
        from cupy_backends.cuda.libs import nvrtc
    except Exception:  # noqa: BLE001
        print("  (CuPy not installed: compile check skipped)")
        return
    # NVRTC alone, for a fixed virtual architecture: no device is queried or touched
    prog = nvrtc.createProgram(src, "io_history_fields.cu", [], [])
    try:
        nvrtc.compileProgram(prog, ["-std=c++17", "--fmad=true", f"-I{kdir}", "-arch=compute_80"])
    except Exception:
        print(nvrtc.getProgramLog(prog))
        raise
    finally:
        nvrtc.destroyProgram(prog)


def test_kernel_matches_reference_on_a_card():
    if os.environ.get("WOOF_NAM_TEST_CARD") != "1":
        print("  (WOOF_NAM_TEST_CARD unset: card check skipped)")
        return
    import cupy as cp
    from namgpu import kernels
    rng = np.random.default_rng(11)
    lm, jm, im = 30, 17, 23
    refl = np.where(rng.uniform(size=(lm, jm, im)) < 0.3, -20.0, rng.uniform(-10, 60, (lm, jm, im))).astype(np.float32)
    t = np.sort(rng.uniform(200, 305, (lm, jm, im)), axis=0).astype(np.float32)
    zz = np.cumsum(rng.uniform(100, 600, (lm, jm, im)), axis=0)[::-1].astype(np.float32).copy()
    cpr = np.where(rng.uniform(size=(jm, im)) < 0.5, rng.uniform(0, 3e-3, (jm, im)), 0).astype(np.float32)
    top = rng.integers(0, lm + 3, (jm, im)).astype(np.float32)
    rate = np.float32(3.6e6 / 50.0)
    want, wcomp = reference(refl, t, cpr, top, rate, zz)
    d = {k: cp.asarray(v) for k, v in dict(refl=refl, t=t, zz=zz, cpr=cpr, top=top).items()}
    out = cp.empty_like(d["refl"])
    comp = cp.empty((jm, im), cp.float32)
    k = kernels.get("io_history_fields", "hist_refl_conv")
    k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (np.int32(im), np.int32(jm), np.int32(lm), d["refl"], d["t"],
                                                    d["zz"], d["cpr"], d["top"], rate, np.int32(1), out, comp))
    got, gcomp = cp.asnumpy(out), cp.asnumpy(comp)
    assert np.abs(got - want).max() < 1e-3 and np.abs(gcomp - wcomp).max() < 1e-3
    unchanged = (cpr <= 0)[None] & np.ones((lm, 1, 1), bool)
    assert np.array_equal(got[unchanged], refl[unchanged])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(tests)} passed")
