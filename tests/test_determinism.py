"""Run-to-run determinism of the init kernels that used float atomics (GPU; skipped without CuPy or a card).

Each kernel is run twice on the same input and the two answers must be the same bits. Before the fix the GWD
orography statistics (init-gwd-orog: float atomicAdd scatter of the per-cell sums) differed in about a fifth of
the 12 km cells between two runs, and the vertical-coordinate fit (init-vcoord-recover: one double atomicAdd per
block) in its sums; the TDFI then spread that over the whole domain, so two identical forecasts differed from
+0 h on (WOOF-NAM determinism defect, 2026-11-06). The whole-forecast check is two real runs of one case
compared with `woof-nam compare`; this is the quick check of the two kernels.

Run: python tests/test_determinism.py   (or pytest). Needs ~3 GB of card memory (a full 30-arcsec synthetic DEM).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

try:
    import cupy as cp  # noqa: E402
    HAVE_CARD = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    cp = None
    HAVE_CARD = False

try:
    import pytest  # noqa: E402
    needs_card = pytest.mark.skipif(not HAVE_CARD, reason="needs CuPy and a CUDA card")
except ImportError:  # run as a script
    def needs_card(f):
        return f


def _gwd_once(seed: int = 7):
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.init import init_gwd_orog as gw

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=())
    im, jm = int(g.im), int(g.jm)
    lat, lon = gw._synthetic_geometry(cfg, im, jm)
    st.add("GLAT", (jm, im))[...] = cp.asarray(lat, dtype=cp.float32)
    st.add("GLON", (jm, im))[...] = cp.asarray(lon, dtype=cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    rng = cp.random.RandomState(seed)
    dem = cp.empty((gw.JSRC, gw.ISRC), dtype=cp.int16)
    for j0 in range(0, gw.JSRC, 2000):
        blk = rng.randint(-50, 3000, size=(min(2000, gw.JSRC - j0), gw.ISRC))
        dem[j0:j0 + blk.shape[0]] = blk.astype(cp.int16)
    dem[dem < 0] = -9999
    ctx.tables["gwd_dem"] = dem
    ctx.tables["grid_sw"] = cp.asarray([lat[0, 0], lon[0, 0]], dtype=cp.float32)
    with ctx.stream:
        out = gw.run(ctx)
    ctx.stream.synchronize()
    res = {n: cp.asnumpy(a) for n, a in out.items()}
    del ctx, st, dem, out
    cp.get_default_memory_pool().free_all_blocks()
    return res


@needs_card
def test_gwd_orog_same_bits_twice():
    a = _gwd_once()
    b = _gwd_once()
    bad = {n: int(np.count_nonzero(a[n].view(np.uint32) != b[n].view(np.uint32))) for n in a}
    bad = {n: k for n, k in bad.items() if k}
    assert not bad, f"init-gwd-orog differs between two runs of the same input (words per field): {bad}"
    for n, v in a.items():
        assert np.isfinite(v).all(), f"{n} has NaN/Inf"


@needs_card
def test_vcoord_fit_same_bits_twice():
    from namgpu import kernels
    from namgpu.init import init_vcoord_recover as vc

    im, jm, lm = 954, 835, 60
    rng = np.random.default_rng(3)
    pd = rng.uniform(5.0e4, 9.0e4, size=(jm, im)).astype(np.float32)
    a_true = np.linspace(2.0e3, 3.0e4, lm)
    b_true = np.linspace(0.0, 1.0, lm)
    pres = (a_true[:, None, None] + b_true[:, None, None] * pd[None]).astype(np.float32)
    pres += rng.normal(0.0, 5.0, size=pres.shape).astype(np.float32)
    pt, pdtop = np.float32(200.0), np.float32(3.0e4)
    psfc = cp.asarray(pd + pdtop + pt)
    pres_d = cp.asarray(pres)
    grid3 = kernels.grid_2d(im, jm, lm)
    nblk = int(grid3[0]) * int(grid3[1])

    def once():
        acc = vc._device_acc(lm, nblk)
        a = cp.zeros((lm,), dtype=cp.float32)
        b = cp.zeros((lm,), dtype=cp.float32)
        kernels.get(vc.UNIT, vc.FIT)(grid3, kernels.BLOCK_2D, (
            pres_d, psfc, acc, pt, pdtop, np.int32(im), np.int32(jm), np.int32(lm)))
        kernels.get(vc.UNIT, vc.SOLVE)(((lm + 127) // 128, 1, 1), (128, 1, 1), (
            acc, np.float64((im - 2) * (jm - 2)), a, b, np.int32(lm), np.int32(nblk)))
        return cp.asnumpy(a), cp.asnumpy(b)

    a1, b1 = once()
    a2, b2 = once()
    assert (a1.view(np.uint32) == a2.view(np.uint32)).all() and (b1.view(np.uint32) == b2.view(np.uint32)).all(), \
        "init-vcoord-recover fit differs between two runs of the same input"
    # and it is the least-squares answer (float64 host reference over the b1 columns)
    x = pd[1:-1, 1:-1].astype(np.float64).ravel()
    for l0 in (0, lm // 2, lm - 1):
        y = pres[l0, 1:-1, 1:-1].astype(np.float64).ravel()
        bb, aa = np.polyfit(x, y, 1)
        assert abs(aa - a1[l0]) < 1e-3 * max(1.0, abs(aa)) and abs(bb - b1[l0]) < 1e-5, (l0, aa, a1[l0], bb, b1[l0])


if __name__ == "__main__":
    if not HAVE_CARD:
        print("skipped: needs CuPy and a CUDA card")
        raise SystemExit(0)
    test_vcoord_fit_same_bits_twice()
    print("vcoord fit: same bits twice, matches the float64 least-squares reference")
    test_gwd_orog_same_bits_twice()
    print("gwd orog: same bits twice")
