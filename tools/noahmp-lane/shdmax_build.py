"""shdmax_build.py OUT_NPZ [--config CFG] [--geog DIR]: SHDMAX for Noah-MP (dveg=4) on an NMMB grid, on the GPU.

SHDMAX is what WRF's geogrid/real give Noah-MP: the annual maximum of the monthly MODIS green-vegetation fraction
(greenfrac_fpar_modis), percent. Same source, interpolation sequence and land mask as init-static-build's VEG
(namgpu/init/static_build.py: average_gcell(4.0)+four_pt+average_4pt+average_16pt+search, water 0), taken over all
12 months instead of interpolated to the valid date. phys-noahmp-driver reads it through NAMGPU_NOAHMP_SHDMAX."""
import argparse, json, time
from pathlib import Path
import numpy as np


def main(argv=None):
    import cupy as cp
    from namgpu import config
    from namgpu.init import static_build as sb
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--geog", default=None)
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    t0 = time.perf_counter()
    roots = sb.geog_roots(a.geog)
    ds = sb.resolve_datasets(roots, None, soil=sb.soil_set(None))
    dom = sb.Domain(cfg)
    src = lambda role, **kw: sb.Source(Path(ds[role]["path"]), **kw)   # noqa: E731
    H = dom.halo
    crop = (slice(H, H + dom.jm), slice(H, H + dom.im))
    lu = src("landuse")
    lu.load(dom.tiles_touched(lu))
    cnt = sb._launch_gcell(dom, lu, 0, lu.ncat)
    tot = cnt.sum(axis=0)
    water = (cnt[sb.IGBP_WATER - 1] + cnt[sb.IGBP_LAKE - 1]).astype(cp.float64) / cp.maximum(tot, 1)
    land = (water < 0.5) & (tot > 0)
    del lu, cnt
    mx = None
    for m in range(12):
        s = src("greenfrac", planes=(m,))
        s.load(dom.tiles_touched(s))
        v = cp.nan_to_num(sb._continuous(dom, s, 0, sb.SEQ_MONTHLY, True, land), nan=0.0)
        mx = v if mx is None else cp.maximum(mx, v)
        del s
    shd = cp.where(land, mx * 100.0, 0.0)
    out = cp.asnumpy(cp.ascontiguousarray(shd[crop])).astype(np.float32)
    rec = {"unit": "shdmax-build", "dataset": ds["greenfrac"]["dataset"], "path": str(ds["greenfrac"]["path"]),
           "grid": [int(dom.im), int(dom.jm)], "months": 12, "seconds": round(time.perf_counter() - t0, 1),
           "land_mean_percent": round(float(out[out > 0].mean()), 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, SHDMAX=out, META=np.array(json.dumps(rec)))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
