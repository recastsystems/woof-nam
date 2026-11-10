"""ruc-hrrr-soilclass: the HRRR static soil categories (BNU 30-arc-sec top-layer soil type, the soil set of NCEP's
operational HRRR static file) as the dominant class on a WOOF-NAM grid, on the GPU.

    python -m namgpu.init.ruc_hrrr_soilclass --config parm/nam_configfile --geog DIR --out static/nam12_hrrrsoil.npz

Only the soil block of init-static-build (namgpu.init.static_build: the same gcell accumulation, nearest fill for
empty cells, dominant non-water class, loam where no land pixel), with soil set "hrrr" (bnu_soiltype_top). Writes
SOTYP [JM, IM] int32. The RUC driver (NAMGPU_RUC_SOIL_STATIC) uses it on the RUC land points where it also takes the
HRRR soil analysis, so HRRR's soil moisture is read against HRRR's own soil classes. Licence: owner ruling 2026-10-21, the
HRRR static soil is used as the operational HRRR uses it (US government data), opt-in.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from namgpu.init import static_build as sb


def fetch_bnu(roots):
    """The BNU top-layer soil set is not staged: fetch it with gpuwm's own pinned fetch-geog (an 8 MB download, about
    0.9 GB unpacked) into the first geog root, as the README's setup line does, instead of stopping the run here."""
    import subprocess
    import sys
    root = roots[0] if roots else Path.home() / "WPS_GEOG"
    cmd = [sys.executable, "-m", "gpuwm", "fetch-geog", "--datasets", "bnu_soiltype_top", "--root", str(root)]
    print(f"ruc-hrrr-soilclass: bnu_soiltype_top is not staged; fetching it into {root} "
          "(gpuwm fetch-geog --datasets bnu_soiltype_top, 8 MB)", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not (root / "bnu_soiltype_top" / "index").is_file():
        raise SystemExit(f"ruc-hrrr-soilclass: bnu_soiltype_top/index not under {[str(r) for r in roots]}, and "
                         f"fetching it failed (rc {rc}). Run: gpuwm fetch-geog --datasets bnu_soiltype_top --root {root}")
    return root / "bnu_soiltype_top"


def build(cfg, geog):
    import cupy as cp
    roots = sb.geog_roots(geog)
    path = next((r / "bnu_soiltype_top" for r in roots if (r / "bnu_soiltype_top" / "index").is_file()), None)
    if path is None:
        path = fetch_bnu(roots)
    dom = sb.Domain(cfg)
    t = time.perf_counter()
    so = sb.Source(path)
    so.load(dom.tiles_touched(so))
    cnt = sb._launch_gcell(dom, so, 0, so.ncat)
    empty = cnt.sum(axis=0) == 0
    if bool(empty.any()):
        v, g = sb._launch_sample(dom, so, 0, ("nearest",), empty)
        cat = cp.clip(cp.rint(v).astype(cp.int64), 1, so.ncat)
        for c in range(1, so.ncat + 1):
            cnt[c - 1] += (empty & g & (cat == c)).astype(cp.int32)
    sl, mx = sb._dominant(cnt, exclude=(sb.SOIL_WATER,))
    sl = cp.where(mx > 0, sl, sb.SOIL_LOAM)
    H = dom.halo
    out = cp.asnumpy(sl[H:H + dom.jm, H:H + dom.im]).astype(np.int32)
    print(f"ruc-hrrr-soilclass: {path} -> {dom.jm}x{dom.im} in {time.perf_counter() - t:.1f} s (GPU), "
          f"classes {np.unique(out).tolist()}", flush=True)
    return out


def main(argv=None):
    from namgpu import config
    ap = argparse.ArgumentParser(prog="ruc-hrrr-soilclass")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--geog", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = config.load(a.config) if a.config else config.load()
    sot = build(cfg, a.geog)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, SOTYP=sot, source=np.array("bnu_soiltype_top (HRRR static soil)"))


if __name__ == "__main__":
    main()
