"""init-ugwp-oro: the twenty GSL (UGWP) orographic drag statistics on the NMMB B grid.

The UGWP drag suite (namgpu.physics.gwd.phys_ugwp_driver, --gwd ugwp) reads, per H point, the RRFS
`ugwp_oro_data` fields: var2d, con, oa1-4 and ol1-4 for the large-scale ("ls") gravity-wave drag and
blocking, and the same ten for the small-scale ("ss") gravity-wave drag and turbulent form drag.
NOAA publishes that file only for the RRFS meshes, so the NAM grid builds its own here.

How: tools/ugwp-oro (rw_hex_ugwp_oro, Rust, every core; the gpuwm-hex builder written from Kim and
Arakawa 1995, Kim and Doyle 2005, Beljaars et al. 2004 and the GSL drag-suite documentation, validated
against RRFS's conus12km and conus3km files) reads the WPS_GEOG 30-arc-second GMTED2010 terrain over a
lat/lon box of side sqrt(dxh * dyh) centred on each H point. The result is cached per grid (origin,
spacing, size, window, terrain index, method) under NAMGPU_UGWP_CACHE (default
~/.cache/woof-nam/ugwp-oro), so a grid pays the build once.

Refusals (named breakage, no fallback): the Rust binary or the terrain tiles missing (the drag would
run on zero statistics, i.e. silently off); a non-finite statistic (NaN drag tendencies in every
column downstream).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

UNIT = "init_ugwp_oro"
METHOD = "rw_hex_ugwp_oro hex-0.4.14 highpass20 ss1.5 ls3.5 x2"
VARIABLES = ("var2dls", "conls", "oa1ls", "oa2ls", "oa3ls", "oa4ls", "ol1ls", "ol2ls", "ol3ls", "ol4ls",
             "var2dss", "conss", "oa1ss", "oa2ss", "oa3ss", "oa4ss", "ol1ss", "ol2ss", "ol3ss", "ol4ss")
FIELDS = tuple("UGWP_" + v.upper() for v in VARIABLES)
HERE = Path(__file__).resolve().parents[2]


def topo_dir() -> Path:
    roots = [Path(os.environ["NAMGPU_GEOG"])] if os.environ.get("NAMGPU_GEOG") else []
    roots += [Path.home() / "WPS_GEOG"]
    for r in roots:
        t = r / "topo_gmted2010_30s"
        if (t / "index").is_file():
            return t
    raise RuntimeError(f"{UNIT}: DEFECT: no topo_gmted2010_30s/index under {[str(r) for r in roots]}; "
                       "UGWP would run on zero orography statistics (drag silently off). Set NAMGPU_GEOG.")


def binary() -> Path:
    cands = [os.environ.get("NAMGPU_UGWP_ORO_BIN", "")]
    cands += [str(HERE / "tools" / "ugwp-oro" / "target" / "release" / "rw_hex_ugwp_oro"),
              str(Path.home() / ".gpuwm" / "bridges" / "rw_hex_ugwp_oro")]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    # an install has no checkout to build in: the pinned release asset (woof-nam fetch-bridges; namgpu/helpers.py)
    from namgpu import exporter, helpers
    try:
        return helpers.ensure("rw_hex_ugwp_oro")
    except exporter.ExporterMissing as e:
        raise RuntimeError(f"{UNIT}: DEFECT: rw_hex_ugwp_oro not found (looked in {[c for c in cands if c]}): {e}; "
                           "or build it with `cargo build --release` in tools/ugwp-oro, or set "
                           "NAMGPU_UGWP_ORO_BIN.") from None


def _key(g, topo: Path) -> str:
    idx = (topo / "index").read_bytes()
    h = hashlib.sha256()
    h.update(json.dumps([int(g.im), int(g.jm), float(g.dlm), float(g.dph), float(g.wb), float(g.sb),
                         float(g.tph0), float(g.tlm0), int(g.win_i0), int(g.win_j0), METHOD]).encode())
    h.update(idx)
    return h.hexdigest()[:20]


def build(g) -> np.ndarray:
    """[20, JM, IM] float32 statistics for grid g (namgpu.grid.BGrid), from the cache or the Rust builder."""
    topo = topo_dir()
    cache = Path(os.environ.get("NAMGPU_UGWP_CACHE", str(Path.home() / ".cache" / "woof-nam" / "ugwp-oro")))
    cache.mkdir(parents=True, exist_ok=True)
    jm, im = int(g.jm), int(g.im)
    key = _key(g, topo)
    out = cache / f"ugwp-oro-{im}x{jm}-{key}.f32"
    if not out.is_file():
        exe = binary()
        n = im * jm
        side = np.sqrt(np.asarray(g.dxh, dtype=np.float64)[:, None] * float(g.dyh)) * np.ones((1, im))
        cells = cache / f".cells-{key}-{os.getpid()}.bin"
        tmp = cache / f".{out.name}.{os.getpid()}"
        with open(cells, "wb") as f:
            f.write(np.uint64(n).tobytes())
            f.write(np.ascontiguousarray(g.glat_h, dtype="<f8").tobytes())
            f.write(np.ascontiguousarray(g.glon_h, dtype="<f8").tobytes())
            f.write(np.ascontiguousarray(side, dtype="<f8").tobytes())
        t0 = time.perf_counter()
        r = subprocess.run([str(exe), "--cells", str(cells), "--topo", str(topo), "--out", str(tmp)],
                           capture_output=True, text=True)
        cells.unlink(missing_ok=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{UNIT}: DEFECT: rw_hex_ugwp_oro failed ({r.returncode}): {r.stderr.strip()[-400:]}")
        os.replace(tmp, out)
        print(f"{UNIT}: built {out.name} in {time.perf_counter() - t0:.1f} s (Rust, every core); "
              f"receipt {r.stdout.strip()[-300:]}", flush=True)
    a = np.fromfile(out, dtype="<f4")
    if a.size != 20 * im * jm:
        raise RuntimeError(f"{UNIT}: DEFECT: {out} holds {a.size} values, expected {20 * im * jm}")
    a = a.reshape(20, jm, im)
    if not np.isfinite(a).all():
        raise RuntimeError(f"{UNIT}: DEFECT: {out} has non-finite statistics (NaN drag downstream)")
    return a


def main(argv=None) -> int:
    """Build (or find in the cache) the statistics for the configured grid and print their ranges."""
    from namgpu import config, grid as gridmod
    g = gridmod.build(config.load())
    a = build(g)
    for k, v in enumerate(VARIABLES):
        x = a[k]
        print(f"  {v:8s} min {x.min():10.4g} mean {x.mean():10.4g} max {x.max():10.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
