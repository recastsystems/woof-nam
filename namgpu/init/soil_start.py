"""Soil moisture at a start from another model's land state: the soil-start rows. (lane dewpoint-and-soil-start,
2026-11-08, NAM modernisation).

Breakage it addresses (TD2-FIX.md "Not closed"): GFS soil moisture is drier than the NAM's own land analysis everywhere
(top layer 0.224 vs 0.266 in the Carolinas, 0.166 vs 0.255 in the Plains, 0.251 vs 0.288 over CONUS, 2026-01-23 12Z), and
with the NAM's soil the wedge 2 m dewpoint bias falls from -7 to -4 F at +30 h. Remapping by soil class does not close it.

Rows (NAMGPU_SOIL_START, read by init-native-sfc for every start whose land state is not NMMB Noah's own):
  source  the source's own soil moisture (the default; output unchanged, bit for bit)
  cdf     CDF match of the source soil moisture to the NAM analysis climatology, per grid point and layer
          (piecewise-linear quantile mapping, Reichle and Koster 2004, GRL 31 L19501): theta' = F_NAM^-1(F_GFS(theta)).
          Tables from build-cdf (NAMGPU_SOIL_CDF = the .npz). Urban points come out in the NAM's own urban range, so the
          urban remap is not applied where the match was.
  hrrr    HRRR (RUC LSM, 9 levels) soil moisture at the same valid time wherever HRRR covers the grid, layer-integrated
          onto the 4 Noah layers and averaged over HRRR land cells around each NAM cell; outside HRRR the source's own,
          blended over BLEND_CELLS HRRR cells at the HRRR edge (NAMGPU_SOIL_HRRR = the subset GRIB2 from soil_fetch).
          HRRR is a foreign land model, so the urban remap still applies.
Everything after the decode runs on the GPU.

  python -m namgpu.init.soil_start build-cdf --case-dir CASE --clim DIR --out cdf.npz   (GPU, ~1 s per date)
  python -m namgpu.init.soil_start check --case-dir CASE --cdf cdf.npz --hrrr hrrr.grib2  (case-date soil vs NAM analysis)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

UNIT = "init-soil-start"
ENV, ENV_CDF, ENV_HRRR = "NAMGPU_SOIL_START", "NAMGPU_SOIL_CDF", "NAMGPU_SOIL_HRRR"
MODES = ("source", "cdf", "hrrr")
NQ = 21
SMIN, SMAX = 0.02, 0.6                   # the GFS start's own clip (gfs_start.stage)
MISSING = -9999.0
# HRRR (RUC) soil levels, m below ground, and the Noah layers they are integrated onto
ZRUC = (0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0)
NOAH = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
# HRRR CONUS grid (GRIB2 GDT 30, shape of earth 6, radius 6,371,229 m), checked against the file's inventory
HRRR = dict(nx=1799, ny=1059, lat1=21.138123, lon1=237.280472, lov=262.5, latin=38.5, dx=3000.0, r=6371229.0)
HALF = 2                                 # HRRR cells either side of the nearest one (5 x 5 = 15 km box per 12 km cell)
BLEND_CELLS = 20                         # HRRR cells over which HRRR fades into the source at the HRRR domain edge


def mode():
    m = os.environ.get(ENV, "source").strip().lower() or "source"
    if m not in MODES:
        raise ValueError(f"{UNIT}: {ENV}={m!r}; rows are {MODES}")
    return m


# ---------------------------------------------------------------------------------------------- CDF match
def cdf_apply(x, qg, qn, ok):
    """theta' = F_n^-1(F_g(theta)) per point and layer. x [4, JM, IM]; qg, qn [NQ, 4, JM, IM] quantiles of the source
    and NAM climatologies (ascending); ok [JM, IM] where the tables hold. Below the lowest / above the highest source
    quantile the NAM quantile is shifted by the same amount. Device arrays; returns float32 [4, JM, IM]."""
    import cupy as cp
    x = x.astype(cp.float32)
    nq = int(qg.shape[0])
    idx = (qg <= x[None]).sum(0)                       # 0..NQ
    lo = cp.clip(idx - 1, 0, nq - 2)
    hi = lo + 1
    g0 = cp.take_along_axis(qg, lo[None], 0)[0]
    g1 = cp.take_along_axis(qg, hi[None], 0)[0]
    n0 = cp.take_along_axis(qn, lo[None], 0)[0]
    n1 = cp.take_along_axis(qn, hi[None], 0)[0]
    span = g1 - g0
    w = cp.clip(cp.where(span > 1e-6, (x - g0) / cp.where(span > 1e-6, span, 1.0), 0.5), 0.0, 1.0)
    y = n0 + w * (n1 - n0)
    y = cp.where(idx == 0, qn[0] + (x - qg[0]), y)
    y = cp.where(idx == nq, qn[-1] + (x - qg[-1]), y)
    y = cp.clip(y, SMIN, SMAX)
    return cp.where(ok[None], y, x).astype(cp.float32)


def cdf_table_path(ctx):
    """NAMGPU_SOIL_CDF: a table file, or a folder of monthly tables cdf-MM.npz (picked by the start month), or `auto`
    = $NAMGPU_ROOT/soil_cdf. A month with no table is refused: a table from another season is wrong (lane
    dewpoint-and-soil-start, "Default" item 1)."""
    v = (os.environ.get(ENV_CDF) or "").strip()
    if v == "auto":
        # $NAMGPU_ROOT/soil_cdf when it holds the start month's table, else the woof-nam fetch-soil-cdf folder
        from namgpu.init import soil_cdf_fetch
        start = getattr(ctx, "start", None)
        local = Path(os.environ.get("NAMGPU_ROOT") or ".") / "soil_cdf"
        mm = f"cdf-{start.month:02d}.npz" if start is not None else None
        v = str(local if (mm is None or (local / mm).is_file()) else soil_cdf_fetch.home())
        if mm is not None and not (Path(v) / mm).is_file():
            # the start month's table is not installed: fetch that one month now (pinned size and SHA-256), as the
            # helper programs fetch themselves, instead of stopping in the init after the start is built
            print(f"{UNIT}: {mm} is not installed; fetching it ({soil_cdf_fetch.FETCH_CMD} --months {start.month}, "
                  "about 268 MB)", flush=True)
            try:
                soil_cdf_fetch.fetch([start.month])
            except OSError as e:
                raise FileNotFoundError(f"{UNIT}: {ENV}=cdf could not fetch {mm}: {e}. Run "
                                        f"{soil_cdf_fetch.FETCH_CMD} --months {start.month}, or set {ENV_CDF} to a "
                                        "table or folder") from e
    p = Path(v) if v else None
    if p is not None and p.is_dir():
        start = getattr(ctx, "start", None)
        if start is None:
            raise ValueError(f"{UNIT}: {ENV_CDF}={v} is a folder of monthly tables but the run has no start time")
        p = p / f"cdf-{start.month:02d}.npz"
    if p is None or not p.is_file():
        raise FileNotFoundError(f"{UNIT}: {ENV}=cdf needs {ENV_CDF} = a build-cdf table or a folder of cdf-MM.npz "
                                f"(got {v!r}; resolved {p}); `woof-nam fetch-soil-cdf` installs the monthly tables")
    return p


def _geo_to_rot(lat, lon, tph0d, tlm0d, xp):
    """Geographic (deg) -> NMMB rotated (tph, tlm) (deg) for the pole (tph0d, tlm0d)."""
    t0, l0 = np.radians(tph0d), np.radians(tlm0d)
    la, lo = xp.radians(lat), xp.radians(lon)
    x = np.cos(t0) * xp.cos(la) * xp.cos(lo - l0) + np.sin(t0) * xp.sin(la)
    y = xp.cos(la) * xp.sin(lo - l0)
    z = -np.sin(t0) * xp.cos(la) * xp.cos(lo - l0) + np.cos(t0) * xp.sin(la)
    return xp.degrees(xp.arcsin(xp.clip(z, -1, 1))), xp.degrees(xp.arctan2(y, x))


def resample_to_grid(ctx, qg, qn, ok, meta):
    """Tables built on another rotated lat-lon grid (the 12 km parent configfile unless the table records its own grid
    in meta_grid = 'tph0d tlm0d wbd sbd im jm') sampled at the nearest table point to every run H point, on the GPU.
    Points outside the table grid, or where it does not hold, come out not-ok (the source's own soil there)."""
    import cupy as cp
    if "meta_grid" in meta:
        tph0d, tlm0d, wbd, sbd, im, jm = (float(v) for v in meta["meta_grid"].split())
    else:
        # tables from before meta_grid (cdf-jan.npz): the NAM 12 km parent grid, parm/nam_configfile's own keys.
        # Not config.parse_configfile: it resolves the default path to NAMGPU_CONFIGFILE, i.e. the run's own grid
        # (the na7p6 smoke refused with "table (21, 4, 835, 954) does not match its grid 1318 x 1506").
        tph0d, tlm0d, wbd, sbd, im, jm = 54.0, -106.0, -60.039, -45.036, 954.0, 835.0
    if (int(jm), int(im)) != tuple(qg.shape[2:]):
        raise ValueError(f"{UNIT}: CDF table {tuple(qg.shape)} does not match its grid {int(jm)} x {int(im)}")
    dl, dp = -2 * wbd / (im - 1), -2 * sbd / (jm - 1)
    lat = cp.degrees(cp.asarray(ctx.st["GLAT"], dtype=cp.float64))
    lon = cp.degrees(cp.asarray(ctx.st["GLON"], dtype=cp.float64))
    tph, tlm = _geo_to_rot(lat, lon, tph0d, tlm0d, cp)
    i = cp.rint((tlm - wbd) / dl).astype(cp.int64)
    j = cp.rint((tph - sbd) / dp).astype(cp.int64)
    inside = (i >= 0) & (i < int(im)) & (j >= 0) & (j < int(jm))
    ic, jc = cp.clip(i, 0, int(im) - 1), cp.clip(j, 0, int(jm) - 1)
    qg2 = cp.ascontiguousarray(qg[:, :, jc, ic])
    qn2 = cp.ascontiguousarray(qn[:, :, jc, ic])
    ok2 = ok[jc, ic] & inside
    return qg2, qn2, ok2, f"nearest from {int(jm)}x{int(im)} (pole {tph0d} {tlm0d}), {int(inside.sum())} points inside"


def load_cdf(path):
    import cupy as cp
    z = np.load(path)
    return (cp.asarray(z["qg"], dtype=cp.float32), cp.asarray(z["qn"], dtype=cp.float32),
            cp.asarray(z["ok"].astype(bool)), {k: str(z[k]) for k in z.files if k.startswith("meta_")})


# ---------------------------------------------------------------------------------------------- HRRR
def _lcc_ij(lat_r, lon_r, p=HRRR):
    """Fractional HRRR (i, j) (0-based, j south to north) of points (radians), Lambert conformal tangent at latin."""
    import cupy as cp
    phi1 = np.radians(p["latin"])
    n = np.sin(phi1)
    F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    lov = np.radians(p["lov"])

    def xy(lat, lon, xp):
        rho = p["r"] * F / xp.tan(np.pi / 4 + lat / 2) ** n
        dl = (lon - lov + np.pi) % (2 * np.pi) - np.pi
        th = n * dl
        return rho * xp.sin(th), -rho * xp.cos(th)
    x1, y1 = xy(np.radians(p["lat1"]), np.radians(p["lon1"]), np)
    x, y = xy(lat_r.astype(cp.float64), lon_r.astype(cp.float64), cp)
    return (x - x1) / p["dx"], (y - y1) / p["dx"]


def ruc_to_noah(levels):
    """[9, ...] RUC level values -> [4, ...] Noah layer means (piecewise linear in depth, exact integral)."""
    import cupy as cp
    z = list(ZRUC)
    out = []
    for top, bot in NOAH:
        acc = cp.zeros_like(levels[0])
        for k in range(len(z) - 1):
            a, b = max(z[k], top), min(z[k + 1], bot)
            if b <= a:
                continue
            ta = levels[k] + (levels[k + 1] - levels[k]) * ((a - z[k]) / (z[k + 1] - z[k]))
            tb = levels[k] + (levels[k + 1] - levels[k]) * ((b - z[k]) / (z[k + 1] - z[k]))
            acc = acc + 0.5 * (ta + tb) * (b - a)
        out.append(acc / (bot - top))
    return cp.stack(out)


def decode_hrrr(ctx, path, tmpdir):
    """HRRR subset (soil_fetch hrrr): SOILW at the 9 RUC levels and LAND, [ny, nx] south to north."""
    import cupy as cp
    from namgpu.init import init_grib2_decode as g2
    from namgpu.init.sources import _selsfc
    inv = g2.inventory(path)
    r0 = inv[0]
    if int(r0["gdt"]) != 30 or (int(r0["nx"]), int(r0["ny"])) != (HRRR["nx"], HRRR["ny"]):
        raise ValueError(f"{UNIT}: {path} is GDT {r0['gdt']} {r0['nx']}x{r0['ny']}, not the HRRR CONUS grid")
    for k in ("lat1", "lon1"):
        if k in r0 and abs(float(r0[k]) - HRRR[k]) > 1e-3:
            raise ValueError(f"{UNIT}: {path} {k} {r0[k]} differs from the HRRR grid {HRRR[k]}")
    sel = {f"W{k}": _selsfc((2, 0, 192, "106", f"{z:g}", f"{z:g}")) for k, z in enumerate(ZRUC)}
    sel["LAND"] = _selsfc((2, 0, 0, "1", "0", None))
    # init-grib2-decode refuses GDT 30 because its callers place records as lat-lon or the NAM rotated grid. Here the
    # gather only orders the values (scan bits honoured, south-to-north rows); placement is this module's own Lambert
    # projection (_lcc_ij), checked against the NAM land mask by `check` (agreement inside HRRR).
    prev = g2.SUPPORTED_GDT
    g2.SUPPORTED_GDT = tuple(prev) + (30,)
    try:
        got = g2.run(ctx, path, sel, missing_value=MISSING, tmpdir=tmpdir, expect_shape=(HRRR["nx"], HRRR["ny"]))
    finally:
        g2.SUPPORTED_GDT = prev
    return cp.stack([got[f"W{k}"] for k in range(len(ZRUC))]), got["LAND"]


def hrrr_on_grid(glat, glon, lev, land_h):
    """HRRR soil [4, JM, IM] on the model grid, its weight w [JM, IM] (1 inside, fading to 0 at the HRRR edge, 0
    outside or with no HRRR land in the box)."""
    import cupy as cp
    ny, nx = int(land_h.shape[0]), int(land_h.shape[1])
    layers = ruc_to_noah(lev)                                          # [4, ny, nx]
    good = (land_h > 0.5) & cp.all((lev > 0.0) & (lev < 0.9), axis=0)
    fi, fj = _lcc_ij(glat, glon)
    ii, jj = cp.rint(fi).astype(cp.int32), cp.rint(fj).astype(cp.int32)
    s = cp.zeros((4,) + glat.shape, dtype=cp.float32)
    n = cp.zeros(glat.shape, dtype=cp.float32)
    for dj in range(-HALF, HALF + 1):
        for di in range(-HALF, HALF + 1):
            a, b = ii + di, jj + dj
            inside = (a >= 0) & (a < nx) & (b >= 0) & (b < ny)
            ac, bc = cp.clip(a, 0, nx - 1), cp.clip(b, 0, ny - 1)
            g = inside & good[bc, ac]
            s += cp.where(g[None], layers[:, bc, ac], 0.0)
            n += g.astype(cp.float32)
    val = s / cp.maximum(n, 1.0)[None]
    edge = cp.minimum(cp.minimum(fi, nx - 1 - fi), cp.minimum(fj, ny - 1 - fj))
    w = cp.clip((edge - HALF) / BLEND_CELLS, 0.0, 1.0) * (n > 0)
    return val.astype(cp.float32), w.astype(cp.float32)


# ---------------------------------------------------------------------------------------------- the hook
def apply(ctx, raw, source):
    """Soil moisture [4, JM, IM] for a foreign-land start under the configured row, and the mask of land points whose
    value is already in NMMB Noah's own range (the urban remap skips them), or None. Records what it did in
    ctx.tables['grid_source']['soil_start']."""
    import cupy as cp
    m = mode()
    x = raw["SOILW"]
    if m == "source":
        return x, None
    land = raw["LAND"] > 0.5
    rec = {"row": m}
    if m == "cdf":
        path = cdf_table_path(ctx)
        qg, qn, ok, meta = load_cdf(path)
        if tuple(qg.shape[1:]) != tuple(x.shape):
            qg, qn, ok, how = resample_to_grid(ctx, qg, qn, ok, meta)
            meta = dict(meta, resampled=how)
        ok = ok & land & (cp.asarray(raw["VGTYP"]) != 13)     # urban: init-native-sfc's urban_soilw (TD2-FIX)
        y = cdf_apply(x, qg, qn, ok)
        keep = ok
        rec.update(table=str(path), points=int(ok.sum()), **meta)
        del qg, qn
    else:
        path = os.environ.get(ENV_HRRR)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"{UNIT}: {ENV}=hrrr needs {ENV_HRRR} = the HRRR soil subset (got {path!r})")
        tmp = str(Path(ctx.case_dir).resolve().parents[1] / "scratch") if getattr(ctx, "case_dir", None) else None
        lev, land_h = decode_hrrr(ctx, path, tmp)
        val, w = hrrr_on_grid(ctx.st["GLAT"], ctx.st["GLON"], lev, land_h)
        w = cp.where(land, w, 0.0)
        y = cp.clip(w[None] * val + (1.0 - w[None]) * x, SMIN, SMAX).astype(cp.float32)
        keep = None
        rec.update(file=str(path), full=int((w >= 1.0).sum()), blended=int(((w > 0) & (w < 1)).sum()),
                   source_only=int((land & (w <= 0)).sum()))
        del lev, land_h, val, w
    d = (y - x)[0][land]
    rec.update(top_mean_change=float(d.mean()), land_points=int(land.sum()))
    gs = ctx.tables.get("grid_source")
    if isinstance(gs, dict):
        gs["soil_start"] = rec
    print(f"{UNIT}: {source} soil moisture, row {m}: {rec}", flush=True)
    return cp.ascontiguousarray(y), keep


# ---------------------------------------------------------------------------------------------- build / check
def _ctx(case_dir):
    from types import SimpleNamespace
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.init import init_grid_geo
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=())
    for k in ("GLAT", "GLON", "VLAT", "VLON"):
        st.add(k, (int(g.jm), int(g.im)))
    ctx = SimpleNamespace(cfg=cfg, grid=g, st=st, tables={}, stream=None, case_dir=str(case_dir), first_step=True)
    init_grid_geo.run(ctx)
    return ctx


def nam_soilw(ctx, path, tmp):
    import cupy as cp
    from namgpu.init import sources as srcmod
    got, _ = srcmod.decode_native(ctx, Path(path), [], [f"SOILW{k}" for k in range(1, 5)], int(ctx.grid.lm), tmp,
                                  lambda n, a: None, top_first=True)
    return cp.stack([got[f"SOILW{k}"] for k in range(1, 5)]).astype(cp.float32)


def gfs_soilw(ctx, path, tmp):
    """GFS 0.25 SOILW on the model grid exactly as the GFS start builds it (gfs_start.stage): source-land ring fill,
    init-gfs-hinterp, clip 0.02-0.6."""
    import cupy as cp
    from namgpu.init import init_grib2_decode as g2, init_gfs_hinterp as hin, gfs_start as gs, source_records as srec
    from namgpu.init.sources import _selsfc
    sfc = srec.GFS["sfc"]
    keys = [f"SOILW{k}" for k in range(1, 5)] + ["LANDSRC"]
    got = g2.run(ctx, path, {k: _selsfc(sfc[k]) for k in keys}, missing_value=MISSING, tmpdir=tmp,
                 expect_shape=(1440, 721))
    land_src = got["LANDSRC"] > 0.5
    out = []
    for k in range(1, 5):
        a = got[f"SOILW{k}"]
        a, _ = gs._ring_fill(a, (a == MISSING) | ~land_src, lon_periodic=True)
        o, _ = hin.interp_full(cp.ascontiguousarray(a, dtype=cp.float32)[None], cp.ones((1,) + a.shape, cp.uint8),
                               ctx.st["GLAT"], ctx.st["GLON"])
        out.append(cp.clip(o[0], SMIN, SMAX))
    return cp.stack(out)


def _smooth(q, ok, r):
    """Land-only (2r+1)^2 box mean of q [..., JM, IM] (device)."""
    import cupy as cp
    w = ok.astype(cp.float32)
    num = cp.zeros_like(q)
    den = cp.zeros_like(w)
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            num += cp.roll(cp.roll(q * w, dj, -2), di, -1)
            den += cp.roll(cp.roll(w, dj, -2), di, -1)
    return cp.where(ok, num / cp.maximum(den, 1.0), q)


def build_cdf(case_dir, clim, out, smooth=2):
    import time
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor
    from namgpu.init import gfs_start as gs
    t0 = time.time()
    ctx = _ctx(case_dir)
    static, static_from = gs.load_static(ctx)
    land = static["LAND"] > 0.5
    sot = cp.rint(static["SOTYP"]).astype(cp.int32)
    tmp = str(Path(clim) / "scratch")
    Path(tmp).mkdir(exist_ok=True)
    dates = sorted(p.stem for p in (Path(clim) / "nam").glob("*.grib2")
                   if (Path(clim) / "gfs" / p.name).is_file())
    if len(dates) < 20:
        raise RuntimeError(f"{UNIT}: only {len(dates)} paired dates in {clim}")
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    N = cp.empty((len(dates), 4, jm, im), dtype=cp.float16)
    G = cp.empty_like(N)
    used = []

    def dec(d):
        return (d, nam_soilw(ctx, Path(clim) / "nam" / f"{d}.grib2", tmp),
                gfs_soilw(ctx, Path(clim) / "gfs" / f"{d}.grib2", tmp))
    n = 0
    with ThreadPoolExecutor(8) as ex:          # the decodes are grib2_dump processes (host I/O), 8 dates at once
        for d, a, b in ex.map(dec, dates):
            N[n] = a.astype(cp.float16)
            G[n] = b.astype(cp.float16)
            used.append(d)
            n += 1
    N, G = N[:n], G[:n]
    nf = N.astype(cp.float32)
    # valid: model land, not land ice (STAS 16), NAM soil a real soil value on every date
    # Urban cells (IGBP 13) are left out: NMMB Noah runs them in their own 0.40-0.45 range and init-native-sfc's
    # urban_soilw already maps them there (TD2-FIX); smoothing them into a land box diluted that range (0.338 against
    # the NAM's 0.428 in the first build) and leaked it into their neighbours' tables.
    urban = cp.rint(static["VGTYP"]).astype(cp.int32) == 13
    ok = land & ~urban & (sot != 16) & cp.all((nf > 0.0) & (nf < 0.7), axis=(0, 1))
    del nf
    p = cp.linspace(0.0, 1.0, NQ, dtype=cp.float32)

    def quant(A):
        A = cp.sort(A.astype(cp.float32), axis=0)
        pos = p * (A.shape[0] - 1)
        lo = cp.floor(pos).astype(cp.int32)
        hi = cp.minimum(lo + 1, A.shape[0] - 1)
        f = (pos - lo)[:, None, None, None]
        return A[lo] * (1 - f) + A[hi] * f
    qn = quant(N)
    del N
    qg = quant(G)
    del G
    if smooth:
        qn = _smooth(qn, ok, smooth)
        qg = _smooth(qg, ok, smooth)
    qn = cp.sort(qn, axis=0)
    qg = cp.sort(qg, axis=0)
    nm, gm = float(qn[NQ // 2, 0][ok].mean()), float(qg[NQ // 2, 0][ok].mean())
    np.savez(out, qg=cp.asnumpy(qg).astype(np.float16), qn=cp.asnumpy(qn).astype(np.float16), ok=cp.asnumpy(ok),
             dates=np.array(used), meta_dates=f"{used[0]}..{used[-1]} ({len(used)} dates, 12Z)",
             meta_smooth=f"{2 * smooth + 1}x{2 * smooth + 1} land box", meta_static=str(static_from),
             meta_method="per-point per-layer 21-quantile piecewise-linear CDF match, GFS 0.25 f000 -> NAM bgrd3d00",
             meta_grid=" ".join(str(ctx.cfg.raw[k]) for k in ("tph0d", "tlm0d", "wbd", "sbd", "im", "jm")))
    print(f"{UNIT}: build-cdf {len(used)} dates, {int(ok.sum())} land points, median top layer NAM {nm:.3f} "
          f"GFS {gm:.3f}, {time.time() - t0:.0f} s -> {out}", flush=True)


BOXES = {"carolinas": (33.0, 38.0, -84.0, -75.0), "plains": (33.0, 43.0, -103.0, -94.0),
         "conus": (25.0, 50.0, -125.0, -67.0)}


def check(case_dir, cdf=None, hrrr=None):
    """Case-date top-layer soil moisture means: GFS start, each row, NAM's own analysis (the target)."""
    import cupy as cp
    from namgpu.init import gfs_start as gs
    case_dir = Path(case_dir)
    ctx = _ctx(case_dir)
    static, _ = gs.load_static(ctx)
    land = static["LAND"] > 0.5
    tmp = str(case_dir.resolve().parents[1] / "scratch")
    Path(tmp).mkdir(exist_ok=True)
    gfsf = sorted((case_dir / "raw" / "gfs").glob("gfs.t*z.pgrb2.0p25.f000"))[0]
    g = gfs_soilw(ctx, gfsf, tmp)
    nam = nam_soilw(ctx, case_dir / "raw" / "nam" / "bgrd3d00.grib2", tmp)
    rows = {"gfs (source)": g, "nam analysis": nam}
    if cdf:
        qg, qn, ok, _ = load_cdf(cdf)
        rows["cdf"] = cdf_apply(g, qg, qn, ok & land)
    if hrrr:
        lev, lh = decode_hrrr(ctx, hrrr, tmp)
        val, w = hrrr_on_grid(ctx.st["GLAT"], ctx.st["GLON"], lev, lh)
        w = cp.where(land, w, 0.0)
        rows["hrrr"] = cp.clip(w[None] * val + (1 - w[None]) * g, SMIN, SMAX)
        # registration check: HRRR land fraction on NAM land and NAM water cells inside HRRR
        fi, fj = _lcc_ij(ctx.st["GLAT"], ctx.st["GLON"])
        ii, jj = cp.rint(fi).astype(cp.int32), cp.rint(fj).astype(cp.int32)
        inside = (ii >= 0) & (ii < HRRR["nx"]) & (jj >= 0) & (jj < HRRR["ny"])
        hl = lh[cp.clip(jj, 0, HRRR["ny"] - 1), cp.clip(ii, 0, HRRR["nx"] - 1)] > 0.5
        agree = float(((hl == land) & inside).sum() / inside.sum())
        print(f"hrrr weight: full {int((w >= 1).sum())}, blended {int(((w > 0) & (w < 1)).sum())}, "
              f"none {int((land & (w <= 0)).sum())} land points; HRRR/NAM land-mask agreement inside HRRR {agree:.4f}")
    lat, lon = cp.degrees(ctx.st["GLAT"]), cp.degrees(ctx.st["GLON"])
    lon = cp.where(lon > 180, lon - 360, lon)
    print("layer 1 (0-10 cm) / layer 2 (10-40 cm) mean m3/m3 on model land:")
    print(f"{'row':16s}" + "".join(f"{b:>22s}" for b in BOXES))
    for name, a in rows.items():
        s = f"{name:16s}"
        for b, (la0, la1, lo0, lo1) in BOXES.items():
            m = land & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
            s += f"{float(a[0][m].mean()):>11.3f} /{float(a[1][m].mean()):>8.3f} "
        print(s)
    m = land & (lat >= 25) & (lat <= 50) & (lon >= -125) & (lon <= -67)
    for name, a in rows.items():
        if name != "nam analysis":
            print(f"{name:16s} CONUS layer-1 vs NAM analysis: MAE {float(cp.abs(a[0] - nam[0])[m].mean()):.4f}, "
                  f"bias {float((a[0] - nam[0])[m].mean()):+.4f}; layer 2 MAE {float(cp.abs(a[1] - nam[1])[m].mean()):.4f}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build-cdf")
    b.add_argument("--case-dir", required=True)
    b.add_argument("--clim", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--smooth", type=int, default=2)
    c = sp.add_parser("check")
    c.add_argument("--case-dir", required=True)
    c.add_argument("--cdf")
    c.add_argument("--hrrr")
    a = ap.parse_args(argv)
    if a.cmd == "build-cdf":
        build_cdf(a.case_dir, a.clim, a.out, a.smooth)
    else:
        check(a.case_dir, a.cdf, a.hrrr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
