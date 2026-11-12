"""woof-nam products: the hourly NAM product files, in the shape NOAA published them.

The 12 km CONUS parent writes the awphys files on grid 218 (below); a nest writes NOAA's nest files,
<prefix>.tHHz.<nest>.hiresfFF.tm00.grib2, on that nest's NOAA output grid (conusnest, alaskanest, hawaiinest,
priconest, firewxnest; any other nest on a Lambert box around it): `--set`, or found from the run
(namgpu/products/sets.py lists the sets; each is a record table and a grid, no code path of its own).

For every history file of a run this writes <out>/woof-nam.tHHz.awphysFF.tm00.grib2 (HH = start hour, FF = forecast
hour) on NCEP grid 218 (Lambert conformal, 614 x 428, 12.19 km, true latitude 25 N, orientation 265 E: the grid of
NOAA's nam.tHHz.awphysFF.tm00.grib2), a wgrib2-style inventory beside it (<file>.idx: record:byte
offset:d=YYYYMMDDHH:VAR:level:time:), and one index for the run (<out>/index.json: file, bytes, records, valid time).

What each file carries, and in what order, is the record table namgpu/products/awphys.tsv: one row per record
of NOAA's awphys file, in NOAA's order, with its discipline/category/parameter, fixed surfaces, statistic and the
source of its plane (namgpu/products/spec.py documents the columns). Rows with no source yet are left out and
named in the run index. The GRIB2 encoding is gpuwm's Rust exporter (`rw_rrfs_upp --profile-json`: the writer of
NOAA-shaped field sets, wx-core's GRIB2 writer, complex packing, every core), which computes the planes with
the WOOF post (rw-post) from one WRF-shaped grid-218 frame. woof-nam's part is that hand-off: the history fields,
regridded from the NMMB rotated lat-lon grid to grid 218 (bilinear, namgpu/io/io_map_regrid.py tables on NCEP's
own sphere) with the winds turned grid-relative as NCEP writes them.

`woof-nam products --compare <archived awphys file> <ours>` checks the grid section and every table record's
inventory identity against NOAA's file.

    woof-nam products --run runs/2025-03-14_12z                  # every history hour of the run
    woof-nam products --run runs/2025-03-14_12z --hours 0-24
    woof-nam products --run runs/2025-03-14_12z --follow        # while the forecast runs (woof-nam run --products)
    woof-nam products --compare cases/.../raw/nam/awphys24.grib2 runs/.../products/woof-nam.t12z.awphys24.tm00.grib2

Reads the history only (no CUDA context in this process); the writer needs a CUDA device (its derived records run
on a card only, whatever --post-device says), and with none visible products stops before the first hour. Each hour
runs as its own process; the exporter runs its own worker threads.
The writer is woof-nam's pinned rw_rrfs_upp (namgpu/exporter.py): the first run downloads it from the woof-nam
release and checks its SHA-256, or `woof-nam fetch-bridges` does that ahead of time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu import cardgate
from namgpu.products import hiresf_carriers as HC
from namgpu.products import land_cloud as LC
from namgpu.products import sets as PSETS

NAM_LEVELS_HPA = list(range(50, 1001, 25))           # the NAM awphys isobaric set: 50-1000 hPa every 25 hPa
RW = "rw_rrfs_upp"                       # gpuwm's NOAA-shaped field-set writer (rw-grib2export crate)
FILE_NAME = "woof-nam.t{HH}z.awphys{FF}.tm00.grib2"
DEFAULT_BITS = 20
# history 2-D fields carried to the exporter as they are (U10/V10 are rotated, below)
PASS_2D = ("HGT", "PSFC", "T2", "Q2", "RAINNC", "RAINC", "TSK", "SST", "SEAICE", "SNOW", "SNOWH", "CANWAT", "TMN",
           "LANDMASK", "SNOWC", "COMPOSITE", "MSLP")
# the NCEP product carriers (io_history_fields.PRODUCT_CARRIERS_2D), as the history names them
PASS_2D += ("TWBS", "QWBS", "USTAR", "Z0", "PBLH", "AKMS", "ALBEDO", "RSWIN", "RLWIN", "RSWOUT", "RADOT", "THS",
            "SICE", "SM", "SNO", "SI", "CMC", "TG", "VEGFRC", "ISLTYP", "IVGTYP", "CFRACL", "CFRACM", "CFRACH",
            "PSHLTR", "POTEVP", "SFCEVP", "T02MAX", "T02MIN", "RH02MAX", "RH02MIN", "SPD10MAX", "REFDMAX",
            "UPVVELMAX", "DNVVELMAX", "PRATEMAX", "UPHLMAX25", "UPHLMIN25", "UPHLMAX03", "UPHLMIN03")
# earth-relative 2-D wind pairs the hand-off turns grid-relative (as U10/V10): the hour's 10 m wind extreme
WIND_2D_EXTRA = ("U10MAX", "V10MAX")
PASS_3D = ("T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "REFL_10CM", "CLDFRA", "TKE_PBL")
# the nest (hiresf) frame also carries the rime factor (RIME records) and the top-of-atmosphere longwave (BRTMP)
PASS_3D_NEST = ("F_RIMEF",)
PASS_2D_NEST = ("RLWTOA",)
# the frame carrier that is 0 off the model domain (rw_rrfs_upp mask_carrier: those points are missing, bitmap)
MASK = "WOOF_VALID"
# the NMM isobaric rule's inputs (rw_rrfs_upp isobaric_rule nmm-isobaric): the total w on the mid layers (W_TOT, NMMB's
# history 'vvel') and the model interface pressures (PINT), when the history has them
PASS_NMM = ("W_TOT", "PINT")
# categories and masks: the nearest source point, never a bilinear blend of class numbers
NEAREST = ("ISLTYP", "IVGTYP", "SM", "LANDMASK")
# NAM's accumulation buckets (awphys): since the last 12 h reset and since the last 3 h reset. The hand-off carries
# each window's total, differenced from the history file at the window start (RAINNC/RAINC are totals since the
# start), as APCP_S12/ACPCP_S12/APCP_S3/ACPCP_S3 (kg m-2); the table labels them with the writer's since:Nh rules.
# The snowfall water (WEASD, accumulation) has NAM's 12 h window only (0-1 ... 0-12 h, then 12-13 ...; NCEP's archived
# awphys files): WEASD_S12, rebuilt from the model's 3 h snowfall bucket ACSNOW (kg m-2, emptied every PRECIPBUCKET = 3 h
# by drv-bucket-reset after that hour's history is written) as the buckets that ended inside the window plus the
# current one.
BUCKETS_H = (12, 3)
SNOW_BUCKET_H = 3
NEED = ("T", "QVAPOR", "PRES", "PH", "U", "V", "HGT", "PSFC", "T2", "Q2", "U10", "V10", "RAINNC", "RAINC")
# The convective scheme's rain at the output time, for the writer's surface visibility (rw_rrfs_upp NAM_VIS reads the
# frame carrier CONV_CARRIER). The history's CPRATE is the BMJ precipitation of the last physics step, m of water per
# DT*NPHS s (the same field and step the history reflectivity's convective term uses).
CONV_RATE = "CPRATE"
CONV_CARRIER = "NAM_CONV_PRATE"


def conv_prate(cprate, ctx):
    """The frame's NAM_CONV_PRATE (kg m-2 s-1): the history CPRATE (m of water per physics step) over the physics step
    DT*NPHS (ctx["dtq2"], s, from the run's configfile: _run_ctx), times 1000 kg m-3; negative values (regrid
    round-off) are 0. None when the history has no CPRATE or the step is unknown.

    Breakage it fixes: the BMJ scheme rains without leaving rain water in the column, and the writer's visibility read
    only the column's water, so a cell under convective rain had the visibility of clear air. Against NCEP's archived
    NAM awphys files (2024-11-05 12Z, +12 h) the cells whose rain was convective in both read 22.9 km in woof-nam and
    13.2 km in the NAM."""
    if cprate is None or not ctx or not ctx.get("dtq2"):
        return None
    a = np.maximum(np.asarray(cprate, dtype=np.float32), np.float32(0.0))
    return (a * np.float32(1000.0 / float(ctx["dtq2"]))).astype(np.float32)


# ---------------------------------------------------------------------------------------------- the hand-off


def frame_q2(q_specific):
    """The hand-off frame's Q2: the history Q2 (NMMB QSHLTR, a specific humidity) as a mixing ratio, q / (1 - q).

    The frame is WRF-shaped and the WOOF post reads the WRF name Q2 as WRF's 2 m mixing ratio: it turns it into a
    specific humidity (w / (1 + w)) for the 2 m SPFH record and derives the 2 m DPT and RH from it. Handed the
    specific humidity itself, the WOOF post applied that conversion to a value that was already converted, so the 2 m
    SPFH record read q / (1 + q): 1.2 % low on the 2024-11-05 12Z analysis (2 % at 20 g/kg), the 2 m dew point
    0.13 K low on average and 0.3 K low over warm water, the 2 m RH 1.1 % low, at every hour, against NCEP's own
    awphys file from the same state. The 3-D QVAPOR of the history is already a mixing ratio."""
    q = np.asarray(q_specific, dtype=np.float64)
    return (q / (1.0 - q)).astype(np.float32)

_TABLES = None


def _history_files(run_dir: Path) -> list:
    return sorted(Path(run_dir).glob("wrfout_d01_*")) or sorted((Path(run_dir) / "run").glob("wrfout_d01_*"))


def _is_parent12(history: Path) -> bool:
    """True when a history is on the 12 km CONUS parent grid the grid-218 (awphys) tables are cut from."""
    from netCDF4 import Dataset
    from namgpu import config
    cfg = config.load()
    with Dataset(history) as d:
        nx, ny = len(d.dimensions["west_east"]), len(d.dimensions["south_north"])
        title = str(getattr(d, "TITLE", ""))
    return (nx, ny) == (int(cfg.im), int(cfg.jm)) and "nest" not in title.lower()


def detect_set(run_dir: Path) -> str | None:
    """The product set a run's history writes: awphys for the 12 km CONUS parent; for a nest, the NCEP nest its run
    names (nest-receipt.json `ncep_nest`, or the NCEP name in its domain), else the generic hiresf set. None when no
    history file is there yet (--follow)."""
    run_dir = Path(run_dir)
    hist = _history_files(run_dir)
    if not hist:
        return None
    try:
        if _is_parent12(hist[0]):
            return "awphys"
    except (ImportError, OSError, KeyError):
        return None
    if run_dir.name in PSETS.NEST_SETS and run_dir.name != "hiresf":
        return run_dir.name                    # a suite run's nest folder (<out>/conusnest/, woof-nam nest --concurrent)
    for rc in (run_dir / "nest-receipt.json", run_dir.parent / "nest-receipt.json"):
        if rc.is_file():
            try:
                d = json.loads(rc.read_text())
            except ValueError:
                continue
            nn = d.get("ncep_nest")                # the door writes {"name": ..., "window": ...}
            name = str((nn.get("name") if isinstance(nn, dict) else nn) or "")
            dom = str(d.get("domain") or "")
            for cand in (name, *[k for k in PSETS.NEST_SETS if k in dom]):
                if cand in PSETS.SETS and cand not in ("awphys", "hiresf"):
                    return cand
    return "hiresf"


class SourceGrid:
    """The NMMB rotated lat-lon H grid of a history, from the history alone: the rotation centre (POLE_LAT/POLE_LON)
    and the rotated coordinates of its first points (XLAT/XLONG), so any nest's history regrids without its
    configfile."""

    def __init__(self, history):
        import netCDF4
        from namgpu.io.io_map_regrid import geo_to_rotated
        with netCDF4.Dataset(history) as s:
            lat = np.asarray(s["XLAT"][0], dtype=np.float64)
            lon = np.asarray(s["XLONG"][0], dtype=np.float64)
            self.tph0 = np.radians(float(s.POLE_LAT))
            self.tlm0 = np.radians(float(s.POLE_LON))
            self.dx_m = float(getattr(s, "DY", 3000.0))
        self.jm, self.im = lat.shape
        tl, tp = geo_to_rotated(self.tph0, self.tlm0, np.radians(lat[:2, :2]), np.radians(lon[:2, :2]))
        self.wb, self.sb = float(tl[0, 0]), float(tp[0, 0])
        self.dlm, self.dph = float(tl[0, 1] - tl[0, 0]), float(tp[1, 0] - tp[0, 0])
        self.lat, self.lon = lat, lon

    def key(self):
        return (round(float(np.degrees(self.tph0)), 6), round(float(np.degrees(self.tlm0)), 6), self.im, self.jm,
                round(self.wb, 9), round(self.sb, 9), round(self.dlm, 12), round(self.dph, 12))


def build_tables_out(g, out) -> dict:
    """Tables from an NMMB H grid `g` to an output grid `out` (sets.OutGrid): bilinear corner and fractions, the
    validity mask (inside the model domain), the nearest point, and 25 sub-cell nearest points per output point for
    NCEP's budget interpolation (wgrib2 -new_grid_interpolation budget: the mean over a 5 x 5 sub-grid of the output
    cell)."""
    from namgpu.io.io_map_regrid import geo_to_rotated

    def frac(lat, lon):
        tlm, tph = geo_to_rotated(g.tph0, g.tlm0, np.radians(lat), np.radians(lon))
        xi = ((tlm - g.wb + np.pi) % (2.0 * np.pi) - np.pi) / g.dlm
        yj = (tph - g.sb) / g.dph
        return xi, yj
    lat, lon = out.latlon()
    xi, yj = frac(lat, lon)
    valid = (xi >= -0.5) & (xi <= g.im - 0.5) & (yj >= -0.5) & (yj <= g.jm - 0.5)
    i0 = np.clip(np.floor(xi), 0, g.im - 2).astype(np.int32)
    j0 = np.clip(np.floor(yj), 0, g.jm - 2).astype(np.int32)
    wx = np.clip(xi - i0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(yj - j0, 0.0, 1.0).astype(np.float32)
    near = (np.clip(np.rint(yj), 0, g.jm - 1).astype(np.int64) * g.im
            + np.clip(np.rint(xi), 0, g.im - 1).astype(np.int64)).ravel()
    jj, ii = np.meshgrid(np.arange(out.ny, dtype=np.float64), np.arange(out.nx, dtype=np.float64), indexing="ij")
    sub = []
    for dj in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for di in (-0.4, -0.2, 0.0, 0.2, 0.4):
            la, lo = out.latlon_ij(ii + di, jj + dj)
            x, y = frac(la, lo)
            ok = (x >= -0.5) & (x <= g.im - 0.5) & (y >= -0.5) & (y <= g.jm - 0.5)
            k = (np.clip(np.rint(y), 0, g.jm - 1).astype(np.int64) * g.im
                 + np.clip(np.rint(x), 0, g.im - 1).astype(np.int64))
            sub.append(np.where(ok, k, -1).ravel().astype(np.int32))
    return {"i0": i0, "j0": j0, "wx": wx, "wy": wy, "valid": valid.astype(np.uint8),
            "lat": lat.astype(np.float32), "lon": (((lon + 180.0) % 360.0) - 180.0).astype(np.float32),
            "_near": near, "_sub": np.stack(sub)}


def _tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = _build_tables()
    return _TABLES


_SET_TABLES: dict = {}


def nest_window(sg, og, margin: int = 2):
    """(i0, j0, window OutGrid) of the part of output grid `og` the NMMB grid `sg` covers, `margin` points wider.
    The writer computes every record on the window and writes it on the whole grid, missing outside (WOOF_FULL_*):
    a nest that covers part of NOAA's grid costs its own size, not the grid's."""
    from dataclasses import replace
    from namgpu.io.io_map_regrid import geo_to_rotated
    lat, lon = og.latlon()
    tlm, tph = geo_to_rotated(sg.tph0, sg.tlm0, np.radians(lat), np.radians(lon))
    xi = ((tlm - sg.wb + np.pi) % (2.0 * np.pi) - np.pi) / sg.dlm
    yj = (tph - sg.sb) / sg.dph
    ok = (xi >= -0.5) & (xi <= sg.im - 0.5) & (yj >= -0.5) & (yj <= sg.jm - 0.5)
    if not ok.any():
        raise SystemExit(f"the nest does not reach the {og.name or og.kind} output grid: no product point lies on it")
    jj, ii = np.nonzero(ok)
    i0, i1 = max(int(ii.min()) - margin, 0), min(int(ii.max()) + margin, og.nx - 1)
    j0, j1 = max(int(jj.min()) - margin, 0), min(int(jj.max()) + margin, og.ny - 1)
    la1, lo1 = og.latlon_ij(float(i0), float(j0))
    win = replace(og, nx=i1 - i0 + 1, ny=j1 - j0 + 1, la1=float(la1), lo1=float(lo1) % 360.0, la2=None, lo2=None)
    return i0, j0, win


def pad_attrs(og, i0: int, j0: int) -> dict:
    """The frame attributes that put a window's records on the whole output grid (rw_rrfs_upp read_pad)."""
    if og.la2 is not None and og.lo2 is not None:
        la2, lo2 = og.la2, og.lo2
    else:
        la2, lo2 = (float(v) for v in og.latlon_ij(float(og.nx - 1), float(og.ny - 1)))
    return {"WOOF_FULL_NX": np.int32(og.nx), "WOOF_FULL_NY": np.int32(og.ny), "WOOF_I0": np.int32(i0),
            "WOOF_J0": np.int32(j0), "WOOF_FULL_LA1": np.float64(og.la1), "WOOF_FULL_LO1": np.float64(og.lo1 % 360.0),
            "WOOF_FULL_LA2": np.float64(la2), "WOOF_FULL_LO2": np.float64(lo2 % 360.0)}


_SG_KEYS: dict = {}


def tables_for(gkey):
    """(tables, output frame) for a job's grid key: None = grid 218 from the 12 km parent (awphys); else
    (set name, sets.OutGrid, history path[, pad]) = that output grid (or window of it) from the history's own NMMB
    grid. Cached per process; the history's grid key is read once per path (a pool worker otherwise read the
    history's XLAT/XLONG again for every field)."""
    if gkey is None:
        return _tables()
    name, out, hist = gkey[:3]
    if hist not in _SG_KEYS:
        sg = SourceGrid(hist)
        _SG_KEYS[hist] = (sg.key(), sg)
    key, sg = _SG_KEYS[hist]
    k = (name, out, key)
    if k not in _SET_TABLES:
        _SET_TABLES[k] = (build_tables_out(sg, out), out)
    return _SET_TABLES[k]


def _build_tables():
    from namgpu import config, grid as gridmod
    from namgpu.io.io_map_regrid import NCEP218, build_tables
    g = gridmod.build(config.load())
    t = build_tables(g, NCEP218)
    if not t["valid"].all():
        raise RuntimeError(f"grid 218: {int((t['valid'] == 0).sum())} points outside the NMMB domain")
    return t, NCEP218


def _regrid(t, a):
    """Bilinear from NMMB H points [.., jm, im] to grid 218 [.., ny, nx] (the io_map_regrid kernel's formula)."""
    jm, im = a.shape[-2:]
    if t.get("_im") != (jm, im):
        k00 = (t["j0"].astype(np.int64) * im + t["i0"]).ravel()
        t.update(_im=(jm, im), _k=(k00, k00 + 1, k00 + im, k00 + im + 1),
                 _fx=t["wx"].ravel(), _fy=t["wy"].ravel())
    flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, jm * im)
    v00, v10, v01, v11 = (np.take(flat, k, axis=1) for k in t["_k"])
    fx, fy = t["_fx"], t["_fy"]
    bot = v00 + fx * (v10 - v00)
    top = v01 + fx * (v11 - v01)
    out = (bot + fy * (top - bot)).astype(np.float32)
    return out.reshape(a.shape[:-2] + t["wx"].shape)


def _budget(t, a):
    """NCEP's budget interpolation (nest APCP, WEASD, SNOD): the mean of the nearest source values at 5 x 5 sub-points
    of each output cell (sub-points off the model domain left out)."""
    jm, im = a.shape[-2:]
    flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, jm * im)
    sub = t["_sub"]
    acc = np.zeros((flat.shape[0], sub.shape[1]), np.float64)
    cnt = np.zeros(sub.shape[1], np.float64)
    for k in sub:
        ok = k >= 0
        acc[:, ok] += np.take(flat, k[ok], axis=1)
        cnt += ok
    out = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    return out.reshape(a.shape[:-2] + t["wx"].shape)


def _nearest(t, a):
    """The source H point nearest each output point (the bilinear corner with the largest weight)."""
    jm, im = a.shape[-2:]
    if "_near" in t:
        flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, jm * im)
        return np.take(flat, t["_near"], axis=1).reshape(a.shape[:-2] + t["wx"].shape)
    j = t["j0"].astype(np.int64) + (t["wy"] >= 0.5)
    i = t["i0"].astype(np.int64) + (t["wx"] >= 0.5)
    flat = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, jm * im)
    return np.take(flat, (j * im + i).ravel(), axis=1).reshape(a.shape[:-2] + t["wx"].shape)


def _stagger(a, axis):
    """Mass points -> the staggered points around them along `axis` (ends copied, interior means)."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    out = np.empty((n + 1,) + a.shape[1:], dtype=np.float32)
    out[0], out[n] = a[0], a[n - 1]
    np.add(a[:-1], a[1:], out=out[1:n])
    out[1:n] *= np.float32(0.5)
    return np.moveaxis(out, 0, axis)


_WT = None




def _derived_tsk(s):
    """Return the hand-off frame's TSK, the skin temperature (K), as a float32 [j, i] array, or None when the history lacks an input.

    TSK = THS * (PSFC / 1e5) ** KAPPA: Poisson's equation at the surface pressure, THS being the surface potential
    temperature (namgpu/io/poisson.py gives the sources and the exponent)."""
    if "THS" not in s.variables or "PSFC" not in s.variables:
        return None
    from namgpu.io import poisson
    ths = np.asarray(s["THS"][0], dtype=np.float32)
    psfc = np.asarray(s["PSFC"][0], dtype=np.float32)
    return poisson.temperature(ths, psfc)


def _derived_landmask(s):
    if "SM" not in s.variables:
        return None
    return (1.0 - np.asarray(s["SM"][0], dtype=np.float32)).astype(np.float32)


# Frame fields the history writer (io-history-fields) does not write under the WRF name, made from the NMMB state
# fields it does write: TSK (awphys rec 299 TMP surface) and LANDMASK (rec 278 HINDEX is land only; NMMB SM is 1 over
# water). A history that has the WRF name is read as it is.
DERIVED = {"TSK": _derived_tsk, "LANDMASK": _derived_landmask}


def _interp_for(ps, name, mode=None):
    """The regrid of one frame field for a product set: awphys bilinear (classes and masks nearest); a nest set
    nearest-neighbour everywhere and NCEP's budget for the APCP/WEASD/SNOD inputs, as nam_prdgen_nest.sh."""
    if ps.interp == "nearest":
        return _budget if (name in ps.budget_fields or mode == "budget") else _nearest
    if mode is not None:
        return _nearest if mode == "nearest" else _regrid
    return _nearest if name in ps.nearest_fields else _regrid


def _read_regrid(job):
    """Worker: read one history field and regrid it to the output grid (one field per process: the netCDF decode and
    the gather run on every core, not one). A namgpu.products.land_cloud (or hiresf_carriers) carrier name is
    computed from the history on the model grid first (that module's formula), then regridded."""
    global _WT
    history, name = job[0], job[1]
    gkey = job[2] if len(job) > 2 else None
    cctx = job[3] if len(job) > 3 else None
    ps = PSETS.AWPHYS if gkey is None else PSETS.get(gkey[0])
    import netCDF4
    if gkey is None:
        if _WT is None:
            _WT = _tables()
        tab = _WT[0]
    else:
        tab = tables_for(gkey)[0]
    if name in LC.CARRIERS:
        a, mode = LC.compute(history, name)
        if a is None:
            return name, None
        dims = LC.SOIL_DIMS if a.ndim == 3 else ("south_north", "west_east")
        return name, (_interp_for(ps, name, mode)(tab, a), dims)
    if name in HC.CARRIERS:
        a, mode = HC.compute(history, name, cctx)
        if a is None:
            return name, None
        return name, (_interp_for(ps, name, mode)(tab, a), ("south_north", "west_east"))
    with netCDF4.Dataset(history) as s:
        if name not in s.variables and name in DERIVED:
            a = DERIVED[name](s)
            if a is None:
                return name, None
            return name, (_interp_for(ps, name)(tab, a), ("Time", "south_north", "west_east"))
        if name not in s.variables:
            return name, None
        v = s[name]
        if v.dimensions[-2:] != ("south_north", "west_east"):
            return name, None
        a = np.asarray(v[0], dtype=np.float32)
        dims = v.dimensions
    return name, (_interp_for(ps, name)(tab, a), dims)


def _read_regrid_to(arg):
    """Worker: _read_regrid, with the regridded array left as <where>/<name>.npy in host RAM for the main process to
    map, instead of pickled back through the pool's pipe (a 3 km nest hour is 7 GB of arrays: received and unpickled
    one after the other in the main process, that was the frame's longest serial step after the pool)."""
    where, job = arg
    _die_with_parent()
    name, got = _read_regrid(job)
    if got is None or where is None:
        return name, got
    a, dims = got
    path = os.path.join(where, f"{name}.npy")
    np.save(path, np.ascontiguousarray(a), allow_pickle=False)
    return name, (path, dims)


_PDEATH_SET = False


def _die_with_parent() -> None:
    """In a field-pool worker (the pool's initializer, so also in a worker that never gets a job): a daemon thread
    that ends the worker when the products process that made the pool has ended (checked every 2 s). Breakage it
    prevents: a products process stopped by SIGKILL left its field workers and their forkserver running for good (nam-differences-2, the CPU products job stopped at 12:21Z on
    2026-11-11: 13 processes removed by hand). Nothing else ends them: a forkserver child holds the forkserver's
    "alive" pipe and both ends of the pool's call queue, so the forkserver never sees its clients go, the workers never
    read an end of file, and a parent-death signal (bound to the forkserver) never fires."""
    global _PDEATH_SET
    if _PDEATH_SET:
        return
    _PDEATH_SET = True
    import multiprocessing
    import threading
    parent = multiprocessing.parent_process()
    pid = getattr(parent, "pid", None)
    if not pid:
        return

    def watch():
        while True:
            time.sleep(2.0)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                os._exit(1)
            except OSError:              # alive, someone else's (EPERM): keep going
                continue
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().rsplit(")", 1)[1].split()[0] == "Z":     # ended, not yet reaped
                        os._exit(1)
            except (OSError, IndexError):
                pass
    threading.Thread(target=watch, name="products-parent-watch", daemon=True).start()


def _mapped(v):
    """A pool result as the frame uses it: (array, dims), the array mapped from host RAM when the worker left it there."""
    if v is None or not isinstance(v[0], str):
        return v
    return np.load(v[0], mmap_mode="r", allow_pickle=False), v[1]


# The longest field-pool jobs, first: a carrier that reads several 3-D fields itself, then every 3-D field. Breakage
# it prevents: NAM_CLDBASE (seven 3-D reads, 15 s on the CONUS nest) was queued last and ran alone after every other
# job had finished, the pool's tail.
HEAVY_FIRST = ("NAM_CLDBASE", "NAM_TCDC")


def _job_order(jobs, three_d) -> list:
    rank = {n: i for i, n in enumerate(HEAVY_FIRST)}
    return sorted(jobs, key=lambda j: rank.get(j[1], len(rank) + (0 if j[1] in three_d else 1)))


def _frame_shm(history: Path, est_bytes: int):
    """A fresh folder in host RAM for one frame's regridded fields ($NAMGPU_PRODUCTS_SHM, default /dev/shm), or None
    with a printed line when it has no room (the fields then come back through the pool's pipe, as before)."""
    from namgpu.io import scratch as scr
    root = Path(os.environ.get("NAMGPU_PRODUCTS_SHM", "") or str(scr.root() or "/dev/shm"))
    free = scr.free_bytes(root)        # free space, less what NAMGPU_SHM_CAP_MIB leaves the package
    if free < 2 * est_bytes:
        print(f"woof-nam products: {history.name}: {root} has {free / 1e9:.1f} GB free, under twice this frame's "
              f"{est_bytes / 1e9:.1f} GB: its fields come back through the pool's pipe (slower)", flush=True)
        return None
    scr.own(root)                      # this process is alive for a sweep in any PID namespace (scratch.own)
    return tempfile.mkdtemp(prefix=f"woof-nam-frame-{os.getpid()}-", dir=root)


# The four precipitation-type carriers (awphys records 336-339, namgpu.products.ptype) are computed here, in the main
# process, from one classification per hour, and the CuPy pool is emptied after it. Breakage it prevents: in the
# field pool every worker assigned one of them made its own CUDA context and kept its CuPy pool (8 workers x 2 GB =
# 16.6 GB in the measured failure), so the awphys writer beside them ran out of GPU memory on every hour, and the
# classification ran up to four times per hour.
PTYPE_CARRIERS = ("NAM_CRAIN", "NAM_CSNOW", "NAM_CFRZR", "NAM_CICEP")


def _ptype_carriers(history: Path, gkey=None, cctx=None) -> dict:
    names = [n for n in PTYPE_CARRIERS if n in LC.CARRIERS]
    # card work: behind the run's card gate (it waits while the forecast claims the card), until the pool is emptied
    with cardgate.hold(Path(history).parent):
        try:
            out = dict(map(_read_regrid, [(str(history), n, gkey, cctx) for n in names]))
        finally:
            _free_pools()
    return out


def _mslet_carrier(history: Path, t, ps):
    """The frame's MSLP carrier, (plane on the output grid, dims) or None: MSLET, the membrane sea-level pressure
    (namgpu.products.memslp), from this history's state, here in the main process (on the device when CuPy sees one,
    as the type carriers; the pool is emptied after), then regridded as every continuous field of the set. rw_rrfs_upp
    writes it as MSLET (carrier MSLP) and takes the 1000 hPa height from it. The history MSLP, the model's in-run
    QUASIPOST membrane field (the tracker's), is not used: it sits under NCEP's MSLET over high ground (NAM analysis
    2024-11-05 12Z, +3 h on grid 218, mean absolute difference 0.52 hPa at 1,500-2,500 m and 0.91 above; this
    carrier: 0.09 and 0.08). A history without T, QVAPOR, PRES, PINT or HGT gives None: MSLET is left out and named in
    the index, never the QUASIPOST field."""
    import netCDF4
    from namgpu.products import memslp

    def free():
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001 - no CuPy or no device: it ran on numpy
            pass
    # on a full shared card it waits and reruns on the GPU, as the writer does (_card_wait)
    counted = reruns = 0
    while True:
        err = None
        # card work: behind the run's card gate, held only while this attempt has memory on the card (not while it
        # waits to rerun: the forecast's claim waits for every hold to end)
        with cardgate.hold(Path(history).parent):
            try:
                with netCDF4.Dataset(history) as s:
                    a = memslp.from_history(s)
            except Exception as e:  # noqa: BLE001 - CuPy's OutOfMemoryError or a CUDA allocation error
                err = e
            free()
        if err is None:
            break
        if counted >= OOM_RETRIES or not (type(err).__name__ == "OutOfMemoryError" or _is_oom(str(err).lower())):
            raise err
        c, why, _ = _card_wait(Path(history).parent, f"MSLET of {Path(history).name}", counted)
        counted += c
        reruns += 1
        print(f"woof-nam products: {Path(history).name}: MSLET out of GPU memory, rerun {reruns} ({why})",
              file=sys.stderr, flush=True)
    if a is None:
        return None
    return _interp_for(ps, "MSLP")(t, a), ("Time", "south_north", "west_east")


def frame_carriers(ps) -> list:
    """Every carrier variable the frame of product set ps holds: the land and cloud carriers, the four
    precipitation-type carriers among them, and the nest extras. Breakage it prevents: the 0.2.0 fold wrote the frame
    from the field pool's job list, which leaves the type carriers out on purpose (_ptype_carriers computes them in
    the main process), so they were classified every hour and never written: no door wrote CSNOW, CICEP, CFRZR or
    CRAIN, on the parent or any nest set (package gate 2026-11-09: 444 awphys records where 448 were due, 4 mapped
    records missing in every set)."""
    return list(LC.CARRIERS) + (list(HC.CARRIERS) if ps.extra_carriers else [])


def wrf_frame(history: Path, dst: Path, pool=None, t=None, f=None, ps=None, gkey=None, cctx=None) -> dict:
    """Write the WRF-shaped frame rw_grib2export reads from one WOOF-NAM history file: on grid 218 (awphys, the
    default), or with `ps`/`gkey` (a nest product set and its output grid, tables_for) on that set's grid. pool: a
    process pool that reads and regrids the fields in parallel (else one after the other here)."""
    import netCDF4
    ps = ps or PSETS.AWPHYS
    if t is None:
        t, f = _tables() if gkey is None else tables_for(gkey)
    nest = gkey is not None
    with netCDF4.Dataset(history) as s:
        missing = [n for n in NEED if n not in s.variables]
        if missing:
            raise RuntimeError(f"{history.name}: products need {missing} (history written with --history-fields "
                               f"full or products has them)")
        names = list(dict.fromkeys(PASS_3D + ("PRES", "PH", "U", "V", "W") + PASS_2D + ("U10", "V10")
                                   + WIND_2D_EXTRA + PASS_NMM + (CONV_RATE,)
                                   + ((PASS_3D_NEST + PASS_2D_NEST) if nest else ())))
        names = [n for n in names if n in s.variables or n in DERIVED]
    # the frame's MSLP is MSLET (_mslet_carrier), computed in this process, not the history's MSLP
    names = [n for n in names if n != "MSLP"]
    carriers = [n for n in frame_carriers(ps) if n not in PTYPE_CARRIERS]      # the field pool's share
    if gkey is None:
        jobs = [(str(history), n) for n in names] + [(str(history), n) for n in carriers]
    else:
        jobs = [(str(history), n, gkey, cctx) for n in names + carriers]
    three_d = set(PASS_3D + ("PRES", "PH", "U", "V", "W") + PASS_NMM + PASS_3D_NEST)
    jobs = _job_order(jobs, three_d)
    shm = None
    if pool is not None:
        ny0, nx0 = t["wx"].shape
        with netCDF4.Dataset(history) as s0:
            nz0 = s0.dimensions["bottom_top"].size
        shm = _frame_shm(history, 4 * ny0 * nx0 * ((nz0 + 1) * len(three_d) + len(names) + len(carriers) + 16))
    try:
        return _wrf_frame_write(history, dst, pool, t, f, ps, gkey, cctx, nest, jobs, shm)
    finally:
        if shm is not None:
            shutil.rmtree(shm, ignore_errors=True)


def _wrf_frame_write(history, dst, pool, t, f, ps, gkey, cctx, nest, jobs, shm) -> dict:
    import netCDF4
    # the pool takes every job now (Executor.map submits them all); the precipitation types are classified here in
    # the main process meanwhile, then the pool's results are collected
    it = pool.map(_read_regrid_to, [(shm, j) for j in jobs]) if pool is not None else map(_read_regrid, jobs)
    ptype = _ptype_carriers(history, gkey, cctx)
    mslet = _mslet_carrier(history, t, ps)
    got = {name: _mapped(v) for name, v in it}
    got.update(ptype)
    got["MSLP"] = mslet
    with netCDF4.Dataset(history) as s:
        get = lambda n: got[n][0]  # noqa: E731
        lat, lon = t["lat"].astype(np.float32), t["lon"].astype(np.float32)
        if not nest:
            # earth-relative -> grid-relative on the Lambert grid (NCEP grid 218 winds are grid-relative)
            cone = np.sin(np.radians(f.lat1))
            alpha = np.radians(((lon - f.lov + 180.0) % 360.0) - 180.0) * cone
            ca, sa = np.cos(alpha).astype(np.float32), np.sin(alpha).astype(np.float32)
        else:
            # earth-relative -> grid-relative on the set's grid (NCEP's nest files: wgrib2 -new_grid_winds grid)
            ca, sa = f.turn(lat, lon)

        def to_grid(ue, ve):
            return ue * ca - ve * sa, ue * sa + ve * ca

        ny, nx = lat.shape
        nz = s.dimensions["bottom_top"].size
        soil = [got[n][0] for n in LC.SOIL if got.get(n) is not None]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(dst, "w", format="NETCDF4") as d:
            for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", nx), ("south_north", ny),
                               ("bottom_top", nz), ("bottom_top_stag", nz + 1), ("west_east_stag", nx + 1),
                               ("south_north_stag", ny + 1)):
                d.createDimension(name, size)
            if soil:
                d.createDimension(LC.SOIL_DIMS[0], soil[0].shape[0])
            S2, S3 = ("Time", "south_north", "west_east"), ("Time", "bottom_top", "south_north", "west_east")

            def put(name, dims, data):
                v = d.createVariable(name, "f4", dims)
                v[0] = data

            tv = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
            tv[:] = s["Times"][:]
            xt = d.createVariable("XTIME", "f4", ("Time",))
            xt[:] = s["XTIME"][:]
            put("XLAT", S2, lat)
            put("XLONG", S2, lon)
            for n in PASS_2D:
                if got.get(n) is not None and got[n][1] == S2:
                    put(n, S2, frame_q2(get(n)) if n == "Q2" else get(n))
            # NAM QSHLTR is specific humidity. The Rust LCL record reads this explicit carrier; the frame's Q2
            # (above) is the same field as a mixing ratio, which is what the WOOF post takes the WRF name Q2 to be.
            put("NAM_Q2_SPECIFIC", S2, get("Q2"))
            u10, v10 = to_grid(get("U10"), get("V10"))
            put("U10", S2, u10)
            put("V10", S2, v10)
            if got.get("U10MAX") is not None and got.get("V10MAX") is not None:
                ux, vx = to_grid(get("U10MAX"), get("V10MAX"))
                put("U10MAX", S2, ux)
                put("V10MAX", S2, vx)
            for n in PASS_3D + (PASS_3D_NEST if nest else ()):
                if got.get(n) is not None:
                    put(n, S3, get(n))
            if nest:
                for n in PASS_2D_NEST:
                    if got.get(n) is not None:
                        put(n, S2, get(n))
                put(MASK, S2, t["valid"].astype(np.float32))
            for name, plane in _buckets(history, s, get, t, ps).items():
                put(name, S2, plane)
            # the convective rain at the output time for the writer's visibility (conv_prate); the physics step is
            # the run's (a nest's cctx, else the configfile beside the parent run)
            conv = conv_prate(got[CONV_RATE][0] if got.get(CONV_RATE) is not None else None,
                              cctx if (cctx or {}).get("dtq2") else _run_ctx(Path(history).parent))
            if conv is not None:
                put(CONV_CARRIER, S2, conv)
            for name in frame_carriers(ps):           # soil, land, snow, flux, radiation, cloud, type, nest records
                if got.get(name) is not None:
                    put(name, ("Time",) + got[name][1], got[name][0])
            pres = get("PRES")
            put("PB", S3, pres)                       # WRF splits pressure into P + PB; the sum is the pressure
            put("P", S3, np.zeros_like(pres))
            # NMMB's mid-layer pressure is hydrostatic (PD and the hybrid coefficients): the exporter takes the
            # interfaces from it by moist hydrostatic integration, not from a WRF eta column
            put("P_HYD", S3, pres)
            ph = get("PH")
            put("PH", ("Time", "bottom_top_stag", "south_north", "west_east"), ph)
            put("PHB", ("Time", "bottom_top_stag", "south_north", "west_east"), np.zeros_like(ph))
            u, v = to_grid(get("U"), get("V"))
            put("U", ("Time", "bottom_top", "south_north", "west_east_stag"), _stagger(u, 2))
            put("V", ("Time", "bottom_top", "south_north_stag", "west_east"), _stagger(v, 1))
            if got.get("W") is not None:
                put("W", ("Time", "bottom_top_stag", "south_north", "west_east"), _stagger(get("W"), 0))
            # NMMB holds w on the mid layers: the exporter takes DZDT and omega from it there as it is (destaggering
            # the interface copy above would smooth it 1-2-1). the products use the total w (W + BARO, history W_TOT);
            # a history written before W_TOT was carried has only the bare W, and the run index says so.
            w_src = "W_TOT" if got.get("W_TOT") is not None else ("W" if got.get("W") is not None else None)
            if w_src is not None:
                put("W_MASS", S3, get(w_src))
            # The exporter's interface pressures: WRF's hybrid coordinate, p_int = C3F MU + C4F + P_TOP with
            # MU + MUB the column below P_TOP. NMMB's is the same form (PINT = PT + SG1 PDTOP + SG2 PD), so the
            # exact 1-D coefficients come from the history itself: each mid-layer pressure is a + b PSFC across
            # the grid (fit per level), and NMMB's interfaces bracket the mid layer evenly (PMID is their mean).
            psfc = get("PSFC")
            c3f, c4f, ptop_i = _hybrid(history, pres=pres, psfc=psfc)
            d.createVariable("P_TOP", "f4", ("Time",))[:] = np.float32(ptop_i)
            put("MUB", S2, psfc - np.float32(ptop_i))
            put("MU", S2, np.zeros_like(psfc))
            d.createVariable("C3F", "f4", ("Time", "bottom_top_stag"))[0] = c3f
            d.createVariable("C4F", "f4", ("Time", "bottom_top_stag"))[0] = c4f
            c3h, c4h = 0.5 * (c3f[:-1] + c3f[1:]), 0.5 * (c4f[:-1] + c4f[1:])
            d.createVariable("C3H", "f4", ("Time", "bottom_top"))[0] = c3h
            d.createVariable("C4H", "f4", ("Time", "bottom_top"))[0] = c4h
            p0 = 1.0e5 - ptop_i
            d.createVariable("ZNW", "f4", ("Time", "bottom_top_stag"))[0] = (c3f + c4f / p0).astype(np.float32)
            d.createVariable("ZNU", "f4", ("Time", "bottom_top"))[0] = (c3h + c4h / p0).astype(np.float32)
            # The turn from grid-218-relative vectors to NAM's native rotated lat-lon orientation, the frame NAM's own
            # post leaves its non-UGRD/VGRD vector records in (VUCSH/VVCSH: rms 0.6-0.7 m/s against the archive that
            # way, 2-4 m/s grid-218-relative): u_native = u COSB - v SINB, v_native = u SINB + v COSB.
            # NCEP's nest files turn every vector grid-relative (wgrib2 regrid), so there the turn is the identity
            cb, sb = _native_turn(lat, lon, ca, sa) if ps.native_vectors else (np.ones_like(lat), np.zeros_like(lat))
            put("NATIVE_COSB", S2, cb)
            put("NATIVE_SINB", S2, sb)
            # WRF's map rotation, u_earth = u COSALPHA - v SINALPHA (the inverse of to_grid)
            # the model interface pressures for the NMM isobaric rule: the history's PINT, else the hybrid
            # reconstruction above (hydrostatic, from PRES and PSFC)
            pint = get("PINT") if got.get("PINT") is not None else (
                c3f[:, None, None] * (psfc - np.float32(ptop_i))[None] + c4f[:, None, None] + np.float32(ptop_i))
            put("PINT", ("Time", "bottom_top_stag", "south_north", "west_east"), pint.astype(np.float32))
            put("COSALPHA", S2, ca)
            put("SINALPHA", S2, -sa)
            put("MAPFAC_M", S2, np.ones_like(lat))
            cy, cx = ny // 2, nx // 2
            if not nest:
                proj = {"TITLE": "WOOF-NAM 12 km on NCEP grid 218 (woof-nam products hand-off)",
                        "MAP_PROJ": np.int32(1), "MAP_PROJ_CHAR": "Lambert Conformal",
                        "TRUELAT1": np.float32(f.lat1), "TRUELAT2": np.float32(f.lat2),
                        "STAND_LON": np.float32(f.lov), "MOAD_CEN_LAT": np.float32(f.lat1),
                        "DX": np.float32(f.dx), "DY": np.float32(f.dy)}
            else:
                fa = f.frame_attrs()
                proj = {"TITLE": f"WOOF-NAM nest on the {ps.name} output grid (woof-nam products hand-off)",
                        "MAP_PROJ": np.int32(fa["MAP_PROJ"]), "MAP_PROJ_CHAR": fa["MAP_PROJ_CHAR"],
                        "TRUELAT1": np.float32(fa["TRUELAT1"]), "TRUELAT2": np.float32(fa["TRUELAT2"]),
                        "STAND_LON": np.float32(fa["STAND_LON"]), "MOAD_CEN_LAT": np.float32(fa["TRUELAT1"]),
                        "DX": np.float32(fa["DX"]), "DY": np.float32(fa["DY"])}
            d.setncatts({
                **proj,
                "START_DATE": s.START_DATE, "SIMULATION_START_DATE": s.SIMULATION_START_DATE,
                "CEN_LAT": np.float32(lat[cy, cx]), "CEN_LON": np.float32(lon[cy, cx]),
                "POLE_LAT": np.float32(90.0), "POLE_LON": np.float32(0.0),
                "GRID_ID": np.int32(1), "PARENT_ID": np.int32(0),
                "WEST-EAST_GRID_DIMENSION": np.int32(nx + 1), "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1),
                "BOTTOM-TOP_GRID_DIMENSION": np.int32(nz + 1),
                "DT": np.float32(25.0 if not nest else float((cctx or {}).get("dt") or 25.0)),
                "MP_PHYSICS": np.int32(5), "BL_PBL_PHYSICS": np.int32(2), "SF_SFCLAY_PHYSICS": np.int32(2),
                "SF_SURFACE_PHYSICS": np.int32(2), "CU_PHYSICS": np.int32(2),
                "BUCKET_MM": np.float32(-1.0), "BUCKET_J": np.float32(-1.0), "HYBRID_OPT": np.int32(2)})
            if gkey is not None and len(gkey) > 3:
                d.setncatts(dict(gkey[3]))
        valid = datetime.strptime(str(netCDF4.chartostring(np.asarray(s["Times"][0]))), "%Y-%m-%d_%H:%M:%S")
        start = datetime.strptime(s.START_DATE, "%Y-%m-%d_%H:%M:%S")
    notes = []
    if conv is None and got.get(CONV_RATE) is not None:
        notes.append("VIS without the convective rain term: no nam_configfile beside the run gives the physics step")
    if w_src != "W_TOT":
        notes.append(f"DZDT/VVEL from the history's {w_src or 'no'} w: it has no W_TOT (W + BARO, the total vertical velocity)")
    if got.get("PINT") is None:
        notes.append("isobaric heights over interface pressures rebuilt from PRES/PSFC: the history has no PINT")
    return {"valid": valid, "start": start, "notes": notes}


def _native_turn(lat, lon, ca, sa):
    """(cos, sin) of the angle from grid 218's axes to the native NMMB rotated lat-lon grid's axes at each grid-218
    point: earth-relative to rotated-grid-relative (the rotated-pole angle beta, tph0d/tlm0d from the run
    configuration) composed with grid 218's own turn (ca, sa: u_grid = u_e ca - v_e sa)."""
    from namgpu import config
    c = config.load()
    tph0, tlm0 = np.radians(c.tph0d), np.radians(c.tlm0d)
    la, lo = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
    dlm = lo - tlm0
    tlat = np.arcsin(np.clip(-np.sin(tph0) * np.cos(la) * np.cos(dlm) + np.cos(tph0) * np.sin(la), -1.0, 1.0))
    cbeta = (np.cos(tph0) * np.cos(la) + np.sin(tph0) * np.sin(la) * np.cos(dlm)) / np.cos(tlat)
    sbeta = np.sin(tph0) * np.sin(dlm) / np.cos(tlat)
    ca, sa = ca.astype(np.float64), sa.astype(np.float64)
    # u_native = u_e cbeta - v_e sbeta with u_e = u_g ca + v_g sa, v_e = -u_g sa + v_g ca
    return (ca * cbeta + sa * sbeta).astype(np.float32), (sbeta * ca - cbeta * sa).astype(np.float32)


def _buckets(history: Path, s, get, t, ps=None) -> dict:
    """APCP/ACPCP over NAM's buckets ending at this hour: total and convective precipitation since the last
    multiple of N hours (N in BUCKETS_H), from this file's RAINNC/RAINC minus those of the history file at the
    bucket start (zero at the initial time). A bucket whose start file is missing is left out (named in the
    writer's manifest as a missing carrier), never filled."""
    import netCDF4
    from datetime import datetime as _dt
    ps = ps or PSETS.AWPHYS
    if ps.interp == "nearest":
        return _buckets_nest(history, s, t, ps)
    lead_s = int(round(float(s["XTIME"][0]) * 60.0))
    if lead_s <= 0:
        # the initial time: every bucket is empty (NAM writes its accumulations there as 0-0 h zeros)
        z = np.zeros(t["wx"].shape, np.float32)
        out = {f"{v}_S{h}": z for h in BUCKETS_H for v in ("APCP", "ACPCP")}
        if "ACSNOW" in s.variables:
            out["WEASD_S12"] = z
        return out
    start = _dt.strptime(s.START_DATE, "%Y-%m-%d_%H:%M:%S")
    tot_now = get("RAINNC") + get("RAINC")
    out = {}
    for hours in BUCKETS_H:
        period = hours * 3600
        b0 = ((lead_s - 1) // period) * period
        if b0 == 0:
            tot0 = conv0 = np.float32(0.0)
        else:
            f0 = history.with_name(f"wrfout_d01_{start + timedelta(seconds=b0):%Y-%m-%d_%H:%M:%S}")
            if not f0.is_file():
                continue
            with netCDF4.Dataset(f0) as s0:
                nc0 = _regrid(t, np.asarray(s0["RAINNC"][0], dtype=np.float32))
                conv0 = _regrid(t, np.asarray(s0["RAINC"][0], dtype=np.float32))
            tot0 = nc0 + conv0
        out[f"APCP_S{hours}"] = np.maximum(tot_now - tot0, 0.0).astype(np.float32)
        out[f"ACPCP_S{hours}"] = np.maximum(get("RAINC") - conv0, 0.0).astype(np.float32)
    snow = _snow_bucket_12(history, s, t, lead_s, start)
    if snow is not None:
        out["WEASD_S12"] = snow
    return out


def _snow_bucket_12(history: Path, s, t, lead_s: int, start):
    """Snowfall water (kg m-2) since the last 12 h reset (NAM's WEASD accumulation record, awphys row 334). The model
    keeps snowfall only in ACSNOW, a 3 h bucket (kg m-2: the Noah driver adds FFROZP PRCP DT, phys_noah_driver.cu) emptied after the history write at every multiple of 3 h; so the
    window's total is ACSNOW of each history file at a 3 h mark after the window start and up to the current 3 h
    bucket's start, plus this file's ACSNOW. Left out (None) when the history has no ACSNOW or a needed file is
    missing, never filled."""
    import netCDF4
    if "ACSNOW" not in s.variables:
        return None
    b12 = ((lead_s - 1) // (12 * 3600)) * (12 * 3600)
    b3 = ((lead_s - 1) // (SNOW_BUCKET_H * 3600)) * (SNOW_BUCKET_H * 3600)
    tot = np.asarray(s["ACSNOW"][0], dtype=np.float64)
    for mark in range(b12 + SNOW_BUCKET_H * 3600, b3 + 1, SNOW_BUCKET_H * 3600):
        f0 = history.with_name(f"wrfout_d01_{start + timedelta(seconds=mark):%Y-%m-%d_%H:%M:%S}")
        if not f0.is_file():
            return None
        with netCDF4.Dataset(f0) as s0:
            if "ACSNOW" not in s0.variables:
                return None
            tot = tot + np.asarray(s0["ACSNOW"][0], dtype=np.float64)
    return np.maximum(_regrid(t, tot.astype(np.float32)), 0.0).astype(np.float32)


def _buckets_nest(history: Path, s, t, ps) -> dict:
    """A nest set's precipitation buckets (NCEP's nest files: the 3 h bucket, 0-1, 0-2, 0-3, 3-4, ...): total (APCP),
    convective (ACPCP) and grid-scale (NCPCP) precipitation since the last multiple of N hours, differenced on the
    model grid from the history file at the bucket start, then put on the output grid: APCP with NCEP's budget
    interpolation, the others nearest-neighbour. A bucket whose start file is missing is left out, never filled."""
    import netCDF4
    from datetime import datetime as _dt
    lead_s = int(round(float(s["XTIME"][0]) * 60.0))
    nc_now = np.asarray(s["RAINNC"][0], dtype=np.float32)
    c_now = np.asarray(s["RAINC"][0], dtype=np.float32)
    start = _dt.strptime(s.START_DATE, "%Y-%m-%d_%H:%M:%S")
    out = {}
    for hours in ps.buckets_h:
        period = hours * 3600
        if lead_s <= 0:
            nc0, c0 = nc_now, c_now
        else:
            b0 = ((lead_s - 1) // period) * period
            if b0 == 0:
                nc0, c0 = np.zeros_like(nc_now), np.zeros_like(c_now)
            else:
                f0 = history.with_name(f"wrfout_d01_{start + timedelta(seconds=b0):%Y-%m-%d_%H:%M:%S}")
                if not f0.is_file():
                    continue
                with netCDF4.Dataset(f0) as s0:
                    nc0 = np.asarray(s0["RAINNC"][0], dtype=np.float32)
                    c0 = np.asarray(s0["RAINC"][0], dtype=np.float32)
        dnc, dc = np.maximum(nc_now - nc0, 0.0), np.maximum(c_now - c0, 0.0)
        out[f"APCP_S{hours}"] = np.maximum(_budget(t, dnc + dc), 0.0).astype(np.float32)
        out[f"ACPCP_S{hours}"] = _nearest(t, dc).astype(np.float32)
        out[f"NCPCP_S{hours}"] = _nearest(t, dnc).astype(np.float32)
    return out


def _hybrid(history: Path, pres=None, psfc=None):
    """(C3F, C4F, P_TOP) on the interfaces, bottom-up, from the history's PRES and PSFC on its own grid.

    PRES(k) = a_k + b_k PSFC holds exactly at every column of an NMMB history (hybrid sigma-pressure, hydrostatic
    mid layers); the fit uses every 7th column. Interfaces: bottom = PSFC (a 0, b 1), then int(k+1) = 2 mid(k) - int(k).
    """
    if pres is not None and psfc is not None:
        ps = np.asarray(psfc, dtype=np.float64).ravel()[::7]
        pm = np.asarray(pres, dtype=np.float64).reshape(pres.shape[0], -1)[:, ::7]
    else:
        import netCDF4
        with netCDF4.Dataset(history) as s:
            ps = np.asarray(s["PSFC"][0], dtype=np.float64).ravel()[::7]
            pm = np.asarray(s["PRES"][0], dtype=np.float64).reshape(s.dimensions["bottom_top"].size, -1)[:, ::7]
    X = np.stack([np.ones_like(ps), ps], axis=1)
    coef, *_ = np.linalg.lstsq(X, pm.T, rcond=None)           # [2, nz]: a, b per mid level
    a_m, b_m = coef[0], coef[1]
    nz = a_m.size
    a_i, b_i = np.zeros(nz + 1), np.zeros(nz + 1)
    a_i[0], b_i[0] = 0.0, 1.0
    for k in range(nz):
        a_i[k + 1] = 2.0 * a_m[k] - a_i[k]
        b_i[k + 1] = 2.0 * b_m[k] - b_i[k]
    b_i[np.abs(b_i) < 1e-5] = 0.0                            # pure-pressure levels (fit noise ~1e-6)
    ptop = float(a_i[-1] + b_i[-1] * ps.mean()) if abs(b_i[-1]) > 0 else float(a_i[-1])
    c3f = b_i
    c4f = a_i + b_i * ptop - ptop
    return c3f.astype(np.float32), c4f.astype(np.float32), ptop



# ---------------------------------------------------------------------------------------------- GRIB2 index

NAMES = {
    "0.0.0": "TMP", "0.0.2": "POT", "0.0.4": "TMAX", "0.0.5": "TMIN", "0.0.6": "DPT", "0.0.10": "LHTFL",
    "0.0.11": "SHTFL", "0.1.0": "SPFH", "0.1.1": "RH", "0.1.3": "PWAT", "0.1.8": "APCP", "0.1.10": "ACPCP",
    "0.1.11": "SNOD", "0.1.13": "WEASD", "0.1.42": "SNOWC", "0.2.1": "WIND", "0.2.2": "UGRD", "0.2.3": "VGRD",
    "0.2.8": "VVEL", "0.2.9": "DZDT", "0.2.10": "ABSV", "0.2.22": "GUST", "0.2.27": "USTM", "0.2.28": "VSTM",
    "0.2.30": "FRICV", "0.3.0": "PRES", "0.3.1": "PRMSL", "0.3.5": "HGT", "0.3.18": "HPBL", "0.3.192": "MSLET",
    "0.3.196": "HPBL", "0.3.198": "MSLMA", "0.4.7": "DSWRF", "0.5.3": "DLWRF", "0.6.1": "TCDC", "0.6.3": "LCDC",
    "0.6.4": "MCDC", "0.6.5": "HCDC", "0.7.6": "CAPE", "0.7.7": "CIN", "0.7.8": "HLCY", "0.7.198": "LAI",
    "0.7.199": "MXUPHL", "0.7.200": "MNUPHL", "0.16.195": "REFD", "0.16.196": "REFC", "0.19.0": "VIS",
    "2.0.0": "LAND", "2.0.1": "SFCR", "2.0.2": "TSOIL", "2.0.4": "VEG", "2.0.10": "GFLUX", "2.0.198": "VGTYP",
    "10.2.0": "ICEC",
    # NCEP local and WMO soil/land/radiation names (wgrib2's table), the soil, land, flux and cloud records
    "2.0.192": "SOILW", "2.3.192": "SOILL", "2.0.194": "MSTAV", "2.0.3": "SOILM", "2.0.196": "CNWAT",
    "0.1.201": "SNOWC", "2.3.196": "SMDRY", "2.3.197": "POROS", "0.2.197": "FRICV", "0.2.196": "CD",
    "2.3.0": "SOTYP", "2.3.193": "RLYRS", "2.0.201": "WILT", "2.3.195": "SMREF", "0.19.1": "ALBDO",
    "0.4.192": "DSWRF", "0.5.192": "DLWRF", "0.4.193": "USWRF", "0.5.193": "ULWRF",
}
LEVELS = {1: "surface", 2: "cloud base", 3: "cloud top", 4: "0C isotherm", 5: "level of adiabatic condensation "
          "from sfc", 6: "max wind", 7: "tropopause", 8: "top of atmosphere", 101: "mean sea level",
          200: "entire atmosphere", 204: "highest tropospheric freezing level", 214: "low cloud layer",
          215: "cloud ceiling", 220: "planetary boundary layer", 224: "middle cloud layer", 234: "high cloud layer",
          245: "lowest level of the wet bulb zero"}


def _scaled(scale, value):
    if scale == 255 or value in (0xFFFFFFFF,):
        return None
    sc = -(scale & 0x7F) if scale & 0x80 else scale
    v = -(value & 0x7FFFFFFF) if value & 0x80000000 else value
    return v / (10.0 ** sc)


def _fmt(x):
    return f"{x:g}"


def _level(t1, v1, t2, v2):
    if t1 == 100:
        return f"{_fmt(v1 / 100)} mb" if t2 in (255, None) else f"{_fmt(v1 / 100)}-{_fmt(v2 / 100)} mb"
    if t1 == 103:
        return f"{_fmt(v1)} m above ground" if t2 in (255, None) else f"{_fmt(v1)}-{_fmt(v2)} m above ground"
    if t1 == 106:
        return f"{_fmt(v1)}-{_fmt(v2)} m below ground" if t2 == 106 else f"{_fmt(v1)} m below ground"
    if t1 == 108:
        return f"{_fmt(v1 / 100)}-{_fmt(v2 / 100)} mb above ground"
    if t1 == 20:
        return f"{_fmt(v1)} K level"
    if t1 in LEVELS and t2 in (255, None):
        return LEVELS[t1]
    return f"level_type={t1} value={v1} second={t2}:{v2}"


def scan(path: Path) -> list:
    """Every GRIB2 message of a file: byte offset, discipline/category/parameter, levels, time (no data decode)."""
    out = []
    data = Path(path).read_bytes()
    pos = 0
    while True:
        pos = data.find(b"GRIB", pos)
        if pos < 0:
            break
        disc, ed = data[pos + 6], data[pos + 7]
        total = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        if ed != 2:
            pos += 4
            continue
        rec = {"offset": pos, "bytes": total, "discipline": disc, "msg": len({r["offset"] for r in out}) + 1}
        nsub = 0
        q = pos + 16
        while q < pos + total - 4:
            ln, num = struct.unpack(">IB", data[q:q + 5])
            sec = data[q:q + ln]
            if num == 1:
                y, mo, dd, hh, mi, ss = struct.unpack(">HBBBBB", sec[12:19])
                rec["ref"] = datetime(y, mo, dd, hh, mi, ss)
                rec["center"] = struct.unpack(">H", sec[5:7])[0]
            elif num == 3:
                rec["gdt"] = struct.unpack(">H", sec[12:14])[0]
                rec["gds"] = bytes(sec[5:]).hex()
                rec["npts"] = struct.unpack(">I", sec[6:10])[0]
            elif num == 4:
                pdt = struct.unpack(">H", sec[7:9])[0]
                rec.update(pdt=pdt, cat=sec[9], param=sec[10], gen_process=sec[13])
                ftime = struct.unpack(">I", sec[18:22])[0]
                t1, s1, v1 = sec[22], sec[23], struct.unpack(">I", sec[24:28])[0]
                t2, s2, v2 = sec[28], sec[29], struct.unpack(">I", sec[30:34])[0]
                v1s, v2s = _scaled(s1, v1), _scaled(s2, v2)
                rec.update(unit=sec[17], ftime=ftime, t1=t1, v1=None if t1 == 255 else (v1s or 0.0), t2=t2,
                           v2=None if t2 == 255 else (v2s or 0.0), s1=s1, s2=s2)
                if pdt == 8 and len(sec) >= 54:
                    rec.update(stat=sec[46], range_unit=sec[48], range_len=struct.unpack(">I", sec[49:53])[0])
            elif num == 5:
                def _sm(b):
                    v = struct.unpack(">H", b)[0]
                    return -(v & 0x7FFF) if v & 0x8000 else v
                rec.update(drt=struct.unpack(">H", sec[9:11])[0], E=_sm(sec[15:17]), D=_sm(sec[17:19]), nbits=sec[19])
            elif num == 7:
                # one field per section 7; NCEP packs U and V as two fields of one message (wgrib2 N.1, N.2)
                sub = dict(rec, key=f"{rec['discipline']}.{rec.get('cat')}.{rec.get('param')}")
                out.append(sub)
                nsub += 1
                sub["sub"] = nsub
            q += ln
        if nsub == 1:
            out[-1]["sub"] = None
        pos += total
    return out


def _hours(unit, n):
    return {0: n / 60.0, 1: n, 2: n * 24, 10: n * 3, 11: n * 6, 12: n * 12, 13: n / 3600.0}.get(unit, n)


_NOAA_TEXT: dict = {}


def _inv_key(r):
    return (r["key"], r.get("t1"), float(r.get("v1") or 0.0), r.get("t2") if r.get("t2") != 255 else None,
            None if r.get("t2") in (None, 255) else float(r.get("v2") or 0.0),
            r.get("stat") if r.get("pdt") == 8 else None)


def _noaa_text(set_name: str = "awphys") -> dict:
    """(inventory identity, n-th occurrence) -> NOAA's own .idx name and level text for that record of a set."""
    if set_name not in _NOAA_TEXT:
        from namgpu.products import spec  # noqa: PLC0415
        ps = PSETS.get(set_name)
        out = {}
        try:
            rows = spec.attach_noaa(spec.read(PSETS.table_path(ps)), noaa_path=PSETS.noaa_path(ps))
        except SystemExit:
            rows = []
        seen = {}
        for t in rows:
            if "noaa_name" not in t:
                continue
            k = spec.inventory_key(t)
            n = seen.get(k, 0)
            seen[k] = n + 1
            out[(k, n)] = (t["noaa_name"], t["noaa_level"])
        _NOAA_TEXT[set_name] = out
    return _NOAA_TEXT[set_name]


def idx_lines(recs, set_name: str = "awphys") -> list:
    """wgrib2 -s lines. A record of the set's table takes NOAA's own name and level text (<set>-noaa.tsv), so
    our .idx and NOAA's read the same to byte-range tools that match on it."""
    lines = []
    text = _noaa_text(set_name)
    seen = {}
    for r in recs:
        i = f"{r['msg']}.{r['sub']}" if r.get("sub") else str(r["msg"])
        k = _inv_key(r)
        n = seen.get(k, 0)
        seen[k] = n + 1
        name, lev = text.get((k, n), (None, None))
        if name is None:
            name = NAMES.get(r["key"], f"var{r['key'].replace('.', '_')}")
            lev = _level(r.get("t1"), r.get("v1"), r.get("t2"), r.get("v2"))
        fh = _hours(r.get("unit", 1), r.get("ftime", 0))
        if r.get("pdt") == 8:
            end = fh + _hours(r.get("range_unit", 1), r.get("range_len", 0))
            kind = {0: "ave", 1: "acc", 2: "max", 3: "min"}.get(r.get("stat"), "stat")
            if r.get("unit") == 2 and r.get("range_unit") == 2:
                # a window written in days (NAM's nest hiresf00: the 0-0 day bucket), as wgrib2 prints it
                tm = f"{r.get('ftime', 0)}-{r.get('ftime', 0) + r.get('range_len', 0)} day {kind} fcst"
            else:
                tm = f"{_fmt(fh)}-{_fmt(end)} hour {kind} fcst"
        else:
            tm = "anl" if fh == 0 else f"{_fmt(fh)} hour fcst"
        lines.append(f"{i}:{r['offset']}:d={r['ref']:%Y%m%d%H}:{name}:{lev}:{tm}:")
    return lines


# ---------------------------------------------------------------------------------------------- one hour

def _exporter() -> Path:
    """The writer (namgpu.exporter: $WOOF_NAM_RW_RRFS_UPP, $GPUWM_BRIDGES, else the pinned copy woof-nam stages)."""
    from namgpu import exporter
    return exporter.ensure(fetch_if_missing=False)


class _Hour:
    """One history hour in flight: its hand-off frame is written, its writer process is running."""

    def __init__(self, history: Path, out: Path, work: Path, when: dict, t0: float, t_frame: float, proc, cmd=None,
                 set_name: str = "awphys", file_name: str | None = None, gate=None, stamps: dict | None = None):
        self.history, self.out, self.work, self.when = history, out, work, when
        self.t0, self.t_frame, self.proc, self.cmd = t0, t_frame, proc, cmd
        self.set_name, self.file_name = set_name, file_name or FILE_NAME
        self.gate = gate or cardgate.Hold()       # the writer's place on the card (cardgate), until it has exited
        self.stamps = stamps if stamps is not None else _stamps(history)


def _stamps(history: Path) -> dict:
    """Where an hour's time goes, in wall-clock seconds after its history file was written (its mtime): the per-hour
    stamps of products.log and the hour's index row. Measured first, so the lag of each GRIB2 file behind its history
    hour can be split into the follower's queue, its frame, its card waits and its writer."""
    try:
        seen = Path(history).stat().st_mtime
    except OSError:
        seen = time.time()
    return {"history": seen, "taken": round(time.time() - seen, 2)}


def _stamp(h_or_stamps, key: str) -> None:
    s = h_or_stamps.stamps if hasattr(h_or_stamps, "stamps") else h_or_stamps
    s[key] = round(time.time() - s["history"], 2)


def stamps_line(name: str, s: dict) -> str:
    """One products.log line: where the time of this hour's file went, seconds after its history file."""
    parts = [f"taken +{s.get('taken', 0):.1f}"]
    for k, label in (("frame_end", "frame done"), ("writer_start", "writer started"), ("writer_end", "writer done"),
                     ("placed", "in place")):
        if k in s:
            parts.append(f"{label} +{s[k]:.1f}")
    if s.get("reruns"):
        parts.append(f"{s['reruns']} reruns, {s.get('rerun_wait_s', 0):.1f} s waiting for card memory")
    return f"{name}: stamps (s after its history file): " + ", ".join(parts)


# A writer that dies on a full shared card (CUDA out of memory in its post or extras upload) is rerun on the GPU,
# never dropped and never moved to the CPU (the GPU stage has no CPU fallback): after a wait until the card has room.
# Every rerun is logged and named in the hour's index row (`writer_retries`); a card that stays full fails the hour.
# "bytes free, the frame needs about" is the writer's own refusal when the card has no room for its post
# (rw-post group_e cuda.rs); it is the same full card, so it is rerun the same way (arctic-front 0.2.1 candidate run
# on a 32 GB card: two hours failed on it at once while the forecast held the card).
OOM_MARKERS = ("out of memory", "OUT_OF_MEMORY", "cudaErrorMemoryAllocation", "bytes free, the frame needs about")
OOM_RETRIES = int(os.environ.get("NAMGPU_PRODUCTS_OOM_RETRIES", "20"))
OOM_WAIT_S = float(os.environ.get("NAMGPU_PRODUCTS_OOM_WAIT_S", "30"))
# While this run's forecast is on the card, a refused stage waits for the forecast's next release of memory (after a
# radiation call or a history hour: every few seconds on the 12 km grid) and reruns then; that refusal is not one of
# the hour's OOM_RETRIES (namgpu.cardgate says what broke with the fixed sleep). A forecast that gives nothing back
# for this long counts as a try, as a card held by someone else does.
RELEASE_WAIT_S = float(os.environ.get("NAMGPU_PRODUCTS_RELEASE_WAIT_S", "120"))

# The follower's heartbeat, <run>/products.state.json: {"pid", "state", "since", "beat"}. Written when the follower
# starts, when it takes an hour, and about every 5 s while one of its stages waits for card memory. The run that waits
# for its follower (run_forecast.products_wait) counts a new beat as progress. Breakage it prevents: that wait counted
# only new files in products/ and new lines in products.log, so a follower alive and waiting for a full card for
# --products-stall-s (900 s) was stopped and the run exited 1. A beat is written by the waits themselves, never by a
# free-running thread: a follower that hangs stops beating and is still stopped.
STATE_FILE = "products.state.json"
_BEAT = {"path": None, "state": "starting", "since": 0.0}
_BEAT_LOCK = __import__("threading").Lock()


def beat(state: str | None = None) -> None:
    """Rewrite the follower's heartbeat (state: what it does now; None keeps the last). Never raises."""
    p = _BEAT.get("path")
    if p is None:
        return
    with _BEAT_LOCK:
        now = time.time()
        if state is not None and state != _BEAT["state"]:
            _BEAT["state"], _BEAT["since"] = state, now
        doc = {"pid": os.getpid(), "state": _BEAT["state"], "since": round(_BEAT["since"], 3), "beat": round(now, 3)}
        try:
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(doc) + "\n")
            os.replace(tmp, p)
        except OSError:
            pass


def _sleep_beating(seconds: float) -> None:
    end = time.monotonic() + max(0.0, float(seconds))
    while True:
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(5.0, left))
        beat()


def _card_wait(run_dir, what: str, counted: int) -> tuple[int, str, float]:
    """After `what` was refused card memory: wait until a rerun may fit. Returns (1 when this refusal counts as one of
    the hour's OOM_RETRIES else 0, the log words, seconds waited). While this run's forecast is on the card the wait
    ends at its next release of memory and does not count (cardgate); otherwise (the forecast has ended or is gone, a
    run with no gate, or a forecast that gave nothing back for RELEASE_WAIT_S) it is the fixed OOM_WAIT_S and counts."""
    t0 = time.monotonic()
    st = cardgate.forecast_state(run_dir) if run_dir is not None else None
    if st is not None and st["state"] == "running":
        beat(f"waiting for card memory ({what}) until the forecast gives memory back")
        how = cardgate.wait_release(run_dir, st["releases"], RELEASE_WAIT_S, beat=beat)
        dt = time.monotonic() - t0
        if how in ("released", "ended"):
            beat("working")
            return 0, f"waited {dt:.1f} s for the forecast to give card memory back ({how}); not counted", dt
        if how == "timeout":
            beat("working")
            return 1, (f"waited {dt:.0f} s and the forecast gave no memory back; try {counted + 1} of "
                       f"{OOM_RETRIES}"), dt
    beat(f"waiting for card memory ({what}), {OOM_WAIT_S:.0f} s")
    _sleep_beating(OOM_WAIT_S)
    beat("working")
    dt = time.monotonic() - t0
    return 1, f"waited {dt:.0f} s; try {counted + 1} of {OOM_RETRIES}", dt


def _is_oom(text: str) -> bool:
    return any(m in (text or "") for m in OOM_MARKERS)


def _is_oom_error(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or _is_oom(str(e)) or _is_oom(str(e).lower())


def _free_pools() -> None:
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 - no CuPy or no device
        pass


def _start_hour_fits(start, name: str, log=None, run_dir=None):
    """start() (an hour's hand-off frame and writer start), rerun after a wait when the frame runs out of GPU
    memory on a full shared card, as the writer is (_card_wait: the forecast's next release while it runs, else
    OOM_WAIT_S and one of OOM_RETRIES); any other error, or a card that stays full, fails the hour. Breakage it
    prevents: a frame that met a full card failed its hour at once, so a 12 km run with --products on a 32 GB card
    lost hours while its own forecast held the card (arctic-front 0.2.1 candidate run, +18 h and +30 h)."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    counted = reruns = 0
    while True:
        try:
            return start()
        except Exception as e:  # noqa: BLE001 - CuPy's OutOfMemoryError or a CUDA allocation error
            _free_pools()
            if counted >= OOM_RETRIES or not _is_oom_error(e):
                raise
            c, why, _ = _card_wait(run_dir, f"frame of {name}", counted)
            counted += c
            reruns += 1
            log(f"woof-nam products: {name}: frame out of GPU memory, rerun {reruns} ({why})")


def _rerun_on_oom(h: "_Hour", stdout: str, stderr: str):
    """(returncode, stdout, stderr, reruns) after rerunning an out-of-memory writer until it fits."""
    reruns = []
    rc = h.proc.returncode
    cmd = list(h.cmd or [])
    counted = 0
    while counted < OOM_RETRIES:
        if not rc or not cmd or not _is_oom((stderr or "") + (stdout or "")):
            break
        h.gate.release()                       # the refused writer has exited: no place held while it waits
        c, why, dt = _card_wait(h.history.parent, f"writer of {h.history.name}", counted)
        counted += c
        st = getattr(h, "stamps", None)
        if st is not None:
            st["reruns"] = st.get("reruns", 0) + 1
            st["reruns_counted"] = counted
            st["rerun_wait_s"] = round(st.get("rerun_wait_s", 0.0) + dt, 2)
        print(f"woof-nam products: {h.history.name}: writer out of GPU memory, rerun {len(reruns) + 1} ({why})",
              file=sys.stderr, flush=True)
        reruns.append(why)
        shutil.rmtree(h.work / "grib2", ignore_errors=True)
        (h.work / "grib2").mkdir()
        h.gate = cardgate.hold(h.history.parent)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    return rc, stdout, stderr, reruns


def _start_hour(history: Path, out: Path, profile: str, device: str, bits: int, pool, ps=None, gkey=None,
                cctx=None, file_name: str | None = None) -> _Hour:
    """The hand-off frame (main thread; the field reads and regrids run in `pool` on every core), then the writer
    process, started and not waited for: the next hour's frame is built while it encodes."""
    ps = ps or PSETS.AWPHYS
    file_name = file_name or FILE_NAME
    t0 = time.perf_counter()
    stamps = _stamps(history)
    work = Path(tempfile.mkdtemp(prefix=".products-", dir=out))
    gate = cardgate.Hold()
    try:
        frame = work / history.name
        when = wrf_frame(history, frame, pool=pool, ps=ps, gkey=gkey, cctx=cctx)
        t_frame = time.perf_counter() - t0
        _stamp(stamps, "frame_end")
        exp = work / "grib2"
        exp.mkdir()
        cmd = [str(_exporter()), "--out", str(exp), "--bits", str(int(bits)), "--post-device",
               {"auto": "auto", "gpu": "gpu", "cpu": "cpu"}[device], "--profile-json", str(profile),
               "--file-name", file_name, "--frames", str(frame)]
        if cctx and cctx.get("dt"):
            cmd[1:1] = ["--dt", str(float(cctx["dtq2"] or cctx["dt"]))]
        # the writer's GPU post: behind the run's card gate (it waits here while the forecast claims the card) from
        # its start until it has exited (_finish_hour)
        gate = cardgate.hold(history.parent)
        _stamp(stamps, "writer_start")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except BaseException:
        gate.release()
        shutil.rmtree(work, ignore_errors=True)
        raise
    return _Hour(history, out, work, when, t0, t_frame, proc, cmd, ps.name, file_name, gate, stamps)


def _writer_manifest(exp: Path) -> Path:
    """The run manifest rw_rrfs_upp wrote in its output folder (exactly one *-manifest.json)."""
    found = sorted(exp.glob("*-manifest.json"))
    if len(found) != 1:
        raise RuntimeError(f"{RW} left {len(found)} manifests in {exp} (expected one *-manifest.json)")
    return found[0]


def _finish_hour(h: _Hour, keep_work: bool = False) -> dict:
    """Wait for the hour's writer, move its file into place, write the .idx; the hour's index row."""
    keep_work = keep_work or os.environ.get("NAMGPU_PRODUCTS_KEEP", "") == "1"
    try:
        try:
            stdout, stderr = h.proc.communicate()
            rc, stdout, stderr, reruns = _rerun_on_oom(h, stdout, stderr)
        finally:
            h.gate.release()                   # the writer has exited: its memory is back with the driver
        _stamp(h, "writer_end")
        if rc:
            raise RuntimeError(f"{RW} exit {rc}: {(stderr or stdout).strip()[-800:]}")
        exp, out, when = h.work / "grib2", h.out, h.when
        fhr = int(round((when["valid"] - when["start"]).total_seconds() / 3600.0))
        name = h.file_name.replace("{HH}", f"{when['start']:%H}").replace("{FF}", f"{fhr:02d}")
        if not (exp / name).is_file():
            raise RuntimeError(f"{RW} wrote no {name} for {h.history.name}: {stdout[-400:]}")
        os.replace(exp / name, out / name)
        # the writer's run manifest: the one *-manifest.json it writes beside the GRIB2 file
        manifest = json.loads(_writer_manifest(exp).read_text())
        frame_rec = manifest["frames"][0]
        omitted = [{"shortname": f["shortname"], "name": f["pname"], "reason": f["reason"]}
                   for f in frame_rec.get("fields", []) if f.get("written", 0) < f.get("messages", 1)]
        recs = scan(out / name)
        # the .idx goes into place whole and last: a follower (the WOOF publisher) takes a file once its .idx exists
        tmp_idx = out / (name + ".idx.part")
        tmp_idx.write_text("\n".join(idx_lines(recs, h.set_name)) + "\n")
        os.replace(tmp_idx, out / (name + ".idx"))
        _stamp(h, "placed")
        return {"file": name, "history": h.history.name, "valid": f"{when['valid']:%Y-%m-%dT%H:%M:%SZ}",
                "fhr": fhr, "bytes": (out / name).stat().st_size, "records": len(recs),
                "frame_s": round(h.t_frame, 2), "writer_s": round(float(frame_rec.get("seconds") or 0.0), 2),
                "writer_timings": frame_rec.get("timings"), "not_written": omitted,
                "notes": when.get("notes", []), "writer_retries": reruns,
                "stamps": {k: v for k, v in h.stamps.items() if k != "history"},
                "seconds": round(time.perf_counter() - h.t0, 2)}
    finally:
        if not keep_work:
            shutil.rmtree(h.work, ignore_errors=True)


def one_hour(history: str, out: str, profile: str, device: str = "auto", bits: int = DEFAULT_BITS,
             keep_work: bool = False, pool=None) -> dict:
    """One history hour -> <out>/nam.tHHz.awphysFF.tm00.grib2 (+ .idx): the grid-218 hand-off frame, then the
    field-set writer over the record table's profile (every record in table order, one file)."""
    return _finish_hour(_start_hour(Path(history), Path(out), profile, device, bits, pool), keep_work)


def _hour_filter(spec):
    if not spec or spec == "all":
        return None
    keep = set()
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def _xtime_hour(path: Path) -> int:
    import netCDF4
    with netCDF4.Dataset(path) as s:
        return int(round(float(s["XTIME"][0]) / 60.0))


def _run_ctx(run_dir: Path) -> dict:
    """The run's physics step for the rate records (PRATE = PREC / (DT NPHS)): from its configfile."""
    for cf in sorted(Path(run_dir).glob("nam_configfile*")) + sorted(Path(run_dir).parent.glob("nam_configfile*")):
        try:
            kv = {}
            for line in cf.read_text().splitlines():
                k, _, v = line.partition(":")
                v = v.split("#")[0].strip()
                if k.strip() and v:
                    kv[k.strip()] = v
            dt = float(kv["dt_int"]) + (float(kv.get("dt_num", 0)) / float(kv.get("dt_den", 1) or 1))
            nphs = int(float(kv.get("nphs", 1)))
            return {"dt": dt, "nphs": nphs, "dtq2": dt * nphs, "configfile": str(cf)}
        except (KeyError, ValueError, OSError):
            continue
    return {}


def _field_context():
    """The field pool's start method: forkserver (spawn where there is none), on every Python. A forked worker
    inherits the SIGTERM/SIGHUP handler (_stop_on_signal), so the stop signal it is sent becomes an exception it
    catches and outlives, and it holds both ends of the pool's pipes, so it never reads an end of file. The forkserver's
    socket goes in a short folder when TMPDIR is long (namgpu.cores.pool_context)."""
    from namgpu import cores as _cores  # noqa: PLC0415
    return _cores.pool_context()


def field_workers() -> int:
    """Processes of the frame's field pool (reads and regrids): half the cores this process may use
    (namgpu.cores.usable: affinity set and cgroup quota), 2 to 16. It counted the host's cores, so a follower pinned
    to 12 of 24 cores beside its forecast ran 12 field processes on them."""
    from namgpu import cores
    return max(2, min(16, cores.usable() // 2))


def run(run_dir: Path, out: Path | None = None, hours=None, workers: int = 2, threads: int = 8,
        device: str = "gpu", bits: int = DEFAULT_BITS, table: Path | None = None, follow: bool = False,
        poll_s: float = 1.0, idle_timeout_s: float = 1800.0, until: Path | None = None,
        set_name: str | None = None) -> dict:
    """Every history hour of a run (or `hours`) -> one awphys file each, plus <out>/index.json.

    follow: keep watching the run folder while the forecast runs and write each hour as soon as its history file
    is complete (the history writer renames each file into place whole; its size held across one poll, or the run's
    forecast-receipt.json is written); stop once the receipt is there and every hour is done, or after
    `idle_timeout_s` with no new history file and no receipt (the run died: exit with an error). <out>/DONE is
    written at the end of a followed run.
    """
    from namgpu.products import spec
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "products"
    out.mkdir(parents=True, exist_ok=True)
    if follow:                                 # the heartbeat the run's wait reads (STATE_FILE)
        _BEAT["path"] = run_dir / STATE_FILE
        beat("waiting for history")
    if set_name in (None, "auto"):
        set_name = detect_set(run_dir) or "awphys"
    ps = PSETS.get(set_name)
    table = table or PSETS.table_path(ps)
    table_rows = spec.read(table)
    nest = ps.name != "awphys"
    profile, left = spec.write_profile(out / f".{ps.name}-profile.json", table_rows,
                                       noaa_path=PSETS.noaa_path(ps), name=f"nam-{ps.name}",
                                       mask_carrier=MASK if (nest and ps.mask) else None, pair_uv=nest)
    file_name = FILE_NAME if not nest else ps.file_name.replace("woof-nam", FILE_NAME.split(".t{HH}z")[0], 1)
    # The nest run's physics step (PRATE = PREC / DTQ2, and the writer's --dt) comes from the configfile the nest door
    # writes. A follower can start before its door has written it (the suite starts every follower at once and the
    # doors as card memory frees up), so it is read when the first history hour is taken, and again for each later
    # hour until it is found; an hour taken without it says so.
    ctx_state: dict = {}

    def cctx_for(f: Path):
        if not nest:
            return None
        if not ctx_state.get("ctx"):
            ctx_state["ctx"] = _run_ctx(run_dir)
            if not ctx_state["ctx"]:
                print(f"woof-nam products: {ps.name}: no nam_configfile* in {run_dir} or its parent when "
                      f"{f.name} was taken: this hour has no PRATE and the writer has no --dt", flush=True)
        return ctx_state["ctx"]
    grid_state = {}

    def gkey_for(f: Path):
        """The set's output grid for this run (resolved once, from the first history: firewx and the generic
        nest box follow the nest's own corners)."""
        if not nest:
            return None
        if "key" not in grid_state:
            sg = SourceGrid(f)
            og = PSETS.resolve(ps, sg.lat, sg.lon, sg.dx_m)
            i0, j0, win = nest_window(sg, og)
            grid_state["key"] = (ps.name, win, str(f), tuple(pad_attrs(og, i0, j0).items()))
            grid_state["grid"] = og
            grid_state["window"] = {"i0": i0, "j0": j0, "nx": win.nx, "ny": win.ny}
            print(f"woof-nam products: {ps.name}: output grid {og.nx} x {og.ny}, the nest covers the window "
                  f"{win.nx} x {win.ny} at ({i0}, {j0}); the rest is missing (bitmap)", flush=True)
        return grid_state["key"]
    if os.environ.get("RAYON_NUM_THREADS") is None and threads:
        os.environ["RAYON_NUM_THREADS"] = str(int(threads))
    want = _hour_filter(hours)
    t0 = time.perf_counter()
    rows_out, errors = [], []
    handled: set[str] = set()
    sizes: dict[str, int] = {}

    def done(item):
        h, fut = item
        try:
            r = fut.result()
            rows_out.append(r)
            print(f"{r['file']}: {r['records']} records, {r['bytes'] / 1e6:.1f} MB, frame {r['frame_s']} s, "
                  f"writer {r['writer_s']} s", flush=True)
            if r.get("stamps"):
                print(stamps_line(r["file"], r["stamps"]), flush=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{h.history.name}: {type(e).__name__}: {e}")
            print("FAILED", errors[-1], flush=True)

    def ready_files() -> tuple[list[Path], bool]:
        receipt = (run_dir / "forecast-receipt.json").is_file() or (until is not None and Path(until).exists())
        files = sorted(run_dir.glob("wrfout_d01_*"))
        ready = []
        for f in files:
            if f.name in handled:
                continue
            size = f.stat().st_size
            stable = sizes.get(f.name) == size and size > 0
            sizes[f.name] = size
            # the history writer (namgpu.io.history.write_payload, and the multi-card merge) renames each hour into
            # place whole: an hour is taken once its file is there and its size held across one poll, not when the
            # next hour's file appears (that held every hour back one model hour)
            if not follow or receipt or stable:
                ready.append(f)
        return ready, receipt

    # Hours go one after the other in this thread: the hand-off frame of hour N+1 (its field reads and regrids
    # in the process pool, every core) is built while hour N's writer process encodes (rayon, every core; the
    # GPU for the post). netCDF is never touched from two threads, and the pool never forks from one.
    # Up to `workers` writer processes are in flight (a writer on a whole NOAA nest grid is many times one on grid 218, so one
    # at a time would fall behind the forecast); the frames are still built one after the other in this thread.
    # A 1 s poll (a glob and a stat): an hour is taken 1-2 s after its whole file appears (the 5 s poll and its
    # one-poll size hold cost 5-10 s of every hour on the 3 km CONUS nest).
    # Each hour in flight is (hour, its finisher): a thread that waits for the hour's writer and puts the file and
    # its .idx in place as soon as the writer exits. Breakage it prevents: the main thread finished hours only when it
    # had nothing new to build, so a written hour waited for the next hour's frame (40-90 s on the 3 km CONUS nest)
    # before it was public; in a replay at the model's pace the last file landed 138 s after its history hour.
    pending: list = []
    in_flight = max(1, int(workers or 1))
    last_new = time.monotonic()

    def harvest():
        for item in [p for p in pending if p[1].done()]:
            pending.remove(item)
            done(item)

    with ProcessPoolExecutor(max_workers=field_workers(), mp_context=_field_context(),
                             initializer=_die_with_parent) as fields, \
            ThreadPoolExecutor(max_workers=in_flight + 1, thread_name_prefix="products-finish") as finisher:
        while True:
            harvest()
            ready, receipt = ready_files()
            for f in ready:
                handled.add(f.name)
                last_new = time.monotonic()
                try:
                    if want is not None and _xtime_hour(f) not in want:
                        continue
                    beat(f"working ({f.name})")
                    h = _start_hour_fits(lambda f=f: _start_hour(f, out, str(profile), device, bits, fields, ps=ps,
                                                                 gkey=gkey_for(f), cctx=cctx_for(f),
                                                                 file_name=file_name), f.name, run_dir=run_dir)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{f.name}: {type(e).__name__}: {e}")
                    print("FAILED", errors[-1], flush=True)
                    continue
                if len(pending) >= in_flight:
                    done(pending.pop(0))           # writers in flight at most `workers`: wait for the oldest
                pending.append((h, finisher.submit(_finish_hour, h)))
                harvest()
            if not follow:
                break
            if receipt and not ready_files()[0]:
                break
            if time.monotonic() - last_new > idle_timeout_s and not receipt:
                errors.append(f"no new history file in {idle_timeout_s:.0f} s and no forecast-receipt.json: "
                              f"the run stopped")
                print("FAILED", errors[-1], flush=True)
                break
            time.sleep(poll_s)
        while pending:
            done(pending.pop(0))
    if not rows_out and not errors and not handled:
        raise SystemExit(f"{run_dir}: no history files (wrfout_d01_*)")
    rows = rows_out
    idx_path = out / "index.json"
    old = json.loads(idx_path.read_text()).get("files", []) if idx_path.exists() else []
    merged = {r["file"]: r for r in old}
    merged.update({r["file"]: r for r in rows})
    doc = {"set": ps.name, "grid": ps.grid_name,
           **({"grid_spec": grid_state["grid"].spec(), "window": grid_state["window"]}
              if grid_state.get("grid") is not None else {}),
           "writer": RW, "table": str(table), "table_records": len(table_rows),
           "table_records_with_source": len(table_rows) - len(left),
           "left_out": [{"rec": r["rec"], "name": r["name"], "dcp": r["dcp"], "level": r["level"],
                         "level2": r["level2"], "note": r["note"]} for r in left],
           "bits": bits, "files": sorted(merged.values(), key=lambda r: r["fhr"]),
           "bytes": sum(r["bytes"] for r in merged.values()), "errors": errors}
    idx_path.write_text(json.dumps(doc, indent=1))
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB in {time.perf_counter() - t0:.1f} s "
          f"-> {out} (index {idx_path})", flush=True)
    if errors:
        print("errors:\n  " + "\n  ".join(errors), file=sys.stderr)
    if follow:
        (out / "DONE").write_text(json.dumps({"files": len(merged), "errors": len(errors)}) + "\n")
    return {"files": rows, "errors": errors}


def _undated(inventory_line: str) -> str:
    return re.sub(r"^d=\d{8,10}:", "", inventory_line)


def compare_table(reference: Path, ours: Path, table: Path | None = None, set_name: str | None = None) -> int:
    """The grid section, and every table record with a source, against NOAA's file: the same inventory identity
    (parameter, surfaces, template, statistic and time), in NOAA's order. Exit 1 on any difference."""
    from namgpu.products import spec
    if set_name in (None, "auto"):
        set_name = next((k for k in PSETS.NEST_SETS if f".{k}." in Path(ours).name), "awphys")
    ps = PSETS.get(set_name)
    table = table or PSETS.table_path(ps)
    ref, mine = scan(reference), scan(ours)
    bad = 0
    g_ref, g_mine = ref[0].get("gds"), mine[0].get("gds") if mine else None
    same_grid = all(r.get("gds") == g_ref for r in mine)
    print(f"grid section (section 3 bytes): {'same as NOAA' if same_grid else 'DIFFERS from NOAA'}")
    if not same_grid:
        bad += 1
        print(f"  NOAA {g_ref}\n  ours {g_mine}")

    def key(r):
        return (r["key"], r.get("t1"), float(r.get("v1") or 0.0), r.get("t2") if r.get("t2") != 255 else None,
                None if r.get("t2") in (None, 255) else float(r.get("v2") or 0.0),
                r.get("stat") if r.get("pdt") == 8 else None)

    def line(r):
        return ":".join(idx_lines([r], ps.name)[0].split(":")[2:])
    # the n-th table row of an identity (APCP over two windows) is the n-th record of it in each file
    ref_at, mine_at = {}, {}
    for i, r in enumerate(ref):
        ref_at.setdefault(key(r), []).append((i, r))
    for i, r in enumerate(mine):
        mine_at.setdefault(key(r), []).append((i, r))
    all_rows = spec.read(table)
    seen_all = {}
    for t in all_rows:
        t["_occ"] = seen_all.setdefault(spec.inventory_key(t), [0])[0]
        seen_all[spec.inventory_key(t)][0] += 1
    rows = [r for r in all_rows if r["source"] not in ("", "-")]
    print(f"table records with a source: {len(rows)}; ours has {len(mine)} records\n")
    last = -1
    absent = 0
    for t in rows:
        k = spec.inventory_key(t)
        n = t["_occ"]
        a = ref_at.get(k, [])[n] if n < len(ref_at.get(k, [])) else None
        b = mine_at.get(k, [])[n] if n < len(mine_at.get(k, [])) else None
        if a is None and b is None:
            print(f"  ABSENT       rec {t['rec']:>5}  {t['name']} {t['level']}: in neither file at this hour (as NOAA)")
            absent += 1
            continue
        if a is None:
            print(f"  NOT IN NOAA  rec {t['rec']:>5}  ours {b[0] + 1}:{line(b[1])}")
            bad += 1
            continue
        if b is None:
            print(f"  MISSING      rec {t['rec']:>5}  NOAA {a[0] + 1}:{line(a[1])}")
            bad += 1
            continue
        # the reference time (d=) is not part of a record identity: a run is compared with an archived NOAA cycle
        # of another date (NOAA retired the NAM on 2026-11-03 12Z, so every live cycle has a different date)
        same = _undated(line(a[1])) == _undated(line(b[1])) and a[1].get("pdt") == b[1].get("pdt") and \
            a[1].get("range_len") == b[1].get("range_len")
        order = b[0] > last
        last = b[0]
        tag = "OK" if same and order else ("ORDER" if same else "DIFF")
        bad += tag != "OK"
        print(f"  {tag:<12} rec {t['rec']:>5}  NOAA {a[0] + 1}:{line(a[1])}  ours {b[0] + 1}:{line(b[1])}")
    n_ok = len(rows) - absent - bad + (0 if same_grid else 1)
    print(f"\n{n_ok} of {len(rows) - absent} table records present at this hour match NOAA's inventory line and order"
          + (f"; {absent} absent from both, as in NOAA's file" if absent else "")
          + ("" if same_grid else "; the grid section differs"))
    return 1 if bad else 0


def compare(reference: Path, ours: Path) -> int:
    """Records of an archived NAM awphys file and of ours, side by side (same discipline/category/parameter and level)."""
    def keyed(path):
        out = {}
        for r in scan(path):
            k = (r["key"], r.get("t1"), r.get("v1"), r.get("t2"), r.get("v2"), r.get("pdt"), r.get("stat"))
            out.setdefault(k, r)
        return out
    ref, mine = keyed(reference), keyed(ours)
    both = [k for k in ref if k in mine]
    only_ref = [k for k in ref if k not in mine]
    only_mine = [k for k in mine if k not in ref]

    def label(k):
        return f"{NAMES.get(k[0], k[0])} ({k[0]}) {_level(k[1], k[2], k[3], k[4])}" + (" [stat]" if k[5] == 8 else "")
    print(f"reference {reference.name}: {len(ref)} records; ours {ours.name}: {len(mine)} records; "
          f"same record in both: {len(both)}")
    print("\nin both:\n  " + "\n  ".join(label(k) for k in both))
    print("\nonly in the reference:\n  " + "\n  ".join(label(k) for k in only_ref))
    print("\nonly in ours:\n  " + "\n  ".join(label(k) for k in only_mine))
    return 0


# Every product set needs a card, whatever --post-device says: the pinned writer computes its RRFS extras (the derived
# records of NOAA's files) on a CUDA device only; --post-device cpu moves the rest of its post to the processor (a
# referee run) and still opens the card for the extras. With no device it stopped at hour 0 with "the RRFS extras
# need a CUDA device", after the frame of hour 0 had been built on the processor (nam-differences-2, 2026-11-11).
NO_CARD = ("woof-nam products: no CUDA device ({why}): these GRIB2 files need a card, with any --post-device (the "
           "writer computes the derived records of NOAA's files, its RRFS extras, on a card only; cpu moves only the "
           "rest of its post to the processor)")
_CARD_PROBE = r"""
import ctypes, sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print("no CUDA driver library: %s" % e); sys.exit(1)
n = ctypes.c_int(0)
r = lib.cuInit(0) or lib.cuDeviceGetCount(ctypes.byref(n))
print("CUresult %d, %d devices visible" % (r, n.value)); sys.exit(0 if r == 0 and n.value > 0 else 1)
"""


def card_visible() -> tuple[bool, str]:
    """(whether a CUDA device opens here, the driver's own words): cuInit and cuDeviceGetCount in a short child, so
    this process holds no CUDA state (CUDA_VISIBLE_DEVICES is honoured as the writer honours it)."""
    try:
        p = subprocess.run([sys.executable, "-c", _CARD_PROBE], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"the device probe did not run: {e}"
    return p.returncode == 0, (p.stdout or p.stderr).strip().splitlines()[-1] if (p.stdout or p.stderr).strip() \
        else f"probe exit {p.returncode}"


def _children(pid: int) -> list[int]:
    """The processes whose parent is `pid` (from /proc; empty where there is none)."""
    out = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(rest) > 1 and rest[1] == str(pid):
            out.append(int(d))
    return out


def _stop_on_signal(signum, _frame):
    """SIGTERM/SIGHUP: end the field workers and the writers now, then leave by SystemExit so every frame folder in
    host RAM (wrf_frame's finally), every hour's work folder (_finish_hour's finally) and the pools are cleaned up.
    Breakage it prevents: a products process stopped by its PID left 13 processes and its /dev/shm frame folder
    behind (nam-differences-2, 2026-11-11), because the default SIGTERM ends Python with no cleanup at all."""
    import multiprocessing
    kids = [p.pid for p in multiprocessing.active_children() if p.pid]
    for pid in _children(os.getpid()):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"--profile-json" in f.read().split(b"\0"):       # a writer (_start_hour's command)
                    kids.append(pid)
        except OSError:
            pass
    for pid in kids:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    raise SystemExit(128 + signum)


def _install_stop_handlers() -> None:
    import signal
    try:
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, _stop_on_signal)
    except (ValueError, AttributeError):        # not the main thread, or no such signal here
        pass


def main(argv=None) -> int:
    global FILE_NAME
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1],
                                 formatter_class=HelpFormatter,
                                 epilog="examples:\n  woof-nam products --run runs/2025-03-14_12z\n"
                                        "  woof-nam products --run runs/2025-03-14_12z --hours 0-6\n"
                                        "  woof-nam products --compare <archived awphys file> <our awphys file>")
    ap.add_argument("--run", type=Path, help="the run folder (its wrfout_d01_* history files)")
    ap.add_argument("--out", type=Path, default=None, help="where the GRIB2 files go (default <run>/products)")
    ap.add_argument("--hours", default="all", help="forecast hours: all (default), 6, 0-24 or 0,3,6")
    ap.add_argument("--workers", type=int, default=2,
                    help="writer processes in flight, one hour each (default 2; each needs its share of host RAM: "
                         "a writer on the whole 1799 x 1059 conusnest grid passed 22 GB)")
    ap.add_argument("--threads", type=int, default=0,
                    help="writer threads per hour (default 0: every core the process may use)")
    ap.add_argument("--bits", type=int, default=DEFAULT_BITS,
                    help=f"bits per packed value, complex packing (default {DEFAULT_BITS}; a table row may set its own)")
    ap.add_argument("--set", default="auto", choices=("auto",) + tuple(PSETS.SETS),
                    help="the NOAA product file: auto (default: awphys for the 12 km CONUS parent; for a nest the NCEP "
                         "nest its run names, else hiresf on a Lambert box around it), or a set by name")
    ap.add_argument("--table", type=Path, default=None, help="the record table (default: the set's, namgpu/products/)")
    ap.add_argument("--follow", action="store_true",
                    help="watch a running forecast: write each hour as its history file completes, stop when the "
                         "run's forecast-receipt.json is written (woof-nam run --products starts this)")
    ap.add_argument("--until", type=Path, default=None,
                    help="with --follow: also stop once this file exists (a caller that knows the forecast ended)")
    ap.add_argument("--prefix", default=FILE_NAME.split(".t{HH}z")[0],
                    help="file name before .tHHz (default %(default)s)")
    ap.add_argument("--packing", default="complex", choices=("complex",),
                    help="GRIB2 packing: complex (NCEP complex packing with spatial differencing, as NOAA's files)")
    ap.add_argument("--post-device", default="gpu", choices=("gpu", "auto", "cpu"),
                    help="where the writer runs its post: gpu (default; a full card is waited for, never replaced by "
                         "the CPU), auto, or cpu (a referee run: the post on the processor). Every choice needs a "
                         "CUDA device: the writer computes the derived records of NOAA's files on a card only, so "
                         "with no device visible products stops before the first hour, in one line")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "OURS"),
                    help="check the grid section and every table record of one of our files against an archived "
                         "NAM awphys file (same inventory line, NOAA's order)")
    ap.add_argument("--list-both", nargs=2, type=Path, metavar=("REFERENCE", "OURS"),
                    help="list every record of an archived NAM awphys file and of ours, side by side")
    a = ap.parse_args(argv)
    if a.compare:
        return compare_table(*a.compare, table=a.table, set_name=a.set)
    if a.list_both:
        return compare(*a.list_both)
    if a.run is None:
        ap.error("--run is required (or --compare)")
    from namgpu import exporter, paths
    rd0 = a.run if a.run.is_absolute() else paths.work_root() / a.run
    set_name = a.set if a.set != "auto" else (detect_set(rd0) or "awphys")
    try:
        exporter.ensure()                  # before any hour is read: stage the pinned writer, or stop in one line
        if set_name != "awphys":
            exporter.require(("pair_next", "mask_carrier"), f"the {set_name} nest files")
        from namgpu.products import spec
        selected_table = a.table or PSETS.table_path(PSETS.get(set_name))
        additions = {"TMP_ON_CLOUD_TOP", "PRES_ON_LVL_OF_ADIAB_COND_FROM_SFC"}
        if any(r["source"].split("|", 1)[0] in additions for r in spec.read(selected_table)):
            exporter.require(("nam_inventory_1",), "cloud-top temperature and LCL pressure")
    except exporter.ExporterMissing as e:
        print(f"woof-nam products: {e}", file=sys.stderr)
        return 2
    ok, why = card_visible()           # before any hour is read: no card, no file (NO_CARD says why)
    if not ok:
        print(NO_CARD.format(why=why), file=sys.stderr)
        return 2
    _install_stop_handlers()
    from namgpu.io import scratch as _scr
    _scr.sweep()                       # frame folders of products runs that were killed (their owner has ended)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", a.prefix):
        ap.error(f"--prefix {a.prefix!r}: letters, digits, '.', '_' and '-' only")
    FILE_NAME = a.prefix + FILE_NAME[FILE_NAME.index(".t{HH}z"):]
    rd = a.run if a.run.is_absolute() else paths.work_root() / a.run
    res = run(rd, a.out, a.hours, a.workers, a.threads, a.post_device, a.bits, a.table, follow=a.follow,
              until=a.until, set_name=set_name)
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
