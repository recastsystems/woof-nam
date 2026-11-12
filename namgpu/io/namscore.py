"""woof-nam score (namscore): score WOOF-NAM arms against observations and draw obs | control | arm sheets, one command.

    woof-nam score --cycle 2026012312 --case-dir cases/2026-01-23_12z --leads 6,12,18,24 \
        --run control=runs/control --run mynn=runs/mynn --out score/mynn

    # control already scored once: reuse its scores and panels, read only the arm's history
    woof-nam score --cycle 2026012312 --case-dir cases/2026-01-23_12z \
        --reuse control=score/control/control --run mynn=runs/mynn --out score/mynn

The first --run/--reuse is the control; the rest are arms. Every run is a WOOF-NAM history folder
(wrfout_d01_YYYY-mm-dd_HH:00:00, hourly or at least at 0 and every lead).

Observations (fetched once into <case-dir>/obs, then reused by every arm):
  ASOS     rw_asos (IEM METARs, the 48 CONUS state networks): T2, Td2, 10 m wind speed at stations
  URMA     NOAA URMA 2.5 km analysis (noaa-urma-pds): the gridded obs panel for T2, Td2 and 10 m wind
  MRMS     the case's raw/mrms composite reflectivity (woof-nam fetch): FSS and the refl obs panel
  Stage IV rw_stage4 6 h QPE (IEM archive): 6 h precipitation scores and the QPE obs panel
  NAM      the case's archived NAM awphys files: T2/Td2/wind10 at the same stations, as a reference column

Scores (scores.json + SCORES.md in --out):
  ASOS:    n, bias, MAE, RMSE per lead for T2 (F), Td2 (F), 10 m wind speed (kt), every run plus archived NAM
  refl:    FSS at 20 and 35 dBZ, gridpoint and 5x5 cells (~60 km), and frequency bias, vs MRMS block-max on the
           model grid
  qpe6:    the 6 h window ending at each lead: bias (domain mean ratio), ETS at 0.10/0.50/1.00 in and FSS 5x5,
           vs Stage IV sampled at the model cells (Stage IV missing cells masked)

Sheets (--out/sheets/<product>/fHH.png): obs | control | arm(s), drawn by the Rust renderer rw_wrfbatch through
its CF path (render law) on one CONUS frame and ONE colour scale per sheet (wind: the 0-140 kt ladder for all
panels when any panel passes 60 kt). Products: t2m, d2m, wind10, composite_reflectivity, qpf_6h.

Runs anywhere with woof-nam and gpuwm installed (gpuwm's renderer and GRIB2 readers). The observation readers
rw_asos and rw_stage4 are release assets of woof-nam: `woof-nam fetch-bridges` stages them, and the first score
that needs one downloads it (pinned by size and SHA-256). Panels render in parallel (--workers).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

KT = 1.9438445
PRODUCTS = ("t2m", "d2m", "wind10", "composite_reflectivity", "qpf_6h")
CONUS_STATES = None  # filled from rw_asos networks (two-letter *_ASOS minus AK HI PR VI GU AS)
URMA_SEL = {"t2m": (0, 0, 0, 103, 2), "d2m": (0, 0, 6, 103, 2), "wind10": (0, 2, 1, 103, 10)}
NAM_SEL = {"t2m": (0, 0, 0, 103, 2), "d2m": (0, 0, 6, 103, 2), "u10": (0, 2, 2, 103, 10), "v10": (0, 2, 3, 103, 10)}
QPE_THR_IN = (0.10, 0.50, 1.00)
REFL_THR = (20.0, 35.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bridges() -> Path:
    from namgpu.bridges import bridges_dir
    return bridges_dir()


def _tool(name: str) -> Path:
    """rw_asos or rw_stage4: the copy in $GPUWM_BRIDGES, else the pinned one `woof-nam fetch-bridges` stages
    (downloaded on first use; namgpu/helpers.py). Before this the command stopped with 'set GPUWM_BRIDGES' on any
    install: neither program is inside the gpuwm wheel."""
    from namgpu import exporter, helpers
    try:
        return helpers.ensure(name)
    except exporter.ExporterMissing as e:
        raise SystemExit(f"woof-nam score: {e}") from None


# ------------------------------------------------------------------ observations (cached per case)
def fetch_obs(case: Path, cycle: datetime, leads: list[int]) -> Path:
    obs = case / "obs"
    obs.mkdir(parents=True, exist_ok=True)
    end = cycle + timedelta(hours=max(leads))
    jobs = []
    if not (obs / "asos_table.csv").is_file():
        rw_asos = _tool("rw_asos")

        def asos():
            subprocess.run([rw_asos, "networks", "--out", obs / "nets.json"], check=True, capture_output=True)
            import re
            nets = [n for n in json.loads((obs / "nets.json").read_text())["networks"]
                    if re.fullmatch(r"[A-Z]{2}_ASOS", n) and n[:2] not in ("AK", "HI", "PR", "VI", "GU", "AS")]
            subprocess.run([rw_asos, "fetch", "--networks", ",".join(nets),
                            "--start", f"{cycle - timedelta(minutes=30):%Y-%m-%dT%H:%M:%SZ}",
                            "--end", f"{end + timedelta(minutes=30):%Y-%m-%dT%H:%M:%SZ}",
                            "--out", obs / "asos.csv"], check=True, capture_output=True)
            subprocess.run([rw_asos, "table", "--obs", obs / "asos.csv", "--out", obs / "asos_table.csv"],
                           check=True, capture_output=True)
            return "asos"
        jobs.append(asos)
    for L in leads:
        v = cycle + timedelta(hours=L)
        u = obs / "urma" / f"urma_{v:%Y%m%d%H}.grb2"
        if not u.is_file():
            def urma(v=v, u=u):
                u.parent.mkdir(exist_ok=True)
                url = (f"https://noaa-urma-pds.s3.amazonaws.com/urma2p5.{v:%Y%m%d}/"
                       f"urma2p5.t{v:%H}z.2dvaranl_ndfd.grb2_wexp")
                subprocess.run(["curl", "-sSf", "-o", u, url], check=True)
                return u.name
            jobs.append(urma)
    st4 = obs / "st4"
    need = [cycle + timedelta(hours=L) for L in leads if L >= 6 and (L % 6 == 0)]
    if need and not all((st4 / f"st4_{v:%Y%m%d%H}.pack").is_file() for v in need):
        rw_stage4 = _tool("rw_stage4")

        def stage4():
            st4.mkdir(exist_ok=True)
            subprocess.run([rw_stage4, "fetch", "--accumulation", "06h",
                            "--start", f"{min(need):%Y-%m-%dT%H:%M:%SZ}", "--end", f"{max(need):%Y-%m-%dT%H:%M:%SZ}",
                            "--cache", st4 / "cache", "--out", st4 / "raw"], check=True, capture_output=True)
            for v in need:
                f = next((st4 / "cache").rglob(f"ST4.{v:%Y%m%d%H}.06h*"))
                subprocess.run([rw_stage4, "decode", "--file", f, "--out", st4 / f"st4_{v:%Y%m%d%H}.pack"],
                               check=True, capture_output=True)
                if not (st4 / "geo.pack").is_file():
                    subprocess.run([rw_stage4, "grid", "--file", f, "--out", st4 / "geo.pack"],
                                   check=True, capture_output=True)
            return "stage4"
        jobs.append(stage4)
    if jobs:
        with ThreadPoolExecutor(len(jobs)) as ex:
            for r in ex.map(lambda j: j(), jobs):
                log("obs fetched:", r)
    return obs


def read_asos(obs: Path, valid: datetime, window_s=600):
    """{station: (lat, lon, T_K, Td_K, speed_ms)} for the report nearest valid within window_s."""
    import csv
    best = {}
    want = {"temperature_k": 0, "dewpoint_k": 1, "wind_u_m_s": 2, "wind_v_m_s": 3}
    with open(obs / "asos_table.csv") as fh:
        for r in csv.DictReader(fh):
            k = want.get(r["variable"])
            if k is None:
                continue
            t = datetime.strptime(r["valid_time"], "%Y-%m-%dT%H:%M:%SZ")
            dt = abs((t - valid).total_seconds())
            if dt > window_s:
                continue
            s = r["station_id"]
            e = best.setdefault(s, {"lat": float(r["latitude_deg"]), "lon": float(r["longitude_deg"]), "v": {}})
            if k not in e["v"] or dt < e["v"][k][0]:
                e["v"][k] = (dt, float(r["value"]))
    out = {}
    for s, e in best.items():
        v = e["v"]
        T = v.get(0, (0, np.nan))[1]
        Td = v.get(1, (0, np.nan))[1]
        sp = np.hypot(v[2][1], v[3][1]) if 2 in v and 3 in v else np.nan
        out[s] = (e["lat"], e["lon"], T, Td, sp)
    return out


def _grib_field(path: Path, sel, work: Path):
    """(values[j,i] south->north, row) of the one record matching sel = (disc, cat, par, lt, lv)."""
    from namgpu.init.init_grib2_decode import inventory
    from namgpu.io.io_referee_panels import _grib_values, _place, _ints
    inv = inventory(path)
    hits = [r for r in inv if (_ints(r, "discipline"), _ints(r, "category"), _ints(r, "parameter"),
                               _ints(r, "level_type"), _ints(r, "level_value")) == tuple(sel)]
    if len(hits) != 1:
        raise ValueError(f"{path.name}: {len(hits)} records match {sel}")
    return _place(_grib_values(path, hits[0], work), hits[0]).astype(np.float64), hits[0]


def _lambert_latlon(row, radius=6371200.0):
    from namgpu.io.io_map_regrid import LambertFrame, lambert_forward, lambert_inverse
    lon1 = float(row["lon1"]); lon1 = lon1 - 360.0 if lon1 > 180 else lon1
    lov = float(row["lov"]); lov = lov - 360.0 if lov > 180 else lov
    f = LambertFrame(name="grid", nx=row["nx"], ny=row["ny"], dx=float(row["dx"]), dy=float(row["dy"]),
                     lat1=float(row["latin1"]), lat2=float(row["latin2"]), lov=lov,
                     lat_ll=float(row["lat1"]), lon_ll=lon1, radius=radius)
    x0, y0 = lambert_forward(f, f.lat_ll, f.lon_ll)
    X, Y = np.meshgrid(x0 + f.dx * np.arange(f.nx), y0 + f.dy * np.arange(f.ny))
    return lambert_inverse(f, X, Y)


def urma_fields(obs: Path, valid: datetime):
    u = obs / "urma" / f"urma_{valid:%Y%m%d%H}.grb2"
    out, lat, lon = {}, None, None
    with tempfile.TemporaryDirectory() as td:
        for p, sel in URMA_SEL.items():
            v, row = _grib_field(u, sel, Path(td) / p)
            out[p] = v
            if lat is None:
                lat, lon = _lambert_latlon(row)
    out["t2m"] = (out["t2m"] - 273.15) * 1.8 + 32
    out["d2m"] = (out["d2m"] - 273.15) * 1.8 + 32
    return out, lat, lon


def nam_archive_fields(case: Path, cycle: datetime, L: int):
    from namgpu.io.io_referee_panels import _archive_file, _nam218_latlon
    f = _archive_file(case / "raw" / "nam", L)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for p, sel in NAM_SEL.items():
            out[p], _ = _grib_field(f, sel, Path(td) / p)
    lat, lon = _nam218_latlon()
    return {"T2": out["t2m"], "TD2": out["d2m"], "SPD": np.hypot(out["u10"], out["v10"])}, lat, lon


def read_mrms(case: Path, valid: datetime):
    from namgpu.init.init_grib2_decode import inventory
    from namgpu.io.io_referee_panels import _grib_values, _place
    g = case / "raw" / "mrms" / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
    inv = inventory(g)
    row = inv[0]
    with tempfile.TemporaryDirectory() as td:
        a = _place(_grib_values(g, row, Path(td) / "m"), row).astype(np.float64)
    a[a < -500.0] = np.nan      # -999 = outside coverage (missing); -99 = no echo, a valid observation
    lat0, lon0 = float(row["lat1"]), float(row["lon1"])
    lon0 = lon0 - 360 if lon0 > 180 else lon0
    lat = (lat0 - float(row["dy"]) * np.arange(row["ny"]))[::-1]
    lon = lon0 + float(row["dx"]) * np.arange(row["nx"])
    return a, lat, lon


# ------------------------------------------------------------------ model history
def history_frame(run: Path, valid: datetime, names):
    from namgpu.io.maps import _read
    p = run / f"wrfout_d01_{valid:%Y-%m-%d_%H}:00:00"
    if not p.is_file():
        raise FileNotFoundError(f"{p}: no history frame at {valid:%Y-%m-%d %HZ}")
    return _read(p, set(names))[2]


def model_fields(run: Path, cycle: datetime, L: int):
    from namgpu.io.maps import _dewpoint_k
    v = cycle + timedelta(hours=L)
    f = history_frame(run, v, {"XLAT", "XLONG", "T2", "Q2", "PSFC", "U10", "V10", "COMPOSITE", "RAINNC", "RAINC"})
    out = {"lat": f["XLAT"], "lon": f["XLONG"], "T2": (f["T2"] - 273.15) * 1.8 + 32,
           "TD2": (_dewpoint_k(np.maximum(f["Q2"], 0.0), f["PSFC"]) - 273.15) * 1.8 + 32,
           "SPD": np.hypot(f["U10"], f["V10"]),
           "REFC": np.maximum(f.get("COMPOSITE", np.full_like(f["T2"], np.nan)), -30.0),
           "ACC": f["RAINNC"] + f["RAINC"]}
    if L >= 6:
        p = history_frame(run, v - timedelta(hours=6), {"RAINNC", "RAINC"})
        out["QPE6"] = out["ACC"] - (p["RAINNC"] + p["RAINC"])
    return out


# ------------------------------------------------------------------ verification maths
def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def nearest_index(lat, lon, qlat, qlon, max_km=15.0):
    from scipy.spatial import cKDTree
    t = cKDTree(_xyz(np.ravel(lat), np.ravel(lon)))
    d, i = t.query(_xyz(np.asarray(qlat), np.asarray(qlon)), workers=-1)
    ok = d * 6371.0 <= max_km
    return i, ok


def point_stats(model, obs):
    m = np.isfinite(model) & np.isfinite(obs)
    e = model[m] - obs[m]
    if not e.size:
        return dict(n=0, bias=None, mae=None, rmse=None)
    return dict(n=int(e.size), bias=round(float(e.mean()), 2), mae=round(float(np.abs(e).mean()), 2),
                rmse=round(float(np.sqrt((e ** 2).mean())), 2))


def _box(a, n):
    """n x n box mean with NaN-free input (zero padded at edges), via integral image."""
    if n == 1:
        return a
    p = n // 2
    s = np.pad(a, ((p + 1, p), (p + 1, p))).cumsum(0).cumsum(1)
    return (s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]) / (n * n)


def fss(fc, ob, mask, thr, n):
    f = np.where(mask, fc >= thr, 0).astype(np.float64)
    o = np.where(mask, ob >= thr, 0).astype(np.float64)
    pf, po = _box(f, n)[mask], _box(o, n)[mask]
    num = np.mean((pf - po) ** 2)
    den = np.mean(pf ** 2) + np.mean(po ** 2)
    return None if den == 0 else round(float(1 - num / den), 3)


def ets(fc, ob, mask, thr):
    f, o = (fc >= thr) & mask, (ob >= thr) & mask
    a = float((f & o).sum()); b = float((f & ~o).sum()); c = float((~f & o).sum()); N = float(mask.sum())
    ar = (a + b) * (a + c) / N if N else 0
    den = a + b + c - ar
    return (None if den <= 0 else round((a - ar) / den, 3)), (None if a + c == 0 else round((a + b) / (a + c), 3))


# ------------------------------------------------------------------ scoring one run
def score_run(name, run: Path, case: Path, cycle: datetime, leads, obs: Path, out: Path):
    from scipy.ndimage import maximum_filter
    from gpuwm.obs.obspack import read_pack
    res = {"run": str(run), "leads": {}}
    fields = {}
    for L in leads:
        v = cycle + timedelta(hours=L)
        M = model_fields(run, cycle, L)
        fields[L] = M
        r = {}
        st = read_asos(obs, v)
        names = sorted(st)
        S = np.array([st[s] for s in names])
        idx, ok = nearest_index(M["lat"], M["lon"], S[:, 0], S[:, 1])
        for key, col, conv in (("T2", 2, lambda k: (k - 273.15) * 1.8 + 32),
                               ("TD2", 3, lambda k: (k - 273.15) * 1.8 + 32), ("SPD", 4, lambda x: x * KT)):
            mv = np.ravel(M[key])[idx] * (KT if key == "SPD" else 1)
            ob = conv(S[:, col])
            r[f"asos_{key}"] = point_stats(np.where(ok, mv, np.nan), ob)
        # No MRMS file for a lead = woof-nam fetch found no complete mosaic within 10 min (an archive radar dropout,
        # e.g. 2023-04-01 11Z): that lead's reflectivity is not scored, and the rest of the case still is.
        if not (case / "raw" / "mrms" / f"mrms_refc_{v:%Y%m%d%H}.grib2").is_file():
            r["refl_missing"] = f"no complete MRMS mosaic within 10 min of {v:%Y-%m-%d %HZ}"
            log(f"{name} +{L}h: {r['refl_missing']}; reflectivity not scored")
        else:
            # reflectivity vs MRMS block max on the model grid (MRMS 0.01 deg; ~12 km = 11 pixels)
            a, mlat, mlon = read_mrms(case, v)
            amax = maximum_filter(np.nan_to_num(a, nan=-999.0), size=11)
            valid_mrms = maximum_filter(np.isfinite(a).astype(np.uint8), size=11) > 0
            i = np.round((M["lat"] - mlat[0]) / (mlat[1] - mlat[0])).astype(int)
            j = np.round((M["lon"] - mlon[0]) / (mlon[1] - mlon[0])).astype(int)
            inside = (i >= 0) & (i < a.shape[0]) & (j >= 0) & (j < a.shape[1])
            ic, jc = np.clip(i, 0, a.shape[0] - 1), np.clip(j, 0, a.shape[1] - 1)
            ob = np.where(inside, amax[ic, jc], np.nan)
            mask = inside & valid_mrms[ic, jc] & np.isfinite(M["REFC"])
            r["refl"] = {f"{int(t)}dBZ": {"fss1": fss(M["REFC"], ob, mask, t, 1), "fss5": fss(M["REFC"], ob, mask, t, 5),
                                          "freq_bias": (None if not (ob[mask] >= t).sum() else
                                                        round(float((M["REFC"][mask] >= t).sum() / (ob[mask] >= t).sum()), 3))}
                         for t in REFL_THR}
        # 6 h QPE vs Stage IV
        if "QPE6" in M and (obs / "st4" / f"st4_{v:%Y%m%d%H}.pack").is_file():
            p = read_pack(obs / "st4" / f"st4_{v:%Y%m%d%H}.pack")
            g = read_pack(obs / "st4" / "geo.pack")
            sv = np.where(p.array("valid") > 0, p.array("values"), np.nan)
            k, ok4 = nearest_index(g.array("latitude"), g.array("longitude"), np.ravel(M["lat"]), np.ravel(M["lon"]), 4.0)
            ob = np.where(ok4, np.ravel(sv)[k], np.nan).reshape(M["lat"].shape) / 25.4
            fc = M["QPE6"] / 25.4
            mask = np.isfinite(ob)
            q = {"mean_ratio": None if not np.nanmean(ob[mask]) else round(float(fc[mask].mean() / ob[mask].mean()), 3)}
            for t in QPE_THR_IN:
                e, fb = ets(fc, ob, mask, t)
                q[f"{t:.2f}in"] = {"ets": e, "freq_bias": fb, "fss5": fss(fc, ob, mask, t, 5)}
            r["qpe6"] = q
        res["leads"][str(L)] = r
        log(name, f"f{L:02d}", json.dumps({k: r[k] for k in r if k.startswith("asos")}))
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "scores.json").write_text(json.dumps(res, indent=1))
    return res, fields


def score_nam_archive(case, cycle, leads, obs):
    res = {"leads": {}}
    for L in leads:
        try:
            F, lat, lon = nam_archive_fields(case, cycle, L)
        except (FileNotFoundError, ValueError) as e:
            log("archived NAM skipped", L, e)
            continue
        st = read_asos(obs, cycle + timedelta(hours=L))
        S = np.array([st[s] for s in sorted(st)])
        idx, ok = nearest_index(lat, lon, S[:, 0], S[:, 1])
        r = {}
        for key, col, conv in (("T2", 2, lambda k: (k - 273.15) * 1.8 + 32), ("TD2", 3, lambda k: (k - 273.15) * 1.8 + 32),
                               ("SPD", 4, lambda x: x * KT)):
            fv = F[key]
            if key in ("T2", "TD2"):
                fv = (fv - 273.15) * 1.8 + 32
            else:
                fv = fv * KT
            r[f"asos_{key}"] = point_stats(np.where(ok, np.ravel(fv)[idx], np.nan), conv(S[:, col]))
        res["leads"][str(L)] = r
    return res


# ------------------------------------------------------------------ panels and sheets
def _render(job):
    """One CF panel through rw_wrfbatch. job = (nc, cf_var, row, valid, png)."""
    from gpuwm import rustwx
    from namgpu.io.maps import BOUNDS
    nc, var, row, valid, png = job
    from namgpu.io import maps
    # NAMGPU_RENDERER (the pinned build), else the installed gpuwm wheel's renderer (built for this machine's
    # C library; a bridges copy from another host may not load), else the bridges copy
    ren = os.environ.get("NAMGPU_RENDERER") or rustwx.find_renderer() or (bridges() / "rw_wrfbatch")
    ren = Path(ren)
    maps.ensure_basemap(ren)
    td = Path(tempfile.mkdtemp(prefix="nsp-", dir=png.parent))
    try:
        tab = td / "t.json"
        tab.write_text(json.dumps([row]))
        rc = rustwx.run_cf_renderer(ren, Path(nc), variable=var, out_dir=td, valid=f"{valid:%Y-%m-%d %H:%M}",
                                    table=tab, bounds=os.environ.get("NAMSCORE_BOUNDS") or BOUNDS)
        pngs = sorted(td.rglob("*.png"))
        if rc or len(pngs) != 1:
            raise RuntimeError(f"renderer rc {rc}, {len(pngs)} png for {png}")
        shutil.move(str(pngs[0]), str(png))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return png


def write_cf(path, valid, var, values, units, lat, lon, source):
    from namgpu.io.io_referee_panels import _write_cf
    return _write_cf(Path(path), valid, var, np.asarray(values, np.float32), units, lat, lon, source=source)


def panel_jobs(src_label, fields_by_lead, lat_lon, cycle, leads, pdir: Path, td: Path, scales):
    """fields_by_lead[L] = {product: values}; lat_lon = (lat, lon) or per-lead dict."""
    from namgpu.io.maps import TABLE, cf_row
    jobs = []
    for L in leads:
        v = cycle + timedelta(hours=L)
        lat, lon = lat_lon[L] if isinstance(lat_lon, dict) else lat_lon
        for prod, vals in fields_by_lead.get(L, {}).items():
            var = TABLE[prod][0]
            units = TABLE[prod][1]["default_units"]
            title = f"{TABLE[prod][1]['title']}  {src_label}  valid {v:%Y-%m-%d %H}Z (+{L}h)"
            row = cf_row(prod, title, scale=scales.get((prod, L)))
            png = pdir / prod / f"f{L:02d}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            nc = write_cf(td / f"{src_label.replace(' ', '_')}_{prod}_{L}.nc", v, var, vals, units, lat, lon, src_label)
            jobs.append((nc, var, row, v, png))
    return jobs


def model_panel_fields(M, L):
    d = {"t2m": M["T2"], "d2m": M["TD2"], "wind10": M["SPD"], "composite_reflectivity": M["REFC"]}
    if "QPE6" in M:
        d["qpf_6h"] = M["QPE6"]
    return d


def compose(sheet, pngs, caption):
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in pngs]
    h = max(i.height for i in ims)
    W = sum(i.width for i in ims)
    cap = 34
    out = Image.new("RGB", (W, h + cap), "white")
    x = 0
    for i in ims:
        out.paste(i, (x, cap)); x += i.width
    ImageDraw.Draw(out).text((10, 8), caption, fill="black")
    sheet.parent.mkdir(parents=True, exist_ok=True)
    out.save(sheet)
    return sheet


def fmt(s, k):
    x = s.get(k) or {}
    return "-" if not x or x.get("n", 1) == 0 else f"{x['bias']:+.1f}/{x['mae']:.1f}/{x['rmse']:.1f} (n {x['n']})"


def scores_md(cycle, leads, runs, nam):
    lines = [f"# Scores, cycle {cycle:%Y-%m-%d %HZ}", "",
             "ASOS cells are bias/MAE/RMSE (n). T2 and Td2 in F, wind speed in kt. Lower MAE/RMSE is better.", ""]
    for key, lab in (("asos_T2", "2 m temperature (F)"), ("asos_TD2", "2 m dewpoint (F)"), ("asos_SPD", "10 m wind speed (kt)")):
        lines += [f"## {lab}", "", "| lead | " + " | ".join(list(runs) + ["archived NAM"]) + " |",
                  "|---|" + "---|" * (len(runs) + 1)]
        for L in leads:
            cells = [fmt(runs[n]["leads"].get(str(L), {}), key) for n in runs]
            cells.append(fmt(nam["leads"].get(str(L), {}), key))
            lines.append(f"| +{L} h | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Composite reflectivity vs MRMS", "", "FSS gridpoint / 5x5 cells, frequency bias.", "",
              "| lead | thr | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    for L in leads:
        for t in REFL_THR:
            k = f"{int(t)}dBZ"
            cells = []
            for n in runs:
                x = runs[n]["leads"].get(str(L), {}).get("refl", {}).get(k, {})
                cells.append(f"{x.get('fss1')} / {x.get('fss5')}, fb {x.get('freq_bias')}" if x else "-")
            lines.append(f"| +{L} h | {int(t)} dBZ | " + " | ".join(cells) + " |")
    lines += ["", "## 6 h precipitation vs Stage IV (window ending at the lead)", "",
              "ETS / frequency bias / FSS 5x5 per threshold; mean ratio = model mean / Stage IV mean over valid cells.", "",
              "| lead | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for L in leads:
        cells = []
        for n in runs:
            q = runs[n]["leads"].get(str(L), {}).get("qpe6")
            if not q:
                cells.append("-"); continue
            parts = [f"ratio {q['mean_ratio']}"] + [f"{t:.2f}in {q[f'{t:.2f}in']['ets']}/{q[f'{t:.2f}in']['freq_bias']}/{q[f'{t:.2f}in']['fss5']}" for t in QPE_THR_IN]
            cells.append("; ".join(parts))
        lines.append(f"| +{L} h | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--winter" in args:
        from namgpu.io import namwinter
        args.remove("--winter")
        return namwinter.main(args)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", action="store_true", help="winter/CAD scorer; use --winter --help for its inputs")
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--leads", default="6,12,18,24")
    class Spec(argparse.Action):
        def __call__(self, parser, ns, value, opt=None):
            ns.specs.append((self.dest, value))
    ap.set_defaults(specs=[])
    ap.add_argument("--run", action=Spec, help="NAME=HISTORY_DIR (the first --run or --reuse is the control)")
    ap.add_argument("--reuse", action=Spec, help="NAME=SCORE_DIR of a run scored before (its scores.json + panels/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--bounds", default=None, help="sheet frame W,E,S,N (default the CONUS frame -126,-66,22,52); "
                    "a regional frame shows differences the CONUS colour scale hides")
    a = ap.parse_args(argv)
    if a.bounds:
        os.environ["NAMSCORE_BOUNDS"] = a.bounds
    t0 = time.time()
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    leads = [int(x) for x in a.leads.split(",")]
    order = [v.split("=", 1)[0] for _, v in a.specs]
    a.out.mkdir(parents=True, exist_ok=True)
    obs = fetch_obs(a.case_dir, cycle, leads)
    runs, fields = {}, {}
    for spec in [v for k, v in a.specs if k == "reuse"]:
        n, d = spec.split("=", 1)
        runs[n] = json.loads((Path(d) / "scores.json").read_text())
        dst = a.out / n
        if Path(d).resolve() != dst.resolve():
            shutil.copytree(d, dst, dirs_exist_ok=True)
    for spec in [v for k, v in a.specs if k == "run"]:
        n, d = spec.split("=", 1)
        runs[n], fields[n] = score_run(n, Path(d), a.case_dir, cycle, leads, obs, a.out)
    nam = score_nam_archive(a.case_dir, cycle, leads, obs)
    (a.out / "nam_archive_scores.json").write_text(json.dumps(nam, indent=1))
    runs = {n: runs[n] for n in order}
    (a.out / "scores.json").write_text(json.dumps({"cycle": a.cycle, "leads": leads, "runs": runs, "nam_archive": nam}, indent=1))
    (a.out / "SCORES.md").write_text(scores_md(cycle, leads, runs, nam))
    log("scores written", a.out / "SCORES.md", f"{time.time() - t0:.0f} s")
    if a.no_sheets:
        return 0
    # ---- panels: obs (cached per case), every scored run; one colour scale per (product, lead)
    from namgpu.io.maps import wind_hi_scale, WIND_LADDER_TOP_KT
    from gpuwm.obs.obspack import read_pack
    # One wind scale per sheet, decided from every panel drawn here INCLUDING the URMA panel. The URMA wind10 panel is
    # drawn per call into <out>/obs_panels, never into the shared case cache: breakage this prevents (gwd-ugwp lane,
    # 2026-11-09): the cached URMA wind panel was redrawn on the 0-140 kt scale by whichever lane's arm last
    # passed 60 kt, so every other lane's sheet showed URMA on 0-140 kt beside model panels on 0-60 kt.
    scales, urma_cache = {}, {}
    for L in leads:
        tops = [np.nanmax(fields[n][L]["SPD"]) for n in fields]
        try:
            urma_cache[L] = urma_fields(obs, cycle + timedelta(hours=L))
            tops.append(float(np.nanmax(urma_cache[L][0]["wind10"])))
        except Exception as e:                       # no URMA for this lead: the sheet is drawn without its obs panel
            log(f"URMA wind for +{L} h not read ({e}); scale decided from the model panels")
        if tops and max(tops) * KT > WIND_LADDER_TOP_KT:
            scales[("wind10", L)] = wind_hi_scale()
    pdir_name = "panels" if not a.bounds else "panels_" + a.bounds.replace(",", "_")
    td = Path(tempfile.mkdtemp(prefix="namscore-", dir=a.out))
    jobs = []
    try:
        opd = a.case_dir / "obs" / ("panels" if not a.bounds else "panels_" + a.bounds.replace(",", "_"))
        ofields, olatlon = {}, {}
        for L in leads:
            v = cycle + timedelta(hours=L)
            need = [p for p in PRODUCTS if p != "wind10" and not (opd / p / f"f{L:02d}.png").is_file()]
            if L in urma_cache:
                U, ulat, ulon = urma_cache[L]
                jobs += panel_jobs("URMA analysis", {L: {"wind10": U["wind10"]}}, {L: (ulat, ulon)}, cycle, [L],
                                   a.out / "obs_panels", td, scales)
            if not need:
                continue
            if any(p in need for p in ("t2m", "d2m")):
                U, ulat, ulon = urma_cache[L] if L in urma_cache else urma_fields(obs, v)
                jobs += panel_jobs("URMA analysis", {L: {p: U[p] for p in ("t2m", "d2m") if p in need}},
                                   {L: (ulat, ulon)}, cycle, [L], opd, td, scales)
            if "composite_reflectivity" in need and (a.case_dir / "raw" / "mrms" / f"mrms_refc_{v:%Y%m%d%H}.grib2").is_file():
                mr, mlat, mlon = read_mrms(a.case_dir, v)
                LON, LAT = np.meshgrid(mlon, mlat)
                jobs += panel_jobs("MRMS observed", {L: {"composite_reflectivity": np.where(mr < -30, np.nan, mr)}},
                                   {L: (LAT, LON)}, cycle, [L], opd, td, scales)
            sp = obs / "st4" / f"st4_{v:%Y%m%d%H}.pack"
            if "qpf_6h" in need and sp.is_file():
                p = read_pack(sp); g = read_pack(obs / "st4" / "geo.pack")
                jobs += panel_jobs("Stage IV observed", {L: {"qpf_6h": np.where(p.array("valid") > 0, p.array("values"), np.nan)}},
                                   {L: (g.array("latitude"), g.array("longitude"))}, cycle, [L], opd, td, scales)
        for n in fields:
            jobs += panel_jobs(n, {L: model_panel_fields(fields[n][L], L) for L in leads},
                               {L: (fields[n][L]["lat"], fields[n][L]["lon"]) for L in leads}, cycle, leads,
                               a.out / n / pdir_name, td, scales)
        log(f"rendering {len(jobs)} panels on {a.workers} workers")
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(_render, jobs))
        for n in fields:                             # which wind scale each drawn panel is on (read by a later --reuse)
            for L in leads:
                mk = a.out / n / pdir_name / "wind10" / f"f{L:02d}.scale"
                mk.parent.mkdir(parents=True, exist_ok=True)
                mk.write_text("hi" if ("wind10", L) in scales else "std")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    made = []
    for L in leads:
        for prod in PRODUCTS:
            obs_png = (a.out / "obs_panels" if prod == "wind10" else opd) / prod / f"f{L:02d}.png"
            pngs = [obs_png] + [a.out / n / pdir_name / prod / f"f{L:02d}.png" for n in order]
            if not all(p.is_file() for p in pngs):
                continue
            off = ""
            if prod == "wind10":                     # a reused panel drawn on another scale is named, not hidden
                want = "hi" if ("wind10", L) in scales else "std"
                mk = lambda n: a.out / n / pdir_name / "wind10" / f"f{L:02d}.scale"
                odd = [n for n in order if (mk(n).read_text().strip() if mk(n).is_file() else "unknown") != want]
                if odd:
                    off = f"   NOTE {', '.join(odd)} panel not on this sheet's {'0-140' if want == 'hi' else '0-60'} kt scale"
            key = {"t2m": "asos_T2", "d2m": "asos_TD2", "wind10": "asos_SPD"}.get(prod)
            if key:
                cap = "  |  ".join(f"{n}: ASOS {fmt(runs[n]['leads'][str(L)], key)}" for n in order)
            elif prod == "composite_reflectivity":
                cap = "  |  ".join(f"{n}: FSS5 35dBZ {runs[n]['leads'][str(L)]['refl']['35dBZ']['fss5']}" for n in order)
            else:
                cap = "  |  ".join(f"{n}: ETS 0.50in {runs[n]['leads'][str(L)].get('qpe6', {}).get('0.50in', {}).get('ets')}" for n in order)
            made.append(compose(a.out / ("sheets" if not a.bounds else "sheets_" + a.bounds.replace(",", "_")) / prod / f"f{L:02d}.png", pngs,
                                f"{prod} +{L} h   obs | " + " | ".join(order) + "      " + cap + off))
    log(f"{len(made)} sheets in {a.out / 'sheets'}  total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
