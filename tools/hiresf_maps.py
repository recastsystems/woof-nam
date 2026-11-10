"""NOAA | WOOF-NAM | difference map sheets of nest GRIB2 records, drawn by the Rust renderer (rw_wrfbatch).

Decodes the chosen records of NOAA's archived nest file and of ours (gpuwm's grib2_dump), puts each on the part of
the nest grid our run covers (the window in our run index, so the sheet shows the nest, not a mostly empty grid),
writes them as WRF-shaped files on that projection and draws `var:` products with --diff-against --diff-sheet.

    python tools/hiresf_maps.py <NOAA hiresf file> <our hiresf file> <our products/index.json> <out dir> NAME:LEVEL...

NAME:LEVEL is the .idx text, e.g. "REFC:entire atmosphere (considered as a single layer)", "TMP:2 m above ground".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.bridges import bridges_dir  # noqa: E402
from namgpu.products import sets as PS  # noqa: E402

B = bridges_dir()


def idx(path):
    out = []
    for ln in Path(str(path) + ".idx").read_text().splitlines():
        f = ln.split(":")
        out.append(":".join(f[3:5]))
    return out


def dump(path, picks, work, ny, nx):
    d = os.path.join(work, os.path.basename(str(path)) + "_d")
    subprocess.run([str(B / "grib2_dump"), str(path), d, *map(str, picks)], check=True, capture_output=True)
    out = {}
    for line in open(os.path.join(d, "metadata.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        out[int(f[0])] = np.fromfile(os.path.join(d, f[-1]), "<f8").reshape(ny, nx).astype(np.float32)
    return out


def write(path, lat, lon, fields, g, valid, start):
    import netCDF4
    ny, nx = lat.shape
    fa = g.frame_attrs()
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("Time", None)
        d.createDimension("DateStrLen", 19)
        d.createDimension("south_north", ny)
        d.createDimension("west_east", nx)
        t = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.frombuffer(valid.encode(), dtype="S1")
        for n, a in (("XLAT", lat), ("XLONG", lon)):
            v = d.createVariable(n, "f4", ("Time", "south_north", "west_east"))
            v[0] = a
        for n, a in fields.items():
            v = d.createVariable(n, "f4", ("Time", "south_north", "west_east"))
            v[0] = a
        cy, cx = ny // 2, nx // 2
        d.setncatts({"MAP_PROJ": np.int32(fa["MAP_PROJ"]), "MAP_PROJ_CHAR": fa["MAP_PROJ_CHAR"],
                     "TRUELAT1": np.float32(fa["TRUELAT1"]), "TRUELAT2": np.float32(fa["TRUELAT2"]),
                     "STAND_LON": np.float32(fa["STAND_LON"]), "CEN_LAT": lat[cy, cx], "CEN_LON": lon[cy, cx],
                     "MOAD_CEN_LAT": np.float32(fa["TRUELAT1"]), "DX": np.float32(g.dx), "DY": np.float32(g.dy),
                     "START_DATE": start, "SIMULATION_START_DATE": start, "GRID_ID": np.int32(1),
                     "WEST-EAST_GRID_DIMENSION": np.int32(nx + 1), "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1)})


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ref, ours, index, out = Path(argv[0]), Path(argv[1]), Path(argv[2]), Path(argv[3])
    want = argv[4:]
    doc = json.loads(index.read_text())
    ps = PS.get(doc["set"])
    gs = doc["grid_spec"]
    full = PS.OutGrid(gs["kind"], gs["nx"], gs["ny"], gs["la1"], gs["lo1"], gs["dx"], gs["dy"], lov=gs.get("lov", 0.0),
                      latin1=gs.get("latin1", 0.0), latin2=gs.get("latin2", 0.0), lad=gs.get("lad", 0.0))
    w = doc["window"]
    i0, j0, nx, ny = w["i0"], w["j0"], w["nx"], w["ny"]
    jj, ii = np.meshgrid(np.arange(j0, j0 + ny, dtype=np.float64), np.arange(i0, i0 + nx, dtype=np.float64),
                         indexing="ij")
    lat, lon = full.latlon_ij(ii, jj)
    lat, lon = lat.astype(np.float32), (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)
    ia, ib = idx(ref), idx(ours)
    pick = {}
    for wname in want:
        if wname in ia and wname in ib:
            pick[wname] = (ia.index(wname), ib.index(wname))
        else:
            print("not in both:", wname)
    out.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(prefix="hiresf-maps-", dir=out)
    try:
        da = dump(ref, sorted(p[0] for p in pick.values()), work, full.ny, full.nx)
        db = dump(ours, sorted(p[1] for p in pick.values()), work, full.ny, full.nx)
        cut = (slice(j0, j0 + ny), slice(i0, i0 + nx))
        fa, fb = {}, {}
        for wname, (a, b) in pick.items():
            nm = "R_" + "".join(c if c.isalnum() else "_" for c in wname.replace(" above ground", "").replace(
                "entire atmosphere (considered as a single layer)", "column"))[:28]
            x, y = da[a][cut], db[b][cut]
            ok = np.isfinite(y)
            fa[nm] = np.where(ok, x, np.nan).astype(np.float32)     # NOAA's values where our nest has values
            fb[nm] = y
        f0 = next(f for f in doc["files"] if f["file"] == ours.name)
        valid = f0["valid"].replace("T", "_").replace("Z", "")
        start = valid
        a_dir, b_dir = Path(work) / "noaa", Path(work) / "ours"
        a_dir.mkdir()
        b_dir.mkdir()
        fa_path, fb_path = a_dir / f"wrfout_d01_{valid}", b_dir / f"wrfout_d01_{valid}"
        write(fa_path, lat, lon, fa, full, valid, start)
        write(fb_path, lat, lon, fb, full, valid, start)
        prods = ",".join(f"var:wrf_{nm.lower()}" for nm in fa)
        cmd = [str(B / "rw_wrfbatch"), "--store-root", str(Path(work) / "store"), "--out-dir", str(out),
               "--products", prods, "--diff-against", str(fa_path), "--diff-sheet",
               "--diff-labels", f"WOOF-NAM {ps.name},NOAA NAM {ps.name} (archive)", "--width", "1500",
               "--height", "1000", "--footer-title", f"{ours.name} vs {ref.name}", str(fb_path)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout[-3000:], p.stderr[-2000:])
        return p.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
