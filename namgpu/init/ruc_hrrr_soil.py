"""ruc-hrrr-soil: the HRRR 9-level RUC soil analysis (TSOIL, SOILW at 0, 1, 4, 10, 30, 60, 100, 160, 300 cm) for a
WOOF-NAM RUC start over CONUS.

    python -m namgpu.init.ruc_hrrr_soil --cycle 2026012312 --out cases/2026-01-23_12z/raw/hrrr/ruc_soil.npz

Source: NOAA HRRR wrfprsf00 on AWS open data (noaa-hrrr-bdp-pds; US government data), the 18 soil records and LAND,
fetched by byte range from the .idx (about 28 MB), decoded by the Rust GRIB2 reader of the installed gpuwm
(grib2_dump, GPUWM_BRIDGES). Written on the HRRR Lambert grid with its projection constants; the RUC driver
(namgpu.physics.ruc.phys_ruc_driver) samples it onto the NMMB H points on the GPU at its first call.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
LEVELS_M = ("0-0", "0.01-0.01", "0.04-0.04", "0.1-0.1", "0.3-0.3", "0.6-0.6", "1-1", "1.6-1.6", "3-3")
WANT = [("TSOIL", f"{lv} m below ground") for lv in LEVELS_M] + \
       [("SOILW", f"{lv} m below ground") for lv in LEVELS_M] + [("LAND", "surface")]


def _get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise


def fetch(cycle: str, out: Path, bridges: Path | None = None) -> Path:
    day, hh = cycle[:8], cycle[8:10]
    url = f"{BUCKET}/hrrr.{day}/conus/hrrr.t{hh}z.wrfprsf00.grib2"
    idx = _get(url + ".idx").decode().splitlines()
    rows = [ln.split(":") for ln in idx]
    starts = [int(r[1]) for r in rows]
    picks = []
    for var, lev in WANT:
        hit = [n for n, r in enumerate(rows) if r[3] == var and r[4] == lev]
        if not hit:
            raise SystemExit(f"ruc-hrrr-soil: {var} {lev} not in {url}.idx")
        n = hit[0]
        picks.append((starts[n], starts[n + 1] - 1 if n + 1 < len(starts) else ""))
    with ThreadPoolExecutor(len(picks)) as ex:            # I/O: one thread per record
        parts = list(ex.map(lambda p: _get(url, p), picks))
    if bridges is None:                                   # $GPUWM_BRIDGES, else the installed gpuwm wheel own folder
        from namgpu.bridges import bridges_dir
        bridges = bridges_dir()
    bridges = Path(bridges)
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "soil.grib2"
        g.write_bytes(b"".join(parts))
        subprocess.run([str(bridges / "grib2_dump"), str(g), td + "/d", *[str(i) for i in range(len(WANT))]],
                       check=True, capture_output=True)
        meta = [ln.split("\t") for ln in (Path(td) / "d" / "metadata.tsv").read_text().splitlines()]
        inv = subprocess.run([str(bridges / "grib2_inventory"), str(g)], check=True, capture_output=True,
                             text=True).stdout.splitlines()
        head = [ln for ln in inv if ln.startswith("index\t")][0].split("\t")
        row0 = [ln for ln in inv if ln.startswith("0\t")][0].split("\t")
        geo = dict(zip(head, row0))
        nx, ny = int(geo["nx"]), int(geo["ny"])
        fields = [np.fromfile(Path(td) / "d" / f"field-{i:04d}.f64le", dtype="<f8").astype(np.float32).reshape(ny, nx)
                  for i in range(len(WANT))]
    tsoil = np.stack(fields[0:9])
    soilw = np.stack(fields[9:18])
    land = fields[18]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tsoil=tsoil, soilw=soilw, land=land,
             lat1=np.float64(geo["lat1"]), lon1=np.float64(geo["lon1"]), dx=np.float64(geo["dx"]),
             dy=np.float64(geo["dy"]), latin1=np.float64(geo["latin1"]), latin2=np.float64(geo["latin2"]),
             lov=np.float64(geo["lov"]), radius=np.float64(6371229.0), source=np.array(url), cycle=np.array(cycle),
             levels_m=np.array([0, .01, .04, .1, .3, .6, 1., 1.6, 3.], np.float32))
    print(f"ruc-hrrr-soil: {url} -> {out} ({nx}x{ny}; land {int((land > 0.5).sum())}; "
          f"SOILW top land mean {float(soilw[0][land > 0.5].mean()):.3f})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ruc-hrrr-soil")
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fetch(a.cycle, Path(a.out))


if __name__ == "__main__":
    main()
