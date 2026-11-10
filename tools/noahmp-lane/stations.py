"""Station 2 m T / Td / q from (a) archived NAM awphys, (b) a WOOF-NAM run's awphys products, (c) its history.
usage: stations.py CASE_RAW_NAM RUN_DIR LEAD [LEAD...]   (prints one table row per station per source)"""
import sys, glob, math, tempfile
from pathlib import Path
import numpy as np
from namgpu.init.init_grib2_decode import inventory, _dump
from namgpu.io.io_referee_panels import _place

ST = {"KGSO": (36.0975, -79.9436), "KRDU": (35.8922, -78.7819), "KINT": (36.1337, -80.2220),
      "KCLT": (35.2225, -80.9543), "KAVL": (35.4319, -82.5375)}
R = 6371229.0

def lcc_ij(row, lat, lon):
    lat1, lon1 = float(row["lat1"]), float(row["lon1"])
    dx = float(row["dx"]); dy = float(row["dy"])
    if dx > 1000.0 * 1000: dx /= 1000.0; dy /= 1000.0      # mm vs m guard
    if dx > 1.0e5: dx /= 1000.0; dy /= 1000.0
    l1 = math.radians(float(row["latin1"])); l2 = math.radians(float(row["latin2"]))
    lov = math.radians(float(row["lov"]))
    n = math.sin(l1) if abs(l1 - l2) < 1e-9 else (math.log(math.cos(l1) / math.cos(l2)) /
        math.log(math.tan(math.pi / 4 + l2 / 2) / math.tan(math.pi / 4 + l1 / 2)))
    F = math.cos(l1) * math.tan(math.pi / 4 + l1 / 2) ** n / n
    def xy(la, lo):
        rho = R * F / math.tan(math.pi / 4 + math.radians(la) / 2) ** n
        th = n * (((math.radians(lo) - lov + math.pi) % (2 * math.pi)) - math.pi)
        return rho * math.sin(th), -rho * math.cos(th)
    x0, y0 = xy(lat1, lon1)
    x, y = xy(lat, lon)
    return (x - x0) / dx, (y - y0) / dy       # i east, j north (from the SW corner)

def grib_vals(path, want):
    inv = inventory(path)
    pick = {}
    for r in inv:
        key = (r["dcp"], r["level_type"], str(r["level_value"]).split(".")[0])
        for name, k in want.items():
            if key == k and name not in pick:
                pick[name] = r
    with tempfile.TemporaryDirectory() as td:
        dumped, dd = _dump(path, [r["index"] for r in pick.values()], td)
        out = {}
        for name, r in pick.items():
            host = np.fromfile(dd / dumped[r["index"]]["filename"], dtype="<f8")
            out[name] = (_place(host, r), r)
    return out

WANT = {"T2": ((0, 0, 0), "103", "2"), "TD2": ((0, 0, 6), "103", "2"), "Q2": ((0, 1, 0), "103", "2"),
        "RH2": ((0, 1, 1), "103", "2"), "PS": ((0, 3, 0), "1", "0")}

def k2f(k): return (k - 273.15) * 1.8 + 32

def at(arr, row, la, lo):
    i, j = lcc_ij(row, la, lo)
    return float(arr[int(round(j)), int(round(i))])

def main():
    raw, run = Path(sys.argv[1]), Path(sys.argv[2])
    for lead in [int(x) for x in sys.argv[3:]]:
        srcs = {}
        a = sorted(raw.glob(f"awphys{lead:02d}*.grib2"))
        if a: srcs["NAM"] = grib_vals(a[0], WANT)
        o = sorted(run.glob(f"products/**/*awphys{lead:02d}*.grib2", ), key=str) or sorted(run.glob(f"**/*awphys{lead:02d}.tm00.grib2"))
        if o: srcs["WOOF"] = grib_vals(o[0], WANT)
        for s, (la, lo) in ST.items():
            line = [f"f{lead:02d}", s]
            for name, d in srcs.items():
                v = {k: at(arr, r, la, lo) for k, (arr, r) in d.items()}
                line.append(f"{name}: T {k2f(v['T2']):6.1f} Td {k2f(v['TD2']) if 'TD2' in v else float('nan'):6.1f} "
                            f"q {v.get('Q2', float('nan'))*1e3:5.2f} g/kg RH {v.get('RH2', float('nan')):5.1f}")
            print(" | ".join(line))

if __name__ == "__main__":
    main()
