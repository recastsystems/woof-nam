"""score_asos.py CYCLE ASOS_CSV RAW_NAM OUT_CSV NAME=RUNDIR [NAME=RUNDIR ...]
Every CONUS ASOS routine report within 30 min of each hour 1..24: 2 m T, 2 m Td (from Q2/PSFC as td2-fix/region.py),
10 m wind speed; archived NAM 12 km (awphys) and each run's history on the SAME station-hours (a station-hour enters
only when every source has it). Rows: lead, region (conus | wedge 33-38N 84-75W | urban), source, n, bias, MAE."""
import csv, math, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, netCDF4
sys.path.insert(0, str(Path(__file__).parent))
import stations as S

WANTW = {"U10": ((0, 2, 2), "103", "10"), "V10": ((0, 2, 3), "103", "10")}


def td_f(q, p):
    e = p * q / (0.622 + 0.378 * q) / 100.0
    t = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    return t * 1.8 + 32


cyc = datetime.strptime(sys.argv[1], "%Y%m%d%H")
raw = Path(sys.argv[3]); out = sys.argv[4]
runs = [a.split("=", 1) for a in sys.argv[5:]]
obs = {}
for r in csv.DictReader(open(sys.argv[2])):
    try:
        t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
        rec = (t, float(r["tmpf"]) if r["tmpf"] not in ("M", "") else math.nan,
               float(r["dwpf"]) if r["dwpf"] not in ("M", "") else math.nan,
               float(r["sknt"]) if r["sknt"] not in ("M", "") else math.nan, float(r["lat"]), float(r["lon"]))
    except ValueError:
        continue
    obs.setdefault(r["station"], []).append(rec)
stn = sorted(obs)
lat = np.array([obs[s][0][4] for s in stn]); lon = np.array([obs[s][0][5] for s in stn])
idx = {}
rows = []
for L in range(1, 25):
    v = cyc + timedelta(hours=L)
    pick = []
    for k, s in enumerate(stn):
        b = min(obs[s], key=lambda x: abs((x[0] - v).total_seconds()))
        if abs((b[0] - v).total_seconds()) <= 1800:
            pick.append((k, b))
    if not pick:
        continue
    K = np.array([p[0] for p in pick])
    O = {"T": np.array([p[1][1] for p in pick]), "Td": np.array([p[1][2] for p in pick]),
         "wspd": np.array([p[1][3] for p in pick])}
    src = {}
    a = sorted(raw.glob(f"awphys{L:02d}*.grib2"))
    if a:
        g = S.grib_vals(a[0], {"T2": S.WANT["T2"], "TD2": S.WANT["TD2"], **WANTW})
        def at(name):
            arr, row = g[name]
            vals = []
            for k in K:
                try:
                    vals.append(S.at(arr, row, lat[k], lon[k]))
                except IndexError:
                    vals.append(math.nan)
            return np.array(vals)
        src["NAM-archive"] = {"T": S.k2f(at("T2")), "Td": S.k2f(at("TD2")),
                              "wspd": np.hypot(at("U10"), at("V10")) * 1.943844}
    urban = None
    for name, d in runs:
        f = Path(d) / f"wrfout_d01_{v:%Y-%m-%d_%H}:00:00"
        if not f.exists():
            continue
        nc = netCDF4.Dataset(f)
        gv = lambda k: np.asarray(nc[k][0] if nc[k].ndim == 3 else nc[k][:])
        if name not in idx:
            XL, XG = gv("XLAT"), gv("XLONG")
            near = np.empty(len(stn), dtype=np.int64)
            fl, fg = XL.ravel(), XG.ravel()
            for c0 in range(0, len(stn), 64):
                la, lo = lat[c0:c0 + 64, None], lon[c0:c0 + 64, None]
                d2 = (fl[None] - la) ** 2 + ((fg[None] - lo) * np.cos(np.radians(la))) ** 2
                near[c0:c0 + 64] = d2.argmin(1)
            idx[name] = near
        n = idx[name][K]
        T2, Q2, PS = gv("T2").ravel()[n], gv("Q2").ravel()[n], gv("PSFC").ravel()[n]
        U, V = gv("U10").ravel()[n], gv("V10").ravel()[n]
        if urban is None and "IVGTYP" in nc.variables:
            urban = gv("IVGTYP").ravel()[n] == 13
        src[name] = {"T": S.k2f(T2), "Td": td_f(Q2, PS), "wspd": np.hypot(U, V) * 1.943844}
        nc.close()
    wedge = (lat[K] >= 33) & (lat[K] <= 38) & (lon[K] >= -84) & (lon[K] <= -75)
    regions = {"conus": np.ones(len(K), bool), "wedge": wedge}
    if urban is not None:
        regions["urban"] = urban
    for var in ("T", "Td", "wspd"):
        ok = np.isfinite(O[var])
        for s in src.values():
            ok &= np.isfinite(s[var])
        for rn, m in regions.items():
            mm = ok & m
            for sn, s in src.items():
                e = s[var][mm] - O[var][mm]
                rows.append([L, f"{v:%Y-%m-%dT%HZ}", var, rn, sn, int(mm.sum()),
                             round(float(e.mean()), 3) if mm.any() else "", round(float(np.abs(e).mean()), 3) if mm.any() else ""])
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lead", "valid", "var", "region", "source", "n", "bias", "mae"]); w.writerows(rows)
print(out, len(rows), "rows")
