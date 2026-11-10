"""Regional ASOS T2 / Td2 scores for one run (and the archived NAM) at 3-hourly leads, reusing namscore's readers.

  python region_score.py --cycle 2026012312 --case-dir cases/2026-01-23_12z --run NAME=HISTORY --out region.json

Regions: wedge = every ASOS 33-38 N, 84-75 W (the TD2-FIX regional check); plains 33-43 N, 103-94 W; conus = all.
Urban split in the wedge (model IVGTYP 13 at the station's cell) when the history carries IVGTYP / LU_INDEX.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import namscore as ns

REG = {"wedge": (33.0, 38.0, -84.0, -75.0), "plains": (33.0, 43.0, -103.0, -94.0), "conus": (20.0, 55.0, -130.0, -60.0)}
F = lambda k: (k - 273.15) * 1.8 + 32


def stats(mv, ob):
    return ns.point_stats(mv, ob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--leads", default="3,6,9,12,15,18,21,24")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nam", action="store_true", help="also score the archived NAM")
    a = ap.parse_args()
    name, run = a.run.split("=", 1)
    run = Path(run)
    cyc = datetime.strptime(a.cycle, "%Y%m%d%H")
    obs = a.case_dir / "obs"
    leads = [int(x) for x in a.leads.split(",")]
    out = {"cycle": a.cycle, "run": name, "leads": {}, "nam_archive": {}}
    lu = None
    for L in leads:
        v = cyc + timedelta(hours=L)
        st = ns.read_asos(obs, v)
        S = np.array([st[s] for s in sorted(st)])
        M = ns.model_fields(run, cyc, L)
        if lu is None:
            try:
                h = ns.history_frame(run, v, {"LU_INDEX"})
                lu = h.get("LU_INDEX")
            except Exception:
                lu = None
        idx, ok = ns.nearest_index(M["lat"], M["lon"], S[:, 0], S[:, 1])
        mt = np.where(ok, np.ravel(M["T2"])[idx], np.nan)
        md = np.where(ok, np.ravel(M["TD2"])[idx], np.nan)
        ot, od = F(S[:, 2]), F(S[:, 3])
        urb = (np.ravel(lu)[idx] == 13) if lu is not None else np.zeros(len(S), bool)
        r = {}
        for reg, (la0, la1, lo0, lo1) in REG.items():
            m = (S[:, 0] >= la0) & (S[:, 0] <= la1) & (S[:, 1] >= lo0) & (S[:, 1] <= lo1)
            r[reg] = {"T2": stats(np.where(m, mt, np.nan), ot), "TD2": stats(np.where(m, md, np.nan), od)}
            if reg == "wedge":
                r["wedge_urban"] = {"TD2": stats(np.where(m & urb, md, np.nan), od)}
                r["wedge_nonurban"] = {"TD2": stats(np.where(m & ~urb, md, np.nan), od)}
        out["leads"][str(L)] = r
        if a.nam:
            try:
                Fn, lat, lon = ns.nam_archive_fields(a.case_dir, cyc, L)
                i2, ok2 = ns.nearest_index(lat, lon, S[:, 0], S[:, 1])
                nt = np.where(ok2, F(np.ravel(Fn["T2"])[i2]), np.nan)
                nd = np.where(ok2, F(np.ravel(Fn["TD2"])[i2]), np.nan)
                rn = {}
                for reg, (la0, la1, lo0, lo1) in REG.items():
                    m = (S[:, 0] >= la0) & (S[:, 0] <= la1) & (S[:, 1] >= lo0) & (S[:, 1] <= lo1)
                    rn[reg] = {"T2": stats(np.where(m, nt, np.nan), ot), "TD2": stats(np.where(m, nd, np.nan), od)}
                out["nam_archive"][str(L)] = rn
            except Exception as e:
                out["nam_archive"][str(L)] = {"skipped": str(e)}
        print(name, L, json.dumps({k: r[k]["TD2"] for k in ("wedge", "plains", "conus")}), flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
