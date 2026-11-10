"""Fetch the soil-moisture records the soil-start rows need, by .idx byte ranges, many dates at once (threads).

  python soil_fetch.py clim OUTDIR MONTH      # MONTH 1-12: monthly table window, +/-30 d of the 15th, 2021-2025
  python soil_fetch.py clim OUTDIR            # NAM bgrd3d00 + GFS f000 SOILW (4 layers) + LAND, every date of the window
  python soil_fetch.py hrrr OUTFILE YYYYMMDDHH  # HRRR wrfprs f00 SOILW 9 levels + wrfsfc f00 LAND

Window (case 2026-01-23 12Z): 12Z every 3 days, Dec 24 - Feb 22, winters 2021-22 .. 2025-26, minus 2026-01-13 .. 2026-02-02
(the case and 10 days either side never enter its own climatology). I/O only: 32 download threads.
"""
import datetime as dt, sys, urllib.request, concurrent.futures as cf, os, time
NAM = "https://noaa-nam-pds.s3.amazonaws.com/nam.{ymd}/nam.t{hh}z.bgrd3d00.tm00.grib2"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000"
HPRS = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfprsf00.grib2"
HSFC = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf00.grib2"

def get(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)

def ranges(url, want):
    lines = get(url + ".idx").decode().strip().splitlines()
    rows = [l.split(":") for l in lines]
    out = []
    for i, r in enumerate(rows):
        key = ":".join(r[3:5])
        if any(key.startswith(w) for w in want):
            end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
            out.append((int(r[1]), end, key))
    return out

def subset(url, want, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, 0
    rs = ranges(url, want)
    if not rs:
        raise RuntimeError(f"no {want} in {url}")
    parts = [get(url, (a, b if b is not None else "")) for a, b, _ in rs]
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for p in parts:
            f.write(p)
    os.replace(tmp, path)
    return path, len(rs)

def dates():
    out = []
    for y in range(2021, 2026):
        d = dt.date(y, 12, 24)
        while d <= dt.date(y + 1, 2, 22):
            if not (dt.date(2026, 1, 13) <= d <= dt.date(2026, 2, 2)):
                out.append(d)
            d += dt.timedelta(days=3)
    return out

def month_dates(month, years=range(2021, 2026), half=30, every=3):
    """12Z every `every` days, +/-`half` days around the 15th of `month`, one window per year (the monthly CDF tables;
    2026-11-09). General tables: no case is held out."""
    out = []
    for y in years:
        c = dt.date(y, month, 15)
        d = c - dt.timedelta(days=half)
        while d <= c + dt.timedelta(days=half):
            out.append(d)
            d += dt.timedelta(days=every)
    return out

def clim(outdir, month=None):
    os.makedirs(f"{outdir}/nam", exist_ok=True); os.makedirs(f"{outdir}/gfs", exist_ok=True)
    jobs = []
    ds = dates() if month is None else month_dates(int(month))
    for d in ds:
        ymd = d.strftime("%Y%m%d")
        jobs.append((NAM.format(ymd=ymd, hh="12"), ["SOILW:"], f"{outdir}/nam/{ymd}12.grib2"))
        jobs.append((GFS.format(ymd=ymd, hh="12"), ["SOILW:", "LAND:surface"], f"{outdir}/gfs/{ymd}12.grib2"))
    ok = bad = 0
    with cf.ThreadPoolExecutor(32) as ex:
        futs = {ex.submit(subset, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                f.result(); ok += 1
            except Exception as e:
                bad += 1; print("MISS", futs[f][0], e, flush=True)
    print(f"clim: {ok} files ok, {bad} missing, {len(ds)} dates", flush=True)

def hrrr(outfile, cyc):
    ymd, hh = cyc[:8], cyc[8:10]
    a, _ = subset(HPRS.format(ymd=ymd, hh=hh), ["SOILW:"], outfile + ".prs")
    b, _ = subset(HSFC.format(ymd=ymd, hh=hh), ["LAND:surface"], outfile + ".sfc")
    with open(outfile, "wb") as f:
        f.write(open(a, "rb").read()); f.write(open(b, "rb").read())
    os.remove(a); os.remove(b)
    print("hrrr:", outfile, os.path.getsize(outfile))

if __name__ == "__main__":
    if sys.argv[1] == "clim":
        clim(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        hrrr(sys.argv[2], sys.argv[3])
