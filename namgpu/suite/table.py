"""The WOOF-NAM suite as data: parm/nam_suite.json (the parent and the five nests NCEP ran beside it), the fire-weather
point file parm/nam_firewx_loc (NCEP's own mechanism), and the plan a cycle runs from them.

Nothing here touches a card: it is read by `woof-nam suite` (namgpu/suite/cli.py) and by its tests.

The fire-weather point (NCEP: util/sorc/setfirewx.fd, util/ush/setup_firewx.sh, exnam_coldstart_prep_nest.sh):
  nam_firewx_loc holds one line per cycle, "HHz lat lon" (Fortran i2.2,"z",1x,f4.1,1x,f6.1). The Senior Duty
  Meteorologist set it; with no point (0 0) the default was Washington DC, 38.9 N 77.0 W. lat >= 52.0 puts the nest
  in the Alaska nest, else the CONUS nest, and the point must leave the 333 x 333 nest at least 5 rows inside that
  nest (nest_domain_check_location_conus/_ak). Ours keeps the mechanism: the file in parm/ is seeded with NCEP's last
  production points (unchanged 2026-10-20 to 2026-11-03), a point that fails the check falls back to the DC default
  and the plan says why.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

PARM = Path(__file__).resolve().parent.parent / "parm"
TABLE = PARM / "nam_suite.json"
FIREWX_LOC = PARM / "nam_firewx_loc"
FIREWX_DEFAULT = (38.9, -77.0)            # setfirewx.fd: xlat = 38.9, ylat = -77.0 when no point is entered
FIREWX_ALASKA_LAT = 52.0                  # setfirewx.fd: alat .ge. 52.0 -> the Alaska nest
FIREWX_EDGE_ROWS = 5                      # nest_domain_check_location_*: 5 rows inside the parent nest
LOC_LINE = re.compile(r"^\s*(\d{2})z\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s*$")
SUITE_ORDER = ("12km", "conusnest", "alaskanest", "hawaiinest", "priconest", "firewxnest")
NEST_NAMES = SUITE_ORDER[1:]


def load(path: Path | None = None) -> dict:
    t = json.loads(Path(path or TABLE).read_text(encoding="utf-8"))
    if t.get("schema") != "woof-nam-suite.v1":
        raise SystemExit(f"{path or TABLE}: schema {t.get('schema')!r} is not woof-nam-suite.v1")
    return t


def domains(t: dict) -> list[str]:
    """Every domain of the suite, parent first, in NCEP's order (parent, conus, alaska, hawaii, prico, firewx)."""
    names = [t["parent"]["domain"], *t["nests"]]
    return sorted(names, key=lambda n: SUITE_ORDER.index(n) if n in SUITE_ORDER else len(SUITE_ORDER))


# ---------------------------------------------------------------------------------------------- fire weather

def read_firewx_loc(path: Path | None = None) -> dict[int, tuple[float, float]]:
    """{cycle hour: (lat, lon)} from an nam_firewx_loc file; a 0 0 line (no point entered) is left out."""
    out = {}
    p = Path(path or FIREWX_LOC)
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        m = LOC_LINE.match(line)
        if not m:
            continue
        lat, lon = float(m[2]), float(m[3])
        if lat == 0.0 and lon == 0.0:
            continue
        out[int(m[1])] = (lat, lon)
    return out


def write_center_latlon(lat: float, lon: float) -> str:
    """The text of NOAA's nam.tHHz.firewxnest_center_latlon side file ("42.0 -122.3")."""
    return f"{lat:.1f} {lon:.1f}\n"


def _rotated(lat, lon, tph0d, tlm0d):
    from namgpu.nest.domain import geographic_to_rotated
    return geographic_to_rotated(lat, lon, tph0d, tlm0d)


A_EARTH = 6376000.0          # grid.A_EARTH (NMMB module_CONSTANTS a), as namgpu.nest.domain.A_EARTH


def nest_grid(row: dict, center=None) -> dict:
    """A suite nest's grid on its own rotated pole at the centre: NCEP's point counts and spacing (dlmd 0.0105 x res,
    dphd 0.009 x res deg, as namgpu.nest.domain.nest_grid lays one out). The door's --ncep-nest grid sits on the parent's
    pole instead; this one is the plan's (corners for the fire-weather check, sizes for a door without --ncep-nest)."""
    lat, lon = center or row["center"]
    res, im, jm = float(row["res_km"]), int(row["im"]), int(row["jm"])
    dlmd, dphd = 0.0105 * res, 0.009 * res
    dx, dy = A_EARTH * math.radians(dlmd) / 1e3, A_EARTH * math.radians(dphd) / 1e3
    lon = ((lon + 180.0) % 360.0) - 180.0
    return {"res_km": res, "im": im, "jm": jm, "lm": 60, "tph0d": round(lat, 4), "tlm0d": round(lon, 4),
            "dlmd": dlmd, "dphd": dphd, "wbd": round(-(im - 1) / 2 * dlmd, 6), "sbd": round(-(jm - 1) / 2 * dphd, 6),
            "dx_km": round(dx, 4), "dy_km": round(dy, 4), "size_km": [(im - 1) * dx, (jm - 1) * dy]}


def _corners(g: dict) -> list[tuple[float, float]]:
    """Geographic corners (lat, lon) of a door grid (its own pole at its centre)."""
    from namgpu.nest.window import rotated_to_geographic
    return [rotated_to_geographic(x, y, g["tph0d"], g["tlm0d"]) for x in (g["wbd"], -g["wbd"]) for y in (g["sbd"], -g["sbd"])]


def firewx_check(t: dict, lat: float, lon: float) -> tuple[bool, str, str]:
    """(ok, region, why): NCEP's rule. region conus or alaska by lat >= 52; ok when the 333 x 333 fire-weather nest at
    (lat, lon) sits FIREWX_EDGE_ROWS rows inside that 3 km nest on every side."""
    region = "alaska" if lat >= FIREWX_ALASKA_LAT else "conus"
    host = t["nests"][f"{region}nest"]
    hg = nest_grid(host)
    fg = nest_grid(t["nests"]["firewxnest"], (lat, lon))
    lim_x = -hg["wbd"] - FIREWX_EDGE_ROWS * hg["dlmd"]
    lim_y = -hg["sbd"] - FIREWX_EDGE_ROWS * hg["dphd"]
    worst = []
    for clat, clon in _corners(fg):
        x, y = _rotated(clat, clon, hg["tph0d"], hg["tlm0d"])
        worst.append(min(lim_x - abs(x), lim_y - abs(y)))
    m = min(worst)
    if m < 0:
        return False, region, (f"the fire-weather nest at {lat}, {lon} reaches {abs(m):.3f} rotated degrees past the "
                               f"{region} nest less {FIREWX_EDGE_ROWS} rows")
    return True, region, f"{m / hg['dlmd']:.0f} {region}-nest columns to spare"


def firewx_point(t: dict, cycle: datetime, loc: Path | None = None) -> dict:
    """The cycle's fire-weather centre: the nam_firewx_loc line for its hour, checked; else (no line, or a point
    that fails NCEP's check) the DC default."""
    pts = read_firewx_loc(loc)
    lat, lon = pts.get(cycle.hour, FIREWX_DEFAULT)
    how = "nam_firewx_loc" if cycle.hour in pts else "default (no point for this cycle in nam_firewx_loc)"
    ok, region, why = firewx_check(t, lat, lon)
    if not ok:
        bad = (lat, lon)
        lat, lon = FIREWX_DEFAULT
        ok2, region, why2 = firewx_check(t, lat, lon)
        how = f"default: the nam_firewx_loc point {bad} failed the check ({why})"
        why = why2
    return {"lat": lat, "lon": lon, "region": region, "how": how, "check": why}


# ---------------------------------------------------------------------------------------------- the plan

def layout(t: dict, cards: list[str]) -> dict[str, list[str]]:
    """domain -> the card indices it runs on, from the table's layout for this many cards (else the largest layout
    that fits, on the first cards of the list)."""
    rows = {int(k): v for k, v in t["layouts"].items() if not k.startswith("_")}
    n = max((k for k in rows if k <= len(cards)), default=None)
    if n is None:
        raise SystemExit(f"woof-nam suite: no layout for {len(cards)} cards (layouts {sorted(rows)})")
    return {d: [cards[int(i)] for i in idx] for d, idx in rows[n].items()}


def file_name(t: dict, domain: str, cycle: datetime, fhr: int) -> str:
    if domain == t["parent"]["domain"]:
        return t["parent"]["hour_file"].format(prefix=t["prefix"], HH=f"{cycle:%H}", FF=f"{fhr:02d}")
    return t["nest_hour_file"].format(prefix=t["prefix"], HH=f"{cycle:%H}", nest=domain, FF=f"{fhr:02d}")


def hour_of(t: dict, domain: str, name: str) -> int | None:
    """The forecast hour of one of the domain's product file names (or its .idx), else None."""
    base = name[:-4] if name.endswith(".idx") else name
    rx = t["parent"]["hour_regex"] if domain == t["parent"]["domain"] else t["nest_hour_regex"]
    m = re.match(rx, base)
    if not m or (domain != t["parent"]["domain"] and m["nest"] != domain):
        return None
    return int(m["FF"])


def parent_hours_for_nest(hours: float) -> int:
    """The parent hours a one-way nest of `hours` reads before it can start: the door's parent fetch
    (nest/cli.py fetch_parent) checks the parent's history every 3 h up to max(6, hours)."""
    return max(6, 3 * -(-int(-(-hours // 1)) // 3))


def plan(t: dict, cycle: datetime, cards: list[str], only: list[str] | None = None, hours_cap: float | None = None,
         loc: Path | None = None) -> dict:
    """What one cycle runs: per domain its hours, cards, grid, door arguments and product file names."""
    if cycle.hour not in t["cycles"]:
        raise SystemExit(f"woof-nam suite: cycle {cycle:%HZ} is not a suite cycle ({t['cycles']})")
    lay = layout(t, cards)
    want = only or domains(t)
    unknown = [d for d in want if d not in domains(t)]
    if unknown:
        raise SystemExit(f"woof-nam suite: unknown domains {unknown} (the table has {domains(t)})")
    out = {"cycle": cycle.strftime("%Y-%m-%dT%HZ"), "cards": cards, "domains": {}}
    p = t["parent"]
    for d in want:
        if d == p["domain"]:
            h = p["hours"] if hours_cap is None else min(p["hours"], hours_cap)
            out["domains"][d] = {"kind": "parent", "hours": h, "cards": lay[d], "row": p["row"],
                                 "configfile": p["configfile"], "source": p["source"]}
            continue
        row = t["nests"][d]
        center = tuple(row["center"]) if "center" in row else None
        extra = {}
        if row.get("center_from") == "nam_firewx_loc":
            fw = firewx_point(t, cycle, loc)
            center = (fw["lat"], fw["lon"])
            extra["firewx"] = fw
        g = nest_grid(row, center)
        product_grid = row.get("product_grid")
        if extra.get("firewx"):
            # the cycle's fire-weather output grid (nam_firewx_gridspecs.fd, as the door's NCEP row computes it), so the
            # cycle.json a converter reads names this cycle's grid, not a description of how it is made
            try:
                from namgpu.nest import ncep_nests
                fwg = ncep_nests.nest_grid("firewxnest", cycle_hour=cycle.hour, firewx_point=center).get("firewx") or {}
                if fwg.get("output_grid"):
                    product_grid = {"wgrib2": fwg.get("wgrib2"), "template": "lambert", **fwg["output_grid"]}
            except (ImportError, KeyError, TypeError, SystemExit):
                pass
        h = row["hours"] if hours_cap is None else min(row["hours"], hours_cap)
        out["domains"][d] = {"kind": "nest", "hours": h, "cards": lay[d], "center": list(center), "res_km": row["res_km"],
                             "grid": {k: g[k] for k in ("im", "jm", "lm", "tph0d", "tlm0d", "dlmd", "dphd", "wbd", "sbd",
                                                        "dx_km", "dy_km")},
                             "size_km": g["size_km"], "product_grid": product_grid,
                             "packing": row.get("packing"), **extra}
    # the parent must outlast every nest it feeds
    if p["domain"] in out["domains"]:
        ph = out["domains"][p["domain"]]["hours"]
        longest = max((v["hours"] for v in out["domains"].values() if v["kind"] == "nest"), default=0)
        if hours_cap is not None and longest > 0:
            # a capped check still feeds its nests: they read the parent's history to max(6, hours), so the parent
            # runs that far when the cap is shorter (with --hours-cap 1 every nest waited for a parent hour 6 that
            # the 1 h parent never wrote, and the cycle failed: release check 2026-11-09)
            need = min(p["hours"], parent_hours_for_nest(longest))
            if ph < need:
                out["domains"][p["domain"]]["hours"] = ph = need
                out["domains"][p["domain"]]["hours_why"] = (f"capped at {hours_cap:g} h; the nests read the parent's "
                                                           f"history to hour {need}")
        if longest > ph:
            raise SystemExit(f"woof-nam suite: a nest runs {longest} h but the parent only {ph} h")
    return out


def ncep_rows() -> set[str]:
    """The nests the installed door runs as NCEP's own (`woof-nam nest --ncep-nest NAME`: ncep_nests.NCEP_NESTS)."""
    try:
        from namgpu.nest import ncep_nests
        return set(ncep_nests.NCEP_NESTS)
    except ImportError:
        return set()


def door_argv(d: str, row: dict, cycle: datetime, parent_run: Path, out: Path, options: str = "",
              rows: set[str] | None = None) -> list[str]:
    """`woof-nam nest` for one suite nest, one-way from the parent's history. options: the installed door's --help
    text; rows: the nests it has an NCEP row for. A door with --ncep-nest and a row for this nest runs NCEP's own nest
    (its grid on the parent's pole, its keys); otherwise NCEP's point counts and spacing at NCEP's centre on the nest's
    own pole (a door built before that nest's row: `--ncep-nest conusnest` refused at its first line, suite proof
    2026-11-06)."""
    if "--ncep-nest" in options and (rows is None or d in rows):
        argv = ["nest", "--ncep-nest", d]
        if row.get("firewx") and "--firewx-center" in options:
            argv += ["--firewx-center", f"{row['firewx']['lat']},{row['firewx']['lon']}"]
    else:
        lat, lon = row["center"]
        sx, sy = row["size_km"]
        argv = ["nest", "--center", f"{lat},{lon}", "--size-km", f"{sx:.6f}", "--size-y-km", f"{sy:.6f}",
                "--res-km", f"{row['res_km']:g}"]
    argv += ["--source", "parent", "--parent-run", str(parent_run),
             "--start", f"{cycle:%Y%m%d%H}", "--hours", f"{row['hours']:g}", "--out", str(out), "--no-maps",
             "--no-sheets", "--no-warm", "--wait-min", "120"]
    if len(row["cards"]) > 1:
        argv += ["--cards", str(len(row["cards"])), "--card-list", ",".join(row["cards"])]
    return argv
