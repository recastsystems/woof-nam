"""NCEP's fire-weather nest placement, as NAM v4.2.11 placed it every cycle: the point, the parent nest, the grid and the
hiresf output grid. The row itself (333 x 333 x 60, ratio 2, 36 h, dt 2+1/12 s, lnsh 3, no TDFI) is
ncep_nests.NCEP_NESTS["firewxnest"]; this module turns a cycle into that row's grid.

Sources (NCEP NAM v4.2.11, commit 83df4e13):
  util/sorc/setfirewx.fd   the Senior Duty Meteorologist's per-cycle point ("HHz lat lon" in
                           com/nam/<ver>/input/nam_firewx_loc), "same firewx points as yesterday", the Washington DC
                           default (38.9 N, -77.0) when no point is given, lat >= 52.0 -> the Alaska nest (else CONUS),
                           and nest_domain_check_location_{conus,ak}: the 333 x 333 nest must lie inside the outer 5
                           rows of its parent nest. It hard-codes the parent nests (CONUS 1827 x 1467 at 295/95, Alaska
                           1189 x 1249 at 139/398, ratio 4 in the 954 x 835 NAM parent); PARENT_NESTS copies them.
  NPS geogrid llxy_module.F90 compute_nest_locations (PROJ_ROTLLB): i_parent_start = nint(1 + (xcenter - (im-1)/2
                           dlmd_nest - wbd_parent) / dlmd_parent), likewise j (written to nest_start_03, then
                           nam.tHHz.firewx_ijstart.txt, then exnam_fcst.sh's I_START_FIREWX/J_START_FIREWX).
  sorc/nam_firewx_gridspecs.fd  the hiresf output grid from the nest's four corners (ush/nam_prdgen_nest.sh): CONUS
                           the corners' lat/lon bounding box on a fixed 1.27 km Lambert master grid (4289 x 2753, La1
                           20.192, Lo1 -121.554, LoV -95, Latin 25); Alaska their i/j range on a 1.488 km polar
                           stereographic master (3297 x 2209, La1 40.530, Lo1 -178.571, LoV -150) with 100 points
                           of padding. These are NCEP grids 90 and 92 of Office Note 388, Table B.
  Projections: Snyder, J. P., 1987: Map Projections: A Working Manual, USGS Professional Paper 1395 (Lambert
                           conformal conic and polar stereographic on a sphere; see products.sets.OutGrid), on the
                           sphere NCEP's GRIB2 grid definitions declare (shape of earth 6, WMO-No. 306 code table
                           3.2); WMO-No. 306 GRIB2 templates 3.30 and 3.20 for the first point and the axes.
  NOAA's files: s3://noaa-nam-pds nam.YYYYMMDD/nam.tHHz.firewxnest_center_latlon (NCEP's points 2026-10-20 to
                           2026-11-03) and nam.20261103 nam.tHHz.firewxnest.hiresf01.tm00.grib2 (their decoded grids).

Checked against NOAA's 2026-11-03 files: the grid this module builds for 42.0 N, -122.3 (i/j_parent_start
259/818 in the CONUS nest) covers NOAA's 184,183 bitmap points with 610 edge points of difference (the nearest-
neighbour half-cell rim), and every shift of one parent cell either way is worse. Both output grids come out exactly
as NOAA's: 00/12Z 543 x 379, La1 36.487, Lo1 244.377; 06/18Z 646 x 389, La1 39.182, Lo1 233.507 (its box starts west
of the master grid, at master column -11).
"""
from __future__ import annotations

import math
from pathlib import Path

# setfirewx.fd's hard-coded parent nests (and NAM's 12 km parent, parm/nam_configfile)
NAM12 = {"im": 954, "jm": 835, "tph0d": 54.0, "tlm0d": -106.0, "wbd": -60.039, "sbd": -45.036, "dlmd": 0.126,
         "dphd": 0.108}
PARENT_NESTS = {"conus": {"name": "conusnest", "im": 1827, "jm": 1467, "i_sw": 295, "j_sw": 95, "ratio": 4},
                "alaska": {"name": "alaskanest", "im": 1189, "jm": 1249, "i_sw": 139, "j_sw": 398, "ratio": 4}}
A_EARTH = 6376000.0          # grid.A_EARTH (NMMB module_CONSTANTS a), as domain.A_EARTH


def parent_nest(location: str) -> dict:
    """The CONUS or Alaska 3 km nest's grid (setfirewx.fd's numbers), in rotated degrees on the NAM pole."""
    p = PARENT_NESTS[location]
    r = p["ratio"]
    return {"name": p["name"], "im": p["im"], "jm": p["jm"], "tph0d": NAM12["tph0d"], "tlm0d": NAM12["tlm0d"],
            "wbd": NAM12["wbd"] + (p["i_sw"] - 1) * NAM12["dlmd"], "sbd": NAM12["sbd"] + (p["j_sw"] - 1) * NAM12["dphd"],
            "dlmd": NAM12["dlmd"] / r, "dphd": NAM12["dphd"] / r}


# The fire-weather point (util/sorc/setfirewx.fd): one line per cycle, "HHz lat lon" (i2.2,"z",1x,f4.1,1x,f6.1), in
# com/nam/<ver>/input/nam_firewx_loc. NCEP's last production points, unchanged on every day sampled 2026-10-20 to
# 2026-11-03 (AWS noaa-nam-pds nam.tHHz.firewxnest_center_latlon): 00/12Z west-central Utah, 06/18Z southern Oregon.
# The suite keeps NCEP's "same firewx points as yesterday" practice with these as the seed.
FIREWX_LOC_SEED = "00z 39.0 -112.0\n06z 42.0 -122.3\n12z 39.0 -112.0\n18z 42.0 -122.3\n"
FIREWX_DEFAULT = (38.9, -77.0)          # setfirewx.fd: no point given (0 0) -> Washington DC
FIREWX_ALASKA_LAT = 52.0                # setfirewx.fd: lat >= 52.0 -> the Alaska nest, else CONUS
FIREWX_BUFFER_ROWS = 5                  # setfirewx.fd nrows_buffer: inside the outer 5 rows of the parent nest

# nam_firewx_gridspecs.fd GRD: the fire-weather output master grids
FIREWX_MASTER = {
    "conus": {"kind": "lambert", "nx": 4289, "ny": 2753, "la1": 20.192, "lo1": -121.554, "lov": -95.0,
              "latin": 25.0, "dx": 1270.0, "pad": 0},
    "alaska": {"kind": "nps", "nx": 3297, "ny": 2209, "la1": 40.530, "lo1": -178.571, "lov": -150.0,
               "lad": 60.0, "dx": 1488.0, "pad": 100},
}
# The output grids NOAA's own files carry for NCEP's last points (decoded from nam.20261103 tHHz.firewxnest.hiresf01:
# grid_def, La1/Lo1, Nx/Ny). firewx_output_grid reproduces both exactly (at 06/18Z the box starts west of the master
# grid: the corners' SW index is about -10.36 and NOAA's box starts at master column -11, the floor). The suite writes
# these strings for these points; any other point takes the computed grid.
FIREWX_NOAA_GRIDS = {
    (39.0, -112.0): "lambert:265:25:25 244.377:543:1270 36.487:379:1270",
    (42.0, -122.3): "lambert:265:25:25 233.507:646:1270 39.182:389:1270",
}


def _round_half_away(x: float) -> int:
    """Fortran NINT."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def to_rotated(lat: float, lon: float, tph0d: float, tlm0d: float) -> tuple[float, float]:
    """(tlm, tph) degrees on the grid rotated to (tph0d, tlm0d): setfirewx.fd / NPS tll."""
    d = math.pi / 180.0
    tph0, tlm0 = tph0d * d, tlm0d * d
    x = math.cos(tph0) * math.cos(lat * d) * math.cos(lon * d - tlm0) + math.sin(tph0) * math.sin(lat * d)
    y = math.cos(lat * d) * math.sin(lon * d - tlm0)
    z = -math.sin(tph0) * math.cos(lat * d) * math.cos(lon * d - tlm0) + math.cos(tph0) * math.sin(lat * d)
    phi = math.atan(z / math.sqrt(x * x + y * y)) / d
    lam = math.atan(y / x) / d
    if x < 0.0:
        lam += 180.0
    return lam, phi


def to_geographic(tlm: float, tph: float, tph0d: float, tlm0d: float) -> tuple[float, float]:
    """(lat, lon) of rotated (tlm, tph) degrees: setfirewx.fd center_of_nest_conus."""
    d = math.pi / 180.0
    phi_t, lam_t = tph * d, tlm * d
    arg = math.sin(phi_t) * math.cos(tph0d * d) + math.cos(phi_t) * math.sin(tph0d * d) * math.cos(lam_t)
    phi_c = math.asin(max(-1.0, min(1.0, arg)))
    arg = (math.cos(phi_t) * math.cos(lam_t) / (math.cos(phi_c) * math.cos(tph0d * d))
           - math.tan(phi_c) * math.tan(tph0d * d))
    lam_c = tlm0d + (-1.0 if lam_t < 0.0 else 1.0) * math.acos(max(-1.0, min(1.0, arg))) / d
    return phi_c / d, ((lam_c + 180.0) % 360.0) - 180.0


# ------------------------------------------------------------------------------------------ fire weather

def default_loc():
    """The nam_firewx_loc the package ships (namgpu/parm/nam_firewx_loc), else None (the seed below)."""
    try:
        from namgpu import paths
        p = Path(paths.PARM) / "nam_firewx_loc"
        return p if p.is_file() else None
    except Exception:
        return None


def read_firewx_loc(path) -> dict:
    """{"06": (42.0, -122.3), ...} from a nam_firewx_loc file ("HHz lat lon" per line, setfirewx.fd's format); None
    reads the seed (NCEP's last production points)."""
    out = {}
    text = FIREWX_LOC_SEED if path is None else Path(path).read_text()
    for line in text.splitlines():
        p = line.split()
        if len(p) != 3 or not p[0].lower().endswith("z"):
            continue
        out[p[0][:-1].zfill(2)] = (float(p[1]), float(p[2]))
    return out


def firewx_center(cycle_hour: int, loc=None, given=None) -> tuple[float, float, str]:
    """The fire-weather centre for a cycle: --firewx-center if given, else the cycle's line of the nam_firewx_loc
    file (the seed of NCEP's last points when no file is given); a missing line or a 0 0 line takes setfirewx's
    Washington DC default."""
    if given is not None:
        return given[0], given[1], "given"
    if loc is None:
        loc = default_loc()
    pts = read_firewx_loc(loc)
    hh = f"{int(cycle_hour):02d}"
    lat, lon = pts.get(hh, (0.0, 0.0))
    src = "NCEP's last production points (FIREWX_LOC_SEED)" if loc is None else Path(loc).name
    if lat == 0.0 and lon == 0.0:
        return FIREWX_DEFAULT[0], FIREWX_DEFAULT[1], f"{src}: no {hh}z point -> setfirewx default (Washington DC)"
    return lat, lon, f"{src} {hh}z"


def firewx_location(lat: float) -> str:
    return "alaska" if lat >= FIREWX_ALASKA_LAT else "conus"


def firewx_check(parent: dict, lat: float, lon: float, im: int = 333, jm: int = 333, ratio: int = 2) -> None:
    """setfirewx.fd nest_domain_check_location_{conus,ak}: the nest centred at (lat, lon) must lie inside the outer
    5 rows of its parent nest (half a parent cell of slack). Refuses with the margin."""
    tlm, tph = to_rotated(lat, lon, parent["tph0d"], parent["tlm0d"])
    b = FIREWX_BUFFER_ROWS - 1
    sbd_n = parent["sbd"] + b * parent["dphd"]
    nbd_n = parent["sbd"] + (parent["jm"] - 1) * parent["dphd"] - b * parent["dphd"]
    wbd_n = parent["wbd"] + b * parent["dlmd"]
    ebd_n = parent["wbd"] + (parent["im"] - 1) * parent["dlmd"] - b * parent["dlmd"]
    hx, hy = 0.5 * (im - 1) * parent["dlmd"] / ratio, 0.5 * (jm - 1) * parent["dphd"] / ratio
    s, n, w, e = tph - hy, tph + hy, tlm - hx, tlm + hx
    hi, hj = 0.5 * parent["dlmd"], 0.5 * parent["dphd"]
    short = {"south": (sbd_n - s) - hj, "north": (n - nbd_n) - hj, "west": (wbd_n - w) - hi, "east": (e - ebd_n) - hi}
    bad = {k: v for k, v in short.items() if v >= 0}
    if bad:
        raise SystemExit(f"fire-weather nest at {lat}, {lon}: outside the outer {FIREWX_BUFFER_ROWS} rows of the "
                         f"{parent.get('name', 'parent')} (setfirewx.fd check), short by "
                         + ", ".join(f"{k} {v / (parent['dlmd'] if k in ('west', 'east') else parent['dphd']):.1f} "
                                     f"parent cells" for k, v in bad.items()))


def firewx_start(parent: dict, lat: float, lon: float, im: int = 333, jm: int = 333, ratio: int = 2) -> tuple:
    """(i_parent_start, j_parent_start): NPS geogrid compute_nest_locations (PROJ_ROTLLB):
    i = nint(1 + (xcenter - (im - 1)/2 x dlmd_nest - wbd_parent) / dlmd_parent), likewise j."""
    tlm, tph = to_rotated(lat, lon, parent["tph0d"], parent["tlm0d"])
    wbd_guess = tlm - (im - 1) * 0.5 * parent["dlmd"] / ratio
    sbd_guess = tph - (jm - 1) * 0.5 * parent["dphd"] / ratio
    return (_round_half_away(1 + (wbd_guess - parent["wbd"]) / parent["dlmd"]),
            _round_half_away(1 + (sbd_guess - parent["sbd"]) / parent["dphd"]))


def h_corners(g: dict) -> dict:
    """Geographic (lat, lon) of the four corner H points: sw, se, nw, ne."""
    w, s = g["wbd"], g["sbd"]
    e, n = w + (g["im"] - 1) * g["dlmd"], s + (g["jm"] - 1) * g["dphd"]
    f = lambda x, y: to_geographic(x, y, g["tph0d"], g["tlm0d"])
    return {"sw": f(w, s), "se": f(e, s), "nw": f(w, n), "ne": f(e, n)}


def master_projection(f: dict):
    """(fwd, inv) on master grid f: fwd(lat, lon) -> 1-based fractional (i, j), inv(i, j) -> (lat, lon) with lon in
    -180..180. f["kind"] "lambert" is a tangent Lambert conformal grid (true at Latin), "nps" a north polar
    stereographic grid true at LaD; point (1, 1) is (La1, Lo1). The projection is products.sets.OutGrid (Snyder 1987,
    USGS PP 1395, on a sphere) with the earth radius of NCEP's GRIB2 grid definitions (shape of earth 6, WMO-No. 306
    code table 3.2: 6371229 m), so these boxes and the files' own section 3 share one sphere."""
    from namgpu.products.sets import OutGrid, _ij
    if f["kind"] == "lambert":
        g = OutGrid("lambert", f["nx"], f["ny"], f["la1"], f["lo1"] % 360.0, f["dx"], f["dx"], lov=f["lov"] % 360.0,
                    latin1=f["latin"], latin2=f["latin"])
    elif f["kind"] == "nps":
        g = OutGrid("polar", f["nx"], f["ny"], f["la1"], f["lo1"] % 360.0, f["dx"], f["dx"], lov=f["lov"] % 360.0,
                    lad=f["lad"])
    else:
        raise ValueError(f"master grid kind {f['kind']!r}")

    def fwd(lat: float, lon: float) -> tuple[float, float]:
        return _ij(g, lat, lon)

    def inv(i: float, j: float) -> tuple[float, float]:
        la, lo = g.latlon_ij(i - 1.0, j - 1.0)
        return float(la), ((float(lo) + 180.0) % 360.0) - 180.0

    return fwd, inv


def firewx_output_grid(g: dict, location: str) -> dict:
    """The fire-weather hiresf grid for this nest (sorc/nam_firewx_gridspecs.fd): the corners in millidegrees
    (truncated), the box on the master grid (CONUS: the bounding box of the corner lat/lons, no
    padding; Alaska: the corners' i/j range with 100 points of padding), its SW point, and the wgrib2 -new_grid spec
    nam_prdgen_nest.sh passes. Master indices are the floor of the fractional (i, j), which is what NOAA's 06/18Z file
    shows where the box starts west of the master grid (column -11 for a SW index of -10.36)."""
    m = FIREWX_MASTER[location]
    corners = {k: (math.trunc(lat * 1000.0) / 1000.0, math.trunc(lon * 1000.0) / 1000.0)
               for k, (lat, lon) in h_corners(g).items()}
    if m["kind"] == "lambert":
        fwd, inv = master_projection(m)
        lats = [c[0] for c in corners.values()]
        lons = [c[1] % 360.0 for c in corners.values()]
        xsw, ysw = fwd(min(lats), min(lons))
        xne, yne = fwd(max(lats), max(lons))
        isw, jsw, ine, jne = math.floor(xsw), math.floor(ysw), math.floor(xne), math.floor(yne)
        lat0, lon0 = inv(float(isw), float(jsw))
        nx, ny = ine - isw + 1, jne - jsw + 1
        spec = (f"lambert:{int(m['lov'] % 360)}:{int(m['latin'])}:{int(m['latin'])} {lon0 % 360.0:.3f}:{nx}:"
                f"{int(m['dx'])} {lat0:.3f}:{ny}:{int(m['dx'])}")
    else:
        fwd, inv = master_projection(m)
        ij = [fwd(lat, lon) for lat, lon in corners.values()]
        ii, jj = [math.floor(x) for x, _ in ij], [math.floor(y) for _, y in ij]
        isw, jsw = min(ii) - m["pad"], min(jj) - m["pad"]
        ine, jne = max(ii) + m["pad"], max(jj) + m["pad"]
        lat0, lon0 = inv(float(isw), float(jsw))
        nx, ny = ine - isw + 1, jne - jsw + 1
        spec = f"nps:{int(m['lov'] % 360)}:60 {lon0 % 360.0:.3f}:{nx}:{int(m['dx'])} {lat0:.3f}:{ny}:{int(m['dx'])}"
    return {"location": location, "kind": m["kind"], "nx": nx, "ny": ny, "la1": round(lat0, 3),
            "lo1": round(lon0 % 360.0, 3), "dx_m": m["dx"], "master_ij_sw": [isw, jsw],
            "corners_millideg": {k: [round(a, 3), round(b, 3)] for k, (a, b) in corners.items()}, "new_grid": spec}




# ------------------------------------------------------------------------------------------ the row's grid

def nest_grid(row: dict, cycle_hour: int, loc=None, given=None) -> dict:
    """The fire-weather row's grid for one cycle, in ncep_nests.nest_grid's form (domain.nest_grid_associated's keys,
    on the NAM pole), plus a "firewx" block: the point and where it came from, the parent nest, the corners and the
    hiresf output grid."""
    lat, lon, how = firewx_center(cycle_hour, loc, given)
    location = firewx_location(lat)
    parent = parent_nest(location)
    im, jm, ratio = int(row["im"]), int(row["jm"]), int(row["ratio"])
    firewx_check(parent, lat, lon, im, jm, ratio)
    i_ps, j_ps = firewx_start(parent, lat, lon, im, jm, ratio)
    dlmd, dphd = parent["dlmd"] / ratio, parent["dphd"] / ratio
    dx = A_EARTH * math.radians(dlmd) / 1e3
    dy = A_EARTH * math.radians(dphd) / 1e3
    g = {"res_km": row["res_km"], "im": im, "jm": jm, "lm": 60, "tph0d": parent["tph0d"], "tlm0d": parent["tlm0d"],
         "dlmd": dlmd, "dphd": dphd, "wbd": round(parent["wbd"] + (i_ps - 1) * parent["dlmd"], 6),
         "sbd": round(parent["sbd"] + (j_ps - 1) * parent["dphd"], 6), "dx_km": round(dx, 4), "dy_km": round(dy, 4),
         "extent_km": [round((im - 1) * dx, 1), round((jm - 1) * dy, 1)], "associated": True,
         "i_parent_start": i_ps, "j_parent_start": j_ps, "parent_child_space_ratio": ratio,
         "parent_cells": [(im - 1) / ratio, (jm - 1) / ratio], "ncep_nest": "firewxnest", "ncep_window": None}
    tlm = g["wbd"] + 0.5 * (im - 1) * dlmd
    tph = g["sbd"] + 0.5 * (jm - 1) * dphd
    clat, clon = to_geographic(tlm, tph, g["tph0d"], g["tlm0d"])
    g["center"] = [round(clat, 5), round(clon, 5)]
    out = firewx_output_grid(g, location)
    noaa = FIREWX_NOAA_GRIDS.get((round(lat, 1), round(lon, 1)))
    g["firewx"] = {"point": [lat, lon], "point_from": how, "location": location, "parent": parent["name"],
                   "center_latlon": f"{lat:4.1f} {lon:6.1f}\n",
                   "corners": {k: [round(a, 4), round(b, 4)] for k, (a, b) in h_corners(g).items()},
                   "output_grid": out, "wgrib2": noaa or out["new_grid"],
                   "wgrib2_from": ("NOAA's own file for this point (FIREWX_NOAA_GRIDS)" if noaa
                                   else "nam_firewx_gridspecs.fd port (firewx_output_grid)")}
    return g


def domain_name(g: dict) -> str:
    """firewxnest_<lat>N_<lon>W: the grid moves between cycles, and each grid needs its own boundary set and static."""
    lat, lon = g["firewx"]["point"]
    return f"firewxnest_{abs(lat):.2f}{'N' if lat >= 0 else 'S'}_{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
