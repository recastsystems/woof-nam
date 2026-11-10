"""Product sets: which NOAA product file `woof-nam products` writes, on which grid, from which record table.

A product set is data: a record table (one row per record of NOAA's file, in NOAA's order: spec.py documents the
columns), NOAA's own facts per record, the file name, the output grid, how fields are put on that grid and which
accumulation buckets the file carries. Adding a NOAA product file is a row in SETS (the arbitrary acceptance test),
never a new code path.

  awphys       NAM 12 km CONUS parent, nam.tHHz.awphysFF.tm00.grib2 on NCEP grid 218 (the 12 km parent history)
  conusnest    NAM CONUS 3 km nest, nam.tHHz.conusnest.hiresfFF.tm00.grib2 on the 3 km Lambert (HRRR) grid
  alaskanest   NAM Alaska 3 km nest, polar stereographic 2976 m
  hawaiinest   NAM Hawaii 3 km nest, Mercator 2500 m
  priconest    NAM Puerto Rico 3 km nest, Mercator 2500 m
  firewxnest   NAM fire-weather 1.5 km nest: a Lambert box of NCEP's 1.27 km master grid around the nest's corners
               (NCEP's nam_firewx_gridspecs rule, per cycle), or of the 1.488 km polar master over Alaska
  hiresf       any other nest: the hiresf record set on a Lambert box around the nest at the nest's spacing

The nest grids are NOAA's own (ush/nam_prdgen_nest.sh, wgrib2 -new_grid; checked against the decoded 2026-11-03
06Z files). NCEP regridded each nest's native post output with nearest-neighbour interpolation, except APCP, WEASD
and SNOD (budget), turned the winds grid-relative, and left points outside the nest missing (bitmap): the frame
here does the same, before the post, so each column's records are the nest column's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R_NCEP = 6371229.0          # GRIB2 shape of earth 6 (NCEP's sphere)
DEG = math.pi / 180.0


@dataclass(frozen=True)
class OutGrid:
    """An output grid as GRIB2 section 3 holds it: first point (la1, lo1, degrees, lon east), counts, spacing in m
    at the true latitude, and the projection's own keys. Rows run south to north, points west to east (scan 0x40)."""
    kind: str                 # lambert | polar | mercator
    nx: int
    ny: int
    la1: float
    lo1: float
    dx: float
    dy: float
    lov: float = 0.0          # lambert / polar orientation (deg east)
    latin1: float = 0.0       # lambert true latitudes
    latin2: float = 0.0
    lad: float = 0.0          # polar / mercator true latitude
    la2: float | None = None  # mercator last point as NCEP writes it (wgrib2 mercator:...:lon1 ... :lat1)
    lo2: float | None = None
    radius: float = R_NCEP
    name: str = ""

    # ---- forward / inverse on the sphere (Snyder 1987, PP 1395: 15-1..15-4, 21-1..21-4, 7-1..7-4)
    def _lcc(self):
        l1, l2 = self.latin1 * DEG, self.latin2 * DEG
        if abs(l1 - l2) < 1e-9:
            n = math.sin(l1)
        else:
            n = (math.log(math.cos(l1)) - math.log(math.cos(l2))) / (
                math.log(math.tan(math.pi / 4 + l2 / 2)) - math.log(math.tan(math.pi / 4 + l1 / 2)))
        F = math.cos(l1) * math.tan(math.pi / 4 + l1 / 2) ** n / n
        return n, F

    def forward(self, lat, lon):
        lat = np.asarray(lat, np.float64) * DEG
        dl = ((np.asarray(lon, np.float64) - self.lov + 180.0) % 360.0 - 180.0) * DEG
        R = self.radius
        if self.kind == "lambert":
            n, F = self._lcc()
            rho = R * F / np.tan(np.pi / 4 + lat / 2) ** n
            return rho * np.sin(n * dl), -rho * np.cos(n * dl)
        if self.kind == "polar":
            k = R * (1.0 + math.sin(self.lad * DEG))
            rho = k * np.tan(np.pi / 4 - lat / 2)
            return rho * np.sin(dl), -rho * np.cos(dl)
        if self.kind == "mercator":
            c = R * math.cos(self.lad * DEG)
            dl0 = ((np.asarray(lon, np.float64) - self.lo1) % 360.0) * DEG
            return c * dl0, c * np.log(np.tan(np.pi / 4 + lat / 2))
        raise ValueError(f"grid kind {self.kind!r}")

    def inverse(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        R = self.radius
        if self.kind == "lambert":
            n, F = self._lcc()
            rho = np.sign(n) * np.hypot(x, y)
            theta = np.arctan2(x, -y)
            lat = 2.0 * np.arctan((R * F / rho) ** (1.0 / n)) - np.pi / 2
            return np.degrees(lat), (self.lov + np.degrees(theta / n)) % 360.0
        if self.kind == "polar":
            k = R * (1.0 + math.sin(self.lad * DEG))
            rho = np.hypot(x, y)
            lat = np.pi / 2 - 2.0 * np.arctan(rho / k)
            return np.degrees(lat), (self.lov + np.degrees(np.arctan2(x, -y))) % 360.0
        if self.kind == "mercator":
            c = R * math.cos(self.lad * DEG)
            lat = 2.0 * np.arctan(np.exp(y / c)) - np.pi / 2
            return np.degrees(lat), (self.lo1 + np.degrees(x / c)) % 360.0
        raise ValueError(f"grid kind {self.kind!r}")

    def latlon_ij(self, fi, fj):
        """Geographic lat/lon (deg, lon east 0-360) of fractional grid indices (0 = the first point)."""
        x0, y0 = self.forward(self.la1, self.lo1)
        return self.inverse(x0 + self.dx * np.asarray(fi, np.float64), y0 + self.dy * np.asarray(fj, np.float64))

    def latlon(self):
        j, i = np.meshgrid(np.arange(self.ny, dtype=np.float64), np.arange(self.nx, dtype=np.float64), indexing="ij")
        return self.latlon_ij(i, j)

    def turn(self, lat, lon):
        """(cos, sin) of the earth-to-grid turn at each point: u_grid = u_e cos - v_e sin, v_grid = u_e sin + v_e cos."""
        if self.kind == "mercator":
            return np.ones_like(lat, np.float32), np.zeros_like(lat, np.float32)
        cone = self._lcc()[0] if self.kind == "lambert" else 1.0
        a = np.radians(((np.asarray(lon, np.float64) - self.lov + 180.0) % 360.0) - 180.0) * cone
        return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)

    def frame_attrs(self) -> dict:
        """The WRF projection attributes the GRIB2 writer turns into section 3 (rw-grib2export grid.rs)."""
        proj = {"lambert": 1, "polar": 2, "mercator": 3}[self.kind]
        t1 = self.latin1 if self.kind == "lambert" else self.lad
        t2 = self.latin2 if self.kind == "lambert" else self.lad
        return {"MAP_PROJ": proj, "TRUELAT1": t1, "TRUELAT2": t2,
                "STAND_LON": ((self.lov + 180.0) % 360.0) - 180.0, "DX": self.dx, "DY": self.dy,
                "MAP_PROJ_CHAR": {"lambert": "Lambert Conformal", "polar": "Polar Stereographic",
                                  "mercator": "Mercator"}[self.kind]}

    def spec(self) -> dict:
        d = {"kind": self.kind, "nx": self.nx, "ny": self.ny, "la1": self.la1, "lo1": self.lo1, "dx": self.dx,
             "dy": self.dy, "earth_radius_m": self.radius, "scan_mode": "0x40"}
        if self.kind == "lambert":
            d.update(lov=self.lov, latin1=self.latin1, latin2=self.latin2)
        elif self.kind == "polar":
            d.update(lov=self.lov, lad=self.lad)
        else:
            d.update(lad=self.lad, la2=self.la2, lo2=self.lo2)
        return d


@dataclass(frozen=True)
class ProductSet:
    name: str
    table: str                       # record table (namgpu/products/)
    noaa: str                        # NOAA's facts per record (generated from NOAA's file)
    file_name: str                   # {HH} cycle hour, {FF} forecast hour; the prefix replaces "woof-nam"
    grid: OutGrid | str              # an OutGrid, or a rule: "firewx" | "from-nest"
    interp: str = "nearest"          # nearest | bilinear (plus `nearest_fields` / `budget_fields`)
    nearest_fields: tuple = ()
    budget_fields: tuple = ()
    buckets_h: tuple = (3,)
    native_vectors: bool = False     # VUCSH/VVCSH in the native rotated grid's orientation (awphys) or grid-relative
    mask: bool = True                # points off the model domain missing (bitmap); False: the grid must be covered
    extra_carriers: bool = True      # the nest (hiresf) carriers (namgpu.products.hiresf_carriers)
    grid_name: str = ""


NEAREST_CLASSES = ("ISLTYP", "IVGTYP", "SM", "LANDMASK")
# NCEP's budget fields (nam_prdgen_nest.sh: APCP, WEASD, SNOD); here the history fields they are made from
BUDGET = ("RAINNC", "RAINC", "NAM_WEASD", "NAM_SNOD", "NAMH_WEASD_ACC")

AWPHYS = ProductSet(
    name="awphys", table="awphys.tsv", noaa="awphys-noaa.tsv", file_name="woof-nam.t{HH}z.awphys{FF}.tm00.grib2",
    grid="ncep218", interp="bilinear", nearest_fields=NEAREST_CLASSES, buckets_h=(12, 3), native_vectors=True,
    mask=False, extra_carriers=False, grid_name="NCEP 218 (Lambert conformal 614 x 428, Dx 12191 m, shape of earth 6)")


def _nest(name, table, grid, grid_name):
    return ProductSet(name=name, table=table, noaa=f"hiresf-{name}-noaa.tsv",
                      file_name="woof-nam.t{HH}z." + name + ".hiresf{FF}.tm00.grib2", grid=grid,
                      budget_fields=BUDGET, grid_name=grid_name)


SETS = {
    "awphys": AWPHYS,
    "conusnest": _nest("conusnest", "hiresf-conusnest.tsv",
                       OutGrid("lambert", 1799, 1059, 21.138, 237.280, 3000.0, 3000.0, lov=262.5, latin1=38.5,
                               latin2=38.5, name="conusnest"),
                       "NAM conusnest output grid (Lambert 1799 x 1059, 3000 m, LoV 262.5, Latin 38.5; the HRRR grid)"),
    "alaskanest": _nest("alaskanest", "hiresf.tsv",
                        OutGrid("polar", 1649, 1105, 40.530, 181.429, 2976.0, 2976.0, lov=210.0, lad=60.0,
                                name="alaskanest"),
                        "NAM alaskanest output grid (polar stereographic 1649 x 1105, 2976 m, LoV 210, LaD 60)"),
    "hawaiinest": _nest("hawaiinest", "hiresf.tsv",
                        OutGrid("mercator", 321, 225, 18.073, 198.475, 2500.0, 2500.0, lad=20.0, la2=23.088,
                                lo2=206.131, name="hawaiinest"),
                        "NAM hawaiinest output grid (Mercator 321 x 225, 2500 m, LaD 20)"),
    "priconest": _nest("priconest", "hiresf.tsv",
                       OutGrid("mercator", 544, 310, 15.0, 284.5, 2500.0, 2500.0, lad=20.0, la2=22.005, lo2=297.491,
                               name="priconest"),
                       "NAM priconest output grid (Mercator 544 x 310, 2500 m, LaD 20)"),
    # NCEP's fire-weather nest empties its accumulators every hour (nam_configfile_firewxnest_* nhrs_* 1): NOAA's
    # firewx files carry 1-2, 2-3, ... windows where the other nests carry the 3 h bucket
    "firewxnest": replace(_nest("firewxnest", "hiresf-firewxnest.tsv", "firewx",
                                "NAM firewxnest output grid (a box of NCEP's 1.27 km Lambert master, per cycle)"),
                          buckets_h=(1,)),
    "hiresf": replace(_nest("nest", "hiresf.tsv", "from-nest", "a Lambert box around the nest at its spacing"),
                      name="hiresf", noaa="hiresf-alaskanest-noaa.tsv"),
}
NEST_SETS = tuple(k for k in SETS if k != "awphys")


def get(name: str) -> ProductSet:
    if name not in SETS:
        raise SystemExit(f"product set {name!r}: one of {', '.join(SETS)}")
    return SETS[name]


def table_path(ps: ProductSet) -> Path:
    return HERE / ps.table


def noaa_path(ps: ProductSet) -> Path:
    return HERE / ps.noaa


# ------------------------------------------------------------------------------------------------ grid rules

def _corners(lat, lon):
    """(lat, lon east) of the SW, SE, NW, NE corner points of a [j, i] grid."""
    return [(float(lat[j, i]), float(lon[j, i]) % 360.0) for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1))]


# NCEP's fire-weather master grids (sorc/nam_firewx_gridspecs.fd/firewx_gridspecs.f, GRD): CONUS Lambert 4289 x 2753,
# La1 20.192 Lo1 -121.554, LoV -95, Latin 25, 1270 m; Alaska polar stereographic 3297 x 2209, La1 40.530 Lo1 -178.571,
# LoV -150, 1488 m (true at 60 N), 100 points of padding.
FIREWX_CONUS = OutGrid("lambert", 4289, 2753, 20.192, 238.446, 1270.0, 1270.0, lov=265.0, latin1=25.0, latin2=25.0,
                       radius=6371200.0)
FIREWX_ALASKA = OutGrid("polar", 3297, 2209, 40.530, 181.429, 1488.0, 1488.0, lov=210.0, lad=60.0, radius=6371200.0)


def _ij(g: OutGrid, lat, lon):
    """1-based fractional (i, j) of a point on a master grid (W3FB11 / W3FB06 convention)."""
    x0, y0 = g.forward(g.la1, g.lo1)
    x, y = g.forward(lat, lon)
    return 1.0 + (x - x0) / g.dx, 1.0 + (y - y0) / g.dy


# NOAA's decoded fire-weather grids for NCEP's last production points (2026-10-20 to 11-03, every day sampled): the
# rule below lands within one 1.27 km master cell of these from a 333 x 333 nest centred on the point, so the exact
# boxes NOAA wrote are kept for those two centres (lat, lon east) and the rule serves any other point.
FIREWX_KNOWN = {
    (42.0, 237.7): OutGrid("lambert", 646, 389, 39.182, 233.507, 1270.0, 1270.0, lov=265.0, latin1=25.0, latin2=25.0,
                           name="firewxnest"),
    (39.0, 248.0): OutGrid("lambert", 543, 379, 36.487, 244.377, 1270.0, 1270.0, lov=265.0, latin1=25.0, latin2=25.0,
                           name="firewxnest"),
}


def firewx_grid(lat, lon, alaska: bool | None = None) -> OutGrid:
    """NCEP's fire-weather output grid for a nest with these point coordinates: CONUS: the box of the 1.27 km master
    grid spanning the bounding box (min/max lat and lon) of the nest's four corners, its SW corner the master point
    at the truncated indices; Alaska: the four corners' master indices, padded by 100 points (firewx_gridspecs.f)."""
    corners = _corners(lat, lon)
    ny, nx = np.shape(lat)
    cla, clo = float(lat[ny // 2, nx // 2]), float(lon[ny // 2, nx // 2]) % 360.0
    for (kla, klo), g in FIREWX_KNOWN.items():
        if abs(cla - kla) < 0.05 and abs(clo - klo) < 0.05:
            return g
    if alaska is None:
        alaska = sum(c[0] for c in corners) / 4.0 >= 52.0
    if not alaska:
        m = FIREWX_CONUS
        la_sw, lo_sw = min(c[0] for c in corners), min(c[1] for c in corners)
        la_ne, lo_ne = max(c[0] for c in corners), max(c[1] for c in corners)
        xs, ys = _ij(m, la_sw, lo_sw)
        xn, yn = _ij(m, la_ne, lo_ne)
        isw, jsw, ine, jne = int(xs), int(ys), int(xn), int(yn)
        la, lo = m.latlon_ij(isw - 1.0, jsw - 1.0)
        la1, lo1 = round(float(la), 3), round(float(lo), 3)
        return OutGrid("lambert", ine - isw + 1, jne - jsw + 1, la1, lo1, m.dx, m.dy, lov=m.lov, latin1=m.latin1,
                       latin2=m.latin2, name="firewxnest")
    m = FIREWX_ALASKA
    ij = [_ij(m, la, lo) for la, lo in corners]
    i0, i1 = min(int(a) for a, _ in ij) - 100, max(int(a) for a, _ in ij) + 100
    j0, j1 = min(int(b) for _, b in ij) - 100, max(int(b) for _, b in ij) + 100
    la, lo = m.latlon_ij(i0 - 1.0, j0 - 1.0)
    return OutGrid("polar", i1 - i0 + 1, j1 - j0 + 1, round(float(la), 3), round(float(lo), 3), m.dx, m.dy, lov=m.lov,
                   lad=m.lad, name="firewxnest")


def from_nest_grid(lat, lon, dx_m: float) -> OutGrid:
    """A Lambert box around a nest of any shape (the fire-weather rule on a grid of the nest's own spacing): true at
    and oriented on the nest centre, spanning the bounding box of the nest's four corners."""
    ny, nx = lat.shape
    cl, co = float(lat[ny // 2, nx // 2]), float(lon[ny // 2, nx // 2]) % 360.0
    lat0 = max(min(cl, 80.0), -80.0)
    if abs(lat0) < 1.0:
        lat0 = 1.0 if lat0 >= 0 else -1.0
    dx = float(round(dx_m))
    m = OutGrid("lambert", 2, 2, cl, co, dx, dx, lov=co, latin1=lat0, latin2=lat0)
    xs, ys = m.forward(lat, lon)
    x0, x1, y0, y1 = float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))
    nxo, nyo = int(math.floor((x1 - x0) / dx)) + 1, int(math.floor((y1 - y0) / dx)) + 1
    la1, lo1 = m.inverse(x0, y0)
    return OutGrid("lambert", nxo, nyo, round(float(la1), 3), round(float(lo1), 3), dx, dx, lov=round(co, 3),
                   latin1=round(lat0, 3), latin2=round(lat0, 3), name="hiresf")


def resolve(ps: ProductSet, lat=None, lon=None, dx_m: float | None = None) -> OutGrid | None:
    """The output grid of a set for a source grid (lat/lon of the source points); None for awphys (grid 218)."""
    if isinstance(ps.grid, OutGrid):
        return ps.grid
    if ps.grid == "ncep218":
        return None
    if ps.grid == "firewx":
        return firewx_grid(lat, lon)
    if ps.grid == "from-nest":
        return from_nest_grid(lat, lon, dx_m or 3000.0)
    raise ValueError(f"set {ps.name}: grid rule {ps.grid!r}")
