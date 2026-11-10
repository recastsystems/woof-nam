"""Projected (regional) source grids for the isobaric start: the geometry gpuwm's GRIB2 inventory does not report,
the point placement init-gfs-hinterp needs, and the turn of grid-relative winds to earth-relative.

The isobaric path (gfs_start.decode_source) was written for global regular lat-lon files (GDT 0). A regional analysis
comes on its own model grid: HRRR and RRFS v1 on a Lambert conformal grid (template 3.30, LambertGrid at the end of
this module), and RAP: NCEP GRIB2 template 3.32769, a rotated latitude-longitude grid given by its first
point (La1, Lo1), the rotation centre (LaC, LoC) and the spacing (Di, Dj). This module turns that into:
  - RotatedGrid.place(lat, lon): fractional source indices (fy, fx) of any geographic points. init-gfs-hinterp then
    runs its own kernel on those indices (Grid.proj: lat0 = lon0 = 0, dlat = dlon = 1, non-periodic), so the
    interpolation rule (four_pt + average_4pt) is the same one every other source gets;
  - RotatedGrid.edge_weight: 1 well inside the source grid, falling to 0 over `blend` source cells at its edge and
    0 outside (the start takes the row's global fallback source there, gfs_start.stage);
  - RotatedGrid.wind_turn: cos/sin of the angle from local east to the grid's +x axis at every source point, so
    grid-relative u, v become earth-relative before the interpolation (as init-gfs-hinterp expects).

Geometry, as measured on rap.t00z.wrfprsf00.grib2 of 2023-07-29 (953 x 834): La1 -10.590603, Lo1 220.914154, LaC 54,
LoC 254 (microdegrees), Di = Dj = 121813000 (nanodegrees: 0.121813 deg), resolution flags 0x38 (bit 0x08: winds
grid-relative), scan 0x40 (rows south to north). gpuwm's grib2_inventory reports zeros for every geometry field and
0x00 for the flags of this template, so the section is read here. The first point's rotated coordinates are
(-50.7434, -57.9925) deg: the grid is centred on (LaC, LoC) with a spacing of 0.121833 deg, which is RAP's 13.545 km
on WRF's 6370 km earth; Di is the same spacing on a 6371.2 km earth (0.121813). Placement uses the spacing the first
point and the centre imply (the grid's own points), and refuses a file whose Di/Dj disagree with it by more than
0.1 % (a grid that is not centred on its rotation centre would be misplaced).

Refusals (each names the breakage it prevents):
  - a template other than 3.32769: its geometry is not parsed, so its points would be placed in the wrong cells;
  - Di/Dj more than 0.1 % from the spacing the centred grid implies: the grid is not centred on (LaC, LoC), and the
    centred placement would put the far edge cells off by more than a cell;
  - scan mode other than 0x40 (+i, +j): init-grib2-decode forces +j for this template, so any other storage would be
    read flipped.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GDT_ROTATED_NCEP = 32769
GDT_LAMBERT = 30             # Lambert conformal (HRRR wrfprsf00, RRFS v1 prslev / 2dfld: 1799 x 1059, 3 km)
# GRIB2 code table 3.2 earth shapes placed here (spheres of these radii; shape 1 reads its radius from the section)
EARTH_RADIUS = {0: 6367470.0, 6: 6371229.0}
GRID_RELATIVE_UV_BIT = 0x08
SPACING_TOL = 1.0e-3
DTR = np.pi / 180.0


def read_grid_section(path) -> dict:
    """The grid definition (section 3) of the first GRIB2 message of a file, as a dict."""
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:4] != b"GRIB" or head[7] != 2:
            raise ValueError(f"source-grids: {path} is not a GRIB2 file")
        while True:
            h = f.read(5)
            if len(h) < 5:
                raise ValueError(f"source-grids: {path}: no section 3 in the first message")
            n, num = struct.unpack(">IB", h)
            body = f.read(n - 5)
            if num == 3:
                s = h + body
                break
    return parse_grid_section(s)


def parse_grid_section(s: bytes) -> dict:
    def u4(o):
        return struct.unpack(">I", s[o:o + 4])[0]

    def i4(o):                       # GRIB2 signed: sign bit then magnitude
        v = u4(o)
        return -(v & 0x7FFFFFFF) if v & 0x80000000 else v
    gdt = struct.unpack(">H", s[12:14])[0]
    if gdt == GDT_LAMBERT:
        # template 3.30 (octet n of the section is s[n - 1]): 15 shape of earth, 16-20 radius, 31-38 Nx Ny, 39-46 La1
        # Lo1, 47 resolution flags, 48-55 LaD LoV, 56-63 Dx Dy (mm), 64 projection centre, 65 scan, 66-73 Latin1 Latin2
        return {"gdt": gdt, "shape": s[14], "radius_scale": s[15], "radius_value": u4(16), "nx": u4(30), "ny": u4(34),
                "la1": i4(38) * 1e-6, "lo1": i4(42) * 1e-6, "resflags": s[46], "lad": i4(47) * 1e-6,
                "lov": i4(51) * 1e-6, "dx": u4(55) * 1e-3, "dy": u4(59) * 1e-3, "pcf": s[63], "scan": s[64],
                "latin1": i4(65) * 1e-6, "latin2": i4(69) * 1e-6}
    if gdt != GDT_ROTATED_NCEP:
        raise ValueError(f"source-grids: grid template 3.{gdt} is not parsed here (3.{GDT_ROTATED_NCEP} and "
                         f"3.{GDT_LAMBERT} only); its points would be placed in the wrong cells")
    return {"gdt": gdt, "shape": s[14], "nx": u4(30), "ny": u4(34),
            "la1": i4(46) * 1e-6, "lo1": i4(50) * 1e-6, "resflags": s[54],
            "lac": i4(55) * 1e-6, "loc": i4(59) * 1e-6, "di": u4(63) * 1e-9, "dj": u4(67) * 1e-9,
            "scan": s[71]}


def geo_to_rotated(xp, lat, lon, tph0, tlm0):
    """Geographic (radians) -> rotated (tph, tlm) radians about the centre (tph0, tlm0) radians, the rotation of
    namgpu.grid.rotated_to_geographic (NMM / WRF rotated lat-lon)."""
    x = xp.cos(lat) * xp.cos(lon - tlm0)
    y = xp.cos(lat) * xp.sin(lon - tlm0)
    z = xp.sin(lat)
    xr = np.cos(tph0) * x + np.sin(tph0) * z
    zr = -np.sin(tph0) * x + np.cos(tph0) * z
    return xp.arcsin(xp.clip(zr, -1.0, 1.0)), xp.arctan2(y, xr)


def rotated_to_geo(xp, tph, tlm, tph0, tlm0):
    """Inverse of geo_to_rotated (radians)."""
    xr = xp.cos(tph) * xp.cos(tlm)
    y = xp.cos(tph) * xp.sin(tlm)
    zr = xp.sin(tph)
    x = np.cos(tph0) * xr - np.sin(tph0) * zr
    z = np.sin(tph0) * xr + np.cos(tph0) * zr
    return xp.arcsin(xp.clip(z, -1.0, 1.0)), tlm0 + xp.arctan2(y, x)


class GlobalLatLon:
    """A global regular lat-lon source (GDT 0: GFS, IFS, AIFS open data) as the merge in gfs_start.stage sees it: it
    covers every point (edge weight 1 everywhere), so a row with a fallback on a global grid merges only above its
    top (the fallback's levels above the source's highest isobaric level)."""
    kind = "global lat-lon"
    grid_relative_winds = False

    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = int(nx), int(ny)

    def describe(self) -> dict:
        return {"template": 0, "nx": self.nx, "ny": self.ny, "global": True}

    def place(self, lat, lon, xp=np):
        z = xp.zeros(xp.asarray(lat).shape, dtype=xp.float64)
        return z, z

    def edge_weight(self, fy, fx, blend, xp=np):
        return xp.ones(xp.asarray(fy).shape, dtype=xp.float32)


@dataclass
class RotatedGrid:
    """A rotated lat-lon source grid (NCEP 3.32769), centred on its rotation centre. Angles in degrees."""
    nx: int
    ny: int
    tph0d: float
    tlm0d: float
    tph1d: float          # rotated latitude of row 0
    tlm1d: float          # rotated longitude of column 0
    dphd: float
    dlmd: float
    grid_relative_winds: bool

    @classmethod
    def from_section(cls, sec: dict) -> "RotatedGrid":
        if sec["scan"] != 0x40:
            raise ValueError(f"source-grids: scan mode 0x{sec['scan']:02x}; init-grib2-decode reads 3.32769 records "
                             "as +i, +j (0x40), so this file would be read flipped")
        tph0, tlm0 = sec["lac"] * DTR, sec["loc"] * DTR
        tph1, tlm1 = geo_to_rotated(np, np.float64(sec["la1"] * DTR), np.float64(sec["lo1"] * DTR), tph0, tlm0)
        tph1d, tlm1d = float(np.degrees(tph1)), float(np.degrees(tlm1))
        dlmd = -2.0 * tlm1d / (sec["nx"] - 1)
        dphd = -2.0 * tph1d / (sec["ny"] - 1)
        for name, implied, said in (("Di", dlmd, sec["di"]), ("Dj", dphd, sec["dj"])):
            if not said > 0 or abs(implied - said) / said > SPACING_TOL:
                raise ValueError(f"source-grids: {name} {said:.6f} deg but a grid centred on ({sec['lac']}, "
                                 f"{sec['loc']}) with its first point at ({sec['la1']}, {sec['lo1']}) has "
                                 f"{implied:.6f} deg; the centred placement would misplace the far edge")
        return cls(nx=int(sec["nx"]), ny=int(sec["ny"]), tph0d=float(sec["lac"]), tlm0d=float(sec["loc"]),
                   tph1d=tph1d, tlm1d=tlm1d, dphd=dphd, dlmd=dlmd,
                   grid_relative_winds=bool(sec["resflags"] & GRID_RELATIVE_UV_BIT))

    kind = "rotated lat-lon"

    def describe(self) -> dict:
        return {"template": GDT_ROTATED_NCEP, "nx": self.nx, "ny": self.ny, "centre": [self.tph0d, self.tlm0d],
                "first_rotated": [self.tph1d, self.tlm1d], "spacing_deg": [self.dphd, self.dlmd],
                "grid_relative_winds": self.grid_relative_winds}

    # ---- points -> source indices ---------------------------------------------------------------------------------
    def place(self, lat, lon, xp=np):
        """Fractional (fy, fx) source indices (row south to north, column west to east) of geographic points given
        in radians; float64, same shape. Points off the grid get indices outside [0, n-1]."""
        tph, tlm = geo_to_rotated(xp, xp.asarray(lat, dtype=xp.float64), xp.asarray(lon, dtype=xp.float64),
                                  self.tph0d * DTR, self.tlm0d * DTR)
        return (xp.degrees(tph) - self.tph1d) / self.dphd, (xp.degrees(tlm) - self.tlm1d) / self.dlmd

    def edge_weight(self, fy, fx, blend, xp=np):
        """1 at least `blend` cells inside the grid, linear to 0 at its edge, 0 outside (float32)."""
        d = xp.minimum(xp.minimum(fx, (self.nx - 1) - fx), xp.minimum(fy, (self.ny - 1) - fy))
        return xp.clip(d / float(blend), 0.0, 1.0).astype(xp.float32)

    # ---- grid-relative winds ----------------------------------------------------------------------------------------
    def latlon(self, j, i, xp=np):
        """Geographic (lat, lon) radians of fractional source indices (row j, column i)."""
        tph = (self.tph1d + xp.asarray(j, dtype=xp.float64) * self.dphd) * DTR
        tlm = (self.tlm1d + xp.asarray(i, dtype=xp.float64) * self.dlmd) * DTR
        return rotated_to_geo(xp, tph, tlm, self.tph0d * DTR, self.tlm0d * DTR)

    def wind_turn(self, xp=np):
        """(cos a, sin a) float32 [ny, nx], a = the angle from local east to the grid's +x axis (counter-clockwise).
        Earth-relative winds: u_e = u_g cos a - v_g sin a, v_n = u_g sin a + v_g cos a (the grid is conformal, so its
        +y axis is +x turned by 90 deg). The +x direction is the chord between the points half a cell either side."""
        j = xp.arange(self.ny, dtype=xp.float64)[:, None] * xp.ones((1, self.nx))
        i = xp.ones((self.ny, 1)) * xp.arange(self.nx, dtype=xp.float64)[None, :]
        la, lo = self.latlon(j, i, xp)
        la_e, lo_e = self.latlon(j, i + 0.5, xp)
        la_w, lo_w = self.latlon(j, i - 0.5, xp)

        def cart(a, b):
            return xp.cos(a) * xp.cos(b), xp.cos(a) * xp.sin(b), xp.sin(a)
        xe, ye, ze = cart(la_e, lo_e)
        xw, yw, zw = cart(la_w, lo_w)
        tx, ty, tz = xe - xw, ye - yw, ze - zw
        te = -tx * xp.sin(lo) + ty * xp.cos(lo)
        tn = -tx * xp.sin(la) * xp.cos(lo) - ty * xp.sin(la) * xp.sin(lo) + tz * xp.cos(la)
        a = xp.arctan2(tn, te)
        return xp.cos(a).astype(xp.float32), xp.sin(a).astype(xp.float32)


def to_earth(u, v, turn):
    """Grid-relative (u, v) -> earth-relative, in place; u, v [..., ny, nx], turn = wind_turn()."""
    c, s = turn
    ue = u * c - v * s
    v[...] = u * s + v * c
    u[...] = ue


@dataclass
class LambertGrid:
    """A Lambert conformal source grid (GRIB2 template 3.30) on a sphere: HRRR's and RRFS v1's 3 km CONUS grid.
    Placement is the spherical Lambert projection (Snyder 1987, eqs. 15-1 to 15-5) about LoV with standard parallels
    Latin1/Latin2; grid indices are (x - x1) / Dx, (y - y1) / Dy from the first point. Angles in degrees, metres.

    Measured on hrrr.t12z.wrfprsf00.grib2 of 2026-01-23: 1799 x 1059, La1 21.138123, Lo1 237.280472, LoV 262.5,
    Latin1 = Latin2 = 38.5, Dx = Dy = 3000 m, earth shape 6 (6371229 m sphere), winds grid-relative, scan 0x40 (+i, +j:
    rows south to north). tests/test_hrrr_source.py checks the far corner against HRRR's published one.

    Refusals (each names the breakage it prevents): scan other than 0x40 (read flipped), a south-pole cone or a cone
    constant <= 0 (every point misplaced), an ellipsoidal earth (the sphere formulas would misplace points)."""
    nx: int
    ny: int
    la1: float
    lo1: float
    lov: float
    latin1: float
    latin2: float
    dx: float
    dy: float
    radius: float
    grid_relative_winds: bool

    kind = "Lambert conformal"

    @classmethod
    def from_section(cls, sec: dict) -> "LambertGrid":
        if sec["scan"] != 0x40:
            raise ValueError(f"source-grids: Lambert scan mode 0x{sec['scan']:02x}; placement assumes +i, +j (0x40), "
                             "so this file would be placed flipped")
        if sec["pcf"] & 0x80:
            raise ValueError("source-grids: Lambert projection centre flag says south pole; placement is for a north "
                             "pole cone, so every point would be misplaced")
        if sec["shape"] == 1:
            radius = sec["radius_value"] / 10.0 ** sec["radius_scale"]
        elif sec["shape"] in EARTH_RADIUS:
            radius = EARTH_RADIUS[sec["shape"]]
        else:
            raise ValueError(f"source-grids: earth shape {sec['shape']} (an ellipsoid) is not placed here; the sphere "
                             "formulas would misplace points")
        return cls(nx=int(sec["nx"]), ny=int(sec["ny"]), la1=float(sec["la1"]), lo1=float(sec["lo1"]),
                   lov=float(sec["lov"]), latin1=float(sec["latin1"]), latin2=float(sec["latin2"]),
                   dx=float(sec["dx"]), dy=float(sec["dy"]), radius=float(radius),
                   grid_relative_winds=bool(sec["resflags"] & GRID_RELATIVE_UV_BIT))

    def __post_init__(self):
        p1, p2 = self.latin1 * DTR, self.latin2 * DTR
        if abs(p1 - p2) < 1e-10:
            n = np.sin(p1)
        else:
            n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
        if not n > 0:
            raise ValueError("source-grids: Lambert cone constant <= 0 (southern cone); not placed here")
        self.n = float(n)
        self.F = float(np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n)
        x1, y1 = self._xy(np, np.float64(self.la1 * DTR), np.float64(self.lo1 * DTR))
        self.x1, self.y1 = float(x1), float(y1)

    def describe(self) -> dict:
        return {"template": GDT_LAMBERT, "nx": self.nx, "ny": self.ny, "first_point": [self.la1, self.lo1],
                "lov": self.lov, "latin": [self.latin1, self.latin2], "spacing_m": [self.dx, self.dy],
                "radius_m": self.radius, "grid_relative_winds": self.grid_relative_winds}

    def _xy(self, xp, lat, lon):
        dl = xp.remainder(lon - self.lov * DTR + np.pi, 2 * np.pi) - np.pi
        rho = self.radius * self.F / xp.tan(np.pi / 4 + lat / 2) ** self.n
        return rho * xp.sin(self.n * dl), -rho * xp.cos(self.n * dl)

    def place(self, lat, lon, xp=np):
        """Fractional (fy, fx) source indices of geographic points given in radians (float64), as RotatedGrid.place."""
        x, y = self._xy(xp, xp.asarray(lat, dtype=xp.float64), xp.asarray(lon, dtype=xp.float64))
        return (y - self.y1) / self.dy, (x - self.x1) / self.dx

    def latlon(self, j, i, xp=np):
        """Geographic (lat, lon) radians of fractional source indices (row j, column i)."""
        x = self.x1 + xp.asarray(i, dtype=xp.float64) * self.dx
        y = self.y1 + xp.asarray(j, dtype=xp.float64) * self.dy
        rho = xp.sqrt(x * x + y * y)
        theta = xp.arctan2(x, -y)
        lat = 2.0 * xp.arctan((self.radius * self.F / rho) ** (1.0 / self.n)) - np.pi / 2
        return lat, self.lov * DTR + theta / self.n

    edge_weight = RotatedGrid.edge_weight
    wind_turn = RotatedGrid.wind_turn


def grid_of(path):
    """The source grid of a regional GRIB2 file: RotatedGrid (3.32769) or LambertGrid (3.30)."""
    sec = read_grid_section(Path(path))
    if sec["gdt"] == GDT_LAMBERT:
        return LambertGrid.from_section(sec)
    return RotatedGrid.from_section(sec)
