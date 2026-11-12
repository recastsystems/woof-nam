"""io-wind-rotate: earth-relative winds at H points from the model's grid-relative winds.

Unit io-wind-rotate (mode "new"); kernel namgpu/kernels/io_wind_rotate.cu: io_wind_rotate_3d (one
thread per H point and level) and io_wind_rotate_10m (one thread per H point).

What it computes
  The model's U and V are the components along increasing grid i and grid j of the rotated
  latitude-longitude grid (the grid-relative components of WMO-No. 306, GRIB2 code table 3.3, bit 5 = 1).
  History, maps and products want the components along geographic east and north (bit 5 = 0).
  At a point with rotated longitude lam and rotated latitude phi (radians) on a grid whose rotated
  equator and prime meridian cross at geographic latitude phi0 (tph0), the grid-i direction makes the
  angle alpha with geographic east, with
      cos(alpha) = (cos(phi0) cos(phi) - sin(phi0) sin(phi) cos(lam)) / cos(lat)
      sin(alpha) = -sin(phi0) sin(lam) / cos(lat)
  where lat is the geographic latitude of the point. The map is conformal, so the grid-j direction is
  the grid-i direction turned 90 degrees anticlockwise, and
      u_east  = cos(alpha) u_grid - sin(alpha) v_grid
      v_north = sin(alpha) u_grid + cos(alpha) v_grid.
  The numerators above are the east and north components of d(position)/d(lam) of the map
  namgpu.grid.rotated_to_geographic divided by cos(phi); their length is cos(lat), so the wrapper
  normalises the pair by its length instead of dividing by cos(lat). This is the inverse of the
  earth-to-grid turn of init-gfs-wind-rotate, which takes its vectors from the same map.

  10 m wind: U10 and V10 are H-point fields (the surface layer computes them from the H-point winds),
  so they are turned in place with the H-point angle.
  3-D wind: U and V sit on V points, V(i,j) north-east of H(i,j) (CONTRACT.md section 1). Each H point
  takes the mean of its four V neighbours V(i-1,j-1), V(i,j-1), V(i-1,j), V(i,j) in grid-relative
  components, as the model's own V-to-H averages do (the U_PHY / V_PHY four-point mean of the
  turbulence driver), and is then turned with the H-point angle. V points in the last row and the last
  column are not part of the grid (CONTRACT.md section 1), so an H point on the edge takes the mean of
  the neighbours it has (two, or one at a corner).

Sources
  - The rotated-pole map: namgpu.grid.rotated_to_geographic, the map behind GLAT/GLON and VLAT/VLON.
    It is the rotated latitude-longitude grid of WMO-No. 306, Manual on Codes, Vol. I.2, GRIB2 grid
    definition template 3.1, with the southern pole of the rotated system at (tph0 - 90, tlm0) and no
    rotation angle; NCEP's template 3.32769 (the NAM native files) states the same rotation (note 4:
    the rotated prime meridian and equator cross at the centre of the grid).
  - The angle: spherical trigonometry of that map (the derivation is in the module's test,
    tests/test_io_wind_rotate.py, which checks it against a finite-difference tangent of the map).
  - Component flags: WMO-No. 306, GRIB2 code table 3.3, bit 5.

In:      st.U, st.V   [LM,JM,IM] grid-relative winds on V points
         st.U10, st.V10 [JM,IM] grid-relative 10 m winds on H points
Out:     st.UEARTH, st.VEARTH [LM,JM,IM] earth-relative winds on H points (history scratch)
         st.UE10, st.VE10 [JM,IM] earth-relative 10 m winds on H points (history scratch)
Scratch: the H-point (east, north) vectors of grid i and grid j, four [JM,IM] float32 arrays, built once
         per grid on the host in float64 and cached on ctx.grid (h_coefficients).
Launch:  two kernels on ctx.stream, no host sync.

h_coefficients(ctx) and v_coefficients(ctx) return (ie_e, ie_n, je_e, je_n): the east and north
components of the unit vectors of increasing grid i and of increasing grid j, at H and at V points.
Grid to earth: u_east = u*ie_e + v*je_e, v_north = u*ie_n + v*je_n. Earth to grid (the vectors are
orthonormal): u = ue*ie_e + vn*ie_n, v = ue*je_e + vn*je_n.
"""
from __future__ import annotations

import contextlib

import numpy as np

UNIT = "io_wind_rotate"
KERNEL_3D = "io_wind_rotate_3d"
KERNEL_10M = "io_wind_rotate_10m"


def grid_vectors(tlm, tph, tph0):
    """East and north components of the unit vectors of increasing grid i and grid j.

    tlm, tph: rotated longitude and latitude of the points (radians, any shape); tph0: geographic
    latitude of the centre of the rotated grid (radians). Returns float64 (ie_e, ie_n, je_e, je_n).
    """
    tlm = np.asarray(tlm, dtype=np.float64)
    tph = np.asarray(tph, dtype=np.float64)
    s0, c0 = np.sin(float(tph0)), np.cos(float(tph0))
    e = c0 * np.cos(tph) - s0 * np.sin(tph) * np.cos(tlm)      # cos(lat) cos(alpha)
    n = -s0 * np.sin(tlm)                                       # cos(lat) sin(alpha)
    m = np.hypot(e, n)
    pole = m < 1.0e-12                       # a geographic pole: east and north are undefined there
    m = np.where(pole, 1.0, m)
    ca = np.where(pole, 1.0, e / m)
    sa = np.where(pole, 0.0, n / m)
    return ca, sa, -sa, ca


def _points(g, at: str):
    """Rotated (lon, lat) of the H or V points of the local grid (row slab and nest window included)."""
    jm, im = int(g.jm), int(g.im)
    half = 0.5 if at == "v" else 0.0
    ii = np.arange(im, dtype=np.float64)[None, :] + int(getattr(g, "win_i0", 0))
    jj = np.arange(jm, dtype=np.float64)[:, None] + int(getattr(g, "joff", 0)) + int(getattr(g, "win_j0", 0))
    tlm = np.broadcast_to(g.wb + (ii + half) * g.dlm, (jm, im))
    tph = np.broadcast_to(g.sb + (jj + half) * g.dph, (jm, im))
    return tlm, tph


def host_vectors(g, at: str = "h"):
    """(ie_e, ie_n, je_e, je_n) as float32 [JM,IM] host arrays at H (at='h') or V (at='v') points."""
    if at not in ("h", "v"):
        raise ValueError(f"io-wind-rotate: points {at!r} are not 'h' or 'v'")
    tlm, tph = _points(g, at)
    return tuple(np.ascontiguousarray(a, dtype=np.float32) for a in grid_vectors(tlm, tph, g.tph0))


def _cached(ctx, at: str):
    g = ctx.grid
    key = (at, int(getattr(g, "win_i0", 0)), int(getattr(g, "win_j0", 0)), int(getattr(g, "joff", 0)),
           int(g.im), int(g.jm))
    cache = getattr(g, "_io_wind_vectors", None)
    if cache is None:
        cache = {}
        g._io_wind_vectors = cache
    if key not in cache:
        import cupy as cp
        cache[key] = tuple(cp.asarray(a) for a in host_vectors(g, at))
    return cache[key]


def h_coefficients(ctx):
    """Device (ie_e, ie_n, je_e, je_n) at H points, float32 [JM,IM], cached per grid."""
    return _cached(ctx, "h")


def v_coefficients(ctx):
    """Device (ie_e, ie_n, je_e, je_n) at V points, float32 [JM,IM], cached per grid."""
    return _cached(ctx, "v")


def to_earth_h(u, v, ie_e, ie_n, je_e, je_n, xp=np):
    """Reference of the unit's arithmetic for any array module: grid-relative V-point U, V [..., JM, IM]
    to earth-relative H-point components with the four-point mean (edges: the neighbours that exist)."""
    jm, im = u.shape[-2:]
    acc_u = xp.zeros(u.shape, dtype=xp.float64)
    acc_v = xp.zeros(u.shape, dtype=xp.float64)
    cnt = xp.zeros((jm, im), dtype=xp.float64)
    uu = u[..., : jm - 1, : im - 1].astype(xp.float64)
    vv = v[..., : jm - 1, : im - 1].astype(xp.float64)
    for dj in (0, 1):
        for di in (0, 1):
            # V(i0, j0) is a neighbour of H(i0 + di, j0 + dj)
            acc_u[..., dj: dj + jm - 1, di: di + im - 1] += uu
            acc_v[..., dj: dj + jm - 1, di: di + im - 1] += vv
            cnt[dj: dj + jm - 1, di: di + im - 1] += 1.0
    ub, vb = acc_u / cnt, acc_v / cnt
    return ub * ie_e + vb * je_e, ub * ie_n + vb * je_n


def run(ctx):
    """Write UEARTH/VEARTH and UE10/VE10 from the grid-relative state winds."""
    from namgpu import kernels

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    for need in ("U10", "V10"):
        if need not in st.fields:
            raise RuntimeError(f"io-wind-rotate: {need} missing (the surface layer or the start writes it)")
    ie_e, ie_n, je_e, je_n = h_coefficients(ctx)
    ue = st.add("UEARTH", (lm, jm, im))
    ve = st.add("VEARTH", (lm, jm, im))
    ue10 = st.add("UE10", (jm, im))
    ve10 = st.add("VE10", (jm, im))
    k3 = kernels.get(UNIT, KERNEL_3D)
    k2 = kernels.get(UNIT, KERNEL_10M)
    args = (np.int32(im), np.int32(jm), ie_e, ie_n, je_e, je_n)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k3(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D, (np.int32(lm),) + args + (st.U, st.V, ue, ve))
        k2(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D, args + (st["U10"], st["V10"], ue10, ve10))
