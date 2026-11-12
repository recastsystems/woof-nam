"""MSLET: membrane sea-level pressure from a NMMB history state.

What it computes
    The sea-level pressure of the "horizontal" (membrane) reduction. On each isobaric surface that cuts the ground,
    the virtual temperature below the ground is filled by solving Laplace's equation, with the virtual temperature
    of the air around the mountain on the same surface as the boundary condition. The hydrostatic equation is then
    integrated down from the ground through those temperatures to the height where it is zero. The field is the
    awphys and hiresf MSLET record (the writer derives the 1000 hPa height from it) and the MSLP maps' field.

    from_history(nc) -> float32 [south_north, west_east], Pa, on the history's own grid (H points); None when the
    history lacks T, QVAPOR, PRES, PINT or HGT.

Method and sources
    Mesinger, F., 1990: "Horizontal" pressure reduction to sea level. Proc. 21st Int. Conf. on Alpine Meteorology,
        Zurich: the method, as Office Note 394 states it.
    Treadon, R. E., 1993: The NMC Eta model post processor: a documentation. NMC Office Note 394 (revised by
        H.-Y. Chuang and G. Manikin, NCEP, 2001):
      sec. 4.1    isobaric surfaces every 25 hPa from 50 to 1000 hPa; temperature and humidity linear in ln p;
                  between the lowest model layer and the ground, the lowest layer's values; below the lowest
                  model layer, values from the second and third layers above the ground, which keeps the
                  surface layer's own signature out.
      eq. (4.1)   dz / d(-ln p) = Rd Tv / g, Tv = T (1 + 0.608 q), q the specific humidity.
      eq. (4.2)   Laplace's equation for Tv at the points of a surface that lie below the ground; the air on the
                  same surface around the mountain gives the boundary condition.
      eq. (4.3)   its relaxation: each below-ground point takes the nine-point average of its neighbours (weight 4
                  on the four nearest, 1 on the four diagonal ones); only below-ground values are replaced.
      sec. 4.2.1  the 6.5 K/km standard lapse rate.
      sec. 4.2.2  for a terrain-following model the reduction is made on constant-pressure surfaces, and the
                  sea-level pressure is found where the height is zero.
    The model's own in-run membrane field (NMMB nmm/module_QUASIPOST.F90, model source): the same steps on isobaric
        surfaces; the first guess below the ground is the temperature (not the virtual temperature) carried down the
        standard lapse rate, and the relaxation takes 100 passes.

    The relaxation is not run to convergence: the field is the first guess smoothed by 100 passes, as the model's
    membrane field is. The first guess, its source layers and the pass count were checked against NCEP's own awphys
    MSLET (cycle 2024-11-05 12Z, NAM analysis start).

    Constants: Rd = 287.04 J kg-1 K-1 and g = 9.8060226 m s-2, the model's own (module_CONSTANTS), so the
    integration matches the history's own hydrostatic heights.

Array module: CuPy when it sees a device (NAMGPU_MEMSLP_XP=auto, the default), NumPy otherwise or with
NAMGPU_MEMSLP_XP=numpy. The result is always a NumPy array.
"""
from __future__ import annotations

import os

import numpy as np

RD = 287.04                # J kg-1 K-1 (NMMB module_CONSTANTS R_D)
G = 9.8060226              # m s-2 (NMMB module_CONSTANTS G)
CAPPA = 287.04 / 1004.6    # R_D / CP of the model: history T is theta - 300 with P00 = 1e5 Pa
CAPPA_NMMB = CAPPA         # the same, by the name tests that build a history use
P00 = 1.0e5
D608 = 0.608               # Tv = T (1 + 0.608 q), Office Note 394 eq. (4.1)
GAMMA = 6.5e-3             # K m-1, standard lapse rate (Office Note 394 sec. 4.2.1)

# Isobaric surfaces, Pa, every 25 hPa from 50 to 1000 hPa (Office Note 394 sec. 4.1). Only those that cut the
# ground somewhere in the domain take part.
LEVELS = np.arange(5000.0, 100000.0 + 1.0, 2500.0)
PASSES = 100               # relaxation passes of eq. (4.3) (the model's membrane field)

NEEDS = ("T", "QVAPOR", "PRES", "PINT", "HGT")


def xp_for(name: str | None = None):
    """CuPy when it sees a device (auto), else NumPy; 'numpy' or 'cupy' forces one."""
    name = (name or os.environ.get("NAMGPU_MEMSLP_XP", "auto")).lower()
    if name == "numpy":
        return np
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        if name == "cupy":
            raise
    return np


def _read(nc, name, top=None):
    v = nc.variables[name]
    a = v[0, :top] if top is not None else (v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:])
    return np.asarray(np.ma.getdata(a), dtype=np.float32)


def from_history(nc, xp=None, passes: int | None = None):
    """MSLET (Pa, float32 [south_north, west_east]) of an open history (netCDF4.Dataset), or None when it lacks
    an input. `passes` overrides the number of relaxation passes."""
    if any(n not in nc.variables for n in NEEDS):
        return None
    xp = xp_for() if xp is None else xp
    hgt = _read(nc, "HGT")
    ps = _read(nc, "PINT", 1)[0]
    # the layers that reach the highest surface in use, plus one above it, are all the column needs
    p = nc.variables["PRES"]
    top = levels_for(ps)
    lm = p.shape[1]
    nk = lm
    if top:
        for k in range(lm):
            if float(np.max(np.ma.getdata(p[0, k]))) < top[0]:
                nk = min(k + 1, lm)
                break
    nk = max(nk, 3)
    ins = [_read(nc, n, nk) for n in ("T", "QVAPOR", "PRES")]
    return membrane_slp(xp, *ins, ps, hgt, passes=PASSES if passes is None else passes)


def levels_for(ps):
    """The isobaric surfaces (Pa, ascending) that lie below the ground somewhere: higher than the lowest ground
    pressure."""
    return [float(v) for v in LEVELS if v > float(np.min(ps))]


def _to_np(xp, a):
    return a.get() if xp is not np else a


def virtual_temperature(xp, theta, w, p):
    """Tv (K) from history T (theta - 300), the vapour mixing ratio and the pressure (eq. 4.1)."""
    t = (theta + np.float32(300.0)) * (p * np.float32(1.0 / P00)) ** np.float32(CAPPA)
    q = w / (np.float32(1.0) + w)
    return t * (np.float32(1.0) + np.float32(D608) * q)


def level_tv(xp, tv, lnp_mid, lnps, levels):
    """Tv on each isobaric surface [L, JM, IM] (float64) and the below-ground mask. Linear in ln p between
    mid-layers; the lowest layer's Tv between the lowest mid-layer and the ground (sec. 4.1). Below the ground:
    NaN."""
    lm = tv.shape[0]
    out = xp.empty((len(levels),) + tuple(lnps.shape), dtype=np.float64)
    under = xp.empty(out.shape, dtype=bool)
    for n, v in enumerate(levels):
        lnlev = np.float32(np.log(v))
        below = xp.zeros(lnps.shape, dtype=np.int32)
        for k in range(lm):
            below += lnp_mid[k] > lnlev                    # mid-layers with higher pressure than the surface
        k_hi = xp.clip(below, 1, lm - 1)[None]             # the layer above the surface
        k_lo = k_hi - 1                                    # the layer below it
        lo = xp.take_along_axis(lnp_mid, k_lo, 0)[0].astype(np.float64)
        hi = xp.take_along_axis(lnp_mid, k_hi, 0)[0].astype(np.float64)
        tlo = xp.take_along_axis(tv, k_lo, 0)[0].astype(np.float64)
        thi = xp.take_along_axis(tv, k_hi, 0)[0].astype(np.float64)
        t = tlo + (float(lnlev) - lo) / (hi - lo) * (thi - tlo)
        t = xp.where(below == 0, tv[0].astype(np.float64), t)   # between the lowest mid-layer and the ground
        u = np.log(v) > lnps
        out[n] = xp.where(u, xp.nan, t)
        under[n] = u
    return out, under


def first_guess(xp, tvl, under, levels, t23, lnp23):
    """Below-ground first guess: the temperature of the second and third layers above the ground (their mean,
    at the mean ln p of their mid-layers; sec. 4.1) carried down the standard lapse rate (sec. 4.2.1), in pressure
    T(p) = T23 (p / p23)^(Rd gamma / g)."""
    for n, v in enumerate(levels):
        g = t23 * xp.exp((np.log(v) - lnp23) * (RD * GAMMA / G))
        tvl[n] = xp.where(under[n], g, tvl[n])
    return tvl


def _neighbour_sum(xp, a):
    """4 x (four nearest) + (four diagonal) neighbours of each point of a [JM, IM] array; outside the array counts
    as zero (the eq. 4.3 stencil on the B grid's H points)."""
    z = xp.pad(a, 1)
    near = z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:]
    diag = z[:-2, :-2] + z[:-2, 2:] + z[2:, :-2] + z[2:, 2:]
    return 4.0 * near + diag


def relax(xp, field, under, passes):
    """Eq. (4.2) by eq. (4.3): `passes` nine-point averaging passes on each surface; only the below-ground points
    (under) take the new value. At the domain edge the average is over the neighbours that exist. field:
    [L, JM, IM] with the first guess in its below-ground points; changed in place and returned.

    Each surface is worked on the box that holds its below-ground points and one ring of neighbours around them:
    the ring is above the ground (fixed) or the domain edge, so the result is that of the whole surface."""
    jm, im = field.shape[1:]
    wsum = _neighbour_sum(xp, xp.ones((jm, im), dtype=field.dtype))
    for k in range(field.shape[0]):
        u = under[k]
        rows = xp.nonzero(u.any(axis=1))[0]
        if rows.size == 0 or passes <= 0:
            continue
        cols = xp.nonzero(u.any(axis=0))[0]
        j0, j1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, jm)
        i0, i1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, im)
        ub = u[j0:j1, i0:i1]
        wb = wsum[j0:j1, i0:i1]
        work = field[k, j0:j1, i0:i1]
        # inside the box a neighbour outside it is outside the domain or beside an above-ground ring point, which
        # is never updated: zero padding is right for every point that is
        for _ in range(passes):
            work = xp.where(ub, _neighbour_sum(xp, work) / wb, work)
        field[k, j0:j1, i0:i1] = work
    return field


def integrate(xp, tvl, under, levels, tv0, lnps, z0):
    """ln of sea-level pressure: eq. (4.1) integrated down from the ground (height z0, ln p lnps) through the
    surfaces below it, each layer's Tv the mean of its two ends; sea level where the height is zero (sec. 4.2.2),
    ln p linear in height inside the layer (constant Tv). Ground at or below sea level: up through the lowest
    layer's Tv. Still above sea level under the last surface: the standard lapse rate from there."""
    z = z0.copy()
    lnp = lnps.copy()
    tvp = tv0.copy()
    done = z0 <= 0.0
    lnpmsl = xp.where(done, lnps + G * z0 / (RD * tv0), 0.0)
    for k, v in enumerate(levels):
        lnv = np.log(v)
        go = under[k] & ~done
        tbar = 0.5 * (tvp + tvl[k])
        znew = z - RD / G * tbar * (lnv - lnp)
        cross = go & (znew <= 0.0)
        frac = z / xp.where(cross, z - znew, 1.0)
        lnpmsl = xp.where(cross, lnp + frac * (lnv - lnp), lnpmsl)
        done = done | cross
        z = xp.where(go, znew, z)
        lnp = xp.where(go, lnv, lnp)
        tvp = xp.where(go, tvl[k], tvp)
    tbar = tvp + 0.5 * GAMMA * z
    return xp.where(done, lnpmsl, lnp + G * z / (RD * tbar))


def prepare(xp, theta, w, p, ps, hgt):
    """Device inputs: (levels, Tv on them with NaN below the ground, below-ground mask, lowest-layer Tv,
    first-guess temperature and its ln p (second and third layers), ln surface pressure, terrain height)."""
    levels = levels_for(ps)
    theta, w, p = xp.asarray(theta), xp.asarray(w), xp.asarray(p)
    tv = virtual_temperature(xp, theta, w, p)
    lnp_mid = xp.log(p)
    lnps = xp.log(xp.asarray(ps, dtype=np.float64))
    tv0 = tv[0].astype(np.float64)
    t = (theta[1:3] + np.float32(300.0)) * (p[1:3] * np.float32(1.0 / P00)) ** np.float32(CAPPA)
    t23 = t.astype(np.float64).mean(axis=0)
    lnp23 = lnp_mid[1:3].astype(np.float64).mean(axis=0)
    tvl, under = level_tv(xp, tv, lnp_mid, lnps, levels)
    return levels, tvl, under, tv0, t23, lnp23, lnps, xp.asarray(hgt, dtype=np.float64)


def membrane_slp(xp, theta, w, p, ps, hgt, passes=PASSES):
    """MSLET (Pa, NumPy float32 [JM, IM]) from NumPy inputs: theta - 300, vapour mixing ratio and pressure on the
    lowest mid-layers [K, JM, IM] (level 0 at the ground; enough layers to reach above the highest surface in
    use), surface pressure [JM, IM] and terrain height [JM, IM]."""
    levels, tvl, under, tv0, t23, lnp23, lnps, z0 = prepare(xp, theta, w, p, ps, hgt)
    if levels:
        tvl = relax(xp, first_guess(xp, tvl, under, levels, t23, lnp23), under, passes)
    lnpmsl = integrate(xp, tvl, under, levels, tv0, lnps, z0)
    return _to_np(xp, xp.exp(lnpmsl)).astype(np.float32)
