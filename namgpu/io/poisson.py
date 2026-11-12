"""Temperature from potential temperature at a level, and the pressure 2 m above the ground.

Poisson's equation for an adiabatic parcel (Wallace, J. M., and P. V. Hobbs, 2006: Atmospheric Science, 2nd ed.,
Academic Press, Sec. 3.4, potential temperature):

    T = theta * (p / P0) ** KAPPA,   theta = T * (P0 / p) ** KAPPA,   P0 = 1000 hPa,   KAPPA = R_d / c_p

with R_d = 287.04 J kg-1 K-1 (the model's module_CONSTANTS r_d) and c_p = 1004 J kg-1 K-1 for dry air, so
KAPPA = 0.2858964. NCEP's own NAM output pins the exponent: in the native-grid surface files of the 2024-11-05 12Z
cycle (+0 h and +9 h) and the native analysis of 2024-05-06 12Z, the surface temperature record (TMP 0.0.0, level 1)
equals the surface potential temperature record (POT 0.0.2, level 1) times (PRES surface / 1e5) ** k at every one of
the 796,590 points to the files' packing precision for k = 0.2858964 (largest difference 0.0063 K, packing step
0.01 K), and the exponent fitted where the surface pressure is below 860 hPa is 0.285896 (median of about 33,000
points).

The level's pressure is the pressure at the level the potential temperature belongs to:
  skin temperature   : the surface pressure (PSFC, the model's lowest interface pressure);
  2 m temperature    : the 2 m pressure (PSHLTR). NCEP's hourly 2 m maximum and minimum (awphys TMAX/TMIN 103:2)
                       equal the 2 m temperature of the same hour at the points where it is the last sample of the
                       hour only with the 2 m pressure; with the surface pressure they sit 0.02 K off (cycles
                       2024-11-05 12Z and 2024-05-06 12Z, 1-24 h).

pressure_2m is the hypsometric equation over 2 m (Wallace and Hobbs 2006, Sec. 3.2, hypsometric equation) with the
2 m temperature as the layer mean: p2 = psfc * exp(-2 g / (R_d T2)), g = 9.8060226 m s-2 (module_CONSTANTS g).

Every function takes numpy or cupy arrays and returns float32 arrays of the same kind, on the same device.
"""
from __future__ import annotations

import numpy as np

P0 = 1.0e5                 # Pa, reference pressure of potential temperature
R_D = 287.04               # J kg-1 K-1, dry-air gas constant (module_CONSTANTS r_d)
CP = 1004.0                # J kg-1 K-1, dry-air specific heat at constant pressure
KAPPA = R_D / CP           # 0.2858964
G = 9.8060226              # m s-2 (module_CONSTANTS g)
Z_SHELTER = 2.0            # m


def _xp(a):
    """numpy or cupy, whichever holds `a`."""
    mod = type(a).__module__.split(".")[0]
    if mod == "cupy":
        import cupy
        return cupy
    return np


def _f32(xp, a):
    return xp.asarray(a, dtype=xp.float32)


def temperature(theta, p):
    """Temperature (K) of potential temperature `theta` (K) at pressure `p` (Pa)."""
    xp = _xp(theta)
    th, pp = _f32(xp, theta), _f32(xp, p)
    return (th * (pp * xp.float32(1.0 / P0)) ** xp.float32(KAPPA)).astype(xp.float32)


def theta(t, p):
    """Potential temperature (K) of temperature `t` (K) at pressure `p` (Pa)."""
    xp = _xp(t)
    tt, pp = _f32(xp, t), _f32(xp, p)
    return (tt * (xp.float32(P0) / pp) ** xp.float32(KAPPA)).astype(xp.float32)


def pressure_2m(psfc, t2):
    """Pressure (Pa) 2 m above a surface at pressure `psfc` (Pa), with `t2` (K) as the layer temperature."""
    xp = _xp(psfc)
    ps, tt = _f32(xp, psfc), _f32(xp, t2)
    return (ps * xp.exp(xp.float32(-Z_SHELTER * G / R_D) / tt)).astype(xp.float32)
