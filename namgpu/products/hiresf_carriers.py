"""The NAM nest (hiresf) record carriers on the native NMMB grid, from one history file.

Each carrier takes the open history Dataset (and the run's physics step) and returns a [j, i] float32 array or None
when the history lacks an input (the record is then left out and named in the run index, never filled). The 3 h
averages and accumulations come from NMMB's own accumulators, which the model empties every nhrs_* hours
(drv_bucket_reset: NCEP's 3 h bucket; 1 h for the fire-weather nest), so at forecast hour F they hold the bucket that
ends at F, the window NOAA's hiresf files carry (0-1, 0-2, 0-3, 3-4, ...; firewx 0-1, 1-2, 2-3, ...).

What each record is, and what it is built from:

- Bucket averages (GRIB2 statistical process 0, average; WMO-No. 306, code table 4.10). The model adds the
  instantaneous flux to an accumulator at every turbulence call and counts the calls (ARDSW, ARDLW, ASRFC), so the
  average over the bucket is the accumulator over its count. The accumulators keep the model's signs: downward
  positive at the surface (ASWIN, ALWIN, SFCSHX, SFCLHX), outgoing negative at the surface (ASWOUT, ALWOUT), outgoing
  positive at the top (ASWTOA, ALWTOA). Units are those of the NCEP GRIB2 tables (discipline 0 categories 0, 1, 4,
  5 and 19; discipline 1 category 1; discipline 2 category 0); the tables give no sign, and the signs are the ones
  NCEP's records carry: each radiation flux is the magnitude in its named direction, the sensible and latent heat
  fluxes are positive from the surface to the air, the ground heat flux keeps the sign of the model's soil heat
  flux (positive towards the surface), and the snow phase-change heat flux is negative where snow melts.
- Bucket accumulations (statistical process 1), kg m-2: evaporation and potential evaporation are accumulated by the
  model as metres of liquid water (the latent heat flux over L_v rho_w, L_v = 2.501e6 J kg-1, rho_w = 1000 kg m-3),
  so x rho_w; snow melt, surface and subsurface runoff and snowfall (ACSNOM, SSROFF, BGROFF, ACSNOW) are accumulated
  in kg m-2 (Noah's SNOMLT, RUNOFF1 and RUNOFF2 + RUNOFF3, and FFROZP PRCP, times the step).
- PRATE (kg m-2 s-1): the model's precipitation of the last physics step PREC (m of water, emptied at each
  turbulence call) over that step DT NPHS, x rho_w.
- CPOFP (%): the percent of that precipitation that is frozen, 100 SR (SR the model's frozen fraction of the surface
  precipitation, from its microphysics), where the last physics step precipitated; NO_PRECIP elsewhere, the value
  NCEP's files carry where there is no precipitation.
- The hourly maxima (statistical process 2): the model's hourly maximum precipitation rate and frozen precipitation
  rate (mm h-1) / 3600 s h-1 as kg m-2 s-1, and the hourly maximum reflectivity at the -10 C level (dBZ).
- Instantaneous fields carried as the model holds them: surface and 10 m specific humidity, 10 m potential
  temperature, the top-of-atmosphere outgoing longwave of the last radiation call, the surface exchange
  coefficient, and the model's low, middle and high cloud fractions (x 100 for %).
- Snow-free and maximum snow albedo (%): the land surface's snow-free albedo and the maximum albedo over deep snow
  (Noah LSM User's Guide 2.7.1, section 5.0, control-file lines 19 and 21), the model's ALBASE and MXSNAL
  (fractions) x 100 on land; over open water and sea ice the constant values NCEP's records carry there
  (WATER_ALB; ICE_SNFALB, ICE_MXSALB).
- WTMP (K): the water temperature SST over open water; elsewhere the surface temperature, as NCEP's records carry
  it over land and sea ice, from the skin potential temperature THS by Poisson's equation T = THS (p_s / p_0)^(R/c_p),
  p_0 = 1000 hPa (Wallace, J. M., and P. V. Hobbs, 2006: Atmospheric Science, 2nd ed., potential temperature),
  R/c_p from the model's constants.

Sources: the model's accumulators and counters (NMMB solver state, drv_bucket_reset) and its Noah land surface
(Ek, M. B., et al., 2003: J. Geophys. Res., 108(D22), 8851; Mitchell, K., 2005: The community Noah land-surface
model (LSM) User's Guide, public release version 2.7.1, NCEP/EMC); WMO-No. 306, Manual on Codes, Vol. I.2, GRIB2
code tables 4.2 and 4.10; the NCEP GRIB2 parameter tables (nco.ncep.noaa.gov/pmb/docs/grib2, tables 4.2-0-0, 4.2-0-1,
4.2-0-19, 4.2-1-1, 4.2-2-0); NCEP's archived NAM output for the choices the definitions leave open (named above).
"""
from __future__ import annotations

import numpy as np

RHOWATER = 1000.0          # kg m-3
SEC_PER_HOUR = 3600.0
CAPPA = 287.04 / 1004.6    # R_d / c_p (NMMB module_CONSTANTS, as namgpu/kernels/nmmb_common.cuh)
P0 = 1.0e5                 # Pa, the reference pressure of the potential temperature
NO_PRECIP = -50.0          # CPOFP where the last physics step did not precipitate (NCEP's hiresf records)
WATER_ALB = 6.0            # snow-free and maximum snow albedo over open water, % (NCEP's records)
ICE_SNFALB = 65.0          # snow-free albedo over sea ice, % (NCEP's records)
ICE_MXSALB = 60.0          # maximum snow albedo over sea ice, % (NCEP's records)


def _get(s, name):
    """Return history variable `name` at its first time as a float32 [j, i] array, or None when the history lacks it."""
    if name not in s.variables:
        return None
    return np.asarray(s[name][0], dtype=np.float32)


def _ave(acc, cnt, sign=1.0):
    """Return a carrier (s, ctx) giving the bucket average of history accumulator `acc` (sample count `cnt`), times
    `sign`. A point whose count is 0 (an empty bucket, at the start) reads 0."""
    def f(s, ctx=None):
        a, n = _get(s, acc), _get(s, cnt)
        if a is None or n is None:
            return None
        ok = n > 0.0
        out = np.where(ok, np.float32(sign) * a / np.where(ok, n, np.float32(1.0)), np.float32(0.0))
        return out.astype(np.float32)
    return f


def _scaled(name, k=1.0):
    """Return a carrier (s, ctx) giving history field `name` in the record's units (factor `k`), or None when absent."""
    def f(s, ctx=None):
        a = _get(s, name)
        return None if a is None else (a * np.float32(k)).astype(np.float32)
    return f


def _dtphs(ctx):
    """The physics step DT NPHS (s) from the run context, or None."""
    if not ctx:
        return None
    if ctx.get("dtq2"):
        return float(ctx["dtq2"])
    if ctx.get("dt") and ctx.get("nphs"):
        return float(ctx["dt"]) * float(ctx["nphs"])
    return None


def prate(s, ctx):
    """Return the instantaneous precipitation rate PRATE (kg m-2 s-1) as a float32 [j, i] array, or None."""
    p, dt = _get(s, "PREC"), _dtphs(ctx)
    if p is None or not dt:
        return None
    return (np.maximum(p, np.float32(0.0)) * np.float32(RHOWATER / dt)).astype(np.float32)


def cpofp(s, ctx):
    """Return the percent of frozen precipitation CPOFP (%) as a float32 [j, i] array, or None."""
    p, sr = _get(s, "PREC"), _get(s, "SR")
    if p is None or sr is None:
        return None
    return np.where(p > 0.0, np.float32(100.0) * np.clip(sr, 0.0, 1.0), np.float32(NO_PRECIP)).astype(np.float32)


def _surface(s):
    """(open water, sea ice) [j, i] booleans: SM is the model's water mask, SICE its sea-ice flag; None without SM."""
    sm = _get(s, "SM")
    if sm is None:
        return None
    sice = _get(s, "SICE")
    water = sm > 0.5
    ice = np.zeros_like(water) if sice is None else (~water) & (sice > 0.5)
    return water, ice


def _albedo(name, ice_value):
    """Carrier: land albedo field `name` (fraction) x 100; WATER_ALB over open water, `ice_value` over sea ice."""
    def f(s, ctx=None):
        a, m = _get(s, name), _surface(s)
        if a is None or m is None:
            return None
        water, ice = m
        out = np.where(water, np.float32(WATER_ALB), np.where(ice, np.float32(ice_value), np.float32(100.0) * a))
        return out.astype(np.float32)
    return f


def wtmp(s, ctx=None):
    """Return WTMP (K): SST over open water, the surface temperature THS (PSFC / P0)^CAPPA elsewhere; or None."""
    sst, ths, ps, m = _get(s, "SST"), _get(s, "THS"), _get(s, "PSFC"), _surface(s)
    if sst is None or ths is None or ps is None or m is None:
        return None
    tsk = ths.astype(np.float64) * (ps.astype(np.float64) / P0) ** CAPPA
    return np.where(m[0], sst, tsk).astype(np.float32)


# frame name -> (function(history Dataset, ctx), regrid mode "nearest" | "budget")
CARRIERS = {
    # 3 h bucket averages (W m-2) and accumulations (kg m-2)
    "NAMH_DSWRF_AVE": (_ave("ASWIN", "ARDSW"), "nearest"),
    "NAMH_USWRF_AVE": (_ave("ASWOUT", "ARDSW", -1.0), "nearest"),
    "NAMH_DLWRF_AVE": (_ave("ALWIN", "ARDLW"), "nearest"),
    "NAMH_ULWRF_AVE": (_ave("ALWOUT", "ARDLW", -1.0), "nearest"),
    "NAMH_USWRF_TOA_AVE": (_ave("ASWTOA", "ARDSW"), "nearest"),
    "NAMH_ULWRF_TOA_AVE": (_ave("ALWTOA", "ARDLW"), "nearest"),
    "NAMH_SHTFL_AVE": (_ave("SFCSHX", "ASRFC", -1.0), "nearest"),
    "NAMH_LHTFL_AVE": (_ave("SFCLHX", "ASRFC", -1.0), "nearest"),
    "NAMH_GFLUX_AVE": (_ave("SUBSHX", "ASRFC"), "nearest"),
    "NAMH_SNOHF_AVE": (_ave("SNOPCX", "ASRFC"), "nearest"),
    "NAMH_EVP_ACC": (_scaled("SFCEVP", RHOWATER), "nearest"),
    "NAMH_PEVAP_ACC": (_scaled("POTEVP", RHOWATER), "nearest"),
    "NAMH_SNOM_ACC": (_scaled("ACSNOM"), "nearest"),
    "NAMH_SSRUN_ACC": (_scaled("SSROFF"), "nearest"),
    "NAMH_BGRUN_ACC": (_scaled("BGROFF"), "nearest"),
    "NAMH_WEASD_ACC": (_scaled("ACSNOW"), "budget"),
    # instantaneous
    "NAMH_SPFH_SFC": (_scaled("QSH"), "nearest"),
    "NAMH_POT_10M": (_scaled("TH10"), "nearest"),
    "NAMH_SPFH_10M": (_scaled("Q10"), "nearest"),
    "NAMH_PRATE": (prate, "nearest"),
    "NAMH_CPOFP": (cpofp, "nearest"),
    "NAMH_SNFALB": (_albedo("ALBASE", ICE_SNFALB), "nearest"),
    "NAMH_MXSALB": (_albedo("MXSNAL", ICE_MXSALB), "nearest"),
    "NAMH_WTMP": (wtmp, "nearest"),
    "NAMH_SFEXC": (_scaled("SFCEXC"), "nearest"),
    "NAMH_ULWRF_TOA": (_scaled("RLWTOA"), "nearest"),
    # hourly maxima kept by io-max-fields (kg m-2 s-1; dBZ)
    "NAMH_PRATE_MAX": (_scaled("PRATEMAX", 1.0 / SEC_PER_HOUR), "nearest"),
    "NAMH_SRWEQ_MAX": (_scaled("FPRATEMAX", 1.0 / SEC_PER_HOUR), "nearest"),
    "NAMH_REFD263K_MAX": (_scaled("REFDMAX263K"), "nearest"),
    # the model's own low/middle/high cloud fractions (%)
    "NAMH_LCDC": (_scaled("CFRACL", 100.0), "nearest"),
    "NAMH_MCDC": (_scaled("CFRACM", 100.0), "nearest"),
    "NAMH_HCDC": (_scaled("CFRACH", 100.0), "nearest"),
}


def compute(history: str, name: str, ctx: dict | None = None):
    """Return (native-grid float32 [j, i] array or None, regrid mode) for carrier `name` of one history file."""
    import netCDF4
    fn, mode = CARRIERS[name]
    with netCDF4.Dataset(history) as s:
        s.set_auto_mask(False)
        return fn(s, ctx or {}), mode
