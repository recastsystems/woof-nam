"""The NAM nest (hiresf) record carriers: NCEP post formulas on the native NMMB grid, from one history file.

Each function takes the open history Dataset (and the run's physics step) and returns a [j, i] array or None when
the history lacks an input (the record is then left out and named in the run index, never filled). The 3 h
averages and accumulations read NMMB's own accumulators, which the model empties every nhrs_* hours
(drv_bucket_reset: NCEP's 3 h bucket; 1 h for the fire-weather nest), so at forecast hour F they hold the bucket that
ends at F, exactly the window NOAA's hiresf files carry (0-1, 0-2, 0-3, 3-4, ...; firewx 0-1, 1-2, 2-3, ...). Averages are the accumulator over its counter (UPP SURFCE.f /
CLDRAD.f: x * RRNUM with RRNUM = 1 / count). Sign conventions are NMMB's (TWBS/QWBS and their sums positive down).
"""
from __future__ import annotations

import numpy as np


def _get(s, name):
    if name not in s.variables:
        return None
    return np.asarray(s[name][0], dtype=np.float32)


def _ave(acc, cnt, sign=1.0):
    def f(s, ctx):
        a, n = _get(s, acc), _get(s, cnt)
        if a is None or n is None:
            return None
        return np.where(n > 0.0, np.float32(sign) * a / np.where(n > 0.0, n, 1.0), 0.0).astype(np.float32)
    return f


def _scaled(name, k=1.0):
    def f(s, ctx):
        a = _get(s, name)
        return None if a is None else (a * np.float32(k)).astype(np.float32)
    return f


def prate(s, ctx):
    """SURFCE.f instantaneous precipitation rate (kg m-2 s-1): PREC (m, the last physics step) * 1000 / DTQ2,
    DTQ2 = DT * NPHS."""
    p = _get(s, "PREC")
    dtq2 = (ctx or {}).get("dtq2")
    if p is None or not dtq2:
        return None
    return (p * np.float32(1000.0 / float(dtq2))).astype(np.float32)


def cpofp(s, ctx):
    """SURFCE.f percent of frozen precipitation: SR * 100 where the step precipitated, -50 elsewhere."""
    p, sr = _get(s, "PREC"), _get(s, "SR")
    if p is None or sr is None:
        return None
    return np.where(p > 0.0, sr * np.float32(100.0), np.float32(-50.0)).astype(np.float32)


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
    "NAMH_EVP_ACC": (_scaled("SFCEVP", 1000.0), "nearest"),
    "NAMH_PEVAP_ACC": (_scaled("POTEVP", 1000.0), "nearest"),
    "NAMH_SNOM_ACC": (_scaled("ACSNOM", 1000.0), "nearest"),
    "NAMH_SSRUN_ACC": (_scaled("SSROFF"), "nearest"),
    "NAMH_BGRUN_ACC": (_scaled("BGROFF"), "nearest"),
    "NAMH_WEASD_ACC": (_scaled("ACSNOW", 1000.0), "budget"),
    # instantaneous
    "NAMH_SPFH_SFC": (_scaled("QSH"), "nearest"),
    "NAMH_POT_10M": (_scaled("TH10"), "nearest"),
    "NAMH_SPFH_10M": (_scaled("Q10"), "nearest"),
    "NAMH_PRATE": (prate, "nearest"),
    "NAMH_CPOFP": (cpofp, "nearest"),
    "NAMH_SNFALB": (_scaled("ALBASE", 100.0), "nearest"),
    "NAMH_MXSALB": (_scaled("MXSNAL", 100.0), "nearest"),
    "NAMH_WTMP": (_scaled("SST"), "nearest"),
    "NAMH_SFEXC": (_scaled("SFCEXC"), "nearest"),
    "NAMH_ULWRF_TOA": (_scaled("RLWTOA"), "nearest"),
    # hourly maxima kept by io-max-fields in mm/h (MAX_FIELDS_HR: 1000 PREC / (DT NPHS) * 3600): kg m-2 s-1
    "NAMH_PRATE_MAX": (_scaled("PRATEMAX", 1.0 / 3600.0), "nearest"),
    "NAMH_SRWEQ_MAX": (_scaled("FPRATEMAX", 1.0 / 3600.0), "nearest"),
    "NAMH_REFD263K_MAX": (_scaled("REFDMAX263K"), "nearest"),
    # the model's own low/middle/high cloud fractions (%)
    "NAMH_LCDC": (_scaled("CFRACL", 100.0), "nearest"),
    "NAMH_MCDC": (_scaled("CFRACM", 100.0), "nearest"),
    "NAMH_HCDC": (_scaled("CFRACH", 100.0), "nearest"),
}


def compute(history: str, name: str, ctx: dict | None = None):
    """Worker side: (native-grid array or None, regrid mode) for one carrier of one history file."""
    import netCDF4
    fn, mode = CARRIERS[name]
    with netCDF4.Dataset(history) as s:
        return fn(s, ctx), mode
