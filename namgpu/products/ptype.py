"""NCEP-style dominant precipitation type for WOOF-NAM (awphys/hiresf CRAIN/CSNOW/CICEP/CFRZR), on the GPU.

The dominant type is a vote over several algorithms run on the model's own profile plus the microphysics' frozen
fraction (Manikin 2005, "An overview of precipitation type forecasting using NAM and SREF data", AMS 21st WAF/17th
NWP, 8A.6).

  ncep          Baldwin, Treadon & Contorno (1994) as described by Manikin (2005) and Jeffrey (2018, UMD MS thesis,
                sec. 1.3): no ice in the column (coldest saturated level warmer than 269 K) -> ZR/RA by the lowest
                temperature; snow when the wet-bulb warm area is under 350 K m and the area between Tw and -4 C is
                not over 3000 K m; ice pellets when the wet-bulb cold area in the lowest 150 hPa is under -2500 K m
                and the surface-based warm area is under 350 K m; else ZR/RA by the surface wet-bulb.
  ncep-revised  the same, with the -4 C check replaced by "no snow when the Tw > 0 area exceeds 500 K m"
                (Manikin 2005: the revised NCEP scheme, biased toward snow).
  ramer         Ramer (1993) as described by Jeffrey (2018) and Manikin (2005): surface Tw >= 2 C -> RA; no Tw > 0
                in the column -> SN; else an ice fraction I from the generating level (the highest layer below
                400 hPa with RH > 90 % over >= 16 hPa; I = 1 if its Tw < -6.6 C, else 0), melted in Tw > 0 layers
                and refrozen in Tw < 0 layers (only where ice is present or Tw < -6.6 C) at E = 0.045 K per unit
                ln p (the same energy as Bourgouin's melting threshold, 13.2 J/kg / Rd); I >= 0.85 frozen (SN, or
                IP if it had melted below 0.85 on the way), I <= 0.04 liquid (ZR/RA by surface Tw), between: IP
                (Manikin 2005: the undecided case "assigns ice pellets").
  bourgouin     Bourgouin (2000, Wea. Forecasting 15, 583-592) energy areas on T (the namwinter code, moved here).
  explicit      the microphysics: frozen when the step's snow ratio SR >= 0.5 at a point with precipitating
                hydrometeors at the lowest level (abstains elsewhere); frozen is IP where graupel (TEMPO's frozen
                drops) outweighs snow there and the column has a Tw > 0 layer, else SN; liquid is ZR where
                T2 <= 0 C, else RA.
  dominant      the majority over the five (an abstaining algorithm has no vote); ties go to the first in
                TIE_ORDER (ZR, IP, SN, RA: the more hazardous type, which NCEP's scheme favours by design).
  explicit-sr   the type WOOF-NAM's winter scoring used before this module (2026-11-09 winter-scoring lane):
                SR >= 0.5 -> IP where Bourgouin says IP/ZR else SN; liquid -> ZR where T2 < 0 C else RA.
                Kept selectable by name.

Mask: a type is given only where the model precipitates (Manikin 2005: "a precipitation type solution is only
generated when the model generates non-zero precipitation at a grid point"). `rate` (default): the last physics
step's rate PREC / (DT NPHS) >= MASK_RATE_MM_H = 0.01 mm/h, the value that reproduces NCEP's own awphys records:
on one history hour of a run from the NAM's own analysis (2024-11-05 12Z, +3 h) the typed share of grid 218 is
20.93 % against the archived NAM's 21.30 % (overlap 0.879; snow over land 13.13 against 13.57 %); at the 0.05 mm/h
this module used before it was 15.62 % (overlap 0.704; snow over land 7.17 %), and with the literal non-zero test
34.39 % (overlap 0.601). NAMGPU_PTYPE_RATE_MM_H sets another rate: a suite whose microphysics leaves sub-trace
precipitation over very large areas (TEMPO: with the non-zero test the 7.6 km nam-modern runs typed precipitation
at 2.4x the ASOS precipitating hours; ptype-winter lane, 2026-11-09) may want 0.05. `prec`: PREC > 0, the literal
non-zero test.
`prec4e-6`: PREC >= 4e-6 m per step or the hour >= 0.25 mm, the winter-scoring lane's mask: 0.48 mm/h at the
7.6 km step (DT NPHS = 30 s), so it dropped almost every light-snow hour, the cause of the "snow forecast half as
often as observed" score. Kept by name.

Classes: 0 none, 1 RA, 2 SN, 3 ZR, 4 IP. Array module: cupy when it imports and sees a device (NAMGPU_PTYPE_XP=
numpy|cupy|auto, default auto), else numpy; every column is independent, so the whole grid is one vectorised pass.
"""
from __future__ import annotations

import os

import numpy as np

NONE, RA, SN, ZR, IP = 0, 1, 2, 3, 4
NAMES = {NONE: "none", RA: "RA", SN: "SN", ZR: "ZR", IP: "IP"}
T0 = 273.15
RD, CP, G, LV, EPS = 287.04, 1004.6, 9.80665, 2.501e6, 0.62197
TIE_ORDER = (ZR, IP, SN, RA)
ALGORITHMS = ("ncep", "ncep-revised", "ramer", "bourgouin", "explicit")
METHODS = ("dominant",) + ALGORITHMS + ("explicit-sr",)
DEFAULT_METHOD = "dominant"
MASKS = ("rate", "prec", "prec4e-6")
PTOP_PA = 40000.0                  # every profile algorithm looks below 400 hPa (Ramer's generating-level search)
MASK_PREC_M = 0.0
MASK_RATE_MM_H = float(os.environ.get("NAMGPU_PTYPE_RATE_MM_H", "0.01"))
SR_FROZEN = 0.5
HYD_MIN = 1.0e-7                   # kg/kg of rain+snow+graupel at the lowest level for the explicit vote
# Baldwin et al. (1994) / revised NCEP thresholds (K m)
B_COLD_SAT_K, B_WARM_SNOW, B_M4_SNOW, B_REV_WARM, B_COLD150, B_SFC_WARM = 269.15, 350.0, 3000.0, 500.0, -2500.0, 350.0
# Ramer (1993) constants
R_SFC_RAIN_C, R_ICE_TW_C, R_RH, R_DEPTH_PA, R_E, R_FROZEN, R_LIQUID = 2.0, -6.6, 0.90, 1600.0, 0.045, 0.85, 0.04


def xp_for(name: str | None = None):
    name = (name or os.environ.get("NAMGPU_PTYPE_XP", "auto")).lower()
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


def _es(xp, t):
    return 611.2 * xp.exp(17.67 * (t - T0) / (t - 29.65))


def _dewpoint(xp, p, w):
    e = xp.maximum(w * p / (EPS + w), 1e-3)
    a = xp.log(e / 611.2)
    return T0 + 243.5 * a / (17.67 - a)


def wetbulb(xp, t, p, w):
    """Wet-bulb temperature (K) from T (K), p (Pa), water-vapour mixing ratio w (kg/kg): Newton on the
    psychrometric balance cp (T - Tw) = Lv (ws(Tw) - w), w capped at saturation."""
    t = t.astype(xp.float64)
    p = p.astype(xp.float64)
    es_t = _es(xp, t)
    ws_t = EPS * es_t / xp.maximum(p - es_t, 1.0)
    w = xp.clip(w.astype(xp.float64), 0.0, ws_t)
    tw = t - 0.33 * (t - xp.minimum(t, _dewpoint(xp, p, w)))
    for _ in range(6):
        es = _es(xp, tw)
        den = xp.maximum(p - es, 1.0)
        ws = EPS * es / den
        dws = ws * (17.67 * 243.5 / (tw - 29.65) ** 2) * (p / den)
        tw = tw - (tw - t + (LV / CP) * (ws - w)) / (1.0 + (LV / CP) * dws)
    return xp.minimum(tw, t)


def _rh(xp, t, p, w):
    es = _es(xp, t)
    return w / xp.maximum(EPS * es / xp.maximum(p - es, 1.0), 1e-12)


def _column(xp, f):
    """Profiles [L+1, N] bottom-up, the surface first (T2/Q2/PSFC), model levels above it up to PTOP_PA (NaN past),
    plus wet-bulb and RH. f: T (K) [L, N], P (Pa) [L, N], QV (mixing ratio) [L, N], T2, Q2, PSFC [N]."""
    T = xp.concatenate([f["T2"][None], f["T"]], 0).astype(xp.float64)
    P = xp.concatenate([f["PSFC"][None], f["P"]], 0).astype(xp.float64)
    W = xp.concatenate([f["Q2"][None], f["QV"]], 0).astype(xp.float64)
    keep = (P >= PTOP_PA) & (P <= P[0:1] + 1.0)
    keep[0] = True
    Tn, Pn, Wn = xp.where(keep, T, 250.0), xp.where(keep, P, 5.0e4), xp.where(keep, W, 0.0)
    TW = xp.where(keep, wetbulb(xp, Tn, Pn, Wn), xp.nan)
    RH = xp.where(keep, _rh(xp, Tn, Pn, Wn), xp.nan)
    return xp.where(keep, T, xp.nan), xp.where(keep, P, xp.nan), Wn, TW, RH, keep


def _segments(xp, X, P, Tv, ref):
    """Area of (X - ref) over each layer, split at the crossing, in K m (dz hypsometric): lo/hi parts per layer
    [L, N] (attributed to the lower/upper level's side), and the valid-layer mask."""
    a = X[:-1] - ref
    b = X[1:] - ref
    with np.errstate(invalid="ignore", divide="ignore"):
        dz = RD * 0.5 * (Tv[:-1] + Tv[1:]) / G * xp.log(P[:-1] / P[1:])
    ok = xp.isfinite(a) & xp.isfinite(b) & xp.isfinite(dz) & (dz > 0)
    a = xp.where(ok, a, 0.0)
    b = xp.where(ok, b, 0.0)
    dz = xp.where(ok, dz, 0.0)
    cross = (a > 0) != (b > 0)
    den = xp.where(cross, a - b, 1.0)
    fr = xp.where(cross, a / xp.where(den == 0, 1e-9, den), 1.0)
    lo = xp.where(cross, 0.5 * a * fr * dz, 0.5 * (a + b) * dz)
    hi = xp.where(cross, 0.5 * b * (1 - fr) * dz, 0.0)
    return lo, hi, ok


def _pos(xp, lo, hi):
    return xp.where(lo > 0, lo, 0.0) + xp.where(hi > 0, hi, 0.0)


def _neg(xp, lo, hi):
    return xp.where(lo < 0, lo, 0.0) + xp.where(hi < 0, hi, 0.0)


def ncep(xp, col, revised=False):
    T, P, W, TW, RH, keep = col
    Tv = xp.nan_to_num(T, nan=250.0) * (1 + 0.608 * W)
    sat = keep & (xp.nan_to_num(RH) >= R_RH)
    sat[0] = False
    tsat = xp.where(sat, xp.nan_to_num(T, nan=999.0), 999.0).min(0)
    no_ice = xp.any(sat, 0) & (tsat > B_COLD_SAT_K)
    lo, hi, ok = _segments(xp, TW, P, Tv, T0)
    warm = _pos(xp, lo, hi).sum(0)
    lo4, hi4, _ = _segments(xp, TW, P, Tv, B_COLD_SAT_K)
    m4 = _pos(xp, lo4, hi4).sum(0)
    within = xp.nan_to_num(P[1:], nan=0.0) >= (P[0:1] - 15000.0)
    cold150 = xp.where(within & ok, _neg(xp, lo, hi), 0.0).sum(0)
    sgn = xp.nan_to_num(TW, nan=0.0) > T0
    seen_cold = xp.cumsum(~sgn, 0) > 0                       # at or above the first level at or below 0 C
    sfc_warm = xp.where(~seen_cold[:-1], xp.where(lo > 0, lo, 0.0), 0.0).sum(0)
    snow = (warm < B_REV_WARM) if revised else ((warm < B_WARM_SNOW) & (m4 <= B_M4_SNOW))
    ip = (cold150 < B_COLD150) & (sfc_warm < B_SFC_WARM)
    liq = xp.where(TW[0] <= T0, ZR, RA)
    out = xp.where(snow, SN, xp.where(ip, IP, liq))
    out = xp.where(no_ice, xp.where(T[0] <= T0, ZR, RA), out)
    return out.astype(xp.int8)


def ramer(xp, col):
    T, P, W, TW, RH, keep = col
    L1, N = T.shape
    twc = xp.nan_to_num(TW - T0, nan=-30.0)
    twsfc = twc[0]
    anywarm = xp.any(xp.where(keep, twc, -1.0) > 0, 0)
    moist = keep & (xp.nan_to_num(RH) > R_RH)
    moist[0] = False
    Pn = xp.nan_to_num(P, nan=0.0)
    gen = xp.full(N, -1, xp.int64)
    run_bot = xp.zeros(N)
    for k in range(1, L1):                                   # bottom-up: each moist run's bottom pressure
        start = moist[k] & ~moist[k - 1] if k > 1 else moist[k]
        run_bot = xp.where(start, Pn[k], run_bot)
        gen = xp.where(moist[k] & ((run_bot - Pn[k]) >= R_DEPTH_PA), k, gen)   # the highest deep run wins
    lev = xp.arange(L1)[:, None]
    hi_moist = xp.where(moist, lev, -1).max(0)
    gen = xp.where(gen < 0, hi_moist, gen)
    gen = xp.where(gen < 0, xp.argmax(xp.where(keep, xp.nan_to_num(RH), -1.0), 0), gen)
    tw_gen = xp.take_along_axis(twc, gen[None], 0)[0]
    I = xp.where(tw_gen < R_ICE_TW_C, 1.0, 0.0)
    Imin = I.copy()
    for k in range(L1 - 1, 0, -1):                           # descend layer (k-1, k) below the generating level
        act = (k <= gen) & keep[k] & keep[k - 1]
        tm = 0.5 * (twc[k] + twc[k - 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            dl = xp.log(Pn[k - 1] / xp.where(Pn[k] > 0, Pn[k], 1.0))
        dl = xp.where(act & xp.isfinite(dl) & (dl > 0), dl, 0.0)
        frz = (tm < 0) & ((I > 0) | (tm < R_ICE_TW_C))
        I = xp.where(tm > 0, xp.maximum(0.0, I - tm * dl / R_E), xp.where(frz, xp.minimum(1.0, I - tm * dl / R_E), I))
        Imin = xp.minimum(Imin, I)
    frozen = xp.where(Imin >= R_FROZEN, SN, IP)
    liq = xp.where(twsfc > 0, RA, ZR)
    out = xp.where(I >= R_FROZEN, frozen, xp.where(I <= R_LIQUID, liq, IP))
    out = xp.where(~anywarm, SN, out)
    out = xp.where(twsfc >= R_SFC_RAIN_C, RA, out)
    return out.astype(xp.int8)


def bourgouin(xp, col, return_pa=False):
    """Bourgouin (2000) on T (not Tw): areas Rd * integral (T - T0) d ln p (J/kg), a layer crossing 0 C split at the
    crossing. Mixed categories go to the nearer side of the middle of his band (RA/SN at 9.4 J/kg; ZR/IP at
    NA = 56 + 0.66 PA). Same decisions as namwinter.bourgouin (winter-scoring lane)."""
    T, P = col[0], col[1]
    L, N = T.shape
    a = T[:-1] - T0
    b = T[1:] - T0
    with np.errstate(invalid="ignore", divide="ignore"):
        lnr = xp.log(P[:-1] / P[1:])
    valid = xp.isfinite(a) & xp.isfinite(b) & xp.isfinite(lnr) & (lnr > 0)
    a = xp.where(valid, a, -1.0)
    b = xp.where(valid, b, -1.0)
    lnr = xp.where(valid, lnr, 0.0)
    cross = (a > 0) != (b > 0)
    den = xp.where(cross, a - b, 1.0)
    fr = xp.where(cross, a / xp.where(den == 0, 1e-9, den), 1.0)
    part_lo = xp.where(cross, 0.5 * a * fr * lnr, 0.5 * (a + b) * lnr) * RD
    part_hi = xp.where(cross, 0.5 * b * (1 - fr) * lnr, 0.0) * RD
    sign = xp.where(xp.isfinite(T), T > T0, False)
    seg = xp.concatenate([xp.zeros((1, N), xp.int64), xp.cumsum(sign[1:] != sign[:-1], 0)], 0)
    NS = 6
    s0 = xp.minimum(seg[:-1], NS - 1)
    s1 = xp.minimum(seg[1:], NS - 1)
    area = xp.stack([xp.where(s0 == s, part_lo, 0.0).sum(0) + xp.where(s1 == s, part_hi, 0.0).sum(0)
                     for s in range(NS)])
    a0, a1, a2 = area[0], area[1], area[2]
    sfc_warm = sign[0]
    out = xp.full(N, SN, xp.int8)
    pa_aloft = xp.zeros(N)

    def warm_sfc(pa_s, frozen_cls):
        return xp.where(pa_s > 9.4, RA, frozen_cls)
    mA = sfc_warm & ~(area[2:] > 0).any(0)
    out = xp.where(mA, warm_sfc(a0, SN), out)
    mB = ~sfc_warm & (a1 > 0)
    PA, NA = a1, -a0
    out = xp.where(mB, xp.where(PA < 2.0, SN, xp.where(NA > 56.0 + 0.66 * PA, IP, ZR)), out)
    pa_aloft = xp.where(mB, PA, pa_aloft)
    mC = sfc_warm & (a1 < 0) & (a2 > 0)
    PA, NA = a2, -a1
    clsC = xp.where(PA < 2.0, warm_sfc(a0, SN), xp.where(NA > 56.0 + 0.66 * PA, warm_sfc(a0, IP), RA))
    out = xp.where(mC, clsC, out).astype(xp.int8)
    pa_aloft = xp.where(mC, PA, pa_aloft)
    return (out, pa_aloft) if return_pa else out


def explicit(xp, col, f):
    """-1 (abstain) where the microphysics has no precipitating hydrometeors at the lowest level or no SR."""
    TW = col[3]
    N = TW.shape[1]
    sr, qr, qs, qg = (f.get(k) for k in ("SR", "QRAIN0", "QSNOW0", "QGRAUP0"))
    if sr is None or qr is None or qs is None:
        return xp.full(N, -1, xp.int8)
    qg = xp.zeros_like(qs) if qg is None else qg
    have = (qr + qs + qg) > HYD_MIN
    frozen = xp.clip(sr, 0, 1) >= SR_FROZEN
    warm_aloft = xp.any(xp.nan_to_num(TW[1:], nan=0.0) > T0, 0)
    fcls = xp.where((qg > qs) & warm_aloft, IP, SN)
    lcls = xp.where(f["T2"] <= T0, ZR, RA)
    return xp.where(have, xp.where(frozen, fcls, lcls), -1).astype(xp.int8)


def explicit_sr(xp, col, f, bcls=None):
    sr = f.get("SR")
    if sr is None:
        return xp.full(col[0].shape[1], -1, xp.int8)
    bcls = bourgouin(xp, col) if bcls is None else bcls
    frozen = xp.clip(sr, 0, 1) >= SR_FROZEN
    return xp.where(frozen, xp.where((bcls == IP) | (bcls == ZR), IP, SN),
                    xp.where(f["T2"] < T0, ZR, RA)).astype(xp.int8)


def dominant(xp, votes):
    """votes: list of [N] classes (-1 abstains): the class with the most votes, ties to the first in TIE_ORDER;
    -1 where nobody voted."""
    V = xp.stack(votes)
    counts = {c: (V == c).sum(0) for c in (RA, SN, ZR, IP)}
    best = xp.maximum(xp.maximum(counts[RA], counts[SN]), xp.maximum(counts[ZR], counts[IP]))
    out = xp.full(V.shape[1], -1, xp.int8)
    for c in reversed(TIE_ORDER):                                   # the first in TIE_ORDER is written last
        out = xp.where((counts[c] == best) & (best > 0), c, out)
    return out.astype(xp.int8)


CHUNK = int(os.environ.get("NAMGPU_PTYPE_CHUNK", 262144))


def classify_all(f, xp=None, chunk: int | None = None):
    """Every method, unmasked, for a dict of arrays (see _column; plus SR, QRAIN0, QSNOW0, QGRAUP0 for the explicit
    votes). Inputs may be numpy; outputs are numpy int8 [N] per method name (0 where nothing voted). Columns go in
    chunks of NAMGPU_PTYPE_CHUNK (262,144: ~3 GB of float64 temporaries on the device), each one vectorised pass."""
    xp = xp or xp_for()
    N = np.asarray(f["T2"]).shape[-1]
    chunk = chunk or CHUNK
    if N > chunk:
        parts = []
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            sub = {k: (None if v is None else (v[..., a:b] if np.ndim(v) else v)) for k, v in f.items()}
            parts.append(_classify(sub, xp))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return _classify(f, xp)


def _classify(f, xp):
    g = {k: (None if v is None else xp.asarray(v)) for k, v in f.items()}
    col = _column(xp, g)
    res = {"ncep": ncep(xp, col), "ncep-revised": ncep(xp, col, revised=True), "ramer": ramer(xp, col),
           "bourgouin": bourgouin(xp, col), "explicit": explicit(xp, col, g)}
    res["explicit-sr"] = explicit_sr(xp, col, g, res["bourgouin"])
    res["dominant"] = dominant(xp, [res[a] for a in ALGORITHMS])
    out = {}
    for k, v in res.items():
        v = xp.where(v < 0, NONE, v).astype(xp.int8)
        out[k] = v.get() if hasattr(v, "get") else np.asarray(v)
    return out


def mask(f, name="rate", hourly_mm=None, dtq2=None):
    """Where a type is given: `rate` PREC / dtq2 >= MASK_RATE_MM_H (dtq2 = DT * NPHS, s); `prec` PREC > 0;
    `prec4e-6` the winter-scoring lane's mask (PREC >= 4e-6 m, or the hour >= 0.25 mm when hourly_mm is given)."""
    prec = np.asarray(f["PREC"])
    if name == "rate":
        if not dtq2:
            raise ValueError("ptype mask rate needs the physics step DT*NPHS (dtq2)")
        return prec * (1000.0 * 3600.0 / float(dtq2)) >= MASK_RATE_MM_H
    if name == "prec":
        return prec > MASK_PREC_M
    if name == "prec4e-6":
        m = prec >= 4.0e-6
        return m | (np.asarray(hourly_mm) >= 0.25) if hourly_mm is not None else m
    raise ValueError(f"ptype mask {name!r}: one of {', '.join(MASKS)}")


def levels_below(P_sample, ptop=PTOP_PA):
    """How many bottom-up model levels any sampled column needs to reach ptop (plus one): P_sample [L, ...]."""
    P = np.asarray(P_sample).reshape(np.asarray(P_sample).shape[0], -1)
    return int((P.max(1) >= ptop - 5000.0).sum()) + 1


def from_history(ds, levels: int | None = None):
    """The classify_all inputs from an open woof-nam history Dataset (wrfout names, bottom-up levels): T is NMMB's
    perturbation theta (T + 300 K) on PRES; QVAPOR and Q2 are mixing ratios."""
    def g(k, sl=None):
        if k not in ds.variables:
            return None
        v = ds[k]
        return np.asarray(v[0] if sl is None else v[0, sl], np.float32)
    K = slice(0, levels) if levels else slice(None)
    P = g("PRES", K)
    th = g("T", K)
    L = P.shape[0]
    f = {"P": P.reshape(L, -1), "T": ((th + 300.0) * (P / 1.0e5) ** 0.28571).reshape(L, -1),
         "QV": g("QVAPOR", K).reshape(L, -1)}
    for k in ("T2", "Q2", "PSFC", "SR", "PREC"):
        a = g(k)
        f[k] = None if a is None else a.reshape(-1)
    for k in ("QRAIN", "QSNOW", "QGRAUP"):
        a = g(k, slice(0, 1))
        f[k + "0"] = None if a is None else a.reshape(-1)
    return f, P.shape[1:]


NEED = ("T", "PRES", "QVAPOR", "T2", "Q2", "PSFC", "PREC")


def dataset_ptype(ds, method: str | None = None, mask_name: str | None = None):
    """[j, i] int8 type classes (0 none, 1 RA, 2 SN, 3 ZR, 4 IP) for one open history Dataset, masked, or None when
    the history lacks an input: the awphys/hiresf precipitation-type product. NAMGPU_PTYPE picks the method
    (default dominant), NAMGPU_PTYPE_MASK the mask (default rate, 0.01 mm/h as NCEP's files)."""
    method = method or os.environ.get("NAMGPU_PTYPE", DEFAULT_METHOD)
    mask_name = mask_name or os.environ.get("NAMGPU_PTYPE_MASK", "rate")
    if method not in METHODS:
        raise ValueError(f"NAMGPU_PTYPE={method!r}: one of {', '.join(METHODS)}")
    if mask_name not in MASKS:
        raise ValueError(f"NAMGPU_PTYPE_MASK={mask_name!r}: one of {', '.join(MASKS)}")
    if any(k not in ds.variables for k in NEED):
        return None
    dtq2 = run_dtq2(ds) if mask_name == "rate" else None
    if mask_name == "rate" and not dtq2:
        # no configfile beside the history: the rate cannot be formed, so the record is left out (named in the run
        # index) rather than typed under another mask
        print(f"ptype: {ds.filepath()}: no nam_configfile beside the history (DT*NPHS unknown); type left out", flush=True)
        return None
    ds.set_auto_mask(False)
    f, shp = from_history(ds, levels=levels_below(ds["PRES"][0, :, ::40, ::40]))
    cls = classify_all(f)[method]
    return np.where(mask(f, mask_name, dtq2=dtq2), cls, NONE).astype(np.int8).reshape(shp)


def run_dtq2(ds):
    """DT * NPHS (s) of the run that wrote this history: NAMGPU_PTYPE_DTQ2, else the run folder's nam_configfile (the
    same source woof-nam products uses for PRATE)."""
    v = os.environ.get("NAMGPU_PTYPE_DTQ2")
    if v:
        return float(v)
    from pathlib import Path
    from namgpu.io.products import _run_ctx
    return _run_ctx(Path(ds.filepath()).parent).get("dtq2")


def history_ptype(path, method: str | None = None, mask_name: str | None = None):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return dataset_ptype(ds, method, mask_name)


# the four NCEP categorical records, one carrier each (awphys recs 336-339, hiresf CSNOW/CICEP/CFRZR/CRAIN): the
# classification runs once per history file per process and the four flags read it
_LAST = {}


def flag(ds, cls_code):
    key = (ds.filepath(), os.environ.get("NAMGPU_PTYPE", DEFAULT_METHOD), os.environ.get("NAMGPU_PTYPE_MASK", "rate"))
    if key not in _LAST:
        _LAST.clear()
        _LAST[key] = dataset_ptype(ds)
    c = _LAST[key]
    return None if c is None else (c == cls_code).astype(np.float32)
