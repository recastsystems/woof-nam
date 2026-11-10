"""rrtmgp-call: RTE+RRTMGP radiation for the NMMB column, a selectable replacement for the GFS-package RRTM
chain (namgpu.physics.rrtm.rrtm_call). Selected with ``shortwave: rrtmgp`` and ``longwave: rrtmgp`` in the
configfile; the NAM default stays ``rrtm``.

Scheme: ArWen's GPU RTE+RRTMGP (gpuwm.core.rrtmgp.RRTMGPRadiation, Apache-2.0; RTE+RRTMGP is BSD-3, the
k-distribution and cloud-optics tables are rrtmgp-data v1.9, BSD-3; the RFMIP-derived trace-gas climatology is
CC-BY-SA-4.0 with its attribution in the gpuwm-data companion). The scheme code is used as shipped in gpuwm
2.9.13; this module is only the NMMB seam. Not bitwise to anything: NAM never ran RRTMGP.

Same call contract as rrtm_call.run(ctx, args): phys-rad-driver calls it on radiation steps with the Fortran
RRTM argument dict, after rad_zero_outputs and radupdate (args['SOLCON'] is NAM's solar constant with its
eccentricity, used as the TOA irradiance here).

Seam (NMMB l = 1 at the model top, gpuwm k = 1 at the bottom):
  layer p    P_PHY = SGML2*PD + PSGML1                       (module_RA_RRTM.F90:486-487)
  interface  P8W(1) = PT, P8W(l+1) = P8W(l) + PDSG1 + DSG2*PD (module_RA_RRTM.F90:480-485)
  T          T (temperature, K)
  qv         Q/(1-Q): NMMB Q is specific humidity, RRTMGP's gas mapping takes mixing ratio
  clouds     Ferrier (fer, fer_hires): QC liquid, QR rain, QS+QI+QG frozen -> the WRF Ferrier coupling
             (mp_physics 15: ice-active, no separate snow species), cal_cldfra1 cloud fraction (icloud=1)
  surface    TSKIN skin temperature (the RRTM path's TSEA), EPSR emissivity, ALBEDO, SM (xland = 1 + SM), SICE
  geometry   GLAT/GLON (radians) -> degrees
Outputs (state names of the RRTM path, every one written on its spectrum's steps):
  RSWTT, RLWTT  temperature tendencies K s-1 (exner = 1 at the seam, so the theta rate IS dT/dt)
  RSWIN = SW down at the surface, RSWOUT = SW up at the surface, RSWINC = clear-sky SW down at the surface,
  RSWTOA = SW up at TOA, RLWIN = LW down at the surface, RLWTOA = LW up at TOA (OLR)
  CZMEAN  the mean of max(cos z, 0) over the SW interval [t, t + NRADS*DT] (12 samples), the cosine the SW
          fluxes are computed at; RDTEMP and TURBL rescale RSWTT/RSWIN/RSWOUT by CZEN/CZMEAN between calls,
          exactly as for the RRTM chain (GFS coszmn has the same meaning: an interval mean including dark samples)
  SIGT4   STBOLT*T(LM)**4 on LW steps (module_RA_RRTM.F90:1232-1240)
  CLDFRA  the cal_cldfra1 fraction RRTMGP radiated through; CFRACL/M/H: maximum overlap inside the NMMB
          PTOPC pressure bands (RRTM's gethml uses max-random; this is a diagnostic only, no feedback);
          ACFRST/NCFRST, ACFRCV/NCFRCV accumulate the total (1-(1-L)(1-M)(1-H)), ACFRCV where CUPPT > 0.

Refused (named breakage):
  - microphysics other than fer/fer_hires/thompson (TEMPO): their effective radii are not wired into this seam yet; a run with
    them would radiate a different cloud than the microphysics carries.
  - shortwave and longwave must both be rrtmgp: mixing RRTMGP with the RRTM chain would double-write RSWTT/RLWTT.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

KEY = "rrtmgp_call:scheme"
NSAMPLE = 12                     # CZMEAN samples over the SW interval
STBOLT = np.float32(5.67051e-8)  # nmmb_common NMMB_STBOLT
PTOPC_DEFAULT_PA = (105000.0, 64200.0, 35000.0, 0.0)   # used only if rrtm_init has no ptopc
WRF_MP = {"fer": 5, "fer_hires": 15, "thompson": 28}
# Thompson-layout microphysics (NMMB thompson tracers; TEMPO via config.MP_COLUMNS): WRF's Thompson coupling (mp 8/28,
# gpuwm cloud-optics family "thompson"): liquid path QC, ice path QI + QS (graupel carries no radiative mass, as WRF),
# the scheme's own radii RE_CLOUD/RE_ICE/RE_SNOW (metres, phys-tempo-driver) in microns, clamped to Thompson's
# calc_effectRad bounds. Before the first microphysics call (radiation runs first at step 0) the radii are those
# bounds' floors, as WRF's has_reqc fields hold their initial value. 2026-11-09: without this TEMPO
# with RRTMGP refused at the first radiation call.
THOMPSON_RE_UM = {"RE_CLOUD": (2.49, 50.0), "RE_ICE": (4.99, 125.0), "RE_SNOW": (9.99, 999.0)}


def _scheme_class():
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    class NmmbRRTMGP(RRTMGPRadiation):
        """RRTMGP with NAM's own solar constant and the interval-mean cosine set per call."""

        def _cosine_zenith(self, valid_time, *, hour_offset_seconds=0.0, generation="wrf_461"):  # noqa: ARG002
            return self._nmmb_czmean

        def _solar_constant(self, valid_time, *, generation="wrf_461") -> float:  # noqa: ARG002
            return float(self._nmmb_solcon)

    return NmmbRRTMGP


def _start(ctx) -> datetime:
    start = ctx.start if getattr(ctx, "start", None) is not None else getattr(ctx, "start_time", None)
    if start is None:
        jd = ctx.tables["JDAT"]
        start = datetime(int(jd[0]), int(jd[1]), int(jd[2]), int(jd[4]), int(jd[5]), int(jd[6]))
    return start


def _scheme(ctx, lat_deg, lon_deg, lw: bool, sw: bool):
    cache = ctx.tables.setdefault(KEY, {})
    key = (bool(lw), bool(sw))
    obj = cache.get(key)
    if obj is None:
        chunk = int(os.environ.get("NAMGPU_RRTMGP_CHUNK", "0") or 0)
        kw = {"column_chunk": chunk} if chunk > 0 else {}
        obj = _scheme_class()(start_time=_start(ctx), latitude_deg=lat_deg, longitude_deg=lon_deg,
                              longwave=lw, shortwave=sw, **kw)
        cache[key] = obj
    return obj


def _czmean(scheme, valid_time, dtsw):
    """Mean of max(cos z, 0) over [t, t + dtsw], NSAMPLE midpoints, WRF calc_coszen geometry (gpuwm wrf_461)."""
    import cupy as cp
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    acc = None
    for k in range(NSAMPLE):
        off = (k + 0.5) * dtsw / NSAMPLE
        mu = RRTMGPRadiation._cosine_zenith(scheme, valid_time, hour_offset_seconds=off, generation="wrf_461")
        mu = cp.maximum(mu, cp.float32(0.0))
        acc = mu if acc is None else acc + mu
    return (acc / cp.float32(NSAMPLE)).astype(cp.float32)


def run(ctx, args: dict, *, stamp=None, rows=None, sw: bool = True) -> None:  # noqa: ARG001
    import cupy as cp

    cfg, st, g = ctx.cfg, ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    mp = str(cfg.microphysics).strip()
    if mp not in WRF_MP:
        raise NotImplementedError(
            f"rrtmgp-call: microphysics {mp!r} is not wired into the RRTMGP seam (fer/fer_hires only); its "
            "effective radii would be dropped and RRTMGP would radiate a different cloud")
    nts = int(args["NTIMESTEP"])
    dt = float(args["DT"])
    nrads, nradl = int(args["NRADS"]), int(args["NRADL"])
    lsswr = nts % nrads == 0
    lslwr = nts % nradl == 0
    if not (lsswr or lslwr):
        # phys-rad-driver does not zero the outputs before such a call when the scheme is rrtmgp, so the
        # held fields of the last call stay in place (the RRTM chain's call here recomputes the same values)
        return
    dtsw = nrads * dt
    ctx.tables["rrtm_call_flags"] = {"lsswr": lsswr, "lslwr": lslwr, "ntimestep": nts,
                                     "solcon": float(args["SOLCON"]), "sw_run": bool(lsswr), "scheme": "rrtmgp"}

    f32 = cp.float32
    pd = st["PD"]
    sgml2 = cp.asarray(st["SGML2"], dtype=f32).reshape(lm, 1, 1)
    dsg2 = cp.asarray(st["DSG2"], dtype=f32).reshape(lm, 1, 1)
    pdsg1 = cp.asarray(np.asarray(g.pdsg1, np.float32)).reshape(lm, 1, 1)
    psgml1 = cp.asarray(np.asarray(g.psgml1, np.float32)).reshape(lm, 1, 1)
    pt = f32(g.pt)

    # NMMB top-down columns -> gpuwm bottom-up (nz, ny, nx)
    p_lay = (sgml2 * pd[None] + psgml1)[::-1]
    dp = pdsg1 + dsg2 * pd[None]                        # (lm, jm, im), top-down
    p_int_td = cp.empty((lm + 1, jm, im), dtype=f32)
    p_int_td[0] = pt
    p_int_td[1:] = pt + cp.cumsum(dp, axis=0)
    p_int = p_int_td[::-1]
    t = st["T"][::-1]
    q = st["Q"][::-1]
    qv = q / (f32(1.0) - q)
    qc = st["QC"][::-1]
    qr = st["QR"][::-1]
    radii = {}
    if mp == "thompson":
        qi_td, qs_td = st["QI"], st["QS"]
        for name, (lo, hi) in THOMPSON_RE_UM.items():
            arr = st.fields.get(name)
            if arr is None or tuple(arr.shape) != (lm, jm, im):
                v = cp.full((lm, jm, im), lo, dtype=f32)
            else:
                v = cp.clip(arr * f32(1.0e6), f32(lo), f32(hi)).astype(f32)
            radii[{"RE_CLOUD": "effc", "RE_ICE": "effi", "RE_SNOW": "effs"}[name]] = cp.ascontiguousarray(v[::-1])
    # fer_hires carries no QI tracer (drv-tracer-index: QC QR QS QG); add whichever frozen species the state holds
    frozen = None
    for name in ("QS", "QI", "QG"):
        arr = st.fields.get(name)
        if arr is not None and tuple(arr.shape) == (lm, jm, im):
            frozen = arr.copy() if frozen is None else frozen + arr
    if frozen is None:
        raise KeyError("rrtmgp-call: the state has none of QS, QI, QG (Ferrier frozen condensate)")
    qi = frozen[::-1]
    zeros3 = cp.zeros_like(qc)
    qs = zeros3
    if mp == "thompson":
        qi, qs = qi_td[::-1], qs_td[::-1]
    ones3 = cp.ones_like(p_lay)
    c = cp.ascontiguousarray
    atmosphere = {"pressure": c(p_lay), "p_interface": c(p_int), "temperature": c(t), "exner": ones3,
                  "qv": c(qv), "qc": c(qc), "qi": c(qi)}
    xland = f32(1.0) + cp.asarray(st["SM"], dtype=f32)
    fields = {"tsk": c(st["TSKIN"]), "albedo": c(st["ALBEDO"]), "emiss": c(st["EPSR"]),
              "xland": xland, "xice": c(st["SICE"]), "glw": st["RLWIN"]}
    elapsed = float(ctx.ntimestep) * float(ctx.dt or cfg.dt)
    state = SimpleNamespace(qc=atmosphere["qc"], qr=c(qr), qi=atmosphere["qi"], qs=c(qs),
                            p_top=float(pt), elapsed_seconds=elapsed, physics=None, **radii)
    gcfg = SimpleNamespace(mp_physics=WRF_MP[mp], dt=dt, radt=dtsw / 60.0, radt_minutes=dtsw / 60.0,
                           use_adaptive_time_step=True, solar_generation="wrf_461", aercu_opt=0, cu_physics=0,
                           bl_pbl_physics=0, icloud_bl=0, icloud=1, shcu_physics=0, specified=False, nested=False,
                           wrf_rrtmg_compatibility="none")

    lat = cp.rad2deg(cp.asarray(st["GLAT"], dtype=f32))
    lon = cp.rad2deg(cp.asarray(st["GLON"], dtype=f32))
    scheme = _scheme(ctx, lat, lon, lslwr, lsswr)
    valid_time = scheme.start_time + timedelta(seconds=elapsed)
    czmean = _czmean(scheme, valid_time, dtsw)
    scheme._nmmb_czmean = czmean
    scheme._nmmb_solcon = float(args["SOLCON"])

    res = scheme(atmosphere=atmosphere, fields=fields, state=state, cfg=gcfg)
    bf = getattr(res, "boundary_fluxes", None) or {}

    if lsswr:
        st["RSWTT"][...] = res.rthratensw[::-1]
        st["RSWIN"][...] = res.swdown
        st["RSWOUT"][...] = res.swdown - res.gsw
        st["RSWINC"][...] = bf["swdnbc"] if "swdnbc" in bf else res.swdown
        st["RSWTOA"][...] = bf["swupt"]
        st["CZMEAN"][...] = czmean
    if lslwr:
        st["RLWTT"][...] = res.rthratenlw[::-1]
        st["RLWIN"][...] = res.glw
        st["RLWTOA"][...] = res.olr
        tl = st["T"][lm - 1]
        st["SIGT4"][...] = STBOLT * tl * tl * tl * tl
    if not lsswr:
        # rad_zero_outputs zeroed CZMEAN on this LW-only step; the RRTM chain rewrites it on every call
        # (grrad coszmn), so this does too.
        st["CZMEAN"][...] = czmean

    # Cloud diagnostics
    cld_td = res.cldfra[::-1]
    st["CLDFRA"][...] = cld_td
    ptopc = None
    tbl = ctx.tables.get("rrtm_init")
    if tbl is not None and "ptopc" in tbl:
        ptopc = cp.asnumpy(tbl["ptopc"]).astype(np.float64)
        if not np.all(np.isfinite(ptopc)) or ptopc.max() <= 0:
            ptopc = None
    if ptopc is None:
        ptopc = np.asarray(PTOPC_DEFAULT_PA)
    p_td = p_lay[::-1]
    bands = []
    for lo, hi in ((ptopc[1], 2.0e5), (ptopc[2], ptopc[1]), (-1.0, ptopc[2])):
        m = (p_td > f32(lo)) & (p_td <= f32(hi))
        bands.append(cp.max(cp.where(m, cld_td, f32(0.0)), axis=0))
    st["CFRACL"][...], st["CFRACM"][...], st["CFRACH"][...] = bands
    tot = f32(1.0) - (f32(1.0) - bands[0]) * (f32(1.0) - bands[1]) * (f32(1.0) - bands[2])
    st["ACFRST"][...] += tot
    st["NCFRST"][...] += 1
    st["ACFRCV"][...] += cp.where(st["CUPPT"] > f32(0.0), tot, f32(0.0))
    st["NCFRCV"][...] += 1
    if stamp is not None:
        stamp("rrtmgp-done")
