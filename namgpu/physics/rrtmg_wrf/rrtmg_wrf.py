"""rad-rrtmg: WRF RRTMG longwave and shortwave as the NMMB radiation, with WRF's paired cloud fraction.

Modernisation lane (2026-11-08). Selected by ``shortwave: rrtmg`` / ``longwave: rrtmg`` in the
configfile (parm/nam_configfile_rrtmg), or by NAMGPU_RADIATION=rrtmg over any configfile. Off by default:
a bare run keeps the frozen NAM radiation (NMMB GFS-lineage RRTM, progcld2 cloud).

What runs: ArWen's GPU port of WRF v4.6.1 RRTMG (gpuwm.core.rrtmg_legacy, AER RRTMG_LW v4.85 / RRTMG_SW v3.9,
BSD-3-Clause via AER's 2016 relicence; WRF lineage, public domain driver), McICA maximum-random overlap
(cldovrlp 2), CAM ozone climatology (o3input 2), fixed greenhouse gases (ghg_input 0), Tegen climatological aerosol (aer_opt 1,
the default; aerosol-aware aer_opt 3 is off; 0 and 3 are knobs: NAMGPU_RRTMG_AER_OPT).

Cloud fraction: WRF's icloud=1 (cal_cldfra1, Xu-Randall) on the Ferrier-Aligo package arm that WRF pairs with
mp_physics 5 (Registry fer_mp_hires: moist qv, qc, qr, qi with qi the total ice; no qs). NMMB fer_hires holds
the total ice in QS (= F_ICE*CWM, drv_update_water), so QS is handed to the adapter as qi. Radii: Ferrier is
outside WRF's use_mp_re disjunction, so the wrapper's own relcalc/reicalc radii are used, as WRF does.

Column map (NMMB l = 1 is the TOP; WRF k = 1 is the BOTTOM): every 3-D field is flipped on the device. Layer
pressure = mean of the two PINT interfaces; dz from the hypsometric equation on virtual temperature; the
interface heights start at FIS/g. qv = Q/(1-Q) (NMMB Q is specific humidity, WRF qv a mixing ratio).

Outputs written into the NMMB fields the old RRTM wrote (so the rest of the step is unchanged):
  RLWTT, RSWTT  = theta tendency * Exner (K s-1, NMMB's sensible-temperature tendency), flipped back
  RLWIN = LW down at the surface, RSWIN = SW down, RSWOUT = SW up, RSWINC = clear-sky SW down,
  RLWTOA = OLR, RSWTOA = SW up at the top, CZMEAN = cos zenith at the interval midpoint (WRF's choice,
  NMMB's rdtemp scales the SW heating by CZEN/CZMEAN between calls), SIGT4 = STBOLT*T(LM)^4,
  CLDFRA = the icloud=1 cloud fraction, CFRACL/M/H = maximum overlap within the 642 / 350 hPa bands,
  ACFRST += total (random between bands), NCFRST, NCFRCV += 1.

Named gap: BMJ convective cloud does not reach this radiation (NMMB's CU_cloud from CUPPT/HTOP/HBOT; WRF's
bmj_rad_feedback carrier is not produced by the NMMB BMJ port). ACFRCV is left unchanged.
"""
from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

KEY = "rrtmg_wrf:adapter"
STBOLT = 5.67051e-8
R_D = 287.04
CP_D = 1004.6
GRAV = 9.80616
P0 = 1.0e5
LOW_TOP = 64200.0
MID_TOP = 35000.0


def selected(cfg) -> bool:
    env = os.environ.get("NAMGPU_RADIATION", "").strip().lower()
    if env:
        return env == "rrtmg"
    return str(cfg.shortwave).strip() == "rrtmg" or str(cfg.longwave).strip() == "rrtmg"


def _aer_opt() -> int:
    # Default 1 (Tegen monthly aerosol climatology, not aerosol-aware): the frozen NAM radiation already carries a
    # climatological aerosol (phys_rrtm_clim_aerinit), and the 2026-01-23 12 h arms put aer_opt 1 closer to ASOS
    # (T2 bias +0.8 F at +6/+12 h vs +0.9/+1.0 with no aerosol, same MAE). 0 (no aerosol) and 3 stay knobs.
    return int(os.environ.get("NAMGPU_RRTMG_AER_OPT", "1"))


def _wrf_cfg(ctx):
    c = ctx.cfg
    radt = float(c.nrads) * float(ctx.dt or c.dt) / 60.0
    return SimpleNamespace(
        mp_physics=5, icloud=1, use_mp_re=1, cldovrlp=2, idcor=0, ghg_input=0, aer_opt=_aer_opt(),
        o3input=2, rrtmg_aer3_type_source="rural", rrtmg_smoke_manifest="", simple_dir_fdb=0,
        sf_surface_physics=2, bl_pbl_physics=2, icloud_bl=0, bl_mynn_version=0, cu_physics=0,
        bmj_rad_feedback=False, shcu_physics=0, aercu_opt=0, aer_ra_feedback=0, use_theta_m=0,
        nssl_2moment_on=1, solar_generation="wrf_480", rrtmg_cloud_optics_form="wrf_461",
        aer_type=1, rrtmg_aer3_source="climatology", radt=radt, radt_minutes=radt,
        dt=float(ctx.dt or c.dt))


def _adapter(ctx):
    ad = ctx.tables.get(KEY)
    if ad is None:
        from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
        st = ctx.st
        lat = np.degrees(np.asarray(st["GLAT"].get(), np.float64)).astype(np.float32)
        lon = np.degrees(np.asarray(st["GLON"].get(), np.float64)).astype(np.float32)
        start = ctx.start if ctx.start is not None else ctx.start_time
        ad = RRTMGLegacyRadiation(start, lat, lon, p_top=float(ctx.grid.pt), o3input=2,
                                  aer_opt=_aer_opt())
        ctx.tables[KEY] = ad
    return ad


def _weights(ctx, lm):
    """WRF fnm/fnp (bottom-up, length nz) from a reference column's layer pressure depths."""
    key = "rrtmg_wrf:fnmfnp"
    w = ctx.tables.get(key)
    if w is None:
        import cupy as cp
        g = ctx.grid
        pdref = float(cp.median(ctx.st["PD"]).get())
        dp = (np.asarray(g.pdsg1, np.float64) + np.asarray(ctx.st["DSG2"].get(), np.float64).reshape(-1) * pdref)
        dnw = dp[::-1].copy()                     # bottom-up layer depths
        fnm = np.zeros(lm, np.float32)
        fnp = np.zeros(lm, np.float32)
        for k in range(1, lm):
            dn = 0.5 * (dnw[k] + dnw[k - 1])
            fnm[k] = 0.5 * dnw[k - 1] / dn
            fnp[k] = 0.5 * dnw[k] / dn
        w = (cp.asarray(fnm), cp.asarray(fnp))
        ctx.tables[key] = w
    return w


def run(ctx, lw_step: bool = True) -> None:
    import cupy as cp
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    f32 = cp.float32

    def bu(a):                         # NMMB top-down [LM, JM, IM] -> WRF bottom-up, contiguous
        return cp.ascontiguousarray(a[::-1])

    pint = st["PINT"]                  # [LM+1, JM, IM], 0 = top
    t = st["T"]
    q = st["Q"]
    p_mid = 0.5 * (pint[:-1] + pint[1:])
    qv = q / (1.0 - q)
    tv = t * (1.0 + 0.608 * q)
    dz = (R_D / GRAV) * tv * cp.log(pint[1:] / pint[:-1])      # [LM] top-down, positive
    dz_bu = bu(dz)
    zw = cp.empty((lm + 1, jm, im), dtype=f32)
    zw[0] = st["FIS"] / GRAV
    zw[1:] = zw[0][None] + cp.cumsum(dz_bu, axis=0)
    pressure = bu(p_mid).astype(f32)
    atmosphere = {
        "pressure": pressure,
        "p_interface": bu(pint).astype(f32),
        "temperature": bu(t).astype(f32),
        "exner": (pressure / P0) ** f32(R_D / CP_D),
        "dz": dz_bu.astype(f32),
        "z_interface": zw,
    }
    fields = {
        "tsk": st["TSKIN"], "emiss": st["EPSR"], "albedo": st["ALBEDO"],
        "xland": (1.0 + st["SM"]).astype(f32), "xice": st["SICE"],
        "snow": (st["SNO"] * 1000.0).astype(f32),
    }
    fnm, fnp = _weights(ctx, lm)
    start = ctx.start if ctx.start is not None else ctx.start_time
    valid = start + timedelta(seconds=float(ctx.ntimestep) * float(ctx.dt or ctx.cfg.dt))
    state = SimpleNamespace(
        qv=bu(qv).astype(f32), qc=bu(st["QC"]), qr=bu(st["QR"]), qi=bu(st["QS"]), qs=None, qg=None,
        p_top=float(g.pt), elapsed_seconds=float((valid - start).total_seconds()),
        fnm=fnm, fnp=fnp, ht=st["FIS"] / GRAV, physics=None, chem=None, _registered_mass_slots=(),
        alt=None)
    if _aer_opt() == 1:
        # aer_opt 1 (Tegen climatology): the monthly field is built once per forecast, not once per call.
        state.aerodm = ctx.tables.get("rrtmg_wrf:aerodm")
    ad = _adapter(ctx)
    if not ctx.tables.get("rrtmg_wrf:logged"):
        print(f"rad-rrtmg: WRF RRTMG LW/SW (gpuwm.core.rrtmg_legacy) is the radiation; icloud=1, cldovrlp=2 McICA, "
              f"o3input=2, aer_opt={_aer_opt()}, every {int(ctx.cfg.nrads)} steps", flush=True)
        ctx.tables["rrtmg_wrf:logged"] = True
    res = ad(atmosphere=atmosphere, fields=fields, state=state, cfg=_wrf_cfg(ctx))
    if _aer_opt() == 1 and getattr(state, "aerodm", None) is not None:
        ctx.tables["rrtmg_wrf:aerodm"] = state.aerodm
    exner = atmosphere["exner"]
    st["RLWTT"][...] = (res.rthratenlw * exner)[::-1]
    st["RSWTT"][...] = (res.rthratensw * exner)[::-1]
    bf = res.boundary_fluxes
    st["RLWIN"][...] = bf["lwdnb"]
    st["RLWTOA"][...] = bf["lwupt"]
    st["RSWIN"][...] = bf["swdnb"]
    st["RSWOUT"][...] = bf["swupb"]
    st["RSWINC"][...] = bf["swdnbc"]
    st["RSWTOA"][...] = bf["swupt"]
    st["CZMEAN"][...] = cp.maximum(res.coszen, f32(0.0))
    if lw_step:
        tl = t[lm - 1]
        st["SIGT4"][...] = f32(STBOLT) * tl * tl * tl * tl
    cf = res.cldfra[::-1]              # top-down [LM, JM, IM]
    st["CLDFRA"][...] = cf
    zero = f32(0.0)
    low = cp.where(p_mid >= LOW_TOP, cf, zero).max(axis=0)
    mid = cp.where((p_mid < LOW_TOP) & (p_mid >= MID_TOP), cf, zero).max(axis=0)
    high = cp.where(p_mid < MID_TOP, cf, zero).max(axis=0)
    st["CFRACL"][...] = low
    st["CFRACM"][...] = mid
    st["CFRACH"][...] = high
    total = 1.0 - (1.0 - low) * (1.0 - mid) * (1.0 - high)
    st["ACFRST"][...] += total.astype(f32)
    st["NCFRST"][...] += 1
    st["NCFRCV"][...] += 1
