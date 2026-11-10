"""phys-noahmp-driver: Noah-MP land surface (WRF v4.6.1 lineage, ArWen's GPU port) in place of NAM Noah.

Selectable, not default: NAMGPU_LSM=noahmp (or land_surface noahmp in the configfile) swaps the call of
phys-noah-driver inside phys-turbl-driver's surface block for this unit. NAM Noah stays the default and is
untouched when this is off.

Physics: namgpu/third_party/arwen_noahmp (ArWen gpuwm Noah-MP, Apache-2.0, from WRF's public-domain
phys/noahmp; WRF v4.6.1 MPTABLE/SOILPARM/GENPARM). Not the UCAR-licensed Noah-MP v5.0. Options are ArWen's
admitted set: dveg=4, opt_crs=1, opt_btr=1, opt_run=1, opt_sfc=1, opt_frz=1, opt_inf=1, opt_rad=3, opt_alb=2,
opt_snf=1, opt_tbot=2, opt_stc=1, opt_soil=1; MODIFIED_IGBP_MODIS_NOAH (= NAM's IGBP, IVEGSRC 1, water 17,
ice 15, urban 13) and STATSGO soils (= NAM's STAS); 4 soil layers 0.1/0.3/0.6/1.0 m (= NAM SLDPTH).

Column interface (NMMB -> WRF noahmplsm), lowest model layer = index LM-1, surface interface = PHINT[LM]:
    T3D(1)<-T[LM-1]  QV3D(1)<-Q/(1-Q) (NAM Q is specific humidity, Noah-MP's driver takes mixing ratio)
    U_PHY/V_PHY[LM-1] (mass-point winds, turbl_uv_phy)  DZ8W(1)<-DZ[LM-1]  P8W3D(1:2)<-PHINT[LM], PHINT[LM-1]
    TSK<-TSFC  HFX<-TWBS  QFX<-QWBS  LH<-ELFLX  GRDFLX<-GRNFLX  QSFC: Noah-MP specific, written to QSH as
    q/(1-q) (the convention NAM Noah writes)  SWDOWN<-RSW_DN_SFC  GLW<-RLW_DN_SFC  RAINBL, SR  XLAND
    XICE<-SICE  IVGTYP ISLTYP  SHDMAX<-VGFRCK (NAM carries no annual-max green fraction)  TMN<-TG
    ALBEDO EMISS<-EPSR SNOWC SNOW (mm) SNOWH (m) CANWAT<-CMC (mm)  SMOIS<-SMC TSLB<-STC SH2O
    SMSTAV SMSTOT SFCRUNOFF<-SSROFF UDRUNOFF<-BGROFF ACSNOW ACSNOM  ZNT/Z0 -> NMP_ZNT/NMP_Z0 (NAM's JSFC keeps
    its own roughness)  COSZEN: WRF calc_coszen at the call time from GLAT/GLON on the device.
Precipitation: RAINBL (mm per surface call) with SR, through noahmplsm's PRCPOTHR branch (the six ARW
partial accumulations are zero), as WRF does when only RAINBL is known.
2 m: WRF module_surface_driver's Noah-MP T2/Q2 ownership (FVEG blend of T2MV/T2MB, Q2MV/Q2MB) over
non-water, non-full-ice points, written into TSHLTR (theta at PSHLTR) and QSHLTR (specific humidity), which
the history writer turns into T2/Q2. Water and full sea ice keep NAM's JSFC values.
Carried Noah-MP state (canopy, snow layers, ...) lives in the State as NMP_<name> whole-domain arrays.
Cold start (NOAHMP_INIT + SNOW_INIT) at the first call, from the NAM soil/snow state.
"""
from __future__ import annotations

import math
import os
from datetime import timedelta

import numpy as np

UNIT = "phys_noahmp_driver"
ENV = "NAMGPU_LSM"
KAPPA = np.float32(0.28589641)          # io_history_fields: T2 = TSHLTR * (PSHLTR*1e-5)**CAPA
NSOIL = 4
DZS = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)

# WRF name -> NAM State name for arrays both models carry.
ALIASES = {
    "xland": "XLAND", "xice": "SICE", "ivgtyp": "IVGTYP", "isltyp": "ISLTYP", "shdmax": "VGFRCK",
    "tmn": "TG", "swdown": "RSW_DN_SFC", "glw": "RLW_DN_SFC", "rainbl": "RAINBL", "sr": "SR",
    "snow": "SNOW", "snowh": "SNOWH", "canwat": "CMC", "tsk": "TSFC", "hfx": "TWBS", "qfx": "QWBS",
    "lh": "ELFLX", "grdflx": "GRNFLX", "albedo": "ALBEDO", "snowc": "SNOWC", "emiss": "EPSR",
    "smstav": "SMSTAV", "smstot": "SMSTOT", "sfcrunoff": "SSROFF", "udrunoff": "BGROFF",
    "acsnow": "ACSNOW", "acsnom": "ACSNOM", "chs2": "CHS2", "cqs2": "CQS2",
    "tslb": "STC", "smois": "SMC", "sh2o": "SH2O",
}
ZERO_2D = ("SMSTAV", "SMSTOT", "SSROFF", "BGROFF", "ACSNOW", "ACSNOM", "SNOWC", "TWBS", "QWBS",
           "ELFLX", "GRNFLX")


def selected(ctx) -> bool:
    v = (os.environ.get(ENV) or getattr(ctx.cfg, "land_surface", "noah") or "noah").strip().lower()
    if v not in ("noah", "noahmp", "ruc"):          # ruc: the RUC option (namgpu/physics/ruc) owns land points
        raise ValueError(f"{ENV}/land_surface={v!r}: the land surface is 'noah' (default), 'noahmp' or 'ruc'")
    return v == "noahmp"


def _rt():
    from namgpu.third_party.arwen_noahmp.core import noahmp_runtime as rt
    return rt


_PARAMS = {}


def _params():
    if "p" not in _PARAMS:
        _PARAMS["p"] = _rt().NoahmpRuntimeParameters()
    return _PARAMS["p"]


def _carrier(st, name, shape, dtype="float32"):
    key = "NMP_" + name.upper()
    if key in st.fields:
        return st.fields[key]
    return st.add(key, shape, dtype)


def _shdmax(ctx, st, s2):
    """NMP_SHDMAX (percent): WRF's SHDMAX, the annual max of the monthly MODIS green fraction, from the
    NAMGPU_NOAHMP_SHDMAX npz (tools/noahmp-lane/shdmax_build.py, same grid), never below the NAM's current
    VGFRCK; without the file, VGFRCK itself (printed once: then FVEG follows the season, not the annual max).
    Written once into the whole-domain parent State (a band ctx reaches it through ctx._p), then carried."""
    import cupy as cp
    parent = getattr(ctx, "_p", ctx)
    pst = parent.st
    if not parent.tables.get("NMP_SHDMAX_SET"):
        pj, pi = int(parent.grid.jm), int(parent.grid.im)
        whole = pst.fields["NMP_SHDMAX"] if "NMP_SHDMAX" in pst.fields else pst.add("NMP_SHDMAX", (pj, pi))
        path = os.environ.get("NAMGPU_NOAHMP_SHDMAX", "").strip()
        h = np.load(path)["SHDMAX"].astype(np.float32) if path else None
        if h is not None and h.shape == (pj, pi):
            whole[...] = cp.maximum(cp.asarray(h), pst["VGFRCK"])
            print(f"{UNIT}: SHDMAX from {path} (annual-max MODIS green fraction, land mean "
                  f"{float(h[h > 0].mean()):.1f} %), floored at VGFRCK", flush=True)
        else:
            whole[...] = pst["VGFRCK"]
            print(f"{UNIT}: SHDMAX = VGFRCK ({'no NAMGPU_NOAHMP_SHDMAX' if h is None else 'SHDMAX file is for another grid'}):"
                  " FVEG follows the current green fraction, not WRF's annual maximum", flush=True)
        parent.tables["NMP_SHDMAX_SET"] = True
    return st["NMP_SHDMAX"]


_SHD = {}


def run(ctx):
    import cupy as cp
    from namgpu.third_party.arwen_noahmp.core.dudhia import wrf_solar_geometry
    from namgpu.third_party.arwen_noahmp.core.surface_forcing import (
        SURFACE_PRECIPITATION_FIELD_NAMES, SurfacePrecipitationForcing)
    rt = _rt()
    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    s2 = (jm, im)
    nsnow = int(rt.NSNOW)
    params = _params()

    for n in ZERO_2D:
        if n not in st.fields:
            st.add(n, s2)
    fields = {w: st[n] for w, n in ALIASES.items()}
    fields["shdmax"] = _shdmax(ctx, st, s2)
    for n in rt.NOAHMP_STATE_2D + rt.NOAHMP_DIAGNOSTICS_2D + ("lai", "qsfc", "z0", "znt", "t2", "q2", "th2"):
        fields[n] = _carrier(st, n, s2)
    for n in rt.NOAHMP_STATE_INT_2D:
        fields[n] = _carrier(st, n, s2, "int32")
    for n in rt.NOAHMP_STATE_SNOW_3D:
        fields[n] = _carrier(st, n, (nsnow, jm, im))
    for n in rt.NOAHMP_STATE_SNOWSOIL_3D:
        fields[n] = _carrier(st, n, (nsnow + NSOIL, jm, im))
    fields["psfc"] = st["PHINT"][lm]
    for name in SURFACE_PRECIPITATION_FIELD_NAMES.values():
        fields[name] = _carrier(st, name, s2)          # stay zero: RAINBL/SR carry the precipitation

    # Step counter: 1 on the first surface call of the run (noahmplsm ITIMESTEP == 1 water/ice presets).
    first = ctx.tables.get("NMP_FIRST_NTSD")
    if first is None:
        first = int(ctx.ntimestep)
        ctx.tables["NMP_FIRST_NTSD"] = first
    itimestep = 1 if int(ctx.ntimestep) == first else int(ctx.ntimestep) + 1
    if itimestep == 1:
        # NAM state -> Noah-MP carriers, then NOAHMP_INIT/SNOW_INIT.
        q = st["QSH"]
        fields["qsfc"][...] = q / (cp.float32(1.0) + q)
        # NAMGPU_NOAHMP_INIT_SNOWH = swe (default: WRF's FNDSNOWH-false branch, SNOWH = SNOW * 0.005, 200 kg/m3)
        # | depth (the start's own snow depth, WRF FNDSNOWH true). GFS depths are ~10:1 (105 kg/m3 at the
        # 2026-01-23 thin-snow ASOS sites): Noah-MP then covers 5-20 mm SWE fully (FSNO 0.98), the pack insulates
        # the skin from the soil and 2 m T at those sites fell to -8.5 F bias by the first evening (MAE 8.8);
        # with swe, FSNO 0.48 and -2.9 F (MAE 4.4; NAM Noah control +3.5 / 4.9). Whole-run T2 MAE +12 h 4.3 -> 3.4.
        if os.environ.get("NAMGPU_NOAHMP_INIT_SNOWH", "swe").strip().lower() == "swe":
            fields["snowh"][...] = fields["snow"] * cp.float32(0.005)
            print(f"{UNIT}: cold-start SNOWH = SNOW * 0.005 (NAMGPU_NOAHMP_INIT_SNOWH=swe, 200 kg/m3)", flush=True)
        rt.noahmp_cold_start(fields, params=params, dzs=DZS)

    # Forcing (2-level views, bottom first), mass-point winds, mixing ratio.
    kl = lm - 1
    qv = st["Q"][kl:kl + 1]
    atmosphere = {
        "temperature": st["T"][kl:kl + 1],
        "qv": qv / (cp.float32(1.0) - qv),
        "u": st["U_PHY"][kl:kl + 1], "v": st["V_PHY"][kl:kl + 1],
        "dz": st["DZ"][kl:kl + 1],
        "p_interface": cp.stack((st["PHINT"][lm], st["PHINT"][lm - 1])),
    }
    # Solar geometry at the call time, WRF calc_coszen, on the device.
    start = ctx.start if ctx.start is not None else ctx.start_time
    elapsed = float(ctx.ntimestep) * float(ctx.dt or ctx.cfg.dt)
    deg = cp.float32(180.0 / math.pi)
    lat_d = st["GLAT"] * deg
    lon_d = st["GLON"] * deg
    cosz, _ = wrf_solar_geometry(start + timedelta(seconds=elapsed), lat_d, lon_d)
    geometry = rt.NoahmpSolarGeometry(start, cp.asnumpy(lat_d), cp.asnumpy(lon_d))
    dt = float(np.float32(ctx.cfg.nphs) * np.float32(ctx.dt or ctx.cfg.dt))
    dx = float(np.mean(g.dxh))

    # NAMGPU_NOAHMP_OPT_RSF: Noah-MP ground surface resistance option, WRF v4.6.1 (1, 2, 3, 4; default 4).
    # 4 = Sakaguchi-Zeng off snow, FSNO-weighted RSURF_SNOW on snow. With 1, RSURF over snow comes from the
    # frozen top layer's liquid water and snow barely sublimates (0123: latent heat over snow ~1/10 of NAM Noah,
    # Td2 at 5-20 mm SWE sites -3.4 F at +12 h); with 4, Td2 there -2.0 and +6 h 5-20 mm MAE 2.9 (Noah 3.3).
    from namgpu.third_party.arwen_noahmp.core import noahmp_energy_slab as _es
    rsf = int(os.environ.get("NAMGPU_NOAHMP_OPT_RSF", "4") or "4")
    if rsf not in (1, 2, 3, 4):
        raise ValueError(f"NAMGPU_NOAHMP_OPT_RSF={rsf}: Noah-MP opt_rsf is 1, 2, 3 or 4")
    if _es.OPT_RSF != rsf:
        _es.OPT_RSF = rsf
        print(f"{UNIT}: opt_rsf = {rsf}", flush=True)
    census = rt.noahmp_lsm_step(
        fields, atmosphere, params=params, geometry=geometry,
        precipitation=SurfacePrecipitationForcing.from_fields(fields),
        coszen=cp.ascontiguousarray(cosz.astype(cp.float32)), dt=dt, dx=dx, dzs=DZS,
        itimestep=itimestep, elapsed_seconds=elapsed, dveg=4, opt_run=1, opt_crop=0, opt_irr=0,
        opt_tdrn=0, opt_soil=1)

    # Hand-back to NAM conventions.
    # CHKLOWQ = 1 at every point, as NAM Noah's prelude writes it (phys_noah_driver.cu:198): MYJ's VDIFH
    # multiplies the surface moisture flux by it (XLV*CHKLOWQ*AKHS_DENS). Unset, it stayed at its zero fill
    # and the first 24 h arm (2026-01-23) had no surface evaporation anywhere, land or sea.
    (st.fields["CHKLOWQ"] if "CHKLOWQ" in st.fields else st.add("CHKLOWQ", s2)).fill(1.0)
    land =(st["XLAND"] < cp.float32(1.5)) & (st["SICE"] < cp.float32(0.5))
    qs = fields["qsfc"]
    st["QSH"][...] = cp.where(land, qs / (cp.float32(1.0) - qs), st["QSH"])
    if "TSHLTR" in st.fields and "PSHLTR" in st.fields:
        own = land & (st["IVGTYP"] != int(params.land_use.iswater))
        th = fields["t2"] * (cp.float32(1.0e5) / st["PSHLTR"]) ** KAPPA
        r2 = fields["q2"]
        st["TSHLTR"][...] = cp.where(own, th, st["TSHLTR"])
        st["QSHLTR"][...] = cp.where(own, r2 / (cp.float32(1.0) + r2), st["QSHLTR"])
    # Flux-consistent MYJ skin for heat. NMMB's MYJ VDIFH applies the surface heat flux as
    # CP*AKHS*RHOK(LM)*(THZ0 - TH(LM)) with THZ0 = TSK*(1E5/PSFC)**CAPPA over land, so the atmosphere only gets
    # the LSM's sensible heat flux when the LSM's TSK is consistent with JSFC's AKHS. NAM Noah is (it runs with
    # CH = AKHS: on the 2026-01-23 control TWBS equals the MYJ-applied flux at every land point). Noah-MP uses its
    # own exchange coefficients, so MYJ applied a different flux (12 h arm, 18Z: 73 W/m2 into the air over snow
    # where Noah-MP's HFX was 113, 130 where it was 109 over bare land; correlation 0.38-0.84) and land and air
    # did not conserve energy. Moisture is already consistent: VDIFH inverts QSFC from ELFLX (= Noah-MP's LH).
    # So MYJ gets NMP_TSK_MYJ: the skin temperature that makes its formula return Noah-MP's HFX exactly.
    # TSFC (radiation, SFCDIAGS, JSFC stability) keeps Noah-MP's own TSK. Points with AKHS <= 1e-4 m/s (flux
    # under ~6 W/m2 anyway) and water/sea ice keep TSFC.
    kl = lm - 1
    rd, cpd = cp.float32(287.04), cp.float32(1004.6)
    p608 = cp.float32(461.6) / rd - cp.float32(1.0)
    rho = st["PHMID"][kl] / (rd * st["T"][kl] * (cp.float32(1.0) + p608 * st["Q"][kl] - st["QC"][kl]))
    akhs = st["AKHS"]
    ok = land & (akhs > cp.float32(1.0e-4))
    dth = fields["hfx"] / (cpd * cp.where(ok, akhs, cp.float32(1.0)) * rho)
    dth = cp.clip(dth, cp.float32(-50.0), cp.float32(50.0))
    thz0 = st["TH"][kl] + dth
    tsk_myj = thz0 * (st["PHINT"][lm] * cp.float32(1.0e-5)) ** (rd / cpd)
    _carrier(st, "tsk_myj", s2)[...] = cp.where(ok, tsk_myj, st["TSFC"])
    ctx.tables["NMP_CENSUS"] = census
    return census


def post_sfcdiags(ctx):
    """After phys-noah-sfcdiags: Noah-MP owns T2/TH2X/Q2X over its land points (WRF module_surface_driver
    under NOAHMPSCHEME). SFCDIAGS inverts the land fluxes with JSFC's CHS2/CQS2, which NAM Noah rewrites at
    land (CHS2 = CQS2, CQS2 = CHS on condensation) and Noah-MP does not, so its land T2 came out at -301 to
    1008 K on the 2026-01-23 smoke and fed io-max-fields' T10. Water and sea ice keep SFCDIAGS."""
    import cupy as cp
    st = ctx.st
    if "NMP_T2" not in st.fields:
        return
    land = (st["XLAND"] < cp.float32(1.5)) & (st["SICE"] < cp.float32(0.5))
    for nam, nmp in (("T2", "NMP_T2"), ("TH2X", "NMP_TH2"), ("Q2X", "NMP_Q2")):
        if nam in st.fields:
            st[nam][...] = cp.where(land, st[nmp], st[nam])
