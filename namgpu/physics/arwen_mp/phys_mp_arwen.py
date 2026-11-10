"""phys-mp-arwen: GSMDRIVE CASE ('thompson') glue with ArWen's GPU microphysics as the scheme (modernise lane
mp-thompson-p3).

Fortran structure followed: nmm/module_MICROPHYSICS.F90 (NAM nam.v4.2.11, 83df4e13c1) GSMDRIVE
  - column prep (216-263): DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY, the same prep kernel the fer_hires route uses;
  - USE_QV (275-300): QV = Q/(1-Q) for the Thompson branch;
  - CASE ('thompson') (418-466): the scheme call on QV, QC, QR, QI, QS, QG, NI, NR, TH_PHY, PI_PHY, P_PHY, DZ, DTPHS;
    RAINNCV is all precipitation (snow and graupel included), SR the frozen fraction;
  - after the select (488-531): Q = QV/(1+QV), CWM = QC+QR+QI+QS+QG, TRAIN and T from TH_PHY*PI_PHY (the fer
    post kernel), PREC/ACPREC/ACPREC_TOT from RAINNCV (the fer precip kernel). B1 interior only, as the Fortran tiles.

The scheme is the ArWen (gpuwm, Apache-2.0, WRF v4.6.1 public-domain lineage) CUDA microphysics, called through
`gpuwm.core.microphysics.apply` on a shim state (namgpu.physics.arwen_mp.shim):
  microphysics thompson       -> ArWen mp_physics 8   (classic Thompson, WRF v4.6.1 module_mp_thompson)
  microphysics thompson_aero  -> ArWen mp_physics 28  (aerosol-aware Thompson: NC, NWFA, NIFA as advected tracers)
  microphysics p3             -> ArWen mp_physics 50  (P3 one-category, specified Nc; raw key p3_predict_nc true -> 51)
NMMB columns are top-down (l = 0 is the model top); ArWen columns are bottom-up. The shim flips the vertical axis
on the way in and out. Everything stays on the card: copies are device-to-device, no host sync per call.

Gridpoint layout: the scheme runs on the whole (JM, IM) plane and only the B1 interior (rows/cols 1..N-2) is written
back, the GSMDRIVE tile (ITS_B1..ITE_B1). The outer ring keeps its boundary values.

ArWen package location: NAMGPU_ARWEN_SITE (a folder holding the `gpuwm` package, e.g. the public gpuwm 2.9.13 wheel
unpacked), else an importable gpuwm. A missing package is a refusal naming the cause (no fallback scheme).
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_driver"            # the prep/post/precip kernels are the fer_hires route's (same Fortran lines)
K_PREP = "phys_fer_driver_prep"
K_POST = "phys_fer_driver_post"
K_PRECIP = "phys_fer_driver_precip"

ARWEN_MP = {"thompson": 8, "thompson_aero": 28, "p3": 50}
G = 9.81


def _import_arwen():
    site = os.environ.get("NAMGPU_ARWEN_SITE")
    if site and site not in sys.path:
        sys.path.insert(0, site)
    try:
        from gpuwm.core import microphysics as amp  # noqa: PLC0415
        from gpuwm.config import RunConfig  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "phys-mp-arwen: the ArWen microphysics package (gpuwm) is not importable "
            f"(NAMGPU_ARWEN_SITE={site!r}): {type(e).__name__}: {e}. Set NAMGPU_ARWEN_SITE to a folder holding the "
            "gpuwm package (pip download gpuwm==2.9.13 --no-deps and unzip it).") from e
    return amp, RunConfig


def arwen_mp_physics(cfg) -> int:
    mp = str(cfg.microphysics).strip()
    n = ARWEN_MP[mp]
    if mp == "p3" and str(cfg.raw.get("p3_predict_nc", False)).strip().lower() in ("true", ".true.", "t", "1", "yes"):
        n = 51
    return n


def _vcoord(ctx, attr: str):
    from namgpu.physics.driver.phys_fer_driver import _vcoord as fer_vcoord  # noqa: PLC0415
    return fer_vcoord(ctx, attr)


def _band_rows(ctx, lm, jm, im) -> int:
    """Rows per band: NAMGPU_MP_BAND_ROWS, else a working set of NAMGPU_MP_BAND_MIB (default 2048 MiB, at most 40 %
    of the free device memory plus the free pool) at ~64 (LM, IM) planes per row (shim + ArWen scheme scratch)."""
    env = os.environ.get("NAMGPU_MP_BAND_ROWS")
    if env:
        return max(1, min(jm, int(env)))
    import cupy as cp
    free, _tot = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    avail = free + (pool.total_bytes() - pool.used_bytes())
    budget = min(float(os.environ.get("NAMGPU_MP_BAND_MIB", 2048)) * 2**20, 0.4 * avail)
    per_row = 64.0 * lm * im * 4
    return max(16, min(jm, int(budget // per_row)))


def _setup(ctx):
    """Build the band shim once (scheme scratch is keyed by band shape, so every band reuses it)."""
    from namgpu.physics.arwen_mp.shim import ShimState  # noqa: PLC0415
    amp, RunConfig = _import_arwen()
    g, cfg = ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nphs = int(cfg.nprecip)
    dtphs = float(np.float32(nphs) * np.float32(ctx.dt))
    caller = ctx.tables.get("fer_hires_caller") or {}
    dx = float(caller.get("delx", 12.0)) * 1000.0
    dy = float(caller.get("dely", 12.0)) * 1000.0
    mpn = arwen_mp_physics(cfg)
    nb = _band_rows(ctx, lm, jm, im)
    acfg = RunConfig(nx=im, ny=nb, nz=lm, dx=dx, dy=dy, ztop=20000.0,
                     dt=dtphs, run_seconds=86400.0, mp_physics=mpn, moist=True)
    shim = ShimState(ctx, lm, nb, im, mpn)
    bands = []
    for own0 in range(0, jm, nb):
        own1 = min(own0 + nb, jm)
        w0 = min(own0, jm - nb)
        bands.append((w0, w0 + nb, own0, own1))
    ent = {"amp": amp, "cfg": acfg, "shim": shim, "mp": mpn, "dtphs": dtphs, "init": False, "calls": 0,
           "bands": bands}
    ctx.tables["arwen_mp"] = ent
    print(f"phys-mp-arwen: ArWen mp_physics={mpn} on the NMMB thompson branch, {len(bands)} row bands of {nb} "
          f"(of {jm}), dx {dx:.0f} m, dtphs {dtphs:.1f} s", flush=True)
    return ent


def run(ctx, use_radar: bool = False, nprecip: int | None = None):
    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ent = ctx.tables.get("arwen_mp") or _setup(ctx)
    amp, acfg, shim = ent["amp"], ent["cfg"], ent["shim"]

    nphs = int(cfg.nprecip if nprecip is None else nprecip)
    dtphs = np.float32(nphs) * np.float32(ctx.dt)
    rdtphs = np.float32(1.0) / dtphs

    s3, s2 = (lm, jm, im), (jm, im)
    dz8w, rho_phy, p_phy, pi_phy, th_phy = (st.add(n, s3) for n in ("DZ8W", "RHO_PHY", "P_PHY", "PI_PHY", "TH_PHY"))
    lowlyr = st.add("LOWLYR", s2, dtype="int32")
    rainnc = st.add("RAINNC", s2)
    rainncv = st.add("RAINNCV", s2)

    block = kernels.BLOCK_2D
    grid3 = kernels.grid_2d(im, jm, lm)
    grid2 = kernels.grid_2d(im, jm, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, K_PREP)(grid3, block, (
            np.int32(im), np.int32(jm), np.int32(lm),
            st["PD"], st["T"], st["Q"], st["DSG2"], st["SGML2"], _vcoord(ctx, "pdsg1"), _vcoord(ctx, "psgml1"),
            dz8w, rho_phy, p_phy, pi_phy, th_phy,
            st["AVRAIN"], rainnc, rainncv, lowlyr,
        ))

        first = not ent["init"]
        cold_aero = False
        if first and ent["mp"] == 28:
            tr = st["TRACERS"]
            tab = ctx.tables.get("indx") or {}
            cold_aero = all(float(tr[int(tab[k]) - 1].max()) == 0.0 for k in ("INDX_NWFA", "INDX_NIFA") if k in tab)
        for (w0, w1, own0, own1) in ent["bands"]:
            shim.load(w0, w1, dz8w=dz8w, rho_phy=rho_phy, p_phy=p_phy, th_phy=th_phy,
                      p3_first=first, itimestep=ent["calls"] + 1)
            if first:
                if cold_aero:          # a cold start: the band sees no filled overlap rows from the band before
                    shim.nwfa[...] = 0.0
                    shim.nifa[...] = 0.0
                # WRF microphysics_init (module_physics_init.F, once per domain): for mp 28 thompson_init's
                # synthetic CCN/IN profile where the aerosol tracers arrive unset (all zero on a cold start).
                rec = dict(amp.microphysics_init(shim, acfg) or {})
                if ent["mp"] == 28:
                    shim.store_aerosol((own0, own1))
                if own0 == 0:
                    print(f"phys-mp-arwen: init {rec}", flush=True)
            diag = amp.apply(shim, acfg, float(dtphs), refl_10cm_due=True)
            shim.store((own0, own1), th_phy=th_phy, diag=diag, rainncv=rainncv, rainnc=rainnc)
        ent["init"] = True
        shim.zero_ring()

        # T, TRAIN (B1 interior) from TH_PHY*PI_PHY
        kernels.get(UNIT, K_POST)(grid3, block, (
            np.int32(im), np.int32(jm), np.int32(lm),
            th_phy, pi_phy, st["DFI_TTEN"], st["T"], st["TRAIN"],
            rdtphs, np.int32(1 if use_radar else 0),
        ))
        kernels.get(UNIT, K_PRECIP)(grid2, block, (
            np.int32(im), np.int32(jm),
            rainncv, st["PREC"], st["ACPREC"], st["ACPREC_TOT"],
        ))
    ent["calls"] += 1
    return {"RAINNC": rainnc, "RAINNCV": rainncv}
