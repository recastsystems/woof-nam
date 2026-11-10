"""phys-ntiedtke-cu: New Tiedtke (WRF cu_physics 16) as the CUCNVC convection scheme on NMMB columns.

Configfile: ``convection: ntiedtke``. Optional port keys (not NAM keys):
  ``ntiedtke_shallow: auto | on | off``   the shallow branch (ktype 2); see "EDMF pairing" below. auto.
  ``ntiedtke_tiedtke_closure: .false.``  ArWen's cu_physics 6 closure switch inside the same kernels.

The scheme is ArWen's GPU New Tiedtke (gpuwm/core/ntiedtke.py + kernels/ntiedtke.cu, WRF v4.6.1
module_cu_ntiedtke.F lineage, WRF public domain; ArWen code Apache-2.0), copied unmodified into
namgpu/third_party/arwen/core/ (ntiedtke.py with its three gpuwm imports replaced, see the README there;
kernels/ntiedtke_assembled.cu is ArWen's exact NVRTC source). Nothing in the scheme is rewritten here: this
unit is the seam (namgpu/kernels/phys_ntiedtke_map.cu) and the call.

Where it runs: inside phys-bmj-driver (CUCNVC), in place of the BMJ call, between CUCNVC's own prep and
post kernels. So everything NAM does around the scheme stays as NAM does it: the PHINT/PHMID/RR/EXNER/DZ/
U_PHY/V_PHY prep, the T/Q update by post from RTHCUTEN*EXNER and RQCUTEN over DTCNVC = NCNVC*DT, the
precipitation buckets (PREC, ACPREC, CUPREC, CPRATE), HTOP/HBOT/CNVTOP/CNVBOT for RRTM and output, and
drv-h-to-v-tend's NPRECIP wind block for DUDT_PHY/DVDT_PHY. CU_PHYSICS is 16 (drv-physics-initialize).

The seam, per column:
  - levels flipped (NMMB L = 1 top; WRF k = 1 ground); interfaces p8w from PHINT.
  - Q (specific humidity) -> qv mixing ratio; RQCUTEN back to d(specific humidity)/dt.
  - qc/qi from the total condensate CW split by F_ICE and F_RAIN (FER_HIRES's own split; CW is the only
    advected condensate, QC/QR/QS are stale between microphysics calls).
  - w at interfaces from the mid-layer W_TOT (0 at the ground and the top).
  - hfx = -TWBS, qfx = -QWBS/ELWV (TURBL leaves both positive downward in W m-2).
  - dx = sqrt(DXH(j)*DYH).
  - Large-scale forcing. WRF hands this scheme the advective + radiative + PBL tendency of T and qv
    (module_cumulus_driver.F:867-886). NMMB keeps no advective tendency, so the unit measures the whole
    non-convective, non-microphysical change over the NCNVC steps since the last call: phys-ntiedtke-snap
    stores T and Q right after GSMDRIVE, and thften = (T - TSNAP)/DTCNVC, qvften = d(qv)/DTCNVC. CLTEND
    spreads the convective + microphysical T change over the NPRECIP steps (Tadj), and by the next call it
    is fully applied, so T - TSNAP holds exactly dynamics + radiation + TURBL + TQADJUST. The first call
    of a run, the first after a restart, and any call whose snapshot is not exactly NCNVC steps old (inside
    the digital filter) gets zero forcing, as WRF's first step does.
  - Tendencies -> RTHCUTEN (theta tendency, post multiplies by EXNER), RQCUTEN, DUDT_PHY, DVDT_PHY;
    raincv (rn/stepcu with stepcu = NCNVC, mm per model step, the BMJ RAINCV meaning) -> RAINCV;
    CUTOP/CUBOT = LM+1-kctop / LM+1-kcbot (counted from the ground, as BMJ writes them).
  - Detrained cloud water and ice go into CW (F_ICE/F_RAIN recoupled), the NMMB SAS template
    (module_CONVECTION.F90:566-593), interior columns only, as post updates T and Q.

EDMF pairing (no double counting of shallow convection). New Tiedtke carries its own shallow branch
(ktype 2, the subcloud moist-static-energy closure). A mass-flux PBL (MYNN-EDMF with bl_mynn_edmf on) also
carries the shallow cumulus updrafts. Running both moves the same subcloud heat and moisture twice. Rule:
  - turbulence myj (NAM), ysu, or mynn with the EDMF mass flux off: shallow ON (the scheme as WRF runs it).
  - turbulence mynn with bl_mynn_edmf on (the default for mynn): shallow OFF, ktype 2 columns write no
    tendency, no detrainment and no rain; deep (1) and mid-level (3) convection stay. This is the RAP/HRRR
    rule for GF beside MYNN-EDMF (ishallow off), applied to New Tiedtke.
  - ``ntiedtke_shallow: on`` keeps both anyway (a sensitivity arm); the log says it double counts.
  The per-call ACTIVE field (NT_KTYPE: 0 none, 1 deep, 2 shallow, 3 mid-level, -2 shallow dropped) shows
  the rule acting.

Memory: no per-call domain-size arrays. The pipeline's workspace is held once at the tile width
(nt_tile_columns, the GF tile); the seam kernels pack straight into it and unpack straight out of it, one
chunk at a time, padding a short last chunk with its first column as ArWen does.
"""
from __future__ import annotations

import contextlib
import logging
import os

import numpy as np

from namgpu import kernels

UNIT = "phys_ntiedtke_map"
CU_PHYSICS = 16
LOG = logging.getLogger("namgpu.ntiedtke")

_EDMF_PBL = {"mynn", "mynnedmf", "mynn_edmf", "mynn-edmf", "mynn25", "mynn3"}
_TRUE = {"1", "t", "true", ".true.", "yes", "on"}

# nt_pack's output buffers in its exact parameter order. A levels-then-interfaces grouping here once passed
# dz8w where the kernel writes p8w (LM+1 rows into an LM-row buffer) and shifted every later field by one
# (smoke 2026-11-08); this tuple is the kernel signature order.
_PACK_OUT = (("t3d", 0), ("qv3d", 0), ("qc3d", 0), ("qi3d", 0), ("u3d", 0), ("v3d", 0), ("pcps", 0),
             ("p8w", 0), ("dz8w", 0), ("rho3d", 0), ("exner", 0), ("w", 0), ("qvften", 0), ("thften", 0),
             ("xland", 1), ("hfx", 1), ("qfx", 1), ("dx", 1))
_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in _TRUE


def edmf_mass_flux_on(cfg) -> bool:
    """True when the PBL carries shallow cumulus mass flux (MYNN with bl_mynn_edmf on; on is MYNN's default).

    Two ways a tree selects MYNN-EDMF: the configfile key (turbulence: mynn...), or the pbl-mynn-edmf lane's
    NAMGPU_PBL=mynn-edmf switch (namgpu/physics/mynn/phys_mynn_edmf.py, which runs bl_mynn_edmf=1). Either one
    turns the shallow branch off under ntiedtke_shallow auto.
    """
    turb = str(getattr(cfg, "turbulence", "")).strip().lower()
    if turb in _EDMF_PBL:
        return _truthy(cfg.raw.get("bl_mynn_edmf", 1))
    try:
        from namgpu.physics.mynn import phys_mynn_edmf    # present once the MYNN-EDMF lane is merged
        if phys_mynn_edmf.selected():
            # that lane's options dict is MYNN_OPTIONS (bl_mynn_edmf=1 by default); an earlier read of a
            # non-existent OPTIONS fell back to "on" even with bl_mynn_edmf=0
            return _truthy(getattr(phys_mynn_edmf, "MYNN_OPTIONS", {}).get("bl_mynn_edmf", 1))
        return False
    except ImportError:
        pass
    env = os.environ.get("NAMGPU_PBL", "").strip().lower().replace("_", "-")
    return env in ("mynn-edmf", "mynnedmf", "mynn")


def shallow_off(cfg) -> bool:
    """The pairing rule: drop the Tiedtke shallow branch when the PBL's EDMF mass flux is on."""
    mode = str(cfg.raw.get("ntiedtke_shallow", "auto")).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"phys-ntiedtke-cu: ntiedtke_shallow={mode!r}; use auto, on or off")
    edmf = edmf_mass_flux_on(cfg)
    if mode == "auto":
        return edmf
    if mode == "on" and edmf:
        LOG.warning("phys-ntiedtke-cu: ntiedtke_shallow on with the MYNN EDMF mass flux on: shallow convection is "
                    "counted twice (sensitivity arm only)")
    return mode == "off"


def _pipeline(ctx, lm: int):
    """One NtPipeline for the run, at the tile width, cached in ctx.tables."""
    key = "ntiedtke_pipeline"
    # "in" and [] (not .get): inside row bands ctx.tables is band.BandTables, a dict subclass whose
    # parent lookup lives in __contains__/__getitem__ only.
    if key in ctx.tables:
        return ctx.tables[key]
    import cupy as cp
    from namgpu.third_party.arwen.core import ntiedtke as nt

    cfg = ctx.cfg
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    ncol_dom = int(ctx.grid.im) * int(ctx.grid.jm)
    width = nt.nt_tile_columns(ncol_dom, sms)
    p = nt.NtPipeline(ncol=width, nz=lm, dt=float(cfg.dt), stepcu=int(cfg.nprecip), itimestep=2,
                      tiedtke_closure=_truthy(cfg.raw.get("ntiedtke_tiedtke_closure", False)))
    ctx.tables[key] = p
    LOG.info("phys-ntiedtke-cu: New Tiedtke pipeline %d columns x %d levels, workspace %.1f MiB, shallow %s",
             width, lm, p.w.bytes_allocated() / 2**20, "off (EDMF pairing)" if shallow_off(cfg) else "on")
    return p


def run(ctx):
    """New Tiedtke over the current (band) domain. Reads the CUCNVC prep arrays; writes what post applies."""
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    for name in ("W_TOT", "TWBS", "QWBS", "CW", "F_ICE", "F_RAIN", "QC", "QR", "QS"):
        if name not in st.fields:
            raise RuntimeError(f"phys-ntiedtke-cu: {name} is not in ctx.st; the seam reads it "
                               "(W_TOT from dyn-vsound, TWBS/QWBS from phys-turbl-driver, CW/F_*/Q* from FER_HIRES)")
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ncol = im * jm
    ncnvc = int(cfg.nprecip)
    dt = np.float32(cfg.dt)
    dtcnvc = np.float32(np.float32(ncnvc) * dt)
    i32, f32 = np.int32, np.float32

    p = _pipeline(ctx, lm)
    width = int(p.w.ncol)
    w = p.w

    snap_step = ctx.tables["ntiedtke_snap_step"] if "ntiedtke_snap_step" in ctx.tables else None
    forcing_ok = int(snap_step is not None and int(ctx.ntimestep) - int(snap_step) == ncnvc
                     and "NT_TSNAP" in st.fields and not getattr(ctx, "filter_active", False))
    tsnap = st["NT_TSNAP"] if forcing_ok else st["T"]
    qsnap = st["NT_QSNAP"] if forcing_ok else st["Q"]

    st.add("NT_KTYPE", (jm, im), dtype="int32")
    from namgpu.physics.bmj import phys_bmj_bmjdrv
    dxh = phys_bmj_bmjdrv._dxh_device(st, g)
    off = int(shallow_off(cfg))

    pack = kernels.get(UNIT, "nt_pack")
    unpack = kernels.get(UNIT, "nt_unpack")
    blk = int(kernels.BLOCK_COL[0])   # BLOCK_COL is the (128, 1, 1) launch tuple
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    # The seam kernels take a (col0, n, width) window of the domain and pack straight into the workspace.
    for col0 in range(0, ncol, width):
        n = min(width, ncol - col0)
        with scope:
            pack(_grid(width, blk), (blk,), (
                i32(im), i32(jm), i32(lm), i32(col0), i32(n), i32(width), i32(forcing_ok), f32(1.0) / dtcnvc,
                st["T"], st["Q"], st["CW"], st["F_ICE"], st["F_RAIN"], st["U_PHY"], st["V_PHY"],
                st["PHMID"], st["PHINT"], st["DZ"], st["RR"], st["EXNER"], st["W_TOT"], tsnap, qsnap,
                st["XLAND"], st["TWBS"], st["QWBS"], dxh, f32(g.dyh),
                *[w.bind(nm, kind) for nm, kind in _PACK_OUT]))
        p.run_chunk()
        with scope:
            unpack(_grid(n, blk), (blk,), (
                i32(im), i32(jm), i32(lm), i32(col0), i32(n), i32(width), dtcnvc, i32(off),
                *[w.bind(nm, 0) for nm in _OUT_LEV], w.bind("raincv", 1),
                w.bind("ktype", 1), w.bind("ldcum", 1), w.bind("kctop", 1), w.bind("kcbot", 1),
                st["Q"],
                st["RTHCUTEN"], st["RQCUTEN"], st["DUDT_PHY"], st["DVDT_PHY"],
                st["RAINCV"], st["CUTOP"], st["CUBOT"],
                st["CW"], st["F_ICE"], st["F_RAIN"], st["QC"], st["QR"], st["QS"], st["NT_KTYPE"],
                i32(1 if str(cfg.microphysics).strip() == "fer_hires" else 0)))
    _hourly_line(ctx, ncnvc, forcing_ok, off)


def _hourly_line(ctx, ncnvc: int, forcing_ok: int, off: int) -> None:
    """Once per forecast hour (one small readback): columns per ktype and the land-mean surface fluxes the
    scheme saw, so the forecast log shows the scheme and the EDMF rule acting. Not on any other call."""
    per_hour = max(1, int(round(3600.0 / float(ctx.cfg.dt))))
    if int(ctx.ntimestep) % per_hour >= ncnvc:
        return
    import cupy as cp
    st = ctx.st
    act = st["NT_KTYPE"]
    land = st["XLAND"] < 1.5
    counts = cp.stack([cp.count_nonzero(act == v) for v in (1, 2, 3, -2)]
                      + [cp.count_nonzero(land)]).get()
    flx = cp.stack([cp.sum(cp.where(land, -st["TWBS"], 0.0)), cp.sum(cp.where(land, -st["QWBS"], 0.0)),
                    cp.sum(st["RAINCV"])]).get()
    nl = max(int(counts[4]), 1)
    rule = "EDMF pairing: shallow off" if off else "shallow on"
    forcing = "on" if forcing_ok else "zero (first call or filter)"
    print(f"ntiedtke step {int(ctx.ntimestep)}: deep {int(counts[0])} shallow {int(counts[1])} mid {int(counts[2])} "
          f"shallow-dropped {int(counts[3])} ({rule}); forcing {forcing}; land-mean hfx {flx[0] / nl:.1f} W m-2, "
          f"latent {flx[1] / nl:.1f} W m-2; domain RAINCV sum {flx[2]:.3g} mm/step", flush=True)


def _grid(n: int, blk: int):
    return ((int(n) + blk - 1) // blk,)
