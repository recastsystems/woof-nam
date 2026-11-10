"""phys-gf-gfdrv: scale-aware Grell-Freitas cumulus on the NMMB column (nam_configfile `convection: gf`).

A WOOF-NAM option, not NAM code: NAM ran BMJ. This replaces the BMJ call inside CUCNVC (phys-bmj-driver):
the glue's prep kernel still builds PHINT, PHMID, EXNER, RR, DZ, U_PHY, V_PHY and zeroes the tendencies, and
its post kernel still applies RTHCUTEN/RQCUTEN to T/Q and RAINCV to the precipitation buckets, HTOP/HBOT,
CUPPT (so RRTM's convective cloud keeps working). In between, this unit runs GFDRV.

Scheme: namgpu/kernels/phys_gf_gf.cu is ArWen's CUDA GFDRV (gpuwm/core/kernels/gf.cu, WRF v4.6.1
module_cu_gf_wrfdrv.F + module_cu_gf_deep.F + module_cu_gf_sh.F, held at max_ulp 0 against WRF in ArWen),
compiled at GF_KMAX = 60 for NAM's 60 layers. Deep (16-member ensemble closure, ichoice 0) and shallow
(ishallow 1, the default here: BMJ also carries shallow convection) arms; momentum transport on.
Scale awareness is GF's own: the updraft area fraction sig = (1 - frh)^2 with frh from the cloud radius
against dx (module_cu_gf_deep.F:455-471), so the scheme shrinks itself as dx falls. At dx <= GF_OFF_DX_M
(9 km on the smallest DXH) the deep and shallow arms are switched off and the domain runs with explicit
convection only, as NAM's own nests do (convection none); the run log says so. Measured on the rotated grids
(DXH = a cos(tph) dlm): conus12 9.9-14.0 km keeps GF; conus7p6 8.35-8.87 km (dy 7.6 km), the 5 km class and
the 4 km class (4.39-4.67 km) run explicit. gf_off_dx_m = 0 keeps GF on at any spacing (a selectable arm).

Column map and forcing: namgpu/kernels/phys_gf_driver.cu (header). The forcing is measured: T and Q are kept
after each convection step's microphysics (gf_snapshot, called by the driver at the end of the physics group),
so at the next convection step (T - T_snap)/(NPRECIP*DT) is the dynamics + radiation + PBL tendency WRF hands
GFDRV. The first convection step of a forecast (and the first after the digital filter) has no snapshot of
the step before it and runs with the PBL lane only.

Knobs (nam_configfile, all optional): gf_ishallow (1), gf_momentum (true), gf_off_dx_m (9000),
gf_blocks_per_sm (4: the workspace tile, ArWen's measured plateau).

Launch: pack -> GFDRV -> unpack per tile of columns, on ctx.stream, no host sync. Workspace and tile buffers
come from the CuPy pool each call (ArWen's global-workspace layout: charged per column in flight).
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from namgpu import kernels

UNIT = "phys_gf_driver"
GF_UNIT = "phys_gf_gf"
CU_PHYSICS_GF = 3          # WRF's GFSCHEME number (CU_PHYSICS table value for this option)
GF_OFF_DX_M = 9000.0     # conus12 (min DXH 9.9 km) keeps GF; conus7p6 (8.35-8.87 km), 5 km and 4 km run explicit

# phys_gf_gf.cu workspace layout (ArWen gpuwm/core/gf.py, checked against the .cu by tests/test_gf_workspace.py there)
GFWS_SLOTS = 90 + 36
GF_BLOCK = 64
GF_KMAX = 60
NIN_LEV, NIN_SCA, NLEV, NSCA, NISCA = 15, 6, 16, 9, 6


def _flag(raw, key, default):
    v = raw.get(key, default)
    if isinstance(v, str):
        v = v.strip().strip(".").lower() in ("true", "t", "1", "yes")
    return bool(v)


def gf_active(ctx) -> bool:
    """False when the grid is at or below the self-off spacing (explicit convection only)."""
    t = ctx.tables.get("gf_active")
    if t is None:
        dx_min = float(np.min(np.asarray(ctx.grid.dxh, dtype=np.float64)))
        off = float(ctx.cfg.raw.get("gf_off_dx_m", GF_OFF_DX_M))
        t = dx_min > off
        ctx.tables["gf_active"] = t
        ctx.tables["gf_dx_min_m"] = dx_min
        print(f"phys-gf-gfdrv: Grell-Freitas {'ON' if t else 'OFF (explicit convection)'}: "
              f"min dx {dx_min:.0f} m, self-off at {off:.0f} m", flush=True)
    return bool(t)


def snapshot(ctx) -> None:
    """T, Q after this convection step's microphysics: the next convection step measures its forcing from them."""
    st = ctx.st
    g = ctx.grid
    lm, jm, im = int(g.lm), int(g.jm), int(g.im)
    ts = st.add("GF_TSNAP", (lm, jm, im))
    qs = st.add("GF_QSNAP", (lm, jm, im))
    n = lm * jm * im
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, "gf_snapshot")(((n + 255) // 256,), (256,), (np.int32(n), st["T"], st["Q"], ts, qs))
    ctx.tables["gf_snap_step"] = int(ctx.ntimestep)


def _tile(ctx, fn, ncol):
    import cupy as cp
    per_sm = int(os.environ.get("NAMGPU_GF_BLOCKS_PER_SM") or ctx.cfg.raw.get("gf_blocks_per_sm", 4))
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    return int(min(ncol, max(1, sms * per_sm) * GF_BLOCK))


def run(ctx):
    """GFDRV over every column of the (band's) domain; writes the CUCNVC arrays the post kernel reads."""
    import cupy as cp

    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > GF_KMAX:
        raise RuntimeError(f"phys-gf-gfdrv: lm={lm}; phys_gf_gf.cu is compiled for GF_KMAX={GF_KMAX}")
    raw = cfg.raw
    ncnvc = int(cfg.nprecip)
    dt = np.float32(cfg.dt)
    dtc = np.float32(np.float32(ncnvc) * dt)
    ishallow = int(raw.get("gf_ishallow", 1))
    momentum = 1 if _flag(raw, "gf_momentum", True) else 0
    snap = ctx.tables.get("gf_snap_step")
    fvalid = 1 if (snap is not None and "GF_TSNAP" in st.fields
                   and int(snap) == int(ctx.ntimestep) - ncnvc) else 0
    deep = st.add("GF_DEEP_COUNT", (jm, im))
    shal = st.add("GF_SHAL_COUNT", (jm, im))
    ts = st["GF_TSNAP"] if fvalid else st["T"]
    qs = st["GF_QSNAP"] if fvalid else st["Q"]

    pack = kernels.get(UNIT, "gf_pack")
    unpack = kernels.get(UNIT, "gf_unpack")
    gfdrv = kernels.get(GF_UNIT, "gf_gfdrv_stage")
    ncol = im * jm
    tile = _tile(ctx, gfdrv, ncol)
    f32, i32 = np.float32, np.int32
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        lvin = cp.empty((tile, NIN_LEV, lm), dtype=cp.float32)
        scin = cp.empty((tile, NIN_SCA), dtype=cp.float32)
        iin = cp.empty((tile, 3), dtype=cp.int32)
        lev = cp.empty((tile, NLEV, lm), dtype=cp.float32)
        sca = cp.empty((tile, NSCA), dtype=cp.float32)
        isc = cp.empty((tile, NISCA), dtype=cp.int32)
        nblk_ws = (tile + GF_BLOCK - 1) // GF_BLOCK
        ws = cp.empty(nblk_ws * GFWS_SLOTS * (GF_KMAX + 9) * GF_BLOCK, dtype=cp.float32)
        dxh = st["BMJ_DXH"]
        for lo in range(0, ncol, tile):
            n = min(tile, ncol - lo)
            blocks = ((n + 127) // 128,)
            pack(blocks, (128,), (
                i32(im), i32(jm), i32(lm), i32(lo), i32(n), i32(fvalid), dtc, i32(ishallow), i32(0),
                st["U_PHY"], st["V_PHY"], st["W"], st["T"], st["Q"],
                st["PHMID"], st["EXNER"], st["RR"], st["DZ"], st["PHINT"], ts, qs,
                st["FIS"], st["TWBS"], st["QWBS"], st["XLAND"], dxh, st["KPBL"],
                lvin, scin, iin))
            gfdrv(((n + GF_BLOCK - 1) // GF_BLOCK,), (GF_BLOCK,), (
                lvin, scin, iin, lev, sca, isc, ws, i32(0), i32(n), i32(lm)))
            unpack(blocks, (128,), (
                i32(im), i32(jm), i32(lm), i32(lo), i32(n), f32(dt), dtc, i32(momentum),
                lev, sca, isc, st["Q"],
                st["RTHCUTEN"], st["RQCUTEN"], st["DUDT_PHY"], st["DVDT_PHY"],
                st["RAINCV"], st["CUTOP"], st["CUBOT"], st["CW"], st["F_ICE"], st["F_RAIN"],
                deep, shal))
        del lvin, scin, iin, lev, sca, isc, ws
