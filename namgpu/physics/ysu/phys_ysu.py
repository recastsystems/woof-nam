"""phys-ysu: the YSU PBL (WRF bl_pbl_physics = 1) in place of MYJPBL, one thread per column.

Kernel: namgpu/kernels/phys_ysu_column.cu, ArWen's WRF v4.6.1 transcription with three edits: it reads and writes
NMMB's top-down [LM, JM, IM] arrays in place through a flipped level stride (no copies), it takes NMMB's radiative
temperature tendencies RLWTT + RSWTT (= rthraten * exner, all the scheme uses) for the top-down mixing, and exch_m is
optional. Called by phys-turbl-driver in MYJPBL's slot (step 6), on the PBL gate, with DTBL = NPHS * DT.

WRF argument -> NMMB name: U, V <- U_PHY, V_PHY; THETA <- TH; QV <- Q and QC <- QC (NMMB's specific humidity: the
scheme's operators are linear in the tracer and its surface source is QFX / rho, the flux of specific humidity, and
thv = theta (1 + 0.608 q) is the specific-humidity form); QI <- zeros (fer_hires runs F_QI false); P <- PHMID;
P8W <- PHINT; PI <- EXNER; DZ <- DZ; PSFC <- PHINT(LM+1); ZNT <- SL_ZNT (the SFCLAY roughness); UST <- USTAR; HFX, QFX <- TWBS, QWBS after
NOAHLSM (W m-2 and kg m-2 s-1, upward); WSPD, BR, PSIM = FM, PSIH = FH, U10, V10 from phys-sfclayrev (WRF's PBL driver
binds PSIM/PSIH to the full similarity denominators). Outputs: DUDT, DVDT (H points, m s-2), RTHBLTEN, RQBLTEN,
RQCBLTEN, RQIBLTEN (zero), EXCH_H, PBLH; LPBL then comes from PBLH through TURBL's own non-MYJ branch (turbl_lpbl).
Q2 (MYJ's 2 TKE) is not a YSU variable and is left as it is (floored at EPSQ2 by TURBL each call).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_ysu_column"
KERNEL = "ysu_column"
KMAX = 128                 # YSU_KMAX
LANES = 32                 # YSUWS_LANES: the launch block, fixed by the kernel's workspace interleave
SLOTS = 18                 # YSUWS_SLOTS
BLOCKS_PER_SM = 16         # ArWen's measured tile (gpuwm physics_inventory.YSU_TILE_BLOCKS_PER_SM)
TOPDOWN_PBLMIX = 1         # WRF v4.6.1 ysu_topdown_pblmix default
G = 9.80665
_REPORTED = False


def topo_wind(ctx):
    """WRF topo_wind: 0 (WRF default, off) or 1 (Jimenez and Dudhia 2012 subgrid-topography drag). Selected by
    NAMGPU_TOPO_WIND (run --topo-wind). 2 (Ovens-Mass) needs VAR2D and is not offered."""
    import os
    v = int(os.environ.get("NAMGPU_TOPO_WIND", "0") or 0)
    if v not in (0, 1):
        raise RuntimeError(f"phys-ysu: NAMGPU_TOPO_WIND={v} refused: 0 or 1 (2 needs WRF's VAR2D, not built here)")
    return v


def _ctopo(ctx):
    """ctopo, ctopo2 once per run on the device (WRF dyn_em/start_em.F 1537-1611, topo_wind 1). VAR_SSO is the
    variance of 30-arcsec terrain in the cell: here HSTDV**2, the NPS 30-arcsec standard deviation init-gwd-orog
    already builds. lap_hgt from FIS/g with the 4 index neighbours, edges clamped as WRF does."""
    import cupy as cp
    if "YSU_CTOPO" in ctx.tables:
        return ctx.tables["YSU_CTOPO"], ctx.tables["YSU_CTOPO2"]
    st = ctx.st
    for n in ("HSTDV", "FIS", "XLAND"):
        if n not in st.fields:
            raise RuntimeError(f"phys-ysu: topo_wind 1 needs {n} (WRF: 'topo_wind requires VAR_SSO data'); refused")
    hs = st["HSTDV"].astype(cp.float32)
    ht = st["FIS"].astype(cp.float32) / cp.float32(G)
    land = st["XLAND"] < 1.5
    ct = cp.where(land, hs, cp.float32(1.0))
    ct = cp.where(ct <= 2.718, cp.float32(1.0), cp.log(cp.maximum(ct, cp.float32(1e-6))))
    hp = cp.pad(ht, 1, mode="edge")
    lap = (hp[1:-1, 2:] + hp[1:-1, :-2] + hp[2:, 1:-1] + hp[:-2, 1:-1] - ht * 4.0) / 4.0
    ct2 = cp.ones_like(ct)
    alpha = (lap + 20.0) / 10.0
    c_mid = alpha * ct + (1.0 - alpha)
    c_deep = (lap + 30.0) / 10.0
    ct = cp.where(lap > -10.0, ct, cp.where(lap >= -20.0, c_mid, cp.where(lap >= -30.0, c_deep, 0.0)))
    ct2 = cp.where(lap >= -20.0, ct2, cp.where(lap >= -30.0, c_deep, 0.0))
    ct = cp.ascontiguousarray(ct, dtype=cp.float32)
    ct2 = cp.ascontiguousarray(ct2, dtype=cp.float32)
    ctx.tables["YSU_CTOPO"], ctx.tables["YSU_CTOPO2"] = ct, ct2
    global _REPORTED
    if not _REPORTED:   # one host read per run, at the first call: the receipt line for the arm
        _REPORTED = True
        n = int(land.sum())
        print("phys-ysu topo_wind 1: land cells %d, ctopo>1 %.1f%%, ctopo<1 %.1f%%, max %.2f, ctopo2<1 %d cells" % (
            n, 100.0 * float(((ct > 1.0) & land).sum()) / max(n, 1), 100.0 * float(((ct < 1.0) & land).sum()) / max(n, 1),
            float(ct.max()), int((ct2 < 1.0).sum())), flush=True)
    return ct, ct2
REQUIRED = ("U_PHY", "V_PHY", "TH", "Q", "QC", "PHMID", "PHINT", "EXNER", "DZ", "RLWTT", "RSWTT", "SL_ZNT", "USTAR",
            "TWBS", "QWBS", "SL_WSPD", "RIB", "SL_FM", "SL_FH", "XLAND", "U10", "V10", "DUDT", "DVDT", "RTHBLTEN",
            "RQBLTEN", "RQCBLTEN", "RQIBLTEN", "EXCH_H", "PBLH")


def _tile(ctx, ncol):
    import cupy as cp
    key = "YSU_TILE"
    if key not in ctx.tables:
        sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
        ctx.tables[key] = sms * BLOCKS_PER_SM * LANES
    return int(max(LANES, min(int(ncol), ctx.tables[key])))


def _workspace(ctx, lm, tile):
    import cupy as cp
    key = f"YSU_WS_{lm}_{tile}"
    if key not in ctx.tables:
        blocks = (tile + LANES - 1) // LANES
        ctx.tables[key] = cp.empty(blocks * SLOTS * (lm + 1) * LANES, dtype=cp.float32)
    return ctx.tables[key]


def run(ctx, dtbl):
    st, g = ctx.st, ctx.grid
    missing = [n for n in REQUIRED if n not in st.fields]
    if missing:
        raise RuntimeError("phys-ysu: state fields missing: %s (refused, no synthetic fallback)" % ", ".join(missing))
    jm, im = (int(s) for s in st["TSFC"].shape)
    lm = int(g.lm)
    if not 4 <= lm <= KMAX:
        raise RuntimeError(f"phys-ysu: lm={lm}; the kernel holds 4..{KMAX} levels (YSU_KMAX)")
    s2 = (jm, im)
    kpbl = st.add("YSU_KPBL", s2, "int32")
    cloud = st.add("YSU_CLOUDFLG", s2, "int32")
    wstar, delta = st.add("YSU_WSTAR", s2), st.add("YSU_DELTA", s2)
    tdr, w32 = st.add("YSU_TOPDOWN_RADSUM", s2), st.add("YSU_WSTAR3_2", s2)
    f = st.fields
    ncol = im * jm
    tile = _tile(ctx, ncol)
    ws = _workspace(ctx, lm, tile)
    k = kernels.get(UNIT, KERNEL)
    tw = topo_wind(ctx)
    ctopo = ctopo2 = None
    if tw == 1:
        ctopo, ctopo2 = _ctopo(ctx)
    i32, f32 = np.int32, np.float32
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            k(((count + LANES - 1) // LANES, 1, 1), (LANES, 1, 1), (
                f["U_PHY"], f["V_PHY"], f["TH"], f["Q"], f["QC"], f["RQIBLTEN"], f["PHMID"], f["PHINT"],
                f["EXNER"], f["DZ"], f["RLWTT"], f["RSWTT"],
                f["PHINT"][lm], f["SL_ZNT"], f["USTAR"], f["TWBS"], f["QWBS"], f["SL_WSPD"], f["RIB"],
                f["SL_FM"], f["SL_FH"], f["XLAND"], f["U10"], f["V10"],
                f["DUDT"], f["DVDT"], f["RTHBLTEN"], f["RQBLTEN"], f["RQCBLTEN"], f["RQIBLTEN"],
                f["PBLH"], kpbl, f["EXCH_H"], np.uint64(0),
                wstar, delta, f32(dtbl), tdr, w32, cloud,
                i32(TOPDOWN_PBLMIX), i32(lm), i32(jm), i32(im),
                ws, i32(lm + 1), i32(lo), ctopo if ctopo is not None else np.uint64(0)))
        if ctopo2 is not None:
            # bl_ysu.F90 1402-1408: in deep valleys U10/V10 blend toward the lowest-layer wind (pre-mixing)
            k1 = lm - 1
            for w10, wl in (("U10", "U_PHY"), ("V10", "V_PHY")):
                f[w10][...] = ctopo2 * f[w10] + (1.0 - ctopo2) * f[wl][k1]
