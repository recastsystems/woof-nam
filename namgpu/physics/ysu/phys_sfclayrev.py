"""phys-sfclayrev: the revised MM5 surface layer (WRF sf_sfclay_physics = 1) in place of JSFC.

Scheme kernel: namgpu/kernels/phys_sfclayrev.cu (ArWen's WRF v4.6.1 sf_sfclayrev transcription, unchanged).
Glue: namgpu/kernels/phys_ysu_glue.cu. Called by phys-turbl-driver in the surface block in JSFC's slot, on the
same NPHS gate, in WRF's surface-driver order: run() (SFCLAY), then NOAHLSM and SFCDIAGS, then diag().

WRF argument -> NMMB TURBL name:
  inputs   U, V <- U_PHY, V_PHY (H-point winds, lowest level); T <- T; QV <- Q / (1 - Q) (mixing ratio);
           P <- PHMID; DZ8W <- DZ; PSFC <- PHINT(LM+1); TSK <- TSFC; PBLH; MAVAIL; XLAND (1 land, 2 water)
  inout    ZNT <- SL_ZNT (land: Noah table Z0 of the vegetation class, as WRF's Noah route; water: the
           Charnock value, written back to Z0), UST <- USTAR, QSFC <- QSH (mixing ratio, as NOAHLSM writes it), and the scheme's own
           step-to-step MOL, ZOL, HFX, QFX (SL_MOL, SL_ZOL, SL_HFX, SL_QFX: whole-domain state, never band-local)
  outputs  CHS, CHS2, CQS2, QGH, CPM, FLHC, FLQC, RMOL, RIB <- BR, ELFLX <- LH, U10, V10, PSIM, PSIH,
           GZ1OZ0; FM, FH and WSPD for YSU (SL_FM, SL_FH, SL_WSPD); AKHS = CHS and AKMS = UST^2 / WSPD for the
           history names MYJ used to fill; TWBS / QWBS = HFX / QFX handed to NOAHLSM (water points keep them).
diag() after SFCDIAGS: TSHLTR / QSHLTR / PSHLTR (the history's T2 and Q2) from SFCDIAGS's TH2X / Q2X, as WRF's Noah
route writes T2/Q2; TH10 / Q10; and the post-LSM HFX / QFX kept for the next SFCLAY call (WRF's INOUT).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_sfclayrev"
GLUE = "phys_ysu_glue"
OPTION = 1                 # revised MM5 (sf_sfclayrev); 91 = the classic MM5 scheme in the same kernel
ISFTCFLX = 0
IZ0TLND = 0
A_EARTH = 6376000.0
DTR = np.pi / 180.0

# SFCLAY_OUTPUTS order of the kernel (gpuwm/core/physics_inventory.py) -> NMMB state name.
OUTPUTS = (
    ("znt", "SL_ZNT"), ("ust", "USTAR"), ("mol", "SL_MOL"), ("hfx", "SL_HFX"), ("qfx", "SL_QFX"), ("qsfc", "QSH"),
    ("zol", "SL_ZOL"), ("regime", "SL_REGIME"), ("psim", "PSIM"), ("psih", "PSIH"), ("fm", "SL_FM"),
    ("fh", "SL_FH"), ("lh", "ELFLX"), ("u10", "U10"), ("v10", "V10"), ("th2", "SL_TH2"), ("t2", "SL_T2"),
    ("q2", "SL_Q2"), ("chs", "CHS"), ("chs2", "CHS2"), ("cqs2", "CQS2"), ("flhc", "FLHC"), ("flqc", "FLQC"),
    ("qgh", "QGH"), ("rmol", "RMOL"), ("wspd", "SL_WSPD"), ("br", "RIB"), ("gz1oz0", "GZ1OZ0"), ("cpm", "CPM"),
    ("ck", "SL_CK"), ("cka", "SL_CKA"), ("cd", "SL_CD"), ("cda", "SL_CDA"),
)
REQUIRED = ("U_PHY", "V_PHY", "T", "Q", "PHMID", "DZ", "PHINT", "TSFC", "PBLH", "MAVAIL", "XLAND", "Z0", "USTAR",
            "IVGTYP")
SHELTER = ("TSHLTR", "QSHLTR", "PSHLTR", "TH10", "Q10")


def _need(st, names):
    missing = [n for n in names if n not in st.fields]
    if missing:
        raise RuntimeError("phys-sfclayrev: state fields missing: %s (refused, no synthetic fallback)"
                           % ", ".join(missing))


def _dx(ctx):
    key = "SFCLAYREV_DX"
    if key not in ctx.tables:
        ctx.tables[key] = np.float32(A_EARTH * float(ctx.cfg.dlmd) * DTR)   # the rotated grid's along-row step
    return ctx.tables[key]


def _z0tbl(ctx):
    """The Noah background roughness by vegetation class (phys-noah-redprm's device table, the one SFLX reads):
    over land WRF's Noah route gives SFCLAY this ZNT, not NMMB's orography-inclusive Z0."""
    from namgpu.physics.noah import phys_noah_redprm as rp
    tabs, _ = rp._bound_tables(ctx)
    return tabs["Z0TBL"]


def _shape(ctx):
    st, g = ctx.st, ctx.grid
    jm, im = (int(s) for s in st["TSFC"].shape)
    return im, jm, int(g.lm)


def run(ctx):
    st = ctx.st
    _need(st, REQUIRED)
    im, jm, lm = _shape(ctx)
    s2 = (jm, im)
    for _, name in OUTPUTS:
        st.add(name, s2)
    qv1 = st.add("SL_QV1", s2)
    lake = st.add("SL_LAKE", s2)
    f = st.fields
    n = im * jm
    i32 = np.int32
    z0tbl = _z0tbl(ctx)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    k1 = lm - 1
    with scope:
        kernels.get(GLUE, "sfclayrev_in")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["Q"], qv1, lake, i32(im), i32(jm), i32(lm)))
        kernels.get(GLUE, "sfclayrev_znt")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["IVGTYP"], f["XLAND"], z0tbl, f["Z0"], f["SL_ZNT"], i32(int(z0tbl.size)), i32(im), i32(jm)))
        ins = (f["U_PHY"][k1], f["V_PHY"][k1], f["T"][k1], qv1, f["PHMID"][k1], f["DZ"][k1], f["PHINT"][lm],
               f["TSFC"], f["PBLH"], f["MAVAIL"], f["XLAND"], lake)
        outs = tuple(f[name] for _, name in OUTPUTS)
        kernels.get(UNIT, "sfclay_column")(((n + 127) // 128, 1, 1), (128, 1, 1), ins + outs + (
            _dx(ctx), i32(OPTION), i32(1), i32(ISFTCFLX), i32(IZ0TLND), i32(n)))
        kernels.get(GLUE, "sfclayrev_znt_back")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["XLAND"], f["SL_ZNT"], f["Z0"], i32(im), i32(jm)))
        kernels.get(GLUE, "sfclayrev_out")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["SL_HFX"], f["SL_QFX"], f["CHS"], f["USTAR"], f["SL_WSPD"],
            f["TWBS"], f["QWBS"], f["AKHS"], f["AKMS"], i32(im), i32(jm)))


def diag(ctx):
    st = ctx.st
    im, jm, lm = _shape(ctx)
    for name in SHELTER:
        st.add(name, (jm, im))
    _need(st, ("TH2X", "Q2X", "TH", "TWBS", "QWBS"))
    f = st.fields
    i32 = np.int32
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(GLUE, "sfclayrev_diag")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["TH2X"], f["Q2X"], f["PHINT"], f["TH"], f["Q"], f["DZ"], f["TWBS"], f["QWBS"],
            f["TSHLTR"], f["QSHLTR"], f["PSHLTR"], f["TH10"], f["Q10"], f["SL_HFX"], f["SL_QFX"],
            i32(im), i32(jm), i32(lm)))
