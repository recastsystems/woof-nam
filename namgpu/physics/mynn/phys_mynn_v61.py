"""phys-mynn-v61 (Route A): the NEWEST MYNN pair, hex's MYNN-EDMF v6.1.1 + MYNN-SFC v6.1.0, in the NMMB TURBL slot.

Selected with NAMGPU_PBL=mynn-v61 (default myj: the frozen-NAM path is untouched). phys_mynn_edmf stays the facade
the TURBL driver and the BMJ pairing rule call; it hands every call to this module when this row is selected.

Schemes (hex branch hex/mynn-v61-option, exact vs native in module replay; Apache-2.0, NCAR/MYNN-EDMF 2b2da61 and
NCAR/MYNN-SFC a038a9e):
  surface layer : hexcore.mpasphys.rrfs.sf_mynnsfc_v610.run_packed   (the MPAS driver, MPAS presence mask)
  PBL           : hexcore.mpasphys.rrfs.bl_mynnedmf_v611.MynnEdmfCuda (the unified v6 driver, RRFS hrrrv5 options:
                  option_arguments({}) = closure 2.6, cloudpdf 2, mixlength 1, edmf 1, edmf_mom 1, edmf_tke 0,
                  cloudmix 1, mixqt 0, ess 1, tkeadvect off, dd off)
Column contract as the NMMB<->ArWen adapter (b07096a): NMMB [L, J, I] with L = 0 the TOP -> MPAS Fortran (i, k, j)
with i = J*IM + I (all columns on one i axis), j = 1, k = 0 the LOWEST layer; kte = LM, kme = LM + 1 (the MPAS
allocation: the interface arrays pint and w use level LM+1). Q -> qv = Q/(1-Q), QC -> qc = QC/(1-Q); tendencies go
back in specific units (x (1-Q)). rthraten = (RLWTT + RSWTT) / EXNER (the NAM radiation heating as a theta rate).
pint = PHINT (TURBL's hydrostatic interface pressure), w = NMMB layer W averaged to interfaces (0 at the ground).
MYNN state (qke, tsq, qsq, cov, el_pbl, sh3d, sm3d, cldfra_bl, qc_bl, qi_bl, qke_adv, kpbl) and the whole MYNN-SFC
field pack are held on the GPU for the run (run a forecast in one segment, as the older MYNN arm). All GPU, no CPU
path; the host only launches (one error-flag read on the first call of each scheme).
"""
from __future__ import annotations

import os

import numpy as np

NAME = "mynn-v61"
_C = {}


def _hex():
    if not _C:
        import sys
        extra = os.environ.get("NAMGPU_HEXCORE_PATH")      # the hex tree's src (hexcore) when it is not installed
        if extra and extra not in sys.path:
            sys.path.insert(0, extra)
        from hexcore import numerics
        try:
            numerics.set_mode(os.environ.get("NAMGPU_MYNN_NUMERICS", "fast"), source="nam-mynn-v61")
        except Exception:
            pass
        from hexcore.mpasphys.rrfs import bl_mynnedmf_v611 as v611
        from hexcore.mpasphys.rrfs import sf_mynnsfc_v610 as v610
        _C.update(v611=v611, v610=v610, edmf=v611.MynnEdmfCuda(), held={})
    return _C


_PROF = os.environ.get("NAMGPU_MYNN61_PROFILE", "") == "1"   # one-off profile runs only (CUDA events + a sync)


class _Prof:
    """NAMGPU_MYNN61_PROFILE=1: CUDA events around each piece of the two calls, medians printed every 50 calls.
    Never on a timed run (the read-back syncs); the default path has no events and no syncs."""

    def __init__(self, tag):
        self.tag, self.ev = tag, []

    def mark(self, name):
        if _PROF:
            import cupy as cp
            e = cp.cuda.Event()
            e.record()
            self.ev.append((name, e))

    def done(self, ncall):
        if not _PROF or len(self.ev) < 2:
            return
        self.ev[-1][1].synchronize()
        import cupy as cp
        acc = _hex()["held"].setdefault(("prof", self.tag), {})
        for (n0, e0), (n1, e1) in zip(self.ev, self.ev[1:]):
            acc.setdefault(n1, []).append(cp.cuda.get_elapsed_time(e0, e1))
        acc.setdefault("TOTAL", []).append(cp.cuda.get_elapsed_time(self.ev[0][1], self.ev[-1][1]))
        if ncall % 50 == 0:
            e = _hex()["edmf"]
            print(f"mynn-v61 profile {self.tag} call {ncall}: " + ", ".join(
                f"{k} {float(np.median(v)):.1f} ms" for k, v in acc.items())
                + f" | edmf lanes {getattr(e, 'last_lane_width', '-')} batch {getattr(e, 'last_batch_columns', '-')}"
                + f" stride {_hex().get('stride', '-')} B", flush=True)


def _held(name, shape, dtype=np.float32, order="C"):
    import cupy as cp
    h = _hex()["held"]
    key = (name, tuple(shape), np.dtype(dtype).str, order)
    b = h.get(key)
    if b is None:
        b = cp.zeros(shape, dtype=dtype, order=order)
        h[key] = b
    return b


def _ncall(ctx):
    n = ctx.tables.get("MYNN61_NCALL")
    if n is None:
        n = [0, 0]
        ctx.tables["MYNN61_NCALL"] = n
    return n


def _mask(jm, im):
    import cupy as cp
    h = _hex()["held"]
    m = h.get(("B1", jm, im))
    if m is None:
        m = cp.zeros((jm, im), dtype=cp.bool_)
        m[1:jm - 1, 1:im - 1] = True
        h[("B1", jm, im)] = m
    return m


def _dx(ctx):
    return float(np.mean(np.asarray(ctx.grid.dxh, dtype=np.float64)))


# ------------------------------------------------------------------------------------------------ surface layer
def run_sfc(ctx):
    """MYNN-SFC v6.1.0 in JSFC's place (before NOAHLSM)."""
    import cupy as cp
    H = _hex()
    v610 = H["v610"]
    FI = v610.FIELDS.index
    st = ctx.st
    f = st.fields
    lm, jm, im = (int(s) for s in f["T"].shape)
    n, s2 = jm * im, (jm, im)
    for nm in ("USTAR", "Z0", "QSH", "RMOL", "U10", "V10", "TH10", "Q10", "TSHLTR", "QSHLTR", "PSHLTR",
               "AKHS", "AKMS", "FLHC", "FLQC", "ELFLX", "RIB", "PSIM", "PSIH", "PBLH", "CT"):
        st.add(nm, s2)
    for nm in ("MYNNB_HFX", "MYNNB_QFX", "MYNNB_WSPD", "MYNN61_CH"):
        st.add(nm, s2)
    nc = _ncall(ctx)
    first = nc[0] == 0
    nc[0] += 1
    pk = _held("sfc_pack", (len(v610.FIELDS), n))
    pr = _Prof("sfc")
    pr.mark("start")

    def put(name, val):
        pk[FI(name)].reshape(s2)[...] = val

    kb, k2 = lm - 1, lm - 2
    q1 = f["Q"][kb]
    if first:
        pk.fill(0)
        put("ust", cp.maximum(f["USTAR"], cp.float32(1.0e-3)))
        put("ustm", cp.maximum(f["USTAR"], cp.float32(1.0e-3)))
        put("znt", cp.maximum(f["Z0"], cp.float32(1.0e-4)))
        put("qsfc", f["QSH"])
        put("hfx", f["TWBS"])
        put("qfx", f["QWBS"])
        pk[FI("dx")].fill(np.float32(_dx(ctx)))
    else:
        put("hfx", f["MYNNB_HFX"])
        put("qfx", f["MYNNB_QFX"])
        put("qsfc", f["QSH"])                      # Noah's surface humidity (MPAS passes the LSM's qsfc)
        put("znt", f["Z0"])                        # NAM roughness over land; MYNN's own (Charnock) z0 over water
    put("u3d", f["U_PHY"][kb]); put("v3d", f["V_PHY"][kb]); put("dz8w", f["DZ"][kb])
    put("u3d2", f["U_PHY"][k2]); put("v3d2", f["V_PHY"][k2]); put("dz8w2", f["DZ"][k2])
    put("t3d", f["T"][kb]); put("qv3d", q1 / (1.0 - q1)); put("p3d", f["PHMID"][kb])
    put("th3d", f["TH"][kb]); put("rho3d", f["RR"][kb])
    put("psfcpa", f["PHINT"][lm]); put("pblh", f["PBLH"]); put("mavail", f["MAVAIL"])
    put("xland", f["XLAND"]); put("tsk", f["TSFC"]); put("snowh", f["SNOWH"]); put("xice", f["SICE"])
    put("tsurf", f["TSFC"])
    step = int(nc[0])
    initflag = 1 if first else 0
    vals = dict(ivegsrc=0, redrag=0, sfc_z0_type=0, spp_pbl=0, sf_mynn_sfcflux_water=0, sf_mynn_sfcflux_land=0,
                shalwater_z0=0, isfflx=1, restart=0, cycling=0, initflag=initflag, flag_iter=1, flagc_lsm=1,
                itimestep=step)
    opts = H["held"].get("sfc_opts")
    if opts is None:
        opts = cp.asarray([vals[k] for k in v610.OPTIONS], cp.int32)
        H["held"]["sfc_opts"] = opts
    pr.mark("pack")
    err = v610.run_packed(pk, opts, errors=_held("sfc_err", (n,), np.int32), itimestep=step, initflag=initflag)
    if first:
        bad = int(cp.count_nonzero(err))
        if bad:
            raise RuntimeError(f"MYNN-SFC v6.1.0 errflg set in {bad} columns on the first call")
    pr.mark("mynnsfc")
    m = _mask(jm, im)

    def get(name):
        return pk[FI(name)].reshape(s2)

    def cpy(dst, val):
        cp.copyto(f[dst], val, where=m)

    cpy("CHS", get("chs")); cpy("CHS2", get("chs2")); cpy("CQS2", get("cqs2")); cpy("CPM", get("cpm"))
    cpy("QGH", get("qgh")); cpy("FLHC", get("flhc")); cpy("FLQC", get("flqc"))
    cpy("TWBS", get("hfx")); cpy("QWBS", get("qfx")); cpy("ELFLX", get("lh"))
    cpy("QSH", get("qsfc")); cpy("USTAR", get("ust")); cpy("Z0", get("znt")); cpy("RMOL", get("rmol"))
    cpy("RIB", get("br")); cpy("PSIM", get("psim")); cpy("PSIH", get("psih"))
    cpy("AKHS", get("chs"))
    wsp = cp.maximum(get("wspd"), cp.float32(0.1))  # whole field: the PBL refuses wspd <= 0 (outer ring)
    cpy("AKMS", get("ust") * get("ust") / wsp)      # = cd * wspd
    cpy("U10", get("u10")); cpy("V10", get("v10"))
    cp.copyto(f["MYNNB_WSPD"], wsp)
    cp.copyto(f["MYNN61_CH"], get("ch"))
    pr.mark("unpack")
    pr.done(nc[0])
    return pk


# ------------------------------------------------------------------------------------------------ PBL
_STATE = ("qke", "qke_adv", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d", "cldfra_bl", "qc_bl", "qi_bl")
_OUT3 = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "exch_h", "exch_m")
_IN3 = ("dz", "u", "v", "w", "th", "tk", "p", "exner", "rho", "qv", "qc", "rthraten", "pint")
_IN2 = ("dx", "xland", "ps", "ts", "qsfc", "ust", "ch", "hfx", "qfx", "wspd", "znt", "uoce", "voce", "pblh")
_OUT2 = ("maxwidth", "maxmf", "ztop_plume", "excess_h", "excess_q", "maxwidth_dd", "maxmf_dd", "maxtkeprod",
         "cldtop_cooling", "ent_eff")


def run_pbl(ctx, dtphs):
    """MYNN-EDMF v6.1.1 in MYJPBL's place."""
    import cupy as cp
    H = _hex()
    v611 = H["v611"]
    st = ctx.st
    f = st.fields
    lm, jm, im = (int(s) for s in f["T"].shape)
    n, s2, s3 = jm * im, (jm, im), (lm, jm, im)
    for nm in ("XLEN_MIX", "EXCH_H"):
        st.add(nm, s3)
    st.add("LPBL", s2, dtype="int32")
    nc = _ncall(ctx)
    first = nc[1] == 0
    nc[1] += 1
    A = {nm: _held("e3_" + nm, (n, lm + 1, 1), order="F") for nm in _IN3 + _STATE + _OUT3}
    B = {nm: _held("e2_" + nm, (n, 1), order="F") for nm in _IN2 + _OUT2}
    B["kpbl"] = _held("e2_kpbl", (n, 1), np.int32, order="F")
    pr = _Prof("pbl")
    pr.mark("start")

    def bot(a):                      # (lm+1, J, I) C-contiguous view of an (n, lm+1, 1) F array, bottom first
        return a.T[0].reshape(lm + 1, jm, im)

    def fill(a, top_first):          # NMMB [L] top-first -> MPAS k bottom-first; level kme copies kte
        v = bot(a)
        v[:lm] = top_first[::-1]
        v[lm] = v[lm - 1]

    def two(a):
        return a.T[0].reshape(jm, im)

    one_m_q = 1.0 - f["Q"]
    fill(A["dz"], f["DZ"]); fill(A["u"], f["U_PHY"]); fill(A["v"], f["V_PHY"]); fill(A["th"], f["TH"])
    fill(A["tk"], f["T"]); fill(A["p"], f["PHMID"]); fill(A["exner"], f["EXNER"]); fill(A["rho"], f["RR"])
    fill(A["qv"], f["Q"] / one_m_q); fill(A["qc"], f["QC"] / one_m_q)
    if "RLWTT" in f and "RSWTT" in f:
        fill(A["rthraten"], (f["RLWTT"] + f["RSWTT"]) / f["EXNER"])
    bot(A["pint"])[...] = f["PHINT"][::-1]
    wl = f["W"][::-1]
    wi = bot(A["w"])
    wi[0] = 0
    wi[1:lm] = 0.5 * (wl[:-1] + wl[1:])
    wi[lm] = wl[lm - 1]
    if first:
        B["dx"].fill(np.float32(_dx(ctx)))
        two(B["pblh"])[...] = f["PBLH"]
    two(B["xland"])[...] = f["XLAND"]; two(B["ps"])[...] = f["PHINT"][lm]; two(B["ts"])[...] = f["TSFC"]
    two(B["qsfc"])[...] = f["QSH"]; two(B["ust"])[...] = f["USTAR"]; two(B["ch"])[...] = f["MYNN61_CH"]
    two(B["hfx"])[...] = f["TWBS"]; two(B["qfx"])[...] = f["QWBS"]; two(B["wspd"])[...] = f["MYNNB_WSPD"]
    two(B["znt"])[...] = f["Z0"]

    vals = dict(ids=1, ide=n, jds=1, jde=1, kds=1, kde=lm + 1, ims=1, ime=n, jms=1, jme=1, kms=1, kme=lm + 1,
                its=1, ite=n, jts=1, jte=1, kts=1, kte=lm,
                f_qc=True, f_qi=False, f_qs=False, f_qoz=False, f_qnc=False, f_qni=False, f_qnifa=False,
                f_qnwfa=False, f_qnbca=False,
                initflag=1 if first else 0, restart=False, cycling=False, delt=float(dtphs),
                mix_chem=False, nchem=0, ndvel=0)
    vals.update(v611.option_arguments({}))
    vals.update(A)
    vals.update(B)
    pr.mark("gather")
    if _PROF and "stride" not in H:
        H["stride"] = v611.workspace_bytes(vals)
    out = H["edmf"].run(vals)
    pr.mark("edmf")
    if first:
        bad = int(cp.count_nonzero(out["errflg"]))
        if bad:
            raise RuntimeError(f"MYNN-EDMF v6.1.1 errflg set in {bad} columns on the first call")

    def top(a):                      # bottom-first layers -> NMMB top-first view
        return bot(a)[:lm][::-1]

    m2 = _mask(jm, im)
    m3 = cp.broadcast_to(m2, s3)

    def sel(name, val):
        dst = f[name]
        if dst.shape[0] == lm + 1:
            dst = dst[1:]            # NMMB interface array: MYNN level k is the bottom interface of its layer
        cp.copyto(dst, val, where=m3)

    sel("DUDT", top(A["rublten"]))
    sel("DVDT", top(A["rvblten"]))
    sel("RTHBLTEN", top(A["rthblten"]))
    sel("RQBLTEN", top(A["rqvblten"]) * one_m_q)
    sel("RQCBLTEN", top(A["rqcblten"]) * one_m_q)
    sel("EXCH_H", top(A["exch_h"]))
    sel("XLEN_MIX", top(A["el_pbl"]))
    sel("Q2", cp.maximum(top(A["qke"]), cp.float32(0.02)))
    cp.copyto(f["PBLH"], two(B["pblh"]), where=m2)
    cp.copyto(f["LPBL"], (lm + 1 - two(B["kpbl"])).astype(cp.int32), where=m2)
    pr.mark("scatter")
    pr.done(nc[1])
    return out
