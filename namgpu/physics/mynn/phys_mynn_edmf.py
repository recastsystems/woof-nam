"""phys-mynn-edmf: MYNN-EDMF PBL + MYNN surface layer in the NMMB TURBL slot (NAM modernisation arm).

Replaces the MYJ pair (JSFC surface layer + MYJPBL) with the WRF v4.6.1 MYNN pair as ArWen runs it on the GPU:
  surface layer : gpuwm.core.mynn_sfclay.launch_mynn_surface_layer  (module_sf_mynn.F SFCLAY1D_mynn, isftcflx 0)
  PBL           : gpuwm.core.mynn_pbl_runtime.mynn_pbl_step          (module_bl_mynn_wrapper.F + module_bl_mynn.F,
                  EDMF on: bl_mynn_edmf=1, edmf_mom=1, edmf_tke=0, mixlength=2, cloudpdf=2, cloudmix=1, closure 2.6)
ArWen's MYNN is WRF public-domain lineage (Apache-2.0 port); nothing UCAR-restricted is used.

Selected with NAMGPU_PBL=mynn-edmf (default myj: the frozen-NAM path is untouched, byte for byte).

Column interface (NMMB B-grid -> ARW-like column): NMMB arrays are [LM, JM, IM] with L = 1 at the TOP; MYNN wants
(nz, ny, nx) bottom-first. Every 3-D input is flipped into a bottom-first working buffer (one copy, on the GPU):
  dz <- DZ, u/v <- U_PHY/V_PHY (TURBL's H-point averages), theta <- TH, p <- PHMID, exner <- EXNER, rho <- RR,
  T <- T, qv <- Q/(1-Q) (NMMB Q is specific humidity; MYNN's wrapper takes mixing ratio and converts back),
  qc <- QC/(1-Q), qi <- 0 (fer_hires carries no separate cloud ice: F_QI false), qs <- QS (flag_qs false, as WRF
  gives Ferrier), w <- W (NMMB layer w; MYNN reads the lower-interface w only in its plume diagnostics).
MYNN's carried state (qke, tsq, qsq, cov, el, sh, sm, qc_bl, qi_bl, cldfra_bl, exch_h, exch_m, kpbl and the plume
diagnostics) is held bottom-first under MYNNB_* names for the whole run (not in restart files: run a forecast in one
segment). Outputs go back top-first to the NMMB names the rest of TURBL and the solver read:
  DUDT, DVDT (H points, m s-2, then drv-h-to-v-tend), RTHBLTEN (K s-1), RQBLTEN and RQCBLTEN (specific, s-1:
  MYNN's mixing-ratio rates times (1-Q), the inverse of its wrapper's conversion), PBLH, LPBL (= LM+1-KPBL),
  EXCH_H, XLEN_MIX (el), Q2 (= qke, 2 TKE, written for output; MYNN keeps its own copy, not advected).
Surface layer -> Noah hand-off (the same names JSFC wrote): CHS, CHS2, CQS2, CPM, QGH, FLHC, FLQC, TWBS (HFX),
  QWBS (QFX), ELFLX (LH), QSH (qsfc), USTAR, Z0 (znt; MYNN updates it over water only), RMOL, RIB, PSIM, PSIH,
  AKHS (= chs), AKMS (= cd * wspd), U10, V10 (H points, grid-relative, as JSFC's). After Noah and SFCDIAGS, the 2 m
  diagnostics come from SFCDIAGS (WRF's Noah path): TSHLTR = T2 (1e5/PSHLTR)^kappa, QSHLTR = Q2X,
  PSHLTR = PSFC exp(-2 g / (R_d T2)).
Tendencies and fluxes are written on the B1 tile only (i0 in [1, IM-2], j0 in [1, JM-2]), as MYJ/JSFC leave the
outer ring.

Not coupled yet (named gaps): MYNN subgrid clouds (icloud_bl) to the NAM RRTM cloud fraction; qke advection
(bl_mynn_tkeadvect false, WRF's default); MYNN state in restart files.
All GPU; the host only launches. No CPU path.
"""
from __future__ import annotations

import os

import numpy as np

ENV = "NAMGPU_PBL"
NAME = "mynn-edmf"

G = np.float32(9.80665)
R_D = np.float32(287.04)
CAPPA = np.float32(287.04 / 1004.6)

# The MYNN identity this arm runs (HRRR / RRFS lineage settings that ArWen's driver admits).
MYNN_OPTIONS = dict(
    closure=2.6, bl_mynn_cloudpdf=2, bl_mynn_mixlength=2, bl_mynn_edmf=1, bl_mynn_edmf_mom=1,
    bl_mynn_edmf_tke=0, bl_mynn_mixscalars=0, bl_mynn_cloudmix=1, bl_mynn_mixqt=0, bl_mynn_output=0,
    bl_mynn_tkeadvect=False, icloud_bl=1, flag_qc=True, flag_qi=True,   # ArWen admits only FLAG_QC and FLAG_QI true; qi is a zero column (fer_hires has no separate cloud ice), so its solve stays zero
)
MP_PHYSICS_WRF = 5          # Ferrier: WRF's Registry gives MYNN flag_qs false for it

STATE_3D = ("qke", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d", "qc_bl", "qi_bl", "cldfra_bl",
            "exch_h", "exch_m")


def selected() -> bool:
    v = os.environ.get(ENV, "myj").strip().lower()
    if v in ("", "myj"):
        return False
    if v in ("mynn", "mynn-edmf", "mynnedmf", "mynn-v61"):
        return True
    raise RuntimeError(f"{ENV}={v!r}: rows are 'myj' (NAM, default), 'mynn-edmf' and 'mynn-v61'")


def v61() -> bool:
    """NAMGPU_PBL=mynn-v61: hex's MYNN-EDMF v6.1.1 + MYNN-SFC v6.1.0 (phys_mynn_v61) in place of the older port."""
    return os.environ.get(ENV, "myj").strip().lower() == "mynn-v61"


def bmj_shallow_off() -> bool:
    """EDMF / shallow-convection pairing rule (as lane cu-tiedtke's for New Tiedtke): with MYNN-EDMF on, BMJ shallow
    convection is off (the EDMF updrafts carry shallow cumulus; running both moves the same subcloud heat and moisture
    twice). NAMGPU_BMJ_SHALLOW=on forces it back on (sensitivity arm), =off forces it off; auto is the default."""
    v = os.environ.get("NAMGPU_BMJ_SHALLOW", "auto").strip().lower()
    if v == "on":
        return False
    if v == "off":
        return True
    if v != "auto":
        raise RuntimeError(f"NAMGPU_BMJ_SHALLOW={v!r}: rows are auto, on, off")
    return selected() and int(MYNN_OPTIONS["bl_mynn_edmf"]) == 1


class _Scratch:
    """The DomainState.scratch shape MynnPblScratch.from_state asks for: one persistent buffer per slot."""

    def __init__(self):
        self.bufs = {}

    def scratch(self, shape, slot, dtype=None):
        import cupy as cp
        dtype = cp.float32 if dtype is None else dtype
        key = (slot, tuple(int(s) for s in shape), np.dtype(dtype).str)
        b = self.bufs.get(key)
        if b is None:
            b = cp.zeros(shape, dtype=dtype)
            self.bufs[key] = b
        return b


_CACHE = {}      # device work buffers per process (never in ctx.tables: restart sets walk the tables)


class _Arm(dict):
    """Work buffers (process cache) + the call counters, which live in ctx.tables["MYNN_NCALL"] as plain ints so
    a restart set carries them (a resumed run must not cold-start MYNN again)."""

    def __init__(self, ctx, cache):
        super().__init__(cache)
        self._ctx = ctx

    def _n(self):
        n = self._ctx.tables.get("MYNN_NCALL")
        if n is None:
            n = [0, 0]
            self._ctx.tables["MYNN_NCALL"] = n
        return n

    def __getitem__(self, k):
        if k == "calls_sfc":
            return self._n()[0]
        if k == "calls_pbl":
            return self._n()[1]
        return dict.__getitem__(self, k)

    def __setitem__(self, k, v):
        if k == "calls_sfc":
            self._n()[0] = int(v)
        elif k == "calls_pbl":
            self._n()[1] = int(v)
        else:
            _CACHE[self._key][k] = v
            dict.__setitem__(self, k, v)


def _t(ctx):
    import cupy as cp
    key = int(cp.cuda.runtime.getDevice())
    if key not in _CACHE:
        _CACHE[key] = {"scratch": _Scratch(), "buf": {}}
    a = _Arm(ctx, _CACHE[key])
    a._key = key
    return a


def _buf(t, name, shape, dtype=None):
    import cupy as cp
    dtype = cp.float32 if dtype is None else dtype
    key = (name, tuple(shape))
    b = t["buf"].get(key)
    if b is None:
        b = cp.zeros(shape, dtype=dtype)
        t["buf"][key] = b
    return b


def _mask(t, jm, im):
    import cupy as cp
    m = t["buf"].get(("B1", (jm, im)))
    if m is None:
        m = cp.zeros((jm, im), dtype=cp.bool_)
        m[1:jm - 1, 1:im - 1] = True
        t["buf"][("B1", (jm, im))] = m
    return m


def _ledger(ctx, t):
    """The run's MYNN health ledger (gpuwm.core.health_ledger), registered on ctx.health_ledgers so the driver
    drains it hourly and at the end of the run; None with NAMGPU_MYNN_DEFER=0 (immediate reads)."""
    if os.environ.get("NAMGPU_MYNN_DEFER", "1").strip() == "0":
        return None
    led = t["buf"].get("ledger")
    if led is None:
        from gpuwm.core.health_ledger import HealthLedger
        led = HealthLedger(capacity=16, label="nam-mynn")
        t["buf"]["ledger"] = led
    leds = getattr(ctx, "health_ledgers", None)
    if leds is None:
        leds = []
        ctx.health_ledgers = leds
    if not any(x is led for x in leds):
        leds.append(led)
    return led


def _dx(ctx):
    g = ctx.grid
    return float(np.mean(np.asarray(g.dxh, dtype=np.float64)))


# Land roughness for the MYNN surface layer: WRF Noah (MODIFIED_IGBP_MODIS_NOAH, VEGPARM.TBL Z0MIN/Z0MAX; WRF public
# domain), the table WRF/RAP pair with MYNN-SFC on IGBP land use (NAM ivegsrc 1 uses the same 20 IGBP classes).
# Why: the NAM SFCR roughness (Z0/Z0BASE) is 1.9-2.75 m over forest because JSFC measures its 10 m wind above an
# equivalent roughness height (phys_jsfc_sfcdif.cu "equivalent Z0 for the shelter winds", zu10e = zu_e + 10).
# MYNN's 10 m wind is log(10/z0)/log(za/z0) from the ground, so the NAM z0 made the first MYNN arm's 10 m wind
# -1.4 kt vs ASOS at +6 h (control +0.5); the table overshot (+1.2 kt) and cost T2/Td2, so NAM roughness is the default.
_IGBP_Z0MIN = (0.50, 0.50, 0.50, 0.50, 0.20, 0.01, 0.01, 0.01, 0.15, 0.10,
               0.30, 0.05, 0.50, 0.05, 0.001, 0.01, 0.0001, 0.30, 0.15, 0.05)
_IGBP_Z0MAX = (0.50, 0.50, 0.50, 0.50, 0.50, 0.05, 0.06, 0.05, 0.15, 0.12,
               0.30, 0.15, 0.50, 0.14, 0.001, 0.01, 0.0001, 0.30, 0.15, 0.10)
_Z0_KERNEL = None


def _land_z0(t, f, m):
    """WRF Noah Z0BRD (green-fraction interpolation between Z0MIN and Z0MAX) plus SNOWZ0 burial, on land cells of
    the B1 tile only; water keeps MYNN's own (Charnock) roughness. One elementwise launch, no host sync."""
    import cupy as cp
    global _Z0_KERNEL
    # Default nam (evidence, 2026-01-23 12Z 12 h arms vs ASOS): with the table, 10 m wind went from -1.4 to +1.2 kt
    # bias, wind MAE 2.9 -> 3.0/3.2 kt, T2 MAE at +12 h 3.8 -> 4.0 F, Td2 4.1 -> 4.3 F (cold, decoupled snow valleys
    # where SNOWZ0 drops z0 toward 1 mm). NAMGPU_MYNN_Z0=table selects the WRF Noah table row.
    if os.environ.get("NAMGPU_MYNN_Z0", "nam").strip().lower() != "table":
        return
    tab = t["buf"].get("z0tab")
    if tab is None:
        tab = cp.asarray(np.array(_IGBP_Z0MIN + _IGBP_Z0MAX, dtype=np.float32))
        t["buf"]["z0tab"] = tab
    if _Z0_KERNEL is None:
        _Z0_KERNEL = cp.ElementwiseKernel(
            "int32 ivg, float32 vegfrc, float32 xland, float32 snowh, float32 snowc, bool b1, raw float32 tab",
            "float32 z0",
            """
            if (b1 && xland < 1.5f) {
                int k = ivg - 1; if (k < 0) k = 0; if (k > 19) k = 19;
                float fr = fminf(fmaxf(vegfrc, 0.f), 1.f);
                float z0brd = (1.f - fr) * tab[k] + fr * tab[20 + k];
                float burial = 7.f * z0brd - snowh;
                float z0eff = (burial <= 0.0007f) ? 0.001f : burial / 7.f;
                float sc = fminf(fmaxf(snowc, 0.f), 1.f);
                z0 = (1.f - sc) * z0brd + sc * z0eff;
            }
            """, "namgpu_mynn_land_z0")
    ivg = f["IVGTYP"]
    if ivg.dtype != cp.int32:
        ivg = ivg.astype(cp.int32)
    _Z0_KERNEL(ivg, f["VEGFRC"], f["XLAND"], f["SNOWH"], f["SNOWC"], m, tab, f["Z0"])


# Shelter (10 m) wind over land: NAM's equivalent-roughness convention, inside MYNN's own similarity functions.
# Cause (route C, 2026-01-23 12Z arms v1-v3): NAM's Z0 (the analysis SFCR, GFS-table canopy roughness: 1.9-2.75 m
# over forest) is the right roughness for the DRAG the column feels, but ASOS anemometers sit in open exposure. JSFC
# keeps Z0 for the fluxes and diagnoses the land 10 m wind above an equivalent roughness
# zu_e = max(0.35 z0, min(0.5 z0, 0.18)) (phys_jsfc_sfcdif.cu "equivalent Z0 for the shelter winds"):
# U10 = U1 * simm10e / simm. MYNN-SFC diagnoses U10 with the canopy z0 itself (and the neutral log(10/z0)/log(za/z0)
# when za < 13 m), so v1 (NAM z0) read -1.4 kt vs ASOS where the control read +0.5; swapping in the WRF Noah table
# z0 (v2/v3) changed the DRAG instead (column too fast: +1.2 kt, T2/Td2 and the Red River rain band worse).
# Fix at cause: keep NAM Z0 for the fluxes (v1's column, T2/Td2 at control level) and diagnose the land shelter wind
# the way NAM defines it: U10 = U1 * psix10e / psix, psix = MYNN's ln((za+z0)/z0) - psim (its flux denominator),
# psix10e = ln((10+z0e)/z0e) - [psim(zol(10+z0e)/za) - psim(zol z0e/za)], MYNN's stable/unstable psi forms.
# NAMGPU_MYNN_U10=mynn keeps MYNN's own diagnostic. Water keeps MYNN's (z0 there is Charnock, no canopy).
_U10_KERNEL = None


def _land_u10(t, res, u1, v1, dz1, xland, m, f):
    import cupy as cp
    global _U10_KERNEL
    if os.environ.get("NAMGPU_MYNN_U10", "nam").strip().lower() == "mynn":
        return
    if _U10_KERNEL is None:
        _U10_KERNEL = cp.ElementwiseKernel(
            "float32 u1, float32 v1, float32 dz1, float32 z0, float32 zol, float32 gz1oz0, float32 psim, "
            "float32 xland, bool b1",
            "float32 u10, float32 v10",
            """
            if (b1 && xland < 1.5f) {
                float za = 0.5f * dz1;
                float zuuz = fminf(0.5f * z0, 0.18f);
                float z0e = fmaxf(0.35f * z0, zuuz);
                float ga = logf((10.f + z0e) / z0e);
                float za10 = zol * (10.f + z0e) / za, zae = zol * z0e / za;
                float p10;
                if (zol > 0.f) {
                    p10 = (-6.1f * logf(za10 + powf(1.f + powf(za10, 2.5f), 0.4f)))
                        - (-6.1f * logf(zae + powf(1.f + powf(zae, 2.5f), 0.4f)));
                } else if (zol < 0.f) {
                    p10 = namgpu_psimu(za10) - namgpu_psimu(zae);
                    p10 = fminf(p10, 0.9f * ga);
                } else {
                    p10 = 0.f;
                }
                float psix = fmaxf(gz1oz0 - psim, 0.1f);
                float r = (ga - p10) / psix;
                u10 = u1 * r; v10 = v1 * r;
            }
            """, "namgpu_mynn_land_u10", preamble="""
            __device__ float namgpu_psimu(float z) {   // module_sf_mynn.F psim_unstable_full (psi_opt 0)
                float x = powf(1.f - 16.f * z, 0.25f);
                float k = 2.f * logf(0.5f * (1.f + x)) + logf(0.5f * (1.f + x * x)) - 2.f * atanf(x) + 1.5707963f;
                float y = powf(1.f - 10.f * z, 0.33f);
                float c = 1.5f * logf((y * y + y + 1.f) / 3.f) - 1.7320508f * atanf((2.f * y + 1.f) / 1.7320508f)
                          + 1.8137994f;
                return (k + z * z * c) / (1.f + z * z);
            }
            """)
    _U10_KERNEL(u1, v1, dz1, res.znt, res.zol, res.gz1oz0, res.psim, xland, m, f["U10"], f["V10"])


def run_sfc(ctx):
    """MYNN surface layer in JSFC's place (before NOAHLSM)."""
    if v61():
        from namgpu.physics.mynn import phys_mynn_v61
        return phys_mynn_v61.run_sfc(ctx)
    import cupy as cp
    from gpuwm.core.mynn_sfclay import (MYNN_SURFACE_OUTPUTS, MynnSurfaceResult, launch_mynn_surface_layer,
                                        seed_mynn_surface_first_step)
    st = ctx.st
    t = _t(ctx)
    lm, jm, im = (int(s) for s in st["T"].shape)
    s2 = (jm, im)
    f = st.fields
    for n in ("USTAR", "Z0", "QSH", "RMOL", "U10", "V10", "TH10", "Q10", "TSHLTR", "QSHLTR", "PSHLTR",
              "AKHS", "AKMS", "FLHC", "FLQC", "ELFLX", "RIB", "PSIM", "PSIH", "PBLH", "CT"):
        st.add(n, s2)
    for n in ("MYNNB_MOL", "MYNNB_USTM", "MYNNB_HFX", "MYNNB_QFX", "MYNNB_WSPD", "MYNNB_QSTAR"):
        st.add(n, s2)
    first = t["calls_sfc"] == 0
    t["calls_sfc"] += 1
    kb, k2 = lm - 1, lm - 2
    q1 = f["Q"][kb]
    qv1 = _buf(t, "qv1", s2)
    cp.divide(q1, 1.0 - q1, out=qv1)
    u1 = cp.ascontiguousarray(f["U_PHY"][kb])
    v1 = cp.ascontiguousarray(f["V_PHY"][kb])
    if first:
        f["MYNNB_USTM"][...] = cp.maximum(f["USTAR"], cp.float32(0.001))
        seed_mynn_surface_first_step(u1, v1, qv1, ust=f["USTAR"], mol=f["MYNNB_MOL"], qsfc=f["QSH"],
                                     qstar=f["MYNNB_QSTAR"])
        f["Z0"][...] = cp.maximum(f["Z0"], cp.float32(1.0e-4))
    _land_z0(t, f, _mask(t, jm, im))
    psfc = cp.ascontiguousarray(f["PHINT"][lm])
    inputs = {
        "u1": u1, "v1": v1, "t1": cp.ascontiguousarray(f["T"][kb]), "qv1": qv1,
        "p1": cp.ascontiguousarray(f["PHMID"][kb]), "rho1": cp.ascontiguousarray(f["RR"][kb]),
        "dz1": cp.ascontiguousarray(f["DZ"][kb]),
        "u2": cp.ascontiguousarray(f["U_PHY"][k2]), "v2": cp.ascontiguousarray(f["V_PHY"][k2]),
        "dz2": cp.ascontiguousarray(f["DZ"][k2]),
        "psfc": psfc, "tsk": f["TSFC"], "pblh": f["PBLH"], "mavail": f["MAVAIL"],
        "hfx": f["MYNNB_HFX"], "qfx": f["MYNNB_QFX"], "znt": f["Z0"], "qsfc": f["QSH"], "ust": f["USTAR"],
        "xland": f["XLAND"], "snowh": f["SNOWH"],
    }
    res = t.get("sfc_result")
    if res is None or res.chs.shape != s2:
        res = MynnSurfaceResult(**{n: cp.zeros(s2, dtype=cp.float32) for n in MYNN_SURFACE_OUTPUTS})
        t["sfc_result"] = res
    launch_mynn_surface_layer(inputs, f["MYNNB_MOL"], f["MYNNB_USTM"], res, dx=_dx(ctx),
                              itimestep=int(t["calls_sfc"]), isfflx=1, isftcflx=0)
    m = _mask(t, jm, im)
    put = lambda name, val: cp.copyto(f[name], val, where=m)
    put("CHS", res.chs); put("CHS2", res.chs2); put("CQS2", res.cqs2); put("CPM", res.cpm)
    put("QGH", res.qgh); put("FLHC", res.flhc); put("FLQC", res.flqc)
    put("TWBS", res.hfx); put("QWBS", res.qfx); put("ELFLX", res.lh)
    put("QSH", res.qsfc); put("USTAR", res.ust); put("Z0", res.znt); put("RMOL", res.rmol)
    put("RIB", res.br); put("PSIM", res.psim); put("PSIH", res.psih)
    put("AKHS", res.chs); put("AKMS", res.cd * res.wspd)
    put("U10", res.u10); put("V10", res.v10)
    _land_u10(t, res, u1, v1, inputs["dz1"], f["XLAND"], m, f)
    put("MYNNB_MOL", res.mol); put("MYNNB_USTM", res.ustm); cp.maximum(res.wspd, cp.float32(0.1), out=f["MYNNB_WSPD"])   # whole field: the PBL refuses wspd <= 0 (outer ring has zero U_PHY)
    put("MYNNB_QSTAR", res.qstar)
    return res


def post_sfcdiags(ctx):
    """After NOAHLSM + SFCDIAGS: the shelter fields the history and products read, from SFCDIAGS (WRF's Noah path);
    and the post-LSM fluxes kept for the next surface-layer call (MYNN's hfx/qfx inputs)."""
    import cupy as cp
    st = ctx.st
    t = _t(ctx)
    f = st.fields
    lm, jm, im = (int(s) for s in f["T"].shape)
    m = _mask(t, jm, im)
    psfc = f["PHINT"][lm]
    t2 = f["T2"]
    pshl = psfc * cp.exp(-2.0 * G / (R_D * cp.maximum(t2, cp.float32(150.0))))
    cp.copyto(f["PSHLTR"], pshl.astype(cp.float32), where=m)
    cp.copyto(f["TSHLTR"], (t2 * (1.0e5 / f["PSHLTR"]) ** CAPPA).astype(cp.float32), where=m)
    cp.copyto(f["QSHLTR"], f["Q2X"], where=m)
    cp.copyto(f["MYNNB_HFX"], f["TWBS"], where=m)
    cp.copyto(f["MYNNB_QFX"], f["QWBS"], where=m)


def run_pbl(ctx, dtphs):
    """MYNN-EDMF PBL in MYJPBL's place."""
    if v61():
        from namgpu.physics.mynn import phys_mynn_v61
        return phys_mynn_v61.run_pbl(ctx, dtphs)
    import cupy as cp
    from gpuwm.core.mynn_pbl_runtime import mynn_pbl_step
    st = ctx.st
    t = _t(ctx)
    f = st.fields
    lm, jm, im = (int(s) for s in f["T"].shape)
    s2, s3 = (jm, im), (lm, jm, im)
    for n in STATE_3D:
        st.add("MYNNB_" + n, s3)
    for n in ("maxwidth", "maxmf", "ztop_plume"):
        st.add("MYNNB_" + n, s2)
    st.add("MYNNB_kpbl", s2, dtype="int32")
    st.add("MYNNB_ktop_plume", s2, dtype="int32")
    for n in ("XLEN_MIX", "EXCH_H"):
        st.add(n, s3)
    st.add("LPBL", s2, dtype="int32")
    t["calls_pbl"] += 1

    # Bottom-first VIEWS of the NMMB arrays (negative stride on L): mynn_pbl_step copies each column chunk into
    # its own staging buffer, so no whole-domain flipped copy is made. Only qv and qc (converted) are buffers.
    q = f["Q"]
    # Work buffers: per call from the memory pool (default), or held for the run with NAMGPU_MYNN_SCRATCH=keep.
    # Held buffers + held chunk scratch (~3.8 GB) did not run faster (full-shape bench: 405 vs 408 ms per
    # call) and pushed the 12 km arm out of a 32 GB card (OOM on an RTX 5090), so per call is the default.
    keep = os.environ.get("NAMGPU_MYNN_SCRATCH", "call") == "keep"
    _b = (lambda name: _buf(t, name, s3)) if keep else (lambda name: cp.zeros(s3, dtype=cp.float32))
    one_m_q = _b("one_m_q")
    cp.subtract(cp.float32(1.0), q[::-1], out=one_m_q)
    qv = _b("qv3")
    cp.divide(q[::-1], one_m_q, out=qv)
    qc = _b("qc3")
    cp.divide(f["QC"][::-1], one_m_q, out=qc)
    qi = _b("qi3_zero")
    atm = {
        "dz": f["DZ"][::-1], "u": f["U_PHY"][::-1], "v": f["V_PHY"][::-1], "theta": f["TH"][::-1],
        "pressure": f["PHMID"][::-1], "exner": f["EXNER"][::-1], "rho": f["RR"][::-1],
        "temperature": f["T"][::-1], "qv": qv, "qc": qc, "qi": qi, "qs": f["QS"][::-1],
    }
    w = f["W"][::-1]
    fields = {n: f["MYNNB_" + n] for n in STATE_3D}
    fields.update({n: f["MYNNB_" + n] for n in ("maxwidth", "maxmf", "ztop_plume", "kpbl", "ktop_plume")})
    fields.update({"pblh": f["PBLH"], "rmol": f["RMOL"], "xland": f["XLAND"], "tsk": f["TSFC"],
                   "psfc": cp.ascontiguousarray(f["PHINT"][lm]), "ust": f["USTAR"], "hfx": f["TWBS"],
                   "qfx": f["QWBS"], "wspd": f["MYNNB_WSPD"]})
    # Column chunk: ArWen's default 16,384 columns is 128 blocks per launch (under one wave on a 170-SM card) and
    # ~49 chunks of ~80 launches at 12 km (host-bound: 1-h smoke, 189 ms device / 353 ms host per call). 65,536
    # columns is 13 chunks, ~3 GB of MYNN working set; the working set is per call (returned to the pool after it),
    # so it never adds to the radiation call's peak.
    chunk = int(os.environ.get("NAMGPU_MYNN_CHUNK", "65536"))
    # MYNN's chunk working set (~3 GB at 65,536 columns): per call by default (back to the pool after the call, so it
    # never adds to the radiation call's peak); held for the run with NAMGPU_MYNN_SCRATCH=keep.
    scratch = t["scratch"] if keep else _Scratch()
    # Validators on the device. MYNN's input refusals (finite inputs; positive dz, rho, p, exner, tk, wspd, psfc,
    # pblh, dx) write device flag words. With no health ledger each site read its words back at once: 4 blocking
    # reads per 65,536-column chunk, 52 per call at 12 km, so the host waited on the GPU and the GPU then idled while
    # the host queued the next chunk. Under the run's ledger the words are ORed on the device and polled without
    # waiting after each call; the driver drains every ledger once an hour at the sync it already pays and at the
    # end of the run. Same refusals, same messages, reported up to one forecast hour late (the message says so).
    # NAMGPU_MYNN_DEFER=0 restores the immediate reads.
    led = _ledger(ctx, t)
    from gpuwm.core import health_ledger as _hl
    with _hl.deferring(led):
        out = mynn_pbl_step(atm, fields, w=w, dx=_dx(ctx), delt=float(dtphs), itimestep=int(t["calls_pbl"]),
                            mp_physics=MP_PHYSICS_WRF, state=scratch, column_chunk=chunk, **MYNN_OPTIONS)
    if led is not None:
        led.poll()
    m3 = cp.broadcast_to(_mask(t, jm, im), s3)
    def sel(name, val):
        dst = f[name]
        if dst.shape[0] == lm + 1:
            # an interface array (NMMB EXCH_H is [LM+1]): MYNN's level k is the BOTTOM interface of its layer,
            # i.e. NMMB interface l0 + 1 of layer l0 = LM-1-k
            dst = dst[1:]
        cp.copyto(dst, val, where=m3)
    sel("DUDT", out["du"][::-1])
    sel("DVDT", out["dv"][::-1])
    sel("RTHBLTEN", out["dtheta"][::-1])
    sel("RQBLTEN", out["dqv"][::-1] * one_m_q[::-1])
    sel("RQCBLTEN", out["dqc"][::-1] * one_m_q[::-1])
    sel("EXCH_H", f["MYNNB_exch_h"][::-1])
    sel("XLEN_MIX", f["MYNNB_el_pbl"][::-1])
    sel("Q2", cp.maximum(f["MYNNB_qke"][::-1], cp.float32(0.02)))
    cp.copyto(f["LPBL"], (lm + 1 - f["MYNNB_kpbl"]).astype(cp.int32), where=_mask(t, jm, im))
    return out
