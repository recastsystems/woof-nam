"""phys-ugwp-driver: NOAA-GSL UGWP orographic drag suite (bl_ugwp_gwdo) in place of NAM's GWD (--gwd ugwp).

Copyright 2026, NOAA, UCAR/NCAR CU/CIRES (NOAA GSL Unified Gravity Wave Physics, UGWP); Apache-2.0
(LICENSE-UGWP-Apache-2.0.txt). Source NOAA-GSL/UGWP c1c893ed (MPAS_20241223) bl_ugwp.F, through the
gpuwm-hex CUDA port (Apache-2.0). The scheme is what RRFS v1/v2 and the hrrrv5 suite run: large-scale
GWD with flow blocking (Kim and Doyle 2005; Lott and Miller 1997), small-scale GWD in stable boundary
layers (Steeneveld 2008; Tsiringakis 2017) and turbulent orographic form drag (Beljaars 2004).

Kernel: namgpu/kernels/phys_ugwp_oro.cu (phys_ugwp_oro), one launch, one thread per H column, reading
the TURBL working arrays directly. Called from phys-turbl-driver in GWD's slot on the physics cadence
(DTPHS = NPHS * DT); its tendencies DUDT_GWD/DVDT_GWD join DUDT/DVDT through turbl_dudt_add exactly as
NAM's GWD tendencies do.

In (ctx.st): U_PHY, V_PHY, T, Q, PHMID, EXNER [LM, JM, IM]; PHINT, TURBL_Z [LM+1, JM, IM]; LPBL int32,
PBLH, RIB, XLAND, CROT, SROT [JM, IM]; UGWP_* (twenty statistics, prepare()). Row vector UGWP_DXM [JM].
Out: DUDT_GWD, DVDT_GWD [LM, JM, IM]; UGWP_DUSFC, UGWP_DVSFC [JM, IM] (Earth-relative surface stress).
No host sync on the step path except a 4-byte status read every STATUS_EVERY calls (the KPBL guard).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_ugwp_oro"
KERNEL = "phys_ugwp_oro"
LM_MAX = 60          # UGWP_LEVELS in the kernel: per-thread column arrays hold 60 + 2 levels
STATUS_EVERY = 72    # calls between status reads (one 4-byte copy)
# NMMB constants (module_CONSTANTS): G, CP, R_D, R_V.
G, CP, RD, RV = 9.806, 1004.6, 287.04, 461.6
FV = RV / RD - 1.0
ARGS = {"U": "U_PHY", "V": "V_PHY", "PMID": "PHMID", "EXNR": "EXNER", "PINT": "PHINT", "Z": "TURBL_Z",
        "LPBL": "LPBL"}


def _fields():
    from namgpu.init.init_ugwp_oro import FIELDS
    return FIELDS


def prepare(ctx):
    """Load the orography statistics (init-ugwp-oro) and the row spacing once, on the whole domain."""
    st, g = ctx.st, ctx.grid
    if "UGWP_DXM" in st.fields and all(n in st.fields for n in _fields()):
        return
    import cupy as cp
    from namgpu.init import init_ugwp_oro as oro
    a = oro.build(g)
    jm, im = int(g.jm), int(g.im)
    for k, name in enumerate(oro.FIELDS):
        st.add(name, (jm, im))[...] = cp.asarray(a[k])
    st.fields["UGWP_DXM"] = cp.asarray(np.sqrt(np.asarray(g.dxh, np.float64) * float(g.dyh)).astype(np.float32))
    ctx.tables["ugwp_status"] = cp.zeros((1,), dtype=cp.int32)
    ctx.tables["ugwp_calls"] = 0
    print(f"phys-ugwp-driver: UGWP orographic drag on ({im} x {jm}), var2dls mean {float(a[0].mean()):.1f} m "
          f"max {float(a[0].max()):.0f} m; var2dss mean {float(a[10].mean()):.1f} m", flush=True)


def run(ctx):
    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > LM_MAX:
        raise ValueError(f"phys-ugwp-driver: kernel column arrays hold {LM_MAX} levels, got {lm}")
    if "UGWP_DXM" not in st.fields:
        prepare(ctx)
    a = lambda n: st[ARGS.get(n, n)]
    for name in ("DUDT_GWD", "DVDT_GWD"):
        if tuple(st[name].shape) != (lm, jm, im):
            raise ValueError(f"{name} shape {st[name].shape}, expected {(lm, jm, im)}")
    dus = st["UGWP_DUSFC"] if "UGWP_DUSFC" in st.fields else st.add("UGWP_DUSFC", (jm, im))
    dvs = st["UGWP_DVSFC"] if "UGWP_DVSFC" in st.fields else st.add("UGWP_DVSFC", (jm, im))
    status = ctx.tables["ugwp_status"]
    dtphs = np.float32(int(ctx.cfg.nphs)) * np.float32(ctx.dt)
    f32, i32 = np.float32, np.int32
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
        st["DUDT_GWD"], st["DVDT_GWD"], dus, dvs, status,
        a("U"), a("V"), st["T"], st["Q"], a("PMID"), a("EXNR"), a("PINT"), a("Z"), a("LPBL"),
        st["PBLH"], st["RIB"], st["XLAND"], st["CROT"], st["SROT"], st["UGWP_DXM"],
        *[st[n] for n in _fields()],
        f32(G), f32(CP), f32(RD), f32(RV), f32(FV), f32(np.pi), dtphs,
        i32(im), i32(jm), i32(lm), i32(0), i32(jm - 1)))
    ctx.tables["ugwp_calls"] = int(ctx.tables.get("ugwp_calls", 0)) + 1
    if ctx.tables["ugwp_calls"] % STATUS_EVERY == 1 and int(status.get()[0]):
        raise RuntimeError("phys-ugwp-driver: DEFECT UGWP_INVALID_KPBL: a column's KPBL search ran past the top "
                           "level (bl_ugwp.F:634-709 would read beyond layer storage)")
