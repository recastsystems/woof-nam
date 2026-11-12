"""drv-dfi-refl-restore: the 0 h radar reflectivity recomputed from the 0 h state.

Why: after TDFI the filtered dynamics (dyn_average) and the restored physics fields (drv-dfi-phy-restore: CW,
F_ICE, F_RAIN, F_RIMEF and the other R-flagged fields of parm/nam_filt_vars_norad.txt) are the 0 h state, but
REFL_10CM is not in the restore table, so it kept the filter's last FER_HIRES values: 1,037 cells over 40 dBZ
at f00 in smoke24 (2026-11-04 11:25Z) that the 0 h condensate does not hold. The no-DFI start writes no
reflectivity at all, so the 0 h map had to be either dropped or made consistent; this makes it consistent.

How, on the fer_hires route (the NAM's): one microphysics call (phys-fer-driver, DT NPRECIP, as every forecast
step makes it) on the 0 h state, keeping only the REFL_10CM it writes; every other field the call writes (T, Q,
the condensate and its fractions, TRAIN, the precipitation buckets, SR, MPRATES and the call's work arrays) is put
back from a host copy, so the forecast starts from the same state bit for bit and only the 0 h history changes.
The forecast hours' REFL_10CM is the same call's (the last one before the hour), so the 0 h map uses the same rain
and snow size rules as the later hours: drizzle (rain with no ice melting above it) takes the raised intercept of
Ferrier (1994) and Aligo et al. (2018), and rain under a melting layer takes the stratiform minimum drop size.
Before this the 0 h map used a fixed intercept (8e6 m-4) for all rain (the diagnostic unit), which is 8-16 dB too
strong for drizzle: on 2024-11-05 12Z, in the 30,339 native-grid columns with rain and no snow, cloud ice or
convective rain, the 0 h composite was 11.3 dB above NCEP's own analysis composite (MAE 11.3 dB); with the
raised intercept on the analysis rain it is -0.4 dB (MAE 1.6 dB). The next microphysics call (step 0 of the
forecast) overwrites REFL_10CM as before.
Other routes (a column suite or the TEMPO / Thompson columns own microphysics) keep the fixed-intercept diagnostic
on CW, F_ICE, F_RAIN and F_RIMEF, with no convective term.
Called by drv-dfi-tdfi-driver right after drv-dfi-phy-restore, and by run_forecast before the time loop of a
no-filter start, on ctx.stream.
"""
from __future__ import annotations

import contextlib

import numpy as np

UNIT = "drv_dfi_refl_restore"

# What one phys-fer-driver call writes besides REFL_10CM (its docstring, "Writes"), put back after the 0 h call.
# The band-local work arrays (DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY) are rewritten by the next call before any read.
MP_WRITES = ("T", "Q", "CW", "QC", "QR", "QS", "F_ICE", "F_RAIN", "F_RIMEF", "TRAIN", "PREC", "ACPREC",
             "ACPREC_TOT", "AVRAIN", "SR", "SNOWNC_TOT", "MPRATES", "LOWLYR", "RAINNC", "RAINNCV")


def keep_only_refl(fields, call, writes=MP_WRITES):
    """Run call() (one microphysics call that writes REFL_10CM and the names in `writes` in `fields`), then put
    every name in `writes` back as it was: the same array object (a call may replace it, as MPRATES' first call
    does) with its values from a host copy taken before the call; a name the call added is removed. Arrays that
    share memory (QC and its TRACERS slot) are restored through the one object. Returns the names put back."""
    saved = {}
    for n in writes:
        a = fields.get(n)
        if a is not None:
            saved[n] = (a, a.get() if hasattr(a, "get") else np.array(a, copy=True))
    call()
    for n in writes:
        if n not in saved:
            fields.pop(n, None)
    for n, (a, h) in saved.items():
        fields[n] = a
        if isinstance(a, np.ndarray):
            a[...] = h
        elif a.flags.c_contiguous:
            a.set(h)                                 # host to device in place, no device temporary
        else:
            import cupy as cp
            a[...] = cp.asarray(h)
    return sorted(saved)


def ferrier_route(cfg) -> bool:
    """True when the forecast's microphysics call is phys-fer-driver's FER_HIRES (the NAM configuration)."""
    if str(getattr(cfg, "microphysics", "")).strip() != "fer_hires" or str(getattr(cfg, "mp_column", "") or ""):
        return False
    from namgpu.steps.drv_physics_initialize import _suite_microphysics
    return _suite_microphysics(cfg) is None


def run(ctx):
    import cupy as cp

    st = ctx.st
    if "REFL_10CM" not in st.fields:
        return None
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    if ferrier_route(ctx.cfg):
        from namgpu.physics.driver import phys_fer_driver
        with scope:
            before = float(st["REFL_10CM"].max())
            kept = keep_only_refl(st.fields, lambda: phys_fer_driver.run(ctx))
            refl = st["REFL_10CM"]
            after = float(refl.max())
            n40 = int((refl.max(axis=0) > 40.0).sum())
        print(f"drv-dfi-refl-restore: 0 h REFL_10CM from one microphysics call on the 0 h state ({len(kept)} other "
              f"fields put back): max {before:.1f} -> {after:.1f} dBZ, {n40} columns over 40 dBZ", flush=True)
        return {"refl": "one microphysics call on the 0 h state, other fields put back", "max_before": before,
                "max_after": after}
    return _fixed_intercept(ctx, cp, scope)


def _fixed_intercept(ctx, cp, scope):
    """The diagnostic reflectivity unit on CW, F_ICE, F_RAIN, F_RIMEF, no convective term."""
    from namgpu.physics.fer_hires import phys_fer_calmict, phys_fer_gpvs, phys_fer_init

    st = ctx.st
    g = ctx.grid
    lm = int(g.lm)
    fer = ctx.tables.get("fer")
    if fer is None:
        phys_fer_init.run(ctx)
        fer = ctx.tables["fer"]
    if "fer_gpvs_tbpvs" not in ctx.tables:
        phys_fer_gpvs.run(ctx)
    if "fer_scalars" not in ctx.tables:
        coef = cp.asnumpy(ctx.tables["fer_gpvs_coef"])        # once per run, at the end of the filter
        sc = dict(fer["scalars"])
        sc["C1XPVS"], sc["C2XPVS"] = float(coef[0]), float(coef[1])
        ctx.tables["fer_scalars"] = sc
        ctx.tables["fer_massi"] = fer["massi"]
    from namgpu.physics.driver.phys_fer_driver import _vcoord
    psgml1 = _vcoord(ctx, "psgml1")
    with scope:
        sgml2 = st["SGML2"].reshape(lm, 1, 1)
        p = sgml2 * st["PD"][None, :, :] + psgml1.reshape(lm, 1, 1)
        p = p.astype(cp.float32, copy=False)
        curefl = ctx.tables.get("dfi_refl_curefl")
        if curefl is None or curefl.shape != st["PD"].shape:
            curefl = cp.zeros(st["PD"].shape, dtype=cp.float32)
            ctx.tables["dfi_refl_curefl"] = curefl
        dbz = phys_fer_calmict.run(ctx, p, st["T"], st["Q"], st["CW"], st["F_ICE"], st["F_RAIN"],
                                   st["F_RIMEF"], curefl)
        before = float(st["REFL_10CM"].max())        # one readback, once per run (the end of the filter)
        cp.copyto(st["REFL_10CM"], dbz)
        after = float(dbz.max())
        n40 = int((dbz.max(axis=0) > 40.0).sum())
    print(f"drv-dfi-refl-restore: 0 h REFL_10CM recomputed (fixed intercept): max {before:.1f} -> {after:.1f} dBZ, "
          f"{n40} columns over 40 dBZ", flush=True)
    return {"refl": "recomputed from the 0 h state (fixed intercept, no convective term)", "max_before": before,
            "max_after": after}
