"""drv-native-diag-restore: the 0 h diagnostics of a start from the NAM's own analysis.

Why: a run started from an archived NAM analysis (init_source native) runs no digital filter and writes its 0 h
history before the first physics step, as NMMB does. The NAM's own 0 h state carried its diagnostics over from its
analysis cycle; ours started them at zero (or at JSFC_INIT's cold-start USTAR of 0.1). In the 2024-11-05 12Z
cycle 94 records of the +0 h awphys file were all zero where NCEP's were not: total cloud cover (NCEP mean 57.5 %),
boundary-layer height (599 m), the four surface radiation fluxes, sensible and latent heat flux, moisture
availability, and the vertical velocity (VVEL, DZDT) on every isobaric level; total soil moisture read 0.1 mm over
land (NCEP 492 mm) and friction velocity 0.1 everywhere.

How: the analysis file carries those fields (the inventory of nam.tHHz.bgrd3d00: HPBL, FRICV, SHTFL, LHTFL, the
instantaneous DSWRF/USWRF/DLWRF/ULWRF at the surface, SOILM 0-2 m, MSTAV 0-1 m, and TCDC on every hybrid level).
init-native (namgpu/init/sources.py: stage_native, NATIVE_DIAG) decodes the 2-D ones; this unit, called once just
before the time loop of a no-filter start, puts them in the state under the names the history and the products read:
  PBLH = HPBL; USTAR = FRICV; TWBS = -SHTFL; QWBS = -LHTFL (NMMB's sign: positive down);
  RSWIN = DSWRF, RSWOUT = USWRF, CZEN = CZMEAN = 1; RLWIN = DLWRF, SIGT4 = STBOL T_lowest^4; RADOT = ULWRF
  (so the 0 h radiation records are the analysis' instantaneous fluxes);
  SMSTOT = SOILM (kg m-2); SMSTAV = MSTAV / 100;
  CLDFRA = TCDC on the hybrid levels / 100 (decoded here, one 3-D field in flight);
  W_TOT = W on the mid layers (W holds the analysis' DZDT record, itself NMMB's W_TOT; the dynamics' VSOUND writes
  W_TOT only from the first step on);
  CPRATE = CPRAT DT NPRECIP / 1000 (m of water over one convection call, as the convection glue writes it), on the
  interior columns only (the glue never writes the outer rows and columns, so a value there would stay all run);
  CNVBOT, CNVTOP = the model layer, counted from the ground (1 the lowest), whose mid-layer pressure is nearest the
  analysis' convective cloud bottom and top pressures (convective_layers); LM + 1 and 0, the glue's empty values,
  where the column has no convective cloud.
The last three are what the history's reflectivity adds the convective rain from (Z = 300 R^1.4 up to CNVTOP) and
what the cloud base takes as the convective cloud bottom. Without them a native start's 0 h products had no
convective cloud at all: the 0 h composite reflectivity lacked the convective echo NCEP's own file has (around the
hurricane near Jamaica on 2024-11-05 12Z, 14-22 N, 82-70 W of grid 218, 1,020 points over 20 dBZ against NCEP's
1,522; in NCEP's analysis the composite of the columns with convective rain and no grid-scale rain or snow is
300 R^1.4 of CPRAT to 0.06 dB, correlation 0.998), and the 0 h cloud base had no base at 34,494 grid-218 points where NCEP's has one (MAE 1,123 m against
208 m at +3 h). The first convection call (step 0) overwrites CPRATE on the interior, and the output-window reset of
the next call puts CNVBOT and CNVTOP back to their empty values before the first hour, so only the 0 h history
changes.
The first step's radiation call (NTIMESTEP 0) recomputes RSWIN, RSWOUT, RLWIN, RADOT, CZMEAN, SIGT4 and CLDFRA, and
the surface layer, turbulence and Noah recompute the rest, so the forecast starts as before except for the first
guess of USTAR and PBLH (the NAM's own instead of JSFC_INIT's cold-start values). A GFS or other source start runs
the digital filter (drv-dfi-tdfi-driver), which fills these by running the physics, and never calls this unit.
The precipitation types stay zero at 0 h: they come from the last physics step's rate, which a start does not have.
"""
from __future__ import annotations

UNIT = "drv_native_diag_restore"
STBOL = 5.67051e-8       # the model's Stefan-Boltzmann constant (NMMB_STBOLT): SIGT4 is the model's sigma T^4


def fill(st, diag, t_lowest, xp):
    """Write the decoded analysis diagnostics (diag: name -> [JM, IM]) into st (a state with .fields, or a dict).
    t_lowest: [JM, IM] temperature of the lowest model layer (K). Returns the state names written."""
    names = st.fields if hasattr(st, "fields") else st
    f32 = xp.float32
    done = []

    def put(name, value):
        if name in names and value is not None:
            st[name][...] = value.astype(f32)
            done.append(name)

    g = diag.get
    put("PBLH", g("HPBL"))
    put("USTAR", g("FRICV"))
    if g("SHTFL") is not None:
        put("TWBS", -g("SHTFL"))
    if g("LHTFL") is not None:
        put("QWBS", -g("LHTFL"))
    if g("DSWRF") is not None or g("USWRF") is not None:
        put("RSWIN", g("DSWRF"))
        put("RSWOUT", g("USWRF"))
        one = xp.ones(t_lowest.shape, dtype=f32)
        put("CZEN", one)
        put("CZMEAN", one)
    if g("DLWRF") is not None:
        put("RLWIN", g("DLWRF"))
        t = t_lowest.astype(f32)
        put("SIGT4", f32(STBOL) * t * t * t * t)
    put("RADOT", g("ULWRF"))
    put("SMSTOT", g("SOILM"))
    if g("MSTAV") is not None:
        put("SMSTAV", xp.clip(g("MSTAV") * f32(0.01), 0.0, 1.0))
    return done


def convective_layers(p_mid, p_cloud, empty, xp):
    """The model layer, counted from the ground (1 the lowest), whose mid-layer pressure is nearest p_cloud.
    p_mid: [LM, JM, IM] mid-layer pressure (Pa), index 0 the top layer; p_cloud: [JM, IM] (Pa), <= 0 where the column
    has no convective cloud, which gets `empty`. Also returns the largest |p_mid - p_cloud| at the chosen layers over
    the columns with a cloud (Pa; 0 when none has one). One level at a time: no [LM, JM, IM] temporary."""
    lm = int(p_mid.shape[0])
    pc = p_cloud.astype(xp.float32)
    best = xp.full(pc.shape, xp.inf, dtype=xp.float32)
    l0 = xp.zeros(pc.shape, dtype=xp.int32)
    for k in range(lm):
        d = xp.abs(p_mid[k] - pc)
        closer = d < best
        best = xp.where(closer, d, best)
        l0 = xp.where(closer, xp.int32(k), l0)
    has = pc > 0
    layer = xp.where(has, (lm - l0).astype(xp.float32), xp.float32(empty)).astype(xp.float32)
    worst = float(best[has].max()) if bool(has.any()) else 0.0
    return layer, worst


def fill_convective(st, diag, p_mid, conv_step_s, xp):
    """Write the analysis' convective state (CPRAT, PCNVBOT, PCNVTOP in diag) into st: CPRATE (m of water per
    conv_step_s = DT NPRECIP seconds, interior columns only), CNVBOT and CNVTOP (convective_layers). Returns the state
    names written and the largest layer-pressure mismatch (Pa)."""
    names = st.fields if hasattr(st, "fields") else st
    lm = int(p_mid.shape[0])
    done, worst = [], 0.0
    g = diag.get
    if g("CPRAT") is not None and "CPRATE" in names:
        rate = xp.maximum(g("CPRAT").astype(xp.float32), xp.float32(0.0)) * xp.float32(float(conv_step_s) * 1.0e-3)
        out = xp.zeros(rate.shape, dtype=xp.float32)
        out[1:-1, 1:-1] = rate[1:-1, 1:-1]
        st["CPRATE"][...] = out
        done.append("CPRATE")
    for src, dst, empty in (("PCNVBOT", "CNVBOT", lm + 1), ("PCNVTOP", "CNVTOP", 0)):
        if g(src) is not None and dst in names:
            layer, w = convective_layers(p_mid, g(src), empty, xp)
            st[dst][...] = layer
            worst = max(worst, w)
            done.append(dst)
    return done, worst


def run(ctx):
    import cupy as cp
    from pathlib import Path
    from namgpu.init import sources

    nd = ctx.tables.get("native_diag")
    if not nd:
        return []
    st, lm = ctx.st, int(ctx.grid.lm)
    done = fill(st, nd["sfc"], st["T"][lm - 1], cp)
    worst = None
    if any(k in nd["sfc"] for k in ("CPRAT", "PCNVBOT", "PCNVTOP")):
        from namgpu.physics.driver.phys_fer_driver import _vcoord
        p_mid = (st["SGML2"].reshape(lm, 1, 1) * st["PD"][None, :, :]
                 + _vcoord(ctx, "psgml1").reshape(lm, 1, 1)).astype(cp.float32)
        got, worst = fill_convective(st, nd["sfc"], p_mid, float(ctx.cfg.dt) * int(ctx.cfg.nprecip), cp)
        done += got
        del p_mid
    if nd.get("cldfra") and "CLDFRA" in st.fields:
        def sink(n, a):
            st["CLDFRA"][...] = cp.clip(a * cp.float32(0.01), 0.0, 1.0)
        sources.decode_native(ctx, Path(nd["file"]), ["CFRAC"], [], lm, nd["tmpdir"], sink,
                              top_first=nd["top_first"])
        done.append("CLDFRA")
    if "W_TOT" in st.fields and "W" in st.fields:
        st["W_TOT"][:lm] = st["W"][:lm]
        done.append("W_TOT")
    ctx.tables["native_diag"] = None             # the decoded fields are not needed again
    tail = "" if worst is None else f"; convective cloud layers within {worst:.1f} Pa of the analysis' pressures"
    print(f"{UNIT}: 0 h diagnostics from the analysis: {' '.join(done)}{tail}", flush=True)
    return done
