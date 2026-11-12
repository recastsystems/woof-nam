"""init-native-sfc: surface and soil initial state from the NAM native analysis into ctx.st.

Unit: init-native-sfc (group init, mode new, no Fortran counterpart). Kernel:
namgpu/kernels/init_native_sfc.cu (kernel init_native_sfc), one launch, one thread per column.

Inputs (device arrays, names fixed by the GRIB decode unit init-grib2-decode):
    ctx.tables["raw_sfc"]          dict of cupy arrays, shapes in the kernel header:
        TSOIL [4, JM, IM] K          SOILW [4, JM, IM] volumetric      TMP_SFC [JM, IM] K
        WTMP [JM, IM] K              ICEC [JM, IM]                     LAND [JM, IM] (1 = land)
        TSOIL_3M [JM, IM] K          SNOD [JM, IM] m                   WEASD [JM, IM] kg m-2
        ALBDO [JM, IM] percent       VEG [JM, IM] percent              SFCR [JM, IM] m
        SNOWC [JM, IM]               CNWAT [JM, IM] kg m-2
        VGTYP [JM, IM] int32         SOTYP [JM, IM] int32
    ctx.tables["soil_hydraulics"]  dict with "bexp", "psisat" (m), "smcmax": cupy float32 [30],
                                   from init-static-missing (soil class table).
Outputs (ctx.st, NMMB names, CONTRACT layout):
    STC, SMC, SH2O [4, JM, IM]; TSKIN, SST, SICE, SM, TG, SI (mm), SNO (mm), SNOWC, ALBEDO, VEGFRC, Z0,
    Z0BASE, CMC [JM, IM] float32; IVGTYP, ISLTYP [JM, IM] int32.
Conversions and the frozen-fraction formula are documented in the kernel header.

Urban soil moisture from a foreign land model (urban_soilw, before the kernel). NAM's SFLX replaces the soil
parameters of an urban point (VEGTYP == IW, 13 for IGBP) with SMCMAX 0.45, SMCREF 0.42, SMCWLT = SMCDRY 0.40
(module_LS_NOAHLSM.F90:1851-1858), so NMMB Noah only evaporates from an urban soil holding 0.40-0.45. The NAM's own
analysis cycles those points in that range (0.43 at the Carolina airports, 2026-01-23 12Z). A start whose land state
comes from another model (GFS, IFS, ERA5, ...) carries that model's soil moisture for the soil class, 0.22 at the
same points: below the urban wilting point, so those cells never evaporate and their 2 m dewpoint runs 10-20 F too
dry under daytime mixing in dry air (KGSO/KRDU/KINT/KCLT, Jan 2026 wedge; 4-8 F in the Dec 2022 Northeast case).
For such a source each urban land point keeps its relative wetness: (theta - SMCWLT)/(SMCMAX - SMCWLT) of the
point's STAS soil class, mapped onto the urban range 0.40-0.45. Every other point, and every point of the NAM's own
analysis or a WOOF-NAM parent (already NMMB Noah's state), is unchanged.
Run the self-test (random plausible inputs, full size, NaN/Inf check):
    source env.sh && python -m namgpu.init.init_native_sfc
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "init-native-sfc"
KERNEL = "init_native_sfc"
KFILE = "init_native_sfc"   # namgpu/kernels/<KFILE>.cu (underscored unit id)
NCLASS = 30
NSOIL = 4
BLOCK = 128

RAW_KEYS = ("TSOIL", "SOILW", "TMP_SFC", "ICEC", "LAND", "TSOIL_3M", "SNOD", "WEASD",
            "ALBDO", "VEG", "SFCR", "CNWAT", "VGTYP", "SOTYP")
OPT_KEYS = ("SNOWC",)   # NAM's snowc record overrides the 0.98/0 rule when present
OUT_FLOAT_2D = ("TSKIN", "SST", "SICE", "SM", "TG", "SI", "SNO", "SNOWC", "ALBEDO", "VEGFRC",
                "Z0", "Z0BASE", "CMC")
OUT_SOIL = ("STC", "SMC", "SH2O")

# Urban soil (module_LS_NOAHLSM.F90:1851-1858, SFLX: IF(VEGTYP==IW) SMCMAX=0.45, SMCWLT=0.40). IW = 13 for IGBP
# (IVEGSRC 1, phys_noah_driver.IVEGSRC). Sources whose soil moisture is NMMB Noah's own keep it as it is.
URBAN_IW = 13
URBAN_SMCWLT = 0.40
URBAN_SMCMAX = 0.45
OWN_LAND_SOURCES = ("native", "parent")


def _stas_wlt_max():
    """STAS SOILPARM wilting point and porosity per soil class (1-based class k at index k-1), host float32."""
    import numpy as np
    from namgpu.init import init_static_missing as ism
    t = ism._noah().load_tables(mminlu="MODIFIED_IGBP_MODIS_NOAH", mminsl="STAS", tbl_dir=ism.TBL_DIR)
    return np.asarray(t.wltsmc, np.float32).ravel(), np.asarray(t.maxsmc, np.float32).ravel()


def urban_soilw(soilw, vgtyp, sotyp, land, wlt=None, smax=None):
    """Soil moisture [4, JM, IM] with urban land points mapped from their soil class's range onto the urban one.

    theta_urban = 0.40 + 0.05 * clip((theta - SMCWLT_k) / (SMCMAX_k - SMCWLT_k), 0, 1), k = the point's STAS class.
    A class with no range (water, ice) takes the urban wilting point. Non-urban or water points are returned as given.
    Works on cupy or numpy arrays. Returns (array, number of urban land points changed)."""
    try:
        import cupy as cp
        xp = cp.get_array_module(soilw)
    except ImportError:          # CPU-only test environment
        import numpy as xp
    if wlt is None or smax is None:
        wlt, smax = _stas_wlt_max()
    wlt_d, max_d = xp.asarray(wlt, dtype=xp.float32), xp.asarray(smax, dtype=xp.float32)
    k = xp.clip(xp.asarray(sotyp).astype(xp.int32) - 1, 0, wlt_d.size - 1)
    w, m = wlt_d[k][None], max_d[k][None]
    span = m - w
    rel = xp.where(span > 0, xp.clip((soilw - w) / xp.where(span > 0, span, 1.0), 0.0, 1.0), 0.0)
    mapped = (xp.float32(URBAN_SMCWLT) + xp.float32(URBAN_SMCMAX - URBAN_SMCWLT) * rel).astype(xp.float32)
    urban = ((xp.asarray(vgtyp) == URBAN_IW) & (xp.asarray(land) > 0.5))[None]
    return xp.ascontiguousarray(xp.where(urban, mapped, soilw).astype(xp.float32)), int(urban.sum())


def soilw_for_source(raw, source, wlt=None, smax=None, ctx=None):
    """The soil moisture init-native-sfc starts from: the source's own, with urban_soilw applied when the source's
    land state is not NMMB Noah's own (OWN_LAND_SOURCES). With ctx, the soil-start row (soil_start.py,
    NAMGPU_SOIL_START: source | cdf | hrrr) is applied first; points a row already put in NMMB Noah's own range
    (the CDF match to the NAM climatology) skip the urban remap. Returns (array, urban points changed or None)."""
    if source is None or source in OWN_LAND_SOURCES:
        return raw["SOILW"], None
    soilw, keep = raw["SOILW"], None
    if ctx is not None:
        from namgpu.init import soil_start
        soilw, keep = soil_start.apply(ctx, raw, source)
    out, n = urban_soilw(soilw, raw["VGTYP"], raw["SOTYP"], raw["LAND"], wlt, smax)
    if keep is not None:
        import cupy as cp
        urban = (cp.asarray(raw["VGTYP"]) == URBAN_IW) & (cp.asarray(raw["LAND"]) > 0.5)
        out = cp.ascontiguousarray(cp.where(keep[None], soilw, out).astype(cp.float32))
        n = int((urban & ~keep).sum())
    return out, n


def _check_raw(raw, im, jm):
    missing = [k for k in RAW_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{UNIT}: ctx.tables['raw_sfc'] lacks {missing} (init-grib2-decode)")
    for k in RAW_KEYS + tuple(o for o in OPT_KEYS if o in raw):
        shp = tuple(raw[k].shape)
        want = (NSOIL, jm, im) if k in ("TSOIL", "SOILW") else (jm, im)
        if shp != want:
            raise ValueError(f"{UNIT}: raw {k} shape {shp}, want {want}")


def run(ctx):
    import cupy as cp
    import numpy as np

    st = ctx.st
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    raw = ctx.tables["raw_sfc"]
    soil = ctx.tables["soil_hydraulics"]
    _check_raw(raw, im, jm)

    bexp = cp.ascontiguousarray(soil["bexp"], dtype=cp.float32)
    psisat = cp.ascontiguousarray(soil["psisat"], dtype=cp.float32)
    smcmax = cp.ascontiguousarray(soil["smcmax"], dtype=cp.float32)
    if bexp.shape != (NCLASS,) or psisat.shape != (NCLASS,) or smcmax.shape != (NCLASS,):
        raise ValueError(f"{UNIT}: soil_hydraulics tables must have {NCLASS} classes")

    f = st.fields
    if "ALBEDO" not in f:
        st.add("ALBEDO", (jm, im))
        f = st.fields
    if "SNOWC" in raw:
        snowc_arr, has_snowc = raw["SNOWC"], 1
    else:
        snowc_arr, has_snowc = cp.zeros((jm, im), dtype=cp.float32), 0
    has_snowc_d = cp.asarray([has_snowc], dtype=cp.int32)
    source = (ctx.tables.get("grid_source") or {}).get("source")
    soilw, n_urban = soilw_for_source(raw, source, ctx=ctx)
    if n_urban is not None:
        print(f"{UNIT}: {source} soil moisture on {n_urban} urban land points mapped onto the NAM urban range "
              f"{URBAN_SMCWLT}-{URBAN_SMCMAX} (relative wetness of the soil class kept)", flush=True)
    k = kernels.get(KFILE, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    grid, block = kernels.grid_cols(im, jm), (BLOCK,)
    with scope:
        k(grid, block, (
            raw["TSOIL"], soilw, raw["TMP_SFC"], raw["ICEC"], raw["LAND"],
            raw["TSOIL_3M"], raw["SNOD"], raw["WEASD"], raw["ALBDO"], raw["VEG"], raw["SFCR"],
            snowc_arr, has_snowc_d, raw["CNWAT"], raw["VGTYP"], raw["SOTYP"],
            bexp, psisat, smcmax,
            f["STC"], f["SMC"], f["SH2O"],
            f["TSKIN"], f["SST"], f["SICE"], f["SM"], f["TG"], f["SI"], f["SNO"], f["SNOWC"],
            f["ALBEDO"], f["VEGFRC"], f["Z0"], f["Z0BASE"], f["CMC"],
            f["IVGTYP"], f["ISLTYP"],
            np.int32(im), np.int32(jm),
        ))
        # A source that carries the model's own liquid soil water (the parent hand-off, parent_start.py) keeps it
        # on land instead of the kernel's frozen-fraction split, capped at the total. So does a native start that has
        # NCEP's bgrdsf00 land records (sources.py). No other source sets this.
        ov = ctx.tables.get("sfc_override") if getattr(ctx, "tables", None) is not None else None
        if ov and "SH2O" in ov:
            land = (f["SM"] < 0.5)[None]
            f["SH2O"][...] = cp.where(land, cp.minimum(cp.maximum(ov["SH2O"], 0.0), f["SMC"]), f["SH2O"])
    return st


def _selftest():
    """Full-size random but plausible state (954 x 835 x 60 grid, the real NAM parent size)."""
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace

    from namgpu import state as statemod

    im, jm, lm = 954, 835, 60
    grid = SimpleNamespace(im=im, jm=jm, lm=lm)
    st = statemod.State(grid)
    rng = cp.random.default_rng(7)
    raw = {
        "TSOIL": 250.0 + 60.0 * rng.random((NSOIL, jm, im), dtype=cp.float32),
        "SOILW": 0.02 + 0.5 * rng.random((NSOIL, jm, im), dtype=cp.float32),
        "TMP_SFC": 250.0 + 60.0 * rng.random((jm, im), dtype=cp.float32),
        "ICEC": rng.random((jm, im), dtype=cp.float32),
        "LAND": (rng.random((jm, im), dtype=cp.float32) > 0.5).astype(cp.float32),
        "TSOIL_3M": 260.0 + 30.0 * rng.random((jm, im), dtype=cp.float32),
        "SNOD": 0.5 * rng.random((jm, im), dtype=cp.float32),
        "WEASD": 200.0 * rng.random((jm, im), dtype=cp.float32),
        "ALBDO": 100.0 * rng.random((jm, im), dtype=cp.float32),
        "VEG": 100.0 * rng.random((jm, im), dtype=cp.float32),
        "SFCR": 0.01 + rng.random((jm, im), dtype=cp.float32),
        "SNOWC": rng.random((jm, im), dtype=cp.float32),
        "CNWAT": 0.1 * rng.random((jm, im), dtype=cp.float32),
        "VGTYP": rng.integers(1, 21, (jm, im), dtype=cp.int32),
        "SOTYP": rng.integers(1, 13, (jm, im), dtype=cp.int32),
    }
    # Test-only soil classes (the real table comes from init-static-missing).
    soil = {
        "bexp": (2.5 + 10.0 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
        "psisat": (0.03 + 0.7 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
        "smcmax": (0.33 + 0.25 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
    }
    ctx = SimpleNamespace(st=st, grid=grid, stream=cp.cuda.Stream(non_blocking=True),
                          tables={"raw_sfc": raw, "soil_hydraulics": soil})
    run(ctx)
    ctx.stream.synchronize()
    bad = []
    for name in OUT_SOIL + OUT_FLOAT_2D:
        a = st.fields[name]
        if not bool(cp.isfinite(a).all()):
            bad.append(name)
    frozen = float(cp.mean(st.fields["SMC"] - st.fields["SH2O"]))
    print(f"{UNIT}: fields {len(OUT_SOIL) + len(OUT_FLOAT_2D) + 2}, "
          f"NaN/Inf in {bad or 'none'}, mean frozen fraction {frozen:.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
