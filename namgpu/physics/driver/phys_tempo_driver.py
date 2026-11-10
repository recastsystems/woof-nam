"""phys-tempo-driver: GSMDRIVE (the NMMB 'thompson' route) with selectable TEMPO microphysics.

Selected by configfile `microphysics: tempo` (config.load maps it onto NMMB's thompson water layout:
cfg.microphysics = "thompson", cfg.mp_column = "tempo"), so every other NMMB unit (tracer slots, UPDATE_WATER,
RRTM np3d 8, cloud fraction, history) runs its Fortran thompson branch, and this unit replaces mp_gt_driver.
TEMPO is the RRFS/HRRRv5 successor of Thompson (NCAR/TEMPO c62699fb660e / 9adf9ef, Apache-2.0); the column kernel is
gpuwm-hex's hexcore.mpasphys.rrfs.tempo_v325 (default) or tempo_v303, tables from
namgpu.physics.tempo.tables (hexcore's ported tempo_init: Rust, no Fortran).

Fortran: nmm/module_MICROPHYSICS.F90:60-558 (GSMDRIVE), thompson branch 418-452, USE_QV 275-298 and 488-503.
Sequence per microphysics call (every NPRECIP steps), all on ctx.stream, in row chunks so the TEMPO work area
(about 30 KB per column) stays bounded (NAMGPU_TEMPO_CHUNK_MIB, default 2048):
  tempo_pack -> hexcore launch_columns (TEMPO main column, reflectivity_10cm on) -> tempo_unpack -> tempo_surface.
Hydrometeors live in the NMMB TRACERS water slots (QC QI QR QS QG NI NR, drv-tracer-index thompson layout),
advected with spec_adv .true.; Q is specific humidity in NMMB and mixing ratio in TEMPO (converted at the seam).
TEMPO options: non-aerosol, no hail prognostics (presence 0): the seven NMMB thompson species, constant droplet
number. REFL_10CM is TEMPO's own 10 cm reflectivity (HRRR-style, from the TEMPO size distributions).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import os

import numpy as np

from namgpu import kernels

UNIT = "phys_tempo_driver"
SPECIES = ("QC", "QR", "QI", "QS", "QG", "NI", "NR")
FLAGS = (0, 0, 0, 0)          # aerosolaware, hailaware, ml_for_bl_nc, refl10cm_from_melting
ENV_MELT = "NAMGPU_TEMPO_REFL_MELT"   # 1: REFL_10CM adds TEMPO's melting snow/graupel reflectivity (bright band);
                                       # diagnostic only, on by default for both named versions.
ENV_VERSION = "NAMGPU_TEMPO_VERSION"
VERSIONS = {"v3.2.5": "325", "v3.0.3": "303"}


def _version(raw=None) -> str:
    name = str(os.environ.get(ENV_VERSION) or (raw or {}).get("tempo_version") or "v3.2.5").strip()
    if name not in VERSIONS:
        raise ValueError(f"{ENV_VERSION}/tempo_version={name!r}: choose one of {tuple(VERSIONS)}")
    return name


def _melt() -> bool:
    return os.environ.get(ENV_MELT, "1").strip().lower() in ("1", "true", "on", "yes")


def _flags(version):
    flags = FLAGS[:3] + (1 if _melt() else 0,)
    # v3.2.5 adds hailhyperaware; upstream forces it off without hail-aware carriers.
    return flags + (flags[1],) if version == "v3.2.5" else flags
PRESENCE = 0                  # nc nwfa nifa ng qb carriers absent
ENV_CHUNK = "NAMGPU_TEMPO_CHUNK_MIB"
ENV_KERNEL = "NAMGPU_TEMPO_KERNEL"   # warp (default) | column


def _hexcore(version):
    try:
        from hexcore.mpasphys.rrfs import tempo_v303 as base
        from hexcore.mpasphys import resources_build as rb
        tv = importlib.import_module(f"hexcore.mpasphys.rrfs.tempo_v{VERSIONS[version]}")
        ph = importlib.import_module(f"hexcore.mpasphys.rrfs.tempo_phase_v{VERSIONS[version]}")
    except ImportError as e:   # the run cannot do microphysics without the column kernel
        raise RuntimeError(f"phys-tempo-driver: TEMPO {version} needs gpuwm-hex>=0.5.0 (hexcore) importable "
                           f"(pip install gpuwm-hex, or NAMGPU_HEXCORE on PYTHONPATH): {e}") from e
    return tv, ph, base, rb


def _setup(ctx):
    import cupy as cp
    t = ctx.tables.get("tempo")
    if t is not None and "tv" in t:
        return t
    # a resumed run: the restart set leaves "tempo" out (restart.TRANSIENT_TABLES; it holds the hexcore module
    # and a ~0.5 GB device blob rebuilt from the cache in seconds). Defect fixed: KeyError tv on --resume.
    version = _version(ctx.cfg.raw)
    tv, ph, base, rb = _hexcore(version)
    flags = _flags(version)
    from namgpu.physics.tempo.tables import load_blob
    host, receipt = load_blob(hailaware=True, melt=bool(flags[3]))
    blob = cp.asarray(np.ascontiguousarray(host))
    if base._fast():              # both versions use the same table ABI and conversion
        blob = base.fast_tables(blob)
    maths_path = os.environ.get("NAMGPU_TEMPO_MATHS")
    if maths_path:
        maths = open(maths_path).read()
    else:
        maths = _maths(rb)
    tab = ctx.tables.get("indx")
    if tab is None:
        from namgpu.steps.drv_tracer_index import host_index
        tab = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
    for n in SPECIES:
        if int(tab[f"INDX_{n}"]) <= 0:
            raise RuntimeError(f"phys-tempo-driver: no TRACERS slot for {n} (microphysics {ctx.cfg.microphysics!r})")
    receipt = dict(receipt, tempo_version=version, flags=flags,
                   device_math_sha256=hashlib.sha256(maths.encode()).hexdigest())
    t = {"tv": tv, "ph": ph, "base": base, "version": version, "flags": flags,
         "blob": blob, "maths": maths, "first": True, "receipt": receipt,
         "slots": {n: int(tab[f"INDX_{n}"]) - 1 for n in SPECIES}}
    ctx.tables["tempo"] = t
    ctx.cfg.raw["tempo_version"] = version   # a restart retains the selected version without the original env
    print(f"phys-tempo-driver: {version}, melting reflectivity={flags[3]}, "
          f"device maths sha256={receipt['device_math_sha256']}", flush=True)
    return t


def _maths(rb):
    """Bind the installed hex physics packet's certified TEMPO device header, shared by both versions."""
    from hexcore.mpasphys.table_resources_v842 import certified_path
    packet = rb.load_packet(rb.resolve_resource("physics-resources"))
    return certified_path(packet["mp_tempo"]["device_math"]).read_text()


def _vec(ctx, attr):
    import cupy as cp
    key = "tempo_" + attr
    if key not in ctx.tables:
        ctx.tables[key] = cp.asarray(np.asarray(getattr(ctx.grid, attr), dtype=np.float32))
    return ctx.tables[key]


def _rows_per_chunk(ctx, t, im, jm, lm):
    mib = float(os.environ.get(ENV_CHUNK, "2048"))
    per_col = t.get("bytes_per_col")
    if per_col is None:
        tv = t["tv"]
        import cupy as cp
        _, (work_b, state_b, _) = tv._compiled(cp.cuda.runtime.getDevice(), t["maths"], lm)
        per_col = work_b + state_b + 4 * lm * (17 + 15 + 15 + 2)
        t["bytes_per_col"] = per_col
    rows = int(mib * 2 ** 20 // (per_col * im))
    return max(1, min(jm, rows))


def run(ctx, use_radar: bool = False, nprecip: int | None = None):
    if os.environ.get(ENV_KERNEL, "warp").strip() != "column":
        return _run_warp(ctx, use_radar=use_radar, nprecip=nprecip)
    import cupy as cp
    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    t = _setup(ctx)
    tv = t["tv"]
    nphs = int(cfg.nprecip if nprecip is None else nprecip)
    dtphs = np.float32(nphs) * np.float32(ctx.dt)
    rdtphs = np.float32(1.0) / dtphs

    trac = st["TRACERS"]
    sp = {n: trac[s] for n, s in t["slots"].items()}
    s2 = (jm, im)
    rainncv = st.add("RAINNCV", s2)
    acs, acg, acr = st.add("ACPCP_SN", s2), st.add("ACPCP_GR", s2), st.add("ACPCP_RA", s2)
    pdsg1, psgml1 = _vec(ctx, "pdsg1"), _vec(ctx, "psgml1")
    re_c, re_i, re_s = (st.add(n, (lm, jm, im)) for n in ("RE_CLOUD", "RE_ICE", "RE_SNOW"))
    rows = _rows_per_chunk(ctx, t, im, jm, lm)
    k_pack, k_unpack, k_sfc = (kernels.get(UNIT, k) for k in ("tempo_pack", "tempo_unpack", "tempo_surface"))
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for j0 in range(0, jm, rows):
            nr = min(rows, jm - j0)
            ncol = nr * im
            f = cp.empty((17, lm, ncol), dtype=cp.float32)
            g3 = kernels.grid_2d(im, nr, lm)
            k_pack(g3, block, (np.int32(im), np.int32(jm), np.int32(lm), np.int32(j0), np.int32(nr),
                               st["PD"], st["T"], st["Q"], st["W"], st["DSG2"], st["SGML2"], pdsg1, psgml1,
                               sp["QC"], sp["QR"], sp["QI"], sp["QS"], sp["QG"], sp["NI"], sp["NR"], f))
            diag = tv.launch_columns(f, t["blob"], dt=float(dtphs), flags=t["flags"], device_math_source=t["maths"],
                                     first_call_main=t["first"], pbl=None, presence=PRESENCE)
            if t["version"] == "v3.0.3":
                t["first"] = False   # retain the older version's first-call behavior
            k_unpack(g3, block, (np.int32(im), np.int32(jm), np.int32(lm), np.int32(j0), np.int32(nr),
                                 f, diag, st["DFI_TTEN"], rdtphs, np.int32(1 if use_radar else 0),
                                 st["T"], st["Q"], st["CW"], st["TRAIN"],
                                 sp["QC"], sp["QR"], sp["QI"], sp["QS"], sp["QG"], sp["NI"], sp["NR"],
                                 st["REFL_10CM"], re_c, re_i, re_s))
            k_sfc(kernels.grid_2d(im, nr, 1), block, (
                np.int32(im), np.int32(jm), np.int32(lm), np.int32(j0), np.int32(nr), diag,
                st["AVRAIN"], rainncv, st["SR"], st["PREC"], st["ACPREC"], st["ACPREC_TOT"], acs, acg, acr))
            del f, diag
    t["first"] = False               # v3.2.5 initializes every column of the first timestep
    return {"RAINNCV": rainncv}


def _warp_module(ctx, t, lm):
    """hexcore's fast TEMPO modules for this card: the warp kernel module and the phase module, both with the
    binary32 blob's constant-bank parameter block written (as launch_mpas does before every slot)."""
    import cupy as cp
    ph, base = t["ph"], t["base"]
    dev = cp.cuda.runtime.getDevice()
    pmod, ms, sizes, resident, words = ph._modules(dev, t["maths"], lm, ph.inline_mode())
    blob = ph.float_blob(t["blob"])
    base._set_scalars(pmod, blob)
    base._set_scalars(ms, blob)
    return ph, dev, blob


def _run_warp(ctx, use_radar: bool = False, nprecip: int | None = None):
    import cupy as cp
    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    t = _setup(ctx)
    ph, dev, blob = _warp_module(ctx, t, lm)
    nphs = int(cfg.nprecip if nprecip is None else nprecip)
    dtphs = np.float32(nphs) * np.float32(ctx.dt)
    rdtphs = np.float32(1.0) / dtphs
    trac = st["TRACERS"]
    sp = {n: trac[s] for n, s in t["slots"].items()}
    s2 = (jm, im)
    rainncv = st.add("RAINNCV", s2)
    acs, acg, acr = st.add("ACPCP_SN", s2), st.add("ACPCP_GR", s2), st.add("ACPCP_RA", s2)
    re_c, re_i, re_s = (st.add(n, (lm, jm, im)) for n in ("RE_CLOUD", "RE_ICE", "RE_SNOW"))
    pdsg1, psgml1 = _vec(ctx, "pdsg1"), _vec(ctx, "psgml1")
    mib = float(os.environ.get(ENV_CHUNK, "2048"))
    per_col = 4 * lm * (12 + 1 + 4) + 4 * 12
    rows = max(1, min(jm, int(mib * 2 ** 20 // (per_col * im))))
    k_pack, k_unpack, k_sfc = (kernels.get(UNIT, k) for k in ("tempo_pack_w", "tempo_unpack_w", "tempo_surface_w"))
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for j0 in range(0, jm, rows):
            nr = min(rows, jm - j0)
            ncol = nr * im
            f = cp.empty((12, lm, ncol), dtype=cp.float32)
            pii = cp.empty((lm, ncol), dtype=cp.float32)
            prof = cp.zeros((4, lm, ncol), dtype=cp.float32)
            sfc = cp.zeros((12, ncol), dtype=cp.float32)
            g3 = kernels.grid_2d(im, nr, lm)
            k_pack(g3, block, (np.int32(im), np.int32(jm), np.int32(lm), np.int32(j0), np.int32(nr),
                               st["PD"], st["T"], st["Q"], st["W"], st["DSG2"], st["SGML2"], pdsg1, psgml1,
                               sp["QC"], sp["QR"], sp["QI"], sp["QS"], sp["QG"], sp["NI"], sp["NR"], f, pii))
            fp = [f[i].data.ptr for i in range(12)] + [0] * 5 + [pii.data.ptr, 0, 0]
            fptr = cp.asarray(np.asarray(fp, dtype=np.uint64))
            sptr = cp.asarray(np.asarray([sfc[i].data.ptr for i in range(12)], dtype=np.uint64))
            pptr = cp.asarray(np.asarray([prof[i].data.ptr for i in range(4)], dtype=np.uint64))
            ph.launch_steps(device=dev, nz=lm, cells=ncol, steps=1, dt=float(dtphs), f=fptr, bl=None, s=sptr,
                            profiles=pptr, tables=blob, flags=t["flags"], first=t["first"], presence=PRESENCE,
                            radii=True, device_math_source=t["maths"], force_warp=True)
            if t["version"] == "v3.0.3":
                t["first"] = False
            k_unpack(g3, block, (np.int32(im), np.int32(jm), np.int32(lm), np.int32(j0), np.int32(nr),
                                 f, pii, prof, st["DFI_TTEN"], rdtphs, np.int32(1 if use_radar else 0),
                                 st["T"], st["Q"], st["CW"], st["TRAIN"],
                                 sp["QC"], sp["QR"], sp["QI"], sp["QS"], sp["QG"], sp["NI"], sp["NR"],
                                 st["REFL_10CM"], re_c, re_i, re_s))
            k_sfc(kernels.grid_2d(im, nr, 1), block, (
                np.int32(im), np.int32(jm), np.int32(j0), np.int32(nr), sfc,
                st["AVRAIN"], rainncv, st["SR"], st["PREC"], st["ACPREC"], st["ACPREC_TOT"], acs, acg, acr))
            del f, pii, prof, sfc, fptr, sptr, pptr
    t["first"] = False
    return {"RAINNCV": rainncv}
