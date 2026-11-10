"""phys-rrtm-progcld8: PROGCLD8 (np3d 8) cloud optical properties for RRTM on the thompson tracer layout.

Fortran phys/radiation_clouds_nmmb.f:1329-1542 (nam.v4.2.11, 83df4e13c1), gethml inlined as in phys-rrtm-progcld2.
Radii: TEMPO's own (RE_CLOUD, RE_ICE, RE_SNOW, metres, written by phys-tempo-driver) in place of Thompson's
calc_effectRad; Fortran initial values where TEMPO has none. modernise/mp-tempo lane.
Kernel: namgpu/kernels/phys_rrtm_progcld8.cu (one thread per column, one launch).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_cld_init as cld_init
from namgpu.physics.rrtm import phys_rrtm_progcld2 as progcld2

UNIT = "phys_rrtm_progcld8"
NF_CLDS = 9


def run(ctx, inputs, *, ivflip: int, llyr: int, iovr: int, cld_fraction: int = 0):
    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk = int(inputs["LMK"])
    if lmk > 128:
        raise RuntimeError(f"{UNIT}: LMK={lmk} exceeds the kernel's per-column array (128)")
    if cld_fraction != 0 and inputs["cldfk1"] is None:
        raise RuntimeError(f"{UNIT}: cld_fraction={cld_fraction} needs cldfk1")
    s3 = (lm, jm, im)
    re = [st.add(n, s3) for n in ("RE_CLOUD", "RE_ICE", "RE_SNOW")]
    clouds = progcld2._scratch(st, "PGC8_CLOUDS", (NF_CLDS, lmk, jm, im))
    clds = progcld2._scratch(st, "PGC8_CLDS", (5, jm, im))
    mtop = progcld2._scratch(st, "PGC8_MTOP", (3, jm, im), dtype="int32")
    mbot = progcld2._scratch(st, "PGC8_MBOT", (3, jm, im), dtype="int32")
    i32 = np.int32
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, "phys_rrtm_progcld8_column")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lmk), i32(lm), i32(ivflip), i32(llyr), i32(iovr), i32(cld_fraction),
            inputs["plyr"], inputs["plvl"], inputs["tlyr"], inputs["qlyr"], inputs["qc2d"], inputs["qi2d"],
            inputs["qs2d"], inputs["cldfk1"], inputs["xlat"], re[0], re[1], re[2], clouds, clds, mtop, mbot))
    return {"clouds": clouds, "clds": clds, "mtopa": mtop, "mbota": mbot}


def dep(name, ctx, inputs):
    """grrad_clouds dependency entry for both NMMB prognostic cloud branches this tree runs."""
    if name == "progcld2":
        return progcld2.dep(name, ctx, inputs)
    if name != "progcld8":
        raise RuntimeError(f"{UNIT}: dep called for {name!r}")
    cld = cld_init.check_status(ctx)
    return run(ctx, inputs, ivflip=int(ctx.cfg.ivflip), llyr=cld["llyr"], iovr=cld["iovr"],
               cld_fraction=int(inputs["cld_fraction"]))
