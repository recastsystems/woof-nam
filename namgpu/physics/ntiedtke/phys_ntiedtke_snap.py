"""phys-ntiedtke-snap: T and Q right after GSMDRIVE, the reference for New Tiedtke's large-scale forcing.

Plan entry after phys-fer-driver (cadence nprecip), on the plan only with ``convection: ntiedtke``
(driver.CONFIG_GATES). It runs before CLTEND mode 0 resets T, so TSNAP holds T after the convective and
microphysical change of this cloud step; by the next call CLTEND has applied that change in full (NPRECIP
increments of Tadj), so T - TSNAP at the next call is dynamics + radiation + TURBL + TQADJUST only. See
phys_ntiedtke_cu's docstring. One copy kernel on ctx.stream; NT_TSNAP and NT_QSNAP are whole-domain State
arrays added once; the step number goes to ctx.tables["ntiedtke_snap_step"] (host int, no sync).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_ntiedtke_map"


def run(ctx):
    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ts = st.add("NT_TSNAP", (lm, jm, im))
    qs = st.add("NT_QSNAP", (lm, jm, im))
    n = lm * jm * im
    blk = 256
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, "nt_snap")(((n + blk - 1) // blk,), (blk,), (np.int32(n), st["T"], st["Q"], ts, qs))
    ctx.tables["ntiedtke_snap_step"] = int(ctx.ntimestep)
