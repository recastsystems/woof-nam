"""rrtm-call: SUBROUTINE RRTM of module_RA_RRTM.F90 as one chain of the RRTM units, in Fortran order.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_RA_RRTM.F90:98-1418 (RRTM) and
phys/grrad_nmmb.f:660-2210 (grrad_nmmb). NOAA US government work; each unit's kernel keeps the
NOAA header as its notice. This module has no kernel of its own: it is the call order.

Called by phys-rad-driver (module_RADIATION.F90:586-625) on radiation steps as run(ctx, args),
args = the Fortran RRTM argument dict built by phys_rad_driver._rrtm_args.

Order (one radiation call):
  module_RA_RRTM.F90
    98-878     phys-rrtm-driver-prep     control scalars and P8W/P_PHY (whole domain: they come from the
                                         J = JTE row, see its header)
  then per row chunk (rrtm_chunk: rows in chunks of ROWS, columns independent):
    98-878     phys-rrtm-driver-prep     cloud layering, TAUTOTAL, CUTOP/CUBOT (kernel 2, per column, on
                                         the chunk's rows: no whole-domain scratch)
    879-881    phys-rrtm-dayparts        COSZMN over the SW interval, sunlit runs per row
    879-1112   phys-rrtm-driver-call     grrad inputs per column (prsi, prsl, gt, gq, gr1, surface)
  grrad_nmmb.f
    660-1396   phys-rrtm-grrad-setup     plvl/plyr/tlyr/prslk1/rhly/qstl, GR1 tracer, cloud fractions
    1397-1407  phys-rrtm-getozn          climatological ozone (ntoz == 0)
    1409-1414  phys-rrtm-coszmn          coszen_loc, coszdg
    1419-1427  phys-rrtm-getgases        gas volume mixing ratios
    1428-1539  phys-rrtm-grrad-gases     tlvl, tvly, qlyr, tsfa
    1540-1576  phys-rrtm-grrad-aerosols  setaer -> faersw, faerlw
    1556-1794  phys-rrtm-grrad-clouds    clw, progcld2 (np3d 3), IALB=2 albedo
    1818-1962  phys-rrtm-grrad-sw        setalb, swrad          (lsswr)
    1963-2201  phys-rrtm-grrad-lw        setemis, lwrad, fluxr  (lslwr inside; fluxr/coszen always)
  module_RA_RRTM.F90
    1113-1418  phys-rrtm-driver-post     RSWTT, RLWTT, RSWIN, RLWIN, ..., cloud fractions
  and the chunk's owned rows of every state field it touched go back to the State.

lsswr = MOD(NTIMESTEP, NRADS) == 0, lslwr = MOD(NTIMESTEP, NRADL) == 0 (module_RA_RRTM.F90:842-846),
NTIMESTEP = NTIMESTEP_RAD. dtsw = NRADS*DT, dtlw = NRADL*DT.

The runtime physparam flags are the ones rad_initialize_nmmb.f:134-183 sets from the NAM solver
values (module_SOLVER_GRID_COMP.F90:10650-10700): ioznflg = NTOZ = 0, ico2flg = ICO2 = 1,
ivflip = IFLIP = 0 (namgpu/config.py carries those runtime values).

Memory: ROWS rows per chunk (default 8, owned 6 per chunk over JM = 835). The chain's scratch is held for
one chunk only (about 1/10 of the whole-domain need); see rrtm_chunk for why the whole domain does
not fit. Chunk scratch is allocated on the first call and reused (one chunk shape).

Batches (steploop-hostbound, 2026-11-09): the row windows of ROWS rows stay the radiation's decomposition
(they decide the SW MINVAL(laytrop) groups of taumol19/23/25/28, so they decide the output), but up to B
consecutive windows run as ONE chunk: one launch set over all their rows. On the 32 GB fit path the windows
were 8 rows, 7,632 columns per launch: 139 chunks, 52,681 launches and 278 host reads per call, the GPU on a
few dozen SMs, and radiation 71 % of the step loop (profile, 2026-11-09).
sw_seam.window_kmins rebuilds each column's groups from its own window inside the batch, so the output is the
same for every B. B is sized each call from the memory the card leaves (batch_windows); B = 1 is the old loop.
The SW day count of each chunk is counted on the host once per call for every row (day_rows), not read back
per chunk.
"""
from __future__ import annotations

import os

from namgpu.physics.rrtm import (
    phys_rrtm_coszmn as coszmn,
    phys_rrtm_dayparts as dayparts,
    phys_rrtm_driver_call as driver_call,
    phys_rrtm_driver_post as driver_post,
    phys_rrtm_driver_prep as driver_prep,
    phys_rrtm_getgases as getgases,
    phys_rrtm_getozn as getozn,
    phys_rrtm_grrad_aerosols as grrad_aerosols,
    phys_rrtm_grrad_clouds as grrad_clouds,
    phys_rrtm_grrad_gases as grrad_gases,
    phys_rrtm_grrad_lw as grrad_lw,
    phys_rrtm_grrad_setup as grrad_setup,
    phys_rrtm_grrad_sw as grrad_sw,
    phys_rrtm_progcld2 as progcld2,
    phys_rrtm_progcld8 as progcld8,
)
from namgpu.physics.rrtm import rrtm_chunk

# NAM iflip = 0 (module_SOLVER_GRID_COMP.F90:10714): input from TOA to surface.
IVFLIP_NAM = 0
ROWS = int(os.environ.get("NAMGPU_RRTM_ROWS", "32"))   # 32 rows: about 6 GB of chain scratch; 8 rows was launch-bound
KEY_CHUNK = "rrtm_call:chunkctx"
BATCH_KEY = "rrtm_call:batch"          # chunk table: window geometry of the current batch (device), B > 1 only
NDAY_KEY = "rrtm_call:nday"            # chunk table: host day-column count of the current batch (SW calls)
NDAY_CHECK_KEY = "rrtm_call:nday_check"  # parent table: device count of host/device day-count mismatches
GEOM_KEY = "rrtm_call:geom"            # parent table prefix: uploaded batch geometry per (JM, ROWS, B)
PLAN_KEY = "rrtm_call:plan"            # parent table: the last batch choice (printed when it changes)
# Pool bytes the radiation call may grow this process to on the 32 GB fit path (CuPy pool, GiB): the fit path is
# the one that fits a 32 GB card, and a 3-way shared 96 GB card gives each process the same ~30 GB. 24 GiB of
# pool + the local-memory reservation (2.7 GiB at 12 km) + context stays under 28 GiB per process.
FIT_POOL_GIB = 24.0
# RRTM chain bytes per row-column for the first batch choice (nest/domain.VRAM_PATH rrtm_mb_per_row_col, measured
# through the door): fit 0.26 MB, full 0.59 MB. Later calls use the larger of this and what the last call held.
ROW_COL_BYTES = {"fit": 0.26e6, "full": 0.59e6}
# driver-call 2-D outputs that are GRRAD actual arguments (module_RA_RRTM.F90 CALL GRRAD_NMMB).
GRRAD_SFC_ARGS = ("XLON", "XLAT", "SINLAT", "COSLAT", "SLMSK", "TSEA", "TISFC", "FICE", "ZORL",
                  "SNWDPH", "SNCOVR", "SNOALB", "HPRIME_V", "SFCALBEDO", "SMX", "CV", "CVB", "CVT",
                  "FLGMIN_L")


def batches(jm: int, rows: int, b: int):
    """[(W0, W1, a, b, windows)] per batch: up to `b` consecutive rrtm_chunk.windows(jm, rows) computed together
    over rows [W0, W1) (every batch the same height, so one chunk context serves them all; the last batch reaches
    back over rows an earlier batch owns, which it computes and does not write back), owning [a, b)."""
    wins = rrtm_chunk.windows(jm, rows)
    if b <= 1:
        return [(w0, w1, a, bb, [(w0, w1, a, bb)]) for (w0, w1, a, bb) in wins]
    if b >= len(wins):
        groups, height = [wins], jm
    else:
        groups = [wins[k:k + b] for k in range(0, len(wins), b)]
        height = max(g[-1][1] - g[0][0] for g in groups)
    out = []
    for g in groups:
        w1 = g[-1][1]
        w0 = min(g[0][0], w1 - height)
        if w0 < 0:
            w0, w1 = 0, height
        out.append((w0, w1, g[0][2], g[-1][3], g))
    return out


def batch_geometry(W0: int, W1: int, wins) -> dict:
    """Host geometry of one batch for sw_seam.window_kmins: batch-local window rows and the owner of each row."""
    import numpy as np
    owner = np.empty(W1 - W0, dtype=np.int32)
    for r in range(W1 - W0):
        g = W0 + r
        k = next((k for k, (_, _, a, b) in enumerate(wins) if a <= g < b), None)
        owner[r] = k if k is not None else (0 if g < wins[0][2] else len(wins) - 1)
    return {"nwin": len(wins), "win_r0": np.array([w[0] - W0 for w in wins], dtype=np.int32),
            "win_r1": np.array([w[1] - W0 for w in wins], dtype=np.int32), "row_owner": owner}


def _device_geometry(ctx, jm: int, rows: int, b: int, plan):
    """Device copies of every batch's geometry, uploaded once per (JM, ROWS, B) and kept in the parent tables."""
    import cupy as cp
    key = f"{GEOM_KEY}:{jm}:{rows}:{b}"
    geo = ctx.tables.get(key)
    if geo is None:
        geo = []
        for (W0, W1, _a, _b, wins) in plan:
            h = batch_geometry(W0, W1, wins)
            geo.append({"nwin": h["nwin"], "win_r0": cp.asarray(h["win_r0"]), "win_r1": cp.asarray(h["win_r1"]),
                        "row_owner": cp.asarray(h["row_owner"])})
        ctx.tables[key] = geo
    return geo


def batch_windows(ctx, jm: int, im: int, rows: int, fit: bool) -> int:
    """Windows per batch, from the memory this process may grow to now. NAMGPU_RRTM_BATCH forces it (A/B, tests).
    1 when a window spans more than one SW chunk (the reference groups would then be the SW chunks') or the
    card has no room for more than one window."""
    from namgpu.physics.rrtm import phys_rrtm_sw_swrad as _swr
    nw = len(rrtm_chunk.windows(jm, rows))
    forced = os.environ.get("NAMGPU_RRTM_BATCH", "").strip()
    if forced:
        return max(1, min(int(forced), nw))
    if nw == 1 or rows * im > _swr.NC_DEFAULT or rows < rrtm_chunk.MIN_ROWS:
        return 1
    if not fit and ctx.tables.get(PLAN_KEY) is not None:
        return int(ctx.tables[PLAN_KEY]["b"])          # full path: the chunk context is kept, so B is kept
    import cupy as cp
    from namgpu import devmem
    try:
        pool = devmem._STATE.get("pool") or cp.get_default_memory_pool()
        used, held = int(pool.used_bytes()), int(pool.total_bytes())
        free = int(cp.cuda.runtime.memGetInfo()[0])
    except Exception:  # noqa: BLE001  (async pool without the counters)
        return 1
    g = 2 ** 30
    cap = held + free - (1 if fit else 2) * g
    ceiling = int(devmem._STATE.get("ceiling") or 0)
    if ceiling:
        cap = min(cap, ceiling - g // 4)
    if fit:
        cap = min(cap, int(float(os.environ.get("NAMGPU_RRTM_POOL_GIB", FIT_POOL_GIB)) * g))
    per_row = max(ROW_COL_BYTES["fit" if fit else "full"] * im, float(ctx.tables.get("rrtm_call:row_bytes", 0.0)))
    # phys-rrtm-lw-lwrad takes at most JC_DEFAULT rows per chunk (its scratch is sized per chunk; larger domains go
    # through rrtm_call), so a batch is never taller than that
    from namgpu.physics.rrtm import phys_rrtm_lw_lwrad as _lwr
    rows_max = min((cap - used) / per_row, float(_lwr.JC_DEFAULT))
    if rows_max < rows + (rows - 2):
        return 1
    b = max(1, min(nw, 1 + int((rows_max - rows) // (rows - 2))))
    if b >= nw and jm > rows_max:
        b = nw - 1                                   # one batch would be the whole domain, taller than rows_max
    return max(1, b)


def day_rows(ctx):
    """Day columns per row for this SW call, host int64 [JM]: DAYPARTS coszen_loc >= 0.0001 (the grrad_sw day
    mask, same kernels on the same per-column inputs as each chunk runs them). The one device-to-host read of
    the call's day list; every chunk's SW_NDAY is the sum over its rows (checked on the device, NDAY_CHECK_KEY)."""
    import contextlib as _cl
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_grrad_sw as _gsw
    st = ctx.st
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    nrads = int(ctx.cfg.nrads)
    scope = ctx.stream if ctx.stream is not None else _cl.nullcontext()
    with scope:
        glat = cp.asarray(st["GLAT"], dtype=cp.float32).reshape(-1)
        glon = cp.asarray(st["GLON"], dtype=cp.float32).reshape(-1)
        s = dayparts.dayparts(glon, cp.sin(glat), cp.cos(glat), float(ctx.clock.solhr),
                              float(nrads) * float(ctx.cfg.dt) / 3600.0, nrads, st[dayparts.SOL_STATE], im)
        day = cp.empty(jm * im, dtype=cp.int32)
        kernels.get(_gsw.UNIT, "grrad_sw_daymask")(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
                                                     (im, jm, s["coszen"], day))
        rows = day.reshape(jm, im).sum(axis=1, dtype=cp.int64)
        return rows.get(stream=ctx.stream) if ctx.stream is not None else rows.get()


def _check_nday(ctx) -> None:
    """Raise if a chunk's device day count differed from its host count (read where the stream is idle)."""
    chk = ctx.tables.get(NDAY_CHECK_KEY)
    if chk is not None and int((chk.get(stream=ctx.stream) if ctx.stream is not None else chk.get())[0]) != 0:
        raise RuntimeError("rrtm-call: a chunk's device SW day count differed from rrtm_call.day_rows "
                           "(the SW launches would have missed day columns)")


def _chunk_ctx(ctx, rows):
    cc = ctx.tables.get(KEY_CHUNK)
    if cc is None or cc.w != rows:
        cc = rrtm_chunk.ChunkCtx(ctx, rows)
        ctx.tables[KEY_CHUNK] = cc
    return cc


def run(ctx, args: dict, *, stamp=None, rows: int | None = None, sw: bool = True) -> None:
    """sw=False is a test-harness knob only (tools/run_rrtm_real.py --lw-only): it skips grrad-sw and says
    so in ctx.tables['rrtm_call_flags']; the forecast path always runs SW on SW steps."""
    cfg = ctx.cfg
    nts = int(args["NTIMESTEP"])
    dt = float(args["DT"])
    nrads, nradl = int(args["NRADS"]), int(args["NRADL"])
    lsswr = nts % nrads == 0
    lslwr = nts % nradl == 0
    dtsw, dtlw = nrads * dt, nradl * dt
    ntoz = int(cfg.ntoz)
    ivflip = int(getattr(cfg, "ivflip", IVFLIP_NAM))
    ctx.tables["rrtm_call_flags"] = {"lsswr": lsswr, "lslwr": lslwr, "ntimestep": nts,
                                     "solcon": float(args["SOLCON"]), "sw_run": bool(sw and lsswr)}

    def _s(name):
        if stamp is not None:
            stamp(name)

    _check_nday(ctx)

    from namgpu.state import fit_on as _fit_on  # noqa: PLC0415
    if _fit_on():
        # 32 GB fit: hand the pool's cached free blocks back before the radiation working set is allocated.
        # They are what the 47 steps since the last call left cached (dynamics and physics temporaries, in
        # their sizes); the chunk working set's large blocks cannot be cut from them, so without this they
        # sit reserved beside it at the radiation peak (333x333: 0.66 GiB reserved over the 2.88 GiB used).
        # The call already frees everything at its end; values do not depend on placement.
        import cupy as _cpf
        if ctx.stream is not None:
            ctx.stream.synchronize()
        _cpf.get_default_memory_pool().free_all_blocks()
    _s("driver-prep")
    driver_prep.run(ctx, ntimestep=nts, jdat=args["JDAT"], dt=dt, nrads=nrads, nradl=nradl, columns=False)
    prep = ctx.tables[driver_prep.PRE_KEY]

    # P_PHY (layer pressure) for this call, full domain, before the chunk loop: driver-call reads it
    # ahead of grrad-setup in each chunk (module_RA_RRTM.F90:486-487 sets it before the column work).
    # Same formula grrad-setup writes; without this the first call has no P_PHY and later calls read
    # the previous call's PD (run lead 0935Z: KeyError 'P_PHY' at step 0). This is the module_RA_RRTM.F90:486-487 prologue
    # on the whole domain, the same step phys-rrtm-driver-prep does per row.
    import cupy as _cp
    _g, _st = ctx.grid, ctx.st
    _lm, _jm, _im = int(_g.lm), int(_g.jm), int(_g.im)
    _ppy = _st.add("P_PHY", (_lm, _jm, _im))
    _ppy[...] = (_cp.asarray(_st["SGML2"]).reshape(_lm, 1, 1) * _st["PD"][None]
                 + _cp.asarray(_g.psgml1).reshape(_lm, 1, 1))

    jm = int(ctx.grid.jm)
    from namgpu.state import FIT_RRTM_ROWS, fit_on  # noqa: PLC0415
    _fit = fit_on()
    if not rows:
        # NAMGPU_RRTM_ROWS wins; otherwise 32 rows on the full path, 8 on the 32 GB fit (16 rows ran out of memory at the 5th radiation call)
        rows = ROWS if ("NAMGPU_RRTM_ROWS" in os.environ or not _fit) else FIT_RRTM_ROWS
    rows = min(int(rows), jm)
    if rows < rrtm_chunk.MIN_ROWS and rows < jm:
        raise SystemExit(f"NAMGPU_RRTM_ROWS {rows}: radiation runs in row chunks of at least {rrtm_chunk.MIN_ROWS} rows "
                         f"(a chunk owns its rows less one halo row each side); set 4 or more")
    # windows of `rows` rows decide the result; batches of nb of them decide the launch shape (see the docstring)
    nb = batch_windows(ctx, jm, _im, rows, _fit)
    plan = batches(jm, rows, nb)
    hrows = plan[0][1] - plan[0][0]
    geo = _device_geometry(ctx, jm, rows, nb, plan) if nb > 1 else None
    last = ctx.tables.get(PLAN_KEY)
    if last is None or (last["rows"], last["b"]) != (rows, nb):
        print(f"rrtm-call: radiation row windows of {rows} rows, {nb} per batch: {len(plan)} chunks of {hrows} rows "
              f"({'fit' if _fit else 'full'} path)", flush=True)
    ctx.tables[PLAN_KEY] = {"rows": rows, "b": nb, "chunks": len(plan), "chunk_rows": hrows}
    ctx.tables["rrtm_call_flags"].update(rrtm_rows=rows, rrtm_batch=nb, rrtm_chunks=len(plan))
    sw_on = bool(lsswr and sw)
    drows = day_rows(ctx) if sw_on else None
    if sw_on and ctx.tables.get(NDAY_CHECK_KEY) is None:
        ctx.tables[NDAY_CHECK_KEY] = _cp.zeros((1,), dtype=_cp.int32)
    if _fit:
        _pool = _cp.get_default_memory_pool()
        _used0 = int(_pool.used_bytes())
    cc = _chunk_ctx(ctx, hrows)
    if _fit and rrtm_chunk.workspace(cc) is None:
        # one g-point workspace shared in time by swrad's chunk scratch and the LW taug/fracs pair (see
        # rrtm_chunk.WS_KEY): max of the two, not their sum
        import numpy as _np
        from namgpu.physics.rrtm import lw_core as _lwc
        from namgpu.physics.rrtm import phys_rrtm_sw_swrad as _swr
        _nlay = _lm + _swr.LTP
        _n = hrows * _im
        _need = max(_lwc.pair_floats(_nlay, hrows, _im),
                    _swr.chunk_scratch_floats(_swr.sw_ncap(_n, _swr.sw_nc(_n, nb > 1)), _nlay, _nlay + 1))
        cc.tables[rrtm_chunk.WS_KEY] = _cp.empty(_need, dtype=_np.float32)
    _s("grrad-chunks")
    for k, (w0, w1, a, b, _wins) in enumerate(plan):
        cc.begin(w0, w1)
        if geo is not None:
            cc.tables[BATCH_KEY] = geo[k]
        if sw_on:
            cc.tables[NDAY_KEY] = int(drows[w0:w1].sum())
        driver_prep.run_columns(cc, prep)
        dayparts.run(cc)
        dc = driver_call.run(cc, ntoz=ntoz, nrads=nrads, nradl=nradl, dt=dt)
        # The GRRAD actual arguments (module_RA_RRTM.F90:1013-1046: XLON, XLAT, SLMSK, TSEA, ...) are the
        # driver-call outputs; the grrad units read them from the chunk state by those names.
        for _n in GRRAD_SFC_ARGS:
            cc.st.local[_n] = dc[_n]
        grrad_setup.run(cc, dtsw=dtsw, dtlw=dtlw, ivflip=ivflip, ntoz=ntoz)
        if ntoz <= 0:
            getozn.run(cc, ivflip=ivflip)
        coszmn.run(cc)
        getgases.run(cc, ico2flg=int(cfg.ico2flg), ivflip=ivflip)
        grrad_gases.run(cc, ivflip=ivflip)
        grrad_aerosols.run(cc, lsswr=lsswr, lslwr=lslwr)
        grrad_clouds.run(cc, dep=progcld8.dep)   # progcld2 (np3d 3) or progcld8 (np3d 8)
        if lsswr and sw:
            grrad_sw.run(cc)
        grrad_lw.run(cc, sw_done=True)
        driver_post.run(cc, ntimestep=nts)
        cc.st.writeback(a, b)
    if _fit:
        # what this call's chunk working set held (pool after the loop over the pool before it), per chunk row:
        # the next call's batch choice uses the larger of this and the model
        ctx.tables["rrtm_call:row_bytes"] = max(0, int(_pool.total_bytes()) - _used0) / max(1, hrows)
    if _fit and os.environ.get("NAMGPU_RRTM_KEEP_CHUNK", "") != "1":
        # 32 GB fit: the chunk context holds the whole RRTM working set (chunk copies, chunk-local
        # scratch and tables: swrad chunk_scratch alone is 2 x 0.9 GB). Held between radiation calls it sat
        # idle through 47 of every 48 steps and pushed the step peak past a 32 GB card. Drop it; the next
        # call rebuilds it exactly as the first call does (zero-filled, one shape).
        _cc = ctx.tables.pop(KEY_CHUNK, None)
        if _cc is not None:
            _cc.release()                    # its arrays back to the pool now, before devmem.release below
        # driver-prep's per-call P8W/P_PHY and record are rebuilt at the start of every call (its column
        # scratch is chunk-local and went with the chunk context); the small rrtm_prep:vec:* vectors stay
        for _k in [k for k in ctx.tables if k.startswith("rrtm_prep:") and not k.startswith("rrtm_prep:vec:")]:
            del ctx.tables[_k]
        # give the freed radiation working set back whole, so the next 47 steps and the next call start from
        # unsplit blocks (pool fragmentation: the 4th call OOMed at ROWS=16 with 27.7 GiB in the pool)
        from namgpu import devmem
        devmem.release(ctx.stream)
        _check_nday(ctx)                     # the stream was just synchronised by release
    _s("rrtm-done")
