"""Concurrent one-way nests on one or N cards: a parent and the nests on its grid, stepped together, every domain
row-slab decomposed over the same N ranks (one process per card), as NCEP's NAM ran its nests inside the live 12 km
parent (nest_mode: 1-way in every NAM configfile).

    python -m namgpu.nest.coupled --plan PLAN.json --case 2026-11-05_00z --case-dir CASE --source gfs \
        --hours 84 --nest-hours 60 --out OUT [--cards 8 --card-list 0,...,7]
    (the door: woof-nam nest --ncep-nest conusnest --concurrent --cards 8 ..., namgpu/nest/suite_door.py)

PLAN.json: {"domains": [{"name": "parent", "config": CFG, "init": DIR, "parent": null},
                        {"name": "conusnest", "config": CFG, "init": DIR, "parent": "parent"}, ...]}
Each init DIR is a saved initial state (python -m namgpu.init.slab_init save, run once per domain on one card); every
rank loads its own rows of it (slab_init.load), as `woof-nam run --cards N` does.

What NMMB does and this follows (nam.v4.2.11, 83df4e13c1; NMMB core, US government code), as namgpu/nest/twoway.py:
  per parent step: the parent integrates, then gives every child boundary values valid at the end of its step
  (PARENT_UPDATE_CHILD_PSFC / PARENT_UPDATE_CHILD_BNDRY + UPDATE_BC_TENDS: kernels/nest_bc.cu), and each child runs
  RATIO steps (module_NMM_INTEGRATE.F90 coupler phases 4 and 2). A child's own children are stepped the same way inside
  its steps. The nests stop at --nest-hours and the parent runs on alone to --hours (exnam_fcst.sh LENGTH=60 for every
  domain, then exnam_fcst_parentext.sh extends the parent to 84 h).
Several cards: the parent points a nest boundary reads lie on whichever rank owns those parent rows. WindowCoupler
gathers them (every rank packs the ones it owns, one allgather per parent step per nest, kernels/nest_gather.cu) into
a window of the parent that covers the nest's boundary frame, and runs the one-card boundary kernels on the window, so
each rank fills its own boundary rows with the same bits one card would. NMMB moves the same values by MPI between the
parent's and the child's task sets (PARENTS_SEND_CHILD_DATA / CHILDREN_RECV_PARENT_DATA; not ported).
Not here: two-way feedback on several cards (NCEP's nests were one-way; one card: namgpu/nest/twoway.py --two-way),
moving nests (the NAM nests did not move).

Output: <out>/<name>/wrfout_d01_* per domain (an ordinary run folder each), <out>/coupled-receipt.json (and
coupled-receipt.r<rank>.json): load, TDFI, ms per parent step with the nests' steps, the sampled split per stage
(parent, per nest: gather + boundary, steps), memory per card, non-finite census per domain on the owned rows.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

UNIT_BC, UNIT_GATHER = "nest_bc", "nest_gather"
BLOCK = 128
PKG_PARENT = str(Path(__file__).resolve().parents[2])


# ---------------------------------------------------------------------------------------------- the coupler

def tables(imp: int, jmpg: int, jmpl: int, imc: int, jmcg: int, jmcl: int, i_ps: int, j_ps: int, ratio: int,
           lnsh: int, lnsv: int, nranks: int, rank: int) -> dict:
    """WindowCoupler's host tables for one rank, numpy only (tests/test_coupled_tables.py checks them for any rank
    count): the parent window, the pack/scatter indices, and this rank's ring and strip points.

    imp/jmpg: parent IM and GLOBAL JM; jmpl: this rank's parent memory rows; imc/jmcg/jmcl likewise for the nest.
    Ring (every rank, global): ic, jc, iw - i0w, js - j0w, w, n. Strips (this rank): ic, jc (global), iw - i0w,
    js - j0w, w, side, offset in the LOCAL strip, lstr[4] (local), n."""
    from namgpu import slab as slabmod
    from namgpu.nest import twoway as tw
    ri, rj = tw.ring_points(imc, jmcg, max(lnsh, lnsv))
    r_iw, r_js, r_w = tw.interp_factors(i_ps, j_ps, ratio, ri, rj, False)
    hi, hj, hs, ho, _hl = tw.strip_points(imc, jmcg, lnsh, False)
    h_iw, h_js, h_w = tw.interp_factors(i_ps, j_ps, ratio, hi, hj, False)
    vi, vj, vs, vo, _vl = tw.strip_points(imc, jmcg, lnsv, True)
    v_iw, v_js, v_w = tw.interp_factors(i_ps, j_ps, ratio, vi, vj, True)
    for iw, js, extra, what in ((r_iw, r_js, 1, "boundary ring"), (h_iw, h_js, 1, "boundary H"),
                                (v_iw, v_js, 2, "boundary V")):
        if min(int(iw.min()), int(js.min())) < 0 or int(iw.max()) + extra >= imp or int(js.max()) + extra >= jmpg:
            raise ValueError(f"coupled nest: {what} interpolation leaves the parent ({iw.min()}..{iw.max()} of "
                             f"{imp}, {js.min()}..{js.max()} of {jmpg})")
    # the parent points the stencils touch (H: 2 x 2 from the SW point; V: 3 x 3, the PD 4-point means), their window
    pts = []
    for iw, js, n in ((r_iw, r_js, 2), (h_iw, h_js, 2), (v_iw, v_js, 3)):
        for da in range(n):
            for db in range(n):
                pts.append((js.astype(np.int64) + db) * imp + iw.astype(np.int64) + da)
    flat = np.unique(np.concatenate(pts))
    pi, pj = (flat % imp).astype(np.int64), (flat // imp).astype(np.int64)
    i0w, j0w = int(pi.min()), int(pj.min())
    iww, jww = int(pi.max()) - i0w + 1, int(pj.max()) - j0w + 1
    rows = slabmod.split_rows(jmpg, nranks)                      # 1-based (jts, jte) per rank
    owner = np.searchsorted(np.array([b for _a, b in rows]), pj + 1, side="left")
    counts = np.bincount(owner, minlength=nranks)
    maxn = max(1, int(counts.max()))
    joff_p = max(rows[rank][0] - slabmod.JHALO, 1) - 1 if nranks > 1 else 0
    wpos = np.full((nranks, maxn), -1, np.int64)
    idx = np.zeros(maxn, np.int64)
    for r in range(nranks):
        m = owner == r
        wpos[r, :int(m.sum())] = (pj[m] - j0w) * iww + (pi[m] - i0w)
        if r == rank and m.any():
            loc = (pj[m] - joff_p) * imp + pi[m]
            if loc.min() < 0 or loc.max() >= jmpl * imp:
                raise AssertionError("coupled nest: an owned parent point is outside this rank's memory rows")
            idx[:len(loc)] = loc
            idx[len(loc):] = loc[0]
    # this rank's strip points (global nest i, j for the factors and PDB; offsets in the LOCAL strips)
    if nranks > 1:
        cr = slabmod.split_rows(jmcg, nranks)[rank]
        jms, jme = max(cr[0] - slabmod.JHALO, 1), min(cr[1] + slabmod.JHALO, jmcg)
        joff_c, s_bdy, n_bdy = jms - 1, jms == 1, jme == jmcg
    else:
        joff_c, s_bdy, n_bdy = 0, True, True

    def local(i, j, sd, off, iw, js, w, width):
        keep = (((sd == 0) & s_bdy) | ((sd == 1) & n_bdy) | ((sd >= 2) & (j >= joff_c) & (j < joff_c + jmcl)))
        off = off.astype(np.int64).copy()
        we = sd >= 2
        off[we] = (j[we] - joff_c) * width + off[we] % width
        lstr = np.array([width * imc, width * imc, jmcl * width, jmcl * width], np.int64)
        k = keep
        return [i[k].astype(np.int32), j[k].astype(np.int32), (iw[k] - i0w).astype(np.int32),
                (js[k] - j0w).astype(np.int32), w[k], sd[k].astype(np.int32), off[k], lstr, int(k.sum())]
    return {"i0w": i0w, "j0w": j0w, "iww": iww, "jww": jww, "maxn": maxn, "npts": len(flat),
            "counts": [int(x) for x in counts], "idx": idx, "wpos": wpos, "joff_c": joff_c,
            "ring": [ri, rj, (r_iw - i0w).astype(np.int32), (r_js - j0w).astype(np.int32), r_w, len(ri)],
            "h": local(hi, hj, hs, ho, h_iw, h_js, h_w, lnsh), "v": local(vi, vj, vs, vo, v_iw, v_js, v_w, lnsv)}


class WindowCoupler:
    """Parent -> nest one-way boundary for a parent and a nest decomposed over the same ranks (or one card).

    Host tables (identical on every rank, built once from the GLOBAL grids): the nest's boundary points
    (twoway.ring_points / strip_points), their bilinear factors on the parent (twoway.interp_factors), the set of
    parent points those stencils touch (H: 2 x 2 from the SW point; V: 3 x 3, the PD 4-point means), the bounding
    window of that set, and which rank owns each point (slab.split_rows of the parent). Per parent step: pack the owned
    points (nest_gather_pack), allgather (one card: none), scatter into the window (nest_gather_scatter), then the
    nest_bc kernels on the window (ring PSFC on every rank for the whole ring, H/V strips for this rank's own strip
    rows: S on the south rank, N on the north rank, W/E for the local memory rows)."""

    def __init__(self, ctx_p, ctx_c, ratio: int, i_ps: int, j_ps: int):
        import cupy as cp
        from namgpu import kernels, parallel, slab as slabmod
        from namgpu.nest import twoway as tw
        self.cp, self.K = cp, kernels
        self.p, self.c = ctx_p, ctx_c
        self.ratio, self.i_ps, self.j_ps = int(ratio), int(i_ps), int(j_ps)
        sp, sc = slabmod.of(ctx_p), slabmod.of(ctx_c)
        self.comm = parallel.get(ctx_p)
        if (sp is None) != (sc is None) or (sp is not None and sp.nranks != sc.nranks):
            raise ValueError("coupled: parent and nest must run on the same ranks")
        self.nranks = sp.nranks if sp is not None else 1
        self.rank = sp.rank if sp is not None else 0
        gp, gc = ctx_p.grid, ctx_c.grid
        self.imp, self.jmpg = int(gp.im), (sp.jmg if sp is not None else int(gp.jm))
        self.imc, self.jmcg = int(gc.im), (sc.jmg if sc is not None else int(gc.jm))
        self.jmpl, self.jmcl = int(gp.jm), int(gc.jm)
        self.lm = int(gp.lm)
        if int(gc.lm) != self.lm:
            raise ValueError(f"coupled nest: parent LM {self.lm} != nest LM {gc.lm} (NMMB nests share the levels)")
        if self.lm > 64:
            raise ValueError(f"coupled nest: LM {self.lm} > NEST_LMAX 64 (kernels/nest_spline.cuh)")
        stp, stc = ctx_p.st, ctx_c.st
        for k in ("SG1", "SG2", "PSGML1", "SGML2"):
            a, b = cp.asnumpy(stp[k]), cp.asnumpy(stc[k])
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-6):
                raise ValueError(f"coupled nest: parent and nest {k} differ (NMMB nests share the vertical coordinate)")
        if abs(float(gp.pt) - float(gc.pt)) > 1e-3 or abs(float(gp.pdtop) - float(gc.pdtop)) > 1e-2:
            raise ValueError(f"coupled nest: parent PT/PDTOP {gp.pt}/{gp.pdtop} != nest {gc.pt}/{gc.pdtop}")
        self.pdtop, self.pt = np.float32(gp.pdtop), np.float32(gp.pt)
        lnsh, lnsv = int(ctx_c.cfg.lnsh), int(ctx_c.cfg.lnsv)
        self.recip = np.float32(1.0) / np.float32(np.float32(ctx_c.cfg.dt) * np.float32(self.ratio))

        t = tables(self.imp, self.jmpg, self.jmpl, self.imc, self.jmcg, self.jmcl, i_ps, j_ps, ratio, lnsh, lnsv,
                   self.nranks, self.rank)
        self.t = t
        if sc is not None and t["joff_c"] != sc.joff:
            raise AssertionError(f"coupled nest: tables put the nest's rank {self.rank} rows at {t['joff_c']}, the slab "
                                 f"at {sc.joff}")
        self.i0w, self.j0w, self.iww, self.jww = t["i0w"], t["j0w"], t["iww"], t["jww"]
        self.maxn, self.npts, self.counts = t["maxn"], t["npts"], t["counts"]
        self.nv = 2 + 5 * self.lm

        def dev(x):
            return cp.asarray(np.ascontiguousarray(x))
        self.idx, self.wpos = dev(t["idx"]), dev(t["wpos"])
        self.pack = cp.zeros((self.nv, self.maxn), cp.float32)
        self.recv = cp.zeros((self.nranks, self.nv, self.maxn), cp.float32) if self.nranks > 1 else None
        wp = self.iww * self.jww
        self.wplane = wp
        self.w2 = {k: cp.zeros(wp, cp.float32) for k in ("PD", "FIS")}
        self.w3 = {k: cp.zeros(self.lm * wp, cp.float32) for k in ("T", "Q", "CW", "U", "V")}
        self.ring = [dev(x) for x in t["ring"][:-1]] + [t["ring"][-1]]
        self.h = [dev(x) for x in t["h"][:-1]] + [t["h"][-1]]
        self.v = [dev(x) for x in t["v"][:-1]] + [t["v"][-1]]
        self.pdb = cp.zeros((self.jmcg, self.imc), cp.float32)          # global rows: the ring is global
        self.fis_c = self._global_rows(ctx_c, stc["FIS"])              # the nest terrain, global rows (once)
        self._bind_strips()

    def _global_rows(self, ctx, a):
        """A [JM_local, IM] field of `ctx` as the [JMG, IM] global array on every rank (one allgather, at setup)."""
        cp = self.cp
        from namgpu import slab as slabmod
        s = slabmod.of(ctx)
        if s is None:
            return cp.ascontiguousarray(a, dtype=cp.float32)
        rows = slabmod.split_rows(s.jmg, s.nranks)
        mr = max(b - a_ + 1 for a_, b in rows)
        im = int(a.shape[-1])
        with ctx.stream:
            send = cp.zeros((mr, im), cp.float32)
            send[:s.nown] = a[s.own0:s.own0 + s.nown]
            recv = cp.zeros((s.nranks, mr, im), cp.float32)
            self.comm.allgather(send, recv, ctx.stream)
            out = cp.ascontiguousarray(cp.concatenate([recv[r, :b - a_ + 1] for r, (a_, b) in enumerate(rows)]))
        return out

    def _bind_strips(self):
        cp, bc = self.cp, self.c.bc
        from namgpu.nest.twoway import H_BC, V_BC
        sides = "SNWE"
        for name in H_BC + V_BC:
            for s in sides:
                a = bc[f"{s}_{name}"]
                if a.dtype != cp.float32 or not a.flags.c_contiguous:
                    raise ValueError(f"coupled nest: ctx.bc {s}_{name} is not a dense float32 strip")
        self._bc_id = {k: id(v) for k, v in bc.items()}
        self.hptr = cp.asarray(np.array([bc[f"{s}_{n}"].data.ptr for n in H_BC for s in sides], np.uint64))
        self.vptr = cp.asarray(np.array([bc[f"{s}_{n}"].data.ptr for n in V_BC for s in sides], np.uint64))

    def gather(self):
        """The parent's values on the window points, from the ranks that own them (end of the parent step)."""
        K, p, i32, i64 = self.K, self.p.st, np.int32, np.int64
        gx = (self.maxn + BLOCK - 1) // BLOCK
        K.get(UNIT_GATHER, "nest_gather_pack")((gx, self.nv), (BLOCK,), (
            p["PD"], p["FIS"], p["T"], p["Q"], p["CW"], p["U"], p["V"], i64(self.imp * self.jmpl), i32(self.lm),
            i32(self.maxn), self.idx, self.pack))
        recv = self.pack
        if self.recv is not None:
            self.comm.allgather(self.pack, self.recv, self.p.stream)
            recv = self.recv
        w2, w3 = self.w2, self.w3
        K.get(UNIT_GATHER, "nest_gather_scatter")((gx, self.nv, self.nranks), (BLOCK,), (
            recv, self.wpos, i32(self.nranks), i32(self.nv), i32(self.maxn), i32(self.lm), i64(self.wplane),
            w2["PD"], w2["FIS"], w3["T"], w3["Q"], w3["CW"], w3["U"], w3["V"]))

    def boundary(self):
        """Parent state at the end of its step -> nest boundary tendencies over the next RATIO nest steps."""
        K, p, c = self.K, self.p.st, self.c.st
        if {k: id(v) for k, v in self.c.bc.items()} != self._bc_id:
            self._bind_strips()
        self.gather()
        f32, i32 = np.float32, np.int32
        w2, w3 = self.w2, self.w3
        iww, jww = i32(self.iww), i32(self.jww)
        ic, jc, iw, js, w, n = self.ring
        K.get(UNIT_BC, "nest_bc_psfc")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
            w2["PD"], w2["FIS"], w3["T"], w3["Q"], iww, jww, i32(self.lm),
            p["SG1"], p["SG2"], f32(self.pdtop), f32(self.pt), self.fis_c, i32(self.imc),
            i32(n), ic, jc, iw, js, w, self.pdb))
        ic, jc, iw, js, w, sd, off, ls, n = self.h
        if n:
            K.get(UNIT_BC, "nest_bc_h")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
                w2["PD"], w3["T"], w3["Q"], w3["CW"], iww, jww, i32(self.lm),
                p["SG1"], p["SG2"], c["PSGML1"], c["SGML2"], f32(self.pdtop), f32(self.pt),
                self.pdb, i32(self.imc), i32(n), ic, jc, iw, js, w, sd, off, self.hptr, ls, f32(self.recip)))
        ic, jc, iw, js, w, sd, off, ls, n = self.v
        if n:
            K.get(UNIT_BC, "nest_bc_v")(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
                w2["PD"], w3["U"], w3["V"], iww, jww, i32(self.lm),
                p["SG1"], p["SG2"], c["PSGML1"], c["SGML2"], f32(self.pdtop), f32(self.pt),
                self.pdb, i32(self.imc), i32(n), ic, jc, iw, js, w, sd, off, self.vptr, ls, f32(self.recip)))

    def receipt(self) -> dict:
        return {"ratio": self.ratio, "i_parent_start": self.i_ps, "j_parent_start": self.j_ps, "two_way": False,
                "boundary_points": {"psfc_ring": self.ring[-1], "h_local": self.h[-1], "v_local": self.v[-1]},
                "window": {"i0": self.i0w, "j0": self.j0w, "iw": self.iww, "jw": self.jww,
                           "mib": round((2 + 5 * self.lm) * self.wplane * 4 / 2**20, 1)},
                "gather": {"points": self.npts, "per_rank": self.counts, "ranks": self.nranks,
                           "mib_per_step": round(self.nranks * self.nv * self.maxn * 4 / 2**20, 2)}}


# ---------------------------------------------------------------------------------------------- one rank

def _dt(cfg) -> Fraction:
    r = cfg.raw
    return Fraction(int(r["dt_int"])) + (Fraction(int(r["dt_num"]), int(r["dt_den"])) if int(r["dt_den"]) else 0)


class Domain:
    def __init__(self, spec: dict):
        self.name, self.config, self.init = spec["name"], Path(spec["config"]), Path(spec["init"])
        self.parent_name = spec.get("parent")
        self.children: list = []
        self.ctx = self.drv = self.cpl = None
        self.ratio = 1


def _build(dom: Domain, stream, comm, rank: int, nranks: int, a, start, shm):
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import slab_init
    from namgpu.slab import Slab, local_grid
    from namgpu.steps import drv_loop_limits
    os.environ[config.CONFIG_ENV] = str(dom.config)    # State() and the loaders read the default configfile
    cfg = config.load(dom.config)
    config.start_filter(cfg, a.source)
    g = gridmod.build(cfg)
    slab = geo = None
    if nranks > 1:
        jmg, im = int(g.jm), int(g.im)
        cen = (jmg // 2, im // 2)
        geo = {"xlat": np.degrees(np.asarray(g.glat_h)).astype(np.float32),
               "xlon": np.degrees(np.asarray(g.glon_h)).astype(np.float32),
               "cen_lat": float(np.degrees(g.glat_h[cen])), "cen_lon": float(np.degrees(g.glon_h[cen])),
               "dx": float(np.asarray(g.dxh)[jmg // 2])}
        slab = Slab.for_rank(rank, nranks, jmg)
        g = local_grid(g, slab)
    st = statemod.State(g, names={k for k, v in statemod.all_specs().items() if v.rank == 0})
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream)

    def attach():
        if slab is not None:
            ctx.slab, ctx.comm, ctx.hist_geo_global = slab, comm, geo
            ctx.decomp = (rank, 1, nranks)
        else:
            ctx.slab = ctx.comm = ctx.decomp = None
    attach()
    load = slab_init.load(ctx, dom.init)
    attach()                                            # the saved attributes are the init process's
    ctx.case_dir = a.case_dir.resolve()
    ctx.start = ctx.start_time = start
    ctx.init_source = a.source
    run_dir = a.out / dom.name
    ctx.out_dir = run_dir if nranks == 1 else run_dir / f"rank{rank}"
    Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
    ctx.hist_pin_key = dom.name
    if shm is not None:
        ctx.hist_shm = str(shm / dom.name)
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    with stream:
        drv_loop_limits.run(ctx)
    dom.ctx, dom.drv = ctx, drv
    return load


def _census(ctx) -> dict:
    import cupy as cp
    from namgpu import slab as slabmod
    out = {}
    for k in ("T", "Q", "CW", "U", "V", "PD"):
        a = ctx.st[k]
        if slabmod.of(ctx) is not None:
            s = ctx.slab
            a = a[..., s.own0:s.own0 + s.nown, :]
        out[k] = int((~cp.isfinite(a)).sum())
    return out


def rank_main(a) -> int:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft != resource.RLIM_INFINITY and soft < want:      # several saved inits are memory-mapped in one process
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    rank, nranks = int(os.environ.get("NAMGPU_RANK", "0")), int(os.environ.get("NAMGPU_NRANKS", "1"))
    if nranks > 1:
        import faulthandler
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    import cupy as cp
    from namgpu import config, devmem, parallel
    from namgpu import restart as rstmod
    from namgpu.boundary import bnd_boundary_init
    from namgpu.io import history_select, io_history
    from namgpu.state import release_init_leftovers
    from namgpu.steps import drv_dfi_tdfi_driver, drv_timing_receipt as timing
    from namgpu.steps.drv_physics_initialize import in_run_dir
    from namgpu.steps.drv_time_loop import release_tdfi

    plan = json.loads(Path(a.plan).read_text())
    doms = {d["name"]: Domain(d) for d in plan["domains"]}
    roots = [d for d in doms.values() if d.parent_name is None]
    if len(roots) != 1:
        raise SystemExit(f"coupled: the plan needs exactly one parent domain (got {[d.name for d in roots]})")
    root = roots[0]
    for d in doms.values():
        if d.parent_name is not None:
            doms[d.parent_name].children.append(d)
    start = datetime.strptime(Path(a.case).name[:14], "%Y-%m-%d_%Hz")
    a.out = a.out.resolve()
    shm = Path(os.environ["NAMGPU_HIST_SHM"]) if nranks > 1 else None
    comm = parallel.make_comm(rank, nranks, os.environ["NAMGPU_NCCL_ID"]) if nranks > 1 else None
    stream = cp.cuda.Stream(non_blocking=True)            # one stream: parent, coupling and nests run in order
    rec = {"door": "woof-nam nest --concurrent (namgpu/nest/coupled.py)", "case": a.case, "source": a.source,
           "hours": a.hours, "nest_hours": a.nest_hours, "rank": rank, "nranks": nranks,
           "comm": os.environ.get("NAMGPU_COMM", "nccl") if nranks > 1 else None,
           "card": os.environ.get("CUDA_VISIBLE_DEVICES"), "start": start.isoformat(), "error": None}
    t0 = time.perf_counter()
    order = []

    def walk(d):
        order.append(d)
        for c in d.children:
            walk(c)
    walk(root)
    rec["load"] = {}
    for d in order:
        tl = time.perf_counter()
        _build(d, stream, comm, rank, nranks, a, start, shm)
        rec["load"][d.name] = round(time.perf_counter() - tl, 2)
        if d.ctx.slab is not None:
            s = d.ctx.slab
            print(f"rank {rank}/{nranks} {d.name}: rows {s.jts}..{s.jte} of {s.jmg}, memory {s.jms}..{s.jme}",
                  flush=True)
    for d in order[1:]:
        q = _dt(doms[d.parent_name].ctx.cfg) / _dt(d.ctx.cfg)
        if q.denominator != 1:
            raise SystemExit(f"coupled: {d.parent_name} dt is not a whole number of {d.name} steps")
        d.ratio = int(q)
    stream.synchronize()
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    hf = a.history_fields or history_select.DEFAULT_PRESET
    fe = history_select.DEFAULT_FULL_EVERY if a.full_every is None else a.full_every
    for d in order:
        d.ctx.history_select = history_select.Selection(hf, rstmod.steps_for(rstmod.parse_hours(fe), d.ctx.cfg.dt))
        d.ctx.restart_writer = None
    rec["grids"] = {d.name: [int(d.ctx.cfg.im), int(d.ctx.cfg.jm), int(d.ctx.cfg.lm)] for d in order}
    rec["dt"] = {d.name: float(d.ctx.cfg.dt) for d in order}
    rec["ratio"] = {d.name: d.ratio for d in order[1:]}
    print("coupled: " + ", ".join(f"{d.name} {rec['grids'][d.name]} dt {rec['dt'][d.name]:g} s" for d in order)
          + f"; {nranks} rank{'s' if nranks > 1 else ''}, one-way", flush=True)

    dtp = float(root.ctx.cfg.dt)
    nsteps = int(round(a.hours * 3600.0 / dtp))
    n_nest = int(round(min(a.nest_hours, a.hours) * 3600.0 / dtp))
    sph = max(1, int(round(3600.0 / dtp)))
    hours_log = []
    acc = {"parent": 0.0}
    sampled, pending, pend_parent = 0, False, False

    def advance(d):
        """One step of d, its history, then each child's boundary and RATIO child steps (recursively)."""
        d.drv.step()
        if d.drv._due("history"):
            d.drv.write_history()
        for c in d.children:
            c.cpl.boundary()
            for _ in range(c.ratio):
                advance(c)

    with in_run_dir(root.ctx), stream:
        try:
            for d in order:
                release_init_leftovers(d.ctx)
                if int(d.ctx.cfg.filter_method) > 0:
                    os.environ[config.CONFIG_ENV] = str(d.config)
                    tf = time.perf_counter()
                    d.ctx.filter_active = True
                    try:
                        drv_dfi_tdfi_driver.run(d.ctx)
                    finally:
                        d.ctx.filter_active = False
                    release_tdfi(d.ctx)
                    d.ctx.ntimestep, d.ctx.first_step = 0, True
                    stream.synchronize()
                    rec[f"tdfi_{d.name}_s"] = round(time.perf_counter() - tf, 2)
                    print(f"tdfi {d.name} done {rec[f'tdfi_{d.name}_s']} s", flush=True)
            for d in order:
                d.ctx.dt = d.ctx.cfg.dt
                d.ctx.start_time = d.ctx.start
            rec["coupler"] = {}
            for d in order[1:]:
                c = d.ctx
                c.concurrent_parent = True          # boundaries from the live parent, never from boco files
                c.i_am_a_nest = True
                c.parent_child_time_ratio = d.ratio
                bnd_boundary_init.run(c)              # the carried values start from the nest's own (filtered) edges
                d.cpl = WindowCoupler(doms[d.parent_name].ctx, c, d.ratio, int(c.cfg.raw["i_parent_start"]),
                                      int(c.cfg.raw["j_parent_start"]))
                rec["coupler"][d.name] = d.cpl.receipt()
            for c in root.children:
                acc[f"{c.name}_boundary"] = 0.0
                acc[f"{c.name}_steps_x{c.ratio}"] = 0.0
            for d in order:
                d.drv._initial_history()
            ev_p = [cp.cuda.Event() for _ in range(2)]
            ev_c = {c.name: [cp.cuda.Event() for _ in range(3)] for c in root.children}
            t1 = time.perf_counter()
            for n in range(1, nsteps + 1):
                nests_on = n <= n_nest
                timed = n % sph == sph // 2                  # one mid-hour parent step per hour (no history in it)
                if timed:
                    ev_p[0].record(stream)
                root.drv.step()
                if timed:
                    ev_p[1].record(stream)
                if root.drv._due("history"):
                    root.drv.write_history()
                if nests_on:
                    for c in root.children:
                        if timed:
                            ev_c[c.name][0].record(stream)
                        c.cpl.boundary()
                        if timed:
                            ev_c[c.name][1].record(stream)
                        for _ in range(c.ratio):
                            advance(c)
                        if timed:
                            ev_c[c.name][2].record(stream)
                if timed:
                    pending = nests_on
                    pend_parent = True
                if n == n_nest and n < nsteps:
                    stream.synchronize()
                    for c in order[1:]:
                        io_history.drain(c.ctx)
                    rec["nests_done_wall_s"] = round(time.perf_counter() - t1, 2)
                    print(f"nests done at parent step {n} ({n * dtp / 3600.0:g} h); the parent runs on alone",
                          flush=True)
                if n % sph == 0:
                    stream.synchronize()                      # once per forecast hour: the only host sync
                    for d in order:
                        timing.drain(d.ctx)
                    devmem.release(stream)
                    if pend_parent:
                        pend_parent = False
                        sampled += 1
                        acc["parent"] += cp.cuda.get_elapsed_time(ev_p[0], ev_p[1]) / 1e3
                        if pending:
                            for c in root.children:
                                ev = ev_c[c.name]
                                acc[f"{c.name}_boundary"] += cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1e3
                                acc[f"{c.name}_steps_x{c.ratio}"] += cp.cuda.get_elapsed_time(ev[1], ev[2]) / 1e3
                        pending = False
                    wall = time.perf_counter() - t1
                    line = (f"fh {n * dtp / 3600.0:6.2f}  parent step {n}  "
                            + "  ".join(f"{d.name} step {d.ctx.ntimestep}" for d in order[1:])
                            + f"  wall {wall:8.1f} s  {wall / n * 1e3:7.1f} ms per parent step"
                            + ("" if nests_on else " (parent alone)"))
                    hours_log.append(line)
                    print(line, flush=True)
            stream.synchronize()
            wall = time.perf_counter() - t1
            rec["run_s"] = round(wall, 2)
            rec["steps"] = {d.name: int(d.ctx.ntimestep) for d in order}
            rec["ms_per_parent_step"] = round(wall / max(1, nsteps) * 1e3, 1)
            if "nests_done_wall_s" in rec:
                rec["ms_per_parent_step_with_nests"] = round(rec["nests_done_wall_s"] / max(1, n_nest) * 1e3, 1)
                rec["ms_per_parent_step_alone"] = round((wall - rec["nests_done_wall_s"]) / max(1, nsteps - n_nest)
                                                        * 1e3, 1)
            rec["sampled_ms_per_parent_step"] = {k: round(v / max(1, sampled) * 1e3, 2) for k, v in acc.items()}
            rec["sampled_steps"] = sampled
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-3000:]
            print("RUN ERROR", rec["error"], flush=True)
            print(rec["trace"], flush=True)
    stream.synchronize()
    for d in order:
        try:
            io_history.drain(d.ctx)
            timing.write_receipt(d.ctx, Path(d.ctx.out_dir))
        except Exception as e:  # noqa: BLE001
            rec.setdefault("receipt_errors", []).append(f"{d.name}: {type(e).__name__}: {e}")
    rec["hour_lines"] = hours_log
    rec["nonfinite_end"] = {d.name: _census(d.ctx) for d in order}
    rec["pool_end_gib"] = round(cp.get_default_memory_pool().total_bytes() / 2**30, 2)
    rec["card_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    if nranks == 1:
        _domain_receipts(a.out, plan, [rec], 0 if not rec["error"] else 1)
    rp = a.out / ("coupled-receipt.json" if nranks == 1 else f"coupled-receipt.r{rank}.json")
    rp.write_text(json.dumps(rec, indent=1, default=str))
    bad = any(any(v.values()) for v in rec["nonfinite_end"].values())
    print(f"coupled receipt {rp}: {rec.get('ms_per_parent_step')} ms per parent step, non-finite "
          f"{rec['nonfinite_end']}", flush=True)
    return 1 if (rec["error"] or bad) else 0


# ---------------------------------------------------------------------------------------------- the launcher

def _domain_receipts(out: Path, plan: dict, ranks: list, code: int) -> None:
    """<out>/<domain>/forecast-receipt.json: the run folder's own receipt (the maps tail, `woof-nam maps` and the
    column store read it), with this domain's numbers from every rank."""
    for d in plan["domains"]:
        nm = d["name"]
        r = {"domain": nm, "coupled": True, "configfile": d["config"], "parent_domain": d.get("parent"),
             "exit_code": code, "grid": (ranks[0] or {}).get("grids", {}).get(nm) if ranks else None,
             "dt": (ranks[0] or {}).get("dt", {}).get(nm) if ranks else None,
             "steps_done": (ranks[0] or {}).get("steps", {}).get(nm) if ranks else None,
             "nonfinite_end_by_rank": [((x or {}).get("nonfinite_end") or {}).get(nm) for x in ranks],
             "receipt": "../coupled-receipt.json"}
        (out / nm).mkdir(parents=True, exist_ok=True)
        (out / nm / "forecast-receipt.json").write_text(json.dumps(r, indent=1, default=str))


def _py_module(mod: str) -> list:
    return [sys.executable, "-c",
            "import sys; sys.path.insert(0, %r); from %s import main; sys.exit(main(sys.argv[1:]))" % (PKG_PARENT, mod)]


def launch(a, argv) -> int:
    """--cards N: one process per card (CUDA_VISIBLE_DEVICES pinned), each stepping every domain on its rows, plus one
    history writer pool per domain (namgpu.io.history_merge). If one rank fails the others are stopped."""
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cards = [c.strip() for c in a.card_list.split(",") if c.strip()] or [str(i) for i in range(a.cards)]
    if len(cards) < a.cards:
        raise SystemExit(f"--cards {a.cards} but --card-list has {len(cards)} cards: {cards}")
    cards = cards[:a.cards]
    if len(set(cards)) < len(cards) and os.environ.get("NAMGPU_COMM", "nccl") != "host":
        raise SystemExit(f"--card-list {cards}: a card twice needs NAMGPU_COMM=host (NCCL refuses two ranks on one "
                         "device)")
    plan = json.loads(Path(a.plan).read_text())
    t0 = time.perf_counter()
    idfile = out / "nccl.id"
    idfile.unlink(missing_ok=True)
    shutil.rmtree(str(idfile) + ".hostcomm", ignore_errors=True)
    shm = Path("/dev/shm") / f"namgpu-coupled-{os.getpid()}"
    from namgpu.io import scratch as _scr
    _scr.own(shm.parent)               # this process is alive for a sweep in any PID namespace (scratch.own)
    shutil.rmtree(shm, ignore_errors=True)
    shm.mkdir(parents=True)
    writers = []
    for d in plan["domains"]:
        (shm / d["name"]).mkdir()
        (out / d["name"]).mkdir(parents=True, exist_ok=True)
        nw = a.hist_workers if d.get("parent") else max(2, a.hist_workers // 2)
        writers.append(subprocess.Popen(_py_module("namgpu.io.history_merge") + [
            "--shm", str(shm / d["name"]), "--out", str(out / d["name"]), "--nranks", str(len(cards)),
            "--workers", str(nw)]))
    procs = []
    for r, card in enumerate(cards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card, NAMGPU_RANK=str(r), NAMGPU_NRANKS=str(len(cards)),
                   NAMGPU_NCCL_ID=str(idfile), NAMGPU_HIST_SHM=str(shm), PYTHONUNBUFFERED="1")
        logf = open(out / f"rank{r}.log", "w")
        procs.append((subprocess.Popen(_py_module("namgpu.nest.coupled") + list(argv), env=env, stdout=logf,
                                       stderr=subprocess.STDOUT), logf))
    print(f"coupled: launched {len(cards)} ranks on cards {cards}; logs {out}/rank*.log", flush=True)
    codes = [None] * len(procs)
    failed = False
    while any(c is None for c in codes):
        for i, (p, _) in enumerate(procs):
            if codes[i] is None and p.poll() is not None:
                codes[i] = p.returncode
                if p.returncode != 0 and not failed:
                    failed = True
                    print(f"rank {i} exited {p.returncode}: stopping the other ranks", flush=True)
                    for q, _ in procs:
                        if q.poll() is None:
                            q.send_signal(signal.SIGTERM)
        time.sleep(0.5)
    for _, f in procs:
        f.close()
    for d in plan["domains"]:
        (shm / d["name"] / "END").write_text("1")
    wcodes = [w.wait() for w in writers]
    shutil.rmtree(shm, ignore_errors=True)
    shutil.rmtree(str(idfile) + ".hostcomm", ignore_errors=True)
    ranks = [json.loads(p.read_text()) if p.exists() else None
             for p in (out / f"coupled-receipt.r{r}.json" for r in range(len(cards)))]
    rec = {"cards": cards, "rank_exit": codes, "history_pool_exit": wcodes, "wall_s": round(time.perf_counter() - t0, 1),
           "comm": os.environ.get("NAMGPU_COMM", "nccl"),
           "ms_per_parent_step_by_rank": [(r or {}).get("ms_per_parent_step") for r in ranks],
           "card_used_gib_by_rank": [(r or {}).get("card_used_gib") for r in ranks],
           "nonfinite_end_by_rank": [(r or {}).get("nonfinite_end") for r in ranks], "ranks": ranks}
    ok = all(c == 0 for c in codes) and not any(wcodes)
    rec["exit_code"] = 0 if ok else 1
    (out / "coupled-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    _domain_receipts(out, plan, ranks, rec["exit_code"])
    print(f"coupled: ranks exit {codes}, history pools {wcodes}, wall {rec['wall_s']} s -> "
          f"{out / 'coupled-receipt.json'}", flush=True)
    return rec["exit_code"]


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m namgpu.nest.coupled", description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--source", default="gfs")
    ap.add_argument("--hours", type=float, required=True, help="the parent's length")
    ap.add_argument("--nest-hours", type=float, default=None, help="the nests' length (default --hours; NCEP: 60 "
                    "for every nest, the parent extended alone to 84)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cards", type=int, default=1)
    ap.add_argument("--card-list", default=os.environ.get("NAM_CARDS", ""))
    ap.add_argument("--hist-workers", type=int, default=4, help="history writer processes per nest (the parent "
                    "gets half), several cards only")
    ap.add_argument("--history-fields", default=None)
    ap.add_argument("--full-every", default=None)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.nest_hours is None:
        a.nest_hours = a.hours
    if a.cards > 1 and "NAMGPU_RANK" not in os.environ:
        return launch(a, argv)
    return rank_main(a)


if __name__ == "__main__":
    _rc = main()
    from namgpu.io import io_history as _ioh
    _ioh.shutdown()             # writer processes hold stdout; os._exit skips their exit hooks
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_rc or 0))
