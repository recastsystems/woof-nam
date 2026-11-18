"""A storm-following nest inside the live parent, in one process: woof-nam nest --concurrent --move-nest.

The nest re-centres on the storm while the 12 km parent runs, by whole parent cells, with no restart, no files and no
second process. It is the in-memory half of ArWen's moving nest (core/nest_relocation.py); the stand-alone door
(woof-nam nest --follow, namgpu/nest/follow*.py) moves by restart sets and fresh starts at the source's 3-hourly times.

The grid. A moving nest is a series of windows of ONE rotated lat-lon grid, the first nest's (the parent's pole, the
nest spacing): window (i0, j0) is the first nest shifted by i0 columns and j0 rows of nest cells (config window keys,
namgpu/nest/move.py). A move is a whole number of parent cells, so i0 and j0 step by the space ratio and the window's
SW corner stays on a parent H point (i_parent_start += di_p). A ground point two windows share has the same bits in
both (the geometry kernels take the reference origin plus the offset), so the kept cells of a move are an index copy.

The envelope. The fine static (terrain, land use, soil type, vegetation, albedo, roughness) and the slow surface state
(soil temperature and moisture, skin and sea-surface temperature, sea ice, snow) for every cell the nest can reach are
built ONCE before the forecast, as one ordinary saved start (namgpu.init.slab_init save) of a big window of the same
grid: the "envelope". It stays on the host, memory-mapped (normally /dev/shm). A move cuts the new window out of it
(load_window: rows and columns of every owner array, one upload per owner) instead of building anything: that is what
makes a move quick, and it is the only host-to-device traffic of a move. The envelope's surface state is the source's
at the start time, on the fine land mask; the new ground's atmosphere does not come from it (below).

A move (LiveMover.move), between two parent steps, on the parent's stream:
  1. the new window's state = the envelope cut (static, surface, geometry, tables), as a fresh start of the window;
     init-grid-consts is run again for the window (its hdac* boundary-block constants depend on where its edges are);
  2. the kept cells: every (.., JM, IM) array of the old nest is laid over it shifted by (dj, di) nest cells, device to
     device, by the restart overlay's own rules (move.overlay: grid arrays carried on the overlap, geometry, per-row
     tables and window constants the window's own, lazily made caches dropped and rebuilt by their units, counters
     and the clock carried); checked bit for bit (check=True);
  3. the new ground (the strip, plus the old window's outermost ring where it lands inside): PD, T, Q, CW from the
     live parent at this step (NMMB PARENT_UPDATE_CHILD_PSFC / _BNDRY, the same kernels as the nest boundary, on the
     nest's own fine terrain), U and V likewise on the V points; then the cold-start derived columns there (PINT, Z,
     RTOP, W = 0, DWDT, OMGALF and the previous-step copies; init-derive-state's kernels), TRACERS_PREV = TRACERS;
  4. the nest boundary strips filled again from the new edges (bnd-boundary-init) and the parent -> nest coupler rebuilt
     for the new i_parent_start/j_parent_start (namgpu/nest/coupled.py WindowCoupler);
  5. a new Driver for the window; the old window's arrays are freed.

Where to move and when (Tracker): every `every_min` minutes of model time the nest's sea-level pressure (surface
pressure reduced with the lowest layer's virtual temperature) is box-smoothed over `smooth_cells` cells on the card,
the minimum outside the boundary band is the storm centre; one 32-byte read per check is the only host sync this adds.
When the centre is more than `cells` parent cells from the window centre in either direction, the window moves by the
offset rounded to whole parent cells, held `margin` parent cells inside the parent (NMMB's own rule for a nest).

Not here (named in the receipt): nests inside a moving nest, several cards, two-way feedback during moves.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import time
import types
from pathlib import Path

import numpy as np

from namgpu.nest import move as mv

UNIT = "nest-live-move"
RD, P608 = np.float32(287.04), np.float32(0.608)
BLOCK = 128
# derived cold-start columns written on the new ground (init-derive-state's cold branch), and kept elsewhere
DERIVED = ("PINT", "Z", "RTOP", "DWDT", "OMGALF", "UP", "VP", "TP", "TOLD", "W")


# ---------------------------------------------------------------------------------------------- the envelope

class Envelope:
    """A saved start (slab_init format) of the big window (ei0, ej0) of the reference grid, memory-mapped. cut()
    returns the skeleton of the window (i0, j0) of size (jm, im), every row/column-shaped owner cut on the host."""

    def __init__(self, src: Path, ei0: int, ej0: int):
        from namgpu.init import slab_init
        self.src = Path(src)
        with open(self.src / "skeleton.pkl", "rb") as f:
            self.d = pickle.load(f)
        if self.d["format"] != slab_init.FORMAT:
            raise SystemExit(f"{UNIT}: envelope {src} has format {self.d['format']}, this code reads {slab_init.FORMAT}")
        self.jme, self.ime, self.lm = int(self.d["jm"]), int(self.d["im"]), int(self.d["lm"])
        self.ei0, self.ej0 = int(ei0), int(ej0)
        self.meta = {int(k): v for k, v in self.d["meta"].items()}
        self.full = {k: tuple(m["shape"]) for k, m in self.meta.items() if m["owner"] == k}
        self._mm: dict = {}

    def axes(self, shape) -> tuple:
        """(row axis, column axis) of an owner shape: the last axis of JME entries and the last of IME entries."""
        shape = tuple(shape)
        ra = max((i for i, s in enumerate(shape) if s == self.jme), default=None)
        ca = max((i for i, s in enumerate(shape) if s == self.ime), default=None)
        if self.jme * self.ime in shape:
            raise SystemExit(f"{UNIT}: an envelope array of shape {shape} has a JM*IM axis; no cut rule")
        return ra, ca

    def mmap(self, o):
        if o not in self._mm:
            self._mm[o] = np.load(self.src / f"a{o}.npy", mmap_mode="r")
        return self._mm[o]

    def cut_owner(self, o, rows: slice, cols: slice) -> np.ndarray:
        h = self.mmap(o)
        ra, ca = self.axes(h.shape)
        idx = [slice(None)] * h.ndim
        if ra is not None:
            idx[ra] = rows
        if ca is not None:
            idx[ca] = cols
        return np.array(h[tuple(idx)], copy=True, order="C")      # a writable copy, never the read-only map


def load_window(ctx, env: Envelope, i0: int, j0: int) -> dict:
    """slab_init.load for the window (i0, j0) of the envelope's grid: ctx.grid must already be the window's grid."""
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor
    from namgpu.init import slab_init
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    r0, c0 = j0 - env.ej0, i0 - env.ei0
    if r0 < 0 or c0 < 0 or r0 + jm > env.jme or c0 + im > env.ime:
        raise SystemExit(f"{UNIT}: window ({i0}, {j0}) {im} x {jm} leaves the envelope ({env.ei0}, {env.ej0}) "
                         f"{env.ime} x {env.jme}: the storm left the area the run prepared (--move-reach-km)")
    if (jm, im) in ((env.jme, env.ime),) or jm == env.jme or im == env.ime or jm == im:
        raise SystemExit(f"{UNIT}: window {im} x {jm} and envelope {env.ime} x {env.jme} must differ in every size")
    rows, cols = slice(r0, r0 + jm), slice(c0, c0 + im)
    meta = env.meta
    t0 = time.perf_counter()
    want = sorted({(m["owner"], m["dev"]) for m in meta.values()})

    def up(od):
        o, dev = od
        h = env.cut_owner(o, rows, cols)
        if dev:
            with cp.cuda.Stream(non_blocking=True) as s_:
                b = cp.asarray(h)
                s_.synchronize()
            return od, b
        return od, h

    bases = {}
    nbytes = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for od, b in ex.map(up, want):
            bases[od] = b
            nbytes += int(b.nbytes)
    built = {}

    def build(k):
        if k in built:
            return built[k]
        m = meta[k]
        o, dev = m["owner"], m["dev"]
        base = bases[(o, dev)]
        fshape = env.full[o]
        ra, ca = env.axes(fshape)
        shape = list(m["shape"])
        if k == o:
            arr = base
        elif ra is None and ca is None:
            st_b = [x * base.itemsize for x in m["strides"]]
            if dev:
                arr = cp.ndarray(shape, dtype=m["dtype"], memptr=base.data + m["off"] * base.itemsize, strides=st_b)
            else:
                arr = np.lib.stride_tricks.as_strided(base.reshape(-1)[m["off"]:], shape, st_b)
        else:
            # a view of a cut owner: whole leading planes of the owner, with the owner's trailing shape from its
            # first cut axis (TRACERS[k], PINT[1:], the bc slot views)
            ax = min(a_ for a_ in (ra, ca) if a_ is not None)
            tr_full = tuple(fshape[ax:])
            plane_g = int(np.prod(tr_full))
            size = int(np.prod(shape))
            tnd = len(tr_full)
            if (not m["c"] or m["off"] % plane_g or size % plane_g or len(shape) < tnd
                    or tuple(shape[len(shape) - tnd:]) != tr_full):
                raise SystemExit(f"{UNIT}: view {k} {m['shape']} of owner {o} {list(fshape)} is not whole planes; "
                                 "no cut rule")
            tr_cut = tuple(base.shape[ax:])
            plane_l = int(np.prod(tr_cut))
            p0, n = m["off"] // plane_g, size // plane_g
            arr = base.reshape(-1)[p0 * plane_l:(p0 + n) * plane_l].reshape(tuple(shape[:len(shape) - tnd]) + tr_cut)
        built[k] = arr
        return arr

    def rebuild(obj):
        if isinstance(obj, slab_init._Ref):
            return build(obj.k)
        if isinstance(obj, dict):
            return {k: rebuild(v) for k, v in obj.items() if not isinstance(v, slab_init._Skipped)}
        if isinstance(obj, list):
            return [rebuild(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(rebuild(v) for v in obj)
        if isinstance(obj, types.SimpleNamespace):
            return types.SimpleNamespace(**{k: rebuild(v) for k, v in vars(obj).items()})
        return obj

    sk = env.d["skel"]
    ctx.st.fields.clear()
    ctx.st.fields.update(rebuild(sk["st"]))
    ctx.bc = rebuild(sk["bc"])
    ctx.tables.clear()
    ctx.tables.update(rebuild(sk["tables"]))
    for k, v in rebuild(sk["attrs"]).items():
        if k in slab_init.CTX_SKIP or isinstance(v, slab_init._Skipped):
            continue
        if v is None and getattr(ctx, k, None) is not None:
            continue
        setattr(ctx, k, v)
    vg = env.d["vgrid"]
    ctx.grid.set_vcoord(vg["pt"], vg["pdtop"], vg["sg1"], vg["sg2"])
    for k in ("dsg1", "dsg2", "sgml1", "sgml2"):
        setattr(ctx.grid, k, vg[k])
    from namgpu.steps import drv_loop_limits as dll
    dll.run(ctx)
    return {"rows": [r0, r0 + jm], "cols": [c0, c0 + im], "owners": len(want), "mib": round(nbytes / 2**20, 1),
            "s": round(time.perf_counter() - t0, 3)}


def window_constants(ctx) -> int:
    """init-grid-consts for the window (its edges set the hdac* boundary blocks and the SW corner scalars): written into
    the loaded tables['grid'] arrays in place, so every alias of them sees it. Returns the arrays written."""
    import cupy as cp
    from namgpu.init import init_grid_consts
    dummy = types.SimpleNamespace(cfg=ctx.cfg, stream=ctx.stream, tables={})
    out = init_grid_consts.run(dummy)
    g = ctx.tables.get("grid") or {}
    n = 0
    with ctx.stream:
        for k, v in out.items():
            d = g.get(k)
            if isinstance(v, cp.ndarray) and isinstance(d, cp.ndarray) and d.shape == v.shape:
                d[...] = v
                n += 1
        sw = ctx.tables.get("grid_sw")
        if isinstance(sw, cp.ndarray) and sw.size == 2:
            sw[0] = ctx.st["GLAT"][0, 0]
            sw[1] = ctx.st["GLON"][0, 0]
    return n


# ---------------------------------------------------------------------------------------------- the overlay

def _bits_differ_dev(a, b) -> object:
    """Device count of differing bit patterns (no sync)."""
    import cupy as cp
    a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
    u = {1: cp.uint8, 2: cp.uint16, 4: cp.uint32, 8: cp.uint64}[a.dtype.itemsize]
    return cp.count_nonzero(a.view(u) != b.view(u))


def overlay_live(new, old, dj: int, di: int) -> dict:
    """move.overlay with the old window live on the card (device to device). new holds the window's fresh state."""
    import cupy as cp
    from namgpu.init import slab_init
    jm, im = int(new.grid.jm), int(new.grid.im)
    o_r, o_c, n_r, n_c = mv.overlap(jm, im, dj, di)
    ring_r, ring_c = mv.ring_lines(jm, im, dj, di)
    rec = {"carried": [], "geometry": [], "window_constants": [], "rows": 0, "cols": 0, "strips": 0,
           "globals_from_old": [], "fresh_only": 0, "old_only_dropped": 0, "relaid": [], "refused": []}
    pairs = []                                   # (new array, old array) carried, for the bit check
    seen = set()
    stream = new.stream

    def ident(a):
        return (int(a.data.ptr), tuple(a.shape), tuple(a.strides), str(a.dtype)) if isinstance(a, cp.ndarray) \
            else (int(a.__array_interface__["data"][0]), tuple(a.shape), tuple(a.strides), str(a.dtype))

    def isarr(x):
        return isinstance(x, (np.ndarray, cp.ndarray))

    def put(f, src_view):
        if isinstance(f, cp.ndarray):
            f[..., n_r, n_c] = src_view if isinstance(src_view, cp.ndarray) else cp.asarray(np.ascontiguousarray(src_view))
        else:
            f[..., n_r, n_c] = cp.asnumpy(src_view) if isinstance(src_view, cp.ndarray) else src_view

    def leaf(path, f, o):
        name = mv._leaf_name(path)
        if o is None:
            rec["fresh_only"] += 1
            rec.setdefault("fresh_only_paths", []).append(path)
            return f
        if f is None:
            rec["old_only_dropped"] += 1      # made lazily in the run: rebuilt by its unit (move.py, CARRY_LAZY {})
            return None
        fs, os_ = tuple(f.shape), tuple(o.shape)
        if fs != os_ or str(f.dtype) != str(o.dtype):
            if (len(fs) >= 2 and len(os_) >= 2 and fs[-2:] == os_[-2:] == (jm, im) and str(f.dtype) == str(o.dtype)
                    and isinstance(f, cp.ndarray) and isinstance(o, cp.ndarray)):
                # re-laid after the first step (move.RESHAPED_AT_RUN, MPRATES): the old layout, overlap carried,
                # strip zero (a cold start's first step zero-fills it)
                a_ = cp.zeros(os_, o.dtype)
                a_[..., n_r, n_c] = o[..., o_r, o_c]
                if ring_r:
                    a_[..., ring_r, :] = 0
                if ring_c:
                    a_[..., :, ring_c] = 0
                rec["relaid"].append(f"{path}: fresh {fs} -> {os_}")
                pairs.append((path, a_, o))
                return a_
            rec["refused"].append(f"{path}: fresh {fs} {f.dtype} vs old {os_} {o.dtype}")
            return f
        kind = mv.classify(fs, jm, im)
        key = ident(f)
        if key in seen:
            return f
        seen.add(key)
        if kind == "grid":
            if name in mv.GEOMETRY:
                rec["geometry"].append((path, _bits_differ_dev(f[..., n_r, n_c], o[..., o_r, o_c]))
                                       if isinstance(f, cp.ndarray) and isinstance(o, cp.ndarray) else (path, None))
                return f
            if name in mv.WINDOW_CONSTANTS:
                rec["window_constants"].append(path)
                return f
            keep_r = f[..., ring_r, :].copy() if ring_r else None
            keep_c = f[..., :, ring_c].copy() if ring_c else None
            put(f, o[..., o_r, o_c])
            if ring_r:
                f[..., ring_r, :] = keep_r
            if ring_c:
                f[..., :, ring_c] = keep_c
            rec["carried"].append(path)
            pairs.append((path, f, o))
            return f
        if kind == "row":
            rec["rows"] += 1
            return f
        if kind == "col":
            rec["cols"] += 1
            return f
        if kind == "strip":
            rec["strips"] += 1
            return f
        if kind == "other":
            rec["refused"].append(f"{path}: shape {fs} has a JM x IM axis")
            return f
        if any(f"/{k}" in path for k in mv.FRESH_GLOBALS):
            return f
        if isinstance(f, cp.ndarray):
            f[...] = o if isinstance(o, cp.ndarray) else cp.asarray(np.ascontiguousarray(o))
        else:
            f[...] = cp.asnumpy(o) if isinstance(o, cp.ndarray) else o
        rec["globals_from_old"].append(path)
        return f

    def walk(f, o, path):
        if isarr(f) or isarr(o):
            return leaf(path, f if isarr(f) else None, o if isarr(o) else None)
        if isinstance(f, dict) and (isinstance(o, dict) or o is None):
            o = o or {}
            for k in list(f.keys()):
                f[k] = walk(f[k], o.get(k), f"{path}/{k}")
            rec["old_only_dropped"] += sum(1 for k in o if k not in f)
            return f
        if isinstance(f, types.SimpleNamespace):
            ov = vars(o) if isinstance(o, types.SimpleNamespace) else {}
            for k, v in vars(f).items():
                setattr(f, k, walk(v, ov.get(k), f"{path}.{k}"))
            return f
        if isinstance(f, list) and isinstance(o, list) and len(f) == len(o):
            for i in range(len(f)):
                f[i] = walk(f[i], o[i], f"{path}[{i}]")
            return f
        if isinstance(f, tuple) and isinstance(o, tuple) and len(f) == len(o):
            return tuple(walk(u, v, f"{path}[{i}]") for i, (u, v) in enumerate(zip(f, o)))
        if (isinstance(f, (int, float, str, bool, type(None))) and f != o and path.startswith("/tables")
                and not any(f"/{k}" in path for k in mv.FRESH_GLOBALS)):
            rec["globals_from_old"].append(path)
            return o
        return f

    with stream:
        walk(new.st.fields, old.st.fields, "/st")
        walk(new.tables, {k: v for k, v in old.tables.items() if k not in slab_init.PER_RANK_TABLES}, "/tables")
        new.bc = walk(new.bc, old.bc, "/bc")
    # counters, flags and the clock: the old window's (what a restart set carries: slab_init.plan's attrs)
    attrs = []
    for k, v in vars(old).items():
        if k in slab_init.CTX_SKIP or k in mv.FRESH_ATTRS or mv._holds_array(v):
            continue
        try:
            pickle.dumps(v)
        except Exception:  # noqa: BLE001
            continue
        if repr(getattr(new, k, None)) != repr(v):
            attrs.append(k)
        setattr(new, k, v)
    rec["attrs_from_old"] = attrs
    og = old.grid
    new.grid.set_vcoord(og.pt, og.pdtop, og.sg1, og.sg2)
    for k in ("dsg1", "dsg2", "sgml1", "sgml2"):
        setattr(new.grid, k, getattr(og, k))
    if rec["refused"]:
        raise SystemExit(f"{UNIT}: refused (no overlay rule): {rec['refused'][:6]}")
    rec["_pairs"] = pairs
    return rec


def carried_check(rec: dict, jm: int, im: int, dj: int, di: int) -> dict:
    """Every carried array's kept cells against the old window's, bit for bit (the old ring lines excepted: they are
    new ground). Device counts, one read."""
    import cupy as cp
    o_r, o_c, n_r, n_c = mv.overlap(jm, im, dj, di)
    rr, cc = mv.ring_lines(jm, im, dj, di)
    keep = np.ones((n_r.stop - n_r.start, n_c.stop - n_c.start), bool)
    for r in rr:
        keep[r - n_r.start, :] = False
    for c in cc:
        keep[:, c - n_c.start] = False
    kd = cp.asarray(keep)
    counts = []
    for _p, f, o in rec["_pairs"]:
        a, b = f[..., n_r, n_c], o[..., o_r, o_c]
        u = {1: cp.uint8, 2: cp.uint16, 4: cp.uint32, 8: cp.uint64}[a.dtype.itemsize]
        d = cp.ascontiguousarray(a).view(u) != cp.ascontiguousarray(b).view(u)
        counts.append(cp.count_nonzero(d & kd))
    tot = [int(x) for x in cp.stack(counts).get()] if counts else []
    geo = [(p, int(x)) for p, x in rec["geometry"] if x is not None]
    bad = {rec["_pairs"][i][0]: n for i, n in enumerate(tot) if n}
    return {"arrays": len(tot), "cells_per_array": int(keep.sum()), "mismatched_arrays": len(bad), "mismatches": bad,
            "geometry_mismatches": sum(n for _p, n in geo), "geometry_arrays": len(geo)}


# ---------------------------------------------------------------------------------------------- new ground

def _band(strip: np.ndarray, width: int) -> np.ndarray:
    """Kept cells within `width` cells (8-neighbour steps) of the strip."""
    from scipy.ndimage import binary_dilation
    if width <= 0:
        return np.zeros_like(strip)
    return binary_dilation(strip, structure=np.ones((3, 3), bool), iterations=int(width)) & ~strip


def _spread(d, known, todo, steps: int):
    """Extend d (device, [..., JM, IM]), known on `known`, into `todo` by repeated means of known 4-neighbours.
    Returns (d, depth): depth is the step at which a cell got its value (0 where known, -1 where never)."""
    import cupy as cp
    kn = known.copy()
    depth = cp.where(kn, 0, -1).astype(cp.int32)
    sh = (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
          ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
          ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
          ((slice(None), slice(None, -1)), (slice(None), slice(1, None))))
    for it in range(1, int(steps) + 1):
        kf = kn.astype(cp.float32)
        dk = d * kf
        num = cp.zeros_like(d)
        cnt = cp.zeros(kn.shape, cp.float32)
        for dst, src in sh:
            num[(Ellipsis,) + dst] += dk[(Ellipsis,) + src]
            cnt[dst] += kf[src]
        grow = (~kn) & todo & (cnt > 0)
        d = cp.where(grow, num / cp.maximum(cnt, 1.0), d)
        depth = cp.where(grow, it, depth)
        kn = kn | grow
    return d, depth


def fill_from_parent(new, parent, i_ps: int, j_ps: int, sratio: int, strip: np.ndarray, seam: int = 3,
                     blend: int = 12) -> dict:
    """PD, T, Q, CW (H points) and U, V (V points) on the strip cells from the live parent, with the nest boundary's
    kernels (kernels/nest_bc.cu) on the whole parent: the parent's surface pressure on the nest's fine terrain, then
    T, Q, CW, U, V on the nest's levels for it. Seam blend (seam > 0): the same parent values are made on the kept cells
    within `seam` cells of the strip; the nest-minus-parent difference there is carried into the strip (means of known
    neighbours) and added with a weight falling from 1 at the seam to 0 `blend` cells in, so the new ground meets the
    kept cells without a step (a low the nest has deepened 2 hPa past its parent stays deepened across the seam).
    Kept cells are never written."""
    import cupy as cp
    from namgpu import kernels
    from namgpu.nest import twoway as tw
    K = kernels
    p, c = parent.st, new.st
    imp, jmp = int(parent.grid.im), int(parent.grid.jm)
    im, jm, lm = int(new.grid.im), int(new.grid.jm), int(new.grid.lm)
    f32, i32 = np.float32, np.int32
    pdtop, pt = f32(parent.grid.pdtop), f32(parent.grid.pt)
    band = _band(strip, seam)
    region = strip | band
    jh, ih = np.nonzero(region)
    vm = region.copy()
    vm[-1, :] = False
    vm[:, -1] = False
    jv, iv = np.nonzero(vm)
    vstrip = strip.copy()
    vstrip[-1, :] = False
    vstrip[:, -1] = False
    vband = band.copy()
    vband[-1, :] = False
    vband[:, -1] = False
    out = {"h_points": int(strip.sum()), "v_points": int(vstrip.sum()), "seam_cells": int(seam),
           "blend_cells": int(blend), "band_points": int(band.sum())}
    if not strip.any():
        return out
    h_iw, h_js, h_w = tw.interp_factors(i_ps, j_ps, sratio, ih, jh, False)
    v_iw, v_js, v_w = tw.interp_factors(i_ps, j_ps, sratio, iv, jv, True)
    for iw, js, ex, what in ((h_iw, h_js, 1, "H"), (v_iw, v_js, 2, "V")):
        if min(int(iw.min()), int(js.min())) < 0 or int(iw.max()) + ex >= imp or int(js.max()) + ex >= jmp:
            raise SystemExit(f"{UNIT}: new ground {what} interpolation leaves the parent")
    dev = lambda x: cp.asarray(np.ascontiguousarray(x))  # noqa: E731
    for k in ("PD", "FIS"):
        if not c[k].flags.c_contiguous:
            raise SystemExit(f"{UNIT}: nest {k} is not C-contiguous")
    nh, nv = len(ih), len(iv)
    with new.stream:
        pdb = c["PD"].copy()                     # the parent's PD on the nest terrain, at the region's points
        ic, jc, iw, js, w = dev(ih.astype(np.int32)), dev(jh.astype(np.int32)), dev(h_iw), dev(h_js), dev(h_w)
        K.get("nest_bc", "nest_bc_psfc")(((nh + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["FIS"], p["T"], p["Q"], i32(imp), i32(jmp), i32(lm), p["SG1"], p["SG2"], pdtop, pt,
            c["FIS"], i32(im), i32(nh), ic, jc, iw, js, w, pdb))
        sc = {v: cp.zeros((2, nh) if v == "PD" else (2, lm, nh), cp.float32) for v in ("PD", "T", "Q", "CW")}
        ptr = dev(np.array([sc[v].data.ptr for v in ("PD", "T", "Q", "CW") for _s in range(4)], np.uint64))
        side, off = dev(np.zeros(nh, np.int32)), dev(np.arange(nh, dtype=np.int64))
        lstr = dev(np.full(4, nh, np.int64))
        K.get("nest_bc", "nest_bc_h")(((nh + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["T"], p["Q"], p["CW"], i32(imp), i32(jmp), i32(lm), p["SG1"], p["SG2"], c["PSGML1"],
            c["SGML2"], pdtop, pt, pdb, i32(im), i32(nh), ic, jc, iw, js, w, side, off, ptr, lstr, f32(1.0)))
        par = {"PD": pdb}
        for v in ("T", "Q", "CW"):
            a_ = cp.zeros((lm, jm, im), cp.float32)
            a_[:, jc, ic] = sc[v][1]
            par[v] = a_
        icv, jcv, iwv, jsv, wv = dev(iv.astype(np.int32)), dev(jv.astype(np.int32)), dev(v_iw), dev(v_js), dev(v_w)
        sv = {v: cp.zeros((2, lm, nv), cp.float32) for v in ("U", "V")}
        ptrv = dev(np.array([sv[v].data.ptr for v in ("U", "V") for _s in range(4)], np.uint64))
        sidev, offv = dev(np.zeros(nv, np.int32)), dev(np.arange(nv, dtype=np.int64))
        lstrv = dev(np.full(4, nv, np.int64))
        K.get("nest_bc", "nest_bc_v")(((nv + BLOCK - 1) // BLOCK,), (BLOCK,), (
            p["PD"], p["U"], p["V"], i32(imp), i32(jmp), i32(lm), p["SG1"], p["SG2"], c["PSGML1"], c["SGML2"],
            pdtop, pt, pdb, i32(im), i32(nv), icv, jcv, iwv, jsv, wv, sidev, offv, ptrv, lstrv, f32(1.0)))
        for v in ("U", "V"):
            a_ = cp.zeros((lm, jm, im), cp.float32)
            a_[:, jcv, icv] = sv[v][1]
            par[v] = a_
        sm, smv = cp.asarray(strip), cp.asarray(vstrip)
        if seam > 0 and band.any():
            bm, bmv = cp.asarray(band), cp.asarray(vband)
            diffs = {}
            for v in ("PD", "T", "Q", "CW", "U", "V"):
                km_, tm_ = (bmv, smv) if v in ("U", "V") else (bm, sm)
                d = cp.where(km_, c[v] - par[v], 0).astype(cp.float32)
                diffs[v] = cp.abs(d).sum() / (cp.count_nonzero(km_) * (1 if d.ndim == 2 else lm))
                d, depth = _spread(d, km_, tm_, 3 * int(blend))
                wgt = cp.clip(1.0 - (depth.astype(cp.float32) - 1.0) / float(blend), 0.0, 1.0)
                wgt = cp.where(depth > 0, wgt, 0.0).astype(cp.float32)
                par[v] = par[v] + wgt * d
            hv = cp.stack(list(diffs.values())).get()
            out["seam_mean_abs_nest_minus_parent"] = {k: round(float(x), 5) for k, x in zip(diffs, hv)}
        c["PD"][...] = cp.where(sm, par["PD"], c["PD"])
        for v in ("T", "Q", "CW"):
            c[v][...] = cp.where(sm, par[v], c[v])
        for v in ("U", "V"):
            c[v][...] = cp.where(smv, par[v], c[v])
        c["Q"][...] = cp.where(sm, cp.maximum(c["Q"], np.float32(1e-12)), c["Q"])
        c["CW"][...] = cp.where(sm, cp.maximum(c["CW"], np.float32(0)), c["CW"])
    return out


def seam_jumps(ctx, strip: np.ndarray, top: int = 60) -> list:
    """Diagnostic (spec "diag"): for every (.., JM, IM) state array, the mean jump across the seam (new-ground cell
    against its kept 4-neighbour) over the mean step between two kept cells one row further in. A field the move left
    stale on the new ground stands out with a large ratio."""
    import cupy as cp
    jm, im = strip.shape
    pairs_s, pairs_k = [], []
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        js, is_ = np.nonzero(strip)
        jn, in_ = js + dj, is_ + di
        jk, ik = jn + dj, in_ + di
        ok = (jn >= 0) & (jn < jm) & (in_ >= 0) & (in_ < im) & (jk >= 0) & (jk < jm) & (ik >= 0) & (ik < im)
        ok &= ~strip[np.clip(jn, 0, jm - 1), np.clip(in_, 0, im - 1)]
        ok &= ~strip[np.clip(jk, 0, jm - 1), np.clip(ik, 0, im - 1)]
        pairs_s.append((js[ok], is_[ok], jn[ok], in_[ok]))
        pairs_k.append((jn[ok], in_[ok], jk[ok], ik[ok]))
    a0 = [np.concatenate([p[k] for p in pairs_s]) for k in range(4)]
    b0 = [np.concatenate([p[k] for p in pairs_k]) for k in range(4)]
    if not len(a0[0]):
        return []
    A = [cp.asarray(x) for x in a0]
    B = [cp.asarray(x) for x in b0]
    names, vals = [], []
    seen = set()
    for k, f in ctx.st.fields.items():
        if not isinstance(f, cp.ndarray) or f.ndim < 2 or tuple(f.shape[-2:]) != (jm, im) or f.dtype.kind != "f":
            continue
        if int(f.data.ptr) in seen:
            continue
        seen.add(int(f.data.ptr))
        x = f.reshape(-1, jm, im)
        js_ = cp.abs(x[:, A[0], A[1]] - x[:, A[2], A[3]]).mean()
        jk_ = cp.abs(x[:, B[0], B[1]] - x[:, B[2], B[3]]).mean()
        names.append(k)
        vals.append(cp.stack([js_, jk_]))
    h = cp.stack(vals).get() if vals else np.zeros((0, 2))
    rows = [(n, float(a), float(b), float(a / b) if b > 0 else (float("inf") if a > 0 else 0.0))
            for n, (a, b) in zip(names, h)]
    rows.sort(key=lambda r: -r[3] if np.isfinite(r[3]) else -1e30)
    return [[n, round(a, 6), round(b, 6), round(r, 2)] for n, a, b, r in rows[:top]]


def seed_strips(new, old_bc: dict, cpl, dj: int, di: int) -> dict:
    """The boundary strips' carried values (slot 0) for the new window. At a parent step's end the old window's slot 0
    is the parent's value interpolated to its edge points (the last parent step's tendency has been stepped in), so the
    new window's slot 0 is the same: the parent's values now at the new edge points (the coupler's own kernels, with
    slot 0 zeroed and RECIP 1, write them into slot 1; copied to slot 0). Where a new edge point is the same ground
    point on the same side and row as an old one (S/N strips when dj = 0, W/E strips when di = 0) the old value is kept,
    so a move of 0 cells changes nothing. Filling slot 0 again from the state's edge rows instead (bnd-boundary-init) is
    not the same: the blending rows relax the state toward slot 0, they do not equal it (a 0-cell move then changed the
    run at +2 h, null-move test N2, 2026-11-18). Slot 1 is written by the coupler at the next parent step."""
    import cupy as cp
    from namgpu.nest.twoway import H_BC, V_BC
    jm, im = int(new.grid.jm), int(new.grid.im)
    o_r, o_c, n_r, n_c = mv.overlap(jm, im, dj, di)
    bc = new.bc
    keys = [f"{s}_{n}" for n in H_BC + V_BC for s in "SNWE"]
    rec = {"from_parent": [], "kept": []}
    with new.stream:
        for k in keys:
            bc[k][0][...] = 0
        r0 = cpl.recip
        cpl.recip = np.float32(1.0)
        try:
            cpl.boundary()                       # slot 1 = parent value - 0
        finally:
            cpl.recip = r0
        for k in keys:
            a = bc[k]
            a[0][...] = a[1]
            side = k[0]
            o = old_bc.get(k)
            if o is None or tuple(o.shape) != tuple(a.shape):
                rec["from_parent"].append(k)
                continue
            # a V strip's last column (S/N) or row (W/E) is the V grid's padding, not a boundary value: it is never
            # carried inward (it put 44 m/s of W into the first minutes after a 2-cell move, smoke S6, 2026-11-18)
            vpad = k.split("_", 1)[1] in V_BC
            if side in "SN" and dj == 0:
                oc, nc = o_c, n_c
                if vpad and di > 0:
                    oc, nc = slice(o_c.start, im - 1), slice(n_c.start, im - 1 - di)
                a[0][..., nc] = o[0][..., oc]
                rec["kept"].append(k)
            elif side in "WE" and di == 0:
                or_, nr = o_r, n_r
                if vpad and dj > 0:
                    or_, nr = slice(o_r.start, jm - 1), slice(n_r.start, jm - 1 - dj)
                a[0][..., nr, :] = o[0][..., or_, :]
                rec["kept"].append(k)
            else:
                rec["from_parent"].append(k)
    return {"kept": len(rec["kept"]), "from_parent": len(rec["from_parent"])}


def derive_strip(new, strip: np.ndarray) -> list:
    """The cold-start derived columns on the new ground (init-derive-state's kernels on scratch copies, then the strip
    cells only): PINT and Z from the new PD/T/Q, RTOP, W = 0, DWDT, OMGALF, UP/VP/TP/TOLD = the new U/V/T; and
    TRACERS_PREV = TRACERS there. Kept cells are not touched."""
    import cupy as cp
    from namgpu import kernels
    from namgpu.init import init_derive_state as ids
    st, g = new.st, new.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    pt, pdtop = np.float32(g.pt), np.float32(g.pdtop)
    lev = (lm, jm, im)
    with new.stream:
        tmp = {k: cp.zeros((lm + 1, jm, im) if k in ("PINT", "Z") else lev, cp.float32) for k in DERIVED}
        o3, q2 = cp.zeros(lev, cp.float32), st["Q2"].copy()
        k = lambda n: kernels.get(ids.UNIT, n)  # noqa: E731
        z0, cold = np.int32(0), np.int32(0)
        k("init_derive_pint")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            st["DSG1"], st["DSG2"], st["PD"], tmp["PINT"], pt, pdtop, cold, np.int32(im), np.int32(jm), np.int32(lm)))
        k("init_derive_levels")(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D, (
            st["T"], st["Q"], st["CW"], st["U"], st["V"], tmp["PINT"], tmp["RTOP"], tmp["DWDT"], tmp["OMGALF"],
            tmp["UP"], tmp["VP"], tmp["TP"], tmp["TOLD"], tmp["W"], o3, q2, cold, z0, np.int32(im), np.int32(jm),
            np.int32(im), np.int32(jm), np.int32(lm)))
        k("init_derive_z")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            st["FIS"], st["PD"], st["T"], st["Q"], st["CW"], tmp["PINT"], st["DSG1"], st["DSG2"], tmp["Z"], pdtop,
            ids.GI, cold, np.int32(im), np.int32(jm), np.int32(lm)))
        m = cp.asarray(strip)
        done = []
        for n_, t in tmp.items():
            a = st.fields.get(n_)
            if a is None or tuple(a.shape) != tuple(t.shape):
                continue
            a[...] = cp.where(m, t, a)
            done.append(n_)
        pdo = st.fields.get("PDO")              # the previous step's PD (dynamics): the new PD on the new ground.
        if pdo is not None and tuple(pdo.shape) == tuple(st["PD"].shape):   # Left at the envelope's: 4.8 Pa/s
            pdo[...] = cp.where(m, st["PD"], pdo)                           # window-mean dPD/dt at the first step
            done.append("PDO")                                              # (smoke4, 2026-11-18)
        tr, tp_ = st.fields.get("TRACERS"), st.fields.get("TRACERS_PREV")
        if tr is not None and tp_ is not None and tr.shape == tp_.shape:
            tp_[...] = cp.where(m, tr, tp_)
            done.append("TRACERS_PREV")
    return done


# ---------------------------------------------------------------------------------------------- where the storm is

class Tracker:
    """The smoothed sea-level-pressure minimum of the nest, on the card; one 32-byte read per fix."""

    def __init__(self, smooth_cells: int, band: int):
        self.smooth, self.band = int(smooth_cells), int(band)

    def fix(self, ctx) -> dict:
        import cupy as cp
        from cupyx.scipy import ndimage as ndi
        st, g = ctx.st, ctx.grid
        lm = int(g.lm)
        with ctx.stream:
            t, q = st["T"][lm - 1], st["Q"][lm - 1]
            psl = (st["PD"] + np.float32(g.pt)) * cp.exp(st["FIS"] / (RD * t * (np.float32(1.0) + P608 * q)))
            sm = ndi.uniform_filter(psl, size=self.smooth, mode="nearest")
            b = self.band
            sm[:b, :] = cp.inf
            sm[-b:, :] = cp.inf
            sm[:, :b] = cp.inf
            sm[:, -b:] = cp.inf
            k = cp.argmin(sm)
            v = cp.stack([k.astype(cp.float64), sm.ravel()[k].astype(cp.float64), psl.min().astype(cp.float64),
                          st["GLAT"].ravel()[k].astype(cp.float64), st["GLON"].ravel()[k].astype(cp.float64)])
            h = v.get()
        im = int(g.im)
        kk = int(h[0])
        return {"j": kk // im, "i": kk % im, "psl_smooth_hpa": round(float(h[1]) / 100.0, 2),
                "psl_min_hpa": round(float(h[2]) / 100.0, 2), "lat": round(float(np.degrees(h[3])), 3),
                "lon": round(float(np.degrees(h[4])), 3)}


# ---------------------------------------------------------------------------------------------- the mover

class LiveMover:
    """Owns one moving nest (a coupled.Domain whose plan entry has "move"). check() at a parent step boundary."""

    def __init__(self, dom, parent_dom, spec: dict, a, start, out: Path, log=print):
        from namgpu.nest import twoway as tw
        self.dom, self.par, self.spec, self.a, self.start, self.log = dom, parent_dom, spec, a, start, log
        self.out = Path(out)
        (self.out / "move").mkdir(parents=True, exist_ok=True)
        self.base = Path(spec["base_config"])
        self.env = Envelope(Path(spec["envelope"]), int(spec["env_i0"]), int(spec["env_j0"]))
        cfg = dom.ctx.cfg
        self.sratio = tw.space_ratio(parent_dom.ctx.cfg, cfg)
        self.i0 = self.j0 = 0
        self.i_ps0, self.j_ps0 = int(cfg.raw["i_parent_start"]), int(cfg.raw["j_parent_start"])
        self.every = float(spec.get("every_min", 15.0))
        self.cells = int(spec.get("cells", 3))
        self.margin = int(spec.get("margin", int(parent_dom.ctx.cfg.lnsh) + 3))
        self.tracker = Tracker(int(spec.get("smooth_cells", 7)), int(cfg.lnsh) + 3)
        self.check_each = bool(spec.get("check", True))
        self.null_at = spec.get("null_move_at_h")          # a forced move of 0 cells (the proof that a move is exact)
        self.gen = 0
        self.fixes, self.moves = [], []
        # the same shock numbers before any move (the reference): from the first step to the first move, at most 3 h
        sph = int(round(3600.0 / float(cfg.dt)))
        n0 = int(dom.ctx.ntimestep)
        self.monitors = [mv.Monitor(dom.ctx, n0 + 1, 3 * sph, None, "no move yet (reference)")]
        self.tracker.smooth = int(spec.get("smooth_cells", self.tracker.smooth))

    # -- where the window may go
    def _room(self, di_p: int, dj_p: int) -> tuple:
        cfgp = self.par.ctx.cfg
        imp, jmp = int(cfgp.im), int(cfgp.jm)
        im, jm = int(self.dom.ctx.cfg.im), int(self.dom.ctx.cfg.jm)
        nxp, nyp = (im - 1) // self.sratio, (jm - 1) // self.sratio
        ips = self.i_ps0 + self.i0 // self.sratio
        jps = self.j_ps0 + self.j0 // self.sratio
        lo_i, hi_i = self.margin + 1, imp - self.margin - nxp
        lo_j, hi_j = self.margin + 1, jmp - self.margin - nyp
        # and inside the reach (the envelope), in whole parent cells: a storm that leaves it holds the nest at its edge
        e_lo_i = self.i_ps0 + -(-self.env.ei0 // self.sratio)
        e_hi_i = self.i_ps0 + (self.env.ei0 + self.env.ime - im) // self.sratio
        e_lo_j = self.j_ps0 + -(-self.env.ej0 // self.sratio)
        e_hi_j = self.j_ps0 + (self.env.ej0 + self.env.jme - jm) // self.sratio
        lo_i, hi_i, lo_j, hi_j = max(lo_i, e_lo_i), min(hi_i, e_hi_i), max(lo_j, e_lo_j), min(hi_j, e_hi_j)
        ni, nj = min(max(ips + di_p, lo_i), hi_i), min(max(jps + dj_p, lo_j), hi_j)
        return ni - ips, nj - jps

    def check(self, n_parent: int, fh: float) -> dict | None:
        ctx = self.dom.ctx
        fx = self.tracker.fix(ctx)
        im, jm = int(ctx.grid.im), int(ctx.grid.jm)
        di_c, dj_c = fx["i"] - (im - 1) / 2.0, fx["j"] - (jm - 1) / 2.0
        fx.update(fh=round(fh, 4), gen=self.gen, off_cells=[round(di_c, 1), round(dj_c, 1)])
        self.fixes.append(fx)
        forced_null = self.null_at is not None and abs(fh - float(self.null_at)) < 1e-6
        lim = self.cells * self.sratio
        if not forced_null and abs(di_c) <= lim and abs(dj_c) <= lim:
            return None
        di_p, dj_p = (0, 0) if forced_null else (int(round(di_c / self.sratio)), int(round(dj_c / self.sratio)))
        di_p, dj_p = self._room(di_p, dj_p)
        if (di_p, dj_p) == (0, 0) and not forced_null:
            fx["held"] = "the parent's edge"
            return None
        written = set().union(*[m.frames for m in self.monitors]) if self.monitors else set()
        if (self.spec.get("frame_before", True) and not ctx.driver._due("history") and not forced_null
                and int(ctx.ntimestep) not in written):
            ctx.driver.write_history()            # the old window at the move time, to set beside the frames after it
        return self.move(di_p, dj_p, n_parent, fh, fx)

    def move(self, di_p: int, dj_p: int, n_parent: int, fh: float, fx: dict) -> dict:
        import cupy as cp
        from namgpu import config, devmem, grid as gridmod, state as statemod
        from namgpu.boundary import bnd_boundary_init
        from namgpu.driver import Ctx, Driver
        from namgpu.io import io_history
        from namgpu.nest import coupled as cpl_mod, domain
        from namgpu.state import release_init_leftovers
        from namgpu.steps import drv_loop_limits, drv_timing_receipt as timing
        stream = self.dom.ctx.stream
        t0 = time.perf_counter()
        old = self.dom.ctx
        rec = {"gen": self.gen + 1, "fh": round(fh, 4), "parent_step": n_parent, "nest_step": int(old.ntimestep),
               "shift_parent_cells": [di_p, dj_p], "fix": fx}
        io_history.drain(old)                      # the old window's history writes hold its buffers
        timing.drain(old)
        stream.synchronize()
        t_a = time.perf_counter()
        di, dj = di_p * self.sratio, dj_p * self.sratio
        ni0, nj0 = self.i0 + di, self.j0 + dj
        ips, jps = self.i_ps0 + ni0 // self.sratio, self.j_ps0 + nj0 // self.sratio
        bcfg = config.load(self.base)
        cfgfile = self.out / "move" / f"nam_configfile_w{self.gen + 1:03d}"
        win = mv.window_configfile(self.base, cfgfile, ni0, nj0, str(bcfg.raw.get("domain_name", "nest")),
                                   Path(str(config.static_npz(bcfg) or "")),
                                   extra={"i_parent_start": ips, "j_parent_start": jps})
        prev_env = os.environ.get(config.CONFIG_ENV)
        os.environ[config.CONFIG_ENV] = str(cfgfile)
        try:
            cfg = config.load(cfgfile)
            config.start_filter(cfg, self.a.source)
            g = gridmod.build(cfg)
            st = statemod.State(g, names={k for k, v in statemod.all_specs().items() if v.rank == 0})
            new = Ctx(cfg=cfg, grid=g, st=st, stream=stream)
            new.slab = new.comm = new.decomp = None
            rec["load"] = load_window(new, self.env, ni0, nj0)
            release_init_leftovers(new)
            rec["window_constants_written"] = window_constants(new)
            t_o = time.perf_counter()
            ov = overlay_live(new, old, dj, di)
            rec["overlay_s"] = round(time.perf_counter() - t_o, 3)
            mk = mv.masks(int(g.jm), int(g.im), dj, di)
            strip = mk["strip"]
            rec["fill"] = fill_from_parent(new, self.par.ctx, ips, jps, self.sratio, strip,
                                           seam=int(self.spec.get("seam_cells", 3)),
                                           blend=int(self.spec.get("blend_cells", 12)))
            rec["derived"] = derive_strip(new, strip)
            if self.spec.get("diag") or os.environ.get("NAMGPU_MOVE_DIAG"):
                rec["seam_jumps"] = seam_jumps(new, strip)
            if self.check_each:
                rec["check"] = carried_check(ov, int(g.jm), int(g.im), dj, di)
            rec["overlay"] = {k: (len(v) if isinstance(v, list) else v) for k, v in ov.items()
                              if k not in ("_pairs", "geometry")}
            new.case_dir, new.start, new.start_time = old.case_dir, old.start, old.start_time
            new.init_source, new.out_dir = old.init_source, old.out_dir
            new.hist_pin_key = getattr(old, "hist_pin_key", None)
            new.forecast_hours, new.dt = old.forecast_hours, cfg.dt
            new.history_select, new.restart_writer = old.history_select, None
            new.concurrent_parent, new.i_am_a_nest = True, True
            new.parent_child_time_ratio = self.dom.ratio
            new.move_monitors = []
            drv = Driver(new)
            new.driver = drv
            drv.check_ready()
            with stream:
                drv_loop_limits.run(new)
                bnd_boundary_init.run(new)        # the strips' buffers at this window's shapes (values set below)
            cpl = cpl_mod.WindowCoupler(self.par.ctx, new, self.dom.ratio, ips, jps)
            rec["strips"] = seed_strips(new, old.bc, cpl, dj, di)
        finally:
            if prev_env is None:
                os.environ.pop(config.CONFIG_ENV, None)
            else:
                os.environ[config.CONFIG_ENV] = prev_env
        ov.pop("_pairs", None)
        self.dom.ctx, self.dom.drv, self.dom.cpl = new, drv, cpl
        self.i0, self.j0 = ni0, nj0
        self.gen += 1
        del old, ov
        gc.collect()
        devmem.release(stream)
        stream.synchronize()
        rec["move_s"] = round(time.perf_counter() - t0, 3)
        rec["move_s_after_history_drain"] = round(time.perf_counter() - t_a, 3)
        rec["window"] = {"i0": ni0, "j0": nj0, "i_parent_start": ips, "j_parent_start": jps,
                         "centre": win["centre_geo"], "configfile": str(cfgfile)}
        rec["strip_cells"] = int(strip.sum())
        # shock numbers after the move (move.Monitor: device scalars, drained at the hourly sync)
        per_min = 60.0 / float(cfg.dt)
        n_ = int(new.ntimestep)
        sph = int(round(3600.0 / float(cfg.dt)))
        # extra frames after the move, never on an hourly history step (two writes of one file would race)
        frames = [f for f in (n_ + int(round(m * per_min)) for m in self.spec.get("extra_frames_min",
                                                                                   mv.EXTRA_FRAMES_MIN)) if f % sph]
        for m_ in self.monitors:                  # a monitor ends at the next move (its PD before is another window's)
            m_.n1 = min(m_.n1, n_ + 1)
        mon = mv.Monitor(new, n_ + 1, int(round(60 * per_min)), mk, f"after move {self.gen}", frames=frames)
        self.monitors.append(mon)
        self.moves.append(rec)
        ck = rec.get("check") or {}
        self.log(f"{UNIT}: move {self.gen} at fh {fh:.2f}: {di_p:+d}, {dj_p:+d} parent cells to centre "
                 f"{win['centre_geo']}, storm {fx['lat']}, {fx['lon']} {fx['psl_smooth_hpa']} hPa; strip "
                 f"{rec['strip_cells']} cells; kept cells {ck.get('arrays')} arrays, {ck.get('mismatched_arrays')} "
                 f"mismatched; {rec['move_s']} s")
        return rec

    def after_step(self):
        ctx = self.dom.ctx
        for m in self.monitors:
            m.after_step(ctx)

    def drain(self):
        for m in self.monitors:
            m.drain()

    def receipt(self) -> dict:
        return {"unit": UNIT, "every_min": self.every, "cells": self.cells, "space_ratio": self.sratio,
                "margin_parent_cells": self.margin, "smooth_cells": self.tracker.smooth,
                "seam_cells": int(self.spec.get("seam_cells", 3)), "blend_cells": int(self.spec.get("blend_cells", 12)),
                "envelope": {"dir": str(self.env.src), "i0": self.env.ei0, "j0": self.env.ej0,
                             "im": self.env.ime, "jm": self.env.jme},
                "moves": self.moves, "fixes": self.fixes, "monitors": [m.receipt() for m in self.monitors],
                "not_here": ["nests inside a moving nest", "several cards", "two-way feedback during moves"]}
