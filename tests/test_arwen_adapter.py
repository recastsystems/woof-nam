"""The ArWen column adapter on a REAL NMMB state (namgpu/physics/arwen_adapter.py).

State: NAMGPU_ADAPTER_STATE=<npz> written by tools/adapter_dump_state.py (the real init chain on the full
grid, e.g. the NAM's own 2026-01-23 12Z analysis). NAMGPU_ADAPTER_WINDOW="j0:j1,i0:i1" crops it (default: the
whole grid); the crop is treated as the domain. Skips when no state or no GPU is there.

Checks:
  1. gather -> scatter with no physics is BITWISE identity on every NMMB field (T, Q, TRACERS, CW, F_*, U, V);
     so is apply_tendencies with all-zero tendencies;
  2. the ArWen views are the NMMB column flipped and converted (t bitwise, qv = Q/(1-Q), p8w/z8w ordered,
     z8w[0] = FIS/g, every value finite);
  3. a change goes back only where it was made: th + 1 K in one column moves only that column's T (by pii);
     a qc change re-couples CW/F_ICE/F_RAIN in that cell only; +1 m/s mass-point u in the interior gives
     +1 m/s at the interior V points and leaves V untouched; the outer row is never written.

    NAMGPU_ADAPTER_STATE=... python -m pytest tests/test_arwen_adapter.py -q      (or python tests/test_arwen_adapter.py)
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

STATE = os.environ.get("NAMGPU_ADAPTER_STATE", "")
NMMB_FIELDS = ("T", "Q", "TRACERS", "CW", "F_ICE", "F_RAIN", "F_RIMEF", "U", "V", "W", "PD", "FIS")


def _load():
    if not STATE or not os.path.exists(STATE):
        pytest.skip("NAMGPU_ADAPTER_STATE not set (write one with tools/adapter_dump_state.py)")
    cp = pytest.importorskip("cupy")
    if os.environ.get("NAMGPU_ADAPTER_MANAGED") == "1" and not getattr(_load, "_managed", False):
        # a shared card: CUDA managed memory spills to host RAM (checks unchanged; timings then are not the card's)
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        _load._managed = True
    z = np.load(STATE)
    win = os.environ.get("NAMGPU_ADAPTER_WINDOW", "")
    sj = si = slice(None)
    if win:
        a, b = win.split(",")
        sj = slice(*map(int, a.split(":")))
        si = slice(*map(int, b.split(":")))
    fields = {}
    for k in z.files:
        if k.startswith(("vc_", "indx_", "meta")):
            continue
        x = z[k]
        x = x[..., sj, si] if x.ndim >= 2 else x
        fields[k] = cp.asarray(np.ascontiguousarray(x))
    indx = dict(zip([str(s) for s in z["indx_names"]], [int(v) for v in z["indx_values"]]))
    meta = [str(s) for s in z["meta"]]
    pt, pdtop = (float(v) for v in z["vc_pt_pdtop"])
    lm, jm, im = fields["T"].shape
    sg1, sg2 = z["vc_sg1"], z["vc_sg2"]
    grid = SimpleNamespace(im=im, jm=jm, lm=lm, pt=pt, pdtop=pdtop, sg2=sg2,
                           psg1=(sg1 * np.float32(pdtop) + np.float32(pt)).astype(np.float32),
                           sgml2=z["vc_sgml2"],
                           psgml1=(z["vc_sgml1"] * np.float32(pdtop) + np.float32(pt)).astype(np.float32))
    # Q is the INDX_Q slot of TRACERS in the real state (one allocation); keep that alias here
    iq = indx["INDX_Q"] - 1
    assert bool((fields["TRACERS"][iq] == fields["Q"]).all()), "dumped Q is not TRACERS[INDX_Q]"
    fields["Q"] = fields["TRACERS"][iq]
    cfg = SimpleNamespace(microphysics=meta[0], raw={}, dt=float(meta[3]), nphs=int(meta[4]),
                          nprecip=int(meta[5]), nrads=int(meta[6]), nradl=int(meta[7]))
    ctx = SimpleNamespace(st=_St(fields), grid=grid, cfg=cfg, tables={"indx": indx})
    return cp, ctx, meta


class _St:
    def __init__(self, fields):
        self.fields = fields

    def __getitem__(self, k):
        return self.fields[k]


def _snapshot(cp, ctx):
    return {k: ctx.st.fields[k].copy() for k in NMMB_FIELDS if k in ctx.st.fields}


def _diff_words(cp, ctx, snap):
    out = {}
    for k, v in snap.items():
        a = ctx.st.fields[k].view(np.uint32)
        out[k] = int((a != v.view(np.uint32)).sum())
    return out


def _all(ctx):
    from namgpu.physics import arwen_adapter as ad
    carried = [s for s in ad.SPECIES if ad._slots(ctx)[s] >= 0]
    read = list(ad.BITS)
    read = [f for f in read if f not in ad.SPECIES or f in carried]
    write = ["th", "t", "qv", *carried, "u", "v"]
    return ad.ArwenColumn(ctx, read=read, write=write, name="test"), carried


def test_round_trip_bitwise():
    cp, ctx, _ = _load()
    col, _ = _all(ctx)
    snap = _snapshot(cp, ctx)
    col.gather()
    col.scatter()
    d = _diff_words(cp, ctx, snap)
    assert all(v == 0 for v in d.values()), d
    zeros = {f: cp.zeros_like(col.a[f]) for f in col.write}
    col.gather()
    col.apply_tendencies(25.0, **zeros)
    d = _diff_words(cp, ctx, snap)
    assert all(v == 0 for v in d.values()), d


def test_views_are_the_flipped_column():
    cp, ctx, _ = _load()
    col, carried = _all(ctx)
    col.gather()
    st = ctx.st.fields
    T = st["T"]
    assert bool((col.t == T[::-1]).all())
    q = st["Q"][::-1]
    assert bool(cp.allclose(col.qv, q / (1 - q), rtol=2e-7, atol=0))
    for f in col.read:
        assert bool(cp.isfinite(col.a[f]).all()), f
    assert bool((col.p8w[:-1] > col.p).all()) and bool((col.p > col.p8w[1:]).all())
    assert bool((col.z8w[1:] > col.z8w[:-1]).all())
    assert bool(cp.allclose(col.z8w[0], st["FIS"] / np.float32(9.8060226), rtol=0, atol=1e-3))
    assert bool(cp.allclose(col.z8w[1:] - col.z8w[:-1], col.dz8w, rtol=1e-5, atol=1e-3))
    assert bool((col.p > 0).all()) and bool((col.pii > 0).all()) and bool((col.rho > 0).all())
    for s in carried:
        from namgpu.physics import arwen_adapter as ad
        slot = ad._slots(ctx)[s]
        assert bool((col.a[s] == st["TRACERS"][slot][::-1]).all()), s


def test_changes_go_back_only_where_made():
    cp, ctx, _ = _load()
    from namgpu.physics import arwen_adapter as ad
    col, carried = _all(ctx)
    lm, jm, im = col.lm, col.jm, col.im
    snap = _snapshot(cp, ctx)
    jc, ic = jm // 2, im // 2
    # 1. theta + 1 K in one column
    col.gather()
    col.th[:, jc, ic] += np.float32(1.0)
    col.scatter()
    dT = ctx.st.fields["T"] - snap["T"]
    assert int((dT != 0).sum()) == lm
    assert bool(cp.allclose(dT[::-1, jc, ic], col.pii[:, jc, ic], rtol=1e-4))
    d = _diff_words(cp, ctx, snap)
    assert all(v == 0 for k, v in d.items() if k != "T"), d
    ctx.st.fields["T"][...] = snap["T"]
    # 2. condensate change in one cell re-couples CW/F_ICE/F_RAIN there only
    if "qc" in carried and "CW" in ctx.st.fields:
        col.gather()
        col.qc[0, jc, ic] += np.float32(1.0e-4)
        col.scatter()
        d = _diff_words(cp, ctx, snap)
        assert d["TRACERS"] == 1 and d["CW"] == 1 and d["T"] == 0 and d["U"] == 0, d
        l0 = lm - 1
        slots = ad._slots(ctx)
        tr = ctx.st.fields["TRACERS"]
        q = {s: float(tr[slots[s], l0, jc, ic]) if slots[s] >= 0 else 0.0 for s in ("qc", "qr", "qi", "qs", "qg")}
        cw = q["qc"] + q["qr"] + q["qi"] + q["qs"] + (0.0 if col.fer_mic else q["qg"])
        assert abs(float(ctx.st.fields["CW"][l0, jc, ic]) - cw) <= 1e-6 * max(cw, 1e-12)
        for k in ("TRACERS", "CW", "F_ICE", "F_RAIN"):
            ctx.st.fields[k][...] = snap[k]
    # 3. +1 m/s mass-point u in the interior
    col.gather()
    col.u[:, 1:jm - 1, 1:im - 1] += np.float32(1.0)
    col.scatter()
    dU = ctx.st.fields["U"] - snap["U"]
    inner = dU[:, 1:jm - 2, 1:im - 2]          # V points whose four H neighbours are all interior
    assert bool(cp.allclose(inner, 1.0, atol=2e-5)), float(cp.abs(inner - 1).max())
    assert _diff_words(cp, ctx, snap)["V"] == 0
    # 4. the outer H row is never written (T, Q)
    col.gather()
    col.th += np.float32(1.0)
    col.qv *= np.float32(1.01)
    col.scatter()
    for k in ("T", "Q"):
        a, b = ctx.st.fields[k], snap[k]
        for edge in (a[:, 0, :] != b[:, 0, :], a[:, -1, :] != b[:, -1, :],
                     a[:, :, 0] != b[:, :, 0], a[:, :, -1] != b[:, :, -1]):
            assert int(edge.sum()) == 0, k
    for k in ("T", "TRACERS", "U"):
        ctx.st.fields[k][...] = snap[k]


def test_clock():
    from namgpu.physics.arwen_adapter import PhysicsClock
    cfg = SimpleNamespace(dt=25.0, nphs=2, nprecip=2, nrads=48, nradl=48)
    c = PhysicsClock(cfg, every_s={"cu": 300})
    assert c.due("pbl", 4) and not c.due("pbl", 5) and c.dt_phys("pbl") == 50.0
    assert c.n["cu"] == 12 and c.dt_phys("sw") == 1200.0
    assert c.substeps("mp", 20.0) == (3, float(np.float32(50.0 / 3)))


def receipt() -> dict:
    """Run the checks once and time a full gather+scatter (the per-call cost a scheme lane adds)."""
    cp, ctx, meta = _load()
    for t in (test_round_trip_bitwise, test_views_are_the_flipped_column, test_changes_go_back_only_where_made,
              test_clock):
        t()
    col, carried = _all(ctx)
    cp.cuda.Device().synchronize()
    ev = [cp.cuda.Event() for _ in range(3)]
    col.gather()
    col.scatter()
    ev[0].record()
    for _ in range(10):
        col.gather()
    ev[1].record()
    for _ in range(10):
        col.scatter()
    ev[2].record()
    ev[2].synchronize()
    lm, jm, im = col.lm, col.jm, col.im
    return {"state": os.path.basename(STATE), "meta": meta, "grid": [lm, jm, im], "species": carried,
            "all_checks": "pass", "gather_ms_all_fields": cp.cuda.get_elapsed_time(ev[0], ev[1]) / 10,
            "scatter_ms_all_writable": cp.cuda.get_elapsed_time(ev[1], ev[2]) / 10,
            "adapter_bytes_all_fields": col.nbytes(), "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


if __name__ == "__main__":
    print(json.dumps(receipt(), indent=1))
