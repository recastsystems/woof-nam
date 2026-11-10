"""The step-loop host-bound defects (profile, 2026-11-09), one test each.

The 12 km fit path ran radiation in 139 row chunks of 8 rows per call, each chunk a set of 7,632-column launches with
two blocking device-to-host reads (phys-rrtm-cld-init check_status from progcld2, and swrad's SW_NDAY), so the main
thread spun a core and the card sat on a few dozen SMs. The fix batches consecutive row windows into one launch set
(rrtm_call.batches) while every column keeps the SW MINVAL(laytrop) groups of its own window (sw_seam.window_kmins),
counts the SW day columns once per call (rrtm_call.day_rows), reads llyr/iovr once per run, and moves the history
encode to writer processes. Each test below fails on the old loop:

1. test_swrad_chunk_has_no_host_sync: one SW chunk captured into a CUDA graph (a host sync inside the capture
   invalidates it). Old: swrad read SW_NDAY back per chunk (ndayp.get()).
2. test_batched_windows_match_single_windows: the radiation SW chain over 4 row windows, run window by window (the old
   loop) and as batches of 3 and of 4 windows: every output bit-identical. Old: no batches; a 3-window chunk forms the
   taumol19/23/25/28 groups over its whole day list and differs.
3. test_progcld2_reads_cld_init_once: progcld2 read llyr/iovr from the device every chunk.
4. test_timing_receipt_keeps_every_stamp: the stamp buffer was full after the 300 start-up filter steps and every
   forecast-hour stamp was dropped.
5. test_history_writer_process_bytes: the process writer writes byte-identical files (CPU only).

Run: python -m pytest tests/test_steploop_hostbound.py  (tests 1, 2, 4 need CuPy and a card; a few hundred MB, seconds)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

try:
    import cupy as cp  # noqa: E402
    HAVE_CARD = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    cp = None
    HAVE_CARD = False

needs_card = pytest.mark.skipif(not HAVE_CARD, reason="needs CuPy and a CUDA card")

LM = 20                 # model layers (LMK = 21 with the extra radiation top layer)
IM = 40
JM = 26                 # rrtm_chunk.windows(26, 8): 4 windows, the last one shifted back
ROWS = 8


class _State:
    """The parts of namgpu.state.State the radiation chunk views use."""

    def __init__(self):
        self.fields, self.specs, self.scalars = {}, {}, {}
        self.xp = cp

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(tuple(int(s) for s in shape), dtype=dtype)
        return self.fields[name]

    def __getitem__(self, name):
        return self.fields[name]

    def __getattr__(self, name):
        f = self.__dict__.get("fields", {})
        if name in f:
            return f[name]
        raise AttributeError(name)


def _parent(seed: int = 3):
    """A parent context holding plausible SW inputs on a JM x IM domain: surface pressure 600-1050 hPa on sigma
    levels, so setcoef's laytrop varies column to column (the MINVAL groups matter), a night band, cloud and aerosol."""
    from namgpu.physics.rrtm import phys_rrtm_sw_swrad as swr
    rng = np.random.default_rng(seed)
    lmk = LM + swr.LTP
    st = _State()
    f = lambda *s: rng.random(s, dtype=np.float32)   # noqa: E731
    psfc = 600.0 + 450.0 * f(JM, IM)
    eta = (np.arange(lmk + 1, dtype=np.float32) / lmk) ** 1.6
    plvl = (0.5 + eta[:, None, None] * (psfc[None] - 0.5)).astype(np.float32)          # top first
    st.fields["RAD_PLVL"] = cp.asarray(plvl)
    st.fields["RAD_PLYR"] = cp.asarray(0.5 * (plvl[:-1] + plvl[1:]))
    st.fields["RAD_TLYR"] = cp.asarray(200.0 + 95.0 * f(lmk, JM, IM))
    st.fields["RAD_QLYR"] = cp.asarray(1e-6 + 1.5e-2 * f(lmk, JM, IM))
    st.fields["RAD_OLYR"] = cp.asarray(1e-7 + 9e-6 * f(lmk, JM, IM))
    st.fields["GASVMR"] = cp.asarray(1e-9 + 4e-4 * f(swr.NF_GAS, lmk, JM, IM))
    clouds = f(swr.NF_CLDS, lmk, JM, IM)
    clouds[0] = np.where(clouds[0] > 0.6, clouds[0], 0.0)                             # cloud fraction, clear areas
    clouds[2] = 5.0 + 25.0 * clouds[2]
    clouds[4] = 20.0 + 60.0 * clouds[4]
    for c in (1, 3, 5):
        clouds[c] = 50.0 * clouds[c]
    st.fields["RAD_CLOUDS"] = cp.asarray(clouds)
    st.fields["SFCALB"] = cp.asarray(0.05 + 0.55 * f(4, JM, IM))
    cz = -0.3 + 1.3 * f(JM, IM)
    cz[:, : IM // 4] = -0.5                                                             # night columns
    st.fields["SW_COSZEN"] = cp.asarray(cz)
    for name, shp in (("HTSWC", (lmk, JM, IM)), ("TOPFLX", (3, JM, IM)), ("SFCFLX", (4, JM, IM)),
                      ("FDNCMP", (6, JM, IM))):
        st.add(name, shp)
    tables = {swr.FAERSW_TABLE: cp.asarray(0.5 * f(3 * swr.NBDSW, lmk, JM, IM))}
    grid = SimpleNamespace(im=IM, jm=JM, lm=LM)
    cfg = SimpleNamespace(lfdncmp=True, lhsw0=False, lflxprf=False, solcon=1368.22)
    return SimpleNamespace(cfg=cfg, grid=grid, st=st, tables=tables, stream=cp.cuda.Stream(non_blocking=True))


def _day_list(cc):
    """grrad_sw step 3 on the chunk: SW_DAY, SW_CSUM, SW_IDXDAY, SW_NDAY from the chunk's SW_COSZEN."""
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_grrad_sw as gsw
    st = cc.st
    im, jm = int(cc.grid.im), int(cc.grid.jm)
    n = im * jm
    for name, shp in (("SW_DAY", (n,)), ("SW_CSUM", (n,)), ("SW_IDXDAY", (n,)), ("SW_NDAY", (1,))):
        st.add(name, shp, dtype="int32")
    with cc.stream:
        kernels.get(gsw.UNIT, "grrad_sw_daymask")(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
                                                  (im, jm, st["SW_COSZEN"].reshape(-1), st["SW_DAY"]))
        cp.cumsum(st["SW_DAY"], dtype=cp.int32, out=st["SW_CSUM"])
        kernels.get(gsw.UNIT, "grrad_sw_scatter")(((n + kernels.BLOCK_COL[0] - 1) // kernels.BLOCK_COL[0],),
                                                  kernels.BLOCK_COL,
                                                  (n, st["SW_DAY"], st["SW_CSUM"], st["SW_IDXDAY"], st["SW_NDAY"]))
    return int((cp.asnumpy(st["SW_COSZEN"]) >= np.float32(1.0e-4)).sum())


def _run_sw(ctx, b: int):
    """The rrtm-call chunk loop for the SW part: batches of b windows of ROWS rows (b = 1: the old loop)."""
    from namgpu.physics.rrtm import phys_rrtm_sw_swrad as swr
    from namgpu.physics.rrtm import rrtm_call, rrtm_chunk
    plan = rrtm_call.batches(JM, ROWS, b)
    geo = rrtm_call._device_geometry(ctx, JM, ROWS, b, plan) if b > 1 else None
    cc = rrtm_chunk.ChunkCtx(ctx, plan[0][1] - plan[0][0])
    with ctx.stream:                     # the driver runs every unit with ctx.stream current (driver.py)
        for k, (w0, w1, a, bb, _wins) in enumerate(plan):
            cc.begin(w0, w1)
            nday = _day_list(cc)
            if geo is not None:
                cc.tables[rrtm_call.BATCH_KEY] = geo[k]
            cc.tables[rrtm_call.NDAY_KEY] = nday
            swr.run(cc)
            cc.st.writeback(a, bb)
    ctx.stream.synchronize()
    return {n: cp.asnumpy(ctx.st[n]) for n in ("HTSWC", "TOPFLX", "SFCFLX", "FDNCMP")}


@needs_card
def test_batched_windows_match_single_windows():
    ref = _run_sw(_parent(), 1)
    assert np.abs(ref["HTSWC"]).max() > 0 and np.isfinite(ref["HTSWC"]).all()
    for b in (3, 4):                     # 3: two batches, the second reaching back over owned rows; 4: one batch
        got = _run_sw(_parent(), b)
        for n in ref:
            assert np.array_equal(ref[n].view(np.uint32), got[n].view(np.uint32)), (b, n)
    # the groups really differ between the windows and one plain chunk over the whole domain: without
    # window_kmins (no batch geometry) a single 26-row chunk does not give the window-by-window answer
    from namgpu.physics.rrtm import phys_rrtm_sw_swrad as swr
    from namgpu.physics.rrtm import rrtm_call, rrtm_chunk
    ctx = _parent()
    cc = rrtm_chunk.ChunkCtx(ctx, JM)
    with ctx.stream:
        cc.begin(0, JM)
        cc.tables[rrtm_call.NDAY_KEY] = _day_list(cc)
        swr.run(cc)
        cc.st.writeback(0, JM)
    ctx.stream.synchronize()
    plain = cp.asnumpy(ctx.st["HTSWC"])
    assert not np.array_equal(plain.view(np.uint32), ref["HTSWC"].view(np.uint32)), \
        "test domain does not exercise the MINVAL(laytrop) groups"


@needs_card
def test_swrad_chunk_has_no_host_sync():
    from namgpu.physics.rrtm import phys_rrtm_sw_swrad as swr
    from namgpu.physics.rrtm import rrtm_call, rrtm_chunk
    ctx = _parent()
    plan = rrtm_call.batches(JM, ROWS, 4)
    geo = rrtm_call._device_geometry(ctx, JM, ROWS, 4, plan)
    cc = rrtm_chunk.ChunkCtx(ctx, JM)
    s = ctx.stream
    with s:
        cc.begin(0, JM)
        nday = _day_list(cc)
        cc.tables[rrtm_call.BATCH_KEY] = geo[0]
        cc.tables[rrtm_call.NDAY_KEY] = nday
        swr.run(cc)                      # warm: kernels compiled, k-tables uploaded
        s.synchronize()
        s.begin_capture(mode=cp.cuda.runtime.streamCaptureModeRelaxed)
        try:
            swr.run(cc)                  # any device->host read or stream sync here invalidates the capture
        finally:
            graph = s.end_capture()
    graph.launch(stream=s)
    s.synchronize()


def test_progcld2_reads_cld_init_once(monkeypatch):
    from namgpu.physics.rrtm import phys_rrtm_cld_init as cld_init
    from namgpu.physics.rrtm import phys_rrtm_progcld2 as progcld2
    reads, calls = [], []
    monkeypatch.setattr(cld_init, "check_status", lambda ctx: reads.append(1) or {"llyr": 2, "iovr": 1})
    monkeypatch.setattr(progcld2, "run", lambda ctx, inputs, **kw: calls.append(kw) or {})
    ctx = SimpleNamespace(tables={cld_init.KEY: {"out": None}}, cfg=SimpleNamespace(ivflip=0))
    for _ in range(139):                 # one radiation call of the 12 km fit path: 139 chunks
        progcld2.dep("progcld2", ctx, {"cld_fraction": 0})
    assert len(calls) == 139 and calls[-1]["llyr"] == 2 and calls[-1]["iovr"] == 1
    assert len(reads) == 1, f"{len(reads)} device reads of llyr/iovr for one radiation call"


@needs_card
def test_timing_receipt_keeps_every_stamp(monkeypatch):
    from namgpu.steps import drv_timing_receipt as timing
    monkeypatch.setattr(timing, "CAPACITY", 16)
    ctx = SimpleNamespace(timings={}, tables={}, stream=cp.cuda.Stream(non_blocking=True))
    for i in range(50):                  # 50 stamps between two drains with 16-slot blocks
        timing.stamp(ctx, f"s{i}")
    rec = timing.receipt(ctx)
    assert rec["stamped"] == 50 and rec["dropped"] == 0 and rec["complete"]
    dev = timing._rec(ctx)["device_ns"]
    assert all(v is not None and v > 0 for v in dev) and dev == sorted(dev)


def test_history_writer_process_bytes(tmp_path):
    pytest.importorskip("netCDF4")
    from datetime import datetime
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from namgpu.io import history
    rng = np.random.default_rng(5)
    jm, im, lm = 12, 17, 6

    def staged():
        ev = SimpleNamespace(synchronize=lambda: None)
        fields = [("T2", rng.random((jm, im), dtype=np.float32), ("south_north", "west_east")),
                  ("T", rng.random((lm, jm, im), dtype=np.float32), ("bottom_top", "south_north", "west_east")),
                  ("PH", rng.random((lm + 1, jm, im), dtype=np.float32),
                   ("bottom_top_stag", "south_north", "west_east"))]
        t0 = datetime(2026, 1, 23, 12)
        return {"event": ev, "keep": [], "staged": fields, "valid": t0, "start": t0,
                "xlat": rng.random((jm, im), dtype=np.float32), "xlon": rng.random((jm, im), dtype=np.float32),
                "xtime_min": 60.0, "dims": {"west_east": im, "south_north": jm, "bottom_top": lm},
                "cen_lat": 40.0, "cen_lon": -95.0, "pole_lat": 54.0, "pole_lon": -106.0, "dx": 12000.0, "dy": 12000.0}

    s = staged()
    s2 = {k: (list(v) if k == "staged" else v) for k, v in s.items()}
    a = history.stage_and_write(tmp_path / "thread.nc", s)
    shm, meta = history.finish_payload_shm(s2)
    try:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context(method)) as ex:
            b = ex.submit(history.write_from_shm, str(tmp_path / "proc.nc"), shm.name, meta).result()
    finally:
        shm.close()
        shm.unlink()
    assert Path(a).read_bytes() == Path(b).read_bytes()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
