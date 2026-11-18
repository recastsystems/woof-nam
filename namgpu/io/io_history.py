"""Write one history NetCDF per output hour (unit io-history, host-io mode).

Driver entry (namgpu/driver.py DIAGNOSTICS): ``("io-history", "history")``, gated by
``minutes_history = 60`` in the configfile, so ``run`` is called once per output hour.

Input: ``ctx.history_fields`` -- name -> (device array, wrf dims), built by unit
io-history-fields (its wrapper sets it on the ctx).  Output: ``<ctx.out_dir>/wrfout_d01_<valid>``.
Start time: ``ctx.start_time`` (datetime, set by the driver from the run's start date).
Output directory: ``ctx.out_dir`` (set by Driver.run; defaults to runs/<start> in the working folder).

No CUDA kernel and no device arithmetic.  The unit does two things per output hour:
  1. the device->host copy of every history field, queued on ctx.stream into pinned host buffers
     (history.stage_history, no host sync on the step path; the writer thread waits on its event);
  2. the NetCDF encode and file write (history.write_payload).  A background thread waits for the copies and
     moves them into one shared-memory block per hour; a pool of NAMGPU_HISTORY_WRITERS writer processes
     (default up to 4) encodes the files, one per process, so hours are written side by side.  The step loop
     waits only for the previous hour's pinned buffers to be copied out (about a second), and for a writer
     when all of them are busy.  Before (one zlib thread, at most one file in flight) the encode took about 40 s
     per 12 km hour on one core, as long as a whole forecast hour on an unshared card, so it would have become
     the critical path once the step loop got faster (profile, 2026-11-09).  NAMGPU_HISTORY_WRITERS=0
     keeps the single writer thread.  ``drain(ctx)`` waits for every file and re-raises any write error;
     Driver.run calls it before the receipt.  The files are byte-identical either way (same write_payload).
  3. FAST FRAMES. The step loop no longer waits for a free writer: files queue in shared memory, bounded by host memory
     (_shm_room: /dev/shm and MemAvailable, each with its reserve), and the step loop waits only when that is full.
     With a dense schedule (more than DENSE_PER_HOUR files a model hour) the pool takes every core but RESERVED_CORES.
     Breakage it prevents: with 5-minute output the four writers (33 s a 12 km products frame on one core) were all
     busy, the step loop waited on the oldest one and the card sat about 70 % idle (5.3-8.3 s of forecast a frame,
     2026-11-13). The run says in one line when it waits, and at the end how long
     it waited in all (summary_line).
Reference for the field list and attributes: io/history.py (module docstring).
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from .history import finish_payload_shm, stage_and_write, stage_history, write_from_shm

_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-history")
_PROCS: ProcessPoolExecutor | None = None
_PROCS_LOCK = threading.Lock()


DENSE_PER_HOUR = 2.0     # files a model hour above which a schedule is dense (FAST FRAMES)
RESERVED_CORES = 4       # left to the forecast's own threads and the site's map processes when dense
_DENSE = False


def writers() -> int:
    """Writer processes (NAMGPU_HISTORY_WRITERS; default min(4, cores / 4), at least 1; 0 = the writer thread; with a
    dense schedule, cores - RESERVED_CORES, at least that). cores = the ones this process may use
    (namgpu.cores.usable: affinity set and cgroup quota), not the host's."""
    v = os.environ.get("NAMGPU_HISTORY_WRITERS", "").strip()
    if v:
        return max(0, int(v))
    from namgpu import cores
    hourly = max(1, min(4, cores.usable() // 4))
    return max(hourly, cores.usable() - RESERVED_CORES) if _DENSE else hourly


def set_schedule(sch) -> None:
    """Size the pool for the run's output schedule (before its first file: the pool is made once)."""
    global _DENSE
    per_h = getattr(sch, "frames_per_hour", None)
    if _PROCS is None:
        _DENSE = bool(per_h is not None and per_h() > DENSE_PER_HOUR)


def _pool() -> ProcessPoolExecutor:
    global _PROCS
    with _PROCS_LOCK:
        if _PROCS is None:
            from namgpu import cores as _cores  # noqa: PLC0415
            # never fork the CUDA process: forkserver (or spawn) children import only numpy/netCDF4; the forkserver's
            # socket goes in a short folder when TMPDIR is long
            _PROCS = ProcessPoolExecutor(max_workers=writers(), mp_context=_cores.pool_context())
    return _PROCS


# Each hour's block is /dev/shm/woof-nam-hist-<pid>-<n> (namgpu/io/scratch.py: a later run removes the blocks of a
# process that has ended). Before a block is written, /dev/shm must hold it plus SHM_RESERVE_B: /dev/shm is a tmpfs,
# and a page written past its end is a bus error that kills the forecast with no message (the 0.2.1 candidate's
# arctic-front rerun, rc 135 at +3 h, with 43 GB of an earlier run's column views in a 62 GB /dev/shm). Short of room,
# the thread waits for this run's own blocks in flight to be written and freed; still short, it writes the hour
# itself from the pinned buffers (the single-writer route: same bytes, slower) and says so in one line.
SHM = "/dev/shm"
SHM_RESERVE_B = int(float(os.environ.get("NAMGPU_HISTORY_SHM_RESERVE_MIB", "2048")) * 1024 * 1024)
_LIVE: dict = {}                 # block name -> SharedMemory, while its file is written
_INFLIGHT: list = []             # the writer-process futures of those blocks
_SEQ = iter(range(1 << 62))


# FAST FRAMES: host memory the queue leaves free, as a share of MemTotal (the forecast, its pinned buffers and the
# site's map processes live there too).
MEM_RESERVE_SHARE = float(os.environ.get("NAMGPU_HISTORY_MEM_RESERVE", "0.25"))


def _meminfo() -> tuple[int, int]:
    """(MemAvailable, MemTotal) in bytes, or (0, 0) where /proc/meminfo cannot be read."""
    got = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("MemAvailable", "MemTotal"):
                got[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        return 0, 0
    return got.get("MemAvailable", 0), got.get("MemTotal", 0)


def _shm_room(need: int) -> tuple[bool, int]:
    """Room for one more queued block: /dev/shm holds it plus SHM_RESERVE_B, and host memory keeps
    MEM_RESERVE_SHARE of MemTotal free after it. Returns (room, the smaller of the two free figures)."""
    from .scratch import free_bytes  # noqa: PLC0415
    if not Path(SHM).is_dir():
        return True, 0           # no /dev/shm here (not Linux): multiprocessing places the block itself
    free = free_bytes(SHM)
    ok = free >= need + SHM_RESERVE_B
    avail, total = _meminfo()
    if total:
        ok = ok and avail >= need + int(MEM_RESERVE_SHARE * total)
        free = min(free, avail)
    return ok, free


def _unlink(name: str) -> None:
    shm = _LIVE.pop(name, None)
    if shm is None:
        return
    try:
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _stage_and_submit(path, staged):
    """Writer thread: copies out of the pinned buffers into shared memory, then hands the file to a process (or,
    when /dev/shm has no room for the block, writes the file here; see SHM_RESERVE_B)."""
    from .history import shm_bytes  # noqa: PLC0415
    need = shm_bytes(staged)
    ok, free = _shm_room(need)
    for f in ([] if ok else list(_INFLIGHT)):
        try:
            f.result()
        except BaseException:  # noqa: BLE001  (the file's own error is raised by drain/_ready_for_next)
            pass
        ok, free = _shm_room(need)
        if ok:
            break
    if not ok:
        print(f"io-history: {SHM} has {free / 1e9:.1f} GB free, under this hour's {need / 1e9:.1f} GB block plus "
              f"{SHM_RESERVE_B / 1e9:.1f} GB: {Path(path).name} is written by the writer thread, not through shared "
              "memory (slower)", flush=True)
        return stage_and_write(path, staged)
    from .scratch import own  # noqa: PLC0415
    own(SHM)                           # this process is alive for a sweep in any PID namespace (scratch.own)
    name = f"woof-nam-hist-{os.getpid()}-{next(_SEQ)}"
    shm, meta = finish_payload_shm(staged, name=name)
    _LIVE[name] = shm
    try:
        fut = _pool().submit(write_from_shm, str(path), shm.name, meta)
    except BaseException:
        _unlink(name)
        raise
    _INFLIGHT.append(fut)

    def _free(f, name=name):
        _unlink(name)
        try:
            _INFLIGHT.remove(f)
        except ValueError:
            pass
    fut.add_done_callback(_free)
    return fut


def _collect(ctx) -> None:
    """The staging thread's result: with writer processes, the file future joins ctx.history_files."""
    fut = getattr(ctx, "history_future", None)
    if fut is not None:
        ctx.history_future = None
        res = fut.result()
        if hasattr(res, "result"):
            files = getattr(ctx, "history_files", None)
            if files is None:
                files = ctx.history_files = []
            files.append(res)


def drain(ctx) -> None:
    """Wait for every history write in flight and re-raise the first error."""
    _collect(ctx)
    files = getattr(ctx, "history_files", None) or []
    ctx.history_files = []
    for f in files:
        f.result()


def shutdown() -> None:
    """Stop the writer processes and their forkserver (call after the last drain). The forecast entries end with
    os._exit, which skips the exit hooks of the executor and of multiprocessing; left running, the forkserver kept the
    stdout pipe of the forecast open and the nest door, reading that pipe, waited forever (a nest run, 2026-11-09)."""
    global _PROCS
    with _PROCS_LOCK:
        p, _PROCS = _PROCS, None
    from .scratch import disown_all  # noqa: PLC0415
    if p is None:
        for name in list(_LIVE):
            _unlink(name)
        disown_all()             # no block is left: this process's owner lock file goes too (scratch.own)
        return
    p.shutdown(wait=True)
    for name in list(_LIVE):     # a block whose writer died before its done-callback ran
        _unlink(name)
    disown_all()                 # no block is left: this process's owner lock file goes too (scratch.own)
    try:
        from multiprocessing import forkserver  # noqa: PLC0415
        stop = getattr(forkserver._forkserver, "_stop", None)
        if stop is not None:
            stop()
    except Exception:  # noqa: BLE001  (spawn start method: nothing to stop)
        pass


# FAST FRAMES: what the step loop waited on the history path (the previous file out of the pinned buffers into the
# queue, and the queue's room), per process; waits() reads it, summary_line() says it at the end of the run.
WAIT_SAY_S = 1.0
_WAITS = {"files": 0, "s": 0.0, "max_s": 0.0, "over": 0, "said": 0}


def waits() -> dict:
    return {"files": _WAITS["files"], "wait_s": round(_WAITS["s"], 2), "max_s": round(_WAITS["max_s"], 2),
            "over_1s": _WAITS["over"], "writers": writers()}


def _note_wait(took: float, queued: int, valid) -> None:
    _WAITS["files"] += 1
    _WAITS["s"] += took
    _WAITS["max_s"] = max(_WAITS["max_s"], took)
    if took > WAIT_SAY_S:
        _WAITS["over"] += 1
        if _WAITS["said"] < 3 or _WAITS["over"] % 25 == 0:
            _WAITS["said"] += 1
            print(f"io-history: the forecast waited {took:.1f} s on the history writers before "
                  f"{valid:%Y-%m-%d_%H:%M:%S} ({queued} files queued, {writers()} writers): host memory or the "
                  f"writers are short on this machine; the run goes on {took:.1f} s later", flush=True)


def summary_line() -> str:
    w = waits()
    return (f"io-history: {w['files']} files, {w['writers']} writers; the forecast waited {w['wait_s']:.1f} s on "
            f"the history path in all (largest {w['max_s']:.1f} s, {w['over_1s']} over {WAIT_SAY_S:g} s)")


def _ready_for_next(ctx, valid=None) -> None:
    """Before staging a file: the previous one is out of the pinned buffers and queued (with room in host memory:
    _stage_and_submit). No wait for a free writer (FAST FRAMES); NAMGPU_HISTORY_QUEUE caps the files queued."""
    t0 = time.monotonic()
    _collect(ctx)
    files = [f for f in (getattr(ctx, "history_files", None) or []) if not f.done() or f.exception()]
    for f in [f for f in files if f.done()]:
        f.result()                       # a failed file stops the run here, as the thread's error did
    cap = int(os.environ.get("NAMGPU_HISTORY_QUEUE", "0") or 0)
    while cap > 0 and len(files) >= cap:
        files.pop(0).result()
    ctx.history_files = files
    if valid is not None:
        _note_wait(time.monotonic() - t0, len(files), valid)


def run(ctx) -> Path:
    fields = getattr(ctx, "history_fields", None)
    if fields is None:
        raise RuntimeError("io-history: ctx.history_fields not set (unit io-history-fields must run first)")
    start = getattr(ctx, "start_time", None)
    if start is None:
        raise RuntimeError("io-history: ctx.start_time not set (driver must give the run start date)")
    if getattr(ctx, "out_dir", None) is None:
        raise RuntimeError("io-history: ctx.out_dir not set")
    from ..steps import drv_timing_receipt as timing  # noqa: PLC0415
    from .history_select import select  # noqa: PLC0415
    fields = select(ctx, fields)         # --history-fields / --full-every (history_select.py)
    if getattr(ctx, "slab", None) is not None:
        return _run_slab(ctx, fields, start, timing)
    if _PROCS is None:
        set_schedule(getattr(ctx, "outputs", None))   # FAST FRAMES: the pool's size, before its first file
    valid = start + timedelta(seconds=ctx.ntimestep * ctx.cfg.dt)
    _ready_for_next(ctx, valid)          # previous file copied out of the pinned buffers (reused) into the queue
    timing.stamp(ctx, "io-history/d2h")
    path = Path(ctx.out_dir) / f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"
    # async copies on ctx.stream into pinned host buffers; the writer thread waits on their event, so the
    # step path no longer syncs or copies (history.stage_history; was 2-65 s per hour with the GPU idle)
    staged = stage_history(ctx, fields, start)
    sch = getattr(ctx, "outputs", None)
    if sch is not None and getattr(sch, "catalog", None) is not None and sch.catalog(int(ctx.ntimestep)):
        staged["zlib"] = False           # a catalog frame (namgpu/outputs.py): the same fields, written raw
    timing.stamp(ctx, "io-history/submit")
    ctx.history_future = (_WRITER.submit(_stage_and_submit, path, staged) if writers() > 0
                          else _WRITER.submit(stage_and_write, path, staged))
    return path


# ---- several cards (MULTICARD-PLAN.md section 6) ------------------------------------------------------------------
# Each rank queues the async copy of its OWNED rows into pinned buffers on ctx.stream (no host sync on the step path);
# its writer thread then writes them, at their global rows, into one full-domain float32 file per field under
# $NAMGPU_HIST_SHM/<valid>/ (/dev/shm) and drops a done.r<rank> flag. A pool of writer processes started by
# tools/run_forecast.py --cards N (tools/history_merge.py) writes each complete hour as one NetCDF, one hour per
# writer, so no single zlib thread sits on the critical path. Rank 0 adds meta.json and the global XLAT/XLONG.


def _run_slab(ctx, fields, start, timing):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    import cupy as cp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from .history import _pinned  # noqa: PLC0415
    s = ctx.slab
    # a run of several domains (namgpu/nest/coupled.py) gives each domain its own folder and writer pool
    shm = getattr(ctx, "hist_shm", None) or os.environ.get("NAMGPU_HIST_SHM")
    if not shm:
        raise RuntimeError("io-history: several cards need NAMGPU_HIST_SHM (tools/run_forecast.py --cards N sets it)")
    drain(ctx)
    timing.stamp(ctx, "io-history/d2h")
    valid = start + timedelta(seconds=ctx.ntimestep * ctx.cfg.dt)
    tag = f"{valid:%Y-%m-%d_%H:%M:%S}"
    jm = int(ctx.grid.jm)
    stream = ctx.stream
    staged, keep = [], []
    from .io_history_fields import device_array  # noqa: PLC0415
    with stream:
        for name, (arr, dims) in fields.items():
            arr = device_array(arr)
            if arr.shape[-2] != jm:
                raise RuntimeError(f"io-history: {name} shape {arr.shape} has no local row axis of {jm}")
            if isinstance(arr, cp.ndarray):
                src = cp.ascontiguousarray(arr[..., s.own0:s.own0 + s.nown, :], dtype=cp.float32)
                dst = _pinned(f"slab:{getattr(ctx, 'hist_pin_key', '')}:{name}", src.shape)
                src.get(stream=stream, out=dst, blocking=False)
                keep.append(src)
            else:
                dst = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)[..., s.own0:s.own0 + s.nown, :])
            staged.append((name, dst, tuple(dims)))
        ev = cp.cuda.Event(block=True, disable_timing=True)
        ev.record(stream)
    meta = None
    if s.rank == 0:
        geo = ctx.hist_geo_global
        meta = {"valid": tag, "start": f"{start:%Y-%m-%d_%H:%M:%S}",
                "xtime_min": float(ctx.ntimestep * ctx.cfg.dt) / 60.0,
                "dims": {"west_east": int(ctx.grid.im), "south_north": int(s.jmg), "bottom_top": int(ctx.grid.lm)},
                "cen_lat": geo["cen_lat"], "cen_lon": geo["cen_lon"],
                "pole_lat": float(ctx.cfg.tph0d), "pole_lon": float(ctx.cfg.tlm0d),
                "dx": geo["dx"], "dy": float(ctx.grid.dyh), "nranks": int(s.nranks),
                "fields": [[n, list(d), list(h.shape[:-2])] for n, h, d in staged]}
    timing.stamp(ctx, "io-history/submit")
    ctx.history_future = _WRITER.submit(_write_slab, Path(shm) / tag, staged, ev, keep, s, meta,
                                        getattr(ctx, "hist_geo_global", None))
    _ = json
    return Path(shm) / tag


def _write_slab(hour, staged, ev, keep, s, meta, geo):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    ev.synchronize()
    del keep
    hour.mkdir(parents=True, exist_ok=True)
    from .history import LEVEL_DIMS  # noqa: PLC0415
    for name, h, _dims in staged:
        if h.ndim == 3 and _dims[0] in LEVEL_DIMS:
            h = h[::-1]                      # NMMB l=1 top -> WRF bottom-up, as finish_payload (soil layers stay)
        full = tuple(h.shape[:-2]) + (int(s.jmg), int(h.shape[-1]))
        path = hour / f"{name}.f32"
        nbytes = int(np.prod(full)) * 4
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size < nbytes:
                os.ftruncate(fd, nbytes)
        finally:
            os.close(fd)
        mm = np.memmap(path, dtype=np.float32, mode="r+", shape=full)
        mm[..., s.jts - 1:s.jte, :] = h
        del mm
    if meta is not None:
        for nm in ("xlat", "xlon"):
            np.ascontiguousarray(geo[nm], dtype=np.float32).tofile(hour / f"{nm}.f32")
        tmp = hour / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, hour / "meta.json")
    (hour / f"done.r{s.rank}").write_text("1")
    return hour
