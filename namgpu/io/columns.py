"""woof-nam columns: the any-point column store (gpuwm's rw_soundings --columns, `.wcol` per history hour).

    woof-nam columns runs/2026-11-04_18z                       (every wrfout_d01_* of a run -> <run>/columns/d01/)
    woof-nam columns runs/x/wrfout_d01_2026-11-04_19:00:00 --out runs/x/columns     (one hour, as it lands)

The store reader takes WRF's conventions, which the WOOF-NAM history does not use (io_history_fields.py): WRF's
C-grid U, V and W (one more point along x, y and z), pressure and geopotential as perturbation plus base (P + PB,
PH + PHB) and the grid-to-earth rotation (COSALPHA, SINALPHA). So each history file is first written as a WRF view
in a scratch folder (NAMGPU_SHM, default /dev/shm; <out>/.scratch on disk when that has no room), the store is
written from the view, and the view is deleted:
  - U, V, W: the history's earth-relative winds and W at mass points, averaged to the cell faces (the end faces
    take the end values); the reader averages them back to mass points, so a sounding's wind is a 1-2-1 average
    of three neighbouring columns along that component's axis (U along x, V along y, W in the vertical);
  - P = PRES (full pressure), PB = 0; PHB = 0 (PH is already the full geopotential);
  - COSALPHA = 1, SINALPHA = 0 (the winds are earth-relative already);
  - every other field as written.
The output is `<out>/d01/<YYYYMMDDTHHMMSSZ>.wcol`, the layout WOOF's worker reads (it takes a store the engine
wrote beside the history as it is). Frames run in parallel processes, one per frame as far as cores, host
memory and the scratch folder allow, with the cores shared out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FRAME = re.compile(r"^wrfout_(d\d\d)_(\d{4})-(\d\d)-(\d\d)_(\d\d):(\d\d):(\d\d)$")


def _faces(x, axis):
    """Mass-point values -> face values along `axis` (n -> n + 1): interior faces are the average of the two
    neighbours, the end faces take the end values."""
    import numpy as np
    axis %= x.ndim
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = n + 1
    y = np.empty(shape, x.dtype)

    def at(s):
        return tuple(s if i == axis else slice(None) for i in range(x.ndim))
    y[at(slice(1, n))] = 0.5 * (x[at(slice(0, n - 1))] + x[at(slice(1, n))])
    y[at(slice(0, 1))] = x[at(slice(0, 1))]
    y[at(slice(n, n + 1))] = x[at(slice(n - 1, n))]
    return y


STAGGER = {"U": ("west_east_stag", -1), "V": ("south_north_stag", -2), "W": ("bottom_top_stag", -3)}


def wrf_view(src: Path, dst: Path) -> Path:
    """Write the WRF-convention view of one WOOF-NAM history file (see the module docstring)."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst, "w", format="NETCDF4") as b:
        for k, d in a.dimensions.items():
            b.createDimension(k, None if d.isunlimited() else len(d))
        b.createDimension("west_east_stag", len(a.dimensions["west_east"]) + 1)
        b.createDimension("south_north_stag", len(a.dimensions["south_north"]) + 1)
        if "bottom_top_stag" not in b.dimensions:
            b.createDimension("bottom_top_stag", len(a.dimensions["bottom_top"]) + 1)
        b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})

        def put(name, dims, data, like=None):
            v = b.createVariable(name, data.dtype if hasattr(data, "dtype") else "f4", dims)
            if like is not None:
                v.setncatts({k: like.getncattr(k) for k in like.ncattrs() if k != "_FillValue"})
            v[:] = data
        for name, v in a.variables.items():
            dims = list(v.dimensions)
            data = v[:]
            if name in STAGGER and v.ndim == 4:
                dim, axis = STAGGER[name]
                dims[axis] = dim
                data = _faces(np.asarray(data), axis)
            put(name, tuple(dims), data, v)
        pres = a.variables["PRES"]
        put("P", pres.dimensions, np.asarray(pres[:]), pres)
        put("PB", pres.dimensions, np.zeros(pres.shape, np.float32))
        ph = a.variables["PH"]
        put("PHB", ph.dimensions, np.zeros(ph.shape, np.float32))
        hgt = a.variables["HGT"]
        put("COSALPHA", hgt.dimensions, np.ones(hgt.shape, np.float32))
        put("SINALPHA", hgt.dimensions, np.zeros(hgt.shape, np.float32))
    return dst


PREFIX = "woof-nam-columns-"


def _one(frame: str, out: str, scratch: str, threads: int, exe: str, owner: int | None = None,
         alt: str | None = None, need: int = 0) -> dict:
    """View + store for one history file, in its own process. The view folder is named after the parent
    (woof-nam-columns-<owner>-*), so the parent removes it whatever ends this worker. Before the view is written the
    scratch folder's room is read again: with less than `need` bytes free (the view's size) the view goes to `alt`
    (the disk scratch beside the store) instead, and the record says so."""
    from namgpu.io import scratch as scr
    t0 = time.perf_counter()
    name = Path(frame).name
    note = None
    if alt and need and scratch and scratch != alt and scr.free_bytes(scratch) < need:
        note = (f"{scratch} had {scr.free_bytes(scratch) / 1e9:.1f} GB free for a {need / 1e9:.1f} GB view: "
                f"view written in {alt}")
        scratch = alt
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{PREFIX}{owner if owner is not None else os.getppid()}-",
                                     dir=scratch) as tmp:
        view = wrf_view(Path(frame), Path(tmp) / name)
        t1 = time.perf_counter()
        req = Path(tmp) / "request.json"
        req.write_text(json.dumps({"sources": [{"path": str(view), "domain": FRAME.match(name)[1], "member": "det"}],
                                   "out": str(out), "tile": 16, "threads": threads}))
        r = subprocess.run([exe, "--columns", str(req)], capture_output=True, text=True)
    rec = {"frame": name, "rc": r.returncode, "view_s": round(t1 - t0, 2), "store_s": round(time.perf_counter() - t1, 2)}
    if note:
        rec["note"] = note
    for line in r.stdout.splitlines():
        if line.startswith("COLUMNS "):
            path, *fields = line[len("COLUMNS "):].rsplit(" ", 2)
            rec.update(path=path, **dict(f.split("=", 1) for f in fields))
    if r.returncode:
        rows = [x for x in r.stderr.splitlines() if x.strip()]
        rec["error"] = (rows[-1] if rows else f"exit {r.returncode}")[:300]
    return rec


def frames_of(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.glob("wrfout_d*") if FRAME.match(f.name))
        elif FRAME.match(p.name):
            out.append(p)
    return out


def _meminfo_available() -> int | None:
    """MemAvailable from /proc/meminfo in bytes (None where there is no /proc)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _cgroup_headroom(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> int | None:
    """Bytes this process's cgroup (v2) and each cgroup above it can still take before its memory.max: the smallest
    of max - (current - reclaimable file cache). Shared memory (/dev/shm, where the views are written) is charged
    and is not reclaimable, so it counts as used. None with no limit anywhere, or no cgroup v2."""
    try:
        rel = next((ln.split("::", 1)[1] for ln in Path(proc_cgroup).read_text().splitlines() if ln.startswith("0::")),
                   None)
    except OSError:
        return None
    if rel is None:
        return None
    best = None
    d = Path(root + rel.rstrip("/")) if rel.strip("/") else Path(root)
    while True:
        try:
            mx = (d / "memory.max").read_text().strip()
            if mx != "max":
                cur = int((d / "memory.current").read_text())
                stat = dict(ln.split() for ln in (d / "memory.stat").read_text().splitlines() if ln.count(" ") == 1)
                cache = max(0, int(stat.get("file", 0)) - int(stat.get("shmem", 0)))
                room = max(0, int(mx) - (cur - cache))
                best = room if best is None else min(best, room)
        except (OSError, ValueError):
            pass
        if d == Path(root) or Path(root) not in d.parents:
            break
        d = d.parent
    return best


def _mem_available() -> int | None:
    """Memory the workers can use: MemAvailable, or the room left under the cgroup's memory.max when that is
    smaller. Breakage it prevents: in a login session capped at 80 GiB on a 123 GB host the count followed the host's
    MemAvailable, and beside the run's GRIB2 follower the kernel killed two workers at the cap (BrokenProcessPool,
    no store; arctic-front 0.2.1 candidate run)."""
    vals = [v for v in (_meminfo_available(), _cgroup_headroom()) if v is not None]
    return min(vals) if vals else None


# The WRF view is the history uncompressed with U, V, W one point longer and P, PB, PHB, COSALPHA, SINALPHA added:
# 4.27 GB (view_bytes) for the 1.24 GB 12 km hour 0 of the arctic-front run (3.60 GB uncompressed; fix/shm-leftovers
# proof run, 2026-11-11). The 2.5-3.0 GB views left by the candidate's broken pool were cut short by the kill. Used
# only when the header cannot be read.
VIEW_RATIO = 3.4


def view_bytes(frames: list[Path]) -> int:
    """Bytes of the largest frame's WRF view, from its header (every variable uncompressed, the staggered winds one
    point longer, the five added fields), 2 % over for the file's own structure."""
    import numpy as np
    if not frames:
        return 0
    big = max(frames, key=lambda f: f.stat().st_size)
    try:
        import netCDF4
        with netCDF4.Dataset(big) as a:
            n = 0
            for name, v in a.variables.items():
                shape = list(v.shape)
                if name in STAGGER and v.ndim == 4:
                    shape[STAGGER[name][1]] += 1
                try:
                    item = np.dtype(v.dtype).itemsize
                except TypeError:
                    item = 1
                n += int(np.prod(shape)) * item
            for src, k in (("PRES", 2), ("PH", 1), ("HGT", 2)):    # P + PB, PHB, COSALPHA + SINALPHA
                if src in a.variables:
                    n += k * int(np.prod(a.variables[src].shape)) * 4
        return int(n * 1.02) + (1 << 20)
    except Exception:  # noqa: BLE001 - no netCDF4, or not a netCDF file
        return int(VIEW_RATIO * big.stat().st_size)


def auto_workers(frames: list[Path], cores: int, scratch: str | None, mem_available: int | None = None,
                 scratch_free: int | None = None, view: int | None = None) -> int:
    """Frames written at once: one per frame, bounded by the cores, by memory and by the room on the scratch folder.
    A worker holds its WRF view on scratch and about as much again in its own memory while it builds the view (a
    worker the kernel killed at the session's memory.max had 2.84 GB anon beside a view still being written,
    arctic-front 0.2.1 candidate, 2026-11-11 08:47Z). So a worker counts one view of memory, two when the scratch is tmpfs (its
    pages are memory, charged to the same cgroup), and one view of the scratch folder's FREE room (free space, less
    what NAMGPU_SHM_CAP_MIB leaves the package). Breakage it prevents: the bound counted three compressed history
    files of memory and one compressed file of scratch, so 16 workers wrote 16 views (45 GB of /dev/shm) beside the
    GRIB2 follower, the kernel killed two at the session's 80 GiB cap, and the store ended in BrokenProcessPool.
    Before any bound the count was fixed at 4, and the 13-frame store of a 7.6 km run took 281 s on 1 to 4 cores of
    a 96-core host (release check 2026-11-09)."""
    from namgpu.io import scratch as scr
    if not frames:
        return 1
    view = max(1, view if view is not None else view_bytes(frames))
    n = min(len(frames), max(1, cores))
    ram = bool(scratch) and scr.in_ram(scratch)
    avail = _mem_available() if mem_available is None else mem_available
    if avail:
        n = min(n, max(1, int(avail * 0.6) // ((2 if ram else 1) * view)))
    if scratch_free is None and scratch:
        scratch_free = scr.free_bytes(scratch)
    if scratch_free is not None and scratch:
        n = min(n, max(1, int(scratch_free * 0.9) // view))
    return max(1, n)


def _pool_context():
    """The workers' start method: forkserver (spawn where there is none), on every Python. A forked worker inherits
    this process's SIGTERM handler, which turns the pool's own terminate() into an exception the worker catches and
    outlives, so a pool broken by a killed worker never finished shutting down; it also inherits both ends of the
    pool's pipes and the state of every thread's locks. The forkserver's socket goes in a short folder when TMPDIR is
    long (namgpu.cores.pool_context)."""
    from namgpu import cores as _cores  # noqa: PLC0415
    return _cores.pool_context()


def _kill_pool(ex) -> None:
    """Stop a pool's worker processes now (a signal or an error in this process): shutdown() would wait for them."""
    for p in list((getattr(ex, "_processes", None) or {}).values()):
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    ex.shutdown(wait=False, cancel_futures=True)


def write(frames: list[Path], out: Path, workers: int | None = None, scratch: str | None = None,
          exe: str | None = None, log=None) -> dict:
    """The store for `frames` under `out` (<out>/<domain>/<stamp>.wcol). Returns {"files", "seconds", "failed"}.

    Scratch: the views go to `scratch` (default NAMGPU_SHM, else /dev/shm) while it has room for them, else to the
    disk folder <out>/.scratch (one line says so). Whatever ends a worker or this process (the OOM killer, a broken
    pool, an error, SIGTERM), every view folder of this run is removed before write returns or raises: the folders
    are named after this process (woof-nam-columns-<pid>-*), and a later run removes those of a process that has
    ended (scratch.sweep). A pool that breaks is run again for the frames it had not finished, with half the workers,
    down to one."""
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool

    from namgpu.io import scratch as scr
    log = log or (lambda m: print(m, flush=True))
    if exe is None:
        from gpuwm.soundings import binary
        exe = str(binary())
    t0 = time.perf_counter()
    # the cores this process may use (affinity set and cgroup quota), not the host's: under `taskset -c 0-11` on a
    # 24-core host the store planned its workers and their threads for 24 cores (2026-11-11)
    from namgpu import cores as coremod
    cores = coremod.usable()
    out.mkdir(parents=True, exist_ok=True)
    alt = str(out / ".scratch")
    if scratch is None:
        r = scr.root()
        scratch = str(r) if r is not None else None
    for d in dict.fromkeys(x for x in (scratch, alt) if x and Path(x).is_dir()):
        scr.sweep(d, log=log)
    view = view_bytes(frames)
    if scratch and frames and scr.free_bytes(scratch) < view * 1.05:
        log(f"columns: {scratch} has {scr.free_bytes(scratch) / 1e9:.1f} GB free, under one "
            f"{view / 1e9:.1f} GB view: the views go to {alt} (disk, slower)")
        scratch = alt
    workers = max(1, min(workers or auto_workers(frames, cores, scratch, view=view), len(frames) or 1))
    threads = max(1, cores // workers)
    owner = os.getpid()
    owned_scratch = scratch
    if owned_scratch:
        scr.own(owned_scratch)         # the views carry this id: alive for a sweep in any PID namespace
    recs: dict = {}
    todo = list(frames)

    def clean():
        for d in dict.fromkeys(x for x in (scratch, alt) if x):
            scr.remove_own(d, PREFIX, owner)
        try:
            Path(alt).rmdir()                 # only when empty
        except OSError:
            pass

    import signal
    import threading
    old_term = None
    if threading.current_thread() is threading.main_thread():
        def _term(signum, _frame):
            raise SystemExit(128 + signum)
        old_term = signal.signal(signal.SIGTERM, _term)
    try:
        while todo:
            broken = False
            ex = ProcessPoolExecutor(workers, mp_context=_pool_context())
            try:
                futs = {ex.submit(_one, str(f), str(out), scratch, threads, exe, owner, alt, view): f for f in todo}
                for fut in as_completed(futs):
                    try:
                        recs[futs[fut]] = fut.result()
                    except BrokenProcessPool:
                        broken = True
            except BaseException:
                _kill_pool(ex)
                raise
            ex.shutdown(wait=True)
            clean()                           # a broken pool's other workers were terminated inside their views
            todo = [f for f in todo if f not in recs]
            if not broken or not todo:
                break
            if workers == 1:
                for f in todo:
                    recs[f] = {"frame": f.name, "rc": -9, "view_s": 0.0, "store_s": 0.0,
                               "error": "its worker process ended abruptly with one worker left (killed: memory?)"}
                break
            workers = max(1, workers // 2)
            threads = max(1, cores // workers)
            log(f"columns: a worker process ended abruptly (the kernel's OOM killer or a signal); the pool's views "
                f"are removed and {len(todo)} frames are written again with {workers} workers")
    finally:
        clean()
        if owned_scratch:
            scr.disown(owned_scratch)         # the views are gone: nothing of this store stays in the scratch folder
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
    for r in recs.values():
        if r.get("note"):
            log(f"columns: {r['frame']}: {r['note']}")
    recs_l = [recs[f] for f in frames if f in recs]
    failed = [r for r in recs_l if r["rc"]]
    return {"files": recs_l, "seconds": round(time.perf_counter() - t0, 2), "failed": len(failed),
            "parallelism": f"{workers} processes x {threads} threads"}


def main(argv=None) -> int:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Write the any-point column store (gpuwm rw_soundings, one .wcol file per history hour) for a "
                    "run folder or for single history files. A sounding at any point and hour is then read from the "
                    "store (gpuwm soundings). Runs on all cores; no GPU.",
        epilog="examples:\n  woof-nam columns runs/2025-03-14_12z\n"
               "  woof-nam columns runs/x/wrfout_d01_2025-03-14_13:00:00 --out runs/x/columns",
        formatter_class=HelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="a run folder (every wrfout_d01_*) or history files")
    ap.add_argument("--out", type=Path, default=None, help="store folder (default <run>/columns)")
    ap.add_argument("--workers", type=int, default=None,
                    help="frames written at once (default: one per frame, bounded by cores, host memory and the "
                         "scratch folder)")
    a = ap.parse_args(argv)
    frames = frames_of(a.paths)
    if not frames:
        ap.error("no wrfout_dNN_YYYY-MM-DD_HH:MM:SS history files in " + ", ".join(map(str, a.paths)))
    out = a.out or ((a.paths[0] if a.paths[0].is_dir() else a.paths[0].parent) / "columns")
    res = write(frames, out, a.workers)
    for r in res["files"]:
        print(f"{r['frame']}: " + (f"{r.get('path')} {r.get('bytes')} bytes" if not r["rc"] else f"FAILED {r['error']}")
              + f" (view {r['view_s']} s, store {r['store_s']} s)", flush=True)
    print(f"columns: {len(frames) - res['failed']} of {len(frames)} in {res['seconds']} s ({res['parallelism']}) "
          f"-> {out}", flush=True)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
