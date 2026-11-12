"""Host-RAM scratch (/dev/shm): the room it really has, and the leftovers of runs that ended.

The package writes four kinds of entry in /dev/shm, each named with the id of the process that owns it:
  woof-nam-columns-<pid>-*   the column store's WRF views (namgpu/io/columns.py), one history hour uncompressed each
  woof-nam-hist-<pid>-<n>    the history writer's shared-memory block of one hour (namgpu/io/io_history.py)
  woof-nam-frame-<pid>-*     a products hand-off frame's regridded fields (namgpu/io/products.py)
  namgpu-hist-<pid>, namgpu-coupled-<pid>   the several-card history folders (run_forecast.launch, nest/coupled.py)
A run that is killed (the kernel's OOM killer, a bus error, SIGKILL from a job runner) leaves them behind, and
/dev/shm is host RAM: on the 0.2.1 candidate's arctic-front run a broken column pool left 16 views (about 43 GB of a
62 GB /dev/shm), and the next run died of a bus error at +3 h when its history block met the full tmpfs.

  free_bytes(path)   room on the filesystem holding `path`, less what NAMGPU_SHM_CAP_MIB leaves the package
  in_ram(path)       whether `path` is on tmpfs/ramfs (its pages are memory, charged to the session's cgroup)
  sweep(root)        remove the entries above whose owner process has ended (one line when it removed any)
  own(root)          called by a process before its first entry under `root`: it holds a lock there for its life

Who is alive. A process id says nothing across PID namespaces: a run in a container that shares the host's /dev/shm
(docker --ipc=host) has ids the host does not know, so by id alone its live entries looked like leftovers and a
sweep on the host would have removed them after MIN_AGE_S. So an owner also holds a shared flock on
<root>/woof-nam-owner-<pid>.lock from before its first entry until it exits (the kernel drops the lock with the
process, however it ends, in any namespace). The sweep asks the lock first: held = alive. Only an entry whose owner
left no lock file (0.2.0, or the 0.2.1 candidates before this) is decided by the process id, as before.

Environment:
  NAMGPU_SHM           the host-RAM scratch root for the column views and the products frames (default /dev/shm)
  NAMGPU_SHM_CAP_MIB   the most the package's own entries may hold in a host-RAM scratch folder (tmpfs/ramfs); the
                       room it sees there is the smaller of the free space and this cap less its entries already there
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

PREFIXES = ("woof-nam-columns-", "woof-nam-hist-", "woof-nam-frame-", "namgpu-hist-", "namgpu-coupled-")
_OWNED = re.compile(r"^(woof-nam-columns|woof-nam-hist|woof-nam-frame|namgpu-hist|namgpu-coupled)-(\d+)(?:-|$)")
# an entry of the 0.2.0 column store (woof-nam-columns-XXXXXXXX: no owner id) is taken as ended after this long
UNOWNED_AGE_S = 3600.0
# an owned entry younger than this is never swept (an owner in another PID namespace looks ended from here)
MIN_AGE_S = 120.0
GB = 1e9
OWNER = "woof-nam-owner-"         # <root>/woof-nam-owner-<pid>.lock, held by the process that owns entries there
_OWN: dict = {}                   # (pid, root) -> [the open descriptor that holds this process's owner lock, users]

try:
    import fcntl
except ImportError:               # no flock on this platform: the process id decides, as before
    fcntl = None


def _owner_path(path, pid: int) -> Path:
    return Path(path) / f"{OWNER}{int(pid)}.lock"


def own(path=None) -> None:
    """Take this process's owner lock under `path` (default root()) and keep it until disown(path) or until the
    process exits. Call it before the first entry is made there; a later call only counts one more user. Breakage it
    prevents: a live run in another PID namespace that shares this /dev/shm losing its entries to a sweep (see the
    module text)."""
    path = root() if path is None else Path(path)
    if fcntl is None or path is None:
        return
    key = (os.getpid(), str(path))
    if key in _OWN:
        _OWN[key][1] += 1
        return
    lock = _owner_path(path, os.getpid())
    try:
        while True:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                if os.fstat(fd).st_ino == os.stat(lock).st_ino:
                    break                              # the file at this name is the one locked
            except FileNotFoundError:
                pass
            os.close(fd)                               # a sweep removed the name between open and lock: again
    except OSError:
        return                                         # a scratch folder that takes no lock: the id decides
    if not _OWN:
        import atexit
        atexit.register(disown_all)
    _OWN[key] = [fd, 1]


def disown(path=None) -> None:
    """One user of this process's owner lock under `path` is done (its entries are removed). With the last one the
    lock is dropped and its file removed, so a finished store leaves nothing in the scratch folder."""
    path = root() if path is None else Path(path)
    key = (os.getpid(), str(path))
    ent = _OWN.get(key)
    if ent is None:
        return
    ent[1] -= 1
    if ent[1] > 0:
        return
    del _OWN[key]
    try:
        # the name goes only if this process is the lock's last holder (the same id in another PID namespace may
        # hold it too); the name is removed under the exclusive lock, as the sweep does
        fcntl.flock(ent[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
        _owner_path(path, key[0]).unlink(missing_ok=True)
    except OSError:
        pass
    finally:
        os.close(ent[0])


def disown_all() -> None:
    """Every owner lock of this process is given back and its file removed. Runs at interpreter exit; the forecast
    entries end with os._exit or exec, which skip exit hooks, so io_history.shutdown() calls it for them."""
    for pid, path in list(_OWN):
        if pid != os.getpid():
            continue                                   # a forked child's copy of the parent's table: not its lock
        _OWN[(pid, path)][1] = 1
        disown(path)


def owner_lock_held(path, pid: int) -> bool | None:
    """Whether the owner lock of process id `pid` under `path` is held: True (its owner is alive, in whatever PID
    namespace), False (the lock file is there and nobody holds it: the owner has ended), None (no lock file)."""
    if fcntl is None:
        return None
    lock = _owner_path(path, pid)
    try:
        fd = os.open(lock, os.O_RDWR)
    except FileNotFoundError:
        return None
    except OSError:
        return True                                    # not ours to open: leave its entries alone
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        return False
    finally:
        os.close(fd)


def root() -> Path | None:
    """The host-RAM scratch root: NAMGPU_SHM, else /dev/shm where it exists, else None."""
    v = os.environ.get("NAMGPU_SHM", "").strip()
    if v:
        return Path(v)
    p = Path("/dev/shm")
    return p if p.is_dir() else None


def owned(name: str) -> bool:
    return name.startswith(PREFIXES)


def _size(p: Path) -> int:
    try:
        if p.is_symlink() or not p.is_dir():
            return p.lstat().st_size
        return sum(f.lstat().st_size for f in p.rglob("*") if not f.is_dir())
    except OSError:
        return 0


def own_bytes(path) -> int:
    """Bytes held by the package's own entries directly under `path`."""
    try:
        return sum(_size(e) for e in Path(path).iterdir() if owned(e.name))
    except OSError:
        return 0


def _cap() -> int | None:
    v = os.environ.get("NAMGPU_SHM_CAP_MIB", "").strip()
    return int(float(v) * 1024 * 1024) if v else None


def disk_free(path) -> int:
    """The filesystem's free bytes at `path` (its nearest existing parent); 0 when it cannot be read."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free)
    except OSError:
        return 0


def free_bytes(path) -> int:
    """The room the package may still fill at `path`: the filesystem's free space, and, on tmpfs or ramfs (host RAM),
    under NAMGPU_SHM_CAP_MIB the cap less what the package's own entries there already hold. The cap is on host RAM:
    a disk scratch folder (the column store's <out>/.scratch) keeps its own free space."""
    free = disk_free(path)
    cap = _cap()
    if cap is not None and in_ram(path):
        free = min(free, max(0, cap - own_bytes(path)))
    return free


def in_ram(path) -> bool:
    """`path` is on tmpfs or ramfs (the longest matching mount in /proc/mounts); /dev/shm is taken as RAM when
    /proc/mounts cannot be read."""
    p = str(Path(path).resolve())
    best, kind = "", None
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            f = line.split()
            if len(f) < 3:
                continue
            mnt = f[1].replace("\\040", " ")
            if (p == mnt or p.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                best, kind = mnt, f[2]
    except OSError:
        return p == "/dev/shm" or p.startswith("/dev/shm/")
    return kind in ("tmpfs", "ramfs")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ended(entry: Path, now: float | None = None) -> bool:
    """An entry of the package whose owner has ended: an owned name with a process id that no longer runs (and
    older than MIN_AGE_S), or a 0.2.0 column view (no owner id) older than UNOWNED_AGE_S."""
    now = time.time() if now is None else now
    try:
        age = now - entry.lstat().st_mtime
    except OSError:
        return False
    m = _OWNED.match(entry.name)
    if m:
        pid = int(m[2])
        if age < MIN_AGE_S or pid == os.getpid():
            return False
        held = owner_lock_held(entry.parent, pid)
        if held is not None:
            return not held            # the owner's lock decides, across PID namespaces
        return not alive(pid)          # an entry of a version that took no owner lock
    return entry.name.startswith("woof-nam-columns-") and age >= UNOWNED_AGE_S


def remove(entry: Path) -> None:
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry, ignore_errors=True)
    else:
        try:
            entry.unlink()
        except FileNotFoundError:
            pass


def sweep(path=None, log=print) -> list[str]:
    """Remove every package entry under `path` (default root() and /dev/shm) whose owner process has ended. Says
    so in one line when it removed any. Returns the names removed."""
    roots = [Path(path)] if path is not None else list(dict.fromkeys(p for p in (root(), Path("/dev/shm"))
                                                                    if p is not None and p.is_dir()))
    gone, size = [], 0
    for r in roots:
        try:
            entries = [e for e in r.iterdir() if owned(e.name)]
        except OSError:
            continue
        for e in entries:
            if ended(e):
                size += _size(e)
                remove(e)
                gone.append(e.name)
        _sweep_owner_locks(r)
    if gone and log is not None:
        log(f"scratch: removed {len(gone)} leftovers of ended woof-nam runs from "
            f"{', '.join(str(r) for r in roots)} ({size / GB:.1f} GB): {', '.join(sorted(gone)[:6])}"
            + (" ..." if len(gone) > 6 else ""))
    return gone


def _sweep_owner_locks(r: Path) -> None:
    """Remove the owner lock files under `r` that nobody holds and that are older than MIN_AGE_S. The name is
    removed while this process holds the lock exclusively, and own() checks the name after locking, so an owner
    never ends up holding a lock on a file that has lost its name."""
    if fcntl is None:
        return
    try:
        locks = [e for e in r.iterdir() if e.name.startswith(OWNER)]
    except OSError:
        return
    for e in locks:
        try:
            if time.time() - e.lstat().st_mtime < MIN_AGE_S:
                continue
            fd = os.open(e, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            e.unlink()
        except OSError:
            pass
        finally:
            os.close(fd)


def remove_own(path, prefix: str, pid: int | None = None) -> list[str]:
    """Remove the entries `prefix`-<pid>-* under `path` (this process's by default). Returns the names removed."""
    pid = os.getpid() if pid is None else pid
    gone = []
    try:
        for e in Path(path).iterdir():
            if e.name.startswith(f"{prefix}{pid}-"):
                remove(e)
                gone.append(e.name)
    except OSError:
        pass
    return gone
