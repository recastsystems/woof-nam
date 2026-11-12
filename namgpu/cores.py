"""How many CPU cores this process may really use: the CPU affinity set, capped by the cgroup CPU quota.

os.cpu_count() reports the host's cores. A rented container can see all of them and be allowed far fewer: the
container (2026-11-04) shows 96 cores with cgroup cpu.max 2304000/100000 (23 cores). Pools sized by
os.cpu_count() there started 96 grib2_dump processes for one IFS boundary time and starved a forecast running
beside it (its history writer stalled for minutes at 73 % CPU pressure). Every worker pool sizes from usable().
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def usable() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max",):                     # cgroup v2
        try:
            quota, period = Path(path).read_text().split()[:2]
            if quota != "max":
                n = min(n, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:                                                          # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            n = min(n, max(1, q // p))
    except (OSError, ValueError):
        pass
    return max(1, n)


# AF_UNIX socket paths hold 107 bytes on Linux and 103 on macOS; keep below the smaller.
_SUN_PATH_MAX = 103
# multiprocessing's forkserver listens on <its temporary folder>/listener-XXXXXXXX
_LISTENER_NAME = len("/listener-XXXXXXXX")


def pool_context():
    """The start method of this package's worker pools: forkserver (spawn where there is none), with the forkserver's
    socket in a folder whose path is short enough for AF_UNIX.

    A forked worker would inherit this process's signal handlers, both ends of the pool's pipes and the state of every
    thread's locks, so the pools never fork. multiprocessing puts the forkserver's listening socket in its own
    temporary folder, $TMPDIR/pymp-XXXXXXXX. Breakage it prevents: with a TMPDIR of 115 characters, on Python 3.12, the
    products and column pools failed at their first task with "OSError: AF_UNIX path too long" (2026-11-12). When the
    path would not fit, the pools' folder is made in /tmp (or /var/tmp) instead, private to this user and removed at
    exit; TMPDIR stays the temporary folder for everything else."""
    import multiprocessing as mp

    if "forkserver" not in mp.get_all_start_methods():
        return mp.get_context("spawn")
    _short_socket_folder()
    return mp.get_context("forkserver")


def _short_socket_folder() -> None:
    """Point multiprocessing's temporary folder (where the forkserver socket goes) at a short private folder when the
    one it would use leaves no room for the socket name. multiprocessing keeps that folder in the process's _config
    under "tempdir" (multiprocessing.util.get_temp_dir); this sets it the same way, before the forkserver starts."""
    import shutil
    import tempfile
    from multiprocessing import process, util

    config = process.current_process()._config
    current = config.get("tempdir")
    planned = current or os.path.join(tempfile.gettempdir(), "pymp-XXXXXXXX")
    if len(os.fsencode(planned)) + _LISTENER_NAME <= _SUN_PATH_MAX:
        return
    for base in ("/tmp", "/var/tmp"):
        try:
            short = tempfile.mkdtemp(prefix="pymp-", dir=base)
        except OSError:
            continue
        if len(os.fsencode(short)) + _LISTENER_NAME <= _SUN_PATH_MAX:
            break
        os.rmdir(short)
    else:
        raise OSError(f"no folder short enough for the worker pools' socket: {planned} is too long for AF_UNIX and "
                      "neither /tmp nor /var/tmp is writable; set TMPDIR to a shorter path")
    util.Finalize(None, shutil.rmtree, args=(short,), kwargs={"ignore_errors": True}, exitpriority=-100)
    config["tempdir"] = short
