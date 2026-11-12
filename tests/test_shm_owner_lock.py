"""The sweep of ended runs' /dev/shm entries asks the owner's lock, not only its process id.

Breakage it prevents: a run in another PID namespace that shares this /dev/shm (a container started with the host's
IPC) has process ids this side does not know. By id alone its live entries looked like leftovers of an ended run, and
a sweep here would have removed them once they were older than two minutes: a history block or a column view pulled
from under a running forecast. An owner now holds a flock on <root>/woof-nam-owner-<pid>.lock for its whole life; the
kernel drops it when the process ends, in any namespace.

CPU only. The owner "in another namespace" is a helper process that holds the lock of a process id which does not
exist on this side."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io import scratch  # noqa: E402

HOLDER = ("import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644); "
          "fcntl.flock(fd, fcntl.LOCK_SH); print('held', flush=True); time.sleep(120)")


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _old(path: Path, seconds: float = 1000.0):
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_live_owner_in_another_pid_namespace_keeps_its_entries(tmp_path):
    pid = _dead_pid()                                   # an id that means nothing on this side
    entry = tmp_path / f"woof-nam-frame-{pid}-abc"
    entry.mkdir()
    (entry / "T.npy").write_bytes(b"x" * 10)
    block = tmp_path / f"woof-nam-hist-{pid}-1"
    block.write_bytes(b"y" * 10)
    lock = tmp_path / f"woof-nam-owner-{pid}.lock"
    owner = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "held"
        for p in (entry, block, lock):
            _old(p)
        assert not scratch.alive(pid)                   # by id the owner looks ended ...
        gone = scratch.sweep(tmp_path, log=None)
        assert gone == [], f"a live owner's entries were swept: {gone}"       # ... and its lock says it is alive
        assert entry.is_dir() and block.is_file() and lock.is_file()
        os.kill(owner.pid, signal.SIGKILL)              # the owner ends, however: the kernel drops its lock
        owner.wait(5)
        for p in (entry, block, lock):
            _old(p)
        gone = scratch.sweep(tmp_path, log=None)
        assert sorted(gone) == sorted([entry.name, block.name]), gone
        assert not lock.exists()                        # and the lock file goes with them
    finally:
        owner.kill()


def test_own_holds_this_process_lock_until_it_exits(tmp_path):
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); from namgpu.io import scratch; "
            "scratch.own(sys.argv[2]); scratch.own(sys.argv[2]); print('held', flush=True); time.sleep(120)")
    p = subprocess.Popen([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), str(tmp_path)],
                         stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert (tmp_path / f"woof-nam-owner-{p.pid}.lock").is_file()
        assert scratch.owner_lock_held(tmp_path, p.pid) is True
        os.kill(p.pid, signal.SIGKILL)
        p.wait(5)
        assert scratch.owner_lock_held(tmp_path, p.pid) is False
    finally:
        p.kill()


def test_disown_leaves_nothing_and_counts_its_users(tmp_path):
    scratch.own(tmp_path)                 # the history writer of this process, say
    scratch.own(tmp_path)                 # and its column store
    lock = tmp_path / f"woof-nam-owner-{os.getpid()}.lock"
    assert lock.is_file()
    scratch.disown(tmp_path)              # the store is done: the writer still owns entries
    assert lock.is_file() and scratch.owner_lock_held(tmp_path, os.getpid()) is True
    scratch.disown(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_history_shutdown_removes_the_lock_file_of_a_forecast_that_ends_with_os_exit(tmp_path):
    """The forecast entries end with os._exit (or exec), which skips exit hooks: io_history.shutdown() gives the
    owner lock back, so a finished run leaves no file of its own in /dev/shm."""
    code = ("import os, sys; sys.path.insert(0, sys.argv[1]); from namgpu.io import io_history, scratch; "
            "scratch.own(sys.argv[2]); scratch.own(sys.argv[2]); io_history.shutdown(); os._exit(0)")
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), str(tmp_path)], check=True)
    assert list(tmp_path.iterdir()) == []


def test_an_entry_without_a_lock_file_is_decided_by_the_process_id_as_before(tmp_path):
    """Entries of 0.2.0 and of the earlier 0.2.1 candidates: no lock file was ever made for them."""
    dead = tmp_path / f"woof-nam-columns-{_dead_pid()}-v"
    dead.mkdir()
    mine = tmp_path / f"woof-nam-columns-{os.getppid()}-v"          # a live process of this namespace
    mine.mkdir()
    _old(dead)
    _old(mine)
    assert scratch.owner_lock_held(tmp_path, os.getppid()) is None
    assert scratch.sweep(tmp_path, log=None) == [dead.name]
    assert mine.is_dir()


def test_the_makers_of_entries_take_the_owner_lock():
    """Each place that makes an entry calls scratch.own first (the five kinds in scratch's module text)."""
    root = Path(__file__).resolve().parents[1] / "namgpu"
    for rel, marker in (("io/products.py", "scr.own(root)"), ("io/io_history.py", "own(SHM)"),
                        ("io/columns.py", "scr.own(owned_scratch)"), ("run_forecast.py", "_scr.own(shm.parent)"),
                        ("nest/coupled.py", "_scr.own(shm.parent)")):
        assert marker in (root / rel).read_text(), f"{rel} makes a /dev/shm entry without scratch.own"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
