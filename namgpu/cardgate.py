"""The card gate of a forecast that runs with its GRIB2 follower on one card: the forecast always gets the card, the
follower waits.

Breakage it prevents: `woof-nam run --products` on a 32 GB card stopped at forecast hour 3 with
`OutOfMemoryError: Out of memory allocating 125,880,832 bytes (allocated so far: 24,386,565,632 bytes, limit set to:
27,663,120,896 bytes)` (warm-season case 2024-05-06 12Z, 2026-11-11). The forecast gives memory back to the driver
after every radiation call and every history hour and takes it again when it needs it; in between, the follower's
frame and its writer process had taken that memory, and the driver refused the forecast below its own limit. Whether
it happened depended on when the follower's hour started. The follower already waits and reruns when it is the one
refused (products._start_hour_fits); nothing made it step aside for the forecast.

How it works. Two lock files in <run>/.card-gate, made by the forecast's door before it starts the follower:

  hold  the follower takes it shared for every piece of card work (the frame's carriers computed on the card, and
        each writer process from its start to its exit) and drops it when that work has given its memory back.
  turn  a turnstile in front of `hold`: the follower passes through it (shared, for an instant) before every hold.

When the driver refuses a request that the forecast's pool may take (devmem's allocator), the forecast claims the
card: it locks `turn` exclusively, so no new follower work can start, then locks `hold` exclusively, which it gets
when every follower hold in flight has ended, that is when their memory is back with the driver. It then asks
the driver again. It keeps both locks until its next devmem.release() (after the radiation call or the history hour
that needed the memory), then the follower goes on. A forecast that is not refused takes no lock and never waits in
the step loop; its only gate work is one pwrite of its state line at each release point (below).

The locks are flock locks: the kernel drops them when their holder exits, however it exits, and they hold across PID
namespaces. A run folder without the gate (woof-nam products on a finished run, the multi-card doors) takes no lock.
A claim that is not granted within NAMGPU_CARD_WAIT_S (600 s) gives up and the allocator raises as before: the memory
is then held by something that is not this run's follower, or a writer hangs.

The forecast's state line, <run>/.card-gate/forecast: "<releases> running|ended <pid> <time>". The forecast rewrites
it (one pwrite of 64 bytes) each time it gives unused memory back to the driver (devmem.release: after every radiation
call and every history hour) and once when its step loop ends. The follower reads it when one of its card stages
(frame, MSLET, writer) was refused card memory: while the forecast runs it waits for the forecast's next release and
reruns then, and that refusal does not use up one of the hour's OOM_RETRIES (the card is shared with this run's own
forecast, which ends); once the forecast has ended, or with no gate, a refusal is someone else's memory and the old
wait (NAMGPU_PRODUCTS_OOM_WAIT_S) and count apply. Breakage it prevents: the follower slept a fixed 30 s after each
refusal, 20 times at most per hour, whatever the forecast did; in a 24 h warm run on a 32 GB card that put the GRIB2
files 99 to 364 s behind their history hour (hour 0 used 9 of its 20 tries), and a busier card could lose an hour.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # a platform without flock: no gate; the follower's own wait-and-rerun is what is left
    fcntl = None

DIR = ".card-gate"
STATE = "forecast"          # the forecast's state line (module text)
STATE_BYTES = 64


def gate_dir(run_dir) -> Path:
    return Path(run_dir) / DIR


def create(run_dir) -> Path | None:
    """Make the run's gate (the forecast's door, before it starts the follower). None where flock does not exist."""
    if fcntl is None:
        return None
    d = gate_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    for n in ("turn", "hold"):
        (d / n).touch()
    (d / STATE).unlink(missing_ok=True)       # a state line of an earlier run in this folder is not this forecast's
    return d


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return True
    try:                                      # a zombie (exited, not yet reaped) is not a forecast
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def forecast_state(run_dir) -> dict | None:
    """The forecast's state line as the follower sees it: {"releases", "state", "pid", "t"}, state running, ended or
    gone (it said running and its process no longer exists). None when the run has no gate or the forecast has not
    written the line (a finished run's folder, a multi-card door, a forecast still in its door's checks)."""
    p = gate_dir(run_dir) / STATE
    for _ in range(3):                        # a read that met a rewrite in flight is read again
        try:
            parts = p.read_text().split()
        except OSError:
            return None
        try:
            n, state, pid, t = int(parts[0]), parts[1], int(parts[2]), float(parts[3])
        except (IndexError, ValueError):
            time.sleep(0.01)
            continue
        if state == "running" and not _alive(pid):
            state = "gone"
        return {"releases": n, "state": state, "pid": pid, "t": t}
    return None


def wait_release(run_dir, since: int, timeout_s: float, beat=None, poll_s: float = 0.1) -> str:
    """The follower, after one of its card stages was refused memory while the forecast runs: wait until the forecast
    gives memory back (its release count passes `since`), ends, or is gone, or `timeout_s` passes. Returns released,
    ended, gone, timeout or none (no state line). `beat` is called about every 5 s while it waits (the follower's
    heartbeat: alive and waiting, namgpu.io.products)."""
    t0 = last_beat = time.monotonic()
    while True:
        st = forecast_state(run_dir)
        if st is None:
            return "none"
        if st["state"] != "running":
            return st["state"]
        if st["releases"] > since:
            return "released"
        now = time.monotonic()
        if now - t0 >= timeout_s:
            return "timeout"
        if beat is not None and now - last_beat >= 5.0:
            beat()
            last_beat = now
        time.sleep(poll_s)


class Hold:
    """One piece of the follower's card work. release() (or the end of the process) drops it."""

    def __init__(self, fd=None):
        self.fd = fd

    def release(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)                      # closing the descriptor drops its lock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def hold(run_dir) -> Hold:
    """The follower, before a piece of card work for a history hour of `run_dir`: waits while the forecast claims the
    card, then holds its place until release(). A run folder with no gate: a hold that locks nothing."""
    d = gate_dir(run_dir)
    if fcntl is None or not (d / "hold").is_file() or not (d / "turn").is_file():
        return Hold()
    turn = os.open(d / "turn", os.O_RDWR)
    try:
        fcntl.flock(turn, fcntl.LOCK_SH)      # waits here while the forecast claims the card
        fd = os.open(d / "hold", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)    # the forecast locks `hold` only while it has `turn`
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(turn)
    return Hold(fd)


class Forecast:
    """The forecast's side of the gate (registered with devmem.card_gate)."""

    def __init__(self, run_dir, wait_s: float | None = None, log=None):
        self.dir = gate_dir(run_dir)
        self.wait_s = float(os.environ.get("NAMGPU_CARD_WAIT_S", "600")) if wait_s is None else float(wait_s)
        self.log = log or (lambda m: print(m, flush=True))
        self._fds = None
        self.claims, self.waited_s = 0, 0.0
        self.releases = 0
        self._state_fd = None
        self._ended = False

    def _state(self, state: str) -> None:
        """Rewrite the state line (one pwrite; never raises: the follower then waits its old fixed time)."""
        if not self.dir.is_dir():
            return
        try:
            if self._state_fd is None:
                self._state_fd = os.open(self.dir / STATE, os.O_RDWR | os.O_CREAT, 0o644)
            line = f"{self.releases} {state} {os.getpid()} {time.time():.3f}".ljust(STATE_BYTES - 1) + "\n"
            os.pwrite(self._state_fd, line.encode(), 0)
        except OSError:
            pass

    def running(self) -> None:
        """The forecast is on the card (devmem.card_gate, when the gate is registered)."""
        self._state("running")

    def released(self) -> None:
        """The forecast gave its unused memory back to the driver (devmem.release)."""
        self.releases += 1
        self._state("ended" if self._ended else "running")

    def ended(self) -> None:
        """The step loop is over: the forecast's card memory is no longer contended for (devmem.card_end)."""
        self._ended = True
        self._state("ended")

    def claimed(self) -> bool:
        return self._fds is not None

    def claim(self, nbytes: int = 0) -> bool:
        """Take the card from the follower: True once every follower hold has ended (ask the driver again), False
        when the card already was the forecast's, there is no gate, or the wait ran out (the refusal stands)."""
        if fcntl is None or self._fds is not None or not (self.dir / "hold").is_file():
            return False
        t0 = time.monotonic()
        fds = []
        try:
            for name in ("turn", "hold"):
                fd = os.open(self.dir / name, os.O_RDWR)
                fds.append(fd)
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - t0 > self.wait_s:
                            raise TimeoutError from None
                        time.sleep(0.02)
        except (TimeoutError, OSError) as e:
            for fd in fds:
                os.close(fd)
            self.log(f"card: the forecast needs {nbytes / 2**20:.0f} MiB more, the card has none, and the GRIB2 "
                     f"follower still held its place after {time.monotonic() - t0:.0f} s ({type(e).__name__})")
            return False
        dt = time.monotonic() - t0
        self._fds = fds
        self.claims += 1
        self.waited_s += dt
        self.log(f"card: the forecast needs {nbytes / 2**20:.0f} MiB more and the card had none: it waited {dt:.1f} s "
                 f"for the GRIB2 follower's work in flight to end, and the follower starts nothing until the "
                 f"forecast gives this memory back")
        return True

    def release(self) -> None:
        fds, self._fds = self._fds, None
        for fd in reversed(fds or []):
            os.close(fd)
