"""A forecast with its GRIB2 follower on one card: the forecast always gets the card, the follower waits.

Breakage it prevents: `woof-nam run --case 2024-05-06_12z --source native --hours 24 --products` on a 32 GB card
stopped at forecast hour 3 with `OutOfMemoryError: Out of memory allocating 125,880,832 bytes (allocated so far:
24,386,565,632 bytes, limit set to: 27,663,120,896 bytes)`: the follower's frame and writer had taken the memory the
forecast had given back to the driver, and nothing made them step aside (2026-11-11). Whether it happened depended on
when the follower's hour started.

CPU only. The card is a counter shared by a stand-in for the forecast's CuPy pool and a stand-in for the follower's
writer; the allocator under test is devmem's own, the locks are the real ones (namgpu.cardgate)."""
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import devmem  # noqa: E402

GIB = 2 ** 30


class OutOfMemoryError(Exception):
    """Stands in for cupy.cuda.memory.OutOfMemoryError."""


class Card:
    """A 32 GiB card: what the forecast's pool holds plus what the follower's work holds."""

    def __init__(self, total=32 * GIB):
        self.total, self.follower = total, 0
        self.lock = threading.Lock()


class Pool:
    """Stands in for CuPy's memory pool: a limit, and a driver (the card) that refuses when it is full."""

    def __init__(self, card):
        self.card, self.limit, self.total = card, 0, 0

    def get_limit(self):
        return self.limit

    def set_limit(self, size=0):
        self.limit = int(size)

    def total_bytes(self):
        return self.total

    def used_bytes(self):
        return self.total

    def free_all_blocks(self):
        pass

    def malloc(self, n):
        with self.card.lock:
            if self.limit and self.total + n > self.limit:
                raise OutOfMemoryError(f"Out of memory allocating {n:,} bytes (allocated so far: {self.total:,} "
                                       f"bytes, limit set to: {self.limit:,} bytes).")
            if self.total + n + self.card.follower > self.card.total:
                raise OutOfMemoryError(f"Out of memory allocating {n:,} bytes (allocated so far: {self.total:,} "
                                       f"bytes, limit set to: {self.limit:,} bytes).")       # the driver's refusal
            self.total += n
        return object()

    def give_back(self, n):
        with self.card.lock:
            self.total -= n


@pytest.fixture
def forecast(tmp_path, monkeypatch):
    """devmem's allocator over the stand-in pool, in a run folder; returns (malloc, pool, card, run folder)."""
    card = Card()
    pool = Pool(card)
    alloc = [None]
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_allocator=lambda: alloc[0], set_allocator=lambda f: alloc.__setitem__(0, f),
                                   memory=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError),
                                   Device=lambda: types.SimpleNamespace(synchronize=lambda: None)),
        get_default_memory_pool=lambda: pool)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(devmem, "_STATE", {})
    monkeypatch.setattr(devmem, "_HITS", [0, 0.0])
    monkeypatch.setattr(devmem, "_GATE", [None], raising=False)
    monkeypatch.delenv("NAMGPU_POOL", raising=False)
    monkeypatch.delenv("NAMGPU_POOL_CAP", raising=False)
    monkeypatch.setenv("NAMGPU_POOL_CAP_STEP_MIB", "0")
    devmem.setup()
    run = tmp_path / "run"
    run.mkdir()
    return alloc[0], pool, card, run


def _door(run):
    """What run_forecast._start_products does before it starts the follower. On code without the gate: nothing."""
    try:
        from namgpu import cardgate
    except ImportError:
        return None
    cardgate.create(run)
    gate = cardgate.Forecast(run, log=lambda m: print(m, flush=True))
    devmem.card_gate(gate)
    return gate


def _hold(run):
    """The follower's place on the card for one writer. On code without the gate: it just takes the card."""
    try:
        from namgpu import cardgate
    except ImportError:
        return contextlib.nullcontext()
    return cardgate.hold(run)


def _writer(run, card, nbytes, seconds, log, name):
    """A follower hour: takes its place, holds `nbytes` of the card for `seconds`, gives it back, leaves."""
    with _hold(run):
        with card.lock:
            card.follower += nbytes
        log.append(f"{name} on the card")
        time.sleep(seconds)
        with card.lock:
            card.follower -= nbytes
        log.append(f"{name} off the card")


def test_forecast_refused_by_a_full_card_waits_for_the_follower_and_goes_on(forecast):
    """The failing run in small: the forecast holds 24 GiB and has given 4 GiB back; the follower's writer takes
    6 GiB of the 8 free; the forecast asks for its 4 GiB again. Before: OutOfMemoryError. Now: the writer ends, the
    forecast gets its memory, and no new writer starts until the forecast has given the memory back."""
    malloc, pool, card, run = forecast
    gate = _door(run)
    log = []
    malloc(24 * GIB)
    w1 = threading.Thread(target=_writer, args=(run, card, 6 * GIB, 0.4, log, "writer 1"))
    w1.start()
    while "writer 1 on the card" not in log:
        time.sleep(0.01)
    t0 = time.monotonic()
    malloc(4 * GIB)                      # 24 + 4 + 6 > 32: the driver refuses while writer 1 is on the card
    waited = time.monotonic() - t0
    log.append("forecast has its 4 GiB")
    assert pool.total == 28 * GIB
    assert log.index("writer 1 off the card") < log.index("forecast has its 4 GiB"), log
    assert 0.2 < waited < 5.0, waited
    # while the forecast keeps the claim (its radiation call is still running) the follower starts nothing
    w2 = threading.Thread(target=_writer, args=(run, card, 6 * GIB, 0.0, log, "writer 2"))
    w2.start()
    time.sleep(0.3)
    assert "writer 2 on the card" not in log, log
    malloc(2 * GIB)                      # and the forecast can still grow: 30 of 32 GiB, nobody else on the card
    # the forecast's release point (after the radiation call): memory back to the driver, the follower goes on
    pool.give_back(6 * GIB)
    devmem.release()
    w2.join(5)
    w1.join(5)
    assert "writer 2 off the card" in log, log
    assert gate.claims == 1 and not gate.claimed()
    assert devmem.stats()["card_claims"] == 1


def test_a_card_that_is_full_without_the_follower_still_raises(forecast):
    """The claim is not a retry loop: with no follower work in flight the refusal is a real one, at once."""
    malloc, pool, card, run = forecast
    _door(run)
    malloc(24 * GIB)
    card.follower = 6 * GIB              # someone else's memory, behind no gate
    t0 = time.monotonic()
    with pytest.raises(OutOfMemoryError):
        malloc(4 * GIB)
    assert time.monotonic() - t0 < 1.0


def test_new_follower_work_does_not_pass_a_waiting_forecast(tmp_path):
    """Two follower hours overlap (workers = 2). The forecast's claim waits for the one in flight; the next one
    must wait behind the forecast, not join the one in flight and keep the forecast waiting."""
    from namgpu import cardgate
    cardgate.create(tmp_path)
    order = []
    a = cardgate.hold(tmp_path)
    gate = cardgate.Forecast(tmp_path, log=lambda m: None)
    tf = threading.Thread(target=lambda: (gate.claim(1), order.append("forecast")))
    tf.start()
    time.sleep(0.15)                     # the forecast now has the turnstile and waits for hold a

    def later():
        with cardgate.hold(tmp_path):
            order.append("hour b")
    tb = threading.Thread(target=later)
    tb.start()
    time.sleep(0.15)
    assert order == [], order            # neither got through while hour a is on the card
    a.release()
    tf.join(5)
    time.sleep(0.15)
    assert order == ["forecast"], order
    gate.release()
    tb.join(5)
    assert order == ["forecast", "hour b"], order


def test_a_killed_follower_does_not_keep_the_card(tmp_path):
    """The hold is a kernel lock: it ends with its process, however that ends."""
    from namgpu import cardgate
    cardgate.create(tmp_path)
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); from namgpu import cardgate; "
            "h = cardgate.hold(sys.argv[2]); print('held', flush=True); time.sleep(60)")
    p = subprocess.Popen([sys.executable, "-c", code, str(Path(__file__).resolve().parents[1]), str(tmp_path)],
                         stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        gate = cardgate.Forecast(tmp_path, wait_s=0.3, log=lambda m: None)
        assert gate.claim(1) is False    # the follower is on the card: the short wait runs out
        os.kill(p.pid, signal.SIGKILL)
        p.wait(5)
        gate = cardgate.Forecast(tmp_path, wait_s=5, log=lambda m: None)
        assert gate.claim(1) is True
        gate.release()
    finally:
        p.kill()


def test_no_gate_no_lock(tmp_path):
    """woof-nam products on a finished run, or a door that made no gate: nothing is locked, nothing is created."""
    from namgpu import cardgate
    with cardgate.hold(tmp_path) as h:
        assert h.fd is None
    assert cardgate.Forecast(tmp_path, log=lambda m: None).claim(1) is False
    assert not (tmp_path / cardgate.DIR).exists()


def test_the_follower_writer_starts_behind_the_gate_and_leaves_it_when_it_exits(tmp_path, monkeypatch):
    """products._start_hour does not start a writer while the forecast claims the card, and _finish_hour gives the
    place back when the writer has exited, also when the hour fails."""
    from datetime import datetime

    from namgpu import cardgate
    from namgpu.io import products
    cardgate.create(tmp_path)
    started = []

    class Proc:
        returncode = 0

        def communicate(self):
            return "", ""

    monkeypatch.setattr(products, "wrf_frame", lambda *a, **k: {"valid": datetime(2024, 5, 6, 13),
                                                                 "start": datetime(2024, 5, 6, 12)})
    monkeypatch.setattr(products, "_exporter", lambda: Path("rw_rrfs_upp"))
    monkeypatch.setattr(products.subprocess, "Popen", lambda *a, **k: (started.append(1), Proc())[1])
    hist = tmp_path / "wrfout_d01_2024-05-06_13:00:00"
    hist.write_bytes(b"x")
    out = tmp_path / "products"
    out.mkdir()
    gate = cardgate.Forecast(tmp_path, log=lambda m: None)
    assert gate.claim(1) is True         # the forecast has the card
    hours = []
    t = threading.Thread(target=lambda: hours.append(products._start_hour(hist, out, "p.json", "gpu", 20, None)))
    t.start()
    time.sleep(0.3)
    assert started == [], "a writer was started while the forecast claimed the card"
    gate.release()
    t.join(5)
    assert started == [1] and len(hours) == 1
    gate2 = cardgate.Forecast(tmp_path, wait_s=0.2, log=lambda m: None)
    assert gate2.claim(1) is False       # the writer is on the card until it has exited
    with pytest.raises(RuntimeError):    # the stand-in writer wrote no file: the hour fails ...
        products._finish_hour(hours[0])
    assert gate2.claim(1) is True        # ... and its place on the card is given back all the same
    gate2.release()


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
