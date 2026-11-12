"""The GRIB2 follower, refused card memory while its run's forecast is on the card, waits for the forecast's next
release of memory (the card gate's state line) and reruns then; that refusal does not use up the hour's tries.

Breakage it prevents (run-hardening-3's 24 h warm run on a 32 GB card, 2026-11-11): after each "out of GPU memory" the
follower slept a fixed 30 s, 20 times at most per hour, whatever the forecast did. Hour 0 used 9 of its 20 tries and the
GRIB2 files landed 99 to 364 s behind their history hour; a busier card could use up 20 and lose an hour. CPU only: a
stand-in forecast (cardgate.Forecast and devmem's registration, no card) and a stand-in writer that is refused
a few times."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import cardgate, devmem  # noqa: E402
from namgpu.io import products  # noqa: E402

# exits 1 with the writer's own full-card refusal while <run>/refuse holds a number above 0 (and counts it down)
WRITER = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); n=int(p.read_text()); p.write_text(str(n-1)); "
          "print('Error: x: CUDA device has 1 bytes free, the frame needs about 9' if n > 0 else 'ok'); "
          "sys.exit(1 if n > 0 else 0)")


class _Proc:
    returncode = 1


def _hour(run: Path, refusals: int):
    (run / "refuse").write_text(str(refusals))
    (run / "work" / "grib2").mkdir(parents=True, exist_ok=True)
    h = products._Hour(run / "wrfout_d01_2024-05-06_12:00:00", run, run / "work", {}, time.perf_counter(), 0.0,
                       _Proc(), [sys.executable, "-c", WRITER, str(run / "refuse")])
    h.stamps = {"history": time.time()}
    return h, "Error: x: CUDA device has 1 bytes free, the frame needs about 9"


class _Releasing:
    """A stand-in forecast on the card: registered with devmem (state line running), it gives memory back every
    `every` s (devmem's release path minus the card) until stopped, then its step loop ends (card_end)."""

    def __init__(self, run: Path, every: float = 0.3):
        cardgate.create(run)
        self.g = cardgate.Forecast(run, log=lambda m: None)
        devmem.card_gate(self.g)
        self.every, self.stop = every, threading.Event()
        self.t = threading.Thread(target=self._loop)

    def _loop(self):
        while not self.stop.wait(self.every):
            devmem.card_release()
            getattr(self.g, "released", lambda: None)()      # (a tree before the state line: nothing to say)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join()
        getattr(devmem, "card_end", devmem.card_release)()
        devmem.card_gate(None)


def test_refusals_while_the_forecast_runs_do_not_use_up_the_tries():
    keep = (products.OOM_RETRIES, products.OOM_WAIT_S)
    products.OOM_RETRIES, products.OOM_WAIT_S = 2, 30.0
    try:
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            h, err = _hour(run, refusals=4)          # the first try's refusal, then 4 more: 5 reruns
            t0 = time.monotonic()
            with _Releasing(run):
                rc, out, _, reruns = products._rerun_on_oom(h, err, "")
            wall = time.monotonic() - t0
            assert rc == 0, f"the writer was given up after {len(reruns)} reruns: {reruns}"
            assert len(reruns) == 5 and all("not counted" in r for r in reruns), reruns
            assert wall < 20, f"it slept the fixed {products.OOM_WAIT_S} s instead of waiting for a release: {wall}"
            assert h.stamps["reruns"] == 5 and h.stamps["reruns_counted"] == 0, h.stamps
    finally:
        products.OOM_RETRIES, products.OOM_WAIT_S = keep


def test_after_the_forecast_ended_refusals_count_as_before():
    keep = (products.OOM_RETRIES, products.OOM_WAIT_S)
    products.OOM_RETRIES, products.OOM_WAIT_S = 2, 0.0
    try:
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            with _Releasing(run):
                pass                                  # the step loop has ended: state line "ended"
            assert cardgate.forecast_state(run)["state"] == "ended"
            h, err = _hour(run, refusals=5)
            rc, _, _, reruns = products._rerun_on_oom(h, err, "")
            assert rc != 0 and len(reruns) == 2 and "try 2 of 2" in reruns[-1], reruns
    finally:
        products.OOM_RETRIES, products.OOM_WAIT_S = keep


def test_frame_refused_while_the_forecast_runs_waits_for_its_release():
    keep = (products.OOM_RETRIES, products.OOM_WAIT_S)
    products.OOM_RETRIES, products.OOM_WAIT_S = 1, 30.0
    calls, logs = [], []

    class OutOfMemoryError(Exception):
        pass

    def start():
        calls.append(1)
        if len(calls) <= 3:
            raise OutOfMemoryError("Out of memory allocating 83,886,080 bytes")
        return "hour"
    try:
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            t0 = time.monotonic()
            import inspect
            kw = {"run_dir": run} if "run_dir" in inspect.signature(products._start_hour_fits).parameters else {}
            with _Releasing(run):
                try:
                    got = products._start_hour_fits(start, "wrfout_d01_x", log=logs.append, **kw)
                except OutOfMemoryError:
                    got = f"given up after {len(calls)} tries"
            assert got == "hour" and len(calls) == 4, (got, calls, logs)
            assert time.monotonic() - t0 < 20 and all("not counted" in m for m in logs), logs
    finally:
        products.OOM_RETRIES, products.OOM_WAIT_S = keep


def test_a_forecast_that_is_gone_is_not_waited_for():
    with tempfile.TemporaryDirectory() as t:
        run = Path(t)
        cardgate.create(run)
        p = __import__("subprocess").Popen([sys.executable, "-c", "pass"])
        p.wait()
        (run / cardgate.DIR / cardgate.STATE).write_text(f"3 running {p.pid} 0.0".ljust(63) + "\n")
        st = cardgate.forecast_state(run)
        assert st["state"] == "gone" and st["releases"] == 3, st
        assert cardgate.wait_release(run, 3, timeout_s=30) == "gone"


if __name__ == "__main__":
    test_refusals_while_the_forecast_runs_do_not_use_up_the_tries()
    test_after_the_forecast_ended_refusals_count_as_before()
    test_frame_refused_while_the_forecast_runs_waits_for_its_release()
    test_a_forecast_that_is_gone_is_not_waited_for()
    print("test_products_card_release_wait: 4 passed")
