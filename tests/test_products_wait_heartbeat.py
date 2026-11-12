"""The run's wait for its GRIB2 follower sees a follower that is alive and waiting for card memory (its heartbeat).

Breakage it prevents (checker of door-products-3, 2026-11-11): products_wait counted only new files in products/ and
new lines in products.log as progress. A follower waiting for a full card may write neither for --products-stall-s
(900 s); the run then stopped a healthy follower and exited 1. Now the follower beats <run>/products.state.json about
every 5 s while one of its stages waits for card memory, and the wait counts a beat as progress. A follower that stops
beating (it hangs) is still stopped. CPU only: stand-in followers, and the follower's own wait code with no card."""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["NAMGPU_PRODUCTS_OOM_WAIT_S"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import run_forecast as rf  # noqa: E402
from namgpu.io import products  # noqa: E402

# beats products.state.json every 0.5 s for 4 s (no file, no log line), then writes DONE and exits 0
BEATING = ("import json,os,sys,time,pathlib; r=pathlib.Path(sys.argv[sys.argv.index('--run')+1]); "
           "o=r/'products'; o.mkdir(exist_ok=True); t0=time.time()\n"
           "while time.time()-t0 < 4:\n"
           "    d={'pid': os.getpid(), 'state': 'waiting for card memory (writer of x), 30 s', 'since': t0, "
           "'beat': time.time()}; p=r/'products.state.json'; q=r/'s.tmp'; q.write_text(json.dumps(d)); "
           "os.replace(q, p); time.sleep(0.5)\n"
           "(o/'DONE').write_text(json.dumps({'files': 1, 'errors': 0}))")
# beats twice, then hangs without beating
HANGS = ("import json,os,sys,time,pathlib; r=pathlib.Path(sys.argv[sys.argv.index('--run')+1]); t0=time.time()\n"
         "for i in range(2):\n"
         "    (r/'products.state.json').write_text(json.dumps({'pid': os.getpid(), 'state': 'working (x)', "
         "'since': t0, 'beat': time.time()})); time.sleep(0.5)\n"
         "time.sleep(600)")


def _start(out: Path, code: str) -> int:
    real = subprocess.Popen

    def popen(argv, **kw):
        return real([argv[0], "-c", code] + list(argv[3:]), **kw)
    rf.subprocess.Popen = popen
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rf._start_products(out, "awphys")
    finally:
        rf.subprocess.Popen = real
        from namgpu import devmem
        devmem.card_gate(None)                 # this test process runs no forecast
    return json.loads((out / rf.FOLLOWER).read_text())["pid"]


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def test_waiting_follower_is_not_stopped():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        pid = _start(out, BEATING)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = rf.products_finish(out, 0, stall_s=2.0)
            assert rc == 0 and (out / "products" / "DONE").is_file(), \
                f"a follower alive and waiting for card memory was stopped: {buf.getvalue()}"
        finally:
            _kill(pid)


def test_follower_that_stops_beating_is_still_stopped():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        pid = _start(out, HANGS)
        try:
            buf = io.StringIO()
            t0 = time.monotonic()
            with contextlib.redirect_stdout(buf):
                rc = rf.products_finish(out, 0, stall_s=2.0)
            assert rc == 1 and "wrote nothing new for 2 s" in buf.getvalue(), buf.getvalue()
            assert "working (x)" in buf.getvalue(), "the stall line names the follower's last state"
            assert time.monotonic() - t0 < 60
        finally:
            _kill(pid)


def test_another_followers_heartbeat_does_not_count():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        (out / rf.STATE_FILE).write_text(json.dumps({"pid": 1, "state": "x", "beat": 1.0}))
        assert rf._follower_beat(out, 2) is None and rf._follower_beat(out, 1)["state"] == "x"


def test_the_follower_beats_while_it_waits_for_card_memory():
    """The follower's own wait (no gate: the fixed wait) rewrites the heartbeat while it waits."""
    assert products.STATE_FILE == rf.STATE_FILE
    keep = (products.OOM_WAIT_S, dict(products._BEAT))
    with tempfile.TemporaryDirectory() as t:
        run = Path(t)
        products._BEAT.update(path=run / products.STATE_FILE, state="starting", since=0.0)
        products.OOM_WAIT_S = 6.0
        seen = []

        def watch():
            for _ in range(70):
                try:
                    seen.append(json.loads((run / products.STATE_FILE).read_text()))
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
        w = threading.Thread(target=watch)
        w.start()
        try:
            counts, why, dt = products._card_wait(run, "writer of wrfout_d01_x", 0)
        finally:
            w.join()
            products.OOM_WAIT_S = keep[0]
            products._BEAT.clear()
            products._BEAT.update(keep[1])
        waiting = [d for d in seen if d["state"].startswith("waiting for card memory")]
        assert waiting, seen[:3]
        assert len({d["beat"] for d in waiting}) >= 2, "no beat while it waited"
        assert waiting[0]["pid"] == os.getpid()
        assert counts == 1 and dt >= 6.0, (counts, why, dt)


if __name__ == "__main__":
    test_waiting_follower_is_not_stopped()
    test_follower_that_stops_beating_is_still_stopped()
    test_another_followers_heartbeat_does_not_count()
    test_the_follower_beats_while_it_waits_for_card_memory()
    print("test_products_wait_heartbeat: 4 passed")
