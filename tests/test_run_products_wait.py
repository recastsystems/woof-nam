"""woof-nam run --products returns only when the GRIB2 follower is done (products/DONE), or fails in one line.

Breakage it prevents (0.2.1 candidate, 2026-11-11): the run door returned while its follower was still writing the
last hours, saying only that DONE would appear later; a batch system that ends the process tree when the command
exits lost those hours, and the proof scripts had to loop on products/DONE after the door. CPU only: the follower that
run_forecast._start_products starts is swapped for a stub (it writes DONE late, fails an hour, or stalls). Also: a
run that stops before its receipt stops its follower (the real run of door-products-3 stopped early and left it)."""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import run_forecast as rf  # noqa: E402

STUBS = {
    # writes one file, then DONE after 3 s, exit 0
    "late": "import json,sys,time,pathlib; o=pathlib.Path(sys.argv[sys.argv.index('--run')+1])/'products'; "
            "o.mkdir(exist_ok=True); (o/'woof-nam.t12z.awphys00.tm00.grib2').write_bytes(b'x'); time.sleep(3); "
            "(o/'DONE').write_text(json.dumps({'files': 1, 'errors': 0}))",
    # hour +3 fails: index.json names it, DONE says 1 error, exit 1
    "fail": "import json,sys,time,pathlib; o=pathlib.Path(sys.argv[sys.argv.index('--run')+1])/'products'; "
            "o.mkdir(exist_ok=True); time.sleep(1); e='wrfout_d01_2024-11-05_15:00:00: RuntimeError: rw_rrfs_upp "
            "exit 1: stub'; (o/'index.json').write_text(json.dumps({'errors': [e]})); "
            "(o/'DONE').write_text(json.dumps({'files': 3, 'errors': 1})); sys.exit(1)",
    # writes nothing, ever
    "stall": "import time; time.sleep(600)",
}


def _start(out: Path, kind: str):
    real = subprocess.Popen
    made = []

    def popen(argv, **kw):
        made.append(real([argv[0], "-c", STUBS[kind]] + list(argv[3:]), **kw))
        return made[-1]
    rf.subprocess.Popen = popen
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rf._start_products(out, "awphys")
        return json.loads((out / rf.FOLLOWER).read_text())["pid"]
    except BaseException:
        for p in made:                    # a failing test leaves no stub running
            p.kill()
            p.wait()
        raise
    finally:
        rf.subprocess.Popen = real


def _finish(out: Path, **kw):
    buf = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(buf):
        rc = rf.products_finish(out, 0, **kw)
    return rc, buf.getvalue(), time.monotonic() - t0


def test_returns_only_after_done():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        _start(out, "late")
        rc, text, _ = _finish(out)
        assert (out / "products" / "DONE").is_file(), "the run returned before products/DONE was written"
        assert rc == 0, text
        assert "1 GRIB2 files written" in text, text
        assert "the run waited" in text and "after the forecast ended" not in text, text


def test_failed_follower_names_the_hour():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        _start(out, "fail")
        rc, text, _ = _finish(out)
        assert rc != 0, text
        line = [ln for ln in text.splitlines() if "FAILED" in ln]
        assert len(line) == 1 and "wrfout_d01_2024-11-05_15:00:00" in line[0], text


def test_stalled_follower_is_stopped():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        pid = _start(out, "stall")
        rc, text, wall = _finish(out, stall_s=2.0)
        try:
            assert rc != 0 and "wrote nothing new for 2 s" in text, text
            assert wall < 60, wall
            try:
                os.kill(pid, 0)
                alive = Path(f"/proc/{pid}").exists() and \
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
            except OSError:
                alive = False
            assert not alive, "the stalled follower is still running"
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def test_no_products_wait_returns_at_once():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        pid = _start(out, "stall")
        try:
            rc, text, wall = _finish(out, wait=False)
            assert rc == 0 and wall < 5, (rc, wall, text)
            assert "still writing" in text, text
        finally:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def test_run_stopped_before_receipt_stops_follower():
    """A run that stops before its receipt (here: a missing analysis file, 2026-11-11) does not leave its follower
    waiting for history; with a receipt the follower is left to finish."""
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        pid = _start(out, "stall")
        try:
            (out / "forecast-receipt.json").write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                assert not rf.stop_orphan_follower(), "a follower of a run with a receipt was stopped"
            (out / "forecast-receipt.json").unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert rf.stop_orphan_follower(), "the follower of a run with no receipt was left running"
            assert rf._FOLLOWER_PROC.poll() is not None and "stopped" in buf.getvalue(), buf.getvalue()
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def test_nest_door_waits_and_fails_with_6():
    from namgpu.nest import cli as nest_cli
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        _start(out, "fail")
        rec = {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = nest_cli.wait_products(out, rec, 0)
        assert code == 6 and "wrfout_d01_2024-11-05_15:00:00" in rec["failed_stage"], (code, rec)


if __name__ == "__main__":
    test_returns_only_after_done()
    test_failed_follower_names_the_hour()
    test_stalled_follower_is_stopped()
    test_no_products_wait_returns_at_once()
    test_run_stopped_before_receipt_stops_follower()
    test_nest_door_waits_and_fails_with_6()
    print("test_run_products_wait: 6 passed")
