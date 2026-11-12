"""woof-nam products stopped by a signal leaves no field workers behind, and on SIGTERM no frame folder either.

Breakage it prevents (nam-differences-2, 2026-11-11): a `woof-nam products` job stopped by its PID while it built a
frame left its 12 field workers, their forkserver and its /dev/shm frame folder (woof-nam-frame-<pid>-*) on the shared
node; they were removed by hand. The default SIGTERM ends Python with no cleanup, and nothing ends a forkserver's
workers once their parent is gone. CPU only: a two-hour run whose frame is a stub (hour 0 hands a writer stub that
sleeps; hour 1 keeps the field pool busy through the real worker entry, _read_regrid_to, with a sleeping field read),
the frame folder in a temporary folder (NAMGPU_PRODUCTS_SHM), stopped while hour 1 is being built."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(r"""
    import json, multiprocessing, os, shutil, sys, time
    from datetime import datetime
    from pathlib import Path
    from namgpu import exporter
    from namgpu.io import products as P
    tmp = Path(sys.argv[1]); rd = tmp / "run"; writer = tmp / "rw_rrfs_upp"
    exporter.ensure = lambda *a, **k: writer
    exporter.require = lambda *a, **k: writer
    P._exporter = lambda: writer
    P.card_visible = lambda: (True, "test")
    P._read_regrid = lambda job: (time.sleep(600), (job[1], None))[1]     # a field read that takes long

    def writers():
        # this process's own children that run the writer command (never another process's)
        out = []
        for d in os.listdir("/proc"):
            try:
                if not d.isdigit():
                    continue
                ppid = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]
                if ppid == str(os.getpid()) and \
                        b"--profile-json" in open(f"/proc/{d}/cmdline", "rb").read().split(b"\0"):
                    out.append(int(d))
            except (OSError, IndexError):
                pass
        return out

    def frame(history, dst, pool=None, ps=None, gkey=None, cctx=None):
        shm = P._frame_shm(Path(history), 1)
        try:
            Path(dst).write_bytes(b"frame")
            if Path(history).name.endswith("13:00:00"):
                futs = [pool.submit(P._read_regrid_to, (shm, (str(history), f"F{i}"))) for i in range(8)]
                time.sleep(3)
                (tmp / "state.json").write_text(json.dumps({
                    "workers": [p.pid for p in multiprocessing.active_children()],
                    "writers": writers(), "shm": shm}))
                [f.result() for f in futs]
            return {"start": datetime(2024, 11, 5, 12), "valid": datetime(2024, 11, 5, 12), "notes": []}
        finally:
            if shm:
                shutil.rmtree(shm, ignore_errors=True)

    if __name__ == "__main__":        # the pool's forkserver imports this file again as __mp_main__
        P.wrf_frame = frame
        sys.exit(P.main(["--run", str(rd), "--set", "awphys", "--workers", "2"]))
""")


def _alive(pid: int) -> bool:
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return st != "Z"


def _ours(tmp: Path) -> list[int]:
    """Live processes whose command line names this test's own temporary folder (the stub run, its writer stub, its
    forkserver and field workers): nothing else on the machine can match."""
    key = str(tmp).encode()
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if key in Path(f"/proc/{d}/cmdline").read_bytes() and _alive(int(d)):
                    out.append(int(d))
            except OSError:
                pass
    return out


def _stub_run(sig: int, settle: int = 40):
    """The stub run, stopped by `sig` while hour 1 is built. Returns (state, [(pid, role)] of it still alive after up
    to `settle` x 0.25 s, the run's exit code, its output, hour work folders left, whether the frame folder is left).
    Every stub process is killed before it returns, pass or fail."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "run").mkdir()
        for h in ("12", "13"):
            (tmp / "run" / f"wrfout_d01_2024-11-05_{h}:00:00").write_bytes(b"")
        (tmp / "shm").mkdir()
        w = tmp / "rw_rrfs_upp"
        w.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(600)\n")
        w.chmod(0o755)
        (tmp / "child.py").write_text(CHILD)
        env = dict(os.environ, PYTHONPATH=str(ROOT), NAMGPU_PRODUCTS_SHM=str(tmp / "shm"), NAMGPU_SHM=str(tmp / "shm"))
        logf = open(tmp / "child.log", "w")
        p = subprocess.Popen([sys.executable, str(tmp / "child.py"), str(tmp)], env=env, cwd=str(tmp),
                             stdout=logf, stderr=subprocess.STDOUT)
        state, left = None, []
        try:
            for _ in range(240):
                if (tmp / "state.json").exists():
                    state = json.loads((tmp / "state.json").read_text())
                    break
                if p.poll() is not None:
                    break
                time.sleep(0.25)
            assert state is not None, f"the stub run did not reach hour 1: {(tmp / 'child.log').read_text()[-2000:]}"
            assert state["workers"] and state["writers"] and state["shm"], state
            p.send_signal(sig)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
            for _ in range(settle):               # a moment for the forkserver and workers to end after the parent
                left = _ours(tmp)
                if not left or (sig == signal.SIGKILL and set(left) <= set(state["writers"])):
                    break
                time.sleep(0.25)
            left = [(x, "writer" if x in state["writers"] else "worker" if x in state["workers"] else "forkserver")
                    for x in left]
        finally:
            if p.poll() is None:
                p.kill()
            for x in _ours(tmp):                  # never leave a stub behind, even when the test fails
                try:
                    os.kill(x, signal.SIGKILL)
                except OSError:
                    pass
            logf.close()
        work = sorted(x.name for x in (tmp / "run" / "products").glob(".products-*"))
        return state, left, p.returncode, (tmp / "child.log").read_text(), work, Path(state["shm"]).exists()


def test_sigterm_leaves_nothing():
    state, left, rc, out, work, shm_left = _stub_run(signal.SIGTERM)
    assert not left, f"processes left after SIGTERM: {left} (state {state}); output: {out[-1500:]}"
    assert not shm_left, f"frame folder left: {state['shm']}"
    assert not work, f"hour work folders left: {work}"
    assert rc == 128 + signal.SIGTERM, f"exit {rc}: {out[-1500:]}"


def test_sighup_leaves_nothing():
    """SIGHUP (the terminal or session that started the job closes) stops the run as SIGTERM does."""
    state, left, rc, out, work, shm_left = _stub_run(signal.SIGHUP)
    assert not left, f"processes left after SIGHUP: {left} (state {state}); output: {out[-1500:]}"
    assert not shm_left, f"frame folder left: {state['shm']}"
    assert not work, f"hour work folders left: {work}"
    assert rc == 128 + signal.SIGHUP, f"exit {rc}: {out[-1500:]}"


def test_sigkill_leaves_no_pool():
    """SIGKILL cannot be caught: the field workers and their forkserver still end within seconds (each worker watches
    its products process). The writer is a separate program that ends with its hour, and the frame folder is removed
    by the next products or run start (scratch.sweep, once its owner has ended)."""
    state, left, rc, out, work, shm_left = _stub_run(signal.SIGKILL, settle=60)
    pool_left = [x for x in left if x[1] != "writer"]
    assert not pool_left, f"pool processes left 15 s after SIGKILL: {pool_left} (state {state}); output: {out[-1500:]}"


if __name__ == "__main__":
    test_sigterm_leaves_nothing()
    test_sighup_leaves_nothing()
    test_sigkill_leaves_no_pool()
    print("test_products_stop_cleanup: 3 passed")
