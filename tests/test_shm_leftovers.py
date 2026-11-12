"""woof-nam host-RAM scratch: a failed run leaves nothing in /dev/shm, and the room is read before it is filled.

Breakage these prevent (0.2.1 candidate, arctic-front 2020-10-24 00Z, 2026-11-11): the column store's 16
workers wrote 16 uncompressed views (45 GB) to /dev/shm beside the GRIB2 follower, the kernel killed two at the
session's 80 GiB memory.max, the pool broke (BrokenProcessPool) and the 16 views, about 43 GB of a 62 GB /dev/shm,
were never removed. The next run died of a bus error at +3 h (rc 135, no message) with those views still in
/dev/shm, and left its two 3.6 GB psm_* blocks (that the tmpfs was full at that moment was not measured). CPU only: small synthetic history files, a stub store writer,
the room and the kills simulated."""
import os
import signal
import subprocess
import sys
import textwrap
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.io import columns  # noqa: E402

try:
    from namgpu.io import scratch  # noqa: E402
except ImportError:              # before the fix there was no scratch module: those tests fail, the others run
    scratch = types.SimpleNamespace(free_bytes=lambda p: 0, in_ram=lambda p: True)

GB = 1024 ** 3


def _history(path: Path, nz=6, ny=20, nx=24, zlib=False):
    """A small WOOF-NAM history with the fields the WRF view reads (winds at mass points, PRES, PH, HGT)."""
    import netCDF4
    rng = np.random.default_rng(len(path.name))
    with netCDF4.Dataset(path, "w", format="NETCDF4") as d:
        d.createDimension("Time", None)
        for k, n in (("bottom_top", nz), ("bottom_top_stag", nz + 1), ("south_north", ny), ("west_east", nx)):
            d.createDimension(k, n)
        d3 = ("Time", "bottom_top", "south_north", "west_east")
        for n in ("U", "V", "W", "PRES", "T", "QVAPOR"):
            d.createVariable(n, "f4", d3, zlib=zlib)[0] = (np.zeros if zlib else rng.random)((nz, ny, nx))
        d.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"), zlib=zlib)[0] = \
            np.zeros((nz + 1, ny, nx))
        d.createVariable("HGT", "f4", ("Time", "south_north", "west_east"), zlib=zlib)[0] = np.zeros((ny, nx))
    return path


def _frames(d: Path, n: int, **kw):
    d.mkdir(parents=True, exist_ok=True)
    return [_history(d / f"wrfout_d01_2020-10-24_{h:02d}:00:00", **kw) for h in range(n)]


STUB = textwrap.dedent("""\
    #!{py}
    # stands in for rw_soundings --columns: reads the request, sleeps, prints the COLUMNS line; with STUB_KILL set it
    # kills its worker (its parent) once, as the kernel's OOM killer did, while the other workers sit in their views
    import json, os, signal, sys, time
    req = json.loads(open(sys.argv[2]).read())
    src = req["sources"][0]["path"]
    kill = os.environ.get("STUB_KILL")
    if kill and kill in src and not os.path.exists(os.environ["STUB_MARK"]):
        open(os.environ["STUB_MARK"], "w").close()
        time.sleep(0.5)
        os.kill(os.getppid(), signal.SIGKILL)
        sys.exit(0)
    time.sleep(float(os.environ.get("STUB_SLEEP", "1.5")))
    out = os.path.join(req["out"], "d01", os.path.basename(src) + ".wcol")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("x")
    print("COLUMNS " + out + " bytes=1 seconds=0.1")
    """)


def _stub(d: Path) -> str:
    p = d / "rw_soundings_stub"
    p.write_text(STUB.format(py=sys.executable))
    p.chmod(0o755)
    # columns.write() looks the store writer up in gpuwm.soundings (not installed in the CPU job)
    g = sys.modules.setdefault("gpuwm", types.ModuleType("gpuwm"))
    s = types.ModuleType("gpuwm.soundings")
    s.binary = lambda: p
    g.soundings = s
    sys.modules["gpuwm.soundings"] = s
    return str(p)


def _left(d: Path) -> list:
    return sorted(e.name for e in d.iterdir()) if d.is_dir() else []


def _views(d: Path) -> list:
    """The column views there now. While a store runs its owner lock file (scratch.own) is beside them."""
    return [n for n in _left(d) if n.startswith("woof-nam-columns-")]


def test_killed_worker_leaves_no_views_and_the_store_completes(tmp_path):
    frames = _frames(tmp_path / "run", 4)
    shm = tmp_path / "shm"
    shm.mkdir()
    _stub(tmp_path)
    os.environ.update(STUB_KILL="02:00:00", STUB_MARK=str(tmp_path / "killed"), STUB_SLEEP="2")
    lines, err = [], None
    try:
        kw = {"log": lines.append} if "log" in columns.write.__code__.co_varnames else {}
        res = columns.write(frames, tmp_path / "run" / "columns", workers=4, scratch=str(shm), **kw)
    except Exception as e:  # noqa: BLE001
        res, err = None, e
    finally:
        for k in ("STUB_KILL", "STUB_MARK", "STUB_SLEEP"):
            os.environ.pop(k, None)
    print("left in scratch:", _left(shm), "| error:", repr(err), "| lines:", lines)
    assert (tmp_path / "killed").exists(), "the stub did not kill its worker"
    assert _left(shm) == [], f"views left in scratch after a killed worker: {_left(shm)}"
    assert err is None, f"the store raised {err!r} instead of writing the frames again"
    assert res["failed"] == 0 and len(res["files"]) == 4, res
    assert any("ended abruptly" in m and "written again" in m for m in lines), lines
    assert not (tmp_path / "run" / "columns" / ".scratch").exists()


def test_short_scratch_room_goes_to_disk_and_says_so(tmp_path, monkeypatch):
    frames = _frames(tmp_path / "run", 2)
    shm = tmp_path / "shm"
    shm.mkdir()
    _stub(tmp_path)
    real = scratch.free_bytes
    monkeypatch.setattr(scratch, "free_bytes", lambda p: 0 if str(p).startswith(str(shm)) else real(p))
    os.environ["STUB_SLEEP"] = "0"
    lines = []
    try:
        kw = {"log": lines.append} if "log" in columns.write.__code__.co_varnames else {}
        res = columns.write(frames, tmp_path / "run" / "columns", workers=2, scratch=str(shm), **kw)
    finally:
        os.environ.pop("STUB_SLEEP", None)
    print("lines:", lines)
    assert res["failed"] == 0, res
    assert any("GB free" in m and ".scratch" in m for m in lines), f"no line about the short room: {lines}"
    assert _left(shm) == [] and not (tmp_path / "run" / "columns" / ".scratch").exists()


def test_scratch_bound_counts_the_uncompressed_view_and_the_free_room(tmp_path):
    # compressed history (zeros, zlib): the file is a small part of its view, as the 12 km files are (1.24 GB -> 2.85 GB)
    frames = _frames(tmp_path / "run", 8, zlib=True)
    size = max(f.stat().st_size for f in frames)
    view = columns.view_bytes(frames) if hasattr(columns, "view_bytes") else None
    assert view is not None and view > 2 * size, (view, size)
    n = columns.auto_workers(frames, 64, str(tmp_path), mem_available=10 ** 15, scratch_free=int(2.5 * view))
    assert n == 2, f"room for 2.5 views must give 2 workers, got {n} (view {view} B, file {size} B)"


def test_arctic_front_numbers(monkeypatch):
    # 37 hourly 12 km files of 1.238 GB (no header readable: VIEW_RATIO), /dev/shm with 14 GB free after an earlier
    # run's leftovers, and a session at its 80 GiB cap: the bound must leave the views and their memory room
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        frames = []
        for i in range(37):
            f = Path(d) / f"wrfout_d01_2020-10-24_{i:02d}:00:00"
            with open(f, "wb") as fh:
                fh.truncate(1_238_400_334)
            frames.append(f)
        monkeypatch.setattr(scratch, "in_ram", lambda p: True)
        view = columns.view_bytes(frames)
        n_room = columns.auto_workers(frames, 24, "/dev/shm", mem_available=10 ** 15, scratch_free=14 * 10 ** 9)
        assert n_room * view <= 14 * 10 ** 9 * 0.9 and n_room == 2, (n_room, view)
        n_mem = columns.auto_workers(frames, 24, "/dev/shm", mem_available=80 * GB, scratch_free=10 ** 13)
        assert n_mem * 2 * view <= 0.6 * 80 * GB and n_mem == 6, (n_mem, view)


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_sweep_removes_only_ended_runs(tmp_path):
    dead, old = _dead_pid(), time.time() - 7200
    (tmp_path / f"woof-nam-columns-{dead}-abc").mkdir()
    (tmp_path / f"woof-nam-columns-{dead}-abc" / "wrfout_d01_x").write_bytes(b"x" * 1000)
    (tmp_path / f"woof-nam-hist-{dead}-0").write_bytes(b"x" * 64)
    (tmp_path / f"woof-nam-hist-{dead}-1").write_bytes(b"x" * 64)            # young: kept (MIN_AGE_S)
    (tmp_path / f"woof-nam-frame-{os.getpid()}-live").mkdir()                 # this process: kept
    (tmp_path / "woof-nam-columns-k2j3h4g5").mkdir()                          # 0.2.0 name, two hours old
    (tmp_path / "psm_5a8c8585").write_bytes(b"x")                             # not the package's: kept
    for n in (f"woof-nam-columns-{dead}-abc", f"woof-nam-hist-{dead}-0", "woof-nam-columns-k2j3h4g5",
              f"woof-nam-frame-{os.getpid()}-live", "psm_5a8c8585"):
        os.utime(tmp_path / n, (old, old))
    lines = []
    gone = scratch.sweep(tmp_path, log=lines.append)
    assert sorted(gone) == sorted([f"woof-nam-columns-{dead}-abc", f"woof-nam-hist-{dead}-0",
                                   "woof-nam-columns-k2j3h4g5"]), gone
    assert sorted(_left(tmp_path)) == sorted([f"woof-nam-hist-{dead}-1", f"woof-nam-frame-{os.getpid()}-live",
                                              "psm_5a8c8585"])
    assert len(lines) == 1 and "removed 3 leftovers" in lines[0], lines


def test_cap_limits_the_room(tmp_path, monkeypatch):
    (tmp_path / f"woof-nam-hist-{os.getpid()}-0").write_bytes(b"x" * (3 << 20))
    monkeypatch.setattr(scratch, "in_ram", lambda p: True)
    monkeypatch.setenv("NAMGPU_SHM_CAP_MIB", "5")
    assert scratch.free_bytes(tmp_path) == 2 << 20
    monkeypatch.setenv("NAMGPU_SHM_CAP_MIB", "1")
    assert scratch.free_bytes(tmp_path) == 0


def test_cap_is_on_host_ram_only(tmp_path, monkeypatch):
    # the cap is on host RAM: the column store's disk scratch (<out>/.scratch) keeps its own free space. Breakage it
    # prevents: under the cap each frame's view, already on disk, was reported as short of room again (proof leg B)
    monkeypatch.setattr(scratch, "in_ram", lambda p: False)
    monkeypatch.setenv("NAMGPU_SHM_CAP_MIB", "1")
    assert scratch.free_bytes(tmp_path) == scratch.disk_free(tmp_path) > (1 << 20)


class _Event:
    def synchronize(self):
        pass


def _staged(ny=30, nx=40, nz=5):
    rng = np.random.default_rng(1)
    return {"event": _Event(), "keep": None,
            "staged": [("T", rng.random((nz, ny, nx), dtype=np.float32), ("bottom_top", "south_north", "west_east")),
                       ("T2", rng.random((ny, nx), dtype=np.float32), ("south_north", "west_east"))],
            "valid": datetime(2020, 10, 24, 3), "start": datetime(2020, 10, 24, 0),
            "xlat": np.zeros((ny, nx), np.float32), "xlon": np.zeros((ny, nx), np.float32), "xtime_min": 180.0,
            "dims": {"west_east": nx, "south_north": ny, "bottom_top": nz}, "cen_lat": 50.0, "cen_lon": -100.0,
            "pole_lat": 50.0, "pole_lon": -111.0, "dx": 12000.0, "dy": 12000.0}


def _blocks() -> list:
    return sorted(p.name for p in Path("/dev/shm").glob(f"woof-nam-hist-{os.getpid()}-*"))


def test_history_block_short_room_writes_in_thread_and_says_so(tmp_path, monkeypatch, capsys):
    from namgpu.io import io_history
    monkeypatch.setattr(scratch, "free_bytes", lambda p: 10 << 20)     # 10 MiB free: under the 2 GiB reserve
    before = sorted(p.name for p in Path("/dev/shm").glob("psm_*"))
    res = io_history._stage_and_submit(tmp_path / "wrfout_d01_2020-10-24_03:00:00", _staged())
    res = res.result() if hasattr(res, "result") else res
    out = capsys.readouterr().out
    assert Path(res).is_file()
    assert "written by the writer thread" in out, out
    assert _blocks() == [] and sorted(p.name for p in Path("/dev/shm").glob("psm_*")) == before


def _die(path, shm_name, meta):
    os.kill(os.getpid(), signal.SIGKILL)


def test_history_block_named_and_freed_also_when_its_writer_is_killed(tmp_path, monkeypatch):
    from namgpu.io import io_history
    if not Path("/dev/shm").is_dir():
        return
    monkeypatch.setenv("NAMGPU_HISTORY_WRITERS", "1")
    io_history.shutdown()
    monkeypatch.setattr(scratch, "free_bytes", lambda p: 100 * GB)
    try:
        fut = io_history._stage_and_submit(tmp_path / "wrfout_d01_2020-10-24_03:00:00", _staged())
        assert hasattr(fut, "result")
        fut.result(timeout=120)
        assert (tmp_path / "wrfout_d01_2020-10-24_03:00:00").is_file()
        time.sleep(0.2)
        assert _blocks() == [], _blocks()
        monkeypatch.setattr(io_history, "write_from_shm", _die)
        fut = io_history._stage_and_submit(tmp_path / "wrfout_d01_2020-10-24_04:00:00", _staged())
        assert any(n.startswith(f"woof-nam-hist-{os.getpid()}-") for n in io_history._LIVE) or fut.done()
        try:
            fut.result(timeout=120)
        except Exception as e:  # noqa: BLE001
            assert type(e).__name__ == "BrokenProcessPool", repr(e)
        time.sleep(0.2)
        assert _blocks() == [], f"a killed writer's block is left: {_blocks()}"
    finally:
        io_history.shutdown()
        for p in Path("/dev/shm").glob(f"woof-nam-hist-{os.getpid()}-*"):
            p.unlink()


def test_sigterm_during_the_store_leaves_no_views(tmp_path):
    frames = _frames(tmp_path / "run", 3)
    shm = tmp_path / "shm"
    shm.mkdir()
    stub = _stub(tmp_path)
    code = textwrap.dedent(f"""
        import sys, types
        sys.path.insert(0, {str(ROOT)!r})
        from pathlib import Path
        from namgpu.io import columns
        frames = [Path(p) for p in {[str(f) for f in frames]!r}]
        columns.write(frames, Path({str(tmp_path / "run" / "columns")!r}), workers=3, scratch={str(shm)!r},
                      **({{"exe": {stub!r}}} if "exe" in columns.write.__code__.co_varnames else {{}}))
        """)
    if "exe" not in columns.write.__code__.co_varnames:
        code = code.replace("from namgpu.io import columns",
                            "g = sys.modules.setdefault('gpuwm', types.ModuleType('gpuwm')); "
                            "s = types.ModuleType('gpuwm.soundings'); s.binary = lambda: %r; g.soundings = s; "
                            "sys.modules['gpuwm.soundings'] = s\nfrom namgpu.io import columns" % stub)
    env = dict(os.environ, STUB_SLEEP="60")
    p = subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=True)
    try:
        t0 = time.time()
        while len(_views(shm)) < 3 and time.time() - t0 < 60:
            time.sleep(0.2)
        assert len(_views(shm)) == 3, _left(shm)
        time.sleep(1.0)
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=60)
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)      # the stubs sleeping on (orphans of the terminated workers)
        except ProcessLookupError:
            pass
    assert _left(shm) == [], f"views left after SIGTERM: {_left(shm)}"
    assert rc != 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
