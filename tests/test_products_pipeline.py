"""The products hand-off path's speed changes keep its output, and a follower takes each hour once its file is whole.

grib-writer-speed 2026-11-10: on the 3 km CONUS nest the product path took 88 s per forecast hour (frame 40-48 s,
writer 46-62 s) against the model's 64-68 s, and a follower took each history hour only when the next hour's file
appeared, one model hour late.
"""
from pathlib import Path

import numpy as np
import pytest

from namgpu.io import history as H
from namgpu.io import products as P


def test_history_file_appears_whole(tmp_path: Path, monkeypatch):
    path = tmp_path / "wrfout_d01_2026-01-23_13:00:00"
    seen = {}

    def fake_write(p, payload):
        seen["writing"] = Path(p).name
        seen["final_exists_while_writing"] = path.exists()
        Path(p).write_bytes(b"netcdf")
        return p

    monkeypatch.setattr(H, "_write_payload", fake_write)
    assert H.write_payload(path, {}) == path
    assert seen == {"writing": ".wrfout_d01_2026-01-23_13:00:00.part", "final_exists_while_writing": False}
    assert path.read_bytes() == b"netcdf"
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_failed_history_write_leaves_nothing(tmp_path: Path, monkeypatch):
    path = tmp_path / "wrfout_d01_2026-01-23_14:00:00"

    def broken(p, payload):
        Path(p).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(H, "_write_payload", broken)
    with pytest.raises(OSError):
        H.write_payload(path, {})
    assert list(tmp_path.iterdir()) == []


def test_follower_takes_an_hour_before_the_next_appears(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    started = []
    polls = {"n": 0}

    def fake_start(f, out, profile, device, bits, pool, ps=None, gkey=None, cctx=None, file_name=None):
        started.append(f.name)
        return f

    def fake_finish(h, keep_work=False):
        return {"file": h.name, "history": h.name, "fhr": 0, "bytes": 1, "records": 1, "frame_s": 0.0, "writer_s": 0.0}

    def model(_seconds):
        # the forecast: hour 0 appears (renamed in whole); the run ends only after the follower has taken it
        polls["n"] += 1
        if polls["n"] == 1:
            (run / "wrfout_d01_2026-01-23_12:00:00").write_bytes(b"x")
        elif started and not (run / "forecast-receipt.json").exists():
            (run / "forecast-receipt.json").write_text("{}")
        elif polls["n"] > 20:
            raise AssertionError("the follower never took hour 0 while the run was still going")

    monkeypatch.setattr(P, "_start_hour", fake_start)
    monkeypatch.setattr(P, "_finish_hour", fake_finish)
    monkeypatch.setattr(P, "_xtime_hour", lambda f: 0)
    monkeypatch.setattr(P.time, "sleep", model)
    P.run(run, out=tmp_path / "out", follow=True, poll_s=0.0, set_name="awphys", workers=1)
    assert started == ["wrfout_d01_2026-01-23_12:00:00"]


def test_heavy_jobs_go_first_and_order_is_otherwise_kept():
    jobs = [("h", "T2"), ("h", "QVAPOR"), ("h", "NAM_TSOIL"), ("h", "NAM_CLDBASE"), ("h", "PH"), ("h", "NAM_TCDC")]
    got = [j[1] for j in P._job_order(jobs, {"QVAPOR", "PH"})]
    assert got == ["NAM_CLDBASE", "NAM_TCDC", "QVAPOR", "PH", "T2", "NAM_TSOIL"]


def test_field_handed_through_host_ram_is_the_same_array(tmp_path: Path, monkeypatch):
    a = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / np.float32(7.0))
    a[0, 0, 0] = np.nan
    monkeypatch.setattr(P, "_read_regrid", lambda job: (job[1], (a, ("Time", "south_north", "west_east"))))
    name, v = P._read_regrid_to((str(tmp_path), ("hist", "QSNOW")))
    assert name == "QSNOW" and isinstance(v[0], str)
    b, dims = P._mapped(v)
    assert dims == ("Time", "south_north", "west_east")
    assert b.dtype == a.dtype and np.array_equal(b.view(np.uint32), a.view(np.uint32))
    # no folder: the pipe, as before
    assert P._read_regrid_to((None, ("hist", "QSNOW")))[1][0] is a
    assert P._mapped(None) is None


def test_a_written_hour_is_published_while_the_next_frame_is_built(tmp_path: Path, monkeypatch):
    """The finisher puts hour 0 in place while the main thread is still building hour 1 (it used to wait for it)."""
    import threading
    run = tmp_path / "run"
    run.mkdir()
    for h in (12, 13):
        (run / f"wrfout_d01_2026-01-23_{h}:00:00").write_bytes(b"x")
    (run / "forecast-receipt.json").write_text("{}")
    published = threading.Event()
    waited = {}

    def fake_start(f, out, profile, device, bits, pool, ps=None, gkey=None, cctx=None, file_name=None):
        if f.name.endswith("13:00:00"):
            waited["hour0_published_during_frame1"] = published.wait(10.0)
        return f

    def fake_finish(h, keep_work=False):
        if h.name.endswith("12:00:00"):
            published.set()
        return {"file": h.name, "history": h.name, "fhr": 0, "bytes": 1, "records": 1, "frame_s": 0.0, "writer_s": 0.0}

    monkeypatch.setattr(P, "_start_hour", fake_start)
    monkeypatch.setattr(P, "_finish_hour", fake_finish)
    monkeypatch.setattr(P, "_xtime_hour", lambda f: 0)
    res = P.run(run, out=tmp_path / "out", follow=True, poll_s=0.0, set_name="awphys", workers=4)
    assert waited == {"hour0_published_during_frame1": True}
    assert len(res["files"]) == 2 and not res["errors"]
