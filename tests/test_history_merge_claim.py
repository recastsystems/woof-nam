"""history_merge: a finished hour can never be claimed twice (woof-nam run --cards N).

Before this fix a writer ended its hour with shutil.rmtree(hour). On /dev/shm (tmpfs lists the newest entry first)
rmtree removed the `claim` folder before meta.json and the done.r* flags, so an idle writer that had just found the
hour complete made `claim` again and read a meta.json that was gone: FileNotFoundError, pool exit 1. A 24 h
every-5-minute Fast run with 24 writers died that way at model hour 16. Now a writer claims an
hour by renaming its folder out of the scanned names, which only one writer can do.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from namgpu.io import history_merge


def _fake_payload(path, payload):
    time.sleep(0.03)                     # the zlib encode: claims overlap
    Path(path).write_text(payload["valid"].strftime("%Y-%m-%d_%H:%M:%S"))


_real_complete = history_merge._complete


def _slow_complete(hour, nranks):
    # A loaded box can stop a writer between finding an hour complete and claiming it; this pause stands in for that.
    ok = _real_complete(hour, nranks)
    if ok:
        time.sleep(0.01)
    return ok


def _feed(shm: Path, n_hours: int, nranks: int):
    for k in range(n_hours):
        tag = f"2026-11-15_{k // 12:02d}:{(k % 12) * 5:02d}:00"
        hour = shm / tag
        hour.mkdir()
        for nm in ("xlat", "xlon", "T2"):
            np.zeros((4, 5), dtype=np.float32).tofile(hour / f"{nm}.f32")
        meta = {"valid": tag, "start": "2026-11-15_00:00:00", "xtime_min": 5.0 * k,
                "dims": {"south_north": 4, "west_east": 5}, "cen_lat": 0, "cen_lon": 0, "pole_lat": 90,
                "pole_lon": 0, "dx": 12000, "dy": 12000, "fields": [["T2", ["south_north", "west_east"], []]]}
        (hour / "meta.json.tmp").write_text(json.dumps(meta))
        os.replace(hour / "meta.json.tmp", hour / "meta.json")
        for r in range(nranks):
            (hour / f"done.r{r}").write_text("1")
        time.sleep(0.005)
    (shm / "END").write_text("1")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="the pool forks")
def test_many_writers_many_hours_no_double_claim(monkeypatch, tmp_path):
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else tmp_path
    shm = base / f"namgpu-hist-test-{os.getpid()}"
    out = tmp_path / "out"
    shm.mkdir()
    import namgpu.io.history as history
    monkeypatch.setattr(history, "write_payload", _fake_payload)
    monkeypatch.setattr(history_merge, "_complete", _slow_complete)
    monkeypatch.setattr(mp, "Process", mp.get_context("fork").Process)
    n_hours, nranks = 288, 4
    t = threading.Thread(target=_feed, args=(shm, n_hours, nranks))
    t.start()
    try:
        rc = history_merge.main(["--shm", str(shm), "--out", str(out), "--nranks", str(nranks), "--workers", "24"])
    finally:
        t.join()
    written = sorted(p.name for p in out.glob("wrfout_d01_*"))
    left = sorted(p.name for p in shm.iterdir()) if shm.exists() else []
    for p in shm.iterdir():
        if p.is_dir():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink()
    shm.rmdir()
    assert rc == 0
    assert len(written) == n_hours
    assert left == ["END"]
