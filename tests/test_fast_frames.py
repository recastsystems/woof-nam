"""FAST FRAMES: dense output never makes the forecast wait on the history writers, and a frame off the hour that is only
drawn carries only what the maps read (namgpu/io/io_history.py, namgpu/outputs.py, namgpu/io/history_select.py).

Breakage it prevents: with `--write 5m` every frame was a whole 12 km products history (1.3 GB after zlib, 33 s on one
core), at most four writers ran, and once four files were in flight the step loop waited on the oldest one with the
card about 70 % idle: 96 s for the first model hour against 32.5 s hourly (one RTX PRO 6000 and
16 cores, 2026-11-13). CPU only."""
import os
import sys
import threading
from concurrent.futures import Future
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import cores, outputs  # noqa: E402
from namgpu.io import history_select, io_history  # noqa: E402


@pytest.fixture
def sixteen(monkeypatch):
    """The site's box: 16 usable cores, no writer pool yet, no override."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.delenv("NAMGPU_HISTORY_WRITERS", raising=False)
    monkeypatch.delenv("NAMGPU_HISTORY_QUEUE", raising=False)
    cores.usable.cache_clear()
    monkeypatch.setattr(io_history, "_PROCS", None)
    monkeypatch.setattr(io_history, "_DENSE", False)
    monkeypatch.setattr(io_history, "_WAITS", {"files": 0, "s": 0.0, "max_s": 0.0, "over": 0, "said": 0})
    yield
    cores.usable.cache_clear()


def test_the_step_loop_does_not_wait_for_a_free_writer(sixteen):
    """Six files still being written (more than the writers): the next file is staged at once. Before, the step loop
    sat in files.pop(0).result() until a writer finished."""
    busy = [Future() for _ in range(6)]
    staged = Future()
    staged.set_result(busy[-1])
    ctx = SimpleNamespace(history_files=busy[:-1], history_future=staged)
    t = threading.Thread(target=io_history._ready_for_next, args=(ctx, datetime(2026, 11, 12, 18, 5)), daemon=True)
    t.start()
    t.join(timeout=1.0)
    alive = t.is_alive()
    for f in busy:
        f.set_result("done")             # let a blocked thread end
    t.join(timeout=5.0)
    assert not alive, "the step loop waited on a busy writer"
    assert len(ctx.history_files) == 6 and io_history.waits()["over_1s"] == 0


def test_a_cap_still_bounds_the_queue_when_asked(sixteen, monkeypatch):
    monkeypatch.setenv("NAMGPU_HISTORY_QUEUE", "2")
    done = Future()
    done.set_result("x")
    pending = Future()
    ctx = SimpleNamespace(history_files=[pending, done], history_future=None)
    pending.set_result("y")
    io_history._ready_for_next(ctx)
    assert ctx.history_files == []


def test_a_dense_schedule_takes_the_cores_and_the_hourly_feed_does_not(sixteen):
    io_history.set_schedule(outputs.Schedule(25.0, 84.0))          # the feed: hourly, 84 h
    assert io_history.writers() == 4
    io_history.set_schedule(outputs.Schedule(25.0, 2.0, write="5m"))
    assert io_history.writers() == 16 - io_history.RESERVED_CORES
    io_history.set_schedule(None)
    assert io_history.writers() == 4


def test_the_queue_is_bounded_by_host_memory(sixteen, monkeypatch, tmp_path):
    from namgpu.io import scratch
    monkeypatch.setattr(io_history, "SHM", str(tmp_path))
    monkeypatch.setattr(scratch, "free_bytes", lambda p: 100 << 30)
    monkeypatch.setattr(io_history, "_meminfo", lambda: (34 << 30, 128 << 30))   # 34 GB available of 128
    ok, _ = io_history._shm_room(4 << 30)
    assert not ok                         # 4 GB more would leave under a quarter of the box free
    monkeypatch.setattr(io_history, "_meminfo", lambda: (40 << 30, 128 << 30))
    ok, _ = io_history._shm_room(4 << 30)
    assert ok


def test_a_wait_is_said_in_one_line_and_counted(sixteen, capsys):
    io_history._note_wait(2.5, 9, datetime(2026, 11, 12, 18, 5))
    io_history._note_wait(0.2, 3, datetime(2026, 11, 12, 18, 10))
    out = capsys.readouterr().out
    assert out.count("io-history: the forecast waited 2.5 s") == 1
    w = io_history.waits()
    assert w["files"] == 2 and w["over_1s"] == 1 and abs(w["wait_s"] - 2.7) < 1e-9
    assert "waited 2.7 s on the history path in all" in io_history.summary_line()


# ---------------------------------------------------------------------------------------------- light frames

def test_light_frames_are_the_off_hour_times_only_drawn():
    s = outputs.Schedule(25.0, 3.0, write="5m", carry="0-3/15m", light_fields=outputs.MAP_FIELDS)
    light = sorted(e["model_h"] for e in s.entries.values() if e["light"])
    assert light and all(abs(h * 60 % 60) > 1e-6 for h in light)               # never a whole hour
    assert not any(s.light(e["step"]) for e in s.entries.values() if e["grib"])
    assert not any(abs(h * 4 - round(h * 4)) < 1e-6 for h in light)            # a nest reads the quarter hours
    assert len(light) == 33 - 9                                                # 33 off-hour 5-min times, 9 carried
    doc = s.doc(datetime(2026, 11, 12, 18))
    assert doc["light_fields"] == list(outputs.MAP_FIELDS)
    assert next(t for t in doc["times"] if t["tag"] == "f000m05")["light"] is True
    assert next(t for t in doc["times"] if t["tag"] == "f001")["light"] is False


def test_no_light_frames_without_light_fields_or_on_the_hourly_feed():
    assert not any(e["light"] for e in outputs.Schedule(25.0, 3.0, write="5m").entries.values())
    feed = outputs.Schedule(25.0, 84.0, light_fields=outputs.MAP_FIELDS)
    assert not any(e["light"] for e in feed.entries.values()) and len(feed.entries) == 85


def test_a_light_frame_writes_what_the_maps_read():
    s = outputs.Schedule(25.0, 2.0, write="5m", light_fields=outputs.MAP_FIELDS)
    names = list(history_select.PRESETS["products"]) + ["NEST_HANDOFF_X"]
    fields = {n: (None, ("south_north", "west_east")) for n in names}
    sel = history_select.Selection("products", 432)
    ctx = SimpleNamespace(outputs=s, history_select=sel, ntimestep=12)        # 5 minutes in: light
    got = history_select.select(ctx, fields)
    assert set(got) == (set(outputs.MAP_FIELDS) & set(names)) | {"NEST_HANDOFF_X"}
    assert {"T", "QVAPOR", "PRES", "PINT", "U", "V", "COMPOSITE", "RAINNC"} <= set(got)
    assert "W" not in got and "QCLOUD" not in got
    ctx.ntimestep = 144                                                         # hour 1: the whole products set
    assert set(history_select.select(ctx, fields)) == set(names)


def test_every_map_need_is_in_a_light_frame():
    from namgpu.io import maps
    need = set()
    for p in ("composite_reflectivity", "t2m", "d2m", "wind10", "wind850", "mslp", "qpf_1h", "qpf_6h"):
        need.update(maps.NEEDS[p])
    need.update(("T", "QVAPOR", "PRES", "PINT", "HGT"))                         # MSLET (maps._mslet)
    assert need <= set(outputs.MAP_FIELDS)


def test_the_sounding_store_skips_light_frames(tmp_path):
    """Breakage it prevents: woof-nam columns read every listed frame, and a light frame has no QCLOUD, W or PH, so the
    store failed on it and the run ended rc 1."""
    from namgpu.io import columns
    s = outputs.Schedule(25.0, 1.0, write="5m", light_fields=outputs.MAP_FIELDS)
    start = datetime(2026, 11, 12, 18)
    s.write(tmp_path, start)
    for t in s.doc(start)["times"]:
        (tmp_path / t["file"]).write_bytes(b"x")
    got = [f.name for f in columns.frames_of([tmp_path])]
    assert got == ["wrfout_d01_2026-11-12_18:00:00", "wrfout_d01_2026-11-12_19:00:00"]


def test_a_frame_off_the_hour_has_no_mrms_overlay(tmp_path):
    from namgpu.io import maps
    (tmp_path / "mrms_refc_2026111218.grib2").write_bytes(b"x")
    assert maps.mrms_file(tmp_path, datetime(2026, 11, 12, 18)) == tmp_path / "mrms_refc_2026111218.grib2"
    assert maps.mrms_file(tmp_path, datetime(2026, 11, 12, 18, 5)) is None
    assert maps.mrms_file(tmp_path, datetime(2026, 11, 12, 19)) is None


def test_a_site_title_says_minutes_off_the_hour():
    from namgpu.io import maps
    start = datetime(2026, 11, 12, 18)
    assert maps.lead_words(start, datetime(2026, 11, 12, 21)) == "+3 h"           # whole hours: as before
    assert maps.lead_words(start, datetime(2026, 11, 12, 19, 35)) == "+1h35m"
