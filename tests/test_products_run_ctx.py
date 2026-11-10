"""A nest products follower that starts before its door has written the configfile still gets the physics step.

suite-1h-e (fa16f9a): the suite started every products follower at 01:39:31Z and the nest doors between 01:39:34Z and
01:48:23Z. The follower read the run context once, when it started, found no nam_configfile, and every nest file lost
the instantaneous PRATE record (786/799 records against 787/800) with no message.
"""
from pathlib import Path

from namgpu.io import products as P

CFG = "dt_int: 2  # s\ndt_num: 0\ndt_den: 4\nnphs: 4\n"


def test_run_ctx_reads_the_configfile_beside_the_run(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    assert P._run_ctx(run) == {}
    (tmp_path / "nam_configfile_firewxnest").write_text(CFG)
    ctx = P._run_ctx(run)
    assert ctx["dt"] == 2.0 and ctx["nphs"] == 4 and ctx["dtq2"] == 8.0


def test_follower_started_before_the_door_gets_the_step_for_every_hour(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    seen = []

    def fake_start(f, out, profile, device, bits, pool, ps=None, gkey=None, cctx=None, file_name=None):
        seen.append((f.name, dict(cctx or {})))
        return f

    def fake_finish(h, keep_work=False):
        return {"file": h.name, "history": h.name, "fhr": len(seen), "bytes": 1, "records": 1,
                "frame_s": 0.0, "writer_s": 0.0}

    def door(_seconds):
        # the nest door, after the follower's first poll: its configfile, two history hours, the receipt
        if not (tmp_path / "nam_configfile_firewxnest").exists():
            (tmp_path / "nam_configfile_firewxnest").write_text(CFG)
            (run / "wrfout_d01_2026-01-23_12:00:00").write_bytes(b"x")
            (run / "wrfout_d01_2026-01-23_13:00:00").write_bytes(b"x")
            (run / "forecast-receipt.json").write_text("{}")

    grid = type("G", (), {"lat": None, "lon": None, "dx_m": 3000.0})
    outg = type("O", (), {"nx": 1, "ny": 1, "spec": lambda self: {}})
    win = type("W", (), {"nx": 1, "ny": 1})
    monkeypatch.setattr(P, "_start_hour", fake_start)
    monkeypatch.setattr(P, "_finish_hour", fake_finish)
    monkeypatch.setattr(P, "_xtime_hour", lambda f: 0)
    monkeypatch.setattr(P, "SourceGrid", lambda f: grid())
    monkeypatch.setattr(P.PSETS, "resolve", lambda ps, lat, lon, dx: outg())
    monkeypatch.setattr(P, "nest_window", lambda sg, og: (0, 0, win()))
    monkeypatch.setattr(P, "pad_attrs", lambda og, i0, j0: {})
    monkeypatch.setattr(P.time, "sleep", door)

    P.run(run, out=tmp_path / "out", follow=True, poll_s=0.0, set_name="firewxnest", workers=1)

    assert [n for n, _ in seen] == ["wrfout_d01_2026-01-23_12:00:00", "wrfout_d01_2026-01-23_13:00:00"]
    assert all(c.get("dtq2") == 8.0 for _, c in seen), seen
