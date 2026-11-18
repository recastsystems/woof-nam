"""The small engine gaps left on 2026-11-13, one test each, every one failing before 0.2.2:

1. `--products` (auto) on a box that is not one of NCEP's nests picked grid 218 and failed.
2. A window from IFS, AIFS or AIGFS longer than the source's own forecast reaches was planned and only failed in the
   fetch; `woof-nam sources --json` said IFS 0.25 posts to 360 h while its fetch read to 90 h.
3. A suite from a GFS hour that is not posted counted its output times from the posted hour, not the hour asked.
4. The nest door had no resume: a box taken back mid-run ran again from hour 0.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


def _history(path: Path, nx: int, ny: int):
    from netCDF4 import Dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as d:
        d.createDimension("west_east", nx)
        d.createDimension("south_north", ny)


# ---- 1. --products auto on a box
def test_products_auto_on_a_box_is_the_box_set_even_when_the_door_names_its_configfile(tmp_path, monkeypatch):
    """The nest door runs with NAMGPU_CONFIGFILE set to the box's own configfile, so a box always matched "the parent
    size" and auto chose awphys (grid 218): exit 6 at 12 km and 3 km. The 12 km grid is still awphys."""
    from namgpu.io import products
    from namgpu.nest import domain
    g = domain.nest_grid(38.5, -97.5, 1500.0, 12)
    cfgfile = tmp_path / "nam_configfile_box"
    domain.write_configfile(cfgfile, g, "box", tmp_path / "static.npz")
    monkeypatch.setenv("NAMGPU_CONFIGFILE", str(cfgfile))
    _history(tmp_path / "box" / "run" / "wrfout_d01_2026-11-12_15:00:00", int(g["im"]), int(g["jm"]))
    _history(tmp_path / "c12" / "wrfout_d01_2026-11-12_15:00:00", 954, 835)
    assert products.detect_set(tmp_path / "box" / "run") == "hiresf"
    assert products.detect_set(tmp_path / "c12") == "awphys"


def test_products_auto_before_the_first_hour_is_the_box_set(tmp_path, monkeypatch):
    """The follower a box's run starts (woof-nam nest --products) begins before any history file exists: it chose
    awphys then (detect_set None -> "awphys") and failed every hour on the 600 km box (2026-11-13). It now reads
    the grid the door gave it (NAMGPU_CONFIGFILE); the 12 km run door, which sets none, keeps awphys."""
    from namgpu.io import products
    from namgpu.nest import domain
    g = domain.nest_grid(36.0, -98.0, 600.0, 12)
    cfgfile = tmp_path / "nam_configfile_box"
    domain.write_configfile(cfgfile, g, "box", tmp_path / "static.npz")
    (tmp_path / "box" / "run").mkdir(parents=True)
    monkeypatch.delenv("NAMGPU_CONFIGFILE", raising=False)
    assert products.detect_set(tmp_path / "box" / "run") is None
    monkeypatch.setenv("NAMGPU_CONFIGFILE", str(cfgfile))
    assert products.detect_set(tmp_path / "box" / "run") == "hiresf"
    from namgpu import config
    monkeypatch.setenv("NAMGPU_CONFIGFILE", str(config.DEFAULT_CONFIGFILE))
    assert products.detect_set(tmp_path / "box" / "run") is None


# ---- 2. windows past what a source's own forecast reaches
NOW = datetime(2026, 11, 13, 20, 0)


@pytest.mark.parametrize("name,hours,reach", [("ifs", 145, 144), ("ifs-0p25", 145, 144), ("aifs", 361, 360),
                                              ("aigfs", 385, 384)])
def test_a_window_past_the_sources_own_forecast_is_refused_by_name(name, hours, reach):
    from namgpu.init import window
    with pytest.raises(ValueError) as e:
        window.resolve(name, hours=hours, at=datetime(2026, 11, 12, 6), now=NOW)
    msg = str(e.value)
    assert f"{reach} h" in msg and name in msg, msg
    assert ("gfs (to 384 h ahead)" in msg) == (reach < 378), msg     # GFS reaches past IFS and AIFS, not past AIGFS
    assert "\n" not in msg


def test_the_reach_counts_the_hours_before_the_wanted_time():
    """An IFS 0.25 window wanted at 05Z starts from the 00Z analysis and runs 5 h longer: 140 h fits 144, 141 h not."""
    from namgpu.init import window
    p = window.resolve("ifs-0p25", hours=139, at=datetime(2026, 11, 12, 5), now=NOW)
    assert p["start"] == datetime(2026, 11, 12, 0) and p["model_hours"] == 144
    with pytest.raises(ValueError, match="144 h"):
        window.resolve("ifs-0p25", hours=140, at=datetime(2026, 11, 12, 5), now=NOW)


def test_ifs_0p25_fetch_reads_what_ecmwf_posts(tmp_path):
    """ECMWF posts IFS 0.25 every 3 h to 144 h at every cycle (its bucket, read 2026-11-13: 06Z and 2025 scda 06Z
    hour 144 present, 00Z hour 147 absent); the fetch read to 90 h, so a 100 h run from IFS 0.25 stopped in its fetch."""
    from namgpu.init import fetch_case
    jobs = fetch_case.plan_source("ifs-0p25", datetime(2026, 11, 12, 6), 100, tmp_path / "raw")
    leads = sorted(args[2] for fn, *args in jobs if fn.__name__ == "_jsonl_lead")
    assert leads[-1] >= 102 and leads == list(range(0, leads[-1] + 1, 3))


def test_sources_json_says_the_reach_the_fetch_reads():
    from namgpu import sources_cmd
    from namgpu.init import sources as S
    rows = {r["name"]: r for r in sources_cmd.table()}
    for name in ("ifs", "ifs-0p25", "aifs", "aigfs"):
        last = max(s for s, _ in S.SOURCES[name]["fetch"]["steps"])
        assert rows[name]["reach_h"] == last, name
    assert rows["ifs-0p25"]["reach_h"] == 144 and rows["aifs"]["reach_h"] == 360 and rows["aigfs"]["reach_h"] == 384
    # every posted table a row lists stays within what its fetch reads with its own edges, or says it is data only
    for name in ("ifs", "ifs-0p25"):
        assert rows[name]["reach_h"] <= max(s for s, _ in rows[name]["posted_by_cycle"]["0"])
    # sources whose edges are GFS's, GEFS's or ERA5's later analyses have no reach of their own
    for name in ("gfs", "gefs", "native", "era5", "rap", "hrrr", "rrfs"):
        assert rows[name]["reach_h"] is None, name
    json.dumps(list(rows.values()))


# ---- 3. a suite from an hour GFS does not post
def test_suite_from_an_unposted_hour_counts_outputs_from_the_hour_asked(tmp_path, monkeypatch):
    """`woof-nam suite --cycle 2026111212 --lead 200` starts the parent at hour 198 (said). Its fetch must plan the
    window (hour 200 asked, so the case's plan carries the 2 h offset the run counts its outputs from), the parent's
    GRIB2 follower must take model hours 2 to 2 + H, and the nests start at the hour asked (20Z), not at 18Z."""
    from namgpu.suite import cli as scli
    calls = []
    monkeypatch.setattr(scli, "visible_cards", lambda: ["0"])

    class FakeSuite:
        def __init__(self, a, t, plan, cycle):
            calls.append((a, plan, cycle))

        def run(self):
            return 0

    monkeypatch.setattr(scli, "Suite", FakeSuite)
    rc = scli.main(["--cycle", "2026111212", "--lead", "200", "--domains", "12km,firewxnest", "--hours-cap", "3",
                    "--out", str(tmp_path / "s"), "--cases-root", str(tmp_path / "cases")])
    assert rc == 0
    a, plan, cycle = calls[0]
    assert cycle.strftime("%Y%m%d%H") == "2026112018"                     # the parent's start (hour 198)
    assert plan["lead_h"] == 198 and plan["wanted_lead_h"] == 200 and plan["offset_h"] == 2
    assert plan["nest_start"] == "2026-11-20T20Z"


def test_suite_parent_fetch_products_and_nests_take_the_offset(tmp_path, monkeypatch):
    """The suite object of that plan: the parent's fetch asks hour 200 (the case then carries the 2 h offset), its
    GRIB2 follower takes model hours 2 to 5 for a 3 h parent, its whole hours are counted from the hour asked (hour 0,
    the posted start, is not an output), and each nest door starts at 20Z."""
    import types
    from namgpu.suite import cli as scli
    from namgpu.suite import table as T
    seen = []

    class FakeProc:
        def __init__(self, name, argv, log, env, cwd=None):
            seen.append((name, list(argv)))
            self.p = types.SimpleNamespace(returncode=0, poll=lambda: 0)
            self.wall_s, self.log = 1.0, log

        def wait(self):
            return 0

    monkeypatch.setattr(scli, "Proc", FakeProc)
    monkeypatch.setattr(scli, "visible_cards", lambda: ["0"])
    made, real = [], scli.Suite
    monkeypatch.setattr(scli, "Suite", lambda a, t, plan, cycle: made.append((a, t, plan, cycle)) or
                        types.SimpleNamespace(run=lambda: 0))
    assert scli.main(["--cycle", "2026111212", "--lead", "200", "--domains", "12km,firewxnest", "--hours-cap", "3",
                      "--out", str(tmp_path / "s"), "--cases-root", str(tmp_path / "cases")]) == 0
    a, t, plan, cycle = made[0]
    s = real(a, t, plan, cycle)
    assert s.offset_h() == 2 and f"{s.nest_start():%Y%m%d%H}" == "2026112020"
    (s.out / "12km").mkdir(parents=True)
    s.run_parent()
    assert s.dom("12km")["state"] == "complete", s.dom("12km")
    fetch = next(argv for name, argv in seen if name == "fetch")
    assert fetch[fetch.index("--cycle") + 1] == "2026111212" and fetch[fetch.index("--lead") + 1] == "200"
    run = next(argv for name, argv in seen if name == "12km")
    assert run[run.index("--lead") + 1] == "198"
    prod = next(argv for name, argv in seen if name.endswith("-products"))
    ph = plan["domains"]["12km"]["hours"]
    assert prod[prod.index("--hours") + 1] == f"2-{2 + int(ph)}"
    # the parent's history: 19Z a carrier, 20Z on (no 18Z file); hours are whole from 20Z
    s.parent_run = tmp_path / "prun"
    s.parent_run.mkdir()
    for hh in ("2026-11-20_19", "2026-11-20_20", "2026-11-20_21", "2026-11-20_22"):
        (s.parent_run / f"wrfout_d01_{hh}:00:00").write_bytes(b"x")
    s.parent_ended, s.parent_failed = None, False
    assert s.parent_hours_written() == (3, False)          # 22Z (hour 4) is still being written
    argv = T.door_argv("firewxnest", plan["domains"]["firewxnest"], s.nest_start(), s.parent_run, tmp_path / "n")
    assert argv[argv.index("--start") + 1] == "2026112020"


# ---- 4. the nest door's restart sets and resume
def _door(monkeypatch, tmp_path, seen, fetched):
    """The nest door with its card, static, fetch and child processes stood in (no card, no network)."""
    import types
    from namgpu.nest import cli as ncli
    from namgpu.nest import domain

    class S:
        samples = []

        def start(self):
            return self

        def mark(self, *a, **k):
            pass

        def stop(self):
            return {}

        def __getattr__(self, k):
            return lambda *a, **kw: {}

    monkeypatch.setattr(ncli, "Sampler", S)
    pick = {"rrtm_rows": 8, "total_gib": 3.0, "memory_path": "full", "pool_limit_gib": None,
            "pool_limit_bytes": None}
    monkeypatch.setattr(domain, "preflight_cards", lambda g, cards, rows=None: {
        "ok": True, "pick": pick, "free_gib": 30.0, "free_by_card": {"0": 30.0}, "tried": [pick], "card": "test"})
    monkeypatch.setattr(domain, "_visible_index", lambda: "0")
    monkeypatch.setattr(ncli, "_static_meta", lambda p: {})
    monkeypatch.setattr(ncli, "_write", lambda *a, **k: None)
    monkeypatch.setattr(ncli, "STATIC_CACHE", tmp_path / "static")

    def run_logged(cmd, env, log, sampler, marks=()):
        seen.append(list(cmd))
        return 0, {}

    monkeypatch.setattr(ncli, "run_logged", run_logged)
    monkeypatch.setattr(ncli, "resolve_fetch", lambda s: ((lambda *a, **k: fetched.append(a) or {}), "lic"))
    return ncli


BOX = ["nest", "--center", "36,-98", "--size-km", "600", "--res-km", "12", "--source", "gfs", "--start",
       "2026111218", "--hours", "6", "--no-maps", "--no-sheets", "--no-warm"]


def test_nest_door_writes_restart_sets_and_resumes_without_fetch_or_start(tmp_path, monkeypatch):
    from namgpu import restart
    seen, fetched = [], []
    ncli = _door(monkeypatch, tmp_path, seen, fetched)
    out = tmp_path / "box"
    assert ncli.main([*BOX, "--out", str(out), "--cases-root", str(tmp_path / "cases"), "--restart-every", "1h"]) == 0
    run = next(c for c in seen if "namgpu.run_forecast" in c)
    i = run.index("--restart-every")
    assert run[i:i + 6] == ["--restart-every", "1h", "--restart-keep", "1", "--restart-dir",
                            str((out / "run" / "restart").resolve())]
    assert len(fetched) == 1
    # a set as woof-nam run writes it (restart.py layout), then the same door words with --resume
    root = out / "run" / "restart"
    st = root / f"{restart.PREFIX}2026-11-12_21:00:00"
    st.mkdir(parents=True)
    (st / "meta.json").write_text(json.dumps({"format": restart.FORMAT, "ntimestep": 432}))
    (st / "COMPLETE").write_text("")
    (root / restart.LATEST).write_text(st.name)
    seen.clear()
    assert ncli.main([*BOX, "--out", str(out), "--cases-root", str(tmp_path / "cases"), "--resume", str(root)]) == 0
    run = next(c for c in seen if "namgpu.run_forecast" in c)
    assert run[run.index("--resume") + 1] == str(st.resolve()) and "--case" not in run and "--hours" not in run
    assert len(fetched) == 1                         # no second fetch


@pytest.mark.parametrize("extra", [["--cards", "2"], ["--two-way"]])
def test_nest_door_refuses_restarts_where_no_set_is_written(tmp_path, monkeypatch, capsys, extra):
    seen, fetched = [], []
    ncli = _door(monkeypatch, tmp_path, seen, fetched)
    rc = ncli.main([*BOX, "--out", str(tmp_path / "b"), "--restart-every", "1h", *extra])
    assert rc == 2 and "restart sets are written for a fixed nest on one card" in capsys.readouterr().err
    assert not seen and not fetched


def test_nest_door_resume_with_no_set_is_refused_by_name(tmp_path, monkeypatch, capsys):
    seen, fetched = [], []
    ncli = _door(monkeypatch, tmp_path, seen, fetched)
    rc = ncli.main([*BOX, "--out", str(tmp_path / "b"), "--resume", str(tmp_path / "b" / "run" / "restart")])
    assert rc == 2 and "no complete restart set" in capsys.readouterr().err
    assert not seen and not fetched
