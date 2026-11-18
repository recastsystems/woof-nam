"""Windows: any source, any start, any length, any output times (namgpu/init/window.py, namgpu/outputs.py).
CPU only, no network. Each test failed before the change that it names.

  - the plan never refuses over an unposted hour: GFS hour 200 starts at 198 and runs 2 h longer; hour 100 for 24 h
    starts at 99 (the window crosses 120, where GFS goes 3-hourly); GEFS past 240 h takes its edges every 6 h; a
    window past the last posted hour is cut there and the line says by how much;
  - an analysis-only source starts at its newest analysis at or before the time asked (the NAM's own at 12Z for
    14Z, ERA5 at any hour), and HRRR at 15Z takes the GFS 12Z run's hour 3 around it;
  - it stops only when nothing can run (a date outside the archive, a time no run reaches), naming who has it;
  - the output schedule: the default is every 60 minutes from step 0 (the files 0.2.1 wrote), a spec is counted from
    the time asked, the carriers a product reads are added and named, GRIB2 stays on whole hours;
  - the run reads the case's plan: the model runs the offset longer and a cut window stops at the cut.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from namgpu import outputs
from namgpu.init import init_fetch_case, sources, window

RUN = datetime(2026, 11, 12, 12)
NOW = datetime(2026, 11, 12, 20, 30)


@pytest.fixture(autouse=True)
def clean_env():
    keys = (sources.LEAD_ENV, "NAMGPU_BOCO_AGE_H", sources.INTERVAL_ENV, "NAMGPU_SOURCE_MEMBER")
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


# ------------------------------------------------------------------------------------------ the plan

def test_gfs_hour_200_starts_at_198_and_runs_2_h_longer():
    p = window.resolve("gfs", hours=6, cycle=RUN, lead=200, now=NOW)
    assert (p["lead_h"], p["offset_h"], p["model_hours"], p["edges_h"], p["cut_h"]) == (198, 2.0, 8.0, 3, 0.0)
    assert p["start"] == datetime(2026, 11, 20, 18) and p["wanted"] == datetime(2026, 11, 20, 20)
    assert "hour 200 is not posted: 2 h earlier" in p["line"] and "first output 2026-11-20 20Z" in p["line"]
    # a posted hour is taken as it is
    q = window.resolve("gfs", hours=6, cycle=RUN, lead=201, now=NOW)
    assert (q["lead_h"], q["offset_h"], q["model_hours"]) == (201, 0.0, 6.0) and "earlier" not in q["line"]


def test_gfs_hour_100_for_24_h_starts_at_99():
    p = window.resolve("gfs", hours=24, cycle=RUN, lead=100, now=NOW)
    assert (p["lead_h"], p["offset_h"], p["model_hours"]) == (99, 1.0, 25.0)


def test_gfs_window_past_the_last_hour_is_cut_and_said():
    p = window.resolve("gfs", hours=12, cycle=RUN, lead=380, now=NOW)
    assert (p["lead_h"], p["model_hours"], p["cut_h"]) == (378, 6.0, 8.0)
    assert "cut 8 h short" in p["line"]
    # hour 381 is posted but carries 3 h of boundaries, too few for any run: the window starts at 378
    q = window.resolve("gfs", hours=6, cycle=RUN, lead=381, now=NOW)
    assert (q["lead_h"], q["offset_h"], q["model_hours"], q["cut_h"]) == (378, 3.0, 6.0, 3.0)
    with pytest.raises(ValueError, match="no run starts near hour 384"):
        window.resolve("gfs", hours=6, cycle=RUN, lead=384, now=NOW)


def test_gfs_at_a_time_picks_the_newest_posted_run():
    have = lambda c, l: c <= RUN                      # noqa: E731 - the 18Z run has posted nothing yet
    p = window.resolve("gfs", hours=6, at=datetime(2026, 11, 20, 20), now=NOW, posted=have)
    assert (p["cycle"], p["lead_h"], p["offset_h"]) == (RUN, 198, 2.0)
    # a past cycle hour with its analysis posted: the analysis
    a = window.resolve("gfs", hours=6, at=datetime(2026, 1, 23, 12), now=NOW, posted=have)
    assert (a["cycle"], a["lead_h"], a["offset_h"]) == (datetime(2026, 1, 23, 12), 0, 0.0)
    # minutes: the hour before, and the first output at the minute asked
    m = window.resolve("gfs", hours=3, at=datetime(2026, 1, 23, 15, 30), now=NOW, posted=have)
    assert (m["lead_h"], m["offset_h"]) == (3, 0.5) and "first output 2026-01-23 15:30Z" in m["line"]
    with pytest.raises(ValueError, match="more than 384 h ahead"):
        window.resolve("gfs", hours=6, at=NOW + timedelta(hours=420), now=NOW, posted=have)


def test_gefs_past_240_takes_6_hourly_edges():
    p = window.resolve("gefs", hours=48, cycle=datetime(2026, 11, 12, 6), lead=246, now=NOW)
    assert (p["lead_h"], p["offset_h"], p["edges_h"]) == (246, 0.0, 6)
    q = window.resolve("gefs", hours=12, cycle=datetime(2026, 11, 12, 6), lead=245, now=NOW)
    assert (q["lead_h"], q["offset_h"], q["edges_h"]) == (240, 5.0, 6) and "edges every 6 h" in q["line"]
    r = window.resolve("gefs", hours=24, cycle=datetime(2026, 11, 12, 6), lead=100, now=NOW)
    assert (r["lead_h"], r["offset_h"], r["edges_h"]) == (99, 1.0, 3)
    rec = window.record(Path("/tmp") / "gefs-win-test", p)
    assert window.env_for(rec) == {"NAMGPU_BOCO_AGE_H": "246", sources.INTERVAL_ENV: "6"}
    os.environ[sources.INTERVAL_ENV] = "6"
    assert sources.boundary_interval_h(sources.SOURCES["gefs"]) == 6
    assert sources.boundary_interval_h(sources.SOURCES["gfs"]) == 6   # a multiple of GFS's 3 h too
    os.environ[sources.INTERVAL_ENV] = "4"
    with pytest.raises(ValueError, match="multiple of the source's own 3 h"):
        sources.boundary_interval_h(sources.SOURCES["gefs"])


def test_analysis_sources_start_at_the_analysis_before_and_run_longer():
    n = window.resolve("native", hours=3, at=datetime(2025, 3, 14, 14), now=NOW)
    assert (n["start"], n["offset_h"], n["model_hours"]) == (datetime(2025, 3, 14, 12), 2.0, 5.0)
    assert "2025-03-14 14Z has no analysis: 2 h earlier" in n["line"] and "first output 2025-03-14 14Z" in n["line"]
    e = window.resolve("era5", hours=3, at=datetime(2014, 2, 11, 15), now=NOW)
    assert (e["start"], e["offset_h"]) == (datetime(2014, 2, 11, 15), 0.0)
    i = window.resolve("ifs-0p25", hours=6, cycle=datetime(2026, 11, 5, 0), lead=150, now=NOW)
    assert i["start"] == datetime(2026, 11, 11, 6) and "forecast hours are not a start" in i["line"]


def test_hrrr_at_15z_takes_gfs_12z_hour_3_around_it():
    p = window.resolve("hrrr", hours=6, at=datetime(2026, 11, 12, 15), now=NOW, posted=lambda c, l: c <= RUN)
    assert (p["start"], p["gfs_cycle"], p["gfs_lead_h"], p["offset_h"]) == (datetime(2026, 11, 12, 15), RUN, 3, 0.0)
    assert "from the GFS 2026-11-12 12Z run hour 3" in p["line"]
    # at a GFS cycle hour the GFS half is that cycle's analysis, as before (no GFS hour)
    q = window.resolve("hrrr", hours=6, at=datetime(2026, 11, 12, 12), now=NOW, posted=lambda c, l: True)
    assert not q.get("gfs_lead_h")


def test_refused_only_when_nothing_can_run():
    with pytest.raises(ValueError, match="native ends 2026-11-03 12Z.*gfs, ifs, era5"):
        window.resolve("native", hours=6, at=datetime(2026, 11, 10, 12), now=NOW)
    with pytest.raises(ValueError, match="analyses only.*ahead of now.*gfs"):
        window.resolve("era5", hours=6, at=NOW + timedelta(hours=30), now=NOW)
    with pytest.raises(ValueError, match="behind the clock"):
        window.resolve("era5", hours=6, at=NOW - timedelta(hours=30), now=NOW)
    with pytest.raises(ValueError, match="rrfs begins 2026-08-12"):
        window.resolve("rrfs", hours=6, at=datetime(2026, 1, 1, 12), now=NOW)


def test_fetch_door_chooses_and_records(tmp_path):
    a = SimpleNamespace(cycle="2026111212", lead=200.0, at=None, hours=6, source=None, spinup_hours=0, boco_age=None)
    c = init_fetch_case.choose(a, now=NOW, posted=lambda c, l: True)
    assert (c["start"], c["lead_h"]) == (datetime(2026, 11, 20, 18), 198)
    rec = window.record(tmp_path / "2026-11-20_18z", c["plan"])
    assert (rec["offset_h"], rec["model_hours"], rec["gfs_lead_h"], rec["wanted"]) == (2.0, 8.0, 198, "2026-11-20T20:00")
    # the run door: the GFS hour from the plan; the run is 2 h longer and its outputs count from 20Z
    case = tmp_path / "2026-11-20_18z"
    assert sources.door_lead(case, "gfs", None, datetime(2026, 11, 20, 18)) == 198
    w = window.for_run(case, 6)
    assert (w["offset_h"], w["model_hours"], w["cut_h"]) == (2.0, 8.0, 0.0)
    # a plain --cycle fetch is planned as before
    assert init_fetch_case.choose(SimpleNamespace(cycle="2026111212", lead=None, at=None, hours=6, source=None,
                                                  spinup_hours=0, boco_age=None))["plan"] is None


def test_a_cut_window_stops_the_run_at_the_cut(tmp_path):
    p = window.resolve("gfs", hours=12, cycle=RUN, lead=380, now=NOW)
    window.record(tmp_path / "c", p)
    w = window.for_run(tmp_path / "c", 12)
    assert (w["model_hours"], w["cut_h"]) == (6.0, 8.0)


def test_hrrr_plan_sets_the_gfs_hour_for_the_run(tmp_path):
    p = window.resolve("hrrr", hours=6, at=datetime(2026, 11, 12, 15), now=NOW, posted=lambda c, l: c <= RUN)
    case = tmp_path / "2026-11-12_15z"
    window.record(case, p)
    assert sources.door_lead(case, "hrrr", None, datetime(2026, 11, 12, 15)) == 3
    assert os.environ[sources.LEAD_ENV] == "3"
    with pytest.raises(ValueError, match="use --source gfs"):
        sources.door_lead(case, "hrrr", 3, datetime(2026, 11, 12, 15))      # --lead itself stays a GFS option


def test_fetch_plan_takes_the_gfs_hour_for_hrrr():
    from namgpu.init import fetch_case
    jobs, skipped = fetch_case.plan(datetime(2026, 11, 12, 15), 6, Path("raw"), source="hrrr", lead_h=3)
    names = sorted(Path(j[2]).name for j in jobs if "gfs" in str(j[2]) or "hrrr" in str(j[2]))
    assert "gfs.t12z.f003.grib2" in names and "hrrr.t15z.wrfprsf00.grib2" in names
    assert not [n for n in names if "f000" in n and "gfs" in n]
    assert "gfs-f000" in dict(skipped)


def test_domain_rows_and_nest_door_take_every_source():
    from namgpu import domains
    from namgpu.nest import cli
    assert set(domains.get("conus12")["sources"]) >= {"native", "gfs", "gefs", "ifs", "ifs-0p25", "era5", "rap",
                                                           "hrrr", "rrfs", "aifs", "aigfs"}
    assert set(cli.FETCH) >= {"gfs", "gefs", "ifs", "ifs-0p25", "era5", "rap", "hrrr", "rrfs", "aifs", "aigfs",
                              "parent"}
    assert "native" not in cli.FETCH


def test_sources_json_lists_every_row():
    from namgpu import sources_cmd
    rows = {r["name"]: r for r in sources_cmd.table()}
    assert set(rows) == set(sources.SOURCES)
    assert rows["gfs"]["starts"] == "forecast-hour" and rows["gfs"]["posted"] == [[120, 1], [384, 3]]
    assert rows["gefs"]["posted"] == [[240, 3], [384, 6]] and rows["hrrr"]["analysis_every_h"] == 1
    assert rows["native"]["valid_before"] == "2026-11-03T12" and rows["parent"]["starts"] == "parent"


# ------------------------------------------------------------------------------------------ the output schedule

def test_default_schedule_is_every_60_minutes_from_step_0():
    s = outputs.Schedule(25.0, 6.0)
    assert sorted(s.entries) == [k * 144 for k in range(7)]
    assert all(s.due(k * 144) for k in range(7)) and not s.due(72)
    assert not s.notes and s.counts()["carriers"] == 0
    assert s.counts() == {"history": 7, "maps": 7, "grib": 7, "carriers": 0, "files": 7}


def test_spec_parsing():
    assert outputs.hours_of("1h", 3) == [0, 1, 2, 3]
    assert outputs.hours_of("0,1.5,3,6", 6) == [0, 1.5, 3, 6]
    assert outputs.hours_of("0-1/15m,1-3/1h", 3) == [0, 0.25, 0.5, 0.75, 1, 2, 3]
    for bad in ("", "1x", "3-1/1h", "0h"):
        with pytest.raises(ValueError):
            outputs.hours_of(bad, 3)


def test_run_a_schedule_counts_from_the_time_asked_and_names_its_carriers():
    """Run A: GFS hour 198 for maps from hour 200: model 8 h, offset 2, --write 0,1.5,3,6, GRIB2 at 0,3,6."""
    s = outputs.Schedule(25.0, 8.0, offset_h=2.0, write="0,1.5,3,6", grib="0,3,6")
    hist = sorted(e["model_h"] for e in s.entries.values() if e["carrier"] is None)
    assert hist == [2.0, 3.5, 5.0, 8.0]
    assert sorted(e["model_h"] for e in s.entries.values() if e["grib"]) == [2.0, 5.0, 8.0]
    rain = sorted(e["model_h"] for e in s.entries.values() if e["carrier"] == "rain")
    assert 3.0 in rain and 6.0 in rain                # the 3 h rain buckets of the 5 h and 8 h GRIB2 files
    assert all(s.lean(e["step"]) for e in s.entries.values() if e["carrier"] == "rain")
    doc = s.doc(datetime(2026, 11, 20, 18))
    assert doc["first_output"] == "2026-11-20T20:00:00"
    first = next(t for t in doc["times"] if t["carrier"] is None)
    assert first["file"] == "wrfout_d01_2026-11-20_20:00:00" and first["h"] == 0.0


def test_grib2_stays_on_whole_hours_and_says_so():
    s = outputs.Schedule(25.0, 3.0, write="30m", grib="30m")
    assert sorted(e["model_h"] for e in s.entries.values() if e["grib"]) == [0, 1, 2, 3]
    assert any("GRIB2 is written on whole forecast hours" in n for n in s.notes)
    assert outputs.tag_of(datetime(2026, 1, 1, 7, 30), datetime(2026, 1, 1, 0)) == "f007m30"
    assert outputs.tag_of(datetime(2026, 1, 1, 7), datetime(2026, 1, 1, 0)) == "f007"


def test_state_carriers_for_a_nest_and_a_time_past_the_end():
    s = outputs.Schedule(25.0, 6.0, write="3h", carry="1h", products=False)
    st = sorted(e["model_h"] for e in s.entries.values() if e["carrier"] == "state")
    assert st == [1.0, 2.0, 4.0, 5.0] and not any(s.lean(n) for n in s.entries)
    t = outputs.Schedule(25.0, 6.0, write="0,9")
    assert any("past the run's end" in n for n in t.notes)


def test_the_list_on_disk_drives_maps_and_grib2(tmp_path):
    s = outputs.Schedule(25.0, 8.0, offset_h=2.0, write="0,1.5,3,6", grib="0,3,6")
    s.write(tmp_path, datetime(2026, 11, 20, 18))
    doc = json.loads((tmp_path / "OUTPUTS.json").read_text())
    assert doc["counts"]["history"] == 4
    assert outputs.files(tmp_path, "grib") == {"wrfout_d01_2026-11-20_20:00:00", "wrfout_d01_2026-11-20_23:00:00",
                                               "wrfout_d01_2026-11-21_02:00:00"}
    assert "wrfout_d01_2026-11-20_21:00:00" in outputs.files(tmp_path, "all")      # a rain carrier
    assert "wrfout_d01_2026-11-20_21:00:00" not in outputs.files(tmp_path, "history")
    assert outputs.files(tmp_path / "none", "grib") is None


def test_driver_asks_the_schedule_not_minutes_history():
    from namgpu.driver import Driver
    s = outputs.Schedule(25.0, 3.0, write="0,1.5")
    ctx = SimpleNamespace(cfg=SimpleNamespace(raw={"minutes_history": 60}, dt=25.0), ntimestep=216, outputs=s)
    drv = Driver.__new__(Driver)
    drv.ctx = ctx
    assert drv._due("history") is True                # 1.5 h = step 216
    ctx.ntimestep = 144
    assert drv._due("history") is False               # hour 1 is not asked
    ctx.outputs = None
    assert drv._due("history") is True                # no schedule: every 60 minutes, as before


def test_map_title_of_a_frame_off_the_hour_and_of_a_regional_start(monkeypatch):
    """Run A titled its 21:30Z frame 'valid 2026-11-20 21Z (+4h)' (the hour rounded): the minutes are said."""
    from namgpu.io import maps
    monkeypatch.delenv("NAMGPU_START_SOURCE", raising=False)
    monkeypatch.setenv(sources.LEAD_ENV, "198")
    t = maps.panel_title("MSLP", datetime(2026, 11, 20, 18), datetime(2026, 11, 20, 21, 30))
    assert t.endswith("from GFS hour 198  valid 2026-11-20 21:30Z (+3h30m)"), t
    assert maps.panel_title("MSLP", datetime(2026, 11, 20, 18), datetime(2026, 11, 20, 20)).endswith("20Z (+2h)")
    monkeypatch.setenv("NAMGPU_START_SOURCE", "hrrr")
    monkeypatch.setenv(sources.LEAD_ENV, "3")
    assert "from HRRR, GFS hour 3 around it" in maps.panel_title("T2", datetime(2026, 11, 12, 15), datetime(2026, 11, 12, 16))
    monkeypatch.setenv("NAMGPU_START_SOURCE", "gefs")
    monkeypatch.setenv("NAMGPU_BOCO_AGE_H", "246")
    assert "from GEFS hour 246" in maps.panel_title("T2", datetime(2026, 11, 22, 12), datetime(2026, 11, 22, 15))


def test_nest_door_edges_of_a_regional_start_follow_the_gfs_hour(monkeypatch):
    """E5 (hrrr --at 15Z on the nest door) stopped in the fetch: the door's boundary age 0 contradicted GFS hour 3."""
    from namgpu.nest import cli
    p = window.resolve("hrrr", hours=3, at=datetime(2026, 11, 12, 15), now=NOW, posted=lambda c, l: c <= RUN)
    monkeypatch.setenv(sources.LEAD_ENV, "3")
    age, why = cli.boco_age_for("hrrr", datetime(2026, 11, 12, 15), None, p)
    assert age == 3 and "GFS forecast hour 3 of the 2026-11-12 12Z run" in why
    from namgpu.init import fetch_case
    jobs, _ = fetch_case.plan(datetime(2026, 11, 12, 15), 6, Path("raw"), source="hrrr", age_h=age, lead_h=3)
    assert any(Path(j[2]).name == "gfs.t12z.f009.grib2" for j in jobs)


def test_grib2_times_are_listed_for_a_follower_too(monkeypatch):
    """Suite D: the parent ran without --products (the suite's own follower writes its GRIB2), so its list named no
    GRIB2 time, the follower wrote nothing and the suite called the domain complete. The times are listed always."""
    from types import SimpleNamespace
    from namgpu import run_forecast
    a = SimpleNamespace(products=None, hours=6.0, output_offset_h=0.0, write="3h", maps_at=None, grib_at=None, carry="1h")
    s = run_forecast.output_schedule(a)
    assert sorted(e["model_h"] for e in s.entries.values() if e["grib"]) == [0.0, 3.0, 6.0]
    plain = run_forecast.output_schedule(SimpleNamespace(**{**vars(a), "write": None, "carry": None}))
    assert plain.counts() == {"history": 7, "maps": 7, "grib": 7, "carriers": 0, "files": 7}
