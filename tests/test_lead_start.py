"""A start from a GFS forecast hour (--lead H) and a start at a chosen time (--at TIME). CPU only, no network.

What these hold (each failed before the change):
  - the file names: a 2026-11-20 21Z start from hour 201 reads the 12Z run of 2026-11-12, its hour 201 with the
    start records and its hours 201, 204, ... as boundaries, and no analysis;
  - the rule for which hours exist and how long a run they carry (GFS posts every hour to 120, every 3 h to 384);
  - the rule for a chosen time (the analysis when there is one, else the newest posted run that reaches it);
  - an analysis start is planned exactly as before;
  - the start record set is the same in an analysis file and in a forecast file (the .idx of f000 and f201 of the
    2026-11-12 12Z and 2021-04-01 00Z runs, tests/fixtures/gfs_idx/).
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from namgpu.init import fetch_case, init_fetch_case, sources

GFS = sources.SOURCES["gfs"]
FIX = Path(__file__).parent / "fixtures" / "gfs_idx"
START = datetime(2026, 11, 20, 21)          # 2026-11-12 12Z + 201 h
RUN = datetime(2026, 11, 12, 12)


@pytest.fixture(autouse=True)
def clean_env():
    """The doors set NAMGPU_START_LEAD_H in os.environ themselves; nothing of it may reach another test."""
    keys = (sources.LEAD_ENV, "NAMGPU_BOCO_AGE_H", "NAMGPU_SOURCE_MEMBER", "NAMGPU_DATA_CREDIT")
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def gfs_jobs(start, hours, **kw):
    jobs, skipped = fetch_case.plan(start, hours, Path("raw"), source="gfs", **kw)
    return [j for j in jobs if "gfs" in str(j[2])], dict(skipped)


# ------------------------------------------------------------------------------------------ file names

def test_file_keys_and_start_file_for_hour_201():
    assert sources.file_keys(GFS, START)["lead"] == 0                # no lead set: the analysis name, as before
    os.environ[sources.LEAD_ENV] = "201"
    keys = sources.file_keys(GFS, START)
    assert (keys["cyc"], keys["bcyc"], keys["lead"]) == (21, 12, 201)
    ctx = SimpleNamespace(case_dir=Path("cases/2026-11-20_21z"), start=START, cfg=None)
    assert sources.source_path(ctx, GFS).as_posix() == "cases/2026-11-20_21z/raw/gfs/gfs.t12z.f201.start.grib2"
    assert sources.boco_age_h(GFS, start=START) == 201               # the boundaries are that run's own hours
    # other rows are untouched by the setting: only a row with "leads" can start from a forecast hour
    assert sources.start_lead_h(sources.SOURCES["ifs"]) == 0 and sources.start_lead_h(sources.SOURCES["rap"]) == 0


def test_boundary_files_come_from_the_same_run():
    from namgpu.init import gfs_bocos
    os.environ[sources.LEAD_ENV] = "201"
    files = gfs_bocos.gfs_files(Path("c"), START, 24)
    assert [fh for fh, _ in files] == list(range(0, 25, 3))
    assert [p.name for _, p in files][:3] == ["gfs.t12z.f201.grib2", "gfs.t12z.f204.grib2", "gfs.t12z.f207.grib2"]
    assert files[-1][1].name == "gfs.t12z.f225.grib2"


def test_plan_hour_201_fetches_the_start_set_once_and_no_analysis():
    jobs, skipped = gfs_jobs(START, 24, lead_h=201)
    names = sorted(Path(j[2]).name for j in jobs)
    assert names == [f"gfs.t12z.f{h:03d}.grib2" for h in range(201, 226, 3)]
    assert not [j for j in jobs if "f000" in str(j[1]) or "f000" in str(j[2])]
    start = [j for j in jobs if j[0] is fetch_case._follow_lead]
    assert len(start) == 1
    fn, url, dest, sdest, pattern, required = start[0]
    assert url.endswith("gfs.20261112/12/atmos/gfs.t12z.pgrb2.0p25.f201") and required
    assert Path(sdest).name == "gfs.t12z.f201.start.grib2" and Path(dest).name == "gfs.t12z.f201.grib2"
    assert pattern is fetch_case.GFS_INIT_VARS                       # soil, snow, ice, canopy water: the start set
    later = [j for j in jobs if j[0] is fetch_case._subset]
    assert all(j[3] is fetch_case.GFS_BOCO_VARS for j in later) and len(later) == 8
    assert "forecast hour 201 of the 2026-11-12 12Z GFS run" in skipped["gfs-f000"]
    # the setting a door puts in the environment plans the same thing
    os.environ[sources.LEAD_ENV] = "201"
    env_jobs, _ = gfs_jobs(START, 24)
    assert [str(j[1]) for j in env_jobs] == [str(j[1]) for j in jobs]


def test_analysis_start_is_planned_as_before():
    """Lead 0: the plan of 0.2.1 (the public feed's job), with or without the new argument."""
    cyc = datetime(2026, 11, 4, 12)
    want = ["gfs.t06z.f006.grib2", "gfs.t06z.f009.grib2", "gfs.t06z.f012.grib2", "gfs.t12z.pgrb2.0p25.f000"]
    for kw in ({}, {"lead_h": 0}, {"lead_h": None}):
        jobs, skipped = gfs_jobs(cyc, 6, **kw)
        assert sorted(Path(j[2]).name for j in jobs) == want
        assert "gfs-f000" not in skipped and not [j for j in jobs if j[0] is fetch_case._follow_lead]
    assert sources.boco_age_h(GFS) == 6 and sources.boco_age_h(GFS, override=0) == 0
    with pytest.raises(ValueError, match="multiple of 6"):
        sources.boco_age_h(GFS, override=201)                        # an age is still a cycle step without a lead


# ------------------------------------------------------------------------------------------ the rule

def test_hour_200_is_refused_naming_198_and_201():
    with pytest.raises(ValueError, match=r"hour 200 is not posted .*every hour to 120, then every 3 hours to 384.*"
                                         r"nearest are 198 and 201"):
        sources.check_lead(GFS, 200, 24, RUN)
    with pytest.raises(ValueError, match="198 and 201"):
        fetch_case.plan(START - timedelta(hours=1), 24, Path("raw"), source="gfs", lead_h=200)


def test_run_length_by_start_hour():
    assert sources.check_lead(GFS, 201, 24, RUN) == 183              # 384 - 201
    assert sources.check_lead(GFS, 198, 186, RUN) == 186
    assert sources.check_lead(GFS, 378, 6, RUN) == 6                 # the latest start
    assert sources.check_lead(GFS, 0, 384, RUN) == 384
    assert sources.check_lead(GFS, 100, 18, RUN) == 18               # off the 3 h grid: must end by hour 120
    with pytest.raises(ValueError, match=r"24 h run from GFS forecast hour 100 needs boundaries to hour 124, and hour "
                                         r"121 is not posted.*longest run from hour 100 is 18 h; hour 99 allows "
                                         r"285 h and hour 102 allows 282 h"):
        sources.check_lead(GFS, 100, 24, RUN)
    with pytest.raises(ValueError, match="longest run from hour 201 is 183 h"):
        sources.check_lead(GFS, 201, 184, RUN)
    with pytest.raises(ValueError, match=r"no run fits from GFS forecast hour 381.*hour 387 is not posted.*latest "
                                         r"start is hour 378"):
        sources.check_lead(GFS, 381, 6, RUN)
    with pytest.raises(ValueError, match=r"no run fits from GFS forecast hour 116.*hour 122 is not posted"):
        sources.check_lead(GFS, 116, 6, RUN)


def test_archived_eras():
    # NCAR's archive (2015-01-15 to 2020): every 3 hours to 240
    assert sources.check_lead(GFS, 201, 24, datetime(2018, 6, 1, 0)) == 39
    with pytest.raises(ValueError, match="every 3 hours to 240.*nearest are 99 and 102"):
        sources.check_lead(GFS, 100, 6, datetime(2018, 6, 1, 0))
    with pytest.raises(ValueError, match="begin 2015-01-15 00Z.*only analyses"):
        sources.check_lead(GFS, 6, 6, datetime(2012, 6, 1, 0))
    with pytest.raises(ValueError, match="only analyses"):
        fetch_case.plan(datetime(2012, 6, 1, 6), 6, Path("raw"), source="gfs", lead_h=6)


def test_a_lead_start_is_gfs_only_and_not_a_cycle_hour_mistake():
    with pytest.raises(ValueError, match="needs --source gfs"):
        fetch_case.plan(START, 24, Path("raw"), source="native", lead_h=201)
    with pytest.raises(ValueError, match="storm-following"):
        fetch_case.plan(START, 24, Path("raw"), source="gfs", lead_h=201, follow=True)
    os.environ[sources.LEAD_ENV] = "201"
    with pytest.raises(ValueError, match="not a 6-hourly cycle"):
        sources.boco_age_h(GFS, start=START + timedelta(hours=1))    # 22Z - 201 h is a 13Z run: there is none
    with pytest.raises(ValueError, match="drop the boundary age"):
        sources.boco_age_h(GFS, override=6, start=START)


# ------------------------------------------------------------------------------------------ a chosen time

def test_at_rule():
    now = datetime(2026, 11, 12, 19, 30)
    have = {(RUN, h) for h in sources.posted_leads(GFS["leads"])}            # the 12Z run is whole
    have |= {(datetime(2026, 1, 23, 12), h) for h in range(0, 49)}
    asked = []

    def posted(cyc, lead):
        asked.append((cyc, lead))
        return (cyc, lead) in have or cyc < datetime(2026, 11, 12, 0)

    # a past cycle hour: its analysis
    assert sources.start_for_time(GFS, datetime(2026, 1, 23, 12), 24, posted, now) == (datetime(2026, 1, 23, 12), 0)
    # a past hour between cycles: the run before it, a few hours in
    assert sources.start_for_time(GFS, datetime(2026, 1, 23, 15), 6, posted, now) == (datetime(2026, 1, 23, 12), 3)
    assert sources.start_for_time(GFS, datetime(2026, 1, 23, 13), 6, posted, now) == (datetime(2026, 1, 23, 12), 1)
    # a time ahead of now: the newest posted run that reaches it (18Z has posted nothing, and is never asked for
    # hours it cannot have; no run after `now` is asked at all)
    asked.clear()
    assert sources.start_for_time(GFS, START, 24, posted, now) == (RUN, 201)
    assert all(c <= now for c, _ in asked)
    # a cycle hour ahead of now has no analysis yet: the newest run's forecast of it
    assert sources.start_for_time(GFS, datetime(2026, 11, 14, 0), 24, posted, now) == (RUN, 36)
    # the newest run has posted the start hour but not yet the run's last boundary hour: not chosen
    part = lambda c, l: (c == datetime(2026, 11, 12, 18) and l <= 30) or posted(c, l)  # noqa: E731
    assert sources.start_for_time(GFS, datetime(2026, 11, 13, 21), 24, part, datetime(2026, 11, 12, 22)) == (RUN, 33)
    # past 120 h only the 3-hourly hours can be started; past 384 h nothing
    with pytest.raises(ValueError, match=r"2026-11-20 20Z: from the 2026-11-12 12Z run it would be forecast hour "
                                         r"200: GFS forecast hour 200 is not posted.*198 and 201"):
        sources.start_for_time(GFS, START - timedelta(hours=1), 24, posted, now)
    with pytest.raises(ValueError, match="more than 384 h ahead"):
        sources.start_for_time(GFS, now + timedelta(hours=400), 24, posted, now)


def fetch_args(**kw):
    base = dict(cycle=None, lead=None, at=None, hours=24, source=None, spinup_hours=0, boco_age=None)
    return SimpleNamespace(**(base | kw))


def test_fetch_says_what_it_chose_and_refuses_in_plain_words():
    start, lead, line = init_fetch_case.choose_start(fetch_args(cycle="2026111212", lead=201))
    assert (start, lead) == (START, 201)
    assert "start 2026-11-20 21Z from GFS forecast hour 201 of the 2026-11-12 12Z run" in line
    assert "hours 201 to 225" in line and "not an analysis" in line
    assert init_fetch_case.choose_start(fetch_args(cycle="2026111212"))[:2] == (RUN, 0)       # as before
    # an hour GFS does not post is not refused: the nearest posted hour before it, the run that much longer
    # (namgpu/init/window.py; tests/test_windows.py holds the rule)
    s200, l200, line200 = init_fetch_case.choose_start(fetch_args(cycle="2026111212", lead=200))
    assert (s200, l200) == (START - timedelta(hours=3), 198) and "hour 200 is not posted: 2 h earlier" in line200
    assert init_fetch_case.choose_start(fetch_args(cycle="2026111212", lead=100))[1] == 99
    for kw, why in ((dict(cycle="2026111212", lead=201, source="native"), "analyses only"),
                    (dict(cycle="2026111212", lead=201, spinup_hours=6), "spin-up reads an analysis"),
                    (dict(cycle="2026111215", lead=3), "a gfs run is at 00, 06, 12 or 18Z"),
                    (dict(cycle="2026111212", at="2026112021"), "--at chooses the source run"),
                    (dict(at="2026112"), "want a time"),
                    (dict(), "give --cycle")):
        with pytest.raises(SystemExit, match=why) as e:
            init_fetch_case.choose_start(fetch_args(**kw))
        assert str(e.value).startswith("woof-nam fetch: refused: ")
    at = init_fetch_case.choose_start(fetch_args(at="2026012315", hours=6), now=datetime(2026, 11, 12, 19),
                                      posted=lambda c, l: True)
    assert at[:2] == (datetime(2026, 1, 23, 15), 3) and "hour 3 of the 2026-01-23 12Z run" in at[2]
    an = init_fetch_case.choose_start(fetch_args(at="2026012312"), now=datetime(2026, 11, 12, 19),
                                      posted=lambda c, l: True)
    assert an[:2] == (datetime(2026, 1, 23, 12), 0) and "from the gfs analysis of 2026-01-23 12Z" in an[2]


def test_case_folder_is_named_for_the_start(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="name the case folder for the start time, 2026-11-20_21z"):
        init_fetch_case.case_folder("cases/2026-11-12_12z", START, 201, "--lead")
    assert init_fetch_case.case_folder("cases/2026-11-20_21z", START, 201, "--lead") == Path("cases/2026-11-20_21z")
    monkeypatch.setenv("NAMGPU_ROOT", str(tmp_path))
    assert init_fetch_case.case_folder(None, START, 201, "--at").name == "2026-11-20_21z"


# ------------------------------------------------------------------------------------------ fetch, run, resume agree

def test_start_record_and_the_run_door(tmp_path, monkeypatch):
    case = tmp_path / "2026-11-20_21z"
    rec = sources.record_start(case, START, 201, "--lead", hours=24)
    assert json.loads((case / "raw" / "START.json").read_text()) == rec
    assert (rec["cycle"], rec["lead_h"], rec["start"]) == ("2026-11-12T12", 201, "2026-11-20T21")
    assert rec["what"] == "GFS forecast hour 201 of the 2026-11-12 12Z run"
    # the run reads it: no --lead needed, and the card processes inherit the setting
    assert sources.door_lead(case, "gfs", None, START) == 201
    assert os.environ[sources.LEAD_ENV] == "201"
    assert sources.door_lead(case, "gfs", 201, START) == 201
    with pytest.raises(ValueError, match="differs.*--lead says 198"):
        sources.door_lead(case, "gfs", 198, START)
    # an analysis fetch into the same folder removes the earlier choice, and the run is an analysis run again
    assert sources.record_start(case, START, 0) is None and not (case / "raw" / "START.json").exists()
    del os.environ[sources.LEAD_ENV]
    assert sources.door_lead(case, "gfs", None, START) == 0 and sources.LEAD_ENV not in os.environ
    with pytest.raises(ValueError, match="use --source gfs"):
        sources.door_lead(case, "native", 201, START)
    with pytest.raises(ValueError, match="198 and 201"):
        sources.door_lead(case, "gfs", 200, START - timedelta(hours=1))
    # --at that chose an analysis is recorded too, as hour 0
    at = sources.record_start(case, RUN, 0, "--at")
    assert at["lead_h"] == 0 and "analysis" in at["what"]


def test_receipt_and_credit_words():
    from namgpu import run_forecast
    r = run_forecast._lead_receipt(201, START)
    assert r["source_cycle"] == "2026-11-12T12" and r["source_lead_h"] == 201
    assert "forecast hour 201" in r["start_is"] and "not an analysis" in r["start_is"]
    assert "analysis" not in sources.lead_credit(201, START)


# ------------------------------------------------------------------------------------------ the records

def idx_names(name, pattern):
    lines = gzip.decompress((FIX / name).read_bytes()).decode().splitlines()
    out = []
    for l in lines:
        f = l.split(":")
        if pattern.search(":" + ":".join(f[3:]) + ":"):
            out.append(":".join(f[3:5]))                 # record name and level
    return out


@pytest.mark.parametrize("run", ["2026111212", "2021040100"])
def test_forecast_file_has_the_whole_start_record_set(run):
    """The start reads the same records from hour 201 as from the analysis: every isobaric field, the surface,
    the four soil layers, snow, sea ice, land mask, canopy water and the hydrometeors."""
    a = idx_names(f"gfs.{run}.f000.idx.gz", fetch_case.GFS_INIT_VARS)
    f = idx_names(f"gfs.{run}.f201.idx.gz", fetch_case.GFS_INIT_VARS)
    assert sorted(a) == sorted(f) and len(a) == 419     # today and 2021 alike (397 in 2021 before CLWMR was taken)
    for need in ("CNWAT:surface", "ICEC:surface", "LAND:surface", "SNOD:surface", "WEASD:surface", "TMP:surface",
                 "TSOIL:0-0.1 m below ground", "SOILW:1-2 m below ground", "ICMR:500 mb",
                 "RWMR:850 mb", "SNMR:500 mb", "HGT:500 mb", "PRMSL:mean sea level"):
        assert need in f, need
    # cloud water: NCEP's 2021 index names it CLWMR, today's CLMR (fetch_case.GFS_VARS); the start takes it either way
    assert ("CLWMR:850 mb" if run.startswith("2021") else "CLMR:850 mb") in f
    # and the boundary set is inside it, so the start hour's file serves as its own boundary file
    assert set(idx_names(f"gfs.{run}.f201.idx.gz", fetch_case.GFS_BOCO_VARS)) <= set(f)


# ------------------------------------------------------------------------------------------ comparison sheets

def test_comparison_sheets_read_the_gfs_run_the_start_came_from():
    """The nest door's sheets put the source forecast beside the run: for a start from hour 201 that is the same
    GFS run's hour 201 + lead, not a cycle at the 21Z start (there is none)."""
    from namgpu.io import referee_rows as rr
    row = next(r for r in rr.ROWS if r["name"] == "gfs-forecast")
    asked = []

    class F(rr.Fetcher):
        def __init__(self):
            pass

        def _grib(self, row, cycle, f, product):
            asked.append((cycle, f))
            raise LookupError("asked")

    os.environ[sources.LEAD_ENV] = "201"
    with pytest.raises(LookupError):
        F().grib_product(row, "mslp", START, START + timedelta(hours=3))
    assert asked == [(RUN, 204)]
    del os.environ[sources.LEAD_ENV]
    asked.clear()
    with pytest.raises(LookupError):
        F().grib_product(row, "mslp", RUN, RUN + timedelta(hours=3))
    assert asked == [(RUN, 3)]                                       # an analysis start: as before
    truth = next(r for r in rr.ROWS if r["name"] == "gfs-analysis")
    os.environ[sources.LEAD_ENV] = "201"
    assert rr._start_lead(truth) == 0 and rr._start_lead(row) == 201


def test_every_map_title_says_a_forecast_hour_start():
    from namgpu.io import maps
    plain = maps.panel_title("Mean sea level pressure", START, START + timedelta(hours=3))
    assert plain == "Mean sea level pressure  init 2026-11-20 21Z  valid 2026-11-21 00Z (+3h)"      # as before
    os.environ[sources.LEAD_ENV] = "201"
    said = maps.panel_title("Mean sea level pressure", START, START + timedelta(hours=3))
    assert said == "Mean sea level pressure  init 2026-11-20 21Z from GFS hour 201  valid 2026-11-21 00Z (+3h)"
    # the GFS column of a comparison sheet is titled with the GFS run's own start and hour
    gfs = maps.panel_title("Mean sea level pressure", RUN, START + timedelta(hours=3), "")
    assert gfs == "Mean sea level pressure  init 2026-11-12 12Z  valid 2026-11-21 00Z (+204h)"
