"""woof-nam suite's table and plan (CPU only): NCEP's nest shapes, the fire-weather point rule, card layouts, names."""
from datetime import datetime, timezone

import pytest

from namgpu.suite import table as T

NCEP = {"conusnest": (1827, 1467), "alaskanest": (1189, 1249), "hawaiinest": (373, 561), "priconest": (401, 325),
        "firewxnest": (333, 333)}


def cyc(h):
    return datetime(2026, 11, 6, h, tzinfo=timezone.utc)


def test_every_nest_has_ncep_point_counts_and_spacing():
    t = T.load()
    p = T.plan(t, cyc(0), [str(i) for i in range(8)])
    assert list(p["domains"]) == list(T.SUITE_ORDER)
    for d, (im, jm) in NCEP.items():
        g = p["domains"][d]["grid"]
        assert (g["im"], g["jm"]) == (im, jm), d
    assert p["domains"]["conusnest"]["grid"]["dlmd"] == pytest.approx(0.0315)
    assert p["domains"]["conusnest"]["grid"]["dphd"] == pytest.approx(0.027)
    assert p["domains"]["firewxnest"]["grid"]["dlmd"] == pytest.approx(0.01575)
    assert p["domains"]["12km"]["hours"] == 84
    assert {d: p["domains"][d]["hours"] for d in NCEP} == {"conusnest": 60, "alaskanest": 60, "hawaiinest": 60,
                                                           "priconest": 60, "firewxnest": 36}


def test_firewx_points_are_ncep_last_production_points():
    t = T.load()
    assert T.read_firewx_loc() == {0: (39.0, -112.0), 6: (42.0, -122.3), 12: (39.0, -112.0), 18: (42.0, -122.3)}
    for h, want in ((0, (39.0, -112.0)), (6, (42.0, -122.3))):
        fw = T.firewx_point(t, cyc(h))
        assert (fw["lat"], fw["lon"], fw["region"], fw["how"]) == (*want, "conus", "nam_firewx_loc")
    assert T.write_center_latlon(42.0, -122.3) == "42.0 -122.3\n"


def test_firewx_default_and_alaska_and_bad_point(tmp_path):
    t = T.load()
    empty = tmp_path / "loc"
    empty.write_text("00z  0.0    0.0\n")                       # no point entered: NCEP's DC default
    fw = T.firewx_point(t, cyc(0), empty)
    assert (fw["lat"], fw["lon"], fw["region"]) == (38.9, -77.0, "conus")
    ak = tmp_path / "ak"
    ak.write_text("00z 61.2 -149.9\n")                          # Anchorage: lat >= 52 -> the Alaska nest
    assert T.firewx_point(t, cyc(0), ak)["region"] == "alaska"
    bad = tmp_path / "bad"
    bad.write_text("00z 12.0 -100.0\n")                         # south of the CONUS nest
    fw = T.firewx_point(t, cyc(0), bad)
    assert (fw["lat"], fw["lon"]) == (38.9, -77.0) and "failed the check" in fw["how"]


def test_layouts_and_door_args():
    t = T.load()
    p = T.plan(t, cyc(6), [str(i) for i in range(8)])
    assert p["domains"]["12km"]["cards"] == ["0"]
    assert p["domains"]["conusnest"]["cards"] == ["1", "2", "3", "4"]
    used = [c for r in p["domains"].values() for c in r["cards"]]
    assert sorted(set(used)) == [str(i) for i in range(8)]
    # 4 cards: the CONUS nest on all four (measured 2026-11-10 at 63-68 s per hour; on two cards it was about twice
    # that), the small nests off card 0, which holds the parent and the CONUS GRIB2 follower
    four = T.plan(t, cyc(6), [str(i) for i in range(4)])
    assert four["domains"]["conusnest"]["cards"] == ["0", "1", "2", "3"]
    assert all("0" not in four["domains"][d]["cards"] for d in ("alaskanest", "hawaiinest", "priconest", "firewxnest"))
    a4 = T.door_argv("conusnest", four["domains"]["conusnest"], cyc(6), __import__("pathlib").Path("/p"),
                     __import__("pathlib").Path("/o"))
    assert a4[a4.index("--card-list") + 1] == "0,1,2,3"
    one = T.plan(t, cyc(6), ["3"])
    assert all(r["cards"] == ["3"] for r in one["domains"].values())
    argv = T.door_argv("conusnest", p["domains"]["conusnest"], cyc(6), __import__("pathlib").Path("/p"),
                       __import__("pathlib").Path("/o"))
    assert argv[:2] == ["nest", "--center"] and "--source" in argv
    ncep = T.door_argv("firewxnest", p["domains"]["firewxnest"], cyc(6), __import__("pathlib").Path("/p"),
                       __import__("pathlib").Path("/o"), "--ncep-nest NAME --firewx-center LAT,LON")
    assert ncep[:3] == ["nest", "--ncep-nest", "firewxnest"] and ncep[ncep.index("--firewx-center") + 1] == "42.0,-122.3"
    assert "--center" not in ncep and ncep[ncep.index("--hours") + 1] == "36"
    # a door with --ncep-nest but no row for this nest: NCEP's counts on the nest's own pole
    norow = T.door_argv("conusnest", p["domains"]["conusnest"], cyc(6), __import__("pathlib").Path("/p"),
                        __import__("pathlib").Path("/o"), "--ncep-nest NAME", {"alaskanest"})
    assert norow[:2] == ["nest", "--center"] and argv[argv.index("--source") + 1] == "parent"
    assert argv[argv.index("--card-list") + 1] == "1,2,3,4"
    assert "--cards" not in T.door_argv("hawaiinest", p["domains"]["hawaiinest"], cyc(6), __import__("pathlib").Path("/p"),
                                        __import__("pathlib").Path("/o"))
    with pytest.raises(SystemExit):
        T.plan(t, datetime(2026, 11, 6, 3, tzinfo=timezone.utc), ["0"])


def test_file_names_are_noaa_names_with_our_prefix():
    t = T.load()
    assert T.file_name(t, "conusnest", cyc(6), 7) == "woof-nam.t06z.conusnest.hiresf07.tm00.grib2"
    assert T.file_name(t, "firewxnest", cyc(0), 36) == "woof-nam.t00z.firewxnest.hiresf36.tm00.grib2"
    assert T.file_name(t, "12km", cyc(18), 84) == "woof-nam.t18z.awphys84.tm00.grib2"
    assert T.hour_of(t, "conusnest", "woof-nam.t06z.conusnest.hiresf07.tm00.grib2.idx") == 7
    assert T.hour_of(t, "alaskanest", "woof-nam.t06z.conusnest.hiresf07.tm00.grib2") is None
    assert T.hour_of(t, "12km", "woof-nam.t18z.awphys84.tm00.grib2") == 84


def test_hours_cap_keeps_the_parent_long_enough_for_its_nests():
    # release check 2026-11-09: `woof-nam suite --hours-cap 1` ran a 1 h parent, and every nest failed waiting for
    # the parent hour 6 its door reads
    t = T.load()
    p = T.plan(t, cyc(12), [str(i) for i in range(2)], hours_cap=1)
    assert p["domains"]["12km"]["hours"] == 6
    assert {d: p["domains"][d]["hours"] for d in NCEP} == {d: 1 for d in NCEP}
    assert T.parent_hours_for_nest(1) == 6 and T.parent_hours_for_nest(7) == 9 and T.parent_hours_for_nest(60) == 60
    p = T.plan(t, cyc(12), [str(i) for i in range(2)], only=["12km"], hours_cap=1)
    assert p["domains"]["12km"]["hours"] == 1          # no nests: the cap holds


def test_a_start_from_a_forecast_hour_need_not_be_a_suite_cycle():
    """woof-nam suite --cycle 2026111212 --lead 201: the suite starts at 2026-11-20 21Z, files are named for that
    start, and the fire-weather nest takes the default point (nam_firewx_loc has no 21Z line)."""
    t = T.load()
    start = datetime(2026, 11, 20, 21, tzinfo=timezone.utc)
    with pytest.raises(SystemExit):
        T.plan(t, start, ["0"])
    p = T.plan(t, start, ["0"], lead=201)
    assert (p["cycle"], p["gfs_cycle"], p["lead_h"]) == ("2026-11-20T21Z", "2026-11-12T12Z", 201)
    assert set(p["domains"]) == set(T.domains(t))
    assert p["domains"]["firewxnest"]["firewx"]["how"].startswith("default")
    assert T.file_name(t, "12km", start, 3).endswith(".t21z.awphys03.tm00.grib2")
    assert "lead_h" not in T.plan(t, cyc(6), ["0"])


def test_suite_door_takes_a_lead_with_or_without_a_parent_run(tmp_path, capsys):
    """--cycle is the GFS run and --lead the hour: the plan's start is cycle + lead, alone or beside --parent-run (the
    nests of a parent that was itself started from that hour, which is how a worker box runs the suite)."""
    from namgpu.suite import cli as suite_cli
    base = ["--cycle", "2026111212", "--lead", "201", "--card-list", "0", "--dry-run", "--hours-cap", "6"]
    assert suite_cli.main([*base, "--out", str(tmp_path / "a")]) == 0
    plan = __import__("json").loads((tmp_path / "a" / "suite-plan.json").read_text())
    assert plan["cycle"] == "2026-11-20T21Z" and plan["lead_h"] == 201 and "12km" in plan["domains"]
    assert suite_cli.main([*base, "--out", str(tmp_path / "b"), "--parent-run", str(tmp_path / "p")]) == 0
    nests = __import__("json").loads((tmp_path / "b" / "suite-plan.json").read_text())
    assert nests["cycle"] == "2026-11-20T21Z" and "12km" not in nests["domains"]
    # hour 200 is not posted: the suite starts at the nearest posted hour before it and says so
    assert suite_cli.main(["--cycle", "2026111212", "--lead", "200", "--card-list", "0", "--dry-run", "--out",
                           str(tmp_path / "c"), "--hours-cap", "6"]) == 0
    assert "hour 200 is not posted" in capsys.readouterr().out
    assert __import__("json").loads((tmp_path / "c" / "suite-plan.json").read_text())["lead_h"] == 198
    with pytest.raises(SystemExit, match="--cycle is the GFS run"):
        suite_cli.main(["--cycle", "2026111215", "--lead", "198", "--card-list", "0", "--dry-run", "--out", str(tmp_path / "d")])
    # the whole 84 h parent does not fit after hour 378: it is cut at hour 384, and the suite says so
    assert suite_cli.main(["--cycle", "2026111212", "--lead", "378", "--card-list", "0", "--dry-run", "--out",
                           str(tmp_path / "e")]) == 0
    assert "cut 78 h short" in capsys.readouterr().out
