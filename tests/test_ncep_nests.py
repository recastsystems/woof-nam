"""NCEP's own NAM nests (namgpu/nest/ncep_nests.py), CPU only: each row's grid is NCEP's (keys from
nam.v4.2.11 parm/nam_configfile_<nest>, centre = NPS dom_cen), the door turns a row into NCEP's configfile keys, and
--ncep-nest refuses options that would move it off NCEP's placement."""
from __future__ import annotations

import pytest

from namgpu.nest import cli, domain, ncep_nests


def test_alaskanest_grid_is_ncep_keys():
    g = ncep_nests.nest_grid("alaskanest")
    # parm/nam_configfile_alaskanest: im 1189, jm 1249, i/j_parent_start 139/398, ratio 4 in the 954 x 835 parent
    assert (g["im"], g["jm"], g["lm"]) == (1189, 1249, 60)
    assert (g["tph0d"], g["tlm0d"]) == (54.0, -106.0)
    assert g["dlmd"] == pytest.approx(0.0315, abs=1e-12) and g["dphd"] == pytest.approx(0.027, abs=1e-12)
    assert (g["wbd"], g["sbd"]) == (-42.651, -2.16)          # -60.039 + 138 x 0.126, -45.036 + 397 x 0.108
    assert (g["i_parent_start"], g["j_parent_start"], g["parent_child_space_ratio"]) == (139, 398, 4)


FIXED = sorted(k for k, r in ncep_nests.NCEP_NESTS.items() if r.get("placement") is None)


@pytest.mark.parametrize("name", FIXED)
def test_start_is_nps_from_dom_cen(name):
    # parm/nam_namelist.nps_<nest> dom_cen_lat/lon is what NPS placed the grid from: its compute_nest_locations
    # (i_parent_start = nint(1 + (xcenter - (im-1)/2 dlmd_nest - wbd_parent) / dlmd_parent)) gives the configfile's
    # i/j_parent_start, so the grid centre lies within half a parent cell of dom_cen (Hawaii 0.04 deg, Alaska 0)
    from namgpu.nest import ncep_firewx
    r = ncep_nests.NCEP_NESTS[name]
    lat, lon = r["nps_center"]
    assert ncep_firewx.firewx_start(dict(ncep_firewx.NAM12), lat, lon, r["im"], r["jm"], r["ratio"]) ==         (r["i_parent_start"], r["j_parent_start"])
    g = ncep_nests.nest_grid(name)
    assert abs(g["center"][0] - lat) < 0.06 and abs(g["center"][1] - lon) < 0.06


def test_config_keys_carry_ncep_nest_keys(tmp_path):
    g = ncep_nests.nest_grid("alaskanest")
    keys = domain.config_keys(g, "alaskanest", tmp_path / "s.npz", spec_adv=True)
    assert keys["dlmd"] == "0.031500000" and keys["dphd"] == "0.027000000"
    assert (keys["i_parent_start"], keys["j_parent_start"], keys["parent_child_space_ratio"]) == ("139", "398", "4")
    assert keys["wbd"] == "-42.651000" and keys["sbd"] == "-2.160000"
    # nam_configfile_alaskanest: dt 6+1/4, filt_dt 2+0/4, nrads/nradl 192, spec_adv .true., convection none
    assert (keys["dt_int"], keys["dt_num"], keys["dt_den"]) == ("6", "1", "4")
    assert (keys["filt_dt_int"], keys["filt_dt_num"], keys["filt_dt_den"]) == ("2", "0", "4")
    assert keys["nrads"] == keys["nradl"] == "192"
    assert keys["spec_adv"] == ".true." and keys["convection"] == "none"


def test_window_is_centred_on_the_same_grid():
    full = ncep_nests.nest_grid("alaskanest")
    w = ncep_nests.nest_grid("alaskanest", (150, 156))
    assert (w["im"], w["jm"]) == (601, 625)
    assert (w["dlmd"], w["dphd"], w["tph0d"], w["tlm0d"]) == (full["dlmd"], full["dphd"], 54.0, -106.0)
    # whole parent cells inside the full nest, so every window point is a point of the full grid
    di = (w["wbd"] - full["wbd"]) / full["dlmd"]
    dj = (w["sbd"] - full["sbd"]) / full["dphd"]
    assert di == pytest.approx(round(di), abs=1e-6) and dj == pytest.approx(round(dj), abs=1e-6)
    assert 0 < di and di + w["im"] - 1 <= full["im"] - 1 and 0 < dj and dj + w["jm"] - 1 <= full["jm"] - 1
    assert ncep_nests.domain_name(w) == "alaskanest-w150x156"


def test_window_offset_reaches_the_north_east_corner():
    full = ncep_nests.nest_grid("alaskanest")
    w = ncep_nests.nest_grid("alaskanest", (150, 156), (147, 156))
    assert (w["i_parent_start"], w["j_parent_start"]) == (139 + 147, 398 + 156)
    # the window's NE H point is the nest's NE H point
    assert w["wbd"] + (w["im"] - 1) * w["dlmd"] == pytest.approx(full["wbd"] + (full["im"] - 1) * full["dlmd"], abs=1e-9)
    assert w["sbd"] + (w["jm"] - 1) * w["dphd"] == pytest.approx(full["sbd"] + (full["jm"] - 1) * full["dphd"], abs=1e-9)
    assert ncep_nests.domain_name(w) == "alaskanest-w150x156+147+156"
    with pytest.raises(SystemExit, match="leaves"):
        ncep_nests.nest_grid("alaskanest", (150, 156), (148, 0))


def test_window_bigger_than_the_nest_is_refused():
    with pytest.raises(SystemExit, match="--ncep-window"):
        ncep_nests.nest_grid("alaskanest", (400, 10))


def test_unknown_nest_is_refused():
    with pytest.raises(SystemExit, match="no row"):
        ncep_nests.nest_grid("marsnest")


@pytest.mark.parametrize("extra", [["--center", "60N,150W"], ["--two-way"], ["--follow", "tc"]])
def test_door_refuses_moving_ncep_nest(extra, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["nest", "--ncep-nest", "alaskanest", "--source", "gfs", "--start", "2026110306",
                  "--out", str(tmp_path / "o"), *extra])


def test_window_needs_ncep_nest(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["nest", "--center", "60N,150W", "--ncep-window", "10x10", "--source", "gfs", "--start", "2026110306",
                  "--out", str(tmp_path / "o")])


def test_hiresf_shape_row():
    h = ncep_nests.NCEP_NESTS["alaskanest"]["hiresf"]
    assert h["wgrib2"] == "nps:210:60 181.429:1649:2976 40.530:1105:2976"     # ush/nam_prdgen_nest.sh
    assert (h["gdt"]["template"], h["gdt"]["nx"], h["gdt"]["ny"]) == (20, 1649, 1105)
    assert h["packing"]["drt"] == 3 and h["bitmap"]
    assert h["records"]["fNN"] == {"fields": 871, "messages": 805}
    assert h["file"].format() == "woof-nam.tHHz.alaskanest.hiresfFF.tm00.grib2"


@pytest.fixture
def big_cards(monkeypatch):
    """Cards of 95.6 GiB, all free; nvidia-smi is never called."""
    monkeypatch.setattr(domain, "card_total_gib", lambda index=None: 95.6)
    monkeypatch.setattr(domain, "free_gib", lambda index=None: 94.0)
    monkeypatch.setattr(domain, "sm_threads", lambda index=None: 188 * 1536)
    monkeypatch.setattr(domain, "card_name", lambda index=None: "test card 95.6 GiB")
    monkeypatch.delenv("NAMGPU_MEMORY_PATH", raising=False)
    monkeypatch.delenv("NAMGPU_SCRATCH_ARENA", raising=False)
    return monkeypatch


def test_alaska_on_two_big_cards_takes_the_fit_path(big_cards):
    # the full path's model (~150 KB/column, ~110 GiB per card here) refused the suite's 2-card Alaska layout
    g = ncep_nests.nest_grid("alaskanest")
    pf = domain.preflight_cards(g, ["5", "6"])
    assert pf["ok"]
    assert pf["pick"]["memory_path"] == "fit" and "memory_path_switched" in pf["pick"]
    assert pf["tried"][0]["memory_path"] == "full"
    env, rec = {}, {}
    cli._pool_cap(env, rec, pf["pick"])
    assert env["NAMGPU_MEMORY_PATH"] == "fit" and rec["memory_path_switched"]


def test_a_path_set_by_the_caller_is_kept(big_cards):
    big_cards.setenv("NAMGPU_MEMORY_PATH", "full")
    g = ncep_nests.nest_grid("alaskanest")
    pf = domain.preflight_cards(g, ["5", "6"])
    assert not pf["ok"]
    assert all(e["memory_path"] == "full" for e in pf["tried"])


def test_small_nest_on_a_big_card_keeps_the_full_path(big_cards):
    g = ncep_nests.nest_grid("alaskanest", (40, 40))
    pf = domain.preflight(g, free=94.0)
    assert pf["ok"] and pf["pick"]["memory_path"] == "full" and "memory_path_switched" not in pf["pick"]


# ---- Hawaii, Puerto Rico, fire weather (nest-small)

@pytest.mark.parametrize("name,wbd,sbd", [("hawaiinest", -54.873, -22.788), ("priconest", 32.823, -29.484)])
def test_small_nests_are_ncep_keys(name, wbd, sbd, tmp_path):
    g = ncep_nests.nest_grid(name)
    r = ncep_nests.NCEP_NESTS[name]
    assert (g["im"], g["jm"], g["tph0d"], g["tlm0d"]) == (r["im"], r["jm"], 54.0, -106.0)
    assert (g["wbd"], g["sbd"]) == (wbd, sbd)          # -60.039 + (i-1) 0.126, -45.036 + (j-1) 0.108
    keys = domain.config_keys(g, name, tmp_path / "s.npz", spec_adv=True, extra=r.get("keys"))
    assert (keys["dt_int"], keys["dt_num"], keys["dt_den"], keys["nrads"]) == ("6", "1", "4", "192")


def test_firewx_ncep_points_and_noaa_grids():
    from namgpu.nest import ncep_firewx
    g06 = ncep_nests.nest_grid("firewxnest", cycle_hour=6)
    g00 = ncep_nests.nest_grid("firewxnest", cycle_hour=0)
    # NCEP's last production points (nam.tHHz.firewxnest_center_latlon 2026-10-20..11-03), in the CONUS nest
    assert g06["firewx"]["point"] == [42.0, -122.3] and g00["firewx"]["point"] == [39.0, -112.0]
    assert g06["firewx"]["parent"] == g00["firewx"]["parent"] == "conusnest"
    assert (g06["i_parent_start"], g06["j_parent_start"]) == (259, 818)
    assert (g00["i_parent_start"], g00["j_parent_start"]) == (495, 662)
    assert g06["dlmd"] == pytest.approx(0.01575) and g06["dphd"] == pytest.approx(0.0135)
    assert g06["firewx"]["center_latlon"] == "42.0 -122.3\n"     # NOAA's file, byte for byte
    # NOAA's decoded grids; the gridspecs port gives the 00/12Z one exactly
    assert g06["firewx"]["wgrib2"] == "lambert:265:25:25 233.507:646:1270 39.182:389:1270"
    assert g00["firewx"]["wgrib2"] == "lambert:265:25:25 244.377:543:1270 36.487:379:1270"
    assert g00["firewx"]["output_grid"]["new_grid"] == g00["firewx"]["wgrib2"]
    assert ncep_nests.domain_name(g06) == "firewxnest_42.00N_122.30W"
    assert ncep_firewx.FIREWX_LOC_SEED.splitlines()[1] == "06z 42.0 -122.3"


def test_firewx_rule_default_alaska_and_refusal(tmp_path):
    loc = tmp_path / "nam_firewx_loc"
    loc.write_text("00z 0.0 0.0\n06z 64.8 -147.7\n")
    dc = ncep_nests.nest_grid("firewxnest", cycle_hour=0, firewx_loc=loc)
    assert dc["firewx"]["point"] == [38.9, -77.0] and "Washington DC" in dc["firewx"]["point_from"]
    ak = ncep_nests.nest_grid("firewxnest", cycle_hour=6, firewx_loc=loc)
    assert ak["firewx"]["parent"] == "alaskanest" and ak["firewx"]["wgrib2"].startswith("nps:210:60 ")
    assert ncep_nests.nest_grid("firewxnest", cycle_hour=12, firewx_loc=loc)["firewx"]["point"] == [38.9, -77.0]
    with pytest.raises(SystemExit, match="outer 5 rows"):
        ncep_nests.nest_grid("firewxnest", cycle_hour=0, firewx_point=(12.0, -100.0))     # south of the CONUS nest


def test_firewx_configfile_keys(tmp_path):
    r = ncep_nests.NCEP_NESTS["firewxnest"]
    g = ncep_nests.nest_grid("firewxnest", cycle_hour=6)
    keys = domain.config_keys(g, "firewxnest_42.00N_122.30W", tmp_path / "s.npz", spec_adv=True, extra=r["keys"])
    # nam_configfile_firewxnest_conus: dt 2+1/12, filt_dt 1+0/3, nrads/nradl 288, lnsh/lnsv 3, no filter, nodeep false
    assert (keys["dt_int"], keys["dt_num"], keys["dt_den"]) == ("2", "1", "12")
    assert (keys["filt_dt_int"], keys["filt_dt_num"], keys["filt_dt_den"]) == ("1", "0", "3")
    assert keys["nrads"] == keys["nradl"] == "288"
    assert (keys["lnsh"], keys["lnsv"], keys["filter_method"], keys["nodeep"]) == ("3", "3", "0", "false")
    assert (keys["i_parent_start"], keys["j_parent_start"], keys["parent_child_space_ratio"]) == ("259", "818", "2")
    assert r["hours"] == 36


def test_firewx_has_no_window_and_center_only_for_firewx(tmp_path):
    with pytest.raises(SystemExit):
        ncep_nests.nest_grid("firewxnest", (10, 10))
    with pytest.raises(SystemExit):
        cli.main(["nest", "--ncep-nest", "hawaiinest", "--firewx-loc", str(tmp_path / "x"), "--source", "gfs",
                  "--start", "2026110306", "--out", str(tmp_path / "o")])
