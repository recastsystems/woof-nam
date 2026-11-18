"""woof-nam nest door, CPU only (no CUDA, no nvidia-smi): the pool cap follows the memory path the forecast takes,
and --parent-run takes the parent door's --out folder or its run/ folder."""
from __future__ import annotations

import pytest

from namgpu.nest import cli, domain

# the Milton Gulf parent (2026-11-05): 12 km, 3000 x 2600 km at 24.5N 87.5W
MILTON_PARENT = dict(lat=24.5, lon=-87.5, size_km=3000.0, res_km=12, size_y_km=2600.0)


@pytest.fixture
def card(monkeypatch):
    """A card of the given total memory with plenty free; nvidia-smi is never called."""
    def use(total_gib, sms=188):
        monkeypatch.setattr(domain, "card_total_gib", lambda index=None: total_gib)
        monkeypatch.setattr(domain, "sm_threads", lambda index=None: sms * 1536)
        monkeypatch.setattr(domain, "card_name", lambda index=None: f"test card {total_gib} GiB")
    monkeypatch.delenv("NAMGPU_MEMORY_PATH", raising=False)
    monkeypatch.delenv("NAMGPU_SCRATCH_ARENA", raising=False)
    return use


def test_milton_parent_grid():
    g = domain.nest_grid(**MILTON_PARENT)
    assert (g["im"], g["jm"], g["lm"]) == (215, 217, 60)


def test_cap_96gb_card_takes_the_full_path(card):
    # 96 GB card: the forecast auto-selects the full memory path (state.FIT_BELOW_GIB = 64), so the cap must be the
    # full path's. The old single model capped this parent at 12.3 GB and it died at the start of its forecast
    # needing 0.9 GB on top of 12.1 GB in use (Milton nest box, 2026-11-05).
    card(95.6)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=95.0)
    assert pf["ok"]
    e = pf["pick"]
    assert e["memory_path"] == "full"
    assert e["pool_limit_bytes"] > 13.0e9          # above what the parent had in use when it died (12.1 + 0.9 GB)
    assert e["pool_limit_gib"] >= domain.MEASURED_POOL_GIB["milton-parent-full-r32"]


def test_cap_32gb_card_takes_the_fit_path(card):
    card(31.8, sms=170)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=31.0)
    assert pf["ok"]
    e = pf["pick"]
    assert e["memory_path"] == "fit"
    full = domain.vram_estimate(g, 32, path="full", index=None)
    assert full["pool_limit_bytes"] > e["pool_limit_bytes"]     # the full path holds every array for the whole run


NEST_333 = dict(lat=38.6, lon=-98.9, size_km=999.0, res_km=3)
NEST_100 = dict(lat=38.6, lon=-98.9, size_km=300.0, res_km=3)


def test_fit_estimate_covers_measured_peaks(card):
    # process peaks measured through the door on the landed code, RTX 5090 (170 SMs), fit path (domain.py note)
    card(31.4, sms=170)
    for grid, rows, key in ((NEST_333, 32, "nest-286x333-fit-r32"), (NEST_333, 4, "nest-286x333-fit-r4"),
                            (NEST_100, 32, "nest-300km-fit-r32")):
        e = domain.vram_estimate(domain.nest_grid(**grid), rows, index=None)
        assert e["memory_path"] == "fit"
        # the estimate covers the measured peak (0.1.3: no longer 1.2x it; the door keeps margin_gib on top)
        assert e["total_gib"] >= domain.MEASURED_PEAK_GIB[key], (key, e)
        # the cap holds the measured pool (peak less local memory and context)
        assert e["pool_limit_gib"] >= domain.MEASURED_PEAK_GIB[key] - e["lmem_gib"] - e["fixed_gib"], (key, e)


def test_333_nest_admitted_on_an_8gb_card(card):
    # the 0.1.0 model wanted about 18 GiB for this nest and refused it on any card with less free
    card(7.6, sms=34)
    pf = domain.preflight(domain.nest_grid(**NEST_333), free=7.3)
    assert pf["ok"], pf
    assert pf["pick"]["memory_path"] == "fit"


def test_300km_nest_admitted_on_a_4gb_card(card):
    card(3.8, sms=24)
    pf = domain.preflight(domain.nest_grid(**NEST_100), free=3.7)
    assert pf["ok"], pf


def test_card_sms_longer_names_first():
    names = [k for k, _ in domain.CARD_SMS]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert a not in b, f"{a!r} is listed before {b!r}, which contains it"


def test_memory_path_setting_wins(card, monkeypatch):
    card(95.6)
    monkeypatch.setenv("NAMGPU_MEMORY_PATH", "fit")
    assert domain.memory_path()["path"] == "fit"
    monkeypatch.setenv("NAMGPU_MEMORY_PATH", "full")
    card(31.8)
    assert domain.memory_path()["path"] == "full"
    monkeypatch.delenv("NAMGPU_MEMORY_PATH")
    monkeypatch.setenv("NAMGPU_SCRATCH_ARENA", "0")      # the older spelling, as state.memory_path reads it
    assert domain.memory_path()["path"] == "full"


def test_no_card_size_no_cap(card):
    # nvidia-smi gave no total: the door cannot tell which path the forecast takes, so it sets no cap
    card(0.0)
    g = domain.nest_grid(**MILTON_PARENT)
    pf = domain.preflight(g, rows=32, free=95.0)
    assert pf["pick"]["pool_limit_bytes"] is None


def _hist(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "wrfout_d01_2024-10-07_00:00:00").write_bytes(b"")


def test_parent_run_door_out_folder(tmp_path):
    out = tmp_path / "parent12"
    _hist(out / "run")
    assert cli.resolve_parent_run(out) == (out / "run").resolve()


def test_parent_run_history_folder(tmp_path):
    out = tmp_path / "parent12"
    _hist(out / "run")
    assert cli.resolve_parent_run(out / "run") == (out / "run").resolve()
    flat = tmp_path / "flat"
    _hist(flat)
    assert cli.resolve_parent_run(flat) == flat.resolve()


def test_parent_run_refusal_names_folders(tmp_path):
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        cli.resolve_parent_run(empty)
    msg = str(e.value)
    assert str(empty) in msg and str(empty / "run") in msg and "wrfout_d01_" in msg
    with pytest.raises(SystemExit):
        cli.resolve_parent_run(None)


def test_free_memory_query_survives_a_slow_nvidia_smi(monkeypatch):
    # 0.2.0 release pass (2026-11-09): on a host with a load average in the thousands nvidia-smi took over 20 s and
    # the nest door died with an uncaught subprocess.TimeoutExpired. One retry with a longer wait, then one plain line.
    import subprocess
    waits = []

    def slow_then_ok(cmd, **kw):
        waits.append(kw["timeout"])
        if len(waits) == 1:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return subprocess.CompletedProcess(cmd, 0, stdout="49152", stderr="")
    monkeypatch.setattr(domain.subprocess, "run", slow_then_ok)
    assert domain.free_gib("0") == 48.0
    assert waits == [20, 120]

    def never(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(domain.subprocess, "run", never)
    with pytest.raises(SystemExit) as e:
        domain.free_gib("3")
    msg = str(e.value)
    assert "card 3" in msg and "nvidia-smi did not answer" in msg and "out of memory" in msg


def test_lead_start_on_the_nest_door(monkeypatch):
    """--lead H: the start is forecast hour H of the GFS run H hours before --start; the boundaries are that run's
    later hours; every map credit and file list says so; what cannot be done is refused in plain words."""
    import os
    from argparse import Namespace
    from datetime import datetime

    from namgpu.init import sources
    monkeypatch.setenv(sources.LEAD_ENV, "")           # restored by monkeypatch whatever the door sets below

    def args(**kw):
        return Namespace(**(dict(lead=201, source="auto", boco_age_h=None, start="2026112021", hours=24.0) | kw))
    a = args()
    assert cli.lead_start(a) == 201 and a.source == "gfs" and os.environ[sources.LEAD_ENV] == "201"
    age, why = cli.boco_age_for("gfs", datetime(2026, 11, 20, 21), None)
    assert age == 201 and "forecast hour 201 of the 2026-11-12 12Z run" in why
    credit, lines = cli.data_credits("gfs")
    assert "start: GFS forecast hour 201" in credit and "forecast hour 201" in lines[0]
    # an hour GFS does not post: the nearest posted hour before it, the door's start moved there (window.py)
    a200 = args(lead=200, start="2026112020")
    assert cli.lead_start(a200) == 198 and a200.start == "2026112018" and a200.window["offset_h"] == 2.0
    # a window past hour 384 is cut there, not refused
    a190 = args(hours=190.0)
    assert cli.lead_start(a190) == 201 and a190.window["cut_h"] == 7.0
    for kw, msg in ((dict(source="parent"), "starts when its parent does"),
                    (dict(boco_age_h=6), "same run's later hours"),
                    (dict(start="2026112022"), "00, 06, 12 or 18Z")):
        with pytest.raises(ValueError, match=msg):
            cli.lead_start(args(**kw))
    with pytest.raises(ValueError, match="--follow"):
        cli.lead_start(args(), following=True)
    # no --lead: an analysis start, as before, and nothing inherited from the calling shell
    a0 = args(lead=None)
    assert cli.lead_start(a0) == 0 and a0.source == "auto" and sources.LEAD_ENV not in os.environ
    assert cli.boco_age_for("gfs", datetime(2024, 11, 5, 12), None)[0] == 0
    assert cli.boco_age_for("gfs", datetime(2024, 11, 5, 12), 6) == (6, "given (--boco-age-h)")
    assert "forecast hour" not in cli.data_credits("gfs")[1][0]
