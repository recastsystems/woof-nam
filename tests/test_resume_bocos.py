"""woof-nam run --resume to a later end than the first leg: the boundary files the first leg never needed are built
from the case's raw inputs before the first step, or the run stops before it in one line naming the fetch.

Breakage it prevents: `woof-nam run --resume <run>/restart --hours 60` after a run made with `--hours 20` stopped at
+21 h with FileNotFoundError .../boco/boco.0021 (nam-gust case 2022-01-03 12Z, logs c4-run-20.log / c4-run-20b.log,
2026-11-11 02:02Z): the init builds boco.NNNN only to the first leg's end, and the resume path only pointed at that
folder. CPU only: the builder is a stub that writes the files it was asked for."""
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.init import gfs_bocos, sources  # noqa: E402

START = datetime(2022, 1, 3, 12)


def _ctx(tmp_path, monkeypatch, first_leg_h=20, raw_to_h=21):
    monkeypatch.setenv("NAMGPU_BOCO_AGE_H", "0")
    case = tmp_path / "cases" / "2022-01-03_12z"
    bdir = case / "boco"
    bdir.mkdir(parents=True)
    for fh in range(0, 21, 3):                    # what the first leg's init built: bc_hours(20, 3) = 21 -> 0..18
        if fh < first_leg_h:
            (bdir / f"boco.{fh:04d}").write_bytes(b"b")
    for fh, p in gfs_bocos.source_files(case, START, 63, sources.SOURCES["gfs"], SimpleNamespace(raw={})):
        if fh <= raw_to_h:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"g")
    return SimpleNamespace(cfg=SimpleNamespace(raw={}), init_source="native", case_dir=case, start=START,
                           boco_dir=str(bdir))


class _Calls(list):
    only = None


def _stub(calls):
    def build(ctx, hours, out_dir, log=print, from_hour=0, generation=None, only=None):
        calls.append((hours, from_hour))
        only_seen.append(None if only is None else list(only))
        for fh in (range(from_hour, hours, 3) if only is None else only):
            (Path(out_dir) / f"boco.{fh:04d}").write_bytes(b"new")
        return out_dir
    only_seen = calls.only = []
    return build


def test_missing_inputs_stop_before_the_first_step_with_the_fetch(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch)
    calls = _Calls()
    with pytest.raises(SystemExit) as e:
        gfs_bocos.resume_check(ctx, 20.0, 60.0, log=lambda m: None, build_fn=_stub(calls))
    msg = str(e.value)
    print(msg)
    assert calls == []
    assert "boco.0021 to boco.0057" in msg
    assert "woof-nam fetch --cycle 2022010312 --hours 60" in msg and "--source native --boco-age 0" in msg
    assert "raw/gfs/gfs.t12z.f024.grib2" in msg


def test_inputs_present_build_the_missing_files_from_the_resume_interval(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, raw_to_h=60)
    calls, lines = _Calls(), []
    built = gfs_bocos.resume_check(ctx, 20.0, 60.0, log=lines.append, build_fn=_stub(calls))
    assert calls == [(60, 21)], calls
    assert calls.only == [list(range(21, 60, 3))], calls.only
    assert built == [f"boco.{fh:04d}" for fh in range(21, 60, 3)]
    assert all((Path(ctx.boco_dir) / f"boco.{fh:04d}").is_file() for fh in range(18, 60, 3))
    assert (Path(ctx.boco_dir) / "boco.0018").read_bytes() == b"b"     # the first leg's files are kept
    assert len(lines) == 1 and "boco.0021 to boco.0057" in lines[0], lines


def test_a_hole_is_filled_and_the_files_that_are_there_are_not_rebuilt(tmp_path, monkeypatch):
    """boco.0006 is gone, boco.0009 to boco.0018 are there: only boco.0006 is asked of the builder, and only its two
    raw times (f006, f009) must be in the case folder. Before, the builder was asked for every file from the first
    missing one to the end, and the later files were written again."""
    ctx = _ctx(tmp_path, monkeypatch, raw_to_h=9)
    (Path(ctx.boco_dir) / "boco.0006").unlink()
    calls = _Calls()
    built = gfs_bocos.resume_check(ctx, 3.0, 20.0, log=lambda m: None, build_fn=_stub(calls))
    assert built == ["boco.0006"]
    assert calls == [(21, 6)] and calls.only == [[6]], (calls, calls.only)
    assert (Path(ctx.boco_dir) / "boco.0006").read_bytes() == b"new"
    for fh in (3, 9, 12, 15, 18):
        assert (Path(ctx.boco_dir) / f"boco.{fh:04d}").read_bytes() == b"b"


def test_nothing_missing_builds_nothing(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch)
    calls = _Calls()
    assert gfs_bocos.resume_check(ctx, 12.0, 20.0, log=lambda m: None, build_fn=_stub(calls)) == []
    assert calls == []


def test_the_resume_door_calls_the_check_before_the_loop():
    src = (Path(__file__).resolve().parents[1] / "namgpu" / "run_forecast.py").read_text()
    i_check = src.find("gfs_bocos.resume_check(ctx")
    i_boco = src.find("rstmod.resume_boco(ctx, a.resume)")
    i_loop = src.find("drv_loop_limits.run(ctx)", i_boco)
    assert 0 < i_boco < i_check < i_loop, (i_boco, i_check, i_loop)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
