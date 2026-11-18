"""The NAM 12 km static surface is one file per start month (static/nam12_static.mMM.npz), and the run reads that month's.

Before: domains.ensure_static built static/nam12_static.npz once, from the archived NAM analysis of the first start's
month and day, and every later start in any month reused it (greenness VEG and albedo ALBDO are monthly; the file
records no date): a July start on a work root first used in November ran on November greenness. CPU only: the
download and the build are stand-ins."""
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_each_month_gets_its_own_file_and_the_run_reads_it(monkeypatch, tmp_path):
    from namgpu import domains
    from namgpu.init import gfs_start, fetch_case
    monkeypatch.setattr(gfs_start, "STATIC_NPZ", tmp_path / "static" / "nam12_static.npz")
    monkeypatch.setattr(domains, "current", lambda: {"name": "nam12", "static": "nam-analysis"})
    monkeypatch.setattr(domains.paths, "work_root", lambda: tmp_path)
    monkeypatch.delenv(gfs_start.NAM_STATIC_ENV, raising=False)
    got = []

    def download(url, dst):
        got.append(url)
        Path(dst).write_bytes(b"grib")
        return str(dst)

    def child(mod, args, card):
        out = Path(args[args.index("--out") + 1])
        out.write_bytes(b"npz " + Path(args[0]).name.encode())
        return 0

    monkeypatch.setattr(fetch_case, "_download", download)
    monkeypatch.setattr(fetch_case, "_subset", lambda url, dst, pat: None)
    monkeypatch.setattr(domains, "_child", child)
    case = tmp_path / "case"
    case.mkdir()
    nov = domains.ensure_static(case, datetime(2026, 11, 14, 12), "gfs", log=lambda *a: None)
    assert nov["action"] == "built" and nov["file"].endswith("nam12_static.m11.npz")
    assert os.environ[gfs_start.NAM_STATIC_ENV] == nov["file"]
    assert gfs_start.nam_static_npz() == Path(nov["file"])
    jul = domains.ensure_static(case, datetime(2026, 7, 4, 0), "gfs", log=lambda *a: None)
    assert jul["action"] == "built" and jul["file"].endswith("nam12_static.m07.npz")
    assert "nam.20260704" in got[-1]                                   # July's own analysis, not November's file
    assert gfs_start.nam_static_npz() == Path(jul["file"])
    again = domains.ensure_static(case, datetime(2026, 11, 15, 0), "gfs", log=lambda *a: None)
    assert again["action"] == "cached" and again["file"] == nov["file"]
    assert gfs_start.nam_static_npz() == Path(nov["file"])
    assert not list((tmp_path / "static").glob("*.part*"))
