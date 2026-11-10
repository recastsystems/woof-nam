"""woof-nam suite --no-products: the run passes on its forecasts alone and keeps every history.

t3k-conus4-12h-noprod (2026-11-10): a clean 12 km + 3 km CONUS forecast with --no-products returned rc 1, "13 GRIB2
hours missing", because the finish counted the GRIB2 hours the run had been told not to write."""
import threading
from types import SimpleNamespace

from namgpu.suite import cli


def _suite(tmp_path, no_products):
    s = cli.Suite.__new__(cli.Suite)
    s.lock = threading.Lock()
    s.a = SimpleNamespace(no_products=no_products, keep_history=False, parent_run=None)
    s.out = tmp_path
    s.pub = tmp_path / "publish"
    s.t = {"parent": {"domain": "12km"}, "prefix": "woof-nam", "nests": {}}
    s.plan = {"domains": {"12km": {"kind": "parent", "hours": 2, "cards": ["0"]},
                          "conusnest": {"kind": "nest", "hours": 2, "cards": ["0"]}}}
    s.cycle = __import__("datetime").datetime(2026, 1, 23, 12)
    s.rec = {"domains": {"12km": {"state": "complete"}, "conusnest": {"state": "complete"}}}
    hist = tmp_path / "conusnest" / "run" / "wrfout_d01_2026-01-23_12:00:00"
    hist.parent.mkdir(parents=True)
    hist.write_bytes(b"x")
    return s, hist


def test_no_products_passes_on_complete_forecasts_and_keeps_history(tmp_path):
    s, hist = _suite(tmp_path, no_products=True)
    for d in s.plan["domains"]:
        s.finish_domain(d)
    assert s.not_complete() == []
    assert hist.is_file()                                   # the history is the run's only output
    assert not (s.pub / "conusnest" / "cycle.json").exists()
    assert s.rec["domains"]["conusnest"]["publish_state"] == "not written (--no-products)"


def test_no_products_still_fails_a_failed_forecast(tmp_path):
    s, _ = _suite(tmp_path, no_products=True)
    s.rec["domains"]["conusnest"]["state"] = "failed"
    for d in s.plan["domains"]:
        s.finish_domain(d)
    assert s.not_complete() == ["conusnest"]


def test_with_products_missing_grib2_hours_still_fail(tmp_path):
    s, hist = _suite(tmp_path, no_products=False)
    for d in s.plan["domains"]:
        s.finish_domain(d)
    assert sorted(s.not_complete()) == ["12km", "conusnest"]
    assert "GRIB2 hours missing" in s.rec["domains"]["conusnest"]["error"]
    assert hist.is_file()                                   # not complete: the history is kept for a rerun
