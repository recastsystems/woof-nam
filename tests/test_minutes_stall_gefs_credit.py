"""A start with minutes counts light frames from the run's start; a stalled S3 range waits 20 s; the GEFS receipt
names no hosting company."""
import urllib.request

from namgpu import outputs
from namgpu.init import fetch_case, sources


def test_start_at_half_past_keeps_hourly_frames_whole():
    # model start 18:00, run start 18:30: hourly output at 18:30, 19:30, ... is lead hour 0, 1, 2, 3
    s = outputs.Schedule(25.0, 3.5, 0.5, write="1h", light_fields=outputs.MAP_FIELDS)
    hourly = [e for e in s.entries.values() if e["carrier"] is None]
    assert [e["h"] for e in hourly] == [0.0, 1.0, 2.0, 3.0]
    assert not any(e["light"] for e in hourly)


def test_start_at_half_past_five_minute_frames_light_off_lead_hour():
    s = outputs.Schedule(25.0, 1.5, 0.5, write="5m", light_fields=outputs.MAP_FIELDS)
    for e in s.entries.values():
        if e["carrier"] is None and not e["grib"]:
            assert e["light"] == (abs(e["h"] - round(e["h"])) > 1e-9), e


def test_start_at_twenty_past_with_inexact_step():
    # dt that does not divide 3600: the step nearest each whole lead hour is whole
    s = outputs.Schedule(26.0 + 2.0 / 3.0, 2.0 + 1.0 / 3.0, 1.0 / 3.0, write="1h", light_fields=outputs.MAP_FIELDS)
    assert not any(e["light"] for e in s.entries.values() if e["carrier"] is None)


def test_s3_range_stall_wait_is_20_s(monkeypatch):
    seen = []

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"x"

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: (seen.append(timeout), R())[1])
    monkeypatch.delenv("RECAST_NAM_FETCH_STALL_S", raising=False)
    fetch_case._get("https://noaa-nam-pds.s3.amazonaws.com/x.grib2", "0-9", timeout=300)
    fetch_case._get("https://www.ncei.noaa.gov/x.grib2", "0-9", timeout=300)
    fetch_case._get("https://noaa-gefs-pds.s3.amazonaws.com/y")
    monkeypatch.setenv("RECAST_NAM_FETCH_STALL_S", "0")
    fetch_case._get("https://noaa-nam-pds.s3.amazonaws.com/x.grib2", "0-9", timeout=300)
    assert seen == [20.0, 300, 20.0, 300]


def test_gefs_licence_names_noaa_not_a_hosting_company():
    lic = sources.SOURCES["gefs"]["licence"] if "gefs" in sources.SOURCES else next(
        r["licence"] for r in sources.SOURCES.values() if "gefs" in str(r.get("fetch", {}).get("host", "")))
    assert "NOAA" in lic and "AWS" not in lic and "Amazon" not in lic
