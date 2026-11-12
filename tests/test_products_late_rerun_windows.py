"""woof-nam products: an hour whose hand-off frame meets a full card and is rerun does not leave the later hours'
window records incomplete or stale.

The question (WOOF site finding, 2026-11-11 08:34Z): in the site's own map worker, an hour refused for card memory
was tried again after later hours had been handed on, and those later hours' window products (1 h rain, 3 h
updraft-helicity maximum, run maxima) came out incomplete. In the package the frame rerun (commit b72c607,
_start_hour_fits) runs inside the follower's hour loop, so hour N is retried before hour N+1 is started, and every
window record of an hour is computed from history files on disk (the precipitation buckets difference this hour's
RAINNC/RAINC against the history file at the bucket start, _buckets; the hourly maxima are the hour's own history
fields; the writer gets one frame per process), never from frames handed on earlier. This test forces it: hours 2 and
3 and 4 are ready together, hour 2's frame fails once with a CuPy out-of-memory error, and the later hours' bucket
and maximum records must equal those of a run with no failure, and the true differences of the history totals.
CPU only: small synthetic history files; the real follower loop, the real rerun and the real bucket code."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ["NAMGPU_PRODUCTS_OOM_WAIT_S"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io import products as P  # noqa: E402

JM, IM = 9, 11
START = "2020-10-24_00:00:00"


class OutOfMemoryError(Exception):
    """Stands in for cupy.cuda.memory.OutOfMemoryError (matched by name, as products does)."""


def _total(h):
    """Accumulated precipitation since the start at hour h (mm), different at every point and hour."""
    j, i = np.mgrid[0:JM, 0:IM]
    nc = (0.7 * h * h + 0.1 * h * (1 + j % 3) + 0.01 * h * i).astype(np.float32)
    c = (0.3 * h + 0.05 * h * (i % 2)).astype(np.float32)
    return nc, c


def _histories(run: Path, hours=range(0, 7)):
    import netCDF4
    run.mkdir(parents=True, exist_ok=True)
    for h in hours:
        p = run / f"wrfout_d01_2020-10-24_{h:02d}:00:00"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("Time", None)
            d.createDimension("south_north", JM)
            d.createDimension("west_east", IM)
            d.START_DATE = START
            d.createVariable("XTIME", "f4", ("Time",))[0] = 60.0 * h
            nc, c = _total(h)
            for name, a in (("RAINNC", nc), ("RAINC", c),
                            ("U10MAX", np.full((JM, IM), 10.0 + h, np.float32)),       # the hour's own maximum
                            ("UPHLMAX25", np.full((JM, IM), 5.0 * h, np.float32))):
                d.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = a


def _tables():
    """Grid-218 tables that put the model grid's interior on the output grid point for point."""
    jj, ii = np.mgrid[0:JM - 1, 0:IM - 1]
    z = np.zeros((JM - 1, IM - 1), np.float32)
    return {"j0": jj, "i0": ii, "wx": z, "wy": z}


def _run(tmp_path: Path, monkeypatch, fail_hour=None, name="run"):
    import netCDF4
    run = tmp_path / name
    _histories(run)
    t = _tables()
    attempts, records, failed = [], {}, set()

    def fake_start(f, out, profile, device, bits, pool, ps=None, gkey=None, cctx=None, file_name=None):
        h = P._xtime_hour(f)
        attempts.append(h)
        if h == fail_hour and h not in failed:
            failed.add(h)
            raise OutOfMemoryError("Out of memory allocating 83,886,080 bytes (allocated so far: 1,346,371,584 bytes)")
        with netCDF4.Dataset(f) as s:
            def get(n):
                return P._regrid(t, np.asarray(s[n][0], dtype=np.float32))
            rec = dict(P._buckets(f, s, get, t, P.PSETS.AWPHYS))
            rec["U10MAX"] = get("U10MAX")
            rec["UPHLMAX25"] = get("UPHLMAX25")
        records[h] = rec
        return h

    def fake_finish(h, keep_work=False):
        return {"file": f"f{h:02d}", "history": str(h), "fhr": h, "bytes": 1, "records": len(records[h]),
                "frame_s": 0.0, "writer_s": 0.0}

    monkeypatch.setattr(P, "_start_hour", fake_start)
    monkeypatch.setattr(P, "_finish_hour", fake_finish)
    res = P.run(run, out=tmp_path / f"{name}-out", follow=False, set_name="awphys", workers=2)
    return attempts, records, res


def test_late_rerun_keeps_later_window_records(tmp_path, monkeypatch):
    clean_attempts, clean, _ = _run(tmp_path, monkeypatch, None, "clean")
    attempts, recs, res = _run(tmp_path, monkeypatch, 2, "oom")
    print("attempts:", attempts)
    assert res["errors"] == [], res["errors"]
    # hour 2 is retried at once, before hour 3 is started: the order is kept
    assert attempts == [0, 1, 2, 2, 3, 4, 5, 6], attempts
    assert clean_attempts == [0, 1, 2, 3, 4, 5, 6]
    for h in (2, 3, 4):
        assert sorted(recs[h]) == sorted(clean[h]), (h, sorted(recs[h]), sorted(clean[h]))
        for k in recs[h]:
            assert np.array_equal(recs[h][k], clean[h][k]), (h, k)

    def tot(h):
        nc, c = _total(h)
        return (nc + c)[:JM - 1, :IM - 1]
    # the true windows: 3 h bucket 3-4 at f04 (since the last 3 h reset), 12 h bucket 0-4, and 0-3 at f03
    assert np.allclose(recs[3]["APCP_S3"], tot(3) - tot(0), atol=1e-5)
    assert np.allclose(recs[4]["APCP_S3"], tot(4) - tot(3), atol=1e-5)
    assert np.allclose(recs[4]["APCP_S12"], tot(4), atol=1e-5)
    assert np.allclose(recs[5]["APCP_S3"], tot(5) - tot(3), atol=1e-5)
    # the hourly maxima are each hour's own
    assert np.all(recs[3]["U10MAX"] == 13.0) and np.all(recs[4]["UPHLMAX25"] == 20.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
