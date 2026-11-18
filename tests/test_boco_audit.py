"""A boundary folder is checked before it is reused, and a boundary time is checked before it is written: winds,
temperature and PD outside what the atmosphere can hold stop the build, and a folder already holding such values is
rebuilt instead of read.

Breakage it prevents: case 2026-11-12_15z on the 12 km Kansas box (woof-nam nest --center 38.5,-97.5 --size-km 1500
--res-km 12). Its boco.0000 and boco.0003, written at 22:31Z on 12 Nov by a HRRR 15Z start, held V from -1.07e6 to
+1.9e5 m/s on every strip (finite; U, T, Q and PD bit-identical to a good build of the same hour). The folder's
BOCO-SOURCE.json matched every later run of that case, so the GFS forecast-hour-3 start and the HRRR 15Z start both
read those files and went to NaN in T at step 2. The same two starts with the boundary files
built afresh ran (rc 0, 2026-11-13). CPU only: the builder is a stub."""
import json
import struct
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import config as configmod  # noqa: E402
from namgpu.domains import bc_hours  # noqa: E402
from namgpu.init import gfs_bocos, sources  # noqa: E402

IM, JM, LM, LNSH, LNSV = 12, 10, 4, 2, 2
START = datetime(2026, 11, 12, 15)
FIELDS = ("PD", "T", "Q", "CW", "U", "V")          # the record order of a boco file (write_boco)


def _strip_sizes(name):
    w = LNSV if name in ("U", "V") else LNSH
    nl = 1 if name == "PD" else LM
    return (nl * w * IM, nl * w * IM, nl * JM * w, nl * JM * w)


def write_np(path, start_vals, tend=None, tboco=10800.0):
    """write_boco's layout from numpy: per field, strips S, N, W, E, each [slot 1 = start_vals, slot 2 = tend]."""
    tend = tend or {}
    with open(path, "wb") as f:
        def rec(b):
            f.write(struct.pack("<i", len(b)))
            f.write(b)
            f.write(struct.pack("<i", len(b)))
        rec(struct.pack("<i3iif", 1, START.day, START.month, START.year, START.hour, np.float32(tboco)))
        for name in FIELDS:
            parts = []
            for k in _strip_sizes(name):
                parts.append(np.full(k, start_vals[name], "<f4").tobytes())
                parts.append(np.full(k, tend.get(name, 0.0), "<f4").tobytes())
            rec(b"".join(parts))


GOOD = {"PD": 9.0e4, "T": 270.0, "Q": 0.004, "CW": 0.0, "U": 20.0, "V": -12.0}


def _folder(tmp_path, monkeypatch, vals):
    case = tmp_path / "cases" / "2026-11-12_15z"
    out = case / "boco-test"
    out.mkdir(parents=True)
    monkeypatch.setattr(sources, "source_row", lambda ctx: ("gfs", sources.SOURCES["gfs"]))
    monkeypatch.setattr(sources, "boco_age_h", lambda row, cfg=None, override=None, start=None, lead=None: 3)
    monkeypatch.setattr(configmod, "boco_dirname", lambda cfg: "boco-test")
    monkeypatch.setattr(configmod, "grid_stamp", lambda cfg: "grid-1")
    monkeypatch.setattr(gfs_bocos, "boco_dir", lambda ctx: out)
    hours = bc_hours(3, 3)
    for fh in range(0, hours, 3):
        write_np(out / f"boco.{fh:04d}", vals)
    (out / "BOCO-SOURCE.json").write_text(json.dumps(
        {"source": "gfs", "start": f"{START:%Y-%m-%dT%H}", "hours": 6, "age_h": 3, "interval_h": 3,
         "files": [], "grid": "grid-1"}))
    calls = []
    monkeypatch.setattr(gfs_bocos, "build", lambda ctx, hours, out_dir, **kw: calls.append(hours) or out_dir)
    ctx = SimpleNamespace(cfg=SimpleNamespace(raw={}, lnsh=LNSH, lnsv=LNSV), grid=SimpleNamespace(im=IM, jm=JM, lm=LM),
                          case_dir=case, start=START, init_source="gfs")
    return ctx, out, calls


def test_a_folder_holding_impossible_v_is_rebuilt_not_reused(tmp_path, monkeypatch, capsys):
    ctx, out, calls = _folder(tmp_path, monkeypatch, dict(GOOD, V=-1.075e6))
    gfs_bocos.run(ctx, 3)
    assert calls, "a boundary folder holding V of -1.07e6 m/s was reused"
    said = capsys.readouterr().out
    assert "boco.0000 holds V from -1.075e+06" in said and "rebuilding" in said, said


def test_a_good_folder_is_reused(tmp_path, monkeypatch):
    ctx, out, calls = _folder(tmp_path, monkeypatch, GOOD)
    gfs_bocos.run(ctx, 3)
    assert calls == []
    assert ctx.boco_dir == str(out)


def test_the_end_of_the_interval_is_checked_too(tmp_path, monkeypatch):
    """slot 2 is the tendency: a start value inside the bounds that runs to 1e5 m/s at the interval's end is caught."""
    ctx, out, calls = _folder(tmp_path, monkeypatch, GOOD)
    write_np(out / "boco.0003", GOOD, tend={"V": 10.0})          # -12 + 10800 x 10 m/s at the end
    gfs_bocos.run(ctx, 3)
    assert calls


def test_read_boco_is_write_boco_inverted(tmp_path):
    p = tmp_path / "boco.0000"
    write_np(p, GOOD, tend={"T": 1e-4})
    got = gfs_bocos.read_boco(p, IM, JM, LM, LNSH, LNSV)
    assert set(got) == set(FIELDS)
    assert np.all(got["V"][0] == np.float32(-12.0)) and got["V"][0].size == sum(_strip_sizes("V"))
    assert np.allclose(got["T"][1], 270.0 + 10800.0 * 1e-4)
    assert gfs_bocos.audit([p], IM, JM, LM, LNSH, LNSV) == ""
    assert "cannot be read" in gfs_bocos.audit([p], IM + 1, JM, LM, LNSH, LNSV)


def test_the_builder_refuses_impossible_strips_before_writing():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    good = {n: tuple(cp.full((3, 4), v, cp.float32) for _ in range(4)) for n, v in GOOD.items()}
    gfs_bocos.check_strips(good, "+0 h")
    bad = dict(good, V=tuple(cp.full((3, 4), -1.075e6, cp.float32) for _ in range(4)))
    with pytest.raises(RuntimeError, match=r"impossible boundary values at \+3 h: {'V': '48 values from -1.075e"):
        gfs_bocos.check_strips(bad, "+3 h")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
