"""The maps' MSLET is not computed in a process that has no CUDA context of its own (namgpu/io/maps.py _MsletStage).

Breakage this prevents: the one-card door gives its card back and re-executes itself as the maps tail, a process
with no CUDA context. render() then computed the MSLP map's MSLET in that process, on the device: the door's PID was
back on the card 3 s after its release, with 1,480 MiB while the maps were prepared (the GRIB2 follower still needed
the card) and 500 MiB through the rest of the maps, the column store and the wait (36 h arctic front-door run of the
0.2.1 candidate a3f8346, 2026-11-11: 324 of 325 nvidia-smi samples after `card_released`).

CPU only. render() is driven with two small synthetic history frames, a stand-in for the renderer and for the frame
worker; the test reads which process called _mslet."""
from __future__ import annotations

import multiprocessing
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from namgpu.io import maps  # noqa: E402
from namgpu.products import memslp  # noqa: E402

P0, T0, LAPSE, RDG = 101325.0, 288.15, 0.0065, 287.04 / 9.80665
LM, NY, NX = 30, 16, 18


def _history(path: Path, valid: str, start: str):
    """A history frame with the five MSLET inputs: a 6.5 K/km atmosphere over a hill, bottom level first."""
    j, i = np.mgrid[0:NY, 0:NX]
    zs = 900.0 * np.exp(-(((j - NY / 2) / 3.0) ** 2 + ((i - NX / 2) / 4.0) ** 2))
    psfc = P0 * (1.0 - LAPSE * zs / T0) ** (1.0 / (LAPSE * RDG))
    sig = np.linspace(0.0, 1.0, LM + 1)[:, None, None] ** 1.5
    pint = 200.0 + sig * (psfc[None] - 200.0)                       # top first
    pmid = 0.5 * (pint[:-1] + pint[1:])
    z = T0 / LAPSE * (1.0 - (pmid / P0) ** (LAPSE * RDG))
    t = np.maximum(T0 - LAPSE * z, 216.65)
    theta = t * (memslp.P00 / pmid) ** memslp.CAPPA_NMMB
    with netCDF4.Dataset(path, "w") as nc:
        for n, s in (("Time", 1), ("DateStrLen", 19), ("bottom_top", LM), ("bottom_top_stag", LM + 1),
                     ("south_north", NY), ("west_east", NX)):
            nc.createDimension(n, s)
        nc.START_DATE = start
        nc.DY = 12009.6
        nc.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.array(list(valid), dtype="S1")
        d2, d3, d3s = ("Time", "south_north", "west_east"), ("Time", "bottom_top", "south_north", "west_east"), \
            ("Time", "bottom_top_stag", "south_north", "west_east")
        nc.createVariable("XLAT", "f4", d2)[0] = 38.0 + 0.1 * j
        nc.createVariable("XLONG", "f4", d2)[0] = -100.0 + 0.1 * i
        nc.createVariable("HGT", "f4", d2)[0] = zs
        nc.createVariable("T", "f4", d3)[0] = (theta - 300.0)[::-1]
        nc.createVariable("PRES", "f4", d3)[0] = pmid[::-1]
        nc.createVariable("QVAPOR", "f4", d3)[0] = np.full_like(pmid, 1.0e-4)
        nc.createVariable("PINT", "f4", d3s)[0] = pint[::-1]


def _render(tmp_path, monkeypatch):
    """render() on two frames with one frame worker in this process. Returns (PIDs that called _mslet in THIS
    process, whether each frame's MSLET file was there when its worker ran)."""
    files = []
    for h in (12, 13):
        f = tmp_path / f"wrfout_d01_2025-03-14_{h}:00:00"
        _history(f, f"2025-03-14_{h}:00:00", "2025-03-14_12:00:00")
        files.append(f)
    here, ready = [], []
    real = maps._mslet

    def mslet(path, work, save=False):
        here.append(os.getpid())
        return real(path, work, save)

    def prep(job):
        path, work = job[0], job[4]
        ready.append(maps._mslet_file(path, work).is_file())
        valid, start = maps._times(Path(path))
        return None, valid, start, [], [], None, (-100.0, -98.3, 38.0, 39.5)

    monkeypatch.setattr(maps, "_mslet", mslet)
    monkeypatch.setattr(maps, "_prep_frame", prep)
    monkeypatch.setattr(maps, "renderer", lambda: Path("renderer-not-run"))
    monkeypatch.setattr(maps, "renderer_env", lambda rw: {})
    monkeypatch.setattr(maps._cores, "usable", lambda: 2)           # one frame worker: the in-process loop
    res = maps.render([*map(str, files), "--out", str(tmp_path / "maps"), "--products", "mslp"])
    assert res["failures"] == [] and res["written"] == []
    return here, ready


def test_a_process_without_cupy_does_not_compute_mslet_itself(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)         # the door's maps tail: CuPy never imported
    here, ready = _render(tmp_path, monkeypatch)
    assert ready == [True, True], "each frame's MSLET file must be there before its worker reads the frame"
    assert here == [], (f"render() computed MSLET {len(here)} times in the calling process (PID {os.getpid()}); a "
                        "process with no CUDA context must leave it to the helper process")
    assert "cupy" not in sys.modules, "render() loaded CuPy into a process that had none"
    assert multiprocessing.active_children() == [], "the MSLET helper process must have ended when render() returns"


def test_a_process_with_cupy_computes_mslet_in_place(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))   # stands for a forecast process' CuPy
    here, ready = _render(tmp_path, monkeypatch)
    assert ready == [True, True]
    assert here == [os.getpid()] * 2, "a process that has CuPy loaded keeps computing MSLET in its own context"
