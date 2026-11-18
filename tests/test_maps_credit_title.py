"""A map drawn by gpuwm's own renderer (a plain pip install) carries the data credit on its title line, whole; a
renderer with the note patch keeps it in the subtitle note as before (maps.note_drawn, maps.credited_title).

Breakage it prevents: maps from a CC BY start (ECMWF IFS, ERA5) or CC BY statics (SoilGrids) drawn by a pip install
carried no credit, because only the patched renderer draws a CF row's "note" (hour-201 proof maps, 2026-11-12; every
box from IFS). CC BY 4.0 asks for the credit wherever the picture goes.
CPU only, numpy only (the history read and the CF write are stubbed). Run: python tests/test_maps_credit_title.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from namgpu.io import maps  # noqa: E402

START = datetime(2026, 11, 12, 6)
LONG = "ECMWF open data CC BY 4.0, SoilGrids CC BY 4.0, ERA5 C3S"


def _read(path, need, upto_pa=None):
    h = int(Path(path).name.split("_f")[1])
    ny, nx = 4, 5
    lat, lon = np.meshgrid(np.linspace(50, 56, ny), np.linspace(-8, 0, nx), indexing="ij")
    f = {"XLAT": lat, "XLONG": lon, "T2": np.full((ny, nx), 280.0), "PSFC": np.full((ny, nx), 1.0e5),
         "Q2": np.full((ny, nx), 0.005), "COMPOSITE": np.full((ny, nx), 20.0)}
    return START + timedelta(hours=h), START, {k: v for k, v in f.items() if k in need | {"XLAT", "XLONG"}}


def table_of(job):
    r = maps._prep_frame(job)
    return json.loads(Path(r[5]).read_text())


def main() -> int:
    maps._read, maps._write_cf = _read, lambda *a, **k: None
    maps._follow_of = lambda path: None
    work = tempfile.mkdtemp()
    need = sorted({"XLAT", "XLONG", "T2", "PSFC", "Q2", "COMPOSITE"})
    prods = ["t2m", "composite_reflectivity"]
    job = ("h_f144", {}, need, prods, work, "WOOF-NAM 12 km")
    os.environ.pop(maps.RENDERER_ENV, None)
    os.environ.pop("NAMGPU_START_LEAD_H", None)
    pinned = maps.PINNED_RENDERER
    maps.PINNED_RENDERER = Path(work) / "no-pinned-build" / "rw_wrfbatch"   # a pip install: no tools/renderer/bin

    # no credit: titles as before, no note
    os.environ.pop(maps.CREDIT_ENV, None)
    rows = table_of(job)
    want = [maps.panel_title(maps.TABLE[p][1]["title"], START, START + timedelta(hours=144)) for p in prods]
    assert [r["title"] for r in rows] == want, rows
    assert all("note" not in r for r in rows), rows

    # a pip install's renderer: the credit whole on the title line, within the line's length, no note
    os.environ[maps.CREDIT_ENV] = LONG
    rows = table_of(job)
    for r in rows:
        assert r["title"].endswith(" · " + LONG) and "data:" not in r["title"], r
        assert len("WOOF-NAM 12 km | " + r["title"]) <= maps.TITLE_LINE_MAX, r
        assert "note" not in r, r
    assert rows[0]["title"].startswith("2 m temperature"), rows[0]
    # the run's own words gave way, the field and the lead stayed
    assert "(+144h)" in rows[1]["title"] and rows[1]["title"].startswith("Composite reflectivity"), rows[1]

    # a short credit keeps the whole title, valid time included
    os.environ[maps.CREDIT_ENV] = "ECMWF open data CC BY 4.0"
    t = table_of(("h_f003", {}, need, ["t2m"], work, "WOOF-NAM 3 km"))[0]["title"]
    whole = maps.panel_title("2 m temperature", START, START + timedelta(hours=3))
    assert t == whole + " · ECMWF open data CC BY 4.0", t

    # a renderer with the note patch (NAMGPU_RENDERER): the subtitle note, the title unchanged
    os.environ[maps.CREDIT_ENV] = LONG
    os.environ[maps.RENDERER_ENV] = "/opt/patched/rw_wrfbatch"
    rows = table_of(job)
    assert all(r["note"] == "data: " + LONG and "data:" not in r["title"] for r in rows), rows
    os.environ.pop(maps.RENDERER_ENV)
    maps.PINNED_RENDERER = pinned
    os.environ.pop(maps.CREDIT_ENV)
    print("test_maps_credit_title: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
