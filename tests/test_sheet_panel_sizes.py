"""A sheet's panels of two sizes are padded to one before the renderer composes them (io_referee_sheets.same_size).
Defect: gpuwm's own renderer (a pip install's) refuses panels of two sizes, and a WOOF-NAM panel whose 10 m wind
passes 60 kt carries the longer wind ladder, so the 1.5 km nest in a live 12 km parent on the South Pacific cyclone
(GFS 2026-11-17 00Z, 1 h, sheets on) ended rc 5 at "sheet f001_wind10.png: panel 1 is 1154x891 and panel 0 is
1163x891" after its maps were drawn. CPU only: the renderer is a stub that reads the request."""
import json
import subprocess
from pathlib import Path

from PIL import Image

from namgpu.io import io_referee_sheets as rs


def _png(path, size, colour=(248, 248, 246)):
    im = Image.new("RGB", size, colour)
    im.putpixel((10, 10), (200, 0, 0))
    im.save(path)
    return path


def test_two_sizes_are_padded_to_the_larger(tmp_path):
    a = _png(tmp_path / "model.png", (1163, 891))
    b = _png(tmp_path / "gfs.png", (1154, 891))
    out = rs.same_size([[a, b]], tmp_path)
    assert out[0][0] == a and out[0][1] != b
    with Image.open(out[0][1]) as im:
        assert im.size == (1163, 891)
        assert im.getpixel((10, 10))[:3] == (200, 0, 0)          # the panel itself is unchanged
        assert im.getpixel((1160, 500))[:3] == (248, 248, 246)   # the pad is the panel's own corner colour
    with Image.open(b) as im:
        assert im.size == (1154, 891)                              # the drawn map is left as it is


def test_one_size_is_passed_as_it_is(tmp_path):
    a = _png(tmp_path / "a.png", (1154, 891))
    b = _png(tmp_path / "b.png", (1154, 891))
    assert rs.same_size([[a, b]], tmp_path) == [[a, b]]


def test_compose_hands_the_renderer_one_size(tmp_path, monkeypatch):
    a = _png(tmp_path / "model.png", (1163, 891))
    b = _png(tmp_path / "gfs.png", (1154, 891))
    seen = {}

    def run(cmd, **kw):
        req = json.loads(Path(cmd[-1]).read_text())
        seen["sizes"] = [Image.open(p).size for p in req["rows"][0]]
        Path(req["output"]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    rs._compose(tmp_path / "rw_wrfbatch", [[a, b]], tmp_path / "sheets" / "f001_wind10.png", "t", "2026-11-17 01:00")
    assert seen["sizes"] == [(1163, 891), (1163, 891)]
