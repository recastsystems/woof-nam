"""The line `woof-nam static` prints names where the albedos came from.

Breakage it prevents (0.2.1 candidate, 2026-11-11): a static build that took the snow-free and the maximum snow albedo
from NCEP's bgrdsf00 file printed "no NCEP bgrdsf00 file given or beside it", because the list of states left to the
fallbacks always names liquid soil water, which the static file never holds. CPU only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.init import gfs_start as gs  # noqa: E402


def test_both_albedos_from_the_file():
    line = gs.static_line("s.npz", Path("nam.t12z.bgrdsf00.tm00"), ["liquid soil water"])
    assert line == "wrote s.npz (albedos from nam.t12z.bgrdsf00.tm00)", line


def test_no_file():
    line = gs.static_line("s.npz", None, ["base albedo under snow", "maximum snow albedo", "liquid soil water"])
    assert "no NCEP bgrdsf00 file given or beside it" in line and "liquid soil water" not in line, line
    assert "base albedo under snow, maximum snow albedo: the vegetation table" in line, line


def test_file_without_one_record():
    line = gs.static_line("s.npz", Path("x.bgrdsf00"), ["maximum snow albedo", "liquid soil water"])
    assert "maximum snow albedo: the vegetation table (x.bgrdsf00 has no such record)" in line, line
