"""The line a GFS start prints about the case's NCEP native surface file says what happened.

Breakage it prevents (0.2.1 candidate a3f8346, warm case 2024-05-06 12Z from the GFS, 2026-11-11): the case held
raw/nam/bgrdsf00.land.grib2 and the start took both albedos from it, yet the run log read "no NCEP native surface
file in the case: liquid soil water from the vegetation table, not the NAM's own; `woof-nam fetch` on the case
brings it". A GFS start never takes liquid soil water from that file (its soil is the GFS's own), so that name is not
a lack there, and the fetch had already brought the file. CPU only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.init import gfs_start as gs  # noqa: E402


def test_both_albedos_from_the_file():
    line = gs.case_land_line(Path("raw/nam/bgrdsf00.land.grib2"), ["liquid soil water"])
    assert line == "init-gfs-start: albedos from the case's bgrdsf00.land.grib2", line


def test_no_file_in_the_case():
    line = gs.case_land_line(None, ["base albedo under snow", "maximum snow albedo", "liquid soil water"])
    assert "no NCEP native surface file in the case" in line and "liquid soil water" not in line, line
    assert "base albedo under snow, maximum snow albedo from the vegetation table" in line, line
    assert "`woof-nam fetch` on the case brings it" in line, line


def test_file_without_one_record():
    line = gs.case_land_line(Path("raw/nam/bgrdsf00.land.grib2"), ["maximum snow albedo", "liquid soil water"])
    assert "no NCEP native surface file" not in line and "fetch" not in line, line
    assert "bgrdsf00.land.grib2 has no maximum snow albedo record: from the vegetation table" in line, line
