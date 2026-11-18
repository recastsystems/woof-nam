"""CPU checks: every year a start can come from has the radiation's fix data (no GPU, no network).

Run: python tests/test_old_years_radiation.py   (or pytest). Breakage it prevents: a 12 km run or a box from ERA5 in
1990 (accepted by the site, which offers ERA5 from 1940) stopped at its first radiation call, after the fetch and the
start were built: aer_update found no volcanic_aerosols_1990-1999.txt in the package, and gas_update, one step on,
would have found no co2historicaldata_1990.txt (the package shipped CO2 from 2004 and no volcanic table).
  - aer_update's volcanic table for every decade from 1850 to 1999 (volc_update reads one for a year in that range);
  - gas_update's CO2 source for every year from 1940 to today: a monthly file of the year or an earlier one (the
    Fortran's search), else, before 1956, the historical global means (co2_from_global_means);
  - the choice keeps the Fortran's wherever the Fortran runs.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import paths  # noqa: E402
from namgpu.config import NamConfig  # noqa: E402

DEF = {k: f.default for k, f in NamConfig.__dataclass_fields__.items() if k in ("co2dat_file", "co2gbl_file", "ictmflg")}
from namgpu.physics.rrtm import phys_rrtm_aer_update as aer  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_gas_update as gas  # noqa: E402

FIRST_START_YEAR = 1940   # ERA5's first year (woof-nam sources era5 valid_from, the site's NAM_SOURCES era5.first)


def _co2_source(year: int, ictmflg: int, fix: Path) -> str:
    """The file gas_update reads for `year` (its own logic: the global means, the year's file, or the latest earlier)."""
    tmpl = DEF["co2dat_file"]
    exists = lambda name: (fix / name).is_file()  # noqa: E731
    if gas.co2_from_global_means(year, ictmflg, tmpl, exists):
        return DEF["co2gbl_file"]
    if exists(gas.co2_file_name(tmpl, year)):
        return gas.co2_file_name(tmpl, year)
    y = year
    while y >= gas.MINYEAR:                    # the Fortran's search for the latest earlier file
        y -= 1
        if exists(gas.co2_file_name(tmpl, y)):
            return gas.co2_file_name(tmpl, y)
    raise AssertionError(f"gas_update finds no CO2 data for {year} (Fortran: Can not find co2 data source file)")


def test_volcanic_tables_shipped():
    missing = [f"volcanic_aerosols_{d}-{d + 9}.txt" for d in range(aer.MINVYR, aer.MAXVYR + 1, 10)
               if not (paths.FIX / f"volcanic_aerosols_{d}-{d + 9}.txt").is_file()]
    assert not missing, missing
    # the table volc_update parses: a title record, then 10 years x 4 bands of 12 i5 values
    lines = (paths.FIX / "volcanic_aerosols_1990-1999.txt").read_text().splitlines()
    assert lines[0].startswith("1990-1999") and len(lines) >= 41
    assert all(len(l.rstrip()) >= 60 and all(l[5 * i:5 * i + 5].strip().lstrip("-").isdigit() for i in range(12))
               for l in lines[1:41])


def test_co2_every_start_year():
    now = datetime.now(timezone.utc).year
    got = {y: _co2_source(y, DEF["ictmflg"], paths.FIX) for y in range(FIRST_START_YEAR, now + 1)}
    assert got[1990] == "co2historicaldata_1990.txt"
    assert got[1956] == "co2historicaldata_1956.txt"
    assert got[1945] == DEF["co2gbl_file"] and got[1940] == DEF["co2gbl_file"]
    assert (paths.FIX / DEF["co2gbl_file"]).read_text().splitlines()[0].startswith("1765 TO")


def test_global_means_only_where_the_fortran_stops():
    tmpl = DEF["co2dat_file"]
    have = {gas.co2_file_name(tmpl, y) for y in range(1956, 2027)}
    exists = have.__contains__
    # ictmflg > 0: the Fortran's own rule, before MINYEAR only
    assert gas.co2_from_global_means(1950, 1, tmpl, exists)
    assert not gas.co2_from_global_means(1957, 1, tmpl, exists)
    # ictmflg 0 (the NAM's): a year with its own 2-D file or MINYEAR on keeps the monthly file, as the Fortran does
    assert not gas.co2_from_global_means(1956, 0, tmpl, exists)
    assert not gas.co2_from_global_means(1990, 0, tmpl, exists)
    assert not gas.co2_from_global_means(1990, 0, tmpl, lambda n: False)   # the Fortran's search and refusal hold
    # ...and a year before the first 2-D file, where the Fortran stops, reads the global means
    assert gas.co2_from_global_means(1955, 0, tmpl, exists)
    assert gas.co2_from_global_means(1940, 0, tmpl, exists)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
