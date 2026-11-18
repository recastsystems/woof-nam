"""GFS cloud water under both of NCEP's index names.

NCEP's .idx names cloud water (GRIB2 0-1-22) CLMR before 2021-02-26 00Z and from 2023-02-03 00Z, and CLWMR in between
(read from the noaa-gfs-bdp-pds f000 .idx files on 2026-11-13). The fetch selects records by those names, so with CLMR
alone every start and boundary file of the two CLWMR years arrived with no cloud water, and the decode, which finds
records by code, left the slot at 0. These tests hold that the start and boundary patterns take cloud water from an
index of either era, at every isobaric level they take the other hydrometeors from, and that this is the only
difference between the two eras' record sets. The index fixtures are NCEP's own files (tests/fixtures/gfs_idx).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from namgpu.init import fetch_case, source_records

FIX = Path(__file__).parent / "fixtures" / "gfs_idx"
ERAS = {"2021040100": "CLWMR", "2026111212": "CLMR"}


def selected(name: str, pattern) -> list[str]:
    out = []
    for line in gzip.decompress((FIX / name).read_bytes()).decode().splitlines():
        f = line.split(":")
        if pattern.search(":" + ":".join(f[3:]) + ":"):
            out.append(":".join(f[3:5]))
    return out


def levels(recs, var):
    return {r.split(":", 1)[1] for r in recs if r.split(":", 1)[0] == var and r.endswith(" mb")}


@pytest.mark.parametrize("run", sorted(ERAS))
@pytest.mark.parametrize("lead", ["f000", "f201"])
@pytest.mark.parametrize("which", ["GFS_INIT_VARS", "GFS_BOCO_VARS"])
def test_cloud_water_taken_at_every_hydrometeor_level(run, lead, which):
    recs = selected(f"gfs.{run}.{lead}.idx.gz", getattr(fetch_case, which))
    cw = levels(recs, ERAS[run])
    assert len(cw) == 22, (run, lead, which, sorted(cw))
    assert cw == levels(recs, "RWMR") == levels(recs, "ICMR") == levels(recs, "SNMR")


@pytest.mark.parametrize("lead", ["f000", "f201"])
def test_the_two_eras_differ_only_in_the_name(lead):
    old = selected(f"gfs.2021040100.{lead}.idx.gz", fetch_case.GFS_INIT_VARS)
    new = selected(f"gfs.2026111212.{lead}.idx.gz", fetch_case.GFS_INIT_VARS)
    assert sorted(r.replace("CLWMR:", "CLMR:") for r in old) == sorted(new)
    assert len(old) == len(new) == 419


def test_both_names_are_the_record_the_decode_reads():
    """The decode finds cloud water by code, so a CLWMR record lands in the same slot as a CLMR one."""
    assert source_records.GFS["pl"]["CLMR"] == (0, 1, 22)
    for name in ("CLMR", "CLWMR"):
        assert fetch_case.GFS_INIT_VARS.search(f":{name}:850 mb:")
        assert fetch_case.GFS_BOCO_VARS.search(f":{name}:850 mb:")
    assert not fetch_case.GFS_BOCO_VARS.search(":CLWMR:1 hybrid level:")
