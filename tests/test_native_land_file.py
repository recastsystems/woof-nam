"""A start from the NAM's own analysis takes the NAM's liquid soil water, base albedo and maximum snow albedo from
NCEP's native surface file of the same cycle (bgrdsf00), which the analysis bgrd3d00 does not carry.

Without it the start split soil water into ice and liquid with a formula of temperature, and the base albedo under
snow and the maximum snow albedo were vegetation-table values. Against NCEP's own bgrdsf00 of 2024-11-05 12Z the
split missed the NAM's liquid water by 0.028 (mean abs, all land; the NAM holds ice at 17,510 land points above
0 C, which no formula gives back), the base albedo under snow by 7.1 points and the maximum snow albedo by 11.7
points. The run then warmed frozen ground too fast (soil 0-10 cm +3 to +4 F, 2 m +1 to +2 F at 21Z, Arizona to
Manitoba) and its snow was too bright in places and too dark in others.

These tests check the pieces on the CPU: the fetch row and its record pattern (against lines of NCEP's real .idx),
the merge of the file's records over the fallbacks, where the files are looked for, and the static builder's
search beside a bgrd3d00. The whole path is proven by a 12 h native run against NCEP's archive."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.init import fetch_case  # noqa: E402
from namgpu.init import sources  # noqa: E402

# lines of NCEP's nam.t12z.bgrdsf00.tm00.grib2.idx of 2024-11-05 (noaa-nam-pds)
IDX = """18:10957098:d=2024110512:TSOIL:0-0.1 m below ground:anl:
19:11331420:d=2024110512:SOILW:0-0.1 m below ground:anl:
20:11624163:d=2024110512:SOILL:0-0.1 m below ground:anl:
23:12565875:d=2024110512:SOILL:0.1-0.4 m below ground:anl:
26:13502808:d=2024110512:SOILL:0.4-1 m below ground:anl:
29:14439929:d=2024110512:SOILL:1-2 m below ground:anl:
34:15405387:d=2024110512:WEASD:surface:anl:
117:43688940:d=2024110512:ALBDO:surface:anl:
118:43909798:d=2024110512:SNFALB:surface:anl:
119:44073220:d=2024110512:MXSALB:surface:anl:""".splitlines()


def test_fetch_row_brings_the_land_records_and_is_optional():
    rows = {r["name"]: r for r in fetch_case.FETCH_ROWS}
    r = rows["nam-bgrdsf00-land"]
    assert r["source"] == "native" and r["role"] == "init" and r.get("optional") is True
    assert r["kind"] == "subset" and r["leads"] == "f000" and r["stem"] == "bgrdsf"
    assert r["dest"] == "nam/bgrdsf00.land.grib2" and r["dest"] in sources.NATIVE_LAND_FILES[0]
    pat = fetch_case.PATTERNS[r["records"]]
    # the same test _subset applies to each idx line
    got = [l.split(":")[3] + ":" + l.split(":")[4] for l in IDX
           if pat.search(":" + ":".join(l.split(":")[3:]) + ":")]
    assert got == ["SOILL:0-0.1 m below ground", "SOILL:0.1-0.4 m below ground", "SOILL:0.4-1 m below ground",
                   "SOILL:1-2 m below ground", "SNFALB:surface", "MXSALB:surface"], got
    era = fetch_case.era_of(r, fetch_case._t("2024-11-05T12"))
    assert fetch_case._era_path(r, fetch_case._t("2024-11-05T12"), 0) == "nam.20241105/nam.t12z.bgrdsf00.tm00.grib2", era


def test_optional_row_is_not_a_required_request():
    src = (ROOT / "namgpu" / "init" / "fetch_case.py").read_text()
    assert 'and not row.get("optional")' in src


def test_land_records_replace_the_fallbacks():
    land = np.array([[1.0, 1.0, 0.0]], np.float32)
    albase = np.array([[0.18, 0.0, 0.0]], np.float32)       # snow-free land, snowy land (missing), water
    rec = {"SNFALB": np.array([[18.0, 21.0, 6.0]], np.float32),
           "MXSALB": np.array([[55.0, 71.0, 6.0]], np.float32)}
    rec.update({f"SOILL{k}": np.full((1, 3), 0.1 * k, np.float32) for k in range(1, 5)})
    extra, sh2o, lacking = sources.land_from_surface_file(rec, albase, land)
    assert lacking == []
    np.testing.assert_allclose(extra["ALBASE"], [[0.18, 0.21, 0.0]], atol=1e-6)    # water keeps the fallback
    np.testing.assert_allclose(extra["MXSNAL"], [[0.55, 0.71, 0.0]], atol=1e-6)    # water: 0, the table fill
    assert extra["ALBASE"].dtype == np.float32 and extra["MXSNAL"].dtype == np.float32
    assert sh2o.shape == (4, 1, 3) and sh2o.dtype == np.float32
    np.testing.assert_allclose(sh2o[:, 0, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_without_the_file_the_start_keeps_its_fallbacks_and_names_them():
    land = np.ones((1, 2), np.float32)
    albase = np.array([[0.2, 0.0]], np.float32)
    extra, sh2o, lacking = sources.land_from_surface_file({}, albase, land)
    assert sh2o is None and "MXSNAL" not in extra and extra["ALBASE"] is albase
    assert lacking == ["base albedo under snow", "maximum snow albedo", "liquid soil water"]
    # an invalid percentage is left to the fallback
    extra, _, _ = sources.land_from_surface_file({"SNFALB": np.array([[0.0, 150.0]], np.float32)}, albase, land)
    np.testing.assert_allclose(extra["ALBASE"], albase)


def test_the_start_looks_for_the_subset_then_the_whole_file(tmp_path):
    case = tmp_path / "cases" / "c"
    (case / "raw" / "nam").mkdir(parents=True)
    assert sources.native_land_file(case) is None
    whole = case / "raw" / "nam" / "bgrdsf00.grib2"
    whole.write_bytes(b"GRIB")
    assert sources.native_land_file(case) == whole
    sub = case / "raw" / "nam" / "bgrdsf00.land.grib2"
    sub.write_bytes(b"GRIB")
    assert sources.native_land_file(case) == sub
    src = (ROOT / "namgpu" / "init" / "sources.py").read_text()
    assert 'ctx.tables["sfc_override"]' in src and "decode_native_land(ctx, lpath" in src


def test_static_build_finds_the_surface_file_beside_the_analysis(tmp_path):
    from namgpu.init import gfs_start
    b = tmp_path / "nam.2024110512.bgrd3d00.grib2"
    b.write_bytes(b"GRIB")
    assert gfs_start.land_file_beside(b) is None
    s = tmp_path / "nam.2024110512.bgrdsf00.land.grib2"
    s.write_bytes(b"GRIB")
    assert gfs_start.land_file_beside(b) == s
    case = tmp_path / "raw" / "nam"
    case.mkdir(parents=True)
    (case / "bgrd3d00.grib2").write_bytes(b"GRIB")
    (case / "bgrdsf00.land.grib2").write_bytes(b"GRIB")
    assert gfs_start.land_file_beside(case / "bgrd3d00.grib2") == case / "bgrdsf00.land.grib2"


if __name__ == "__main__":
    import tempfile
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            n += 1
    print(f"{Path(__file__).name}: {n} passed")
