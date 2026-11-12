"""NAM nest (hiresf) product sets: the tables, the two-field U/V pairing, NOAA's nest grids and the fire-weather box.

    python tests/test_products_hiresf.py        (or pytest)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.products import sets as PS  # noqa: E402
from namgpu.products import spec  # noqa: E402


def test_every_nest_set_reads_and_pairs_noaas_66_messages():
    for name in PS.NEST_SETS:
        ps = PS.get(name)
        rows = spec.read(PS.table_path(ps))
        doc, left = spec.profile(rows, noaa_path=PS.noaa_path(ps), name=f"nam-{name}", mask_carrier="WOOF_VALID",
                                 pair_uv=True)
        n = len(rows)
        assert n == (876 if name == "conusnest" else 871), (name, n)
        assert sum(1 for f in doc["fields"] if f.get("pair_next")) == 66, name
        assert doc["mask_carrier"] == "WOOF_VALID" and doc["initial_window_stats"] == [0, 1] and "initial_window_unit" not in doc
        # every left-out row says why, and every written row has NOAA's precision
        assert all(r["note"] for r in left), name
        assert all("decimal_scale" in f for f in doc["fields"]), name


def test_awphys_profile_is_unchanged_by_the_nest_options():
    rows = spec.read()
    doc, _ = spec.profile(rows)
    assert "mask_carrier" not in doc and "initial_window_unit" not in doc
    assert not any(f.get("pair_next") for f in doc["fields"])


def test_nest_grids_round_trip_and_first_points():
    for name in ("conusnest", "alaskanest", "hawaiinest", "priconest"):
        g = PS.get(name).grid
        lat, lon = g.latlon_ij(np.array([0.0, g.nx - 1.0]), np.array([0.0, g.ny - 1.0]))
        assert abs(lat[0] - g.la1) < 1e-9 and abs(lon[0] - g.lo1) < 1e-9, name
        if g.kind == "mercator":                       # NOAA's last point (wgrib2 mercator:...:lon1 ...:lat1)
            assert abs(lat[1] - g.la2) < 2e-3 and abs(lon[1] - g.lo2) < 2e-3, (name, lat[1], lon[1])
        x, y = g.forward(lat, lon)
        la, lo = g.inverse(x, y)
        assert np.allclose(la, lat, atol=1e-9) and np.allclose(lo, lon, atol=1e-9), name


def test_nest_grids_agree_with_the_ncep_nest_rows():
    """namgpu/nest/ncep_nests.py records each NCEP nest's decoded hiresf grid too: one truth."""
    from namgpu.nest import ncep_nests
    for name, row in ncep_nests.NCEP_NESTS.items():
        gdt = (row.get("hiresf") or {}).get("gdt")
        if not gdt or name not in PS.SETS:
            continue
        g = PS.get(name).grid
        if isinstance(g, str):                         # placed per cycle (firewxnest): its grid comes with the cycle
            continue
        assert (g.nx, g.ny) == (gdt["nx"], gdt["ny"]), name
        assert abs(g.la1 - gdt["la1"]) < 1e-9 and abs(g.lo1 - gdt["lo1"]) < 1e-9, name
        assert abs(g.dx - gdt["dx_m"]) < 1e-9 and abs(g.dy - gdt["dy_m"]) < 1e-9, name
        assert abs(g.lov - gdt.get("lov", g.lov)) < 1e-9, name


def test_firewx_box_for_ncep_production_points():
    d = np.pi / 180.0
    for (clat, clon), (nx, ny) in (((42.0, -122.3), (646, 389)), ((39.0, -112.0), (543, 379))):
        lat = np.full((333, 333), np.nan)
        lon = np.full((333, 333), np.nan)
        lat[166, 166], lon[166, 166] = clat, clon
        lat[0, 0], lat[0, -1], lat[-1, 0], lat[-1, -1] = clat - 2, clat - 2, clat + 2, clat + 2
        lon[0, 0], lon[0, -1], lon[-1, 0], lon[-1, -1] = clon - 3, clon + 3, clon - 3, clon + 3
        g = PS.firewx_grid(lat, lon)
        assert (g.nx, g.ny) == (nx, ny), (clat, clon, g)
    assert d > 0


def test_firewx_runs_ncep_1h_bucket():
    """NCEP's fire-weather nest empties its accumulators every hour (nam_configfile_firewxnest_* nhrs_* 1): NOAA's
    firewx files carry 1-2, 2-3, ... windows. The set, its table and the nest row all say 1 h; the others keep 3 h."""
    fw = PS.get("firewxnest")
    assert fw.buckets_h == (1,) and fw.table == "hiresf-firewxnest.tsv"
    for name in ("conusnest", "alaskanest", "hawaiinest", "priconest"):
        assert PS.get(name).buckets_h == (3,), name
    shared = spec.read(PS.table_path(PS.get("alaskanest")))
    rows = spec.read(PS.table_path(fw))
    assert [(r["name"], r["level"], r["stat"]) for r in rows] == [(r["name"], r["level"], r["stat"]) for r in shared]
    assert not any("since:3h" in r["window"] or r["source"].endswith("_S3") for r in rows)
    assert sum(1 for r in rows if r["window"] == "since:1h") == sum(1 for r in shared if r["window"] == "since:3h") > 0
    from namgpu.nest import ncep_nests
    keys = ncep_nests.NCEP_NESTS["firewxnest"]["keys"]
    assert all(keys[k] == "1" for k in ("nhrs_prec", "nhrs_heat", "nhrs_clod", "nhrs_rdlw", "nhrs_rdsw", "nhrs_srfc"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("test_products_hiresf: PASS")
