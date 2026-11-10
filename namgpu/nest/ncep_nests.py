"""NCEP's own NAM nests as table rows: the native grid (NMMB keys on the NAM 12 km parent's rotated grid), the run
length, and the hiresf GRIB2 shape NOAA published for each. A suite nest is a row here, never a code path: the door
(`woof-nam nest --ncep-nest NAME`) turns a row into its configfile through domain.config_keys, exactly as a
nest-anywhere grid, and the products writer reads the row's `hiresf` block.

Sources (every number below is copied from them, not derived by hand):
  NCEP NAM v4.2.11 (nam.v4.2.11, commit 83df4e13, 2025-02-25):
    parm/nam_configfile           the 12 km parent: 954 x 835 x 60, tph0d 54, tlm0d -106, wbd -60.039, sbd -45.036,
                                  dlmd 0.126, dphd 0.108 (the same keys as namgpu/parm/nam_configfile)
    parm/nam_configfile_<nest>    im, jm, i_parent_start, j_parent_start, parent_child_space_ratio, dt, filt_dt,
                                  nrads/nradl, lnsh/lnsv, spec_adv, the nest physics (domain.NEST_PHYSICS)
    parm/nam_namelist.nps_<nest>  dom_cen_lat/lon (the geographic centre NPS places the grid at; checked by
                                  tests/test_ncep_nests.py against the grid this row builds)
    scripts/exnam_fcst.sh         LENGTH=60 for every nest of the tm00 cycle
    ush/nam_prdgen_nest.sh        the wgrib2 -new_grid spec of each hiresf file
  NOAA's files of the last NAM cycle, 2026-11-03 06Z (s3://noaa-nam-pds/nam.20261103/nam.t06z.<nest>.hiresfFF.tm00
    .grib2), decoded: the grid template values, packing, bitmap, product templates and the record counts below.

Grid (NMMB module_NESTING SET_NEST_GRIDS, as domain.nest_grid_associated lays out a nest): the parent's pole,
dlmd/dphd = the parent's / ratio, the SW H point on parent H point (i_parent_start, j_parent_start) (1-based), so
wbd = parent wbd + (i_parent_start - 1) parent dlmd and sbd likewise.
"""
from __future__ import annotations

import math

# parm/nam_configfile (NAM 12 km parent; namgpu/parm/nam_configfile carries the same keys)
NAM_PARENT = {"im": 954, "jm": 835, "lm": 60, "tph0d": 54.0, "tlm0d": -106.0, "wbd": -60.039, "sbd": -45.036,
              "dlmd": 0.126, "dphd": 0.108, "res_km": 12}
A_EARTH = 6376000.0          # grid.A_EARTH (NMMB module_CONSTANTS a), as domain.A_EARTH

# One row per NCEP nest. `hiresf`: the product file NOAA published (name, grid, packing, records); `wgrib2` is
# nam_prdgen_nest.sh's -new_grid argument verbatim, `gdt` the decoded grid definition of NOAA's own file.
NCEP_NESTS = {
    "conusnest": {
        # im - 1 = 1826 and jm - 1 = 1466 are not whole multiples of the ratio 4: NCEP's CONUS nest ends half a parent
        # cell past a parent row/column on the E and N sides (NMMB does not need whole cells; the boundary factors are
        # per point, PARENT_TO_CHILD_INTERP_FACTORS)
        "im": 1827, "jm": 1467, "i_parent_start": 295, "j_parent_start": 95, "ratio": 4, "res_km": 3,
        "hours": 60, "spec_adv": True, "nps_center": (38.61676, -98.87040),
        "src": "nam.v4.2.11 parm/nam_configfile_conusnest (im 1827, jm 1467, i/j_parent_start 295/95, ratio 4, "
               "dt 6+1/4 s, filt_dt 2+0/4, nrads/nradl 192, lnsh/lnsv 5, spec_adv .true., convection none, "
               "nhrs_* 3); parm/nam_namelist.nps_conusnest e_we 1827 e_sn 1467 dx .0315 dy .027, dom_cen 38.61676, "
               "-98.87040; scripts/exnam_fcst.sh LENGTH=60",
        "hiresf": {
            "file": "woof-nam.tHHz.conusnest.hiresfFF.tm00.grib2",
            "noaa_file": "nam.tHHz.conusnest.hiresfFF.tm00.grib2",
            "leads": (0, 60, 1),
            "wgrib2": "lambert:262.5:38.5:38.5 237.280:1799:3000 21.138:1059:3000",
            "gdt": {"template": 30, "earth_shape": 6, "nx": 1799, "ny": 1059, "la1": 21.138, "lo1": 237.28,
                    "lad": 38.5, "lov": 262.5, "latin1": 38.5, "latin2": 38.5, "dx_m": 3000.0, "dy_m": 3000.0,
                    "projection_centre": 0, "resolution_flags": 0x38, "scan_mode": 0x40},
            "packing": {"drt": 3, "name": "complex packing, 2nd-order spatial differencing (wgrib2 c3)"},
            "bitmap": True,
            "winds": "grid-relative (-new_grid_winds grid); U/V and USTM/VSTM paired, one message per pair",
            "interpolation": {"default": "neighbor", "budget": ("APCP", "WEASD", "SNOD")},
            "records": {"f00": {"fields": 860, "messages": 794}, "fNN": {"fields": 876, "messages": 810},
                        "uv_pairs": 66},
            "recordlist": "conusnest",
            "pdt": (0, 8),
            "noaa_header": {"centre": 7, "subcentre": 0, "generating_process": 84, "master": 2, "local": 1},
            "noaa_mb_per_file": (962, 1067),
            "decoded_from": "nam.t06z.conusnest.hiresf01.tm00.grib2 (2026-11-03): Lambert 1799 x 1059, 3000 m, "
                            "La1 21.138 Lo1 237.28, LoV 262.5, Latin 38.5/38.5 (the HRRR CONUS grid), DRT 5.3, bitmap "
                            "(1,905,065 of 1,905,141 points on most fields); NAM-SUITE-2026-11-06/SPEC.md 3.2, 3.4",
        },
    },
    "alaskanest": {
        "im": 1189, "jm": 1249, "i_parent_start": 139, "j_parent_start": 398, "ratio": 4, "res_km": 3,
        "hours": 60, "spec_adv": True, "nps_center": (59.80179, -157.29422),
        "src": "nam.v4.2.11 parm/nam_configfile_alaskanest (im 1189, jm 1249, i/j_parent_start 139/398, ratio 4, "
               "dt 6+1/4 s, filt_dt 2+0/4, nrads/nradl 192, lnsh/lnsv 5, spec_adv .true., convection none); "
               "parm/nam_namelist.nps_alaskanest dom_cen 59.80179, -157.29422; scripts/exnam_fcst.sh LENGTH=60",
        "hiresf": {
            "file": "woof-nam.tHHz.alaskanest.hiresfFF.tm00.grib2",
            "noaa_file": "nam.tHHz.alaskanest.hiresfFF.tm00.grib2",
            "leads": (0, 60, 1),
            "wgrib2": "nps:210:60 181.429:1649:2976 40.530:1105:2976",
            "gdt": {"template": 20, "earth_shape": 6, "nx": 1649, "ny": 1105, "la1": 40.53, "lo1": 181.429,
                    "lad": 60.0, "lov": 210.0, "dx_m": 2976.0, "dy_m": 2976.0, "projection_centre": 0,
                    "resolution_flags": 0x38, "scan_mode": 0x40},
            "packing": {"drt": 3, "name": "complex packing, 2nd-order spatial differencing (wgrib2 c3)"},
            "bitmap": True,
            # points inside the nest of the 1649 x 1105 = 1,822,145 (NOAA's 2 m T bitmap, hiresf01); this row's native
            # grid covers the same 1,439,016 to 4 points (nearest-neighbour half-cell edge, tests/test_ncep_nests.py note)
            "bitmap_points": 1439016,
            "winds": "grid-relative (-new_grid_winds grid); U/V and USTM/VSTM paired, one message per pair",
            "interpolation": {"default": "neighbor", "budget": ("APCP", "WEASD", "SNOD")},
            "records": {"f00": {"fields": 859, "messages": 793}, "fNN": {"fields": 871, "messages": 805},
                        "uv_pairs": 66},
            "recordlist": "alaskanest",
            "pdt": (0, 8),
            "noaa_header": {"centre": 7, "subcentre": 0, "generating_process": 84, "master": 2, "local": 1},
            "noaa_mb_per_file": (770, 863),
            "decoded_from": "nam.t06z.alaskanest.hiresf01.tm00.grib2 (2026-11-03): 805 messages, 871 fields, GDT 3.20 "
                            "as above, DRT 5.3 on all 871, bitmap on all 871, PDT 4.0 x 840 and 4.8 x 31",
        },
    },
    # nam.v4.2.11 parm/nam_configfile_hawaiinest: 373 x 561, i/j_parent_start 42/207, ratio 4, dt 6+1/4, filt_dt 2+0/3,
    # nrads/nradl 192, lnsh/lnsv 5, spec_adv .true., convection none, TDFI (exnam_fcst.sh filteropt=3)
    "hawaiinest": {
        "im": 373, "jm": 561, "i_parent_start": 42, "j_parent_start": 207, "ratio": 4, "res_km": 3,
        "hours": 60, "spec_adv": True, "nps_center": (20.91, -157.22),
        "src": "nam.v4.2.11 parm/nam_configfile_hawaiinest (im 373, jm 561, i/j_parent_start 42/207, ratio 4, "
               "dt 6+1/4 s, filt_dt 2+0/3, nrads/nradl 192, lnsh/lnsv 5, spec_adv .true., convection none); "
               "parm/nam_namelist.nps_hawaiinest dom_cen 20.91, -157.22; scripts/exnam_fcst.sh LENGTH=60",
        "hiresf": {
            "file": "woof-nam.tHHz.hawaiinest.hiresfFF.tm00.grib2",
            "noaa_file": "nam.tHHz.hawaiinest.hiresfFF.tm00.grib2",
            "leads": (0, 60, 1),
            "wgrib2": "mercator:20 198.475:321:2500:206.131 18.073:225:2500:23.088",
            "gdt": {"template": 10, "earth_shape": 6, "nx": 321, "ny": 225, "la1": 18.073, "lo1": 198.475,
                    "la2": 23.088, "lo2": 206.131, "lad": 20.0, "dx_m": 2500.0, "dy_m": 2500.0,
                    "resolution_flags": 0x38, "scan_mode": 0x40},
            "packing": {"drt": 40, "name": "JPEG2000 (wgrib2 jpeg)"},
            # the whole 321 x 225 Mercator box lies inside the nest: NOAA's file carries a bitmap on 28 of its 871
            # fields only (HINDEX and the flight-level TMP/UGRD/VGRD at 305-3658 m above sea)
            "bitmap": False,
            "bitmap_points": 72225,
            "winds": "grid-relative (-new_grid_winds grid); U/V and USTM/VSTM paired, one message per pair",
            "interpolation": {"default": "neighbor", "budget": ("APCP", "WEASD", "SNOD")},
            "records": {"f00": {"fields": 859, "messages": 793}, "fNN": {"fields": 871, "messages": 805},
                        "uv_pairs": 66},
            "recordlist": "hawaiinest",
            "pdt": (0, 8),
            "noaa_header": {"centre": 7, "subcentre": 0, "generating_process": 84, "master": 2, "local": 1},
            "noaa_mb_per_file": (19, 29),
            "decoded_from": "nam.t06z.hawaiinest.hiresf01.tm00.grib2 (2026-11-03): 805 messages, 871 fields, GDT 3.10 "
                            "as above, DRT 5.40 on all 871, bitmap on 28, PDT 4.0 x 840 and 4.8 x 31",
        },
    },
    # nam.v4.2.11 parm/nam_configfile_priconest: 401 x 325, i/j_parent_start 738/145, ratio 4, otherwise as hawaiinest
    "priconest": {
        "im": 401, "jm": 325, "i_parent_start": 738, "j_parent_start": 145, "ratio": 4, "res_km": 3,
        "hours": 60, "spec_adv": True, "nps_center": (18.550, -68.958),
        "src": "nam.v4.2.11 parm/nam_configfile_priconest (im 401, jm 325, i/j_parent_start 738/145, ratio 4, "
               "dt 6+1/4 s, filt_dt 2+0/3, nrads/nradl 192, lnsh/lnsv 5, spec_adv .true., convection none); "
               "parm/nam_namelist.nps_priconest dom_cen 18.550, -68.958; scripts/exnam_fcst.sh LENGTH=60",
        "hiresf": {
            "file": "woof-nam.tHHz.priconest.hiresfFF.tm00.grib2",
            "noaa_file": "nam.tHHz.priconest.hiresfFF.tm00.grib2",
            "leads": (0, 60, 1),
            "wgrib2": "mercator:20 284.5:544:2500:297.491 15.0:310:2500:22.005",
            "gdt": {"template": 10, "earth_shape": 6, "nx": 544, "ny": 310, "la1": 15.0, "lo1": 284.5,
                    "la2": 22.005, "lo2": 297.491, "lad": 20.0, "dx_m": 2500.0, "dy_m": 2500.0,
                    "resolution_flags": 0x38, "scan_mode": 0x40},
            "packing": {"drt": 40, "name": "JPEG2000 (wgrib2 jpeg)"},
            "bitmap": True,
            # points inside the nest of the 544 x 310 = 168,640 (NOAA's hiresf01 first field); this row's native grid
            # covers the same 145,165 to 402 rim points (nearest-neighbour half cell)
            "bitmap_points": 145165,
            "winds": "grid-relative (-new_grid_winds grid); U/V and USTM/VSTM paired, one message per pair",
            "interpolation": {"default": "neighbor", "budget": ("APCP", "WEASD", "SNOD")},
            "records": {"f00": {"fields": 859, "messages": 793}, "fNN": {"fields": 871, "messages": 805},
                        "uv_pairs": 66},
            "recordlist": "priconest",
            "pdt": (0, 8),
            "noaa_header": {"centre": 7, "subcentre": 0, "generating_process": 84, "master": 2, "local": 1},
            "noaa_mb_per_file": (69, 97),
            "decoded_from": "nam.t06z.priconest.hiresf01.tm00.grib2 (2026-11-03): 805 messages, 871 fields, GDT 3.10 "
                            "as above, DRT 5.40 on all 871, bitmap on all 871, PDT 4.0 x 840 and 4.8 x 31",
        },
    },
    # nam.v4.2.11 parm/nam_configfile_firewxnest_{conus,alaska}: 333 x 333, ratio 2 inside the CONUS or Alaska 3 km nest
    # (1.5 km, dlmd 0.01575; the 1.33 km nest was pre-v4), dt 2+1/12, filt_dt 1+0/3, nrads/nradl 288, lnsh/lnsv 3,
    # nodeep false, no digital filter (exnam_fcst.sh sets filteropt=0 for it), 36 h (exnam_pmgr_nest.sh TEND=36).
    # Placed per cycle (placement "firewx": ncep_firewx.nest_grid); stays fixed through the forecast.
    "firewxnest": {
        "im": 333, "jm": 333, "ratio": 2, "res_km": 1.5, "hours": 36, "spec_adv": True, "placement": "firewx",
        "keys": {"lnsh": "3", "lnsv": "3", "filter_method": "0", "nodeep": "false",
                 # nam_configfile_firewxnest_{conus,alaska}: accumulators emptied every hour (the other nests: 3)
                 "nhrs_prec": "1", "nhrs_heat": "1", "nhrs_clod": "1", "nhrs_rdlw": "1", "nhrs_rdsw": "1",
                 "nhrs_srfc": "1"},
        "src": "nam.v4.2.11 parm/nam_configfile_firewxnest_{conus,alaska} (im/jm 333, ratio 2, dt 2+1/12 s, filt_dt "
               "1+0/3, nrads/nradl 288, lnsh/lnsv 3, nodeep false, spec_adv .true.); scripts/exnam_fcst.sh "
               "filteropt=0 and I/J_START_FIREWX from firewx_ijstart.txt; scripts/exnam_pmgr_nest.sh TEND=36; "
               "util/sorc/setfirewx.fd (placement rule)",
        "hiresf": {
            "file": "woof-nam.tHHz.firewxnest.hiresfFF.tm00.grib2",
            "noaa_file": "nam.tHHz.firewxnest.hiresfFF.tm00.grib2",
            "side_file": "woof-nam.tHHz.firewxnest_center_latlon",
            "leads": (0, 36, 1),
            # per cycle: the grid's "firewx" block (ncep_firewx.nest_grid) carries the wgrib2 spec for its point
            "wgrib2": None,
            "gdt": {"template": 30, "earth_shape": 6, "lad": 25.0, "lov": 265.0, "latin1": 25.0, "latin2": 25.0,
                    "dx_m": 1270.0, "dy_m": 1270.0, "projection_centre": 0, "resolution_flags": 0x38,
                    "scan_mode": 0x40, "per_cycle": ("nx", "ny", "la1", "lo1")},
            "gdt_alaska": {"template": 20, "earth_shape": 6, "lad": 60.0, "lov": 210.0, "dx_m": 1488.0,
                           "dy_m": 1488.0, "per_cycle": ("nx", "ny", "la1", "lo1")},
            "packing": {"drt": 40, "name": "JPEG2000 (wgrib2 jpeg)"},
            "bitmap": True,
            "winds": "grid-relative (-new_grid_winds grid); U/V and USTM/VSTM paired, one message per pair",
            "interpolation": {"default": "neighbor", "budget": ("APCP", "WEASD", "SNOD")},
            "records": {"f00": {"fields": 859, "messages": 793}, "fNN": {"fields": 871, "messages": 805},
                        "uv_pairs": 66},
            "recordlist": "firewxnest",
            "pdt": (0, 8),
            "noaa_header": {"centre": 7, "subcentre": 0, "generating_process": 84, "master": 2, "local": 1},
            "noaa_mb_per_file": (94, 136),
            "decoded_from": "nam.t06z.firewxnest.hiresf01.tm00.grib2 (2026-11-03, point 42.0 -122.3): 805 messages, "
                            "871 fields, GDT 3.30 646 x 389 La1 39.182 Lo1 233.507, DRT 5.40 and a bitmap on all 871 "
                            "(184,183 points inside the nest), PDT 4.0 x 840 and 4.8 x 31",
        },
    },
}


def parent_grid() -> dict:
    p = dict(NAM_PARENT)
    p["dx_km"] = A_EARTH * math.radians(p["dlmd"]) / 1e3
    p["dy_km"] = A_EARTH * math.radians(p["dphd"]) / 1e3
    p["extent_km"] = [round((p["im"] - 1) * p["dx_km"], 1), round((p["jm"] - 1) * p["dy_km"], 1)]
    return p


def names() -> list[str]:
    return sorted(NCEP_NESTS)


def row(name: str) -> dict:
    if name not in NCEP_NESTS:
        raise SystemExit(f"--ncep-nest {name}: no row in namgpu/nest/ncep_nests.py NCEP_NESTS (rows: {names()})")
    return NCEP_NESTS[name]


def nest_grid(name: str, window: tuple[int, int] | None = None, offset: tuple[int, int] | None = None,
              cycle_hour: int | None = None, firewx_loc=None, firewx_point=None) -> dict:
    """The row's native grid in domain.nest_grid_associated's form (so domain.config_keys writes dlmd/dphd and the
    NMMB nest keys). window = (nxp, nyp): a centred sub-box of the row's grid, in whole parent cells, on the same pole
    and spacing (a smoke on a card the whole nest does not fit; never a product grid). offset = (i0, j0): the window's
    SW corner in parent cells from the nest's SW corner instead of centred (e.g. the Alaska nest's north-east part,
    which holds the dateline and reaches 83.8 N). A row placed per cycle (placement "firewx") takes the cycle hour,
    the nam_firewx_loc file and an explicit point instead (ncep_firewx.nest_grid)."""
    r = row(name)
    if r.get("placement") == "firewx":
        if window:
            raise SystemExit(f"--ncep-window: the {name} nest is placed per cycle; it has no window")
        from namgpu.nest import ncep_firewx
        return ncep_firewx.nest_grid(r, 0 if cycle_hour is None else cycle_hour, firewx_loc, firewx_point)
    p = parent_grid()
    ratio = int(r["ratio"])
    # NCEP's grids need not span whole parent cells (conusnest: im - 1 = 1826 = 456.5 x 4); a window is whole cells
    im, jm = int(r["im"]), int(r["jm"])
    nxp, nyp = (im - 1) / ratio, (jm - 1) / ratio
    i_ps, j_ps = int(r["i_parent_start"]), int(r["j_parent_start"])
    if window:
        fx, fy = (im - 1) // ratio, (jm - 1) // ratio
        wx, wy = int(window[0]), int(window[1])
        if not (2 <= wx <= fx and 2 <= wy <= fy):
            raise SystemExit(f"--ncep-window {wx}x{wy}: must be 2..{fx} x 2..{fy} parent cells (the {name} grid)")
        oi, oj = offset if offset is not None else ((fx - wx) // 2, (fy - wy) // 2)
        if not (0 <= oi <= fx - wx and 0 <= oj <= fy - wy):
            raise SystemExit(f"--ncep-window {wx}x{wy}+{oi}+{oj}: the window leaves the {name} grid "
                             f"(offset 0..{fx - wx} x 0..{fy - wy} parent cells)")
        i_ps += oi
        j_ps += oj
        nxp, nyp = wx, wy
        im, jm = wx * ratio + 1, wy * ratio + 1
    dlmd, dphd = p["dlmd"] / ratio, p["dphd"] / ratio
    dx, dy = p["dx_km"] / ratio, p["dy_km"] / ratio
    g = {"res_km": int(r["res_km"]), "im": im, "jm": jm, "lm": 60, "tph0d": p["tph0d"], "tlm0d": p["tlm0d"],
         "dlmd": dlmd, "dphd": dphd,
         "wbd": round(p["wbd"] + (i_ps - 1) * p["dlmd"], 6), "sbd": round(p["sbd"] + (j_ps - 1) * p["dphd"], 6),
         "dx_km": round(dx, 4), "dy_km": round(dy, 4),
         "extent_km": [round((im - 1) * dx, 1), round((jm - 1) * dy, 1)],
         "associated": True, "i_parent_start": i_ps, "j_parent_start": j_ps, "parent_child_space_ratio": ratio,
         "parent_cells": [nxp, nyp], "ncep_nest": name, "ncep_window": list(window) if window else None,
         "ncep_window_offset": list(offset) if (window and offset is not None) else None}
    lon, lat = center(g)
    g["center"] = [round(lat, 5), round(lon, 5)]
    return g


def center(g: dict) -> tuple[float, float]:
    """(lon, lat) degrees of the grid's middle H point (the point NPS's dom_cen names for an odd-sized grid)."""
    import numpy as np
    from namgpu import grid
    tlm = math.radians(g["wbd"] + 0.5 * (g["im"] - 1) * g["dlmd"])
    tph = math.radians(g["sbd"] + 0.5 * (g["jm"] - 1) * g["dphd"])
    lon, lat = grid.rotated_to_geographic(np.float64(tlm), np.float64(tph), math.radians(g["tlm0d"]),
                                          math.radians(g["tph0d"]))
    return math.degrees(float(lon)), math.degrees(float(lat))


def domain_name(g: dict) -> str:
    if g.get("firewx"):
        from namgpu.nest import ncep_firewx
        return ncep_firewx.domain_name(g)
    n = g["ncep_nest"]
    w, o = g.get("ncep_window"), g.get("ncep_window_offset")
    return n if not w else f"{n}-w{w[0]}x{w[1]}" + (f"+{o[0]}+{o[1]}" if o else "")
