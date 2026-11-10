"""Record tables of the isobaric init sources: which GRIB2 records each source row reads, and the generic
conversions that turn them into the fields the isobaric start (gfs_start.py) and boundaries (gfs_bocos.py) use.

Adding a source is a table here plus a row in sources.SOURCES (the arbitrary acceptance test): the conversions
are named physics operations (scale, dewpoint -> q, q -> RH, conservative soil-layer remap, ratio, constant,
donor field from another source's file), never per-model code.

A record table has:
  pl            name -> (discipline, category, parameter) on isobaric levels (level type 100, Pa). Names are
                gfs_start.PL_ORDER slots; a slot a source does not publish stays 0 (the condensate species of a
                source without hydrometeors: init-gfs-vinterp's do_clouds gate then sets CW = 0).
  pl_need       slots every level used must carry (the level set is their intersection).
  pl_derive     slot -> (op, args): computed on the device from the decoded slots, before the decode sanitiser.
  sfc           name -> (d, c, p, level_type, level_value, second_level_value) single-level records.
  sfc_derive    ordered [(out, op, args)]: computed from decoded sfc fields (and earlier outputs).
  sfc_columns   the sfc records the vertical interpolation needs (boundary times decode only these, plus what
                their sfc_derive needs).
  derive_columns  the sfc_derive outputs a boundary time needs.
  donors        name -> {"file": template under the case dir, "spec": sfc selector}: a field the source does not
                publish, taken from another source's file on its own grid (e.g. ECMWF open data has no sea-ice
                concentration; the same cycle's GFS f000 ICEC is the donor, as gpuwm's fix/ifs-open-data-sea-ice).
  container     "grib2" (default: init_grib2_decode; pl and sfc are (d, c, p, ...) record specs) or "regular-netcdf"
                (gpuwm's regular-forcing NetCDF reader, gpuwm.ingest.regular_netcdf, the files gpuwm.era5_arco writes:
                pl and sfc map each slot to the file's field name; levels in hPa; missing points are NaN).
  lat_crop      "south": keep source rows north of the domain's south edge (the GFS path as it has always been);
                "both": keep only the domain's latitude band plus the margin (0.1 deg global fields are 6.5 M points
                per record; the band keeps the device copy small).

Operations (all elementwise, float32, on the device):
  scale(src, factor)                 out = src * factor
  const(value)                       out = value everywhere
  ratio(num, den, floor)             out = num / max(den, floor)
  q_from_dewpoint(td, p)             specific humidity from dewpoint (K) and pressure (Pa):
                                     e = es(td), q = eps e / (p - (1 - eps) e)
  rh_from_q(q, t, p)                 RH (percent) that init-gfs-vinterp's own rh_to_mxrat (liquid branch,
                                     kernels/init_gfs_vinterp.cu init_gfs_vinterp_rhq) turns back into q:
                                     w = q / (1 - q), RH = 100 w / qs(t, p). So the source's q is what the model gets
                                     (inside the kernel's 0.001..100 % clamp), whatever the source's own RH definition.
  layer_remap(srcs, src_layers, tgt_layers)
                                     conservative overlap weights, target layer mean = sum_k w_k src_k
                                     (gpuwm ingest/soil_contract.py conservative_overlap_weights, same rule).
"""
from __future__ import annotations

import numpy as np

EPS = 0.622
SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15     # init_gfs_vinterp_rhq's constants (hPa formula)
G0 = 9.80665                                               # WMO standard gravity: IFS z (m2 s-2) -> gpm

# GFS 0.25 deg pgrb2 (NCEP). Byte for byte the tables gfs_start has always used.
GFS = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "RH": (0, 1, 1), "UGRD": (0, 2, 2),
           "VGRD": (0, 2, 3), "CLMR": (0, 1, 22), "ICMR": (0, 1, 23), "RWMR": (0, 1, 24), "SNMR": (0, 1, 25)},
    "pl_need": ("HGT", "TMP", "RH", "UGRD", "VGRD"),
    "pl_derive": {},
    "sfc": {
        "PRES": (0, 3, 0, "1", "0", None), "HGT": (0, 3, 5, "1", "0", None),
        "TSK": (0, 0, 0, "1", "0", None),
        "T2": (0, 0, 0, "103", "2", None), "RH2": (0, 1, 1, "103", "2", None),
        "Q2": (0, 1, 0, "103", "2", None),
        "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
        "TSOIL1": (2, 0, 2, "106", "0", "0.1"), "TSOIL2": (2, 0, 2, "106", "0.1", "0.4"),
        "TSOIL3": (2, 0, 2, "106", "0.4", "1"), "TSOIL4": (2, 0, 2, "106", "1", "2"),
        "SOILW1": (2, 0, 192, "106", "0", "0.1"), "SOILW2": (2, 0, 192, "106", "0.1", "0.4"),
        "SOILW3": (2, 0, 192, "106", "0.4", "1"), "SOILW4": (2, 0, 192, "106", "1", "2"),
        "CNWAT": (2, 0, 196, "1", "0", None), "WEASD": (0, 1, 13, "1", "0", None),
        "SNOD": (0, 1, 11, "1", "0", None), "ICEC": (10, 2, 0, "1", "0", None),
        "LANDSRC": (2, 0, 0, "1", "0", None),
        # Archived GFS before the 2015-01-14 GFS upgrade (NCEI gfsanl_4, 0.5 deg) publishes soil temperature as TMP
        # (0-0-0) on the soil layers, not TSOIL (2-0-2): read whichever the file carries (sfc_optional, fill_from).
        "STMP1": (0, 0, 0, "106", "0", "0.1"), "STMP2": (0, 0, 0, "106", "0.1", "0.4"),
        "STMP3": (0, 0, 0, "106", "0.4", "1"), "STMP4": (0, 0, 0, "106", "1", "2"),
    },
    # Records an archived GFS file may lack (fetch_case.GFS_ERAS before the S3 bucket; checked by inventory
    # 2026-11-06): the 0.5 deg analyses (2004-2015) carry no TSOIL (TMP on the soil layers instead), no SNOD and
    # no CNWAT; the 0.25 deg GDEX files of 2015-2016 carry no CNWAT. A file that carries them decodes them as before.
    "sfc_optional": ("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4", "STMP1", "STMP2", "STMP3", "STMP4", "CNWAT", "SNOD"),
    # fill_* ops run only where the file did not carry the output (a current GFS file is decoded unchanged)
    "sfc_derive": [
        ("TSOIL1", "fill_from", ("STMP1",)), ("TSOIL2", "fill_from", ("STMP2",)),
        ("TSOIL3", "fill_from", ("STMP3",)), ("TSOIL4", "fill_from", ("STMP4",)),
        ("SNOD", "fill_scale", ("WEASD", 0.005)),   # depth (m) at WRF real.exe's fixed 200 kg m-3 (as ERA5 here)
        ("CNWAT", "fill_const", (0.0,)),            # no canopy water in the file: 0 (declared fill, as IFS/ERA5)
    ],
    # the GFS forecast files (f006 ...) carry no CNWAT, so boundary times ask only for these (2026-11-04)
    "sfc_columns": ("PRES", "HGT", "T2", "RH2", "U10", "V10"),
    "derive_columns": (),
    "donors": {},
    "lat_crop": "south",
}

# IFS layers sot/vsw 1..4 (levtype sol, GRIB2 level type 151 levels 0-1 .. 3-4): 0-7, 7-28, 28-100, 100-289 cm
# (IFS documentation, Part IV physical processes, HTESSEL soil); Noah's four: 0-10, 10-40, 40-100, 100-200 cm.
IFS_SOIL_LAYERS = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))
NOAH_SOIL_LAYERS = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))

# ECMWF open data IFS HRES (0.1 deg pl + sfc pair, or 0.25 deg single file): parameter numbers as the files
# carry them (grib2_inventory of 2026-11-03 12Z f000: sp 0-3-0, z 0-3-4, skt 0-0-17, 2t 0-0-0/103, 2d 0-0-6/103,
# 10u/10v 0-2-2/0-2-3 at 103 10 m, lsm 2-0-0, sd 0-1-254 (m of water equivalent), rsn 0-1-61 (kg m-3),
# sot 2-3-18 and vsw 2-0-25 on level type 151 layers 0-1 .. 3-4).
IFS = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "UGRD": (0, 2, 2), "VGRD": (0, 2, 3)},
    "pl_need": ("HGT", "TMP", "SPFH", "UGRD", "VGRD"),
    "pl_derive": {"RH": ("rh_from_q", ("SPFH", "TMP", "@p"))},
    "sfc": {
        "PRES": (0, 3, 0, "1", "0", None), "Z": (0, 3, 4, "1", "0", None),
        "TSK": (0, 0, 17, "1", "0", None),
        "T2": (0, 0, 0, "103", "2", None), "D2": (0, 0, 6, "103", "2", None),
        "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
        "SOT1": (2, 3, 18, "151", "0", "1"), "SOT2": (2, 3, 18, "151", "1", "2"),
        "SOT3": (2, 3, 18, "151", "2", "3"), "SOT4": (2, 3, 18, "151", "3", "4"),
        "VSW1": (2, 0, 25, "151", "0", "1"), "VSW2": (2, 0, 25, "151", "1", "2"),
        "VSW3": (2, 0, 25, "151", "2", "3"), "VSW4": (2, 0, 25, "151", "3", "4"),
        "SD": (0, 1, 254, "1", "0", None), "RSN": (0, 1, 61, "1", "0", None),
        "LANDSRC": (2, 0, 0, "1", "0", None),
    },
    "sfc_derive": [
        ("HGT", "scale", ("Z", 1.0 / G0)),
        ("Q2", "q_from_dewpoint", ("D2", "PRES")),
        ("RH2", "rh_from_q", ("Q2", "T2", "PRES")),
        (("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4"), "layer_remap",
         (("SOT1", "SOT2", "SOT3", "SOT4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        (("SOILW1", "SOILW2", "SOILW3", "SOILW4"), "layer_remap",
         (("VSW1", "VSW2", "VSW3", "VSW4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        ("WEASD", "scale", ("SD", 1000.0)),                  # m of water -> kg m-2
        ("SNOD", "ratio", ("WEASD", "RSN", 50.0)),           # physical depth (m) = SWE / snow density
        ("CNWAT", "const", (0.0,)),                          # IFS open data publishes no canopy water
    ],
    "sfc_columns": ("PRES", "Z", "T2", "D2", "U10", "V10"),
    "derive_columns": ("HGT", "Q2", "RH2"),
    "donors": {"ICEC": {"file": "raw/ifs/gfs.t{cyc:02d}z.icec.f000.grib2", "spec": (10, 2, 0, "1", "0", None)}},
    # Snow: ECMWF open data carries sd/rsn only in later files (the 2025-03-14 12Z 0.25 deg file has neither). A file
    # without them takes snow water and depth from the same cycle's GFS f000 instead (receipt: donors_used).
    "sfc_optional": ("SD", "RSN"),
    "donor_fallback": {
        "WEASD": {"file": "raw/ifs/gfs.t{cyc:02d}z.weasd.f000.grib2", "spec": (0, 1, 13, "1", "0", None)},
        "SNOD": {"file": "raw/ifs/gfs.t{cyc:02d}z.snod.f000.grib2", "spec": (0, 1, 11, "1", "0", None)},
    },
    "lat_crop": "both",
}

# ERA5 reanalysis as gpuwm.era5_arco writes it (fields named by gpuwm's regular-forcing schema, REGULAR_FIELD_UNITS:
# Z and SOILGEO m2 s-2, SNOW_EC m of water equivalent, ST/SM the four ERA5 layers 0-7, 7-28, 28-100, 100-289 cm, the
# same HTESSEL layers as IFS; Q2 is gpuwm's specific humidity from the 2 m dewpoint). Snow depth: gpuwm's ERA5 field
# list has the water equivalent only (no snow density), so the physical depth takes WRF real.exe's fixed 5:1
# depth-to-water ratio (200 kg m-3, module_initialize_real.F), as gpuwm's ingest/soil.py does for SNOW_EC.
ERA5 = {
    "container": "regular-netcdf",
    "pl": {"HGT": "Z", "TMP": "T", "SPFH": "SPFH", "UGRD": "U", "VGRD": "V"},
    "pl_need": ("HGT", "TMP", "SPFH", "UGRD", "VGRD"),
    # applied in this order: geopotential height from geopotential, then RH from q
    "pl_derive": {"HGT": ("scale", ("HGT", 1.0 / G0)), "RH": ("rh_from_q", ("SPFH", "TMP", "@p"))},
    "sfc": {
        "PRES": "PSFC", "Z": "SOILGEO", "TSK": "SKINTEMP", "T2": "T2", "Q2": "Q2",
        "U10": "U10", "V10": "V10",
        "SOT1": "ST000007", "SOT2": "ST007028", "SOT3": "ST028100", "SOT4": "ST100289",
        "VSW1": "SM000007", "VSW2": "SM007028", "VSW3": "SM028100", "VSW4": "SM100289",
        "SNOW_EC": "SNOW_EC", "ICEC": "SEAICE", "LANDSRC": "LANDSEA",
    },
    "sfc_derive": [
        ("HGT", "scale", ("Z", 1.0 / G0)),
        ("RH2", "rh_from_q", ("Q2", "T2", "PRES")),
        (("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4"), "layer_remap",
         (("SOT1", "SOT2", "SOT3", "SOT4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        (("SOILW1", "SOILW2", "SOILW3", "SOILW4"), "layer_remap",
         (("VSW1", "VSW2", "VSW3", "VSW4"), IFS_SOIL_LAYERS, NOAH_SOIL_LAYERS)),
        ("WEASD", "scale", ("SNOW_EC", 1000.0)),             # m of water -> kg m-2
        ("SNOD", "scale", ("SNOW_EC", 5.0)),                 # physical depth (m) at 200 kg m-3
        ("CNWAT", "const", (0.0,)),                          # canopy water is not in gpuwm's ERA5 field list
    ],
    "sfc_columns": ("PRES", "Z", "T2", "Q2", "U10", "V10"),
    "derive_columns": ("HGT", "RH2"),
    "donors": {},
    "lat_crop": "both",
}

# RUC LSM soil: nine depth NODES (m), values at a depth, not layer means. RAP, HRRR and RRFS publish TSOIL/SOILW at
# exactly these depths (GRIB2 level type 106 with second level = first). Noah's layers are read at their midpoints
# (0.05, 0.25, 0.7, 1.5 m) by linear interpolation between the bracketing nodes: gpuwm's own RAP/HRRR start does
# this (gpuwm/authorities/rw-wps-rap-awip32-grib2.composition.json soil remap kind "linear_node_samples",
# gpuwm/ingest/soil.py _remap_declared_soil), and so does WRF real.exe for a level-form soil source going into Noah
# (module_soil_pre.F init_soil_2_real). Neither rescales moisture between the RUC and Noah soil tables, so neither
# does this: the volumetric moisture is RAP's, read at Noah's depths. --soil-from gfs takes Noah-native soil instead.
RUC_SOIL_NODES = (0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0)

# RAP 13 km analysis on its own rotated lat-lon grid (rap.tHHz.wrfprsf00.grib2, NCEP template 3.32769, 953 x 834,
# isobaric 1013.2 and 1000-50 hPa every 25 hPa). Codes from grib2_inventory of rap.t00z.wrfprsf00 2023-07-29: cloud
# ice is NCEP 0-1-82 (CIMIXR), not GFS's 0-1-23; graupel 0-1-32; TKE 0-19-11; RUC soil 2-0-2 / 2-0-192 at the nine
# nodes; no canopy water record. Grid and winds: source_grids.py (grid-relative winds turned to earth-relative).
#   pl_add    slot -> extra isobaric records summed into it at decode: graupel joins snow, the Ferrier-Aligo ice
#             category (NMMB carries one ice mass, F_ICE of CWM, with a rime factor; RAP's graupel has no rime factor
#             to carry, so F_RIMEF stays 1: graupel is kept as ice mass, its density is dropped).
#   pl_extra  name -> extra isobaric records kept beside the slots: TKE (NMMB Q2 = 2 TKE, as the native warm start).
#   warm      the start carries the source's condensate (CW, F_ICE, F_RAIN) and TKE (Q2) instead of the cold CW = 0.
RAP = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "UGRD": (0, 2, 2), "VGRD": (0, 2, 3),
           "CLMR": (0, 1, 22), "ICMR": (0, 1, 82), "RWMR": (0, 1, 24), "SNMR": (0, 1, 25)},
    "pl_need": ("HGT", "TMP", "SPFH", "UGRD", "VGRD"),
    # q is RAP's own: RH derived from it with the vinterp kernel's own formula (rh_from_q), not RAP's RH record
    "pl_derive": {"RH": ("rh_from_q", ("SPFH", "TMP", "@p"))},
    "pl_add": {"SNMR": ((0, 1, 32),)},
    "pl_extra": {"TKE": (0, 19, 11)},
    "warm": True,
    "sfc": {
        "PRES": (0, 3, 0, "1", "0", None), "HGT": (0, 3, 5, "1", "0", None),
        "TSK": (0, 0, 0, "1", "0", None),
        "T2": (0, 0, 0, "103", "2", None), "Q2": (0, 1, 0, "103", "2", None),
        "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
        **{f"STN{k}": (2, 0, 2, "106", f"{z:g}", f"{z:g}") for k, z in enumerate(RUC_SOIL_NODES)},
        **{f"SMN{k}": (2, 0, 192, "106", f"{z:g}", f"{z:g}") for k, z in enumerate(RUC_SOIL_NODES)},
        "WEASD": (0, 1, 13, "1", "0", None), "SNOD": (0, 1, 11, "1", "0", None),
        "ICEC": (10, 2, 0, "1", "0", None), "LANDSRC": (2, 0, 0, "1", "0", None),
    },
    "sfc_derive": [
        ("RH2", "rh_from_q", ("Q2", "T2", "PRES")),
        (("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4"), "node_interp",
         (tuple(f"STN{k}" for k in range(len(RUC_SOIL_NODES))), RUC_SOIL_NODES, NOAH_SOIL_LAYERS)),
        (("SOILW1", "SOILW2", "SOILW3", "SOILW4"), "node_interp",
         (tuple(f"SMN{k}" for k in range(len(RUC_SOIL_NODES))), RUC_SOIL_NODES, NOAH_SOIL_LAYERS)),
        ("TDEEP", "scale", (f"STN{len(RUC_SOIL_NODES) - 1}", 1.0)),     # the 3 m node: Noah's TG lower boundary
        ("CNWAT", "const", (0.0,)),                                     # RAP's wrfprs has no canopy water record
    ],
    "sfc_columns": ("PRES", "HGT", "T2", "Q2", "U10", "V10"),
    "derive_columns": ("RH2",),
    "donors": {},
    "lat_crop": "both",
}

# HRRR 3 km analysis (hrrr.tHHz.wrfprsf00.grib2, CONUS, NCEP template 3.30 Lambert conformal 1799 x 1059, isobaric
# 1013.2 and 1000-50 hPa every 25 hPa). The same UPP record set as RAP (grib2 .idx of 2026-01-23 12Z): CLMR, CIMIXR,
# RWMR, SNMR, GRLE on every level, the nine RUC soil nodes, LAND, ICEC, WEASD, SNOD, 2 m T/q, 10 m winds. Two
# differences: the file has canopy water (CNWAT surface), so it is read rather than set to 0, and it has no isobaric
# TKE, so Q2 starts at the cold floor (gfs_start.warm_state). Grid and winds: source_grids.LambertGrid.
HRRR = {**RAP,
        "pl_extra": {},
        # canopy water is WMO 2-0-13 in the HRRR wrfprsf00 (grib2_inventory of 2026-01-23 12Z), not NCEP local 2-0-196
        "sfc": {**RAP["sfc"], "CNWAT": (2, 0, 13, "1", "0", None)},
        "sfc_derive": [d for d in RAP["sfc_derive"] if d[0] != "CNWAT"]}

# RRFS v1 (NOAA's operational FV3 regional, 3 km North America) as published on noaa-rrfs-ops-pds: the CONUS cut on
# HRRR's Lambert grid, the state split over two files per hour. prslev (45 isobaric levels 1000-2 hPa): HGT TMP SPFH
# UGRD VGRD CLMR ICMR RWMR SNMR GRLE; cloud ice is 0-1-23 there (GFS's code, measured on rrfs.t00z.prslev f000
# 2026-10-14), not HRRR's 0-1-82. 2dfld: the surface, 2 m / 10 m, LAND, ICEC, snow, canopy water and the nine RUC
# soil nodes, the same records as HRRR's wrfprsf00 surface set. sfc_file_from: the surface records are read from the
# file whose name has prslev replaced by 2dfld (gfs_start.decode_projected). No isobaric TKE: Q2 starts at the floor.
RRFS = {**HRRR,
        "pl": {**HRRR["pl"], "ICMR": (0, 1, 23)},
        "sfc_file_from": ("prslev", "2dfld")}

# ECMWF AIFS Single 0.25 deg as an isobaric BOUNDARY source (its start is the IFS analysis, sources row aifs
# start_from ifs-0p25). The IFS table's records, except that AIFS publishes geopotential z (0-3-4) on the pressure
# levels rather than gh, so HGT is z / g0 (as ERA5). Boundary times read only the columns.
AIFS = {**IFS,
        "pl": {**IFS["pl"], "HGT": (0, 3, 4)},
        "pl_derive": {"HGT": ("scale", ("HGT", 1.0 / G0)), "RH": ("rh_from_q", ("SPFH", "TMP", "@p"))}}

# NOAA AIGFS v1.1 as an isobaric BOUNDARY source (its start is the GFS analysis, sources row aigfs start_from gfs).
# The pres file: HGT TMP SPFH UGRD VGRD on 13 levels (NCEP codes, as GFS). The sfc file: 2 m T and 10 m winds only:
# no surface pressure, no terrain, no 2 m humidity. So:
#   same_grid_donors  HGT (terrain) from the same cycle's GFS analysis, the same 0.25 deg global grid (cut with the
#                     source's own window): terrain does not change over the run.
#   column_derive     PRES: the pressure at the terrain height, hydrostatic in the source's own isobaric heights and
#                     temperatures (between two levels: T linear in height, p = p_k (T/T_k)^(-g/(R a)); below the
#                     lowest level: the standard 6.5 K/km lapse from it, as metgrid/real.exe extrapolate);
#                     RH2: the isobaric RH interpolated linearly in ln p to that surface pressure.
AIGFS = {
    "pl": {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "UGRD": (0, 2, 2), "VGRD": (0, 2, 3)},
    "pl_need": ("HGT", "TMP", "SPFH", "UGRD", "VGRD"),
    "pl_derive": {"RH": ("rh_from_q", ("SPFH", "TMP", "@p"))},
    "sfc": {"T2": (0, 0, 0, "103", "2", None),
            "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None)},
    "same_grid_donors": {"HGT": {"file": "raw/gfs/gfs.t{cyc:02d}z.pgrb2.0p25.f000", "spec": (0, 3, 5, "1", "0", None)}},
    "column_derive": [("PRES", "psfc_from_isobaric", ("HGT",)), ("RH2", "isobaric_at_psfc", ("RH", "PRES"))],
    "sfc_derive": [],
    "sfc_columns": ("T2", "U10", "V10"),
    "derive_columns": (),
    "donors": {},
    "lat_crop": "both",
}

TABLES = {"gfs_0p25": GFS, "ifs": IFS, "era5": ERA5, "rap": RAP, "hrrr": HRRR, "rrfs": RRFS, "aifs": AIFS,
          "aigfs": AIGFS}


def table(row: dict) -> dict:
    """The record table of a SOURCES row (GFS when the row names none: the rows that predate the tables)."""
    return TABLES[row.get("records", "gfs_0p25")]


def boundary_table(row: dict) -> dict:
    """The record table a row's boundary files are read with: its boundary entry's own "records" (aifs, aigfs: the
    start is another row's analysis, the boundaries are the AI forecast's files), else the row's."""
    b = row.get("boundary") or {}
    return TABLES[b["records"]] if b.get("records") else table(row)


RD_ = 287.04
LAPSE = 0.0065                                             # K/m, the standard lapse below the lowest level


def psfc_from_isobaric(hgt, tmp, plev, zs, xp=None):
    """Pressure (Pa) at height zs [ny, nx] from isobaric heights hgt and temperatures tmp [nlev, ny, nx] on levels
    plev (Pa, ground up: highest pressure first). Between the two levels that bracket zs, T is linear in height
    (lapse a = -(T_up - T_lo)/(z_up - z_lo)) and p = p_lo (T(zs)/T_lo)^(g/(R a)) (isothermal form when a ~ 0).
    Below the lowest level: the standard 6.5 K/km from it. Above the highest: that level's isothermal extension.
    Missing (NaN) heights are skipped level by level (a below-ground level a source leaves undefined)."""
    if xp is None:
        import cupy as xp
    f32 = xp.float32
    nlev = hgt.shape[0]
    g_r = f32(G0 / RD_)
    out = xp.full(zs.shape, f32(np.nan), dtype=f32)

    def hyd(p0, t0, z0, a):
        t = t0 - a * (zs - z0)
        small = xp.abs(a) < f32(1e-5)
        iso = p0 * xp.exp(-g_r * (zs - z0) / t0)
        safe_a = xp.where(small, f32(1.0), a)
        lap = p0 * (xp.maximum(t, f32(1.0)) / t0) ** (g_r / safe_a)
        return xp.where(small, iso, lap).astype(f32)

    # below the lowest valid level (ground up: level 0 is the highest pressure)
    z0, t0 = hgt[0], tmp[0]
    below = xp.isfinite(z0) & (zs <= z0)
    out = xp.where(below, hyd(f32(plev[0]), t0, z0, f32(LAPSE)), out)
    for k in range(nlev - 1):
        zl, zu, tl, tu = hgt[k], hgt[k + 1], tmp[k], tmp[k + 1]
        ok = xp.isfinite(zl) & xp.isfinite(zu) & (zs > zl) & (zs <= zu) & ~xp.isfinite(out)
        a = -(tu - tl) / xp.maximum(zu - zl, f32(1.0))
        out = xp.where(ok, hyd(f32(plev[k]), tl, zl, a), out)
    # a column whose lowest level is undefined (below ground in the source): lowest defined level, standard lapse
    for k in range(nlev):
        ok = ~xp.isfinite(out) & xp.isfinite(hgt[k]) & (zs <= hgt[k])
        out = xp.where(ok, hyd(f32(plev[k]), tmp[k], hgt[k], f32(LAPSE)), out)
    top = ~xp.isfinite(out)
    out = xp.where(top, hyd(f32(plev[-1]), tmp[-1], hgt[-1], f32(0.0)), out)
    return out.astype(f32)


def isobaric_at_psfc(field, plev, ps, xp=None):
    """field [nlev, ny, nx] on plev (Pa, ground up) interpolated linearly in ln p to ps [ny, nx]; held at the
    nearest level outside the levels."""
    if xp is None:
        import cupy as xp
    f32 = xp.float32
    lp = np.log(np.asarray(plev, np.float64))
    lps = xp.log(ps)
    out = xp.where(lps >= f32(lp[0]), field[0], field[-1]).astype(f32)
    for k in range(len(plev) - 1):
        hi, lo = f32(lp[k]), f32(lp[k + 1])
        ok = (lps < hi) & (lps >= lo)
        w = (lps - lo) / (hi - lo)
        out = xp.where(ok, w * field[k] + (f32(1.0) - w) * field[k + 1], out)
    return out.astype(f32)


def derive_columns_sfc(rec: dict, pl, plev, order, sfc) -> None:
    """Apply rec['column_derive'] in order: surface fields computed from the isobaric slots (pl [nslot, nlev, ny, nx])
    and earlier surface fields. Works on NumPy (CPU tests) and CuPy arrays alike."""
    if not rec.get("column_derive"):
        return
    xp = np if isinstance(pl, np.ndarray) else __import__("cupy")
    for out, op, args in rec["column_derive"]:
        if op == "psfc_from_isobaric":
            sfc[out] = psfc_from_isobaric(pl[order.index("HGT")], pl[order.index("TMP")], plev, sfc[args[0]], xp=xp)
        elif op == "isobaric_at_psfc":
            sfc[out] = isobaric_at_psfc(pl[order.index(args[0])], plev, sfc[args[1]], xp=xp)
        else:
            raise ValueError(f"source_records: unknown column_derive op {op!r}")


def conservative_overlap_weights(source, target) -> np.ndarray:
    """Target-by-source overlap weights for layer-mean remapping (gpuwm ingest/soil_contract.py, same rule)."""
    w = np.zeros((len(target), len(source)), dtype=np.float64)
    for ti, (t0, t1) in enumerate(target):
        for si, (s0, s1) in enumerate(source):
            w[ti, si] = max(0.0, min(t1, s1) - max(t0, s0)) / (t1 - t0)
        if abs(float(w[ti].sum()) - 1.0) > 2e-15:
            raise ValueError(f"soil remap: target layer {ti} {target[ti]} is not covered by the source layers {source}")
    return w


def node_interp_weights(nodes, target) -> np.ndarray:
    """Target-layer-by-node weights: each target layer's midpoint linearly between its two bracketing depth nodes
    (gpuwm ingest/soil.py _interp_nodes, the "linear_node_samples" remap). A midpoint outside the nodes is refused:
    there is no node pair to read it from (WRF leaves such a level unset)."""
    z = np.asarray(nodes, dtype=np.float64)
    if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError(f"soil node remap: nodes {nodes} must be at least two strictly increasing depths")
    w = np.zeros((len(target), z.size), dtype=np.float64)
    for ti, (t0, t1) in enumerate(target):
        m = 0.5 * (t0 + t1)
        if not z[0] <= m <= z[-1]:
            raise ValueError(f"soil node remap: target layer {ti} midpoint {m} m is outside the nodes {nodes}")
        k = min(int(np.searchsorted(z, m, side="right")) - 1, z.size - 2)
        f = (m - z[k]) / (z[k + 1] - z[k])
        w[ti, k], w[ti, k + 1] = 1.0 - f, f
    return w


def _es_hpa(t):
    import cupy as cp
    return cp.float32(SVP1 * 10.0) * cp.exp(cp.float32(SVP2) * (t - cp.float32(SVPT0)) / (t - cp.float32(SVP3)))


def q_from_dewpoint(td, p):
    import cupy as cp
    e = _es_hpa(td) * cp.float32(100.0)
    return (cp.float32(EPS) * e / (p - cp.float32(1.0 - EPS) * e)).astype(cp.float32)


def rh_from_q(q, t, p):
    import cupy as cp
    es = _es_hpa(t)
    qs = cp.float32(EPS) * es / (p / cp.float32(100.0) - es)
    q = cp.maximum(q, 0.0)
    w = q / (cp.float32(1.0) - q)
    return (cp.float32(100.0) * w / qs).astype(cp.float32)


def derive_sfc(rec: dict, sfc: dict, only=None) -> None:
    """Apply rec['sfc_derive'] in order, in place on sfc (dict of cupy [ny, nx]). only: output names wanted.
    The arrays may be NumPy (the CPU tests); the operations are the same elementwise float32 arithmetic."""
    first = next(iter(sfc.values()), None)
    if isinstance(first, np.ndarray):
        cp = np
    else:
        import cupy as cp
    for out, op, args in rec["sfc_derive"]:
        outs = out if isinstance(out, tuple) else (out,)
        if only is not None and not any(o in only for o in outs):
            continue
        # an optional record the file does not carry: its outputs come from rec["donor_fallback"] instead
        if all(o in rec.get("donor_fallback", {}) for o in outs) and \
                any(isinstance(a, str) and a not in sfc for a in args):
            continue
        if op.startswith("fill_"):
            # fill_from / fill_scale / fill_const: only where the file did not carry `out` (sfc_optional)
            if out in sfc:
                continue
            if op == "fill_from":
                sfc[out] = sfc[args[0]]
            elif op == "fill_scale":
                sfc[out] = (sfc[args[0]] * cp.float32(args[1])).astype(cp.float32)
            elif op == "fill_const":
                ref = next(iter(sfc.values()))
                sfc[out] = cp.full(ref.shape, args[0], dtype=cp.float32)
            else:
                raise ValueError(f"source_records: unknown sfc_derive op {op!r}")
            continue
        if op == "scale":
            sfc[out] = (sfc[args[0]] * cp.float32(args[1])).astype(cp.float32)
        elif op == "const":
            ref = next(iter(sfc.values()))
            sfc[out] = cp.full(ref.shape, args[0], dtype=cp.float32)
        elif op == "ratio":
            sfc[out] = (sfc[args[0]] / cp.maximum(sfc[args[1]], cp.float32(args[2]))).astype(cp.float32)
        elif op == "q_from_dewpoint":
            sfc[out] = q_from_dewpoint(sfc[args[0]], sfc[args[1]])
        elif op == "rh_from_q":
            sfc[out] = rh_from_q(sfc[args[0]], sfc[args[1]], sfc[args[2]])
        elif op in ("layer_remap", "node_interp"):
            srcs, sl, tl = args
            w = conservative_overlap_weights(sl, tl) if op == "layer_remap" else node_interp_weights(sl, tl)
            for ti, o in enumerate(outs):
                acc = None
                for si, s in enumerate(srcs):
                    if w[ti, si]:
                        term = np.float32(w[ti, si]) * sfc[s]
                        acc = term if acc is None else acc + term
                sfc[o] = acc.astype(np.float32)
        else:
            raise ValueError(f"source_records: unknown sfc_derive op {op!r}")


def derive_pl(rec: dict, pl, plev, order) -> None:
    """Apply rec['pl_derive'] in place on pl [nslot, nlev, ny, nx]; '@p' is the level pressure (Pa)."""
    import cupy as cp
    for slot, (op, args) in rec["pl_derive"].items():
        if op == "scale":
            pl[order.index(slot)] = pl[order.index(args[0])] * cp.float32(args[1])
            continue
        if op != "rh_from_q":
            raise ValueError(f"source_records: unknown pl_derive op {op!r}")
        q, t = pl[order.index(args[0])], pl[order.index(args[1])]
        p = cp.asarray(np.asarray(plev, np.float32))[:, None, None]
        pl[order.index(slot)] = rh_from_q(q, t, p)
