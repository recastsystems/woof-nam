"""Init sources: one table row per analysis source, staged into the state the init chain reads.

This is the NPS replacement's front half. NAM was retired on 2026-11-03 12Z, so (CONTRACT "Init after
NAM retirement"):
  native  the archived NAM native B-grid analysis (bgrd3d00), valid only for cycles before the
          retirement. Same grid and levels as the model: no interpolation.
  gfs     GFS 0.25 deg analysis (pgrb2.0p25 f000) for any date, interpolated onto the rotated B grid
          and the NMMB hybrid levels by the init-gfs-* units (nemsinterp port), gfs_start.py.
  parent  our own WOOF-NAM history (a run with history_handoff: true): a one-way nest on its own grid,
          parent_start.py (start) and parent_bocos.py (boundaries).
  ifs     ECMWF open data IFS HRES 0.1 deg (from 2026-10-20 12Z), the same isobaric path; ifs-0p25 for
          earlier cycles (from 2025-03-14 12Z; snow from the same-cycle GFS where the file has no sd/rsn).
          Record tables and conversions: source_records.py.
  era5    ECMWF ERA5 reanalysis 0.25 deg, hourly, 00/06/12/18Z starts from 1940-01-01 00Z to about 6 days before
          today, the same isobaric path: start and boundaries are ERA5 analyses (hindcasts and case studies). Read
          without an account from Google's public ARCO-ERA5 store by gpuwm's own ERA5 route (gpuwm.era5_arco, Rust
          Zarr bridge) and decoded by gpuwm's regular-forcing NetCDF reader (source_records container
          "regular-netcdf").
  rap     NOAA's RAP analysis (13 km, radar-cycled, North America, wrfprsf00 on RAP's rotated lat-lon grid), the same
          isobaric path with the source's own condensate and TKE (a warm start), GFS boundaries, and the same cycle's
          GFS analysis outside RAP's grid and above its 50 hPa top (row "fallback"); soil from RAP's RUC nodes or, with
          --soil-from gfs, from GFS. Grid placement: source_grids.py; records: source_records RAP.
  hrrr    NOAA's HRRR analysis (3 km CONUS, radar-cycled, wrfprsf00 on HRRR's Lambert conformal grid), the rap row's
          path: own condensate, GFS outside HRRR's CONUS grid and above 50 hPa, GFS boundaries, HRRR RUC soil or GFS.
  rrfs    NOAA's RRFS v1 analysis (3 km, radar-cycled), the hrrr row's path; atmosphere from prslev, surface and RUC
          soil from 2dfld. Public archive (noaa-rrfs-ops-pds) from 2026-08-12.
  aifs    ECMWF AIFS Single (CC BY 4.0): the IFS open-data analysis it starts from (ifs-0p25) as the start, AIFS
          leads every 6 h as the boundaries (start_from + boundary records). From 2025-03-14 12Z.
  aigfs   NOAA AIGFS (NOMADS, the last 10 days only): the GFS analysis it starts from as the start, AIGFS leads every
          6 h as the boundaries, surface pressure and 2 m humidity derived from the isobaric profile.
Adding a source is a row in SOURCES (file pattern, kind, grid), never a new code path.

Entry: stage(ctx) is what the chain stage init-grib2-decode runs when it is called with no file
(init_grib2_decode.run(ctx)). It reads ctx.init_source or ctx.cfg.raw["init_source"] (default
"native") and fills:
  ctx.st      TMP SPFH UGRD VGRD CLMR CICE RWMR SNMR PRES [LM, JM, IM] (top first), PSFC, HGT_M,
              SOILHGT, FIS [JM, IM]; for the native source also TCOND, FICE, FRAIN, RIME, TKE
              [LM, JM, IM] (the analysis' own condensate, fractions and TKE; init-native-atm reads them).
  ctx.st      PSHLTR, TSHLTR (2 m theta), QSHLTR, U10, V10 (grid-relative, H points), TH10, Q10 [JM, IM]
              from the source's 2 m / 10 m analysis (shelter_state), as NMMB reads them from its input.
  ctx.tables  raw_sfc (init-native-sfc input), soil_hydraulics (Noah STAS SOILPARM b, psi_s, theta_s),
              grid_source (receipt: source row, file, level order found).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "init-sources"
G_NMMB = np.float32(9.8060226)        # module_CONSTANTS G (FIS = HGT * G)
NCLASS = 30                           # soil_hydraulics table length (init-native-sfc)

# One row per source. "file" is relative to the case directory; {cyc} is the cycle hour.
SOURCES = {
    "native": {"kind": "nam_native", "file": "raw/nam/bgrd3d00.grib2", "gdt": 32769,
               "valid_before": "2026-11-03T12",
               "attribution": "NOAA/NCEP NAM analysis (public domain)",
               "licence": "public-domain (US government work, NOAA NOMADS / AWS NODD noaa-nam-pds)"},
    "gfs":    {"kind": "isobaric", "file": "raw/gfs/gfs.t{cyc:02d}z.pgrb2.0p25.f000", "gdt": 0,
               "grid": "gfs_0p25", "records": "gfs_0p25",
               # the archives the fetch reads (fetch_case.GFS_ERAS): NCEI 0.5 deg analyses from 2004-03-01 00Z
               # (analysis boundaries), NCAR GDEX 0.25 deg forecasts from 2015-01-15 00Z, noaa-gfs-bdp-pds from
               # 2021-01-01 00Z
               "valid_from": "2004-03-01T00",
               "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{cyc:02d}/atmos/"
                      "gfs.t{cyc:02d}z.pgrb2.0p25.f000",
               # Lateral boundaries (init-gfs-bocos, fetch_case row gfs-boco): forecast leads every
               # boco_interval_h of the cycle boco_age_h hours older than the start. 6 h is what
               # operational NAM uses (CONTRACT section 5); 0 h (the start cycle itself) is the knob for
               # past cases (NAMGPU_BOCO_AGE_H, configfile key boco_age_h, or --boco-age).
               "boco_age_h": 6, "boco_interval_h": 3,
               "boco_file": "raw/gfs/gfs.t{bcyc:02d}z.f{lead:03d}.grib2",
               # published_after_h: every lead the door asks for (f000-f120) is on the bucket by then (GFS posts
               # f000 about 3.5 h after the cycle and f384 by about 5 h); the door takes age 0 after it
               "boundary": {"age_h": 6, "interval_h": 3, "file": "raw/gfs/gfs.t{bcyc:02d}z.f{lead:03d}.grib2",
                            "published_after_h": 6},
               "attribution": "NOAA/NCEP GFS (public domain)", "licence": "public-domain"},
    # ECMWF open data IFS HRES (CC BY 4.0). Rows copied from gpuwm authorities/rw-wps-fetch-routes.v1.json
    # (routes ecmwf-open-data-0p1 measured 2026-10-24/29, ecmwf-open-data measured 2026-09-28/30): hosts, file
    # templates, cycles, publication lag and the ECMWF jsonl .index record selection. One file per lead here: the
    # selected records of the pl and sfc objects concatenated (fetch_case.plan_source), with the f000 surface
    # geopotential z appended to every later lead (published at step 0 only). No sea-ice concentration in
    # ECMWF open data: ICEC comes from the same cycle's GFS f000 (source_records.IFS donors). Boundaries are the
    # same cycle's leads every 3 h (age 0: the start and the boundaries are one forecast).
    "ifs":    {"kind": "isobaric", "file": "raw/ifs/ifs.t{cyc:02d}z.0p1.f000.grib2", "gdt": 0, "grid": "ifs_0p1",
               "records": "ifs", "valid_from": "2026-10-20T12", "before_use": "ifs-0p25",
               "boundary": {"age_h": 0, "interval_h": 3, "file": "raw/ifs/ifs.t{bcyc:02d}z.0p1.f{lead:03d}.grib2"},
               "attribution": "ECMWF open data, CC BY 4.0", "licence": "CC-BY-4.0",
               "fetch": {
                   "kind": "ecmwf-jsonl", "route": "gpuwm rw-wps-fetch-routes.v1.json ecmwf-open-data-0p1",
                   "hosts": ["https://storage.googleapis.com/ecmwf-open-data",        # 90 MB/s, 72 h rolling
                             "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",  # every cycle
                             "https://data.ecmwf.int/forecasts"],                       # publisher, 72 h
                   "eras": [{"from": "2026-10-20T12", "until": None, "cycles": [0, 6, 12, 18],
                             "files": ["{ymd}/{hh}z/ifs/0p1/oper/{ymd}{hh}0000-{lead}h-oper-fc-pl.grib2",
                                       "{ymd}/{hh}z/ifs/0p1/oper/{ymd}{hh}0000-{lead}h-oper-fc-sfc.grib2"]}],
                   "steps": [[144, 3]],
                   "publication_lag_h": {"0": 9.6, "12": 9.6, "6": 8.5, "18": 8.5},
                   "select": [{"levtype": "pl", "param": ["gh", "q", "t", "u", "v"]},
                              {"levtype": "sfc", "param": ["10u", "10v", "2d", "2t", "lsm", "skt", "sp", "z",
                                                           "sd", "rsn", "msl"]},
                              {"levtype": "sol", "param": ["sot", "vsw"]}],
                   "f000_only": [{"levtype": "sfc", "param": ["z"]}],
                   "donors": [{"file": "raw/ifs/gfs.t{hh}z.icec.f000.grib2",
                               "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                      "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":ICEC:surface:"},
                               {"file": "raw/ifs/gfs.t{hh}z.weasd.f000.grib2",
                                "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                       "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":WEASD:surface:"},
                               {"file": "raw/ifs/gfs.t{hh}z.snod.f000.grib2",
                                "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                       "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":SNOD:surface:"}]}},
    "ifs-0p25": {"kind": "isobaric", "file": "raw/ifs/ifs.t{cyc:02d}z.0p25.f000.grib2", "gdt": 0, "grid": "ifs_0p25",
                 # valid_from: the first cycle whose step-0 .index lists every record "select" reads, counting
                 # snow (sd, rsn) as optional: a file without them takes snow from the same-cycle GFS f000
                 # (source_records donor_fallback). ECMWF's own .index files: sol sot/vsw from 2024-11-15, sfc z by
                 # 2025-03-15 (present in the 2025-03-14 12Z file, which ran the IFS start), sd by 2025-12-01, rsn
                 # from 2026-02-11 12Z. Was 2024-03-18 12Z, whose files lack sot/vsw; earlier cycles take gfs.
                 "records": "ifs", "valid_from": "2025-03-14T12",
                 "boundary": {"age_h": 0, "interval_h": 3,
                              "file": "raw/ifs/ifs.t{bcyc:02d}z.0p25.f{lead:03d}.grib2"},
                 "attribution": "ECMWF open data, CC BY 4.0", "licence": "CC-BY-4.0",
                 "fetch": {
                     "kind": "ecmwf-jsonl", "route": "gpuwm rw-wps-fetch-routes.v1.json ecmwf-open-data",
                     "hosts": ["https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
                               "https://data.ecmwf.int/forecasts"],
                     # from the first cycle with every required record (valid_from); 06/18Z were 'scda' until
                     # 2026-05-12 06Z
                     "eras": [{"from": "2025-03-14T12", "until": None, "cycles": [0, 12],
                               "files": ["{ymd}/{hh}z/ifs/0p25/oper/{ymd}{hh}0000-{lead}h-oper-fc.grib2"]},
                              {"from": "2025-03-14T12", "until": "2026-05-12T06", "cycles": [6, 18],
                               "files": ["{ymd}/{hh}z/ifs/0p25/scda/{ymd}{hh}0000-{lead}h-scda-fc.grib2"]},
                              {"from": "2026-05-12T06", "until": None, "cycles": [6, 18],
                               "files": ["{ymd}/{hh}z/ifs/0p25/oper/{ymd}{hh}0000-{lead}h-oper-fc.grib2"]}],
                     "steps": [[90, 3]],
                     "publication_lag_h": {"0": 7.5, "12": 7.5, "6": 6.4, "18": 6.4},
                     "select": [{"levtype": "pl", "param": ["gh", "q", "t", "u", "v"]},
                                {"levtype": "sfc", "param": ["10u", "10v", "2d", "2t", "lsm", "skt", "sp", "z",
                                                             "sd", "rsn", "msl"]},
                                {"levtype": "sol", "param": ["sot", "vsw"]}],
                     "f000_only": [{"levtype": "sfc", "param": ["z"]}],
                     "donors": [{"file": "raw/ifs/gfs.t{hh}z.icec.f000.grib2",
                                 "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                        "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":ICEC:surface:"},
                               {"file": "raw/ifs/gfs.t{hh}z.weasd.f000.grib2",
                                "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                       "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":WEASD:surface:"},
                               {"file": "raw/ifs/gfs.t{hh}z.snod.f000.grib2",
                                "url": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/"
                                       "gfs.t{hh}z.pgrb2.0p25.f000", "idx_match": ":SNOD:surface:"}]}},
    # ECMWF ERA5 reanalysis (Copernicus C3S, CC BY 4.0: the CDS catalogue lists the ERA5 pressure- and single-level
    # sets under CC-BY-4.0, checked 2026-11-05). gpuwm's route reused as it is: provider "arco" of gpuwm's era5 source
    # (gpuwm/era5_arco.py ARCO_STORE and its field list; gpuwm/source_adapters.py _ERA5_ARCO_ARCHIVE: from 1940-01-01
    # 00Z, the store's own .zattrs names its last preliminary ERA5T day). One NetCDF per boundary time, written by
    # gpuwm.era5_arco.retrieve_era5_arco into its own folder (the name is gpuwm's era5_combined_name("arco")).
    # Every boundary time is an analysis (age 0): the start and the boundaries are one reanalysis, every 3 h (the NAM's
    # boco interval, tboco 10800 s) out of the hourly record. No donors: ERA5 carries sea ice, snow, soil and SST.
    "era5":   {"kind": "isobaric", "file": "raw/era5/t{cyc:02d}z.f000/era5-combined.nc", "gdt": 0, "grid": "era5_0p25",
               "records": "era5", "valid_from": "1940-01-01T00",
               "boundary": {"age_h": 0, "interval_h": 3, "file": "raw/era5/t{bcyc:02d}z.f{lead:03d}/era5-combined.nc"},
               "attribution": "Contains modified Copernicus Climate Change Service information (ERA5 reanalysis, "
                              "Hersbach et al. 2020, CC BY 4.0), read from Google's public ARCO-ERA5 copy",
               "licence": "CC-BY-4.0",
               "fetch": {
                   "kind": "arco-zarr", "route": "gpuwm era5 provider arco (gpuwm.era5_arco.retrieve_era5_arco)",
                   "provider": "arco",
                   "bounds_url": "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
                                 "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs",
                   "bounds_stop_keys": ["valid_time_stop_era5t", "valid_time_stop"],
                   # source rows kept around the domain (the decode's own margin, gfs_start.LAT_MARGIN), every longitude
                   "lat_margin_deg": 2.0,
                   # gpuwm Rust Zarr readers at once (one per boundary time, ~1.5 GB of host memory each on the NAM band)
                   "max_parallel": 4}},
    # GEFS v12 members (NCEP, public domain), 0.5 deg pgrb2a + pgrb2b: one file per member and lead holds the
    # selected records of both halves (fetch kind "ncep-idx-pair", GFS_INIT_VARS by .idx byte ranges). The records
    # carry the GFS dcp numbers, so the record table is GFS's (RH on 26 levels sets the level set; GEFS has no
    # ICMR/RWMR/SNMR, those slots stay 0). lead_start: the start may be a forecast lead of the boundary cycle,
    # so a start 3 h after a GEFS cycle (the SREF cycles 03/09/15/21Z) reads that cycle's f003 as its start and
    # f003, f006, ... as boundaries; age is start.hour mod 6 unless set (boco_age_h). Members: c00 (control) and
    # p01-p30, from the configfile key source_member or NAMGPU_SOURCE_MEMBER (woof-nam fetch/run --member).
    "gefs":   {"kind": "isobaric", "file": "raw/gefs/{mem}/gefs.t{bcyc:02d}z.f{lead:03d}.grib2", "gdt": 0,
               "grid": "gefs_0p5", "records": "gfs_0p25", "valid_from": "2020-09-23T12", "lead_start": True,
               "cycle_step_h": 6, "member_default": "c00", "members": ["c00"] + [f"p{k:02d}" for k in range(1, 31)],
               "boundary": {"interval_h": 3, "file": "raw/gefs/{mem}/gefs.t{bcyc:02d}z.f{lead:03d}.grib2"},
               "attribution": "NOAA/NCEP GEFS v12 (public domain)",
               "licence": "public-domain (US government work, AWS NODD noaa-gefs-pds)",
               "fetch": {"kind": "ncep-idx-pair", "host": "https://noaa-gefs-pds.s3.amazonaws.com",
                         "files": ["gefs.{ymd}/{hh}/atmos/pgrb2ap5/ge{mem}.t{hh}z.pgrb2a.0p50.f{lead:03d}",
                                   "gefs.{ymd}/{hh}/atmos/pgrb2bp5/ge{mem}.t{hh}z.pgrb2b.0p50.f{lead:03d}"],
                         "records": "gfs_init", "steps": [[240, 3], [384, 6]]}},
    # NOAA's RAP analysis (13 km, North America, hourly, cycled with radar reflectivity): the wrfprsf00 file on RAP's own
    # rotated lat-lon grid (NCEP template 3.32769, source_grids.py), the same isobaric path (records: source_records
    # RAP). RAP's grid stops about 17 NAM columns short of the NAM 12 km grid's west and east edges and its isobaric
    # levels stop at 50 hPa (the NAM top is 2 hPa), so the row names a global fallback: the same cycle's GFS analysis
    # gives the start outside RAP's grid, above 50 hPa (blended over 50-70 hPa and 8 RAP cells at its edge) and, with
    # --soil-from gfs, the soil. Boundaries: none of its own, so GFS's (boco_age_h 6 as the operational NAM), like the
    # native start: a RAP start differs from a GFS start only in the start.
    "rap":    {"kind": "isobaric", "file": "raw/rap/rap.t{cyc:02d}z.wrfprsf00.grib2", "gdt": 32769, "grid": "rap_rotated",
               "records": "rap", "valid_from": "2021-02-22T12",
               "fallback": {"source": "gfs", "edge_cells": 8, "top_blend_pa": 2000.0},
               "soil_from": ("source", "gfs", "gfs-moisture"),
               "attribution": "NOAA/NCEP Rapid Refresh (RAP) analysis (public domain)",
               "licence": "public-domain (US government work, AWS NODD noaa-rap-pds)"},
    # NOAA's HRRR analysis (3 km CONUS, hourly, cycled with radar reflectivity): wrfprsf00 on HRRR's Lambert conformal
    # grid (template 3.30, source_grids.LambertGrid), the RAP row's isobaric path and fallback (records: source_records
    # HRRR). HRRR covers CONUS only, so on the NAM 12 km grid the start is HRRR over CONUS and the same cycle's GFS
    # analysis outside it and above 50 hPa (blended over 8 HRRR cells = 24 km and 20 hPa), with GFS boundaries 6 h
    # older, as the native and rap starts. Soil: HRRR's nine RUC nodes at Noah's layer midpoints, or GFS's.
    "hrrr":   {"kind": "isobaric", "file": "raw/hrrr/hrrr.t{cyc:02d}z.wrfprsf00.grib2", "gdt": 30,
               "grid": "hrrr_lambert", "records": "hrrr",
               # AWS NODD noaa-hrrr-bdp-pds holds wrfprsf00 from 2014-07-30 18Z; the record table is HRRR v4's
               # (2020-12-02 on), measured on 2026-01-23 12Z. An older file missing a record is refused at decode.
               "valid_from": "2014-07-30T18",
               "fallback": {"source": "gfs", "edge_cells": 8, "top_blend_pa": 2000.0},
               "soil_from": ("source", "gfs", "gfs-moisture"),
               # default gfs-moisture (HRRR soil temperature, GFS soil moisture): 2026-01-23 12Z, 12 h, frozen NAM physics,
               # CONUS ASOS MAE at +6/+12 h: T2 2.5/3.0 F (own RUC soil 2.5/2.9, GFS soil 3.4/3.9, control 3.4/3.7), Td2
               # 3.8/4.7 F (own RUC soil 4.1/5.2: RUC moisture read at Noah depths is too dry). --soil-from source keeps it.
               "soil_default": "gfs-moisture",
               "attribution": "NOAA/NCEP High-Resolution Rapid Refresh (HRRR) analysis (public domain)",
               "licence": "public-domain (US government work, AWS NODD noaa-hrrr-bdp-pds)"},
    # NOAA's RRFS v1 analysis (FV3 LAM, 3 km, radar-cycled): the CONUS cut on HRRR's Lambert grid, atmosphere in
    # prslev, surface and RUC soil in 2dfld (source_records RRFS, sfc_file_from), the hrrr row's fallback and GFS
    # boundaries. Public archive noaa-rrfs-ops-pds from 2026-08-12 (no earlier cycles there), f000 at every 3 h cycle.
    "rrfs":   {"kind": "isobaric", "file": "raw/rrfs/rrfs.t{cyc:02d}z.prslev.3km.f000.conus.grib2", "gdt": 30,
               "grid": "rrfs_lambert", "records": "rrfs", "valid_from": "2026-08-12T00",
               "fallback": {"source": "gfs", "edge_cells": 8, "top_blend_pa": 2000.0},
               "soil_from": ("source", "gfs", "gfs-moisture"),
               "attribution": "NOAA/NCEP Rapid Refresh Forecast System (RRFS v1) analysis (public domain)",
               "licence": "public-domain (US government work, AWS NODD noaa-rrfs-ops-pds)"},
    # ECMWF AIFS Single (the machine-learned forecast ECMWF runs operationally, open data CC BY 4.0): route
    # copied from gpuwm authorities/rw-wps-fetch-routes.v1.json "aifs" (aifs-single 0p25 oper, steps 0-360 h every 6 h,
    # 00/06/12/18Z). The .index of 2026-01-23 12Z (measured 2026-11-08): pl q t u v w z on 13 levels 1000-50 hPa (z is
    # geopotential, not gh), sfc sp z(step 0) skt 2t 2d 10u 10v lsm msl, sol sot/vsw on 2 of IFS's 4 layers, no snow
    # depth, no sea ice. AIFS starts from the IFS analysis, so the start is that analysis (start_from ifs-0p25: all
    # four soil layers, snow, the GFS sea-ice donor) and the boundaries are AIFS leads every 6 h (tboco 21600 s).
    # start_fallback: IFS open data stops at 50 hPa and the NAM top is 2 hPa; extrapolated, the start was up to 35 K
    # too cold at 7.6 hPa (min 149 K against the GFS start's 184 K, 2026-11-09), so above 50 hPa (blended over
    # 20 hPa) the start is the same cycle's GFS analysis, as the rap/hrrr rows. Horizontally the IFS covers everything.
    "aifs":   {"kind": "isobaric", "start_from": "ifs-0p25",
               "start_fallback": {"source": "gfs", "edge_cells": 8, "top_blend_pa": 2000.0},
               "file": "raw/aifs/aifs.t{cyc:02d}z.0p25.f000.grib2",
               "gdt": 0, "grid": "ifs_0p25", "records": "aifs", "valid_from": "2025-03-14T12",
               "boundary": {"age_h": 0, "interval_h": 6, "records": "aifs",
                            "file": "raw/aifs/aifs.t{bcyc:02d}z.0p25.f{lead:03d}.grib2"},
               "attribution": "ECMWF AIFS Single open data (CC BY 4.0); start: ECMWF IFS open data analysis",
               "licence": "CC-BY-4.0",
               "fetch": {
                   "kind": "ecmwf-jsonl", "route": "gpuwm rw-wps-fetch-routes.v1.json aifs",
                   "hosts": ["https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
                             "https://data.ecmwf.int/forecasts"],
                   "eras": [{"from": "2025-03-14T12", "until": None, "cycles": [0, 6, 12, 18],
                             "files": ["{ymd}/{hh}z/aifs-single/0p25/oper/{ymd}{hh}0000-{lead}h-oper-fc.grib2"]}],
                   "steps": [[360, 6]],
                   "publication_lag_h": {"0": 8.0, "6": 8.0, "12": 8.0, "18": 8.0},
                   "select": [{"levtype": "pl", "param": ["z", "q", "t", "u", "v"]},
                              {"levtype": "sfc", "param": ["10u", "10v", "2d", "2t", "lsm", "skt", "sp", "z"]}],
                   "f000_only": [{"levtype": "sfc", "param": ["z"]}]}},
    # NOAA AIGFS v1.1 (NCEP's operational GraphCast-based forecast, US government work): route copied from gpuwm
    # rw-wps-fetch-routes.v1.json "aigfs": NOMADS only (the S3 graphcastgfs bucket's same-named files are the
    # experimental EAGLE stream, a different product), 10-day rolling window, pres + sfc files every 6 h to 384 h,
    # whole files (no grib filter). Records (.idx of 2026-11-08 00Z): HGT TMP SPFH UGRD VGRD VVEL on 13 levels
    # 1000-50 hPa; surface only 2 m T, 10 m winds and MSLP. AIGFS starts from NCEP's analysis, so the start is the
    # same cycle's GFS analysis (start_from gfs) and the boundaries are AIGFS leads every 6 h, their surface pressure
    # and 2 m humidity derived from the isobaric profile at the model terrain (source_records AIGFS).
    "aigfs":  {"kind": "isobaric", "start_from": "gfs", "file": "raw/aigfs/aigfs.t{cyc:02d}z.f000.grib2", "gdt": 0,
               "grid": "gfs_0p25", "records": "aigfs", "valid_from": "2025-12-01T00", "retention_h": 240,
               "boundary": {"age_h": 0, "interval_h": 6, "records": "aigfs",
                            "file": "raw/aigfs/aigfs.t{bcyc:02d}z.f{lead:03d}.grib2"},
               "attribution": "NOAA/NCEP AIGFS (public domain); start: NOAA/NCEP GFS analysis",
               "licence": "public-domain (US government work, NOAA NOMADS)",
               "fetch": {
                   "kind": "whole-files", "route": "gpuwm rw-wps-fetch-routes.v1.json aigfs",
                   "hosts": ["https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod"],
                   "eras": [{"from": "2025-12-01T00", "until": None, "cycles": [0, 6, 12, 18],
                             "files": ["aigfs.{ymd}/{hh}/model/atmos/grib2/aigfs.t{hh}z.pres.f{lead:03d}.grib2",
                                       "aigfs.{ymd}/{hh}/model/atmos/grib2/aigfs.t{hh}z.sfc.f{lead:03d}.grib2"]}],
                   "steps": [[384, 6]],
                   "publication_lag_h": {"0": 6.0, "6": 6.0, "12": 6.0, "18": 6.0}}},
    # One-way nest from our own WOOF-NAM history (offline, like WRF ndown): the parent run directory comes from
    # --parent-run / configfile parent_run / NAMGPU_PARENT_RUN; the run must have written the hand-off set
    # (configfile history_handoff: true). Start: parent_start.py; boundaries: parent_bocos.py.
    # Its licence is the parent run's own start source: the history's DATA_LICENCES attribute (nest.cli.parent_credit).
    "parent": {"kind": "nam_history", "file": "{parent_run}/wrfout_d01_{valid}", "grid": "parent_rotated",
               "attribution": "our own WOOF-NAM run (its start source's credit is read from its history)",
               "licence": "inherits the parent run's start source (history attribute DATA_LICENCES)"},
}

# NAM native analysis records (NCEP's GRIB2 file of the NMMB analysis, GDT 32769, resolution and component
# flags 56 = 32 + 16 + 8: i and j increments given, u and v relative to the grid; WMO-No. 306 flag table 3.3),
# (discipline, category, parameter) on hybrid levels.
NATIVE_3D = {
    "PRES": (0, 3, 0), "HGT3D": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0),
    "UGRD": (0, 2, 2), "VGRD": (0, 2, 3), "CLMR": (0, 1, 22), "CICE": (0, 6, 0),
    "RWMR": (0, 1, 24), "SNMR": (0, 1, 25),
    # NCEP local tables: 0-6-195 TCOND, 0-6-199 FICE, 0-1-202 FRAIN, 0-1-203 RIME, 0-19-11 TKE
    "TCOND": (0, 6, 195), "FICE": (0, 6, 199), "FRAIN": (0, 1, 202), "RIME": (0, 1, 203),
    "TKE": (0, 19, 11),
    "DZDT": (0, 2, 9),        # vertical velocity (m/s): the analysis' W_TOT = W + BARO (BARO, mm/s, not subtracted)
    "CFRAC": (0, 6, 1),       # cloud fraction (%) on the hybrid levels: the NAM's CLDFRA (drv-native-diag-restore)
}
# Staged inputs dropped from the state once init-native-atm / init-vcoord-recover have read them.
STAGED_3D = ("PRES", "TMP", "SPFH", "UGRD", "VGRD", "CLMR", "CICE", "RWMR", "SNMR", "ZERO3")


def release_staged(ctx, keep=("PRES",)):
    """Drop the staged decode names (aliases of T/Q/U/V and of the zero species array, PRES) once the
    state is built; the model arrays stay in the state under their own names."""
    for n in STAGED_3D:
        if n not in keep:
            ctx.st.fields.pop(n, None)


# Surface records: name -> (d, c, p, level_type, level_value, second_level_value or None)
NATIVE_SFC = {
    "PSFC": (0, 3, 0, "1", "0", None),
    "HGT_SFC": (0, 3, 5, "1", "0", None),
    "TMP_SFC": (0, 0, 0, "1", "0", None),
    "TSOIL1": (2, 0, 2, "106", "0", "0.1"), "TSOIL2": (2, 0, 2, "106", "0.1", "0.4"),
    "TSOIL3": (2, 0, 2, "106", "0.4", "1"), "TSOIL4": (2, 0, 2, "106", "1", "2"),
    "SOILW1": (2, 0, 192, "106", "0", "0.1"), "SOILW2": (2, 0, 192, "106", "0.1", "0.4"),
    "SOILW3": (2, 0, 192, "106", "0.4", "1"), "SOILW4": (2, 0, 192, "106", "1", "2"),
    "TSOIL_3M": (2, 0, 2, "106", "3", None),
    "CNWAT": (2, 0, 196, "1", "0", None),
    "WEASD": (0, 1, 13, "1", "0", None),          # instantaneous (pdt 0); the pdt-8 twin is an accumulation
    "SNOD": (0, 1, 11, "1", "0", None),
    "SNOWC": (0, 1, 201, "1", "0", None),
    "SFCR": (2, 0, 1, "1", "0", None),
    "VEG": (2, 0, 4, "1", "0", None),
    "VGTYP": (2, 0, 198, "1", "0", None),
    "SOTYP": (2, 3, 0, "1", "0", None),
    "LAND": (2, 0, 0, "1", "0", None),
    "ICEC": (10, 2, 0, "1", "0", None),
    "WTMP": (10, 3, 0, "1", "0", None),
    "ALBDO": (0, 19, 1, "1", "0", None),
}



def albase_from_analysis(albdo_pct, snowc, weasd, land):
    """The snow-free base albedo (NMMB ALBASE, a fraction) from an archived NAM analysis' own surface albedo.

    NAM's Noah sets ALBEDO = ALBASE where there is no snow, so on snow-free land the analysis' ALBDO record is the
    NAM's ALBASE. In NCEP's awphys files of the 2024-11-05 12Z cycle the albedo of 76,057 snow-free land cells of
    grid 218 is the same at +9 h as at +0 h (mean abs difference 0.000 points). The analysis file carries no ALBASE
    record, and before this the whole field was init-static-missing's vegetation-class fill (the mid-range of a
    WRF table's ALBEDOMIN/ALBEDOMAX): from the first land step the run's albedo sat 3.2 points above the NAM's on
    those cells (mean abs 4.0, 99th percentile 12, pattern correlation 0.60; 5-15 points over the Mojave and the
    Colorado Plateau, where the 2 m temperature ran 1-3 F cool at 21Z).

    Returns ALBDO / 100 on land where the analysis has no snow (snow cover and snow water both zero) and the
    albedo is a valid percentage; 0 (missing) elsewhere: snow-covered land, where the record is the snow blend,
    and water. init-static-missing fills the missing cells from the table, as it did for every cell before.
    numpy or cupy arrays; snowc in percent or as a fraction."""
    ok = (land > 0.5) & (snowc <= 0) & (weasd <= 0) & (albdo_pct > 0) & (albdo_pct <= 100)
    return (albdo_pct * 0.01 * ok).astype("float32")



# NCEP's native surface file of the same cycle (nam.tHHz.bgrdsf00.tm00.grib2 on noaa-nam-pds, the same NMMB H grid as
# bgrd3d00; fetch_case row nam-bgrdsf00-land fetches these records only) carries three land states the analysis
# bgrd3d00 does not:
#   SOILL  the liquid part of the soil water, four Noah layers (2-3-192). The NAM's cycled soil holds ice that no
#          formula of temperature and total water gives back: on 2024-11-05 12Z the top layer has ice at 115,396 of
#          217,650 land points, 17,510 of them above 0 C, and the frozen-fraction split of init-native-sfc missed
#          the NAM's liquid water by 0.028 (volume fraction, mean abs, all land; 0.052 where the NAM has ice). With
#          more liquid and less ice than the NAM, the run warmed the frozen ground too fast: soil 0-10 cm +3 to +4 F
#          and 2 m +1 to +2 F at 21Z in a band from Arizona to Manitoba.
#   SNFALB the snow-free base albedo (0-19-193, percent): the NAM's ALBASE on every land point, under snow too. On
#          snow-free land it equals the analysis albedo (mean abs 0.0002 points); under snow albase_from_analysis
#          has nothing and the vegetation-table fill missed it by 7.1 points (mean abs).
#   MXSALB the maximum snow albedo (0-19-192, percent): the NAM's MXSNAL. The table fill missed it by 11.7 points
#          (mean abs, land with snow).
# With both albedos, NCEP's own 9 h albedo over snow is ALBASE + SC (MXSNAL - ALBASE) with SC = min(SNFRAC(SNO,
# 4 SNUP), 0.98), the rule phys_noah_alcalc already applies, to 0.08 points (mean abs, 102,112 snowy land points).
NATIVE_LAND = {
    "SOILL1": (2, 3, 192, "106", "0", "0.1"), "SOILL2": (2, 3, 192, "106", "0.1", "0.4"),
    "SOILL3": (2, 3, 192, "106", "0.4", "1"), "SOILL4": (2, 3, 192, "106", "1", "2"),
    "SNFALB": (0, 19, 193, "1", "0", None),
    "MXSALB": (0, 19, 192, "1", "0", None),
}
# the fetch row's subset first, then the whole file (the referee row nam-bgrdsf, lead 0) when a case has that instead
NATIVE_LAND_FILES = ("raw/nam/bgrdsf00.land.grib2", "raw/nam/bgrdsf00.grib2")


def native_land_file(case_dir):
    """The case's NCEP native surface file of the start (NATIVE_LAND_FILES), or None."""
    for rel in NATIVE_LAND_FILES:
        p = Path(case_dir) / rel
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def land_from_surface_file(rec, albase, land):
    """The NAM's own land states from the native surface file's records (rec: NATIVE_LAND name -> [JM, IM] array,
    any subset), merged over what the start has without them.
    albase: the base albedo the start has without the file (albase_from_analysis, 0 = missing), a fraction.
    Returns (static_extra, sh2o, lacking): static_extra {"ALBASE", and "MXSNAL" when the file has it} for
    init-static-missing (0 = missing: its table fill); sh2o [4, JM, IM] (volume fraction) for init-native-sfc's
    sfc_override (capped there at the total), or None; lacking: the states left to the fallbacks. numpy or cupy."""
    if type(albase).__module__.split(".")[0] == "cupy":
        import cupy as xp  # noqa: PLC0415
    else:
        xp = np
    island = land > 0.5
    extra, lacking = {}, []
    if "SNFALB" in rec:
        a = rec["SNFALB"]
        extra["ALBASE"] = xp.where(island & (a > 0) & (a <= 100), a * 0.01, albase).astype(xp.float32)
    else:
        extra["ALBASE"] = albase
        lacking.append("base albedo under snow")
    if "MXSALB" in rec:
        a = rec["MXSALB"]
        extra["MXSNAL"] = xp.where(island & (a > 0) & (a <= 100), a * 0.01, 0.0).astype(xp.float32)
    else:
        lacking.append("maximum snow albedo")
    if all(f"SOILL{k}" in rec for k in range(1, 5)):
        sh2o = xp.ascontiguousarray(xp.stack([rec[f"SOILL{k}"] for k in range(1, 5)]).astype(xp.float32))
    else:
        sh2o = None
        lacking.append("liquid soil water")
    return extra, sh2o, lacking


def decode_native_land(ctx, path, tmpdir):
    """Decode the NATIVE_LAND records a native surface file carries (init-grib2-decode, on the model grid)."""
    from namgpu.init import init_grib2_decode as dec
    inv = dec.inventory(path)
    keys = [k for k, spec in NATIVE_LAND.items() if any(_selsfc(spec)(r) for r in inv)]
    if not keys:
        return {}
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    return dec.run(ctx, path, {k: _selsfc(NATIVE_LAND[k]) for k in keys}, missing_value=0.0, tmpdir=tmpdir,
                   expect_shape=(im, jm))


# The analysis' 0 h diagnostics (optional; drv-native-diag-restore puts them in the state before the time loop of a
# no-filter start, so the 0 h history and products carry the NAM's own values instead of zeros): boundary-layer
# height, friction velocity, the instantaneous surface fluxes (template 4.0; the 4.8 twins are averages), total
# soil moisture 0-2 m and moisture availability 0-1 m. The hybrid-level cloud fraction is NATIVE_3D CFRAC.
# The convective state of the analysis: the convective precipitation rate (CPRAT, kg m-2 s-1) and the pressures of the
# convective cloud bottom and top (level types 242 and 243, Pa; -50000 where the column has no convective cloud). In
# nam.t12z.bgrd3d00 of 2024-11-05 both pressures are a mid-layer pressure of the column's hybrid levels at every
# point that has one (to 2.4 Pa, the record's packing), so each names a model layer; the cloud-base pressure record
# is the higher of the convective and the grid-scale cloud bottom pressures at every point.
NATIVE_DIAG = {
    "HPBL": (0, 3, 196, "1", "0", None), "FRICV": (0, 2, 197, "1", "0", None),
    "SHTFL": (0, 0, 11, "1", "0", None), "LHTFL": (0, 0, 10, "1", "0", None),
    "DSWRF": (0, 4, 192, "1", "0", None), "USWRF": (0, 4, 193, "1", "0", None),
    "DLWRF": (0, 5, 192, "1", "0", None), "ULWRF": (0, 5, 193, "1", "0", None),
    "SOILM": (2, 0, 3, "106", "0", "2"), "MSTAV": (2, 0, 194, "106", "0", "1"),
    "CPRAT": (0, 1, 196, "1", "0", None),
    "PCNVBOT": (0, 3, 0, "242", "0", None), "PCNVTOP": (0, 3, 0, "243", "0", None),
}


def native_diag_keys(inv) -> list:
    """The NATIVE_DIAG names whose record the analysis inventory carries (instantaneous ones only)."""
    return [k for k, spec in NATIVE_DIAG.items() if any(_selsfc(spec)(r) for r in inv)]


# 2 m / 10 m analysis records. NMMB reads tshltr, qshltr, pshltr, u10, v10, th10, q10 from its input file
# (module_INIT_READ_NEMSIO.F90:802-820, 2178-2217, 2589-2694); without them they start at the
# SOLVER_INITIALIZE defaults (TSHLTR 273, QSHLTR 0, U10 0; module_SOLVER_GRID_COMP.F90:1105-1118) and the
# 0 h history shows those defaults. Optional: a file without a record keeps the derived fallback.
NATIVE_SHELTER = {
    "T2M": (0, 0, 0, "103", "2", None), "Q2M": (0, 1, 0, "103", "2", None),
    "P2M": (0, 3, 0, "103", "2", None),
    "U10M": (0, 2, 2, "103", "10", None), "V10M": (0, 2, 3, "103", "10", None),
    "TH10M": (0, 0, 2, "103", "10", None), "Q10M": (0, 1, 0, "103", "10", None),
}


def shelter_state(ctx, t2, q2, u10, v10, p2=None, th10=None, q10=None):
    """Fill ctx.st PSHLTR, TSHLTR (2 m potential temperature), QSHLTR, U10, V10, TH10 and Q10 [JM, IM] at H points from analysis 2 m and 10 m fields (t2 K, q2 kg/kg, grid-relative u10/v10 m/s, optional p2 Pa, th10, q10).

    PSHLTR: the analysis' 2 m pressure p2 where it is positive; elsewhere, and when the source has none, the
    hypsometric pressure 2 m above PSFC (poisson.pressure_2m). TSHLTR = t2 (1e5 / PSHLTR) ** KAPPA, the inverse of
    the history's T2 = TSHLTR (PSHLTR / 1e5) ** KAPPA, so the 0 h history T2 is the analysis t2. QSHLTR = q2
    (specific humidity). TH10 and Q10: the analysis' 10 m potential temperature and specific humidity; a source
    without them gets the 2 m values (TSHLTR, q2). Arrays may be numpy or cupy; the state's own kind is written."""
    from namgpu.io import poisson
    st = ctx.st
    s2 = (int(ctx.grid.jm), int(ctx.grid.im))
    xp = poisson._xp(t2)
    t2 = xp.asarray(t2, dtype=xp.float32)
    if "PSFC" not in st.fields:
        raise RuntimeError(f"{UNIT}: shelter_state needs PSFC in the state before the 2 m pressure can be set")
    p_hyps = poisson.pressure_2m(xp.asarray(st["PSFC"]), t2)
    if p2 is None:
        pshltr = p_hyps
    else:
        p2 = xp.asarray(p2, dtype=xp.float32)
        # a missing value decodes as 0, which would make TSHLTR infinite
        pshltr = xp.where(p2 > 0, p2, p_hyps).astype(xp.float32)
    tshltr = poisson.theta(t2, pshltr)
    q2 = xp.asarray(q2, dtype=xp.float32)
    fill = {
        "PSHLTR": pshltr, "TSHLTR": tshltr, "QSHLTR": q2,
        "U10": u10, "V10": v10,
        "TH10": tshltr if th10 is None else th10,
        "Q10": q2 if q10 is None else q10,
    }
    for name, val in fill.items():
        st.add(name, s2)[...] = xp.asarray(val, dtype=xp.float32)


def source_row(ctx) -> tuple[str, dict]:
    name = getattr(ctx, "init_source", None) or ctx.cfg.raw.get("init_source", "native")
    name = str(name).lower()
    if name not in SOURCES:
        raise ValueError(f"{UNIT}: init source {name!r} is not a SOURCES row ({sorted(SOURCES)})")
    return name, SOURCES[name]


def boco_age_h(row: dict, cfg=None, override=None, start=None) -> int:
    """Hours between the start and the cycle the boundaries come from, for one SOURCES row. First set
    wins: override (a CLI value), NAMGPU_BOCO_AGE_H, configfile key boco_age_h, the row's boco_age_h.
    A lead_start row (gefs) takes start.hour mod cycle_step_h when nothing is set (the start is that lead of
    the newest cycle at or before it), and any set age must land on one of its cycles."""
    import os
    step = int(row.get("cycle_step_h", 6))
    for v in (override, os.environ.get("NAMGPU_BOCO_AGE_H"),
              (cfg.raw.get("boco_age_h") if cfg is not None else None), row.get("boco_age_h"),
              (row.get("boundary") or {}).get("age_h")):
        if v is not None and str(v).strip() != "":
            h = int(v)
            if row.get("lead_start"):
                if h < 0 or (start is not None and (start.hour - h) % step):
                    raise ValueError(f"{UNIT}: boundary age {h} h from a {start:%HZ} start is not a {step}-hourly "
                                     "cycle of this source")
                return h
            if h < 0 or h % 6:
                raise ValueError(f"{UNIT}: boundary age {h} h: must be a non-negative multiple of 6 (GFS and IFS cycles)")
            return h
    if row.get("lead_start"):
        if start is None:
            raise ValueError(f"{UNIT}: a lead_start source needs the start time to find its cycle")
        return int(start.hour) % step
    raise ValueError(f"{UNIT}: source row has no boco_age_h")


def source_member(row: dict, cfg=None) -> str:
    """The source member a row with members reads (gefs: c00, p01-p30): NAMGPU_SOURCE_MEMBER, else the
    configfile key source_member, else the row's member_default. '' for a row without members."""
    import os
    if not row.get("members"):
        return ""
    v = os.environ.get("NAMGPU_SOURCE_MEMBER") or (cfg.raw.get("source_member") if cfg is not None else None) \
        or row.get("member_default")
    v = str(v).strip()
    if v not in row["members"]:
        raise ValueError(f"{UNIT}: source member {v!r} is not one of this source's members "
                         f"({row['members'][0]} .. {row['members'][-1]})")
    return v


def file_keys(row: dict, start, cfg=None) -> dict:
    """The template keys of a row's file names for a run starting at `start`: cyc (start hour), bcyc (the
    boundary cycle's hour), lead (the start's lead in that cycle: 0 unless the row is lead_start) and mem."""
    from datetime import timedelta
    if start is None:
        return {"cyc": 0, "bcyc": 0, "lead": 0, "mem": source_member(row, cfg)}
    age = boco_age_h(row, cfg, start=start) if row.get("lead_start") else 0
    return {"cyc": int(start.hour), "bcyc": int((start - timedelta(hours=age)).hour), "lead": age,
            "mem": source_member(row, cfg)}


def source_path(ctx, row) -> Path:
    return Path(ctx.case_dir) / row["file"].format(**file_keys(row, ctx.start, getattr(ctx, "cfg", None)))


def soil_hydraulics():
    """Noah STAS SOILPARM b, psi_s (m), theta_s for init-native-sfc, padded to NCLASS (host)."""
    from namgpu.init import init_static_missing as ism
    noah = ism._noah()
    t = noah.load_tables(mminlu="MODIFIED_IGBP_MODIS_NOAH", mminsl="STAS", tbl_dir=ism.TBL_DIR)
    out = {}
    for key, arr in (("bexp", t.bb), ("psisat", t.satpsi), ("smcmax", t.maxsmc)):
        a = np.asarray(arr, np.float32).ravel()
        pad = np.full(NCLASS, a[-1] if a.size else 0.0, np.float32)
        pad[: min(a.size, NCLASS)] = a[:NCLASS]
        out[key] = pad
    return out


def _sel3(dcp, level):
    def sel(r):
        return (r["dcp"] == dcp and str(r["level_type"]) == "105"
                and int(float(r["level_value"])) == level
                and str(r["second_level_type"]) == "255")      # not the layer-mean records
    return sel


def _selsfc(spec):
    d, c, p, lt, lv, lv2 = spec

    def sel(r):
        if r["dcp"] != (d, c, p) or str(r["level_type"]) != lt:
            return False
        if abs(float(r["level_value"]) - float(lv)) > 1e-6:
            return False
        if lv2 is not None:
            try:
                if abs(float(r["second_level_value"]) - float(lv2)) > 1e-6:
                    return False
            except ValueError:
                return False
        # instantaneous records: template 4.0, or 4.1 (one member of an ensemble: GEFS writes every record so)
        return str(r["pdt"]) in ("0", "1", "-")
    return sel


def stage(ctx):
    name, row = source_row(ctx)
    if row.get("start_from"):
        # aifs / aigfs: the start is the analysis the AI forecast itself starts from (another row, staged as that
        # row's own start), the boundaries are the AI forecast's leads (gfs_bocos.boundary_row reads this row)
        sname = row["start_from"]
        from namgpu.init import gfs_start
        gfs_start.check_valid(ctx, name, row)
        srow = dict(SOURCES[sname])
        if row.get("start_fallback"):
            srow["fallback"] = row["start_fallback"]
        info = gfs_start.stage(ctx, sname, srow)
        info.update(start_from=sname, boundaries_from=name, boundary_attribution=row.get("attribution"),
                    boundary_licence=row.get("licence"))
        print(f"{UNIT}: {name}: start from the {sname} analysis, boundaries from {name} leads every "
              f"{row['boundary']['interval_h']} h", flush=True)
        return info
    if row["kind"] == "nam_native":
        return stage_native(ctx, row)
    if row["kind"] == "nam_history":
        from namgpu.init import parent_start
        return parent_start.stage(ctx, name, row)
    if row["gdt"] not in (0, 32769, 30):
        raise NotImplementedError(
            f"{UNIT}: source {name!r} is on GRIB2 GDT {row['gdt']} (Lambert conformal); init-grib2-decode "
            "places only GDT 0 and 32769 (source_grids.py: rotated lat-lon), so its fields would land in the wrong "
            "cells. Not done: a Lambert source grid in source_grids.py.")
    from namgpu.init import gfs_start
    return gfs_start.stage(ctx, name, row)


def decode_native(ctx, path, names3, sfc, lm, tmpdir, sink, top_first=None):
    """Decode hybrid fields one name at a time (top first, written through sink(name, [LM,JM,IM]))
    and the surface fields (returned). One name in flight keeps the device peak at one 3-D field."""
    import cupy as cp
    from namgpu.init import init_grib2_decode as dec

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    # NAMGPU_ROW_WINDOW (config.row_window): the native file is on the parent grid; decode it there and keep the
    # band's rows (no interpolation: the band is the same grid).
    win = ctx.cfg.raw.get("row_window") if getattr(ctx, "cfg", None) is not None else None
    cut = (lambda x: x) if not win else (lambda x: x[..., win[0] - 1:win[1], :])
    kw = dict(missing_value=0.0, tmpdir=tmpdir,
              expect_shape=(im, int(ctx.cfg.raw["jm_full"]) if win else jm))
    p1 = plm = None
    if top_first is None:
        # NCEP's GRIB2 output numbers hybrid levels from one end; NMMB wants l = 1 at the model top. Decide from PRES.
        pr = dec.run(ctx, path, {"P1": _sel3(NATIVE_3D["PRES"], 1),
                                 "PL": _sel3(NATIVE_3D["PRES"], lm)}, **kw)
        p1, plm = float(cp.mean(cut(pr["P1"]))), float(cp.mean(cut(pr["PL"])))
        top_first = p1 < plm
        del pr
    order = list(range(1, lm + 1)) if top_first else list(range(lm, 0, -1))
    for n in names3:
        got = dec.run(ctx, path, {f"{n}@{lev}": _sel3(NATIVE_3D[n], lev) for lev in range(1, lm + 1)}, **kw)
        a = cp.empty((lm, jm, im), dtype=cp.float32)
        for l0, lev in enumerate(order):
            a[l0] = cut(got.pop(f"{n}@{lev}"))
        sink(n, a)
        del a, got
    got = dec.run(ctx, path, {n: _selsfc(NATIVE_SFC[n]) for n in sfc}, **kw) if sfc else {}
    if win:
        got = {k: cp.ascontiguousarray(cut(v)) for k, v in got.items()}
    return got, {"top_first_in_file": top_first, "p_level1_mean": p1, "p_levelLM_mean": plm}


# Decode destinations. The native file is on the model grid and levels, so the dynamics fields are
# decoded straight into the state arrays init-native-atm writes (it reads and writes the same index,
# T = TMP etc., so the alias is safe). That keeps the init peak at the state plus about two 3-D
# fields, which is what lets init fit beside the 10.6 GiB state on a 16 GB card.
ALIAS_INTO = {"TMP": "T", "SPFH": "Q", "UGRD": "U", "VGRD": "V"}
SPECIES = ("CLMR", "CICE", "RWMR", "SNMR")
WARM_INTO = {"TCOND": "CW", "FICE": "F_ICE", "FRAIN": "F_RAIN", "RIME": "F_RIMEF", "TKE": "Q2", "DZDT": "W"}
# The archived NAM analysis' TKE record (0-19-11 on the hybrid levels) is the NMMB q2 as it is, not q2 / 2: NAM's
# post writes q2 under the TKE name (the awphys table says so for the isobaric TKE rows, which write history
# TKE_PBL x 2). Its free-air value in bgrd3d00 is 0.02, the EPSQ2 floor of q2 itself (2024-11-05 12Z: level
# medians 0.02, minimum 0.02). Reading it as a TKE and doubling it started every native run with twice the
# NAM's q2: the +0 h awphys TKE was 2.00 times NCEP's at every level (1000 hPa mean 1.085 against 0.542; free air
# 0.04 against 0.02) and equal to it from +1 h on. A RAP start's TKE record is a real TKE (gfs_start.warm_state
# keeps its 2 x).
NATIVE_TKE_TO_Q2 = 1.0


def decode_warm(ctx):
    """init-native-atm warm start: decode the analysis' TCOND, FICE, FRAIN, RIME, TKE, DZDT straight into
    CW, F_ICE, F_RAIN, F_RIMEF, Q2, W (one field in flight). Returns the names written, also kept in
    ctx.tables["native_warm_kept"] so the later cold-start fills leave them alone (init-derive-state: Q2, W;
    phys-fer-init: F_ICE, F_RAIN, F_RIMEF), as the operational NAM forecast read them from its input."""
    import cupy as cp
    w = ctx.tables.get("native_warm")
    if not w or not w["names"]:
        return []
    st, lm = ctx.st, int(ctx.grid.lm)
    done = []

    def sink(n, a):
        dst = st[WARM_INTO[n]]
        if n == "TCOND":
            dst[...] = cp.maximum(a, 0.0)
        elif n in ("FICE", "FRAIN"):
            dst[...] = cp.clip(a, 0.0, 1.0)
        elif n == "RIME":
            dst[...] = cp.maximum(a, 1.0)
        elif n == "DZDT":
            dst[...] = a
        else:                                   # the file's TKE record is the NMMB q2 itself; EPSQ2 floor 0.02
            dst[...] = cp.maximum(NATIVE_TKE_TO_Q2 * a, cp.float32(0.02))
        done.append(WARM_INTO[n])

    decode_native(ctx, Path(w["file"]), w["names"], [], lm, w["tmpdir"], sink, top_first=w["top_first"])
    ctx.tables["native_warm_kept"] = tuple(done)
    return done


def stage_native(ctx, row):
    import cupy as cp
    from namgpu.init import init_grib2_decode as dec

    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    path = source_path(ctx, row)
    if ctx.start is not None and ctx.start.strftime("%Y-%m-%dT%H") >= row["valid_before"]:
        raise ValueError(f"{UNIT}: no NAM native analysis exists for {ctx.start:%Y-%m-%d %HZ} (NAM retired "
                         "2026-11-03 12Z); use init_source gfs (or hrrr/rrfs once done)")
    if not path.is_file():
        raise FileNotFoundError(f"{UNIT}: {path} missing (run init-fetch-case)")
    inv = dec.inventory(path)
    have = {(r["dcp"], str(r["level_type"])) for r in inv}
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    Path(tmpdir).mkdir(exist_ok=True)

    names3 = ["PRES", "TMP", "SPFH", "UGRD", "VGRD"]
    if cfg.do_clouds:
        names3 += list(SPECIES)
    else:
        zero = st.add("ZERO3", (lm, jm, im))     # the species are not read when do_clouds is off
        for n in SPECIES:
            st.fields[n] = zero

    def sink(n, a):
        dst = ALIAS_INTO.get(n, n)
        st.add(dst, (lm, jm, im))[...] = a
        st.fields[n] = st[dst]

    got, order = decode_native(ctx, path, names3, list(NATIVE_SFC), lm, tmpdir, sink)
    shelter_keys = [k for k, spec in NATIVE_SHELTER.items() if any(_selsfc(spec)(r) for r in inv)]
    if shelter_keys:
        sh = dec.run(ctx, path, {k: _selsfc(NATIVE_SHELTER[k]) for k in shelter_keys},
                     missing_value=0.0, tmpdir=tmpdir, expect_shape=(im, jm))
    else:
        sh = {}
    diag_keys = native_diag_keys(inv)
    diag = dec.run(ctx, path, {k: _selsfc(NATIVE_DIAG[k]) for k in diag_keys},
                   missing_value=0.0, tmpdir=tmpdir, expect_shape=(im, jm)) if diag_keys else {}
    ctx.tables["native_diag"] = {"file": str(path), "tmpdir": tmpdir, "top_first": order["top_first_in_file"],
                                 "sfc": diag, "cldfra": (NATIVE_3D["CFRAC"], "105") in have}
    warm = [n for n in WARM_INTO if (NATIVE_3D[n], "105") in have]
    ctx.tables["native_warm"] = {"file": str(path), "names": warm, "tmpdir": tmpdir,
                                 "top_first": order["top_first_in_file"]}

    st.add("PSFC", (jm, im))[...] = got["PSFC"]
    hgt = got["HGT_SFC"]
    st.add("HGT_M", (jm, im))[...] = hgt
    st.add("SOILHGT", (jm, im))[...] = hgt
    st.add("FIS", (jm, im))[...] = hgt * G_NMMB

    # Shelter / 10 m state from the analysis' own 2 m and 10 m records (NCEP grid-relative winds, flags 56);
    # missing records fall back to the lowest model layer (l = LM, index LM-1).
    t2 = sh.get("T2M", st["T"][lm - 1])
    q2 = sh.get("Q2M", st["Q"][lm - 1])
    shelter_state(ctx, t2, q2, sh.get("U10M", st["U"][lm - 1]), sh.get("V10M", st["V"][lm - 1]),
                  p2=sh.get("P2M"), th10=sh.get("TH10M"), q10=sh.get("Q10M"))
    del sh

    raw = {
        "TSOIL": cp.ascontiguousarray(cp.stack([got[f"TSOIL{k}"] for k in range(1, 5)])),
        "SOILW": cp.ascontiguousarray(cp.stack([got[f"SOILW{k}"] for k in range(1, 5)])),
        "VGTYP": cp.rint(got["VGTYP"]).astype(cp.int32),
        "SOTYP": cp.rint(got["SOTYP"]).astype(cp.int32),
    }
    for k in ("TMP_SFC", "ICEC", "LAND", "TSOIL_3M", "SNOD", "WEASD", "ALBDO", "VEG", "SFCR", "CNWAT",
              "SNOWC", "WTMP"):
        raw[k] = got[k]
    # SNOWC is percent in NCEP's GRIB2 output (NCEP 0-1-201); init-native-sfc takes a fraction.
    raw["SNOWC"] = cp.clip(raw["SNOWC"] / cp.float32(100.0), 0.0, 1.0).astype(cp.float32)
    ctx.tables["raw_sfc"] = raw
    # The snow-free base albedo from the analysis itself (albase_from_analysis); init-static-missing puts it in the
    # state before its vegetation-table fill, which then only touches the snow-covered and water cells.
    albase = albase_from_analysis(raw["ALBDO"], raw["SNOWC"], raw["WEASD"], raw["LAND"])
    # The NAM's liquid soil water, base albedo (under snow too) and maximum snow albedo from NCEP's native surface
    # file of the same cycle (NATIVE_LAND); a case without it keeps the split and the table fill, and says so.
    lpath = native_land_file(ctx.case_dir)
    rec = decode_native_land(ctx, lpath, tmpdir) if lpath is not None else {}
    extra, sh2o, lacking = land_from_surface_file(rec, albase, raw["LAND"])
    ctx.tables["static_extra"] = extra
    if sh2o is not None:
        ctx.tables["sfc_override"] = {**(ctx.tables.get("sfc_override") or {}), "SH2O": sh2o}
    if lacking:
        what = f"{lpath.name} lacks records" if lpath else "no NCEP native surface file (raw/nam/bgrdsf00.land.grib2)"
        print(f"{UNIT}: {what}: {', '.join(lacking)} from the frozen-fraction split and the vegetation table, not "
              "the NAM's own; `woof-nam fetch` on the case brings the file", flush=True)
    else:
        print(f"{UNIT}: {lpath.name}: the NAM's own liquid soil water, base albedo and maximum snow albedo",
              flush=True)
    ctx.tables["soil_hydraulics"] = {k: cp.asarray(v) for k, v in soil_hydraulics().items()}
    ctx.tables["grid_source"] = {"source": "native", "file": str(path), **order}
    print(f"{UNIT}: native analysis {path.name}: {len(names3)} hybrid fields x {lm} levels, "
          f"{len(NATIVE_SFC)} surface fields, warm {warm}; file level 1 = "
          f"{'top' if order['top_first_in_file'] else 'bottom'}", flush=True)
    return ctx.tables["grid_source"]
