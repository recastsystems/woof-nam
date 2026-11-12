"""Fetch one case's inputs from AWS NODD into cases/<case>/raw/ (parallel, byte-range subsets), for any domain.

    python -m namgpu.init.fetch_case --cycle 2026110412 --hours 48 [--config parm/nam_configfile_X]
        [--source gfs] [--boco-age 6] [--case-dir DIR] [--dry-run]

What is fetched is the FETCH_ROWS table: one row per source file family, with the archive's date range, its
path template per era, the record subset, the role (init / boco / referee / truth) and the region it covers.
A row is planned only when the cycle is inside its archive dates and the domain (the configfile's grid, every
H point) overlaps its region; anything else is skipped with the reason printed. Adding a source or a region
is a row here, never a new code path.

Rows today:
- gfs-f000   GFS analysis (init), global: 2021-01-01 00Z on from the NOAA S3 archive (subset by .idx byte ranges
             to the records the start reads: pressure levels, surface, soil, snow, ice, canopy water); before it
             the archived eras of GFS_ERAS (NCAR GDEX 0.25 deg from 2015-01-15, NCEI 0.5 deg analyses from
             2004-03), whole files. Written where sources.SOURCES["gfs"]["file"] expects it.
- gfs-boco   GFS 0.25 deg forecast leads every boco_interval_h covering the run, from the cycle boco_age_h
             older than the start (SOURCES["gfs"]: 6 h as operational NAM; 0 h for past cases), global.
- nam-*      NAM native B-grid analysis bgrd3d00 (the native start), native surface bgrdsfFF and the grid 218
             awphysFF picture referee: only where the domain overlaps the NAM 12 km grid, only for cycles
             before the NAM retirement (2026-11-03 12Z).
- rap-wrfprs RAP 13 km analysis wrfprsf00 (the rap start), subset by .idx to the records source_records.RAP reads,
             only where the domain overlaps RAP's rotated grid, from 2021-02-22. A rap start also needs gfs-f000 (its
             fallback outside RAP's grid and above 50 hPa) and gfs-boco (its boundaries).
- mrms       MRMS MergedReflectivityQCComposite_00.50 (truth), the file nearest each hour, only where the
             domain overlaps MRMS CONUS (20-55 N, 130-60 W).
All requests run in a thread pool (I/O bound); nothing here is a single-core stage.

Any other init source with a "fetch" entry in sources.SOURCES (ECMWF open data IFS 0.1 / 0.25 deg):

    python -m namgpu.init.fetch_case --cycle 2026110312 --hours 48 --source ifs

fetches that row's start and boundary leads (every 3 h, the row's own cycle) into cases/<case>/raw/, by the row's
record selection on the ECMWF jsonl .index beside each object (byte ranges, adjacent records merged), its hosts in
order (a host that refuses the index or an object is skipped), plus the row's donor fields (GFS f000 ICEC by .idx).
One output file per lead: the selected records of the lead's objects concatenated, and the row's f000_only
records (the surface geopotential z, published at step 0 only) appended to every later lead. Prints the bytes.

A row whose fetch kind is "arco-zarr" (era5) is fetched by gpuwm's own ERA5 route, gpuwm.era5_arco.retrieve_era5_arco
(Google's public ARCO-ERA5 Zarr store, no account): one NetCDF per boundary time into the row's per-time folder, the
latitude band of the run's domain (the configfile grid, plus the row's margin) and every longitude. The store's own
bounds document is read first, so an hour past its last published day is refused with that day.
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

NAM = "https://noaa-nam-pds.s3.amazonaws.com"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MRMS = "https://noaa-mrms-pds.s3.amazonaws.com"
RAP = "https://noaa-rap-pds.s3.amazonaws.com"
HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
RRFS = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
MRMS_MAX_OFFSET_S = 600       # MRMS writes every 2 min; a file more than 10 min off is not that hour
# A composite written while radars drop out of the mosaic is a partial field: much of CONUS turns to -999 (no radar
# coverage) and storms vanish. Defect (MRMS partial granule, 2023-04-01 00Z): the nearest granule, 00:00:35, had 67.6 %
# of its pixels at -999 against 34.7 % in a full mosaic that night; the 2023-03-31 squall line was missing and FSS scored
# 0.03. A full mosaic's no-coverage fraction depends on the era's radar set (34.7 % in 2023, 38.2-38.9 % in 2021-22),
# so a granule is judged against that day's own full mosaics: the lowest fraction among the granules nearest the valid
# time and valid +/- 1, 2, 3 h, plus a tolerance of about two radars.
MRMS_NOCOVER = -500.0         # MRMS -999 = no radar coverage (-99 = covered, no echo)
MRMS_REF_HOURS = 3
MRMS_NOCOVER_TOL = 0.01
# Radars that stay down for hours (Hurricane Ian, 2022-09-27 14-19Z: 0.359-0.364 against 0.347 that day) leave the archive
# with no better mosaic in the window; that best-available field is the truth there. A mosaic dropout sits far above it
# (0.426-0.676 at 2023-04-01 00Z, 0.59-0.80 at 11Z, against 0.347). So when no granule in the window comes within
# MRMS_NOCOVER_TOL of the day's full mosaic, the window's own best is taken if it is within MRMS_OUTAGE_TOL (about
# eight radars) of it; beyond that the hour is an outage and gets no file.
MRMS_OUTAGE_TOL = 0.05
ROOT = Path(__file__).resolve().parents[2]

GFS_VARS = re.compile(
    r":(HGT|TMP|SPFH|RH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR|GRLE|O3MR):(\d+(\.\d+)? mb):"
    r"|:(PRES|HGT|TMP|PRMSL|LAND|ICEC|WEASD|SNOD|TSOIL|SOILW):(surface|mean sea level|"
    r"0-0.1 m below ground|0.1-0.4 m below ground|0.4-1 m below ground|1-2 m below ground):"
    r"|:(TMP|SPFH|RH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# Boundary leads: only what init-gfs-bocos reads (gfs_start PL_ORDER levels, SFC_COLUMNS) plus the 2 m, 10 m and
# PRMSL records the source-forecast referee panels draw. Was GFS_VARS: O3MR, GRLE, soil, snow, ice and land are
# about 1/6 of the bytes and are never read from a boundary file.
# Isobaric levels at p >= 1 hPa only ("[1-9]\d* mb"): gfs_start.decode_source keeps p >= 1 hPa, so the 8 levels
# above it (0.01-0.7 mb) were downloaded in every boundary file and dropped (the door lane's 1b6435d pattern,
# 259 -> 183 MB per file with the GRLE/O3MR cut, was lost in the nest/all merge).
GFS_BOCO_VARS = re.compile(
    r":(HGT|TMP|SPFH|RH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR):([1-9]\d*(\.\d+)?) mb:"
    r"|:(PRES|HGT|TMP):surface:|:PRMSL:mean sea level:"
    r"|:(TMP|SPFH|RH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# The start also reads canopy water (gfs_start.SFC CNWAT), which only the analysis carries.
GFS_INIT_VARS = re.compile(GFS_VARS.pattern + r"|:CNWAT:surface:")
NAM_HOURLY = re.compile(
    r":(REFC):entire atmosphere|:APCP:surface:|:TMP:2 m above ground:|:DPT:2 m above ground:"
    r"|:(UGRD|VGRD):10 m above ground:|:PRMSL:mean sea level:|:MSLET:mean sea level:")
# RAP wrfprsf00: what the rap start reads (source_records.RAP): the isobaric ladder of its slots, graupel and TKE, the
# surface, 2 m, 10 m, sea ice, land, snow and the nine RUC soil nodes.
RAP_INIT_VARS = re.compile(
    r":(HGT|TMP|SPFH|UGRD|VGRD|CLWMR|CIMIXR|RWMR|SNMR|GRLE|TKE):(\d+(\.\d+)?) mb:"
    r"|:(PRES|HGT|TMP|LAND|ICEC|WEASD|SNOD):surface:"
    r"|:(TSOIL|SOILW):\d+(\.\d+)?-\d+(\.\d+)? m below ground:"
    r"|:(TMP|SPFH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# HRRR wrfprsf00: the same record set (source_records.HRRR) plus canopy water; HRRR's .idx names cloud water CLMR
# (0-1-22), where RAP's says CLWMR, so without the CLMR term the start would have no cloud water record
HRRR_INIT_VARS = re.compile(RAP_INIT_VARS.pattern + r"|:CLMR:(\d+(\.\d+)?) mb:|:CNWAT:surface:")
# RRFS v1: the isobaric records from prslev, the surface and soil records from 2dfld (source_records.RRFS)
RRFS_PRS_VARS = re.compile(r":(HGT|TMP|SPFH|UGRD|VGRD|CLMR|ICMR|RWMR|SNMR|GRLE):(\d+(\.\d+)?) mb:")
RRFS_SFC_VARS = re.compile(
    r":(PRES|HGT|TMP|LAND|ICEC|WEASD|SNOD|CNWAT):surface:"
    r"|:(TSOIL|SOILW):\d+(\.\d+)?-\d+(\.\d+)? m below ground:"
    r"|:(TMP|SPFH):2 m above ground:|:(UGRD|VGRD):10 m above ground:")
# NAM bgrdsf00: the land states of the NAM's own start that its analysis bgrd3d00 lacks (sources.NATIVE_LAND): liquid
# soil water on the four Noah layers, the snow-free base albedo and the maximum snow albedo. About 1.6 MB of 45.
NAM_LAND = re.compile(r":SOILL:\d+(\.\d+)?-\d+(\.\d+)? m below ground:anl:|:(SNFALB|MXSALB):surface:anl:")
PATTERNS = {"gfs_init": GFS_INIT_VARS, "gfs_boco": GFS_BOCO_VARS, "nam_hourly": NAM_HOURLY, "rap_init": RAP_INIT_VARS,
            "hrrr_init": HRRR_INIT_VARS, "rrfs_prs": RRFS_PRS_VARS, "rrfs_sfc": RRFS_SFC_VARS, "nam_land": NAM_LAND}

# Regions a row covers. "global": everywhere. "bbox": geographic lat/lon box (degrees, lon -180..180).
# "rotated": a rotated lat-lon grid given by its centre (tph0d, tlm0d) and wbd/sbd, tested in its own rotated
# coordinates, so the NAM 12 km footprint is exact.
REGIONS = {
    "global": {"kind": "global"},
    # The NAM 12 km parent grid (parm/nam_configfile tph0d/tlm0d/wbd/sbd). The keys are written here, not read
    # through config.load: with NAMGPU_CONFIGFILE set, loading the default configfile returns the run's own
    # domain, so every nest "overlapped" NAM and fetched its archive (Gabrielle case, 2026-11-04: 1 GB of bgrdsf
    # for a New Zealand domain).
    "nam12": {"kind": "rotated", "tph0d": 54.0, "tlm0d": -106.0, "wbd": -60.039, "sbd": -45.036},
    # NCEP grid 218 (Lambert, 614 x 428, 12.19 N 133.46 W .. 57.29 N 49.39 W corners): its lat/lon hull.
    "grid218": {"kind": "bbox", "lat": (12.0, 61.5), "lon": (-153.0, -49.0)},
    "mrms_conus": {"kind": "bbox", "lat": (20.0, 55.0), "lon": (-130.0, -60.0)},
    # RAP's 13 km rotated grid (wrfprsf00, 953 x 834, centre 54 N 106 W; source_grids.py measured its first point at
    # rotated -50.743, -57.993 deg)
    # HRRR's 3 km Lambert grid (1799 x 1059): the lat/lon hull of its corners (21.14-47.84 N, 134.10-60.92 W)
    "hrrr3": {"kind": "bbox", "lat": (21.1, 47.9), "lon": (-134.1, -60.9)},
    "rap13": {"kind": "rotated", "tph0d": 54.0, "tlm0d": -106.0, "wbd": -57.99, "sbd": -50.74},
}

# Path eras: the GFS bucket moved files under atmos/ with GFS v16 (first cycle 2021-03-22 12Z); before it the
# files sit directly under the cycle folder. The bucket's archive begins 2021-01-01 00Z.
# Before the bucket, two archived eras of the same GFS GRIB2 (same decoder, same records; only host, grid and
# cadence differ). An era's optional third item overrides the row: host, kind ("whole": these archives publish no
# .idx, so whole files), and "chain": "analyses" for an era with no forecast archive online, where every
# boundary time is the analysis (000) or the 3 h first guess (003) of the GFS cycle at or before it.
#   2015-01-15 00Z  NCAR GDEX d084001 (NCEP GFS 0.25 deg global forecast grids, historical archive), the full
#                   pgrb2.0p25 files 3-hourly to 240 h, read over OSDF without an account (verified 2026-11-06:
#                   2016021512 f003 217.6 MB, 2021021400 f084 342.8 MB, 2015011512 f000 195.3 MB).
#   2004-03-01 00Z  NCEI THREDDS model-gfs-g4-anl-files-old (GFS 0.5 deg analyses gfsanl_4, 000/003/006 of every
#                   cycle; online 2004-03..2020-05). NCEI's 0.5 deg FORECAST archive (model-gfs-004-files) is not
#                   online before 2019-08 except a few months, so a start before 2015-01-15 takes analysis
#                   boundaries (verified 2026-11-06: 20110109 12Z 000 48.6 MB, 20140211 12Z 003 52.9 MB).
GFS_GDEX = "https://osdf-director.osg-htc.org/ncar/rda/d084001"
GFS_NCEI_G4ANL = "https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files-old"
GFS_ERAS = (("2004-03-01T00", "{vym}/{vymd}/gfsanl_4_{vymd}_{vhh}00_{vofs:03d}.grb2",
             {"host": GFS_NCEI_G4ANL, "kind": "whole", "chain": "analyses", "grid": "gfs_0p5",
              "licence": "NOAA/NCEP GFS 0.5 deg analyses, US government work (public domain), NCEI THREDDS"}),
            ("2015-01-15T00", "{yyyy}/{ymd}/gfs.0p25.{ymd}{hh}.f{lead:03d}.grib2",
             {"host": GFS_GDEX, "kind": "whole",
              "licence": "NOAA/NCEP GFS 0.25, US government work (public domain), NCAR GDEX d084001 over OSDF"}),
            ("2021-01-01T00", "gfs.{ymd}/{hh}/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"),
            ("2021-03-22T12", "gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"))
# NAM eras. AWS NODD noaa-nam-pds begins with nam.20210916 (checked 2026-11-09: no keys for 20210214, 20210301,
# 20210601). Before it, the awphys (grid 218) family is NCEI THREDDS model-nam218, the same CONUS 12 km grid-218
# product (whole files, no .idx), online from 2020-05-18 (catalog 2026-11-09). Breakage it prevents: the
# texas-freeze-2021-02 case staged with no archived NAM (19 x 404), so its scores lost the archived-NAM column.
# The bgrd and hourly-subset families have no archive before NODD and are skipped with that reason.
NAM_ERAS = (("2021-09-16T00", "nam.{ymd}/nam.t{hh}z.{stem}{lead:02d}.tm00.grib2"),)
NAM_NCEI_218 = "https://www.ncei.noaa.gov/thredds/fileServer/model-nam218"
NAM_AWPHYS_ERAS = (("2020-05-18T00", "{yyyy}{mm}/{ymd}/nam_218_{ymd}_{hh}00_{lead:03d}.grb2",
                    {"host": NAM_NCEI_218, "kind": "whole",
                     "licence": "NOAA/NCEP NAM grid 218, US government work (public domain), NCEI THREDDS model-nam218"}),
                   ) + NAM_ERAS
NAM_RETIRED = "2026-11-03T12"
# RAP on AWS NODD: the archive begins 2021-02-22 (some early days hold only some cycles' wrfprs; a missing file is a
# failed required request, named by the fetch)
RAP_ERAS = (("2021-02-22T00", "rap.{ymd}/rap.t{hh}z.wrfprsf{lead:02d}.grib2"),)
HRRR_ERAS = (("2014-07-30T18", "hrrr.{ymd}/conus/hrrr.t{hh}z.wrfprsf{lead:02d}.grib2"),)
RRFS_PRS_ERAS = (("2026-08-12T00", "rrfs.{ymd}/{hh}/rrfs.t{hh}z.prslev.3km.f{lead:03d}.conus.grib2"),)
RRFS_SFC_ERAS = (("2026-08-12T00", "rrfs.{ymd}/{hh}/rrfs.t{hh}z.2dfld.3km.f{lead:03d}.conus.grib2"),)

# One row per file family. cycle: "start" or "boco" (the boundary cycle, boco_age_h older). leads: "f000",
# "boco" (boco_interval_h steps covering the run), "every3" (0..hours by 3), "hourly36" (1..min(hours, 36),
# not multiples of 3: NAM writes hourly awphys only to 36 h). kind: whole | subset (records by .idx) | mrms.
# dest is relative to raw/. role init/boco rows are required when their source is the run's source. licence: the
# data's licence and host, beside the row (permissive data only; nothing NC-licensed is fetched).
FETCH_ROWS = (
    {"name": "gfs-f000", "licence": "NOAA/NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds",
     "source": "gfs", "role": "init", "host": GFS, "eras": GFS_ERAS, "cycle": "start",
     "leads": "f000", "kind": "subset", "records": "gfs_init", "region": "global",
     "dest": "gfs/gfs.t{hh}z.pgrb2.0p25.f000"},
    # follow_dest / follow_records: a storm-following run (fetch(follow=True)) lands a window move at a boundary
    # lead, where the new window's fresh start is made from that lead (namgpu/nest/prepare.py), so each lead is
    # fetched with the START record set (soil, snow, ice, canopy water: gfs_init) as follow_dest, and the boundary
    # file name is a hard link to it (the start set holds every boundary record). Lead 0 of the start cycle is
    # the gfs-f000 row's file.
    {"name": "gfs-boco", "licence": "NOAA/NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds",
     "source": "gfs", "role": "boco", "host": GFS, "eras": GFS_ERAS, "cycle": "boco",
     "leads": "boco", "kind": "subset", "records": "gfs_boco", "region": "global",
     "dest": "gfs/gfs.t{hh}z.f{lead:03d}.grib2",
     "follow_dest": "gfs/gfs.t{hh}z.f{lead:03d}.start.grib2", "follow_records": "gfs_init"},
    {"name": "nam-bgrd3d00", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "init", "host": NAM, "eras": NAM_ERAS,
     "stem": "bgrd3d", "cycle": "start", "leads": "f000", "kind": "whole", "region": "nam12",
     "until": NAM_RETIRED, "dest": "nam/bgrd3d00.grib2"},
    # optional: a native start without it keeps the frozen-fraction split and the table albedos and says so in one
    # line (sources.stage_native), so a missing file is reported, not counted as a failed required request
    {"name": "nam-bgrdsf00-land", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "init", "optional": True, "host": NAM, "eras": NAM_ERAS,
     "stem": "bgrdsf", "cycle": "start", "leads": "f000", "kind": "subset", "records": "nam_land", "region": "nam12",
     "until": NAM_RETIRED, "dest": "nam/bgrdsf00.land.grib2"},
    {"name": "nam-bgrdsf", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_ERAS,
     "stem": "bgrdsf", "cycle": "start", "leads": "every3", "kind": "whole", "region": "nam12",
     "until": NAM_RETIRED, "dest": "nam/bgrdsf{lead:02d}.grib2"},
    {"name": "nam-awphys", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_AWPHYS_ERAS,
     "stem": "awphys", "cycle": "start", "leads": "every3", "kind": "whole", "region": "grid218",
     "until": NAM_RETIRED, "dest": "nam/awphys{lead:02d}.grib2"},
    {"name": "nam-awphys-hourly", "licence": "NOAA/NCEP NAM, US government work (public domain), AWS NODD noaa-nam-pds",
     "source": "native", "role": "referee", "host": NAM, "eras": NAM_ERAS,
     "stem": "awphys", "cycle": "start", "leads": "hourly36", "kind": "subset", "records": "nam_hourly",
     "region": "grid218", "until": NAM_RETIRED, "dest": "nam/awphys{lead:02d}.sub.grib2"},
    {"name": "rap-wrfprs", "licence": "NOAA/NCEP RAP, US government work (public domain), AWS NODD noaa-rap-pds",
     "source": "rap", "role": "init", "host": RAP, "eras": RAP_ERAS, "cycle": "start", "leads": "f000",
     "kind": "subset", "records": "rap_init", "region": "rap13", "since": "2021-02-22T00",
     "dest": "rap/rap.t{hh}z.wrfprsf00.grib2"},
    {"name": "hrrr-wrfprs", "licence": "NOAA/NCEP HRRR, US government work (public domain), AWS NODD noaa-hrrr-bdp-pds",
     "source": "hrrr", "role": "init", "host": HRRR, "eras": HRRR_ERAS, "cycle": "start", "leads": "f000",
     "kind": "subset", "records": "hrrr_init", "region": "hrrr3", "since": "2014-07-30T18",
     "dest": "hrrr/hrrr.t{hh}z.wrfprsf00.grib2"},
    {"name": "rrfs-prslev", "licence": "NOAA/NCEP RRFS, US government work (public domain), AWS NODD noaa-rrfs-ops-pds",
     "source": "rrfs", "role": "init", "host": RRFS, "eras": RRFS_PRS_ERAS, "cycle": "start", "leads": "f000",
     "kind": "subset", "records": "rrfs_prs", "region": "hrrr3", "since": "2026-08-12T00",
     "dest": "rrfs/rrfs.t{hh}z.prslev.3km.f000.conus.grib2"},
    {"name": "rrfs-2dfld", "licence": "NOAA/NCEP RRFS, US government work (public domain), AWS NODD noaa-rrfs-ops-pds",
     "source": "rrfs", "role": "init", "host": RRFS, "eras": RRFS_SFC_ERAS, "cycle": "start", "leads": "f000",
     "kind": "subset", "records": "rrfs_sfc", "region": "hrrr3", "since": "2026-08-12T00",
     "dest": "rrfs/rrfs.t{hh}z.2dfld.3km.f000.conus.grib2"},
    {"name": "mrms", "licence": "NOAA MRMS, US government work (public domain), AWS NODD noaa-mrms-pds",
     "source": "mrms", "role": "truth", "host": MRMS, "cycle": "start", "leads": "hourly",
     "kind": "mrms", "region": "mrms_conus", "since": "2020-10-14T00", "dest": "mrms"},
)


def _retry(fn, tries=5, slow_tries=10):
    """Retry a request. A throttle answer (HTTP 503 SlowDown / 429) backs off exponentially with jitter, up to
    slow_tries: the ECMWF open-data S3 bucket answered 16 parallel ranged GETs with SlowDown and the old 2-8 s steps
    (20 s in all) failed 4 of 7 files of an AIFS case fetch (2026-11-08). Other errors keep the short linear steps."""
    import random
    k = slow = 0
    while True:
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and slow < slow_tries - 1:
                time.sleep(min(60.0, 2.0 ** slow) * random.uniform(0.5, 1.5))
                slow += 1
                continue
            if k >= tries - 1 or e.code in (429, 503):
                raise
            time.sleep(2.0 * (k + 1))
            k += 1
        except (OSError, urllib.error.URLError):
            if k >= tries - 1:
                raise
            time.sleep(2.0 * (k + 1))
            k += 1


def _get(url: str, rng: str | None = None, timeout=120) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})

    def once():
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    return _retry(once)


def _download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    def once():
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=8 << 20)
    _retry(once)
    tmp.rename(dest)
    return f"got {dest.name} {dest.stat().st_size / 1e6:.0f} MB"


def _download_linked(url: str, start_dest: Path, dest: Path) -> str:
    """A storm-following lead from a whole-file era: the whole file as start_dest, dest a hard link to it."""
    msg = _download(url, start_dest)
    if not (dest.exists() and dest.stat().st_size > 0):
        try:
            os.link(start_dest, dest)
        except OSError:
            shutil.copy2(start_dest, dest)
    return msg


def _follow_lead(url: str, dest: Path, start_dest: Path, pattern: re.Pattern) -> str:
    """A storm-following run's boundary lead: the start record set as start_dest, and dest (the boundary file
    name) a hard link to it when dest is not already there (a boundary-only file from an earlier run is kept)."""
    msg = _subset(url, start_dest, pattern)
    if not (dest.exists() and dest.stat().st_size > 0):
        try:
            os.link(start_dest, dest)
        except OSError:
            shutil.copy2(start_dest, dest)
        msg += f"; {dest.name} -> {start_dest.name}"
    return msg


def _subset(url: str, dest: Path, pattern: re.Pattern) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name}"
    idx = _get(url + ".idx").decode().splitlines()
    rows = [l.split(":") for l in idx if l.strip()]
    offs = [int(r[1]) for r in rows]
    want = [k for k, l in enumerate(idx) if pattern.search(":" + ":".join(l.split(":")[3:]) + ":")]
    if not want:
        # An empty match would write a zero-byte dest that the st_size > 0 skip check refetches
        # on every run, so it is an error, not a file.
        raise ValueError(f"{dest.name}: no idx records of {url} match the pattern ({len(idx)} records)")
    tmp = dest.with_suffix(".part")
    # Sub-messages (idx "329.1", "329.2") share one GRIB2 message and one byte offset: fetch each message
    # once, ending before the next larger offset. (Ending at offs[k + 1] - 1 made an inverted range for the
    # first sub-message, which S3 ignores and answers with the whole file.)
    starts = sorted({offs[k] for k in want})
    ranges = []                                    # [start, end] byte ranges; end None = to end of file
    for s in starts:
        nxt = next((o for o in offs if o > s), None)
        end = nxt - 1 if nxt is not None else None
        if ranges and ranges[-1][1] is not None and ranges[-1][1] + 1 == s:
            ranges[-1][1] = end                    # adjacent records: one request (same bytes, ~10x fewer GETs)
        else:
            ranges.append([s, end])
    # each range is a new connection; one at a time they ran at ~2 MB/s (118 GETs, 129 s for a GFS file),
    # so the ranges of one file are fetched in parallel and written in order (same bytes)
    with ThreadPoolExecutor(min(8, len(ranges))) as ex, open(tmp, "wb") as f:
        for chunk in ex.map(lambda r: _get(url, f"{r[0]}-{r[1] if r[1] is not None else ''}"), ranges):
            f.write(chunk)
    tmp.rename(dest)
    return f"got {dest.name} {len(want)} recs {dest.stat().st_size / 1e6:.0f} MB"


@lru_cache(maxsize=None)
def _list_mrms_day(day: str) -> tuple[str, ...]:
    """Every key under one UTC day prefix, following continuation tokens (MRMS has ~720 a day). Cached: each
    hour lists its own day and both neighbours, so a 48 h case asked for the same three days ~49 times."""
    prefix = f"CONUS/MergedReflectivityQCComposite_00.50/{day}/"
    keys, token = [], None
    while True:
        q = f"list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            q += "&continuation-token=" + urllib.parse.quote(token, safe="")
        xml = _get(f"{MRMS}/?{q}").decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml or not m:
            return tuple(keys)
        token = m.group(1)


def _mrms_nocover_frac(grib: Path) -> float:
    """Fraction of an MRMS composite's pixels at -999 (no radar coverage), decoded on the scorer's own path."""
    import tempfile  # noqa: PLC0415

    from namgpu.init.init_grib2_decode import inventory  # noqa: PLC0415
    from namgpu.io.io_referee_panels import _grib_values  # noqa: PLC0415
    row = inventory(grib)[0]
    with tempfile.TemporaryDirectory(prefix="mrms-cov-", dir=grib.parent) as td:
        v = _grib_values(grib, row, Path(td) / "m")
    return float(np.count_nonzero(v < MRMS_NOCOVER)) / v.size


@lru_cache(maxsize=None)
def _mrms_key_frac(key: str, work: str) -> float:
    """No-coverage fraction of one archive granule (downloaded to a scratch file, then removed). Cached: the reference
    granules of one hour are the candidates of its neighbours."""
    tmp = Path(work) / f".cov-{threading.get_ident()}-{key.rsplit('_', 1)[-1]}.grib2"
    try:
        tmp.write_bytes(gzip.decompress(_get(f"{MRMS}/{key}")))
        return _mrms_nocover_frac(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _mrms_keys_near(valid: datetime) -> list[tuple[float, str]]:
    """(seconds off, key) of every granule listed for valid's UTC day and both neighbours, nearest first."""
    keys = []
    for dd in (-1, 0, 1):
        keys += _list_mrms_day((valid + timedelta(days=dd)).strftime("%Y%m%d"))
    out = []
    for k in keys:
        m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", k)
        if m:
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            out.append((abs((t - valid).total_seconds()), k))
    return sorted(out)


def _mrms_reference(valid: datetime, work: Path) -> float | None:
    """Lowest no-coverage fraction among the granules nearest valid +/- 1..MRMS_REF_HOURS h: the day's full mosaic.
    A radar outage rarely spans 6 h, so at least one of them is complete."""
    fr = []
    for h in range(-MRMS_REF_HOURS, MRMS_REF_HOURS + 1):
        if h == 0:
            continue
        near = _mrms_keys_near(valid + timedelta(hours=h))
        if near and near[0][0] <= MRMS_MAX_OFFSET_S:
            fr.append(_mrms_key_frac(near[0][1], str(work)))
    return min(fr) if fr else None


def _mrms_hour(valid: datetime, dest_dir: Path) -> str:
    # The nearest file can sit in the previous or next UTC day's prefix (valid times near 00Z), so list valid-1d, valid
    # and valid+1d, and take the nearest COMPLETE mosaic within MRMS_MAX_OFFSET_S (see MRMS_NOCOVER_TOL).
    dest = dest_dir / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
    ok = dest.with_suffix(dest.suffix + ".ok")      # written with a checked file: granule, its fraction, the reference
    if dest.exists() and ok.exists():
        return f"have {dest.name}"
    cands = _mrms_keys_near(valid)
    if not cands:
        return f"MRMS none for {valid}"
    if cands[0][0] > MRMS_MAX_OFFSET_S:
        # A live case asks for valid times that have not happened yet; the nearest key is then the newest file,
        # hours early, and writing it under this hour would show the wrong truth (live case 2026-11-04 12Z: 19Z-12Z
        # next day all got 18:36Z). No file, so a later fetch picks the real one up.
        return f"MRMS none within {MRMS_MAX_OFFSET_S} s of {valid} (nearest {cands[0][1].rsplit('_', 1)[-1]})"
    cands = [(d, k) for d, k in cands if d <= MRMS_MAX_OFFSET_S]
    ref = _mrms_reference(valid, dest_dir)
    if dest.exists():
        # cached before this check: kept if it is as complete as the day's full mosaic, else the window decides below
        frac = _mrms_nocover_frac(dest)
        if ref is None or frac <= ref + MRMS_NOCOVER_TOL:
            ok.write_text(f"cached no-coverage {frac:.4f} reference {ref}\n")
            return f"have {dest.name} (checked: no-coverage {frac:.3f}, reference {ref})"
    fracs: list[tuple[str, float]] = []              # (key, no-coverage fraction), nearest first
    pick = None
    for _, k in cands:
        fracs.append((k, _mrms_key_frac(k, str(dest_dir))))
        if ref is not None and fracs[-1][1] <= ref + MRMS_NOCOVER_TOL:
            pick = fracs[-1]
            break
    if pick is None:
        # nothing as complete as the day's full mosaic: the window's best, unless even that is a dropout
        local = min(f for _, f in fracs)
        if ref is None or local <= ref + MRMS_OUTAGE_TOL:
            pick = next((k, f) for k, f in fracs if f <= local + MRMS_NOCOVER_TOL)
    rs = "none" if ref is None else f"{ref:.3f}"
    seen = [f"{k.rsplit('_', 1)[-1]} {f:.3f}" for k, f in fracs]
    if pick is None:
        # A long radar outage: no file, so the hour is not scored against a partial field.
        if dest.exists():
            dest.unlink()
        return f"MRMS none complete within {MRMS_MAX_OFFSET_S} s of {valid} (reference {rs}; {', '.join(seen)})"
    k, frac = pick
    name = k.rsplit("_", 1)[-1]
    skipped = seen[:fracs.index(pick)]
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(gzip.decompress(_get(f"{MRMS}/{k}")))
    tmp.rename(dest)
    ok.write_text(f"{name} no-coverage {frac:.4f} reference {rs}\n")
    why = f"; skipped partial {', '.join(skipped)}" if skipped else ""
    return f"got {dest.name} ({name}, no-coverage {frac:.3f}, reference {rs}{why})"


# ---------------------------------------------------------------------------------------------------------
# Domain footprint and region tests
# ---------------------------------------------------------------------------------------------------------
def domain_points(cfg=None, stride: int = 4):
    """Geographic (lat, lon) degrees of the configured grid's H points (every stride-th row/column plus the
    whole boundary ring), lon in -180..180. cfg None = the run's configfile (NAMGPU_CONFIGFILE)."""
    from namgpu import config, grid as gridmod
    cfg = cfg or config.load()
    g = gridmod.build(cfg)
    lat, lon = np.degrees(g.glat_h), np.degrees(g.glon_h)
    m = np.zeros(lat.shape, bool)
    m[::stride, ::stride] = True
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return lat[m], lon[m]


def _geo_to_rotated(lat, lon, tph0d, tlm0d):
    """Geographic -> rotated lat-lon (degrees) for the rotated pole of namgpu.grid (inverse of
    rotated_to_geographic)."""
    la, lo = np.radians(lat), np.radians(lon)
    p0, l0 = np.radians(tph0d), np.radians(tlm0d)
    x = np.cos(la) * np.cos(lo - l0)
    y = np.cos(la) * np.sin(lo - l0)
    z = np.sin(la)
    xr = np.cos(p0) * x + np.sin(p0) * z
    zr = -np.sin(p0) * x + np.cos(p0) * z
    return np.degrees(np.arcsin(np.clip(zr, -1.0, 1.0))), np.degrees(np.arctan2(y, xr))


def overlaps(region: str, lat, lon) -> bool:
    r = REGIONS[region]
    if r["kind"] == "global":
        return True
    if r["kind"] == "bbox":
        (a, b), (c, d) = r["lat"], r["lon"]
        lo = (np.asarray(lon) + 180.0) % 360.0 - 180.0
        return bool(np.any((lat >= a) & (lat <= b) & (lo >= c) & (lo <= d)))
    if r["kind"] == "rotated":
        tph, tlm = _geo_to_rotated(lat, lon, r["tph0d"], r["tlm0d"])
        return bool(np.any((tlm >= r["wbd"]) & (tlm <= -r["wbd"]) & (tph >= r["sbd"]) & (tph <= -r["sbd"])))
    raise ValueError(f"fetch-case: region {region!r} kind {r['kind']!r}")


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H")


def era_of(row, cyc: datetime) -> dict:
    """The row's era for a cycle: {"since", "template"} plus the era's own overrides (host, kind, chain, ...)."""
    era = None
    for e in row["eras"]:
        if cyc >= _t(e[0]):
            era = {"since": e[0], "template": e[1], **(e[2] if len(e) > 2 else {})}
    if era is None:
        raise ValueError(f"fetch-case: {row['name']} has no archive before {row['eras'][0][0]} (cycle {cyc:%Y-%m-%d %HZ})")
    return era


def _era_path(row, cyc: datetime, lead: int) -> str:
    era = era_of(row, cyc)
    # chain "analyses": lead f of cycle cyc is the analysis/first guess of the GFS cycle at or before cyc + f
    v = cyc + timedelta(hours=lead)
    vc = v.replace(hour=v.hour - v.hour % 6)
    return era["template"].format(ymd=f"{cyc:%Y%m%d}", hh=f"{cyc:%H}", yyyy=f"{cyc:%Y}", mm=f"{cyc:%m}", lead=lead,
                                  stem=row.get("stem", ""), vym=f"{vc:%Y%m}", vymd=f"{vc:%Y%m%d}", vhh=f"{vc:%H}",
                                  vofs=int((v - vc).total_seconds() // 3600))


def _leads(row, hours: int, age: int, interval: int):
    spec = row["leads"]
    if spec == "f000":
        return [0]
    if spec == "boco":
        # exactly the boundary times a run of `hours` reads (domains.bc_hours: at least 6 h, in 3-hourly steps)
        from namgpu.domains import bc_hours
        return list(range(age, bc_hours(hours) + age + 1, interval))
    if spec == "every3":
        return list(range(0, hours + 1, 3))
    if spec == "hourly36":
        return [f for f in range(1, min(hours, 36) + 1) if f % 3]
    if spec == "hourly":
        return list(range(0, hours + 1))
    raise ValueError(f"fetch-case: {row['name']} leads {spec!r}")


def plan(cycle: datetime, hours: int, raw: Path, cfg=None, source: str = "gfs", age_h: int | None = None,
         rows=FETCH_ROWS, follow: bool = False):
    """Jobs [(fn, *args, required)] for one case on one domain, plus the skipped rows [(name, reason)].
    follow: a storm-following run; rows with follow_dest fetch each lead's start record set (see FETCH_ROWS)."""
    from namgpu.init import sources as srcmod
    srow = srcmod.SOURCES["gfs"]
    own = srcmod.SOURCES.get(source, {})
    bsrc = source if "boundary" in own else "gfs"
    fsrc = (own.get("fallback") or {}).get("source")
    age = srcmod.boco_age_h(srow, cfg, age_h)
    interval = int(srow.get("boco_interval_h", 3))
    lat, lon = domain_points(cfg)
    jobs, skipped = [], []
    for row in rows:
        if "until" in row and cycle >= _t(row["until"]):
            skipped.append((row["name"], f"archive ends {row['until']} (cycle {cycle:%Y-%m-%d %HZ})"))
            continue
        if "since" in row and cycle < _t(row["since"]):
            skipped.append((row["name"], f"archive starts {row['since']}"))
            continue
        if not overlaps(row["region"], lat, lon):
            skipped.append((row["name"], f"domain outside region {row['region']}"))
            continue
        # a start whose row has no boundary entry (native NAM analysis, rap) takes the GFS boundaries, as operational
        # NAM (CONTRACT section 5); a row with a fallback (rap) also needs the fallback's start file
        required = ((row["role"] == "init" and row["source"] in (source, fsrc)) or
                    (row["role"] == "boco" and row["source"] in (source, bsrc))) and not row.get("optional")
        # A regional analysis (a SOURCES row with a fallback: rap, hrrr) is read only by its own start. Fetched as an
        # optional extra it added 101 MB (RAP) + 193 MB (HRRR) to every CONUS fetch, gfs and native starts included.
        if row["role"] == "init" and not required and (srcmod.SOURCES.get(row["source"]) or {}).get("fallback"):
            skipped.append((row["name"], f"start source is {source}, not {row['source']}"))
            continue
        cyc = cycle - timedelta(hours=age) if row["cycle"] == "boco" else cycle
        try:
            era = era_of(row, cyc) if "eras" in row else {}
        except ValueError as e:
            # a cycle before the row's first archived era (NAM: NODD from 2021-09-16, NCEI grid 218 from 2020-05-18): skipped, named
            skipped.append((row["name"], str(e).replace("fetch-case: ", "")))
            continue
        # an analysis-chain era (chain "analyses") names its files by the boundary cycle and lead like any era, and
        # _era_path maps each lead to the analysis valid then, so the boundary age changes only the names
        host, kind = era.get("host", row.get("host")), era.get("kind", row["kind"])
        if row["kind"] == "mrms":
            for f in _leads(row, hours, age, interval):
                jobs.append((_mrms_hour, cycle + timedelta(hours=f), raw / row["dest"], required))
            continue
        for f in _leads(row, hours, age, interval):
            url = f"{host}/{_era_path(row, cyc, f)}"
            dest = raw / row["dest"].format(hh=f"{cyc:%H}", lead=f)
            if follow and row.get("follow_dest") and not (cyc == cycle and f == 0):
                sdest = raw / row["follow_dest"].format(hh=f"{cyc:%H}", lead=f)
                if kind == "whole":
                    jobs.append((_download_linked, url, sdest, dest, required))
                else:
                    jobs.append((_follow_lead, url, dest, sdest, PATTERNS[row["follow_records"]], required))
                continue
            if kind == "whole":
                jobs.append((_download, url, dest, required))
            else:
                jobs.append((_subset, url, dest, PATTERNS[row["records"]], required))
    return jobs, skipped


def fetch(cycle: datetime, case_dir: Path, hours: int = 48, workers: int = 16, cfg=None, source: str = "gfs",
          age_h: int | None = None, log=print, rows=FETCH_ROWS, follow: bool = False) -> int:
    """Fetch every planned input of one case. Returns the number of failed REQUIRED requests (0 = the run's
    init and boundary files are complete); failed referee/truth requests are printed and not counted.
    follow: a storm-following run (start record sets at every boundary lead; plan()). The ifs rows need nothing
    more: each of their leads already carries the whole start selection (sources.SOURCES fetch "select")."""
    raw = Path(case_dir) / "raw"
    from namgpu.init import sources as srcmod
    if "fetch" in srcmod.SOURCES.get(source, {}):
        # ifs, ifs-0p25: the row's own fetch entry (ECMWF jsonl .index byte ranges, the GFS ICEC donor)
        raw.mkdir(parents=True, exist_ok=True)
        sjobs = plan_source(source, cycle, hours, raw, cfg)
        t0 = time.perf_counter()
        bad = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fn, *args): next(x for x in args if isinstance(x, Path)) for fn, *args in sjobs}
            for fu in as_completed(futs):
                try:
                    log(fu.result())
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    log(f"FAILED {type(e).__name__}: {e}")
        total = sum(p.stat().st_size for p in futs.values() if p.exists())
        log(f"fetch-case: {source} {len(sjobs)} files, {total / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s, "
            f"{bad} failed")
        return bad
    jobs, skipped = plan(cycle, hours, raw, cfg, source, age_h, rows, follow=follow)
    for name, why in skipped:
        log(f"fetch-case: skip {name}: {why}")
    for d in {Path(j[2]).parent if j[0] is not _mrms_hour else Path(j[2]) for j in jobs}:
        d.mkdir(parents=True, exist_ok=True)
    bad = soft = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *args[:-1]): args[-1] for fn, *args in jobs}
        for fu in as_completed(futs):
            try:
                log(fu.result())
            except Exception as e:  # noqa: BLE001
                if futs[fu]:
                    bad += 1
                else:
                    soft += 1
                log(f"FAILED{'' if futs[fu] else ' (optional)'} {type(e).__name__}: {e}")
    log(f"fetch-case: {len(jobs)} requests in {time.perf_counter() - t0:.0f} s, {bad} required failed, "
        f"{soft} optional failed")
    return bad


def fetch_spinup(cycle: datetime, case_dir: Path, spinup_hours: int, workers: int = 16, cfg=None,
                 source: str = "gfs", age_h: int | None = None, log=print) -> int:
    """The inputs a spin-up start (woof-nam run --start spinup, namgpu/spinup.py) reads beyond the cycle's own: the
    source analysis spinup_hours before the cycle and the boundary leads over that window, from the cycle the
    forecast takes its boundaries from when it is valid there (GFS age 6 and a 6 h spin-up: the T0-6 cycle, leads
    0, 3, 6; the forecast reads 6..90 of the same cycle), else the spin-up start cycle. Plus the cycle's own analysis
    if it is not there yet (the update reads it). Returns the number of failed required requests."""
    from namgpu import spinup
    from namgpu.init import sources as srcmod
    row = srcmod.SOURCES[source]
    main_age = srcmod.boco_age_h(row, cfg, age_h)
    s = cycle - timedelta(hours=int(spinup_hours))
    bad = 0
    if "fetch" in row:
        # the row's own fetch entry (ifs, ifs-0p25, era5): its start and boundary leads of the T0-S cycle (age 0)
        bad += fetch(s, case_dir, int(spinup_hours), workers, cfg, source, None, log)
        if not (Path(case_dir) / row["file"].format(cyc=cycle.hour)).is_file():
            bad += fetch(cycle, case_dir, 6, workers, cfg, source, None, log)
        return bad
    rows = tuple(r for r in FETCH_ROWS if r["source"] == source and r["role"] in ("init", "boco"))
    bad += fetch(s, case_dir, int(spinup_hours), workers, cfg, source, spinup.spinup_age(main_age, spinup_hours),
                 log, rows=rows)
    if not (Path(case_dir) / row["file"].format(cyc=cycle.hour)).is_file():
        bad += fetch(cycle, case_dir, 6, workers, cfg, source, main_age, log,
                     rows=tuple(r for r in rows if r["role"] == "init"))
    return bad


def _jsonl_index(url: str) -> list[dict]:
    import json
    return [json.loads(l) for l in _get(url).decode().splitlines() if l.strip()]


def _jsonl_pick(rows: list[dict], select: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for sel in select:
            if r.get("levtype") == sel["levtype"] and r.get("param") in sel["param"]:
                out.append(r)
                break
    return out


def _ranges(rows: list[dict]) -> list[tuple[int, int]]:
    """Byte ranges [start, end] of the rows, adjacent records merged into one request."""
    spans = sorted((int(r["_offset"]), int(r["_offset"]) + int(r["_length"]) - 1) for r in rows)
    out = []
    for a, b in spans:
        if out and a == out[-1][1] + 1:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _era(fetch: dict, cycle: datetime) -> dict:
    t = cycle.strftime("%Y-%m-%dT%H")
    for e in fetch["eras"]:
        if t >= e["from"] and (e["until"] is None or t < e["until"]) and cycle.hour in e["cycles"]:
            return e
    raise ValueError(f"fetch_case: no publication era of {fetch['route']} holds the {t}Z cycle")


def _object_path(tmpl: str, cycle: datetime, lead: int) -> str:
    return tmpl.format(ymd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H"), lead=lead)


def _jsonl_lead(fetch: dict, cycle: datetime, lead: int, dest: Path) -> str:
    """One lead: every object of the era, its selected records (and the f000_only records of f000 for a later
    lead), concatenated into dest. Hosts are tried in order per lead."""
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name} {dest.stat().st_size / 1e6:.0f} MB"
    era = _era(fetch, cycle)
    errors = []
    for host in fetch["hosts"]:
        try:
            parts = []
            for tmpl in era["files"]:
                obj = f"{host}/{_object_path(tmpl, cycle, lead)}"
                rows = _jsonl_index(obj.rsplit(".grib2", 1)[0] + ".index")
                want = _jsonl_pick(rows, fetch["select"])
                if lead and fetch.get("f000_only"):
                    o0 = f"{host}/{_object_path(tmpl, cycle, 0)}"
                    r0 = _jsonl_pick(_jsonl_index(o0.rsplit(".grib2", 1)[0] + ".index"), fetch["f000_only"])
                    parts += [(o0, rg) for rg in _ranges(r0)]
                parts += [(obj, rg) for rg in _ranges(want)]
            if not parts:
                raise ValueError("no records selected")
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for url, (a, b) in parts:
                    f.write(_get(url, f"{a}-{b}", timeout=300))
            tmp.rename(dest)
            return f"got {dest.name} {len(parts)} ranges {dest.stat().st_size / 1e6:.0f} MB from {host.split('/')[2]}"
        except (OSError, urllib.error.URLError, ValueError) as e:
            errors.append(f"{host.split('/')[2]}: {type(e).__name__} {e}")
    raise OSError(f"{dest.name}: every host failed: {errors}")


def domain_band(margin_deg: float, cfg=None) -> str:
    """'lat_s,-180,lat_n,180': the latitude band of the run's domain (the configfile grid's H and V points, on the
    host; NAMGPU_CONFIGFILE moves it with the run) widened by margin_deg, every longitude (gpuwm's area string)."""
    import numpy as np
    from namgpu import config, grid
    g = grid.build(cfg if cfg is not None else config.load())
    lat = np.degrees(np.concatenate([g.glat_h.ravel(), g.glat_v.ravel()]))
    south = max(-90.0, float(np.floor(lat.min() - margin_deg)))
    north = min(90.0, float(np.ceil(lat.max() + margin_deg)))
    return f"{south:g},-180,{north:g},180"


def store_stop(fetch: dict) -> datetime | None:
    """The last hour a store publishes, from its bounds document (the first of bounds_stop_keys it carries, a day:
    the store grows in whole days), or None when the document cannot be read."""
    import json
    try:
        doc = json.loads(_get(fetch["bounds_url"], timeout=60))
    except (OSError, ValueError):
        return None
    for k in fetch.get("bounds_stop_keys", ()):
        if doc.get(k):
            return datetime.strptime(str(doc[k])[:10], "%Y-%m-%d") + timedelta(hours=23)
    return None


_ARCO_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_ARCO_LOCK = threading.Lock()


def _arco_time(fetch: dict, valid: datetime, area: str, dest: Path) -> str:
    """One ERA5 analysis hour through gpuwm's route (gpuwm.era5_arco.retrieve_era5_arco) into dest's own folder, at
    most fetch["max_parallel"] at once. The folder keeps gpuwm's acquisition receipt beside the file."""
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.parent.name}/{dest.name} {dest.stat().st_size / 1e6:.0f} MB"
    try:
        from gpuwm.era5_arco import retrieve_era5_arco
    except ImportError as e:
        raise ImportError("fetch_case: the era5 source is fetched by gpuwm's ERA5 route: pip install "
                          "'gpuwm>=2.9.11'") from e
    with _ARCO_LOCK:
        slot = _ARCO_SLOTS.setdefault(int(fetch["max_parallel"]), threading.BoundedSemaphore(int(fetch["max_parallel"])))
    t0 = time.perf_counter()
    with slot:
        got = Path(retrieve_era5_arco(cycle=valid, hours=0, area=area, out=dest.parent, cadence=1,
                                      progress=lambda s: None))
    if got.resolve() != dest.resolve():
        raise ValueError(f"fetch_case: gpuwm wrote {got}, the source row reads {dest}")
    return (f"got {dest.parent.name}/{dest.name} {valid:%Y-%m-%d %HZ} {dest.stat().st_size / 1e6:.0f} MB "
            f"in {time.perf_counter() - t0:.0f} s (gpuwm ARCO-ERA5, area {area})")


def _arco_jobs(name: str, row: dict, cycle: datetime, hours: int, raw: Path, cfg=None):
    fetch, b = row["fetch"], row["boundary"]
    last = cycle + timedelta(hours=hours)
    stop = store_stop(fetch)
    if stop is not None and last > stop:
        raise ValueError(f"fetch_case: source {name!r}: the store's last published hour is {stop:%Y-%m-%d %H}Z, "
                         f"this run needs {last:%Y-%m-%d %H}Z")
    area = domain_band(float(fetch["lat_margin_deg"]), cfg)
    jobs = []
    for lead in range(0, hours + 1, int(b["interval_h"])):
        dest = raw.parent / b["file"].format(bcyc=cycle.hour, lead=lead)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((_arco_time, fetch, cycle + timedelta(hours=lead), area, dest))
    return jobs


def start_jobs(name: str, cycle: datetime, raw: Path, cfg=None):
    """Jobs for the f000 start file of SOURCES row `name` only (a start_from row reads it; its boundaries are its own)."""
    from namgpu.init.sources import SOURCES
    row = SOURCES[name]
    if "fetch" in row:
        if row["fetch"]["kind"] != "ecmwf-jsonl":
            raise ValueError(f"fetch_case: start_from {name!r}: only an ecmwf-jsonl or the gfs row can be a start")
        dest = raw.parent / row["file"].format(cyc=cycle.hour)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(_jsonl_lead, row["fetch"], cycle, 0, dest)]
        for d in row["fetch"].get("donors", []):
            ddest = raw.parent / d["file"].format(hh=cycle.strftime("%H"))
            url = d["url"].format(ymd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H"))
            jobs.append((_subset, url, ddest, re.compile(re.escape(d["idx_match"]))))
        return jobs
    if name != "gfs":
        raise ValueError(f"fetch_case: start_from {name!r}: only an ecmwf-jsonl or the gfs row can be a start")
    # the gfs analysis file, as plan() fetches it for a gfs start (FETCH_ROWS row gfs, role init)
    jobs, _ = plan(cycle, 0, raw, cfg, "gfs", 0, rows=[r for r in FETCH_ROWS
                                                         if r["source"] == "gfs" and r["role"] == "init"])
    out = [(fn, *args[:-1]) for fn, *args in jobs]
    for _, *args in out:          # plan() leaves the folders to fetch(); a start_from job writes on its own
        for a in args:
            if isinstance(a, Path):
                a.parent.mkdir(parents=True, exist_ok=True)
    return out


def _whole_lead(fetch: dict, cycle: datetime, lead: int, dest: Path) -> str:
    """One lead of a "whole-files" fetch: every object of the era, whole, concatenated into dest (the AI forecasts
    publish no record subsetting: aigfs has no grib filter and its .idx sidecars cover whole small files)."""
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.name} {dest.stat().st_size / 1e6:.0f} MB"
    era = _era(fetch, cycle)
    errors = []
    for host in fetch["hosts"]:
        try:
            urls = [f"{host}/{_object_path(t, cycle, lead)}" for t in era["files"]]
            with ThreadPoolExecutor(len(urls)) as ex:
                parts = list(ex.map(lambda u: _get(u, timeout=600), urls))
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for b in parts:
                    f.write(b)
            tmp.rename(dest)
            return f"got {dest.name} {len(urls)} files {dest.stat().st_size / 1e6:.0f} MB from {host.split('/')[2]}"
        except (OSError, urllib.error.URLError, ValueError) as e:
            errors.append(f"{host.split('/')[2]}: {type(e).__name__} {e}")
    raise OSError(f"{dest.name}: every host failed: {errors}")
def _subset_pair(urls: list, dest: Path, pattern: re.Pattern) -> str:
    """The records of several objects that `pattern` selects (by their .idx), written one after another into one
    file (GEFS pgrb2a + pgrb2b of one member and lead: a lead's start record set lives in both halves)."""
    if dest.exists() and dest.stat().st_size > 0:
        return f"have {dest.parent.name}/{dest.name}"
    parts = [dest.with_name(f"{dest.name}.half{k}") for k in range(len(urls))]
    for url, part in zip(urls, parts):
        _subset(url, part, pattern)
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for part in parts:
            f.write(part.read_bytes())
    tmp.rename(dest)
    for part in parts:
        part.unlink()
    return f"got {dest.parent.name}/{dest.name} {dest.stat().st_size / 1e6:.0f} MB"


def _pair_jobs(name: str, row: dict, start: datetime, hours: int, raw: Path, cfg=None):
    """A lead_start row (gefs): the start's lead of the boundary cycle and every boundary lead after it, for the
    run's member (sources.source_member)."""
    from namgpu.domains import bc_hours
    from namgpu.init import sources as srcmod
    fetch, b = row["fetch"], row["boundary"]
    age = srcmod.boco_age_h(row, cfg, start=start)
    cyc = start - timedelta(hours=age)
    mem = srcmod.source_member(row, cfg)
    last = max(s for s, _ in fetch["steps"])
    if age + hours > last:
        raise ValueError(f"fetch_case: source {name!r} publishes leads to {last} h; this run needs {age + hours} h")
    jobs = []
    for lead in range(age, age + bc_hours(hours) + 1, int(b["interval_h"])):
        dest = raw.parent / b["file"].format(bcyc=cyc.hour, lead=lead, mem=mem)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urls = [f"{fetch['host']}/" + f.format(ymd=f"{cyc:%Y%m%d}", hh=f"{cyc:%H}", mem=mem, lead=lead)
                for f in fetch["files"]]
        jobs.append((_subset_pair, urls, dest, PATTERNS[fetch["records"]]))
    return jobs


def plan_source(name: str, cycle: datetime, hours: int, raw: Path, cfg=None):
    """Jobs for an init source row with a "fetch" entry: its start and boundary leads plus its donor fields."""
    from namgpu.init.sources import SOURCES
    row = SOURCES[name]
    fetch = row["fetch"]
    if fetch["kind"] == "arco-zarr":
        return _arco_jobs(name, row, cycle, hours, raw, cfg)
    if fetch["kind"] == "ncep-idx-pair":
        return _pair_jobs(name, row, cycle, hours, raw, cfg)
    if fetch["kind"] not in ("ecmwf-jsonl", "whole-files"):
        raise ValueError(f"fetch_case: source {name!r} fetch kind {fetch['kind']!r} has no fetcher")
    last = max(s for s, _ in fetch["steps"])
    if hours > last:
        raise ValueError(f"fetch_case: source {name!r} publishes leads to {last} h, not {hours} h")
    b = row["boundary"]
    jobs = []
    if row.get("start_from"):
        # a row whose start is another row's analysis (aifs: the IFS analysis it starts from; aigfs: the GFS
        # analysis): that row's f000 file, by that row's own fetch route
        jobs += start_jobs(row["start_from"], cycle, raw, cfg)
        if row.get("start_fallback"):
            jobs += start_jobs(row["start_fallback"]["source"], cycle, raw, cfg)
    from namgpu.domains import bc_hours
    if fetch["kind"] == "whole-files":
        for lead in range(0, min(bc_hours(hours, int(b["interval_h"])), last) + 1, int(b["interval_h"])):
            dest = raw.parent / b["file"].format(bcyc=cycle.hour, lead=lead)
            dest.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((_whole_lead, fetch, cycle, lead, dest))
        return jobs
    # the boundary times a run of `hours` reads (boco.NNNN every 3 h needs the lead 3 h after it)
    for lead in range(0, min(bc_hours(hours, int(b["interval_h"])), last) + 1, int(b["interval_h"])):
        dest = raw.parent / b["file"].format(bcyc=cycle.hour, lead=lead)
        dest.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((_jsonl_lead, fetch, cycle, lead, dest))
    for d in fetch.get("donors", []):
        dest = raw.parent / d["file"].format(hh=cycle.strftime("%H"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = d["url"].format(ymd=cycle.strftime("%Y%m%d"), hh=cycle.strftime("%H"))
        jobs.append((_subset, url, dest, re.compile(re.escape(d["idx_match"]))))
    return jobs


def main_source(a) -> int:
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    from namgpu import paths
    raw = (getattr(a, "case_dir", None) or paths.work_root() / "cases" / cycle.strftime("%Y-%m-%d_%Hz")) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = plan_source(a.source, cycle, a.hours, raw)
    t0 = time.perf_counter()
    bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fn, *args): next(x for x in args if isinstance(x, Path)) for fn, *args in jobs}
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
    total = sum(p.stat().st_size for p in futs.values() if p.exists())
    wall = time.perf_counter() - t0
    print(f"fetch {a.source} {a.cycle} +{a.hours} h: {len(jobs)} files, {total / 1e6:.0f} MB in {wall:.0f} s "
          f"({total / 1e6 / max(wall, 1e-3):.0f} MB/s, {a.workers} threads), {bad} failed")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--config", type=Path, default=None, help="domain configfile (default NAMGPU_CONFIGFILE)")
    ap.add_argument("--source", default="gfs", help="init source row (its init/boco rows are required; ifs, ifs-0p25, "
                    "era5 fetch through the row's own fetch entry)")
    ap.add_argument("--boco-age", type=int, default=None, help="boundary cycle age, h (default: the row's)")
    ap.add_argument("--case-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--follow", action="store_true", help="storm-following run: the start record set at every "
                    "boundary lead (a window move lands on a lead)")
    a = ap.parse_args(argv)
    from namgpu.init import sources as srcmod
    if "fetch" in srcmod.SOURCES.get(a.source, {}):
        return main_source(a)          # ifs, ifs-0p25: the row's own fetch entry (jsonl .index)
    from namgpu import config, paths
    cfg = config.load(a.config) if a.config else config.load()
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    case = a.case_dir or paths.work_root() / "cases" / cycle.strftime("%Y-%m-%d_%Hz")
    if a.dry_run:
        jobs, skipped = plan(cycle, a.hours, case / "raw", cfg, a.source, a.boco_age, follow=a.follow)
        for name, why in skipped:
            print(f"skip {name}: {why}")
        for fn, *args in jobs:
            print(fn.__name__, *(str(x) for x in args[:2]), "required" if args[-1] else "optional")
        return 0
    bad = fetch(cycle, case, a.hours, a.workers, cfg, a.source, a.boco_age, follow=a.follow)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
