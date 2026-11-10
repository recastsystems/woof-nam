# WOOF-NAM nest GRIB2 files and NOAA's NAM nest files

NOAA's NAM ran five nests inside its 12 km parent and wrote one GRIB2 file per nest and forecast hour, `nam.tHHz.<nest>.hiresfFF.tm00.grib2`. WOOF-NAM writes the same files for the same nests, named `woof-nam.tHHz.<nest>.hiresfFF.tm00.grib2`, each with a `.idx` beside it (what `wgrib2 -s` prints). This page says, record by record, how close ours are to NOAA's, and what a site or converter that read NOAA's files changes to read ours. The 12 km files are covered by [../COMPATIBILITY.md](../COMPATIBILITY.md).

| nest | `<nest>` | forecast hours | output grid (NOAA's) |
|---|---|---|---|
| CONUS 3 km | `conusnest` | 00-60 | Lambert conformal 1799 x 1059 at 3 km (the HRRR CONUS grid) |
| Alaska 3 km | `alaskanest` | 00-60 | polar stereographic 1649 x 1105 at 2.976 km |
| Hawaii 3 km | `hawaiinest` | 00-60 | Mercator 321 x 225 at 2.5 km |
| Puerto Rico 3 km | `priconest` | 00-60 | Mercator 544 x 310 at 2.5 km |
| Fire weather 1.5 km | `firewxnest` | 00-36 | Lambert conformal at 1.27 km: the box around the nest on NCEP's 1.27 km master grid, so it moves with the nest's centre. `woof-nam.tHHz.firewxnest_center_latlon` gives the centre of each cycle. |

## How we checked

`tools/nest_compat_docs.py` compares WOOF-NAM nest files with NOAA's files from its last complete NAM forecast, 2026-11-03 06Z. The WOOF-NAM files come from 3-hour runs started from GFS for the same cycle (f00-f03): NCEP's own Hawaii, Puerto Rico and fire-weather nests (woof-nam nest --ncep-nest), a window of NCEP's Alaska nest, and a 3 km nest over part of the CONUS grid written with the conusnest product set. Where a run covers only part of a grid, its file still holds the whole grid with the rest marked missing; that changes neither the records nor the grid section. For each nest it decodes hour 01 of both files in full with ecCodes and goes through NOAA's records in NOAA's order (`tools/grib2_compat_nest.py`). For each record it checks whether ours has it, its inventory text, every key a decoder selects on (name, level, time range, statistic, units, the GRIB2 codes, bitmap), its record number (wgrib2's `N`, or `N.1`/`N.2` for the two winds of one message), its packing template and the grid section. Then it compares the `.idx` of every hour both sides have (f00, f01, f02, f03), line by line without the date. That checks the record list and its time windows at f00 and at the later hours. Values are not compared here. Every number on these pages comes from that run.

## Results

| nest | NOAA's records (f01) | ours | ours with NOAA's text and keys | NOAA's not in ours | grid section | packing | record list at every hour |
|---|---|---|---|---|---|---|---|
| CONUS | 876 | 799 | 799 | 77 | identical | as NOAA's | NOAA's, in NOAA's order, with 77 left out |
| Alaska | 871 | 794 | 794 | 77 | identical | as NOAA's | NOAA's, in NOAA's order, with 77 left out |
| Hawaii | 871 | 794 | 767 | 77 | identical | 5.3 (NOAA's 5.40) | NOAA's, in NOAA's order, with 77 left out |
| Puerto Rico | 871 | 794 | 794 | 77 | identical | 5.3 (NOAA's 5.40) | NOAA's, in NOAA's order, with 77 left out |
| Fire weather | 871 | 794 | 794 | 77 | identical | 5.3 (NOAA's 5.40) | NOAA's, in NOAA's order, with 77 left out |

Each nest's page lists every record: [CONUS](COMPATIBILITY-conusnest.md), [Alaska](COMPATIBILITY-alaskanest.md), [Hawaii](COMPATIBILITY-hawaiinest.md), [Puerto Rico](COMPATIBILITY-priconest.md), [Fire weather](COMPATIBILITY-firewxnest.md).

## What works unchanged

- **The grids.** Each file's grid section (GRIB2 section 3) is identical to NOAA's for the same nest (for fire weather, for the same centre). Every reader gives the same latitude and longitude at every point.
- **The records we write.** They come in NOAA's order, each with NOAA's inventory text and every GRIB2 key a decoder selects on (CONUS, Alaska, Puerto Rico, Fire weather; the exception is listed under "NOAA's records that are not in our files, or differ"). So `wgrib2 -match` patterns, pygrib `select` and cfgrib `filter_by_keys` written for NOAA's files find the same records.
- **The record lists, at every hour.** At f00, f01, f02, f03 each `.idx` is NOAA's `.idx` with the same records left out and nothing else changed: same lines, same order, same time text. f00 has NOAA's analysis-time list; the later hours have NOAA's forecast list.
- **The time windows.** They are NOAA's: hourly maxima and minima cover the last hour (F-1 to F); accumulations and averages run in NOAA's 3-hour buckets (0-1, 0-2, 0-3, 3-4, ...); in the fire-weather nest, which NCEP emptied every hour, over the last hour (0-1, 1-2, 2-3, ...); at f00 they are NOAA's 0-0 windows.
- **U and V.** Each U/V pair (and USTM/VSTM) is one two-field message, as in NOAA's files (`N.1`/`N.2` in the `.idx`), and the winds are grid-relative, as NOAA's.
- **Missing values.** Points of the output grid outside the nest are marked missing with a bitmap, as in NOAA's files; readers give them as NaN.

## What a site must change

- **The file name** starts with `woof-nam.` where NOAA's starts with `nam.`.
- **The generating process number** is 254 (WOOF-NAM) where NOAA's is 84 (NAM). Only a script that filters on it changes.
- **Record numbers.** Our files leave out the records we do not write yet (below), and a record number counts the records in the file. So after the first record left out (CICE 1 hybrid level: NOAA's record 4 or 5) our numbers run behind NOAA's. Select records by their inventory text (variable, level and time, as `wgrib2 -match` does), not by NOAA's record number. Byte ranges come from our own `.idx`, as for any GRIB2 file.
- **Packing.** Hawaii: complex packing with spatial differencing (template 5.3) where NOAA's is JPEG 2000 (template 5.40); Puerto Rico: complex packing with spatial differencing (template 5.3) where NOAA's is JPEG 2000 (template 5.40); Fire weather: complex packing with spatial differencing (template 5.3) where NOAA's is JPEG 2000 (template 5.40). A pipeline that fixes the codec per file family sees one template per nest, the same in every hour and cycle.
- **Packing that is unchanged:** CONUS complex packing with spatial differencing (template 5.3); Alaska complex packing with spatial differencing (template 5.3), as NOAA's.

## NOAA's records that are not in our files, or differ

**0.2.0 writes six more records on every nest:** the four precipitation types (CSNOW, CICEP, CFRZR, CRAIN), the cloud-top temperature (TMP cloud top) and the LCL pressure (PRES level of adiabatic condensation from sfc). So 71 of NOAA's records are left out, not 77, and each file has six more records than the tables on this page and the nest pages show (CONUS 805, the others 800). Checked with `woof-nam inventory` against NOAA's 2026-11-03 06Z f01 files for CONUS, Alaska, Hawaii and Puerto Rico: 71 named, none unexplained. The tables and the list below were made with 0.1 files and still count those six as left out.

The same 77 records are left out of every nest (CONUS 77 of 876, Alaska 77 of 871, Hawaii 77 of 871, Puerto Rico 77 of 871, Fire weather 77 of 871), at every hour compared: BMIXL 1 hybrid level; BRTMP top of atmosphere; CCOND surface; CFRZR surface; CICE (41 levels); CICEP surface; CRAIN surface; CSDSF surface; CSNOW surface; GFLUX surface; HGT highest top level of supercooled liquid water layer; HGT lowest bottom level of supercooled liquid water layer; LRGHR entire atmosphere (considered as a single layer); LTNG surface; LWHR entire atmosphere (considered as a single layer); MDIV (4 levels); PRES grid scale cloud bottom level; PRES grid scale cloud top level; PRES level of adiabatic condensation from sfc; RCQ surface; RCS surface; RCSOL surface; RCT surface; RSMIN surface; STRM 250 mb; SWHR entire atmosphere (considered as a single layer); TCOLI entire atmosphere (considered as a single layer); TCOLM entire atmosphere (considered as a single layer); TMP cloud top; UFLX surface; UPHL 5000-2000 m above ground; VFLX surface; VIS cloud top; VWSH tropopause. A record we do not write is left out, never written as zeros or under another record's name.
- **Hawaii:** 27 records, the flight-level TMP, UGRD and VGRD, have a bitmap in NOAA's file and none in ours: NOAA's whole Hawaii grid lies inside the nest, so its only missing points are those below the ground (next bullet).
- **Flight levels below the ground.** TMP, UGRD and VGRD at 305-3658 m above mean sea level: NOAA's files mark the points where that level lies below the ground missing; ours carry a value there (points missing at 305 m beyond those missing at 2 m: CONUS NOAA 789,537, ours 0; Alaska NOAA 322,142, ours 0; Hawaii NOAA 1,795, ours 0; Puerto Rico NOAA 6,592, ours 0; Fire weather NOAA 132,923, ours 0). Read these records above the ground only, or mask them with the surface height.

## Precision

NOAA's nests chose each record's decimal and binary scale per file from the data: the same record carries different scales in NOAA's CONUS and Alaska files of the same hour. So precision is listed apart from the keys, as our packing step (2^E / 10^D) against NOAA's in the file compared: CONUS: 488 same, 158 finer, 5 coarser, 148 constant fields; Alaska: 542 same, 144 finer, 29 coarser, 79 constant fields; Hawaii: 524 same, 9 finer, 86 coarser, 175 constant fields; Puerto Rico: 471 same, 10 finer, 174 coarser, 139 constant fields; Fire weather: 529 same, 19 finer, 124 coarser, 122 constant fields. Each nest's page names the records where ours is coarser.
