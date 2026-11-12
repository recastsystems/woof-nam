# WOOF-NAM GRIB2 files and NOAA's NAM files

WOOF-NAM writes one GRIB2 file per forecast hour in the shape of NOAA's NAM 12 km CONUS product,
`nam.tHHz.awphysFF.tm00.grib2`. Our file is `woof-nam.tHHz.awphysFF.tm00.grib2`, with a `.idx` beside it. A site or
script that read NOAA's NAM files should read ours with only the changes listed under "What a site must change".

## How we checked

We read NOAA's archived f06 (2025-03-14 12Z) and our f06 for the same case the way model sites do. We used three
tools:

- wgrib2 3.8.0: `-s`, `-grid`, `-match` and `-netcdf`;
- pygrib 2.1.8: `select` by shortName, typeOfLevel and level;
- xarray with cfgrib 0.9.15 and eccodes 2.50: `open_dataset` with `filter_by_keys`.

We read 2 m temperature, the 10 m winds, composite reflectivity, both sea-level pressures (PRMSL and MSLET), the
0-6 h precipitation, the 500 mb height and surface CAPE. We also compared every record's GRIB2 keys with NOAA's.
`tools/grib2_compat_table.py` repeats the record check on any pair of files.

## What works unchanged

- **The grid.** The grid section is byte-identical to NOAA's: NCEP grid 218, Lambert conformal, 614 x 428 points,
  Dx = Dy = 12191 m, earth shape 6 (sphere of 6,371,229 m), first point 12.19 N 226.541 E, orientation 265 E,
  true latitude 25 N, scanning mode and flags as NOAA's. wgrib2, pygrib and cfgrib give the same latitude and longitude
  at every point as they do for NOAA's file.
- **Names, levels and times.** Every record has NOAA's parameter codes, levels, time ranges and order. Its `wgrib2 -s`
  line is NOAA's, apart from the date. So `-match` patterns written for NAM find the same records.
- **The .idx.** The `.idx` is exactly what `wgrib2 -s` prints, the same as NOAA's index files. Tools that download
  byte ranges from the index work with the same search strings.
- **Precision.** Each record is packed with NOAA's decimal and binary scale factors for that record and the same
  complex packing: all 444 records of the 2025-03-14 f06 check, and 408 of the 450 of the 2026-01-23 f06 check (the
  other 42 differ in a scale factor; see "Not in the files" and the table). Records are within 15 % of NOAA's size.
- **Accumulations.** Precipitation buckets match NOAA's. APCP and ACPCP run from the last 12 h reset, and the 3 h
  bucket appears in the 3-hourly files, with NOAA's time-range template.
- **Python.** pygrib `select` and cfgrib `filter_by_keys` return the same variable names, units, level types and
  steps as they do on NOAA's file. As with NOAA's file, cfgrib needs a `filter_by_keys`, because the file mixes level
  types.

## What a site must change

- **The file name.** The file name starts with `woof-nam.` where NOAA's starts with `nam.`.
- **The process number.** The generating process number in each record is 254 (WOOF-NAM) where NOAA's is 84 (NAM).
  Only a script that filters on that number needs to change.

## Not in the files

The table in `COMPATIBILITY-TABLE.md` gives the state of each of NOAA's 454 records, from a 0.2.0 run (2026-01-23 12Z,
f06): 450 are present with NOAA's name, level and time keys, and 4 are not produced. Of the 450, 408 also carry NOAA's
decimal and binary scale factors exactly; 42 differ in one or both (16 TKE, 13 HGT, 4 VVEL, 3 UGRD/VGRD pairs, DPT at
2 m, SNOD, LHTFL), and the table gives NOAA's value and ours for each.

- **Records we do not produce (3).** Lightning (LTNG), visibility at cloud top and the Blackadar mixing length (NOAA's
  file holds zero there). The snowfall accumulation (WEASD, NOAA's 12 h window) is written from 0.2.1, rebuilt from
  the model's 3 h snowfall bucket in the history files.
- **Written from 0.2.0.** The dominant precipitation type (CSNOW, CICEP, CFRZR, CRAIN), and MAXUW and MAXVW (the
  hourly 10 m maximum wind).

A record that is missing is left out of the file. It is never written as zeros or under another record's name.
