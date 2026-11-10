# Changelog

## 0.2.0 (candidate)

- GRIB2 products are written sooner, with the same files byte for byte: each hour is taken as soon as its history
  file is complete and published as soon as its writer ends, the hand-off frame runs its longest jobs first, and the
  pinned GRIB2 writer does its shared column work once per hour and stops a failed parcel lift early.
- gpuwm-hex is optional. Only TEMPO microphysics, MYNN v6.1 and the suites that use TEMPO (`hrrrv5`, `rapv5`,
  `woof-nam-next-12km`, `woof-nam-next-7p6km`) need it (`pip install "recast-woof-nam[modern]"`, gpuwm-hex 0.5.0 or
  newer). Without it, a run that asks for one of them stops before it starts, with one line, where it used to stop at
  the physics start with a traceback. `woof-nam doctor` says whether it is installed.
- The CDF soil start and RUC with HRRR soil classes (both presets use them) fetch what they need on first use: the
  soil table of the start month (268 MB, pinned) and the BNU soil set (8 MB, through `gpuwm fetch-geog`). From a
  fresh install the run used to build its start and then stop with a traceback.
- `woof-nam run --case X` with no case folder X stops at once with one line that names the `woof-nam fetch` command
  for it; it used to stop in the start with a traceback.
- `woof-nam suite --no-products` passes when every forecast completes and keeps each nest's history; it returned 1 after
  a clean forecast, counting the GRIB2 hours it had been told not to write.
- `woof-nam suite` on 4 cards runs the CONUS nest on all four (63-68 s per forecast hour measured on 4 x RTX PRO 6000;
  it had two cards) and keeps the other nests off card 0, which holds the parent and the CONUS GRIB2 writer.
- `woof-nam run` takes the fit memory path by itself when the card cannot hold the grid on the full path (the 7.6 km
  North America grid on a 96 GB card).
- `woof-nam run` no longer leaves its history writer processes behind after a run with maps.
- `woof-nam nest` no longer dies with a traceback when `nvidia-smi` is slow to answer on a loaded host: it waits longer
  once, then stops with one line.
- The RUC land option with the HRRR soil start runs from a plain install (it looked for `grib2_dump` only in
  `~/.gpuwm/bridges`).
- The awphys and nest GRIB2 files carry the four precipitation types (CSNOW, CICEP, CFRZR, CRAIN); the compatibility pages
  are rechecked on a 0.2.0 run.
- `woof-nam fetch-bridges` also stages `rw_asos`, `rw_stage4` (for `woof-nam score`) and `rw_hex_ugwp_oro` (for the
  UGWP option), pinned by size and SHA-256; a door that needs one downloads it on first use.

- Pin the matching manylinux GRIB2 writer for the inventory additions and nest output.

- TEMPO defaults to v3.2.5 when selected; v3.0.3 remains available by name.
- `woof-nam inventory` checks every archived record, including explicit omissions and unchanged baseline messages.
- Dominant precipitation-type products use the physics-step trace-rate mask. The classification is shared by all
  four flags in the main process, and `woof-nam run` records the resolved physics step for product generation.

- New physics, each a selectable option, with the NAM suite still the default and unchanged bit for bit:
  - `--physics-suite hrrrv5`: the RRFS v2 hrrrv5 suite on the NMMB dycore; `rapv5`: NOAA RAP v5 pairing (aerosol-aware
    Thompson, Grell-Freitas, MYNN, RUC, RRTMG, GSL drag).
  - Two presets, neither a default: `--physics-suite woof-nam-next-12km` (the NAM 12 km grid with RUC land from the HRRR
    soil analysis, TEMPO, RTE+RRTMGP and UGWP drag; MYJ and BMJ kept) and `woof-nam-next-7p6km` (the same on the NAM North
    America footprint at 7.6 km, with New Tiedtke convection).
  - Single slots: RUC land surface (HRRR soil analysis start, HRRR static soil classes) and Noah-MP; a CDF soil-moisture
    start matched to the NAM soil climate by month; TEMPO (with melting-layer reflectivity), Thompson, aerosol-aware Thompson and P3 microphysics;
    RTE+RRTMGP and WRF RRTMG radiation; YSU with the revised MM5 surface layer, and MYNN-EDMF with the MYNN surface layer (also MYNN v6.1 through gpuwm-hex);
    Grell-Freitas and New Tiedtke convection; UGWP drag.
- New domains: the NAM North America footprint at 7.6 km (`na7p6`), 7.6 km CONUS (`conus7p6`) and a 16 km SREF-shape grid
  (`sref16`).
- New start sources: NOAA RAP, HRRR, RRFS v1 and GEFS v12 members (`--member`), and the ECMWF AIFS and NOAA AIGFS forecasts as
  boundaries.
- `woof-nam suite`: one cycle of the NAM suite, the 12 km parent and the CONUS, Alaska, Hawaii, Puerto Rico and fire-weather
  nests NCEP ran beside it, with NCEP-shaped hiresf GRIB2 files, on the cards of one machine (`woof-nam nest --ncep-nest` for one nest).
- `woof-nam run --start spinup`: a 6 h spin-up before the cycle, brought to the cycle analysis by centred IAU or nudging.
- `woof-nam ensemble`: a named member roster, one forecast per member; `parm/ensembles/sref_v7_nmb.json` is NCEP's SREF v7.1
  NMMB members on the 16 km `sref16` grid, each from its GEFS v12 member (`--source gefs --member`). The RAP analysis as a
  start (`--source rap`).
- `woof-nam score`: scores runs against ASOS, MRMS and Stage IV, with sheets.
- `woof-nam fetch-soil-cdf`: the monthly tables the CDF soil start reads.
- History: `SNOWNC` (grid-scale snowfall since the start) under the NAM microphysics.
- Faster step loop: the host waits on a step event instead of spinning a core, history is written by separate processes,
  and the radiation windows are batched.
- MRMS truth: the fetch takes the nearest complete mosaic, not a partial granule written during a radar dropout.
- Two runs of different months started together no longer write over one shared static file.
- Docs: NOAA's NAM nests and the whole suite (`docs/nests/`); the fire-weather nest uses NCEP's 1 h accumulation bucket.

### Details

- `woof-nam run --start spinup`: the forecast starts spun up. It begins 6 h before the cycle from that cycle's GFS
  analysis and runs the model to the cycle. The state is then brought to the cycle's analysis by a centred IAU
  (`--spinup-update iau`, the default) or by 1 h of nudging (`nudge`), so hour 0 holds clouds and precipitation.
  `woof-nam fetch --spinup-hours 6` fetches its inputs. `--start cold` (the default) is unchanged, bit for bit.

- History: `SNOWNC`, the grid-scale snowfall since the start (mm of water, the microphysics' surface precipitation times
  its frozen fraction each call, summed as WRF's SNOWNC; written when the NAM's Ferrier microphysics runs), and `SR`, the frozen fraction of the last call's surface
  precipitation. Both are in the default `products` history preset. Nothing in the forecast reads them; the forecast's
  own fields are unchanged.

- `woof-nam fetch/run --source rap`: start from NOAA's RAP analysis (13 km, North America, radar-cycled, AWS NODD
  noaa-rap-pds from 2021-02-22), the wrfprsf00 file on RAP's own rotated lat-lon grid. Table work on the shared
  isobaric path: a SOURCES row, a record table (`source_records.RAP`) and a fetch row. New in the path, for any
  source on NCEP grid template 3.32769: the grid geometry read from the file (`source_grids.py`), point placement in
  the source's rotated coordinates inside init-gfs-hinterp, and grid-relative winds turned to earth-relative. The start
  carries RAP's condensate (cloud water, rain, cloud ice, snow + graupel as Ferrier CWM, F_ICE, F_RAIN; F_RIMEF 1) and
  TKE (Q2). Outside RAP's grid (about 17 columns at the NAM grid's west and east edges) and above its 50 hPa top the
  start is the same cycle's GFS analysis (row `fallback`, blended over 8 RAP cells and 20 hPa). Boundaries are GFS's,
  6 h older, as the native start. Soil: RAP's nine RUC nodes read at Noah's layer midpoints (gpuwm's own RAP/HRRR
  rule, no rescaling between the RUC and Noah soil tables); `--soil-from gfs` takes GFS's Noah soil instead.

- Two runs of the same case now give the same forecast, bit for bit. The terrain statistics the gravity-wave drag
  reads (standard deviation, convexity, slope, anisotropy and angle of the sub-grid orography) were summed with
  floating-point atomics, in a different order on every run, so about a fifth of the 12 km cells differed in the
  last bit between two runs. The digital filter spread that over the domain before the 0 h output, and convective
  cells grew it to several kelvin and millimetres within hours. The sums are now gathered per cell in a fixed
  order, as is the vertical-coordinate fit. Measured on the 12 km grid (2026-11-03 06Z GFS start, digital filter,
  all physics, 6 h, one card): two runs before the fix differed in 3,140,115,881 of 6,295,450,910 history values;
  after it, in none. No change in speed.

## 0.1.3 (2026-11-05)

- Radiation no longer hangs when it runs in 2-row chunks. A chunk of 2 rows owned no row, so the chunk loop never
  advanced and grew in host memory until the machine ran out; on an 8 GB card every 3 km nest from about 361 x 360 to
  405 x 405 points was given 2-row chunks by the nest door and hung at the first radiation call. Chunks now need 3 rows
  or more, the door picks 32, 16, 8 or 4 rows, and `--rrtm-rows` below 3 (or `NAMGPU_RRTM_ROWS` below 3 on
  `woof-nam run`) is refused in one line before the run starts.

- The nest door's memory estimate matches the card. It was about 2.5 GB above the measured peak at every size on an
  8 GB card, so the door gave smaller radiation chunks than the card could hold (twice the step time) and refused
  nests that fit. It now covers every measured peak: 1-4 % above at 4- and 8-row radiation chunks, 15-17 % above at 16, and the door
  still keeps 0.75 GiB free beyond it. On an 8 GB card the door now admits nests up to about 490 x 490 points; 378 x 377
  and 415 x 414 ran at 94.9 and 112.9 ms per step (0.1.2: 360 x 359 at 142.8).

- `woof-nam run` checks the card before it starts: a grid that cannot fit (the 12 km grid on an 8 GB card) is refused
  in one line with the estimate and the card's size, where 0.1.2 stopped with a CuPy out-of-memory traceback.

- `woof-nam nest --source era5`: nests start from the ERA5 reanalysis, as `woof-nam run` does. A source with no row
  is now refused before any stage, `--dry-run` included.

- `woof-nam products` on a nest's history says in one line that the awphys GRIB2 files are cut from the 12 km CONUS
  grid, instead of an IndexError for every hour.

- New `woof-nam doctor`: the card, CuPy and NVRTC (a test kernel), the terrain data, the map renderer, the GRIB2
  writer, and what the card can run by the doors' memory estimate (the 12 km grid, 100 x 100 and 333 x 333 nests,
  the largest square nest).

- A case folder whose name is not the start time (YYYY-MM-DD_HHz) is refused in one line by `woof-nam run`.

- `nest-receipt.json` reads the card's memory and load in containers that hide the card's process ids: it read
  0.0 GiB and flagged the busy forecast as a single-core stage there.

- README: memory and speed measured on an 8 GB card (RTX 3060 Ti, fresh `pip install recast-woof-nam==0.1.2`): 3 km
  nests of 99 x 100, 328 x 330 and 360 x 359 points peak at 2.0, 4.1 and 4.2 GB ("Speed and memory").

- Composite and simulated reflectivity now match how NOAA's NAM post computed them. The history `COMPOSITE` and
  `REFL_10CM`, the maps, and the awphys GRIB2 `REFC` (record 3) and `REFD` (records 2, 295, 296) add the reflectivity
  of the convection scheme's rain (Z = 300 R^1.4, full strength below the freezing level, 20 dBZ weaker at the
  convective cloud top) to the microphysics reflectivity, as the NAM post did. Before, they carried the microphysics
  part only and ran 3-4 dB colder than the NAM's own REFC at 30 dBZ for the same state. Columns without convective
  rain are unchanged. The hourly maximum `REFDMAX` already had the term and is unchanged.

- `--source native` starts from the NAM analysis as the operational NAM forecast did. The analysis' rain and ice
  fractions and rime factor, its turbulent kinetic energy and its vertical velocity are kept (before, the rain and ice
  started as cloud water, the TKE at its floor and the vertical velocity at 0), and no second digital filter runs
  over it, because the NAM analysis file is already the NAM's filtered 0 h state. `--filter on` restores the filter;
  `--filter off|on` works for every source. GFS, ECMWF, ERA5 and parent starts are unchanged.

## 0.1.2 (2026-11-05)

- Nests run one-way by default, as NOAA's NAM did: a live parent (`--parent-center`, `--parent-size-km`,
  `--parent-size-y-km`) without `--two-way` now runs as `--concurrent`. `--two-way` enables feedback; two-way made
  Ian 27 hPa deeper than observed in our test (908.8 hPa against 937). In 0.1.1 a live parent ran two-way unless
  `--concurrent` was given.

- Maps of a domain across 180 (a nest over the Aleutians or Fiji) are no longer drawn wrong on gpuwm 2.9.12. That
  release's renderer accepts such a frame but smears every row east-west, so with it the map is not drawn and the
  log says to set `NAMGPU_RENDERER` to the patched renderer (`tools/renderer/install_renderer.sh`). The 12 km
  parent still draws on the CONUS frame, as gpuwm 2.9.12 also smears its own whole-circle frame. woof-nam keeps
  working with gpuwm 2.9.11 and 2.9.12.

- `woof-nam products` works on a plain `pip install`. In 0.1.1 it stopped every hour with `FileNotFoundError: ...
  rw_rrfs_upp ... not in a gpuwm release yet`, because the GRIB2 writer it calls ships in no gpuwm release. woof-nam
  now carries its own pinned writer: the first `woof-nam products` downloads `rw_rrfs_upp` (Linux x86-64, 13 MB,
  glibc 2.28 or later) from the woof-nam release and checks its size and SHA-256 before installing it. New command
  `woof-nam fetch-bridges` (`--from FILE`, `--where`). If the writer cannot be had, `products` and `run --products`
  stop at once with one line naming it and the command that fetches it, before any hour is read or any forecast
  step runs.

- `woof-nam nest --follow tc | low | object:refl | object:uh | path`: a nest that follows a storm. Start it with
  `--storm LAT,LON` or `--storm-id` (an IBTrACS id). The window moves by whole grid cells at the source's boundary
  times. The nest keeps its own state where the old and new windows overlap, and the new strip starts from the
  source. Maps and sheets follow the window. On Typhoon Ampil (48 h from GFS) the mean track error against IBTrACS
  was 62 km, against 95 km for the same nest held still, which lost the storm at +30 h. README "Moving nests".

## 0.1.1 (2026-11-05)

- `woof-nam nest --two-way` and `--concurrent`: a 12 km parent and a 3 km (or 4 km) nest on the parent's own grid, run
  together in one process on one card. The nest takes its lateral boundaries from the live parent every parent step
  (NMMB's PARENT_UPDATE_CHILD_PSFC / _BNDRY and UPDATE_BC_TENDS), as the operational NAM nests ran (`--concurrent`).
  `--two-way` adds NMMB's two-way feedback: box means of the nest's T, Q, CW, U and V replace the parent's in the
  overlap (GENERATE_2WAY_DATA, PARENT_2WAY_UPDATE; `--2way-wgt`, default 1.0). History for both domains in
  `<out>/parent/` and `<out>/nest/`. A live parent runs two-way by default (any `--parent-*` option); the
  `--source parent --parent-run` hand-off and the one-domain door stay one-way.

- On PyPI as `recast-woof-nam` (`pip install recast-woof-nam`); the command stays `woof-nam` and the import name
  stays `namgpu`.
- 3 km nests anywhere (`woof-nam nest`), from GFS or from a WOOF-NAM 12 km parent run.
- GRIB2 products in the NAM awphys shape (`woof-nam products`).
- ERA5 (`--source era5`) and ECMWF IFS (`--source ifs`, `ifs-0p25`) starts, beside GFS and the archived NAM analysis.
- Restarts: `woof-nam run --restart-every N` and `--resume`, byte-identical to a straight run.
- SPD10MAX is the hourly maximum 10 m speed in m/s; wind maps over 60 kt use a 0-140 kt scale.
- Maps draw coastlines, borders and states from a bare pip install (the basemap ships in the wheel).
- `woof-nam licence-check` finds the installed distribution that ships namgpu, whatever its name.

- Less GPU memory, same output: one device memory pool per run that grows only when a request needs it (never past
  `CUPY_GPU_MEMORY_LIMIT` when that is set) and hands unused blocks back after each init stage, radiation call and
  history write; the digital filter keeps its save arrays in host memory and frees everything it owns when it ends; the
  3-D history fields are computed one at a time into one buffer as they are written; init-only scratch is dropped
  after the init; banded column physics and radiation chunks free their arrays as soon as a call ends; the radiation
  and MYJ/BMJ table builds keep their work arrays in global memory. The per-thread local-memory reservation the step
  loop needs is 10,016 bytes (2.44 GiB on an RTX 5090, was 11,264), and `woof-nam run --init-from` and `--resume`
  set it before the load, as a fresh start does. Every history value is unchanged.
- Restart sets on the full memory path (cards of 64 GiB and more): a set leaves out the step, history, unit and
  init-only scratch, the RRTM chunk context and driver-prep tables and the init-only 30-arcsec terrain, exactly what the
  32 GB fit path has already released at the hour (`restart.transient`). Every one of them is written before it is read
  in a step, and the resumed run allocates them again on first use as a fresh start does. A 12 km set on a 96 GB card
  was 36.7 GiB with them; it is now the size of a fit-path set.
- `woof-nam run --resume` from a full-path set no longer stops with `AttributeError: '_Skipped' object has no attribute
  'w'`: an entry the set cannot hold (the RRTM chunk context) is left unset, so the radiation call builds it as on a
  fresh start; sets written before this fix load the same way.
- `woof-nam nest`: the CuPy pool cap follows the memory path the forecast takes on the card (the 32 GB fit below
  64 GiB of card, the full path at and above, `NAMGPU_MEMORY_PATH` as set). A 12 km parent on a 96 GB card ran out
  of memory under a cap sized for the fit path. When the card size cannot be read and no path is set, no cap is set.
- `woof-nam nest` sizes its memory check from this release's memory use: a 333 x 333 nest is admitted on an 8 GB
  card (the check asked for about 18 GiB before), and common cards are counted by their own SM count.
- `woof-nam nest --source parent --parent-run DIR` takes the parent door's `--out` folder or its `run/` folder (the
  one holding `wrfout_d01_*`); the refusal names the folders it looked in.

## 0.1.1.dev2 (private build, not released)

- `woof-nam run --domain conus5`: the 5 km CONUS grid (1097 x 881 x 60, GFS start), a row of `parm/domains.json`.
- Static surfaces built from WPS_GEOG (conus4, conus5) read the permissive soil set by default: SoilGrids texture
  categories and the ERA5 deep-soil temperature, as gpuwm 2.9.4 does. `NAMGPU_GEOG_SOIL=wrf` (or
  `static_build --soil wrf`) selects the WPS STATSGO/FAO set. A cached static built from another soil set is rebuilt.
- Each WPS_GEOG dataset is looked up across NAMGPU_GEOG, the tree its topo_gmted2010_30s link points into,
  WPS_GEOG and ~/WPS_GEOG, so a geography folder that holds only the terrain link still builds a 4 or 5 km static.

## 0.1.1.dev1 (private build, not released)

- `woof-nam run --domain NAME` (and `init`): grids are rows of `namgpu/parm/domains.json`; `conus4` (4 km CONUS,
  GFS start) runs from the wheel. Its static surface is built from WPS_GEOG on first use and per start month.
- A GFS start on the 12 km grid with no archived analysis in the case and no `static/nam12_static.npz` builds that
  file once from the archived NAM analysis of the same month and day.
- `woof-nam fetch --gfs-only` also fetches the start's own GFS analysis (f000); before, a fetched GFS case could not
  start. `--hours N` fetches exactly the GFS boundary times an N-hour run reads (at least 6 h, 3-hourly), and the
  init chain fetches for the run's own length (it fetched 48 h). `--inputs-only` skips the comparison files (MRMS,
  archived NAM forecast). Each fetch writes `<case>/fetch-receipt.json` (seconds, bytes, requests).
- Boundaries for a run whose length is not a multiple of 3 h cover its last hours.
- `forecast-receipt.json` has a `stages` table (wall seconds and parallelism per stage: static, init, filter,
  forecast, maps), plus domain, grid, cards and version. The memory-path log line no longer prints the card size.

## 0.1.0 (2026-11-04)

- First package of the NAM GPU port: NMMB dynamics and the NAM 12 km physics suite on one card,
  native-analysis and GFS starts, hourly history, Rust-rendered maps, timing receipt.
- `woof-nam` command: fetch, static, init, run, maps, side-by-side, licence-check (the `python -m` forms
  still work).
- Everything a run reads ships in the wheel: NAM parm files, NOAA fix files, Noah tables, UNITS.json,
  the reused ArWen files. Working files go to the folder the command runs in (or NAMGPU_ROOT).
- Side-by-side: WOOF-NAM panels are drawn on the same CONUS frame and image size as the MRMS and
  archived NAM panels, so the three compose into one sheet.
