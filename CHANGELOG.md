# Changelog

## 0.2.2

Fixed: a 1.5 km nest inside a live 12 km parent (published 0.2.0 and 0.2.1)
- In 0.2.0 and 0.2.1 (the same code; shown on 0.2.1), `woof-nam nest --res-km 1.5` with a live 12 km parent (`--concurrent`, `--two-way`, or a parent
  given with `--parent-center`, `--parent-size-km` or `--parent-size-y-km`) read the nest's edges from the wrong part of
  the parent. The couplers interpolated the parent onto the nest with the time ratio (12 nest steps to a parent step)
  where the space ratio (8 nest cells to a parent cell) was needed. They now interpolate on the space ratio and step on
  the time ratio.
- How to tell whether a run was affected: its log line `two-way: parent ... ratio 12, ...` names one ratio; 0.2.2
  prints `ratio 12 in time and 8 in space`. Only nests whose two ratios differ were wrong, which in these versions is a
  1.5 km nest in a 12 km parent. 3 km and 4 km nests in a live 12 km parent (ratios 4 and 4, 3 and 3), NCEP's CONUS,
  Alaska, Hawaii and Puerto Rico nests, `woof-nam suite`, `--source parent` and every nest without a live parent were
  right. 0.1.x had no 1.5 km nest.
- What it looked like: a 1.5 km nest of 240 km inside a live 12 km parent of 1,200 km, on the South Pacific tropical
  cyclone of GFS 2026-11-17 00Z (`woof-nam nest --concurrent --center 9S,163W --res-km 1.5 --size-km 240
  --parent-size-km 1200 --parent-size-y-km 1200 --source gfs --start 2026111700 --hours 1`). At +1 h, on 0.2.1: up to 16.6 m/s of vertical
  motion in the nest's outer 12 rows (6.8 m/s inside), the nest's surface pressure 5.6 hPa from the parent's on those
  rows (mean of the absolute difference), and the nest's lowest pressure 62 km from the parent's. On 0.2.2: 3.4 m/s
  (4.6 m/s inside), 0.24 hPa and 6 km.
- `woof-nam nest --ncep-nest firewxnest --concurrent` is refused in words. In 0.2.0 and 0.2.1 it coupled the
  fire-weather nest to the 12 km parent at the cell numbers of NCEP's 3 km nest, which holds it in NCEP's suite, so the
  nest took its boundaries from parent cells that do not hold it. The fire-weather nest runs one-way as before
  (`woof-nam suite`, or `woof-nam nest --ncep-nest firewxnest` from the source's boundaries).
- A run with a live parent draws its maps. In 0.2.1 such a run stopped at `maps nest rc 1` with no maps, and the
  earlier versions with this door have the same code (the history was right; `woof-nam maps --out DIR
  <run>/nest/wrfout_d01_*` draws the maps afterwards).

Added
- Ladders: nests inside nests in one process, `woof-nam nest --concurrent --ladder RES:SIZE[,RES:SIZE...]`, e.g.
  12 -> 3 -> 1.5 km. Each level steps inside the one outside it (a 3 km level takes 4 steps to a 12 km step, a 1.5 km
  level 3 steps to a 3 km step) and is written to its own folder (`parent/`, `nest/`, `nest2/`), with maps.
  `--parent-res-km 3` starts the ladder at 3 km (for example IFS 0.1 degree -> 3 -> 1.5 km), and a step can take its
  own north-south size (`1.5:300x400`). Each level must be a whole number of times finer than the one outside it;
  anything else is refused in words. `--two-way` and `--ladder` together are refused.
- A storm-following nest inside the live parent: `woof-nam nest --concurrent --move-nest`, off by default. The nest
  moves by whole parent cells when the storm (the sea-level-pressure minimum after a 60 km smooth) is more than 3 parent
  cells from its centre, checked every 15 minutes of model time. Kept cells are carried over bit for bit; new ground is
  filled from the live parent at that step, on the nest's own terrain. Before the forecast the door builds the static
  fields of the whole area the nest may visit (`--move-reach-km`). One-way only: two-way feedback during moves, a
  ladder inside a moving nest, and a moving nest on several cards are refused in words. With `--move-nest` off a
  fixed nest gives the same bytes as before. `--follow` (the moving nest that runs one-way from GFS, IFS or a
  WOOF-NAM 12 km run) is unchanged.
- A run can start from a GFS forecast hour instead of an analysis: `--lead H` on `woof-nam fetch`, `run`, `nest`
  and `suite`. The run starts at the GFS cycle plus H hours, reads its start state from that forecast file (soil,
  snow, sea ice, canopy water, skin temperature and the hydrometeors are in every GFS forecast file) and takes the
  same GFS run's later hours as boundaries. `woof-nam fetch --at TIME` starts at a chosen time: the analysis when
  there is one, else the newest posted GFS run that reaches it. The fetch prints the run and hour it chose and
  writes them to `<case>/raw/START.json`, which `woof-nam run` reads.
- The rule is one function for every command (`namgpu/init/window.py`). An hour the source does not post is not
  refused: the run starts at the nearest posted hour before it and runs that much longer, and one line says so
  (GFS hour 200: start at 198, 2 h longer, first output at hour 200; hour 100 for 24 h: start at 99). A window past
  the last posted hour is cut there. It stops only when nothing can run: a date outside the source's archive, or a
  time no run reaches, naming the sources that do. GFS runs before 2015-01-15 have no forecast files.
- Every source takes `woof-nam fetch --at TIME` or `--cycle C --lead H`, and `woof-nam nest --at TIME` or
  `--start T --lead H` (`woof-nam run` reads the fetch's choice; `woof-nam suite` takes `--cycle C --lead H` of a
  GFS run): GFS and GEFS from their forecast hours (GEFS past 240 h takes its edges every 6 h), the other sources from their newest analysis at or before the
  time (hourly for ERA5, RAP, HRRR, RRFS; RAP, HRRR and RRFS take the GFS forecast hour of the same valid time
  around them). A time may have minutes. `woof-nam sources [--json]` lists every source with its archive and posted
  hours. The 12 km domain rows and the nest door take every source (the nest door's own list is gone).
- Output times: `woof-nam run --write SPEC` (`1h`, `15m`, `0,1.5,3,6`, `0-6/15m,6-24/1h`), `--maps-at`, `--grib-at`,
  on run, nest (`--write`) and suite (`--write`, `--nest-write`). A schedule beside `minutes_history`, which the
  convection driver reads and which stays 60, so the forecast is the same whatever is written. The run adds the
  history times its products read (rain bucket starts, the frames before each rain map, a nest's edge hours),
  written lean, never drawn or exported. `<out>/OUTPUTS.json` lists them all; the maps, the GRIB2 follower, the
  column store and a resume read it. GRIB2 stays on whole hours. `woof-nam compare --shared` compares two runs of
  one case written at different times.
- The receipt (`source_cycle`, `source_lead_h`, `start_is`) and the start log name the GFS run and forecast hour,
  and every map's title or subtitle names the forecast hour. Nothing calls such a start an analysis.
- Proof run (one 32 GB card): the 12 km grid for 9 h from hour 201 of the 2026-11-12 12Z GFS run (start valid
  2026-11-20 21Z) with GRIB2 products.

Fixed
- Maps leave a grid's boundary rows off. The outer LNSH rows (5) on each side hold the start's boundary values,
  written there every step, not the forecast, and the next two rows the seam between them. A 3 km box over the
  southern Plains from GFS 2026-11-13 00Z showed a band of stronger 10 m wind along its edges (east edge at +3 h:
  9.3 to 10.0 kt in the outer five rows, 4.0 to 5.0 kt from the seventh row in) and a warm strip along its west
  edge. Against METARs over hours 1 to 20, stations in the outer five rows were 7.6 F too warm and 4.1 kt too windy
  (60 pairs); stations 30 or more rows in, +0.2 F and -0.4 kt (691). Every map (`woof-nam maps`, a run's and a
  nest's own maps) is now drawn on the grid less LNSH + 2 rows a side, framed on what it draws. The history, GRIB2
  and column store keep every row. The history now carries LNSH and LNSV; an older history is read as LNSH 5.
- Dense output no longer stalls the forecast. With `--write 5m` every frame was a whole products history (3.3 GB,
  33 s to compress on one core), at most four writers ran, and the forecast waited for them: on the 12 km grid on one
  RTX PRO 6000 (16 cores) a model hour took 102.7 s against 27.3 s hourly, 6.8 s for each extra frame. Now the
  forecast hands each frame to a queue in shared memory bounded by host memory and goes on; with more than two
  frames a model hour the writers take every core but four. A frame off the whole hour that no GRIB2 file, nest or
  carrier reads (a light frame, marked in `OUTPUTS.json`) holds only what the maps read (0.68 GB in place of 1.33 GB).
  Same box: 28.2 s for that model hour, under 0.1 s a frame; the history at the shared hours and the GRIB2 files are
  bitwise identical to the hourly run's. A run whose writers cannot keep up says so in one line and waits; the
  receipt's `history.waits` gives the total. Named `--history-fields` are kept at every time.
- `--write 5m --products` ended MAPS ERROR, rc 1: a frame off the hour took its hour's MRMS file for the reflectivity
  overlay and the renderer refused it. Off the hour the map is drawn without the overlay.
- `woof-nam columns` skips light frames (they hold no whole column).
- GFS files for starts from 2021-02-26 00Z to 2023-02-02 18Z come with their cloud water again. NCEP's index names
  GFS cloud water CLWMR in those two years and CLMR before and after; the fetch asked for CLMR only, so each start
  file of those years had 397 records in place of 419 and each boundary file lacked its 22 cloud water levels (the
  2021-12-10 18Z start: 397 records, now 419). Both names are taken now. No forecast changes: a GFS start is a cold
  condensate start (cloud water 0, as NCEP's NPS does for the NAM), so no route reads these records today. A 12 km
  6 h run from that start gives the same bits with and without them (819 of 819 history fields, every hour).
- A box south of the equator gets its Lambert product grid where it is. The grid's inverse dropped the cone's sign, so
  a southern box's GRIB2 grid landed 180/n degrees of longitude away: `--products hiresf` wrote 0-byte files ("0
  records") while the run said the files were written, and no catalog map frame was drawn. A 3 km box at 9S 163W now
  writes 786 and 799 records at 0 h and +1 h. Northern grids are unchanged.
- `woof-nam nest --products` with no set (auto) on a box takes `hiresf`. It took `awphys`, whose grid the box does not
  cover, and wrote nothing (exit 6, "points outside the NMMB domain").
- ERA5, IFS and AIFS starts in arctic air: a pressure level under the ground takes the 2 m air carried down at
  6.5 K/km for the interpolation and the humidity. ECMWF carries such a level down from above a surface inversion, so
  these starts were up to 20 K warm on their lowest levels and their 0 h 2 m temperature 6.9 K warm on the coldest
  tenth of the land (ERA5 2015-01-26 12Z). NCEP sources are unchanged.
- Sea and lake ice start from their own column, and near a pole the soil fill follows the sphere. Every water cell
  with ice cover over half takes Noah's sea-ice column (skin temperature down to 271.16 K under 1.5 m of ice) in
  place of land soil filled out from the nearest coast, and north of 60 degrees one ring of the fill reaches as far
  east and west as it does north and south. A box holding the pole showed rays from it in the deep-soil and soil
  temperatures. Open water takes the NAM's own deep-soil temperature, 280 K, as the NAM's analysis does, in place of
  land soil filled out from the coasts (it is written as TG and TMN; Noah does not run on open water).
- A GFS start at forecast hour 3 no longer goes to NaN at step 2: the wind-rotation coefficients reach the card fresh
  on every call, and a boundary folder holding impossible values (finite but far outside physical bounds) is rebuilt,
  not reused; the builder refuses to write one.
- An ERA5 run shorter than 6 h fetches the boundary times it reads. It stopped after its fetch.
- An IFS run may be as long as ECMWF's open data reaches at every cycle, 144 h. The fetch read only to 90 h, the old
  length of the 06Z and 18Z runs, so a 100 h run stopped in its fetch.
- A start from a forecast hour of a GFS or GEFS run that has not happened, or has not yet posted the hours the
  window reads, is refused in one line before any fetch (`woof-nam fetch --cycle C --lead H`, `woof-nam nest --start
  T --lead H`, `woof-nam suite --cycle C --lead H`). It printed its plan and ended with bare "HTTP Error 404" lines.
- An ERA5 start before 2000 runs: the CO2 (1956-2003) and volcanic aerosol (1850-1999) radiation files are shipped
  (NOAA fix files, unmodified), and a year before 1956 reads the historical global means. It stopped at its first
  radiation call.
- The fit memory path (cards under 64 GiB) sizes its radiation batches at 0.34 MB a row-column; at 0.26 MB it ran out
  of card memory.
- The NAM static surface file is built per start month (`static/nam12_static.mMM.npz`). One file was built for the
  first start and reused for every later month, so a July start on a work root first used in November took
  November's greenness and albedo.
- A finished history hour is claimed once (its folder is renamed), never twice.
- A stalled download is retried after 20 s, the ECMWF open-data bucket's regional name included (it kept 300 s per
  ranged request). GEFS receipts credit NOAA open data.
- Maps: on a pip install the data credit is on the title line; with the renderer built from a checkout
  (`tools/renderer/install_renderer.sh --from-src`) a box that holds a pole or crosses the date line keeps its mesh
  (stripes over a quarter of the map and a 10-15 hPa seam before); a map frame of the 12 km grid takes about
  5 s, not 11-14 s, on a host with fragmented memory (NumPy huge pages off for the frame pool; the frame is byte for
  byte the same).
- A comparison sheet whose panels come out at two sizes is composed: they are padded to one size first. A WOOF-NAM
  panel whose 10 m wind passes 60 kt carries the longer wind ladder and is wider than the GFS panel beside it, and
  on a pip install the run ended rc 5 at that sheet after its maps were drawn (the 1.5 km nest command above).
- The live-parent door's memory estimate counts the GFS decode once for the process, not once per domain (a
  53,000-column ladder was estimated at 13.15 GiB and held 4.80).

Changed in a plain run (no new option)
- The deep-soil temperature (TG, TMN) over open water is 280 K, and so is the 3 m soil temperature record of the
  GRIB2 files there, on every isobaric start (every source but `native`, the NAM's own analysis, and `parent`). A
  3 km box over the Gulf of Mexico (30N 90W, GFS 2026-11-17 00Z, 1 h, `--products hiresf`) differs from 0.2.1 in 7,218 TG values, all on open water, and in that one GRIB2 record; the other 19.8 million
  history values and 786 of 787 (0 h) and 799 of 800 (+1 h) GRIB2 records are bit for bit the same.
- Sea and lake ice, and soil north of 60 degrees, start as above, so a run whose grid holds ice or land north of 60N
  (the NAM 12 km grid holds Hudson Bay and the Canadian Arctic) differs from 0.2.1 in its forecast there. A 12 km box
  of 1,200 km over the Canadian Arctic (70N 90W, GFS 2026-11-17 00Z, 1 h): 8,885,902 of 19,964,104 history values
  differ from 0.2.1. At +1 h over its 2,773 sea-ice cells the skin temperature is 0.64 K and the 2 m temperature
  0.29 K lower on average (at most 4.3 K and 2.2 K); over land and open water the mean change is under 0.01 K, and
  surface pressure moves at most 0.04 hPa anywhere.
- The history carries MSLET, written by the forecast on the card.
- ECMWF starts (ERA5, IFS, AIFS) where a pressure level is under the ground (arctic air) start colder near the ground
  (above). ERA5 2015-01-26 12Z, a 12 km box over Baffin Island: the 0 h 2 m temperature of the coldest tenth of the
  land is 2.7 K lower than in 0.2.1.
- Maps leave the boundary rows off and carry the data credit on the title line (above).

Not in this release
- A storm-following nest (`--follow`) from a forecast hour; `--start spinup` with a forecast-hour start; forecast-hour starts from IFS,
  AIFS, AIGFS, RAP, HRRR and RRFS (a window starts from their analysis before it), and from GEFS on a box; GRIB2 off the
  whole hour; boundary spacing that changes inside one run.
- Two-way feedback during moves, a ladder inside a moving nest, a moving nest on several cards; the fire-weather nest
  inside a live 3 km nest.
- Held for a later version: where the winds of a native (NAM analysis) start sit on the grid, a below-ground
  temperature ramp for ECMWF starts, and larger radiation batches on the fit memory path.

## 0.2.1

Fixed
- Snow cover and albedo over snow match the NAM: the Noah vegetation and general tables are now NCEP's NMMB tables
  (the NAM's snow-cover depth SNUP and shape SALP). The package had shipped WRF V3.0 tables, which gave about half the
  NAM's snow cover on grass and crop, so albedo over deep plains snow was about 32 % where the NAM had 52 %.
- The surface wind gust (GUST) is the 10 m wind plus the largest weighted excess of the boundary-layer wind over
  it, the weight 1 at the ground and 0.5 from 1 km up, the boundary-layer top where the bulk Richardson number
  from the lowest level reaches 0.25 (Vogelezang and Holtslag 1996). Against NCEP's awphys files of the 2024-11-05 12Z
  cycle, in a run started from the NAM's analysis, the mean absolute difference is 0.40 m s-1 at 0 h and
  0.42 m s-1 at +3 h (correlation 0.994).
- The column records (CAPE, CIN, the lifted indices, storm-relative helicity, the 0-6 km shear, the Haines index
  and the echo top) are made in one pass over each column. At +3 h of that run the mean absolute differences
  from NCEP's file are 8.7 J kg-1 (surface CAPE), 1.7 J kg-1 (surface CIN), 6.7 m2 s-2 (0-3 km helicity),
  0.07 K (lifted index) and 0.06 (Haines index).
- RH on isobaric levels is over liquid water at every level, as NCEP's files are. At +3 h of that run the mean
  difference from NCEP's file is -0.01 % at 250 hPa, -0.04 % at 500 hPa, -0.11 % at 700 hPa and -0.22 % at
  850 hPa.
- A native start's 0 h files carry the analysis's convective rain and convective cloud, and the 0 h reflectivity
  comes from one microphysics call on the 0 h state, as at the forecast hours. At 0 h of that run the mean
  absolute difference from NCEP's file is 1.1 dBZ for the composite reflectivity (correlation 0.986) and 177 m
  for the cloud base.
- Nest files: snow melt (SNOM) is in kg m-2. Over water and sea ice the snow-free and maximum snow albedo records
  carry NCEP's values (6 % on open water, 65 % and 60 % on sea ice), and over land and sea ice the water
  temperature record (WTMP) carries the surface temperature, as NCEP's files do. Checked against NCEP's native-grid
  files of the 2024-11-05 12Z cycle; no nest run was made.
- Nest hour-0 files write their zero-length statistical windows in hours (unit 1, length 0), as NCEP's do.
- Above the 0 C level the convective part of the history reflectivity falls off with height: -3.5 dB at the
  0 C level and 4 dB more per km above it (Smith 1984; Heymsfield et al. 2010), fitted to NCEP's 4 km and
  hybrid-level reflectivity over 10 cycles. Below the 0 C level it is unchanged.
- MSLET, and the 1000 hPa height that follows it, is the membrane sea-level reduction, computed from each history
  hour. It carried the model's in-run tracker field, which read 0.5-1 hPa low over the Rockies. The run's own
  MSLP maps draw the same field; the history MSLP is unchanged.
- `woof-nam run --resume` of a 12 km run no longer stops with "Too many open files" when it loads the restart set.
- A resumed run with `--products` writes GRIB2 for its new hours; it stopped at the hours of the earlier leg.
- With a long TMPDIR (over about 70 characters) the products, column and history writer pools start: their
  worker socket goes in a short private folder under /tmp, removed at exit, and TMPDIR stays the temporary folder
  for everything else. They failed at the first task with "OSError: AF_UNIX path too long" and the run lost its
  GRIB2 hours.
- `woof-nam run --resume` to a later end than the first leg (`--hours 60` after a 20 h run) builds the boundary
  files it lacks from the raw inputs in the case folder before its first step. When those inputs are missing it stops
  there in one line naming the `woof-nam fetch` that brings them. It stopped at +21 h with a missing boco.0021.
- `woof-nam doctor` no longer reports missing terrain after the README install: the terrain every run reads has its
  own line, and the land sets only nests and other grids read are listed as optional.
- `woof-nam compare` names the variables or hours only one run has when no shared value differs.
- `woof-nam products` on a full card waits and reruns an hour whose hand-off frame ran out of GPU memory, or whose
  writer refused for lack of room; those hours were lost beside a 12 km forecast on a 32 GB card.
- The column store sizes its workers by the memory limit of the session it runs in as well as the host's free
  memory; under a capped session it lost workers and wrote no store.
- A failed run no longer leaves host RAM full. The column store's views and the history writer's shared-memory
  blocks in /dev/shm are removed whatever ends the run (a killed worker, a broken pool, an error, SIGTERM), and the
  next run removes those of a run that was killed outright. A broken column pool writes its unfinished hours again
  with half the workers. Before, one failed 12 km run left 43 GB in /dev/shm.
- The column store and the history writer read the room actually free on /dev/shm before they write to it, counted
  at the uncompressed size of what they write. Short of room they use fewer workers, the disk, or the writer thread,
  and say so in one line. A fatal signal now prints its stack to the run log instead of ending the run silently.
- The snow-free albedo is the NAM's own where the run has an archived NAM analysis: it is taken from that analysis
  (`woof-nam static` writes it into the static file as ALBASE). It was a vegetation-class table value, 3 points high
  on average over snow-free land; where it was 10 or more points high (parts of the south-western deserts) the 2 m
  temperature of a sunny afternoon read about 1 F cool.
  A run whose static file was built by an earlier version keeps the table value until the file is rebuilt.
- The precipitation-type records (CRAIN, CSNOW, CFRZR, CICEP) are written where the rate is at least 0.01 mm/h, the
  rate that reproduces NCEP's files. At 0.05 mm/h they covered about half the NAM's snow area.
  `NAMGPU_PTYPE_RATE_MM_H` sets another rate.
- The 2 m dew point, relative humidity and specific humidity records no longer read low (dew point 0.2-0.6 F,
  specific humidity about 1 %): the 2 m humidity was converted from a mixing ratio twice.
- A run started from the NAM's own analysis (`--source native`) no longer starts with twice its turbulent kinetic
  energy. The difference was gone by +1 h.
- The awphys files carry the accumulated snowfall water record (WEASD, accumulation) in NOAA's 12 h window. It is
  rebuilt from the model's 3 h snowfall bucket. Over land at +12 h of the 2024-11-05 12Z case it matches the
  archived NAM as closely as precipitation does (correlation 0.997).
- A run started from the NAM's own analysis writes the analysis' cloud cover, boundary-layer height, surface and
  radiation fluxes, soil moisture and vertical velocity in its +0 h file. 94 of its records were all zero there;
  5 still are (the four precipitation types and the 0-0 h snowfall window). The forecast is not changed: at +1 h
  and +3 h the records over land compare with the NAM as before.
- The Noah soil table is NCEP's operational NMMB table, replacing WRF V3.0's. The effect is small: soil
  temperature and latent heat flux over land move a little towards the NAM.
- Snowfall water accumulated by the RUC land model and the column suites is in kg m-2, as Noah's is, and the nest
  files' snowfall accumulation is no longer multiplied by 1000. Shown by a unit test; no nest run was made.
- A run started from the NAM's own analysis takes the NAM's liquid soil water, snow-free albedo and maximum snow
  albedo from NCEP's native surface file of the same cycle (bgrdsf00, six records, fetched by `woof-nam fetch`).
  Before, frozen ground started with too little ice and warmed too fast. In the 2024-11-05 12Z case at +9 h, soil
  temperature over snow-free ground that the NAM had frozen was 2 F too warm on average and is now within 0.1 F;
  the 2 m temperature over land differs from the archived NAM by 0.22 F on average, down from 0.33 F; albedo where
  the NAM has snow is within 0.1 point on the model grid, down from 8. A case without the file runs as before and
  says so in one line.
- `woof-nam static` takes the maximum snow albedo and the snow-free albedo from a bgrdsf00 file given with `--land`
  or lying beside the analysis. A static file built by an earlier version still works and says what it lacks.
- `woof-nam run --resume`: boundary files rebuilt for the resumed hours are the same bytes as the first build, and
  only the missing ones are built. Before, a rebuild could copy boundary rows before they were filled, which gave
  wrong boundary tendencies and once a stop at the first step.
- `woof-nam run --products` on a 32 GB card no longer stops with an out-of-memory error when the GRIB2 writer holds
  card memory at the moment the forecast needs it: the writer steps aside and the forecast waits a few seconds. The
  receipt shows `pool.card_claims` and `pool.card_wait_s`. Shown on a 24 h 12 km run (one wait of 5.6 s, 25 of 25
  GRIB2 files) and on a 3 h run beside a deliberately greedy stand-in writer (6 waits; its history and GRIB2 files
  are the same bytes as those of the hours of the first run, which had no wait in them).
- The column store, the history writers and the GRIB2 field pool size themselves to the cores the process may use
  (affinity and cgroup), not the host's count. Shown by unit tests.
- The cleanup of ended runs in /dev/shm decides who is alive by an owner lock, so a live run in another PID
  namespace that shares /dev/shm is not taken for a leftover. Shown by unit tests.
- `woof-nam products` with no CUDA device stops before the first hour in one line, with any `--post-device`. The
  writer computes the derived records of NOAA's files on a card only; `--post-device cpu` moves only the rest of
  the post to the processor, and its files are byte-identical to the card's. It used to fail at hour 0 after
  building the frame on the processor.
- A `woof-nam products` run stopped by SIGTERM or SIGHUP leaves no worker, no writer and no /dev/shm frame folder;
  after SIGKILL its field workers end within seconds.
- A run with `--products` that stops before its receipt stops its GRIB2 follower instead of leaving it waiting.
- Surface visibility (VIS) counts the convective scheme's rain at the output time: its rain water, from the
  Marshall-Palmer drop sizes, is added to the rain term of the Stoelinga-Warner extinction. Under convective rain the
  record read near clear air. In the 2024-11-05 12Z case started from the NAM's own analysis, over cells with
  convective rain in both models, the mean visibility at +12 h was 22.8 km against the archived NAM's 13.3 km and is
  now 14.2 km; at +24 h 22.8 against 13.4, now 14.6. Every other record is the same bytes.
- `woof-nam run --products` gives its card back before it waits for the GRIB2 follower, on every exit with one
  card (also with `--no-maps` and after a failed forecast): the wait runs in a fresh process with no CUDA context,
  and the receipt records `card_released`. On a 12 km run on a 32 GB card the finished forecast held 15,900 MiB of
  card memory through its wait; now it holds none.
- The run's wait no longer stops a GRIB2 follower that is alive and waiting for card memory: the follower writes a
  heartbeat (`<out>/products.state.json`) while it waits. Shown by unit tests.
- `woof-nam static` prints where the albedos came from; it said no bgrdsf00 file was given when it had read one.
  The run's last products line gives the time the run waited for its GRIB2 files.
- The run's maps no longer take the card again after `woof-nam run --products` has given it back. The MSLP map's
  MSLET is computed in a helper process that ends with the maps when the calling process holds no CUDA context (the
  run's maps step, `woof-nam maps`). After a 36 h 12 km run the run's own process was back on the card 3 s after
  it released it, with 1,480 MiB and then 500 MiB until it ended 11 minutes later. On a 3 h run of the fix it is
  not on the card again, and the maps are the same bytes.
- A run started from the GFS prints "albedos from the case's bgrdsf00.land.grib2" when it took both albedos from
  that file. It printed that the case had no NCEP native surface file.

Changed
- The GRIB2 writer `rw_rrfs_upp` and the helpers are fetched from the v0.2.1 release.
- The column store runs one worker per frame, bounded by cores, host memory and scratch space, instead of four.
- `woof-nam run --help` lists every physics suite, the two presets included.
- `woof-nam run --products` returns only when `<out>/products/DONE` is written. A failed follower makes the run
  exit non-zero with one line naming the hour; a follower that writes nothing for `--products-stall-s` (900 s) is
  stopped. `--no-products-wait` returns at once and names the file that marks the products done.
  `woof-nam nest --products` waits the same way (exit 6 when the follower failed; shown by a unit test, no nest run
  was made). With `--follow` or a live parent the nest door writes no GRIB2 and now says so instead of ignoring
  the option.
- The suite progress line says "parent: no hour complete yet" before the parent's first hour.
- A GRIB2 follower refused card memory while its forecast runs reruns the stage as soon as the forecast gives
  memory back, and that refusal does not use one of the hour's 20 tries. On a busy processor the follower's own
  work still sets how far the files are behind their hour.
- The GRIB2 follower starts after the run's case and memory checks, so a run that stops in them starts none.
- `products.log` and `products/index.json` give, per hour, when the file was taken, framed, written and in place.
- The README gives the GRIB2 timing of the 4-card CONUS run: each file 65-80 s after its history hour, and products
  costing the CONUS ranks about 14 % from hour 3 on.

## 0.2.0 (2026-11-10)

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
- `woof-nam suite` on 4 cards runs the CONUS nest on all four (63-68 s per forecast hour measured on 4 cards of 96 GB;
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

- README: memory and speed measured on an 8 GB card (fresh `pip install recast-woof-nam==0.1.2`): 3 km
  nests of 99 x 100, 328 x 330 and 360 x 359 points peak at 2.0, 4.1 and 4.2 GB ("Speed and memory").

- Composite and simulated reflectivity now match NOAA's NAM files. The history `COMPOSITE` and
  `REFL_10CM`, the maps, and the awphys GRIB2 `REFC` (record 3) and `REFD` (records 2, 295, 296) add the reflectivity
  of the convection scheme's rain (Z = 300 R^1.4, full strength below the freezing level, 20 dBZ weaker at the
  convective cloud top) to the microphysics reflectivity. Before, they carried the microphysics
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
  manylinux_2_28) from the woof-nam release and checks its size and SHA-256 before installing it. New command
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
  loop needs is 10,016 bytes (2.44 GiB on a 32 GB card, was 11,264), and `woof-nam run --init-from` and `--resume`
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
