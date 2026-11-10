# Winter and cold-air-damming verification

`woof-nam score --winter` selects the winter/CAD scorer in namscore. It reads
existing output and never changes the forecast suite or its settings. `nam`
remains the default. Forecast ports remain selectable options, each compared
with the frozen control. An archived operational NAM run is a reference, not
proof of identity with that frozen control.

Build the Rust reader once with a Rust toolchain and the NetCDF C development
library (for example `libnetcdf-dev`). The wheel includes this source:

```sh
rustc --edition=2021 -O namgpu/io/winter_bridge.rs -o /path/to/rw_namwinter
export NAMGPU_WINTER_BRIDGE=/path/to/rw_namwinter
# On a shared node use its build-slot wrapper around rustc.
woof-nam score --winter --cycle 2026012312 --hours 12 \
  --run control=runs/control --run option=runs/option \
  --asos cases/2026012312/obs/asos.csv --out score/winter
```

The reader also resolves as `rw_namwinter` on PATH. There is no automatic build
or Python data-reader fallback. Rust reads CSV, locates stations on the sphere,
reads surface slabs and individual vertical columns, and emits small station
receipts. `--workers` controls independent readers (default 2, maximum 4).
This is observation verification on the host, not a CPU forecast fallback.
NumPy scores the small receipts; Pillow draws a statistical time-series sheet.
Geographic weather maps continue to use the package's `rw_wrfbatch` renderer.

Each history directory must contain hourly `wrfout_d01_YYYY-mm-dd_HH:MM:SS`
files, including hour zero. The reader checks the embedded `Times` and preceding
hour before differencing accumulations. Changed grids and negative accumulation
resets are errors. It does not silently make a multi-hour total into an hourly
amount. A missing field stays missing. Fill values and packed NetCDF values are
handled before sampling. The nearest grid point must be within `--max-km` (15 km
by default).

Use the existing `rw_asos` acquisition/cache workflow. This scorer does not add
another HTTP downloader or retry policy. Input is a UTF-8 IEM-style CSV with
`station,valid,lat,lon,tmpf,dwpf`, and optional `wxcodes,p01i,ice_accretion_1hr`.
Times must be UTC. Only the nearest report within 10 minutes of the valid hour
is used, with an earlier-time tie break. Present-weather scoring requires
`wxcodes`; a cached rw_asos export without that column supplies temperature and
dewpoint only. It does not establish a dry weather report.

Present weather uses classes 0 none, 1 rain, 2 snow, 3 freezing rain, 4 sleet.
Mixed reports use freezing rain, sleet, snow, rain priority. Nonfreezing drizzle
is in rain; freezing drizzle, unknown precipitation, hail and missing codes are
unscored. Vicinity and blowing/drifting snow alone are not falling precipitation
at the station. An empty *present* wxcodes column means no reported precipitation;
a missing column or `M` is unknown. The matrix has forecast rows and observation
columns. Overall Heidke includes dry pairs and penalizes missed precipitation.
The separate `both_wet` Heidke is conditional on both sides being wet. Per-class
hit rate is hits/(hits+misses); counts, false-alarm ratio, CSI and ETS accompany
it. Undefined scores are JSON null, never zero or perfect skill.

Model methods are explicit selections, not physics settings:

- `explicit-sr` (default) reuses the recovered draft's diagnostic: wet at hourly
  RAINNC+RAINC increase of at least 0.25 mm or PREC at least 4e-6 m; explicit
  SR at least 0.5 yields snow or sleet according to the profile diagnostic;
  otherwise liquid is freezing rain below 273.15 K or rain above it.
- `bourgouin` reuses the draft's pressure-energy diagnostic on T/PRES columns.
  T is perturbation potential temperature, PRES and PSFC are Pa, T2 is K,
  and Q2 is kg/kg. At least three valid levels including the surface, reaching
  500 hPa, are required. This is the inherited simplified diagnostic, not a
  claim to implement every revision of that method.
- `stored` uses a PTYPE field with the above class numbering, or all four
  CRAIN/CSNOW/CFRZR/CICEP indicator fields. Do not use a PTYPE field with a
  different code convention without converting it first.

Type scores omit hour zero, where hourly precipitation is unavailable. ASOS is
an instantaneous report, so agreement with hourly endpoint diagnostics is a
verification proxy. Snowfall is hourly liquid precipitation times SR times an
explicit 10:1 depth ratio. Freezing-rain liquid is hourly precipitation where
the endpoint diagnostic says freezing rain. Neither is measured snow depth or
ice accretion. ASOS p01i conditional on a freezing-rain report is compared only
as a liquid-equivalent reporting-hour proxy. Ice accretion is retained in the
receipt but is not compared with liquid precipitation as though units alone
made the quantities equivalent.

The default CAD verification subset is a Southeast Piedmont polygon east of
the Appalachians, recorded in JSON. `--stations GSO,INT,HKY,...` selects a known
case's station set. Hourly T2 and Td2 bias, MAE and RMSE are in F. Arms use common
station-hours and common finite data for each variable. Observation values must
agree between arms. Sample counts are reported for every hour.

Erosion is a **thermal cold-pool proxy**, not a synoptic CAD diagnosis. A fixed
panel of stations with T2 in every hour and every arm prevents station outages
from changing the cold fraction. Default cold means T2 below 40 F. Establishment
requires at least half of the panel cold; erosion is the first of two consecutive
hours below that fraction after establishment. Set `--cold-f`, `--cold-fraction`
and `--sustain-hours` to a declared case criterion. A transient dip does not end
the event. Missing hourly coverage, no established wedge, and a wedge surviving
the forecast window have distinct statuses. Timing error is model minus observed
lead, reported only when both events resolve.

Retained station receipts can be scored without old grids or the reader binary:

```sh
woof-nam score --winter --cycle 2026012312 --hours 12 \
  --pairs control=control-pairs.csv --pairs option=option-pairs.csv \
  --out score/replay
```

Required CSV columns are `cycle,station,lead,latitude,longitude,obs_t2_f,model_t2_f`.
Optional columns are `obs_td2_f,model_td2_f,obs_type,model_type`, and
`model_precip_1h_mm,obs_precip_1h_mm,model_fzra_proxy_1h_mm,model_snow_proxy_1h_mm`.
Use blanks for missing values. Duplicate station-hours and wrong cycles fail.
Record conversion provenance alongside the receipt; do not label an old option
combination or an operational NAM archive as the frozen package control.

`--reports reports.csv` adds station-matched reports or analysis samples:

```csv
cycle,kind,station,start_lead,end_lead,amount,units,source
2026012312,snowfall,GSO,0,6,25.4,mm,NOHRSC sampled at GSO
2026012312,freezing_rain_liquid,GSO,0,6,2.54,mm,liquid-equivalent report identifier
```

Windows are `(start_lead,end_lead]`; every hourly model amount must exist.
`snowfall` means snow depth amount and `freezing_rain_liquid` means liquid
equivalent, not radial ice thickness. Units must be mm and the source must be
named. Negative or missing amounts are excluded, never replaced by zero.
Absent reports produce an explicit unavailable result. Arms' report counts
remain visible if model coverage differs.

Output includes `winter_scores.json`, `WINTER.md`, `cad-timeseries.png`, timing,
source hashes and sampled pair receipts. Inspect the sheet before interpreting
the table. `--no-sheets` omits the optional statistical chart. Save the small
receipts and delete disposable forecast data under the workspace's retention
rules. These products alone do not prove an option-off forecast identity run.

CI runs Rust tests and Python score/NetCDF integration tests:

```sh
rustc --edition=2021 --test namgpu/io/winter_bridge.rs -o /tmp/winter-tests
/tmp/winter-tests
NAMGPU_WINTER_BRIDGE=/path/to/rw_namwinter PYTHONPATH=. \
  python -m pytest -q -p no:cacheprovider tests/test_winter_score.py
```
