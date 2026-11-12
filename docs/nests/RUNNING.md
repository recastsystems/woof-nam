# Running NOAA's NAM nests and the whole NAM suite

NOAA's NAM ran a 12 km parent over North America and five nests beside it in the same job, four times a day:

| nest | `--ncep-nest` | native grid (points x levels) | spacing | step | length | files |
|---|---|---|---|---|---|---|
| CONUS | `conusnest` | 1827 x 1467 x 60 | 3 km | 6.25 s | 60 h | `woof-nam.tHHz.conusnest.hiresfFF.tm00.grib2`, FF 00-60 |
| Alaska | `alaskanest` | 1189 x 1249 x 60 | 3 km | 6.25 s | 60 h | `woof-nam.tHHz.alaskanest.hiresfFF.tm00.grib2`, FF 00-60 |
| Hawaii | `hawaiinest` | 373 x 561 x 60 | 3 km | 6.25 s | 60 h | `woof-nam.tHHz.hawaiinest.hiresfFF.tm00.grib2`, FF 00-60 |
| Puerto Rico | `priconest` | 401 x 325 x 60 | 3 km | 6.25 s | 60 h | `woof-nam.tHHz.priconest.hiresfFF.tm00.grib2`, FF 00-60 |
| Fire weather | `firewxnest` | 333 x 333 x 60 | 1.5 km | 2 1/12 s | 36 h | `woof-nam.tHHz.firewxnest.hiresfFF.tm00.grib2`, FF 00-36 |

Each nest is a row of `namgpu/nest/ncep_nests.py`, taken from NAM v4.2.11's own `parm/nam_configfile_<nest>` and
`parm/nam_namelist.nps_<nest>`: NCEP's grid on the 12 km parent's rotated pole, NCEP's start point inside the parent,
step, radiation interval, boundary blending rows, digital filter and length, and no parameterised convection. The whole
suite is the table `namgpu/parm/nam_suite.json`; a different card layout or another nest is a row there, not code.

Each nest writes NOAA's nest GRIB2 files, on NOAA's output grid for that nest, with NOAA's record list and time
windows ([nests/COMPATIBILITY.md](COMPATIBILITY.md) compares them with NOAA's files record by record).

## One nest

```
woof-nam nest --ncep-nest hawaiinest --source gfs --start 2026110306 --out runs/hawaii --products
```

- **Length.** The row's own: 60 h for the four 3 km nests, 36 h for fire weather. `--hours N` runs fewer.
- **Start and boundaries.** `--source gfs` starts the nest straight from GFS (start and boundaries). In the suite
  the nests start from WOOF-NAM's own 12 km run: `--source parent --parent-run <12 km run folder>` (one-way, the 12 km
  run must have written its hand-off set: `woof-nam run --domain conus12-suite` does).
- **Products.** `--products` writes NOAA's nest files hour by hour while the nest runs, into `<out>/run/products/`:
  the GRIB2 file, then its `.idx`. `woof-nam products --run <out>/run` writes them afterwards; the nest the run is
  picks its own set (`--set conusnest|alaskanest|hawaiinest|priconest|firewxnest` to name one).
- **Several cards.** `--cards N` (or `--card-list 1,2,3,4`) splits the nest into rows over N cards of one machine.
  The CONUS and Alaska nests need several cards on most machines; the door's pre-flight checks the memory on each card
  before anything runs, and refuses in one line, naming what is needed, if the nest does not fit. `--wait-min M`
  waits up to M minutes for memory on a shared card.
- **A smoke on a small card.** `--ncep-window NXxNY[+I+J]` runs a window of NX x NY parent (12 km) cells of the nest:
  the same pole, spacing, step and physics, at a size that fits.

The four 3 km nests:

```
woof-nam nest --ncep-nest conusnest  --source parent --parent-run runs/12km --start 2026110306 --cards 4 --out runs/conusnest  --products
woof-nam nest --ncep-nest alaskanest --source parent --parent-run runs/12km --start 2026110306 --cards 2 --out runs/alaskanest --products
woof-nam nest --ncep-nest hawaiinest --source parent --parent-run runs/12km --start 2026110306 --out runs/hawaiinest --products
woof-nam nest --ncep-nest priconest  --source parent --parent-run runs/12km --start 2026110306 --out runs/priconest  --products
```

### The fire-weather nest

NCEP moved its fire-weather nest between cycles, never during one. Its point for each cycle came from a file,
`nam_firewx_loc`, with one line per cycle, `HHz lat lon`. WOOF-NAM keeps that mechanism:

```
woof-nam nest --ncep-nest firewxnest --source parent --parent-run runs/12km --start 2026110306 --out runs/firewx --products
woof-nam nest --ncep-nest firewxnest ... --firewx-center 44.1,-121.3      # this cycle's point, in place of the file
woof-nam nest --ncep-nest firewxnest ... --firewx-loc my_firewx_loc       # another point file
```

- **The default points** (`namgpu/parm/nam_firewx_loc`) are NCEP's last production points, unchanged from 2026-10-20
  to 2026-11-03: 39.0 N 112.0 W (west-central Utah) at 00Z and 12Z, 42.0 N 122.3 W (southern Oregon) at 06Z and 18Z.
- **NCEP's rules:** a point at 52 N or north places the nest in the Alaska nest, otherwise in the CONUS nest; the
  333 x 333 nest must lie at least 5 rows inside that nest, or the door refuses the point by name. With no point for
  the cycle, NCEP's default was Washington DC (38.9 N, 77.0 W), and so is ours. `woof-nam suite` does not stop on a
  point that fails the check: it uses the Washington DC default for that cycle and its plan says why.
- **Its output grid** is the box around the nest on NCEP's 1.27 km Lambert master grid (CONUS) or 1.488 km polar
  stereographic master grid (Alaska), as NCEP computed it. For NCEP's two production points that is exactly NOAA's
  grid: 543 x 379 points at 00/12Z and 646 x 389 at 06/18Z.
- **Its centre** is written to `woof-nam.tHHz.firewxnest_center_latlon` (`42.0 -122.3`, NOAA's format) and to the
  cycle's `cycle.json`.
- **How it is driven:** NCEP ran the fire-weather nest as a 2:1 child of its 3 km CONUS (or Alaska) nest. Here it takes
  its start and boundaries from the 12 km run (a 1:8 step). Its point, size, spacing, step and length are NCEP's.

## The whole suite, one cycle

```
woof-nam suite --cycle 2026110306 --out runs/suite-2026110306             # 12 km parent and every nest, every card
woof-nam suite --cycle 2026110306 --out runs/s --dry-run                   # the plan only
woof-nam suite --cycle 2026110306 --out runs/s --domains 12km,hawaiinest   # some of it
woof-nam suite --cycle 2026110306 --out runs/s --parent-run runs/12km --parent-ended runs/12km/.ended   # nests only
```

1. **Parent.** `woof-nam fetch` (GFS) and `woof-nam run --domain conus12-suite`: the NAM 12 km grid, 84 h, with the
   hand-off set the nests read in its hourly history.
2. **Nests.** Each nest starts once the parent's history covers its hours, one-way from it, on the cards its layout row
   gives it. NCEP ran its nests one-way too.
3. **Products.** While each domain runs, `woof-nam products --follow` writes its GRIB2 files into
   `<out>/publish/<domain>/`: NOAA's awphys files on grid 218 for the parent, NOAA's hiresf files for each nest. A
   GRIB2 file is written first and its `.idx` renamed in after it, so a file whose `.idx` exists is complete.
4. **Finish.** `<out>/publish/<domain>/cycle.json` for each domain: its state (`complete` or `failed`), grid, packing,
   and every file with its size and SHA-256. Each nest's history is deleted once all its GRIB2 hours are there
   (`--keep-history` keeps it). `<out>/suite-receipt.json` has every stage's wall time, step time and peak memory.

A domain that fails never stops the others: its `cycle.json` and the receipt say `failed` and why.

**Card layouts** are rows of `nam_suite.json` (`layouts`), chosen by the number of cards the machine has:

| cards | 12 km parent | CONUS | Alaska | Hawaii | Puerto Rico | fire weather |
|---|---|---|---|---|---|---|
| 8 | 0 | 1-4 | 5-6 | 7 | 7 | 7 |
| 4 | 0 | 0-3 | 3 | 1 | 2 | 2 |
| 2 | 0 | 1 | 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |

On 4 cards the CONUS nest takes all four: 63-68 s per forecast hour on 4 cards of 96 GB (2026-01-23 12Z, 12 h), with
18.3-18.6 GiB per card in the step loop. The other nests stay off card 0, which holds the parent and the CONUS GRIB2
writer; every CONUS rank waits on the slowest one.
With GRIB2 products following (6 h, 2026-01-23 12Z), each CONUS file was public 65-80 s after its history
hour, and products cost the CONUS ranks about 14 % (9.4-9.7 s per forecast hour) from hour 3 on.

Domains that share a card each check its free memory before they start and wait for room. The parent is the critical
path: every nest waits for its hours.

## Output layout

```
<out>/publish/12km/        woof-nam.tHHz.awphysFF.tm00.grib2 (+ .idx)              FF 00-84
<out>/publish/conusnest/   woof-nam.tHHz.conusnest.hiresfFF.tm00.grib2 (+ .idx)    FF 00-60
<out>/publish/alaskanest/  woof-nam.tHHz.alaskanest.hiresfFF.tm00.grib2 (+ .idx)   FF 00-60
<out>/publish/hawaiinest/  woof-nam.tHHz.hawaiinest.hiresfFF.tm00.grib2 (+ .idx)   FF 00-60
<out>/publish/priconest/   woof-nam.tHHz.priconest.hiresfFF.tm00.grib2 (+ .idx)    FF 00-60
<out>/publish/firewxnest/  woof-nam.tHHz.firewxnest.hiresfFF.tm00.grib2 (+ .idx)   FF 00-36
                           woof-nam.tHHz.firewxnest_center_latlon
every domain:              cycle.json (written last)
```

The names are NOAA's (`nam.tHHz.<nest>.hiresfFF.tm00.grib2`) with `woof-nam` in place of `nam` (`--prefix` changes
it), so they never get mixed up with NOAA's own files.

## What is different from NCEP's NAM

- **Start.** NCEP started its CONUS and Alaska nests from their own cycled analyses and cold-started Hawaii, Puerto
  Rico and fire weather each cycle. Here every domain starts from the same source as the 12 km run (GFS by default).
- **Coupling.** NCEP ran all six domains in one executable, the nests reading the parent at every parent step. Here
  each nest reads the 12 km run's hourly history (boundaries every 3 h). `woof-nam nest --ncep-nest conusnest
  --concurrent` runs the parent and the CONUS nest together in one process, the nest reading the live parent every
  step, as NCEP did.
- **Fire weather** is driven by the 12 km run, not by the 3 km nest (above).
- **Products.** The records not written yet, the record numbers and the packing are listed in
  [COMPATIBILITY.md](COMPATIBILITY.md).
