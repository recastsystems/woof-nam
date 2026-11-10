# Archived NAM record gate

Run after `woof-nam products`, on the same forecast lead as the archive:

```sh
woof-nam inventory nam.t06z.awphys01.tm00.grib2 ours/nam.t12z.awphys01.tm00.grib2 \
  --set awphys --baseline before/nam.t12z.awphys01.tm00.grib2 \
  --json inventory.json --md inventory.md
```

The archive defines the expected records. The command reads every GRIB2 field,
including both fields of a paired wind message, without unpacking the data.
Parameter, both fixed surfaces, statistical process, forecast start, duration,
and duplicate count must match. Reference dates and packing may differ.

Each archived field is reported as `written`, `left_out` with the table's release
note, or `unexplained`. A mapped field missing from the actual output is always
unexplained. A new archive field or level cannot disappear because the package
has no source mapping. Extra output records also fail this comparison.

`--baseline` additionally requires every old field, in order, to retain its
complete original message SHA-256, including its grid, dates, bitmap, packing,
and packed values. Added records may occur between the old messages. Comparing
reference and forecast values is a separate scientific check; this command does
not claim forecast skill or numerical agreement with NOAA.

Exit codes: 0 means all records are written or explicitly named as left out,
and the optional baseline is preserved; 1 means a coverage or identity failure;
2 means unreadable or unsupported input. The reader accepts product templates
4.0 and 4.8 with one statistical time range, the templates used by these NAM
products, and refuses unsupported templates instead of skipping them.

The regression fixture `tests/fixtures/nam-20261103-06-f01-inventory.json` was
read from NOAA's 2026-11-03 06Z f01 awphys and five nest GRIB2 files. It stores all
4,812 field headers, source URLs, byte sizes, and SHA-256 of each NOAA text index.
The tests cover missing unmapped records, missing mapped records, unknown
levels, layers, duplicate fields, time windows, paired messages, changed packed
bytes, truncated inputs, and complete table coverage of all six real archives.

`products --compare` is the older mapped-row comparison. It does not replace
this archive-driven release gate.

The nest tables now map cloud-top temperature and surface-parcel LCL pressure.
They require a Rust writer advertising `nam_inventory_1`; older builds are
refused before history is read. Temperature uses the existing cloud-top height
and column temperature interpolation. Pressure uses native PSHLTR, T2 and NAM Q2 specific humidity, converted to
mixing ratio on the GPU for this record only, with the writer's existing LCL
thermodynamics. Generic WRF frames use their Q2 mixing ratio and shelter estimate. These additive diagnostics
use a separate GPU image, leaving the legacy image and output planes intact.
The frozen forecast suite is unchanged. A release must fold the matching Rust
writer patch and qualify/pin its bridge artifact before shipping these tables.
