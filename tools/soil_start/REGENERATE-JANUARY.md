# Regenerate the January soil CDF

When the v0.2.0 release asset returns HTTP 404, the package's existing
producer can rebuild the pinned table from NOAA analyses. This command
keeps the original relative static-input path, which is part of the NPZ
metadata and therefore its SHA-256. It accepts the output only when its
size and full SHA-256 match `namgpu/parm/soil_cdf_pins.json`.

Use one shared prepared directory on the compute host. First fetch the
climatology without a GPU:

```bash
python tools/soil_start/regenerate_january.py fetch --work /workspace/shared-prepared/nam-soil-cdf-january
```

The recipe requests 105 dates at 12Z, every three days within 30 days of
January 15 in 2021 through 2025. The first window predates the NAM
bgrd3d00 archive. Exactly 84 paired dates remain, from 2021121612 through
2025021412. Missing pairs in those four archived windows are refused.

Reuse an existing 2026-01-23 12Z NAM analysis. Its public source is:
`https://noaa-nam-pds.s3.amazonaws.com/nam.20260123/nam.t12z.bgrd3d00.tm00.grib2`.
Then run the build stage through the compute host's GPU scheduler:

```bash
python tools/soil_start/regenerate_january.py build \
  --work /workspace/shared-prepared/nam-soil-cdf-january \
  --nam-analysis /path/to/bgrd3d00.grib2
```

The normal package GPU runtime and GRIB bridges must be available.
The builder uses the full NAM 954 x 835 parent grid, 21 quantiles per
soil layer, and the original land-only 5 x 5 smoothing. It writes
`soil_cdf/cdf-01.npz` and `regenerated-january.json` under `--work`.
Set `NAMGPU_SOIL_CDF_HOME` to that `soil_cdf` folder for the options nest.
The existing fetch command will recognize the pinned output:

```bash
python -m namgpu.init.soil_cdf_fetch --months 1 \
  --dest /workspace/shared-prepared/nam-soil-cdf-january/soil_cdf
```

A mismatched build writes its actual hash and a failed pin check in the
receipt, then exits nonzero. It never substitutes an unverified table or
changes the release pins. Existing output is never overwritten.

After the proving forecast and receipt are complete, remove only the
inputs and outputs created for this recipe. Retain the JSON receipt,
fetch/build logs, and the exact source revision used.
