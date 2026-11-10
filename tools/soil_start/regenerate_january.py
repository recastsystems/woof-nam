"""Regenerate the pinned January CDF when the release asset is unavailable.

Run `fetch` without a GPU, then run `build` through the site's GPU scheduler.
The output is accepted only if it matches the release's existing SHA-256 pin.
No pin, forecast default, or soil-start capability is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import soil_fetch

ROOT = Path(__file__).resolve().parents[2]
CASE = Path("cases/2026-01-23_12z")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_dates(clim):
    actual = sorted(p.stem for p in (clim / "nam").glob("*.grib2")
                    if (clim / "gfs" / p.name).is_file())
    # NAM bgrd3d00 begins in September 2021. The first January window
    # therefore has no pairs; the other four windows each have 21 dates.
    expected = [d.strftime("%Y%m%d") + "12" for d in soil_fetch.month_dates(1)
                if d.year >= 2021 and d.isoformat() >= "2021-09-16"]
    if actual != expected:
        raise RuntimeError(f"January recipe requires all 84 archived pairs: "
                           f"missing={sorted(set(expected) - set(actual))}; "
                           f"extra={sorted(set(actual) - set(expected))}")
    return actual


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fetch", "build"))
    ap.add_argument("--work", type=Path, required=True,
                    help="one shared prepared directory for this recipe")
    ap.add_argument("--nam-analysis", type=Path,
                    help="2026-01-23 12Z NAM bgrd3d00 GRIB2, required for build")
    a = ap.parse_args(argv)
    work = a.work.resolve()
    clim = work / "clim"
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "fetch":
        soil_fetch.clim(str(clim), 1)
        print(f"January climatology ready: {len(paired_dates(clim))} paired dates")
        return 0
    if a.nam_analysis is None or not a.nam_analysis.is_file():
        ap.error("build requires --nam-analysis pointing to the 2026-01-23 12Z NAM analysis")
    dates = paired_dates(clim)
    analysis = a.nam_analysis.resolve()
    # meta_static is part of the NPZ bytes. Keep the producer's original
    # relative case path, while reusing the supplied analysis without a copy.
    link = work / CASE / "raw/nam/bgrd3d00.grib2"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != analysis:
            raise RuntimeError(f"refusing to replace existing input {link}")
    else:
        link.symlink_to(analysis)
    dest = work / "soil_cdf"
    dest.mkdir(exist_ok=True)
    output = dest / "cdf-01.npz"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}; use the receipt from its build")
    env = dict(os.environ, NAMGPU_ROOT=str(work), NAMGPU_PHYSICS_SUITE="nam",
               NAMGPU_CONFIGFILE=str(ROOT / "namgpu/parm/nam_configfile"), NAMGPU_CONFIG_SET="",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    command = [sys.executable, "-u", "-m", "namgpu.init.soil_start", "build-cdf",
               "--case-dir", str(CASE), "--clim", str(clim), "--out", str(output), "--smooth", "2"]
    start = time.monotonic()
    subprocess.run(command, cwd=work, env=env, check=True)
    pin = json.loads((ROOT / "namgpu/parm/soil_cdf_pins.json").read_text())["files"][output.name]
    actual = {"bytes": output.stat().st_size, "sha256": sha256(output)}
    receipt = {"schema": "woof-nam.regenerated-january/v1", "command": command,
               "cwd": str(work), "dates": dates, "analysis": str(analysis),
               "analysis_sha256": sha256(analysis), "output": str(output),
               **actual, "pin": pin, "pin_matches": actual == pin,
               "build_s": round(time.monotonic() - start, 3)}
    receipt_path = work / "regenerated-january.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if actual != pin:
        raise RuntimeError(f"regenerated CDF does not match the existing release pin; see {receipt_path}")
    print(f"Ready: NAMGPU_SOIL_CDF_HOME={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
