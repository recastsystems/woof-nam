"""woof-nam fetch-soil-cdf: the monthly soil-moisture CDF tables the CDF soil start reads (NAMGPU_SOIL_START=cdf), downloaded from the woof-nam release and checked against the size and SHA-256 pinned in
parm/soil_cdf_pins.json.

    woof-nam fetch-soil-cdf                 # every month (12 files, about 268 MB each)
    woof-nam fetch-soil-cdf --months 1,2    # only January and February
    woof-nam fetch-soil-cdf --from DIR      # install tables you already have (checked the same way)
    woof-nam fetch-soil-cdf --where         # print the folder the soil start reads

Each table maps a GFS soil-moisture value at each NAM 12 km land point onto the NAM's own soil climate by quantile
(the GFS and NAM soil analyses at 12Z every 3 days, +/-30 days around the 15th of the month, 2021-2025; tools/soil_start).
They are built from NOAA's public-domain GFS and NAM archives. A table on another rotated grid is sampled to the run
grid on the GPU by nearest point.

Folder: $NAMGPU_SOIL_CDF_HOME, else ~/.cache/woof-nam/soil_cdf. NAMGPU_SOIL_CDF=auto reads $NAMGPU_ROOT/soil_cdf when
that folder has the month's table, else this folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

PINS_FILE = Path(__file__).resolve().parents[1] / "parm" / "soil_cdf_pins.json"
FETCH_CMD = "woof-nam fetch-soil-cdf"


def pins() -> dict:
    return json.loads(PINS_FILE.read_text())


def home() -> Path:
    v = os.environ.get("NAMGPU_SOIL_CDF_HOME")
    return Path(v) if v else Path.home() / ".cache" / "woof-nam" / "soil_cdf"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _ok(p: Path, pin: dict) -> bool:
    return p.is_file() and p.stat().st_size == pin["bytes"] and _sha256(p) == pin["sha256"]


def _url(name: str) -> str:
    pp = pins()
    base = os.environ.get("WOOF_NAM_SOIL_CDF_URL") or f"https://github.com/{pp['repo']}/releases/download/{pp['release']}"
    return f"{base.rstrip('/')}/{name}"


def _download(url: str, dst: Path, tries: int = 3) -> None:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            return
        except OSError as e:   # urllib.error.URLError is an OSError
            last = e
    raise OSError(f"{url}: {last}")


def fetch(months, source: Path | None = None, dest: Path | None = None, quiet: bool = False) -> list:
    """Install the tables of `months` into dest (default home()); return the installed paths. Raises on a bad file."""
    pp = pins()["files"]
    dest = dest or home()
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    from concurrent.futures import ThreadPoolExecutor

    def one(m):
        name = f"cdf-{m:02d}.npz"
        pin, out = pp[name], dest / name
        if _ok(out, pin):
            return out, "have"
        tmp = dest / f".{name}.part{os.getpid()}"
        if source is not None:
            shutil.copyfile(Path(source) / name, tmp)
        else:
            _download(_url(name), tmp)
        size, sha = tmp.stat().st_size, _sha256(tmp)
        if size != pin["bytes"] or sha != pin["sha256"]:
            tmp.unlink(missing_ok=True)
            raise OSError(f"{name}: {size} bytes, sha256 {sha[:16]}.. is not the table this release pins "
                          f"({pin['bytes']} bytes, sha256 {pin['sha256'][:16]}..); not installed")
        os.replace(tmp, out)
        return out, "installed"

    with ThreadPoolExecutor(min(6, len(months)) or 1) as ex:     # several downloads at once (link-bound, not CPU)
        for out, what in ex.map(one, months):
            got.append(out)
            if not quiet:
                print(f"{FETCH_CMD}: {what} {out}", flush=True)
    return got


def _months(s: str) -> list:
    if s in ("", "all"):
        return list(range(1, 13))
    out = sorted({int(x) for x in s.replace(" ", "").split(",") if x})
    bad = [m for m in out if not 1 <= m <= 12]
    if bad:
        raise SystemExit(f"{FETCH_CMD}: months must be 1-12 (got {bad})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=FETCH_CMD, description=__doc__.split("\n\n")[0])
    ap.add_argument("--months", default="all", help="comma list of months 1-12, or all (default)")
    ap.add_argument("--from", dest="source", type=Path, default=None,
                    help="a folder holding cdf-MM.npz to install instead of downloading (checked against the pins)")
    ap.add_argument("--dest", type=Path, default=None, help=f"install folder (default {home()})")
    ap.add_argument("--where", action="store_true", help="print the install folder and which months it holds")
    a = ap.parse_args(argv)
    if a.where:
        d = a.dest or home()
        have = [m for m in range(1, 13) if (d / f"cdf-{m:02d}.npz").is_file()]
        print(f"{d} months {have or 'none'}")
        return 0
    try:
        fetch(_months(a.months), a.source, a.dest)
    except OSError as e:
        print(f"{FETCH_CMD}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
