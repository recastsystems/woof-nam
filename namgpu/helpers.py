"""Helper programs some woof-nam doors run, as pinned release assets beside the GRIB2 writer (namgpu/exporter.py).

  rw_asos, rw_stage4   `woof-nam score`: ASOS reports and Stage IV precipitation (gpuwm's Rust crate rw-obs)
  rw_hex_ugwp_oro      the UGWP drag option: orographic statistics from 30-arc-second terrain (tools/ugwp-oro)

None of them is inside the gpuwm wheel, so before this a `pip install` could not run `woof-nam score` (it asked for a
GPUWM_BRIDGES folder holding rw_asos and rw_stage4) or the UGWP option on a grid with no cached statistics (it asked
for a `cargo build` in a checkout). woof-nam carries prebuilt Linux x86-64 copies attached to the woof-nam GitHub
release named in parm/helper_pins.json, built like rw_rrfs_upp (manylinux_2_28: glibc 2.28 and later). Their exact size
and SHA-256 are pinned there. `woof-nam fetch-bridges` downloads them with the writer; a door that needs one and does
not find it downloads it by itself, checks both pins and only then installs it. A file that fails a pin is deleted.

Where a helper is looked for, in order:
  1. $GPUWM_BRIDGES/<name>, when that folder holds one (a gpuwm bridge overlay, as on a run box);
  2. the staged pinned copy, ~/.woof-nam/bridges/<release>-<sha12>/<name> (or under $WOOF_NAM_HOME);
  3. the installed gpuwm's bridges folder, then ~/.gpuwm/bridges (a gpuwm that ships it).

$WOOF_NAM_BRIDGE_URL, $WOOF_NAM_NO_FETCH and `woof-nam fetch-bridges --from FILE` work as for the writer.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from namgpu import exporter

PINS_FILE = Path(__file__).resolve().parent / "parm" / "helper_pins.json"
USED_BY = {"rw_asos": "woof-nam score (ASOS)", "rw_stage4": "woof-nam score (Stage IV)",
           "rw_hex_ugwp_oro": "the UGWP drag option"}


def pins() -> dict:
    return json.loads(PINS_FILE.read_text())


def names() -> tuple:
    return tuple(pins()["helpers"])


def _pin(name: str, key: str | None = None) -> dict | None:
    return pins()["helpers"].get(name, {}).get(key or exporter.platform_key() or "")


def staged_path(name: str, key: str | None = None) -> Path | None:
    pin = _pin(name, key)
    if pin is None:
        return None
    return exporter._home() / "bridges" / f"{pins()['release']}-{pin['sha256'][:12]}" / name


def find(name: str) -> Path | None:
    """The helper to run, or None (the order is in the module docstring)."""
    env = os.environ.get("GPUWM_BRIDGES")
    if env and (Path(env) / name).is_file():
        return Path(env) / name
    p = staged_path(name)
    if p is not None and p.is_file() and os.access(p, os.X_OK):
        return p
    try:
        import gpuwm  # noqa: PLC0415
        cands = [Path(gpuwm.__file__).resolve().parent / "libexec" / "bridges" / name]
    except ImportError:
        cands = []
    for c in cands + [Path.home() / ".gpuwm" / "bridges" / name]:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def missing_message(name: str, why: str = "") -> str:
    tail = f" ({why})" if why else ""
    return (f"{name}, which {USED_BY.get(name, 'this command')} runs, is not installed{tail}; run "
            f"`{exporter.FETCH_CMD}` to download it, or put a build of it in the folder GPUWM_BRIDGES names")


def fetch(name: str, source: Path | None = None, force: bool = False, quiet: bool = False) -> Path:
    """Stage the pinned helper (from the release asset, or from `source` on disk) and return its path."""
    pin = _pin(name)
    if pin is None:
        raise exporter.ExporterMissing(f"no prebuilt {name} for {sys.platform}: build it and put it in the folder "
                                       f"GPUWM_BRIDGES names")
    dst = staged_path(name)
    if not force and exporter._pinned(dst, pin):
        dst.chmod(0o755)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if source is not None:
            what = str(source)
            if not Path(source).is_file():
                raise exporter.ExporterMissing(f"{source}: no such file")
            shutil.copyfile(source, tmp)
        else:
            base = os.environ.get("WOOF_NAM_BRIDGE_URL") or (f"https://github.com/{pins()['repo']}/releases/download/"
                                                              f"{pins()['release']}")
            if "://" not in base:
                base = Path(base).resolve().as_uri()
            what = base.rstrip("/") + "/" + pin["asset"]
            if not quiet:
                print(f"woof-nam: fetching {name} ({pin['bytes'] / 1e6:.1f} MB) from {what}", file=sys.stderr,
                      flush=True)
            exporter._download(what, tmp)
        size, sha = tmp.stat().st_size, exporter._sha256(tmp)
        if size != pin["bytes"] or sha != pin["sha256"]:
            raise exporter.ExporterMissing(
                f"{what} is not the {name} this release pins ({size} bytes, sha256 {sha[:16]}..; want "
                f"{pin['bytes']} bytes, sha256 {pin['sha256'][:16]}..), not installed")
        tmp.chmod(0o755)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def ensure(name: str) -> Path:
    """The helper, staging the pinned copy first if none is on this machine. Raises ExporterMissing (one line)."""
    p = find(name)
    if p is not None:
        return p
    if os.environ.get("WOOF_NAM_NO_FETCH") == "1":
        raise exporter.ExporterMissing(missing_message(name))
    try:
        return fetch(name)
    except exporter.ExporterMissing as e:
        raise exporter.ExporterMissing(missing_message(name, str(e))) from None


def by_asset(path: Path) -> str | None:
    """The helper a file on disk is (by its release asset name or its own name), or None."""
    for n, rows in pins()["helpers"].items():
        if path.name == n or any(path.name == r["asset"] for r in rows.values()):
            return n
    return None
