"""The GRIB2 writer `woof-nam products` drives: rw_rrfs_upp, gpuwm's Rust field-set writer (crate rw-grib2export).

`woof-nam products` hands each history hour to `rw_rrfs_upp --profile-json`, which computes the derived fields and
writes the GRIB2 file. The build with the NAM awphys profile is not in a gpuwm release yet, so woof-nam carries its
own copy: a prebuilt Linux x86-64 binary attached to the woof-nam GitHub release named by ASSET_RELEASE, built in
the manylinux_2_28 image gpuwm builds its own Linux bridges in (it runs on any manylinux_2_28 Linux: RHEL 8, Debian 10, Ubuntu 20.04 and later). Its exact size and
SHA-256 are pinned below. `woof-nam fetch-bridges` (or the first `woof-nam products`) downloads it, checks both pins
and only then installs it under ~/.woof-nam/bridges/. A file that fails a pin is deleted, never installed.

Where the writer is looked for, in order:
  1. $WOOF_NAM_RW_RRFS_UPP: the path of a writer build (used as given, no pin check);
  2. $GPUWM_BRIDGES/rw_rrfs_upp, when that folder holds one (a gpuwm bridge overlay, as on a run box);
  3. the staged pinned copy, ~/.woof-nam/bridges/<release>-<sha12>/rw_rrfs_upp (or under $WOOF_NAM_HOME).

$WOOF_NAM_BRIDGE_URL replaces the download folder (an https:// or file:// URL, or a local folder) that holds the
asset, e.g. a mirror inside a closed network. `woof-nam fetch-bridges --from FILE` stages a copy already on disk;
it is checked against the same pins. $WOOF_NAM_NO_FETCH=1 stops `woof-nam products` from downloading by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

NAME = "rw_rrfs_upp"
REPO = "recastsystems/woof-nam"
# the woof-nam release whose assets carry this writer (a dev build fetches from the release it leads to)
ASSET_RELEASE = "v0.2.1"
# built from gpuwm f127ff79d4118a322cb144cc1a63e4e063d16d8d (crate rw-grib2export, bin rw_rrfs_upp) with Rust 1.94.0
# in quay.io/pypa/manylinux_2_28_x86_64@sha256:c2261579b9c2e5d45aa93312f73e2a302182e3e977b558581a1838d6fed3d8e6
PINS = {
    "linux-x86_64": {"asset": "rw_rrfs_upp-linux-x86_64", "bytes": 16850680,
                     "sha256": "09c2dff1d8461b21577e83272bb31d07dd71d149d39a453e6f40c9d32742274f"},
}
FETCH_CMD = "woof-nam fetch-bridges"


class ExporterMissing(RuntimeError):
    """No writer on this machine, and none could be staged. The message is one line for the user."""


def platform_key() -> str | None:
    m = platform.machine().lower()
    if sys.platform.startswith("linux") and m in ("x86_64", "amd64"):
        return "linux-x86_64"
    return None


def _home() -> Path:
    env = os.environ.get("WOOF_NAM_HOME")
    return Path(env) if env else Path.home() / ".woof-nam"


def staged_path(key: str | None = None) -> Path | None:
    pin = PINS.get(key or platform_key() or "")
    if pin is None:
        return None
    return _home() / "bridges" / f"{ASSET_RELEASE}-{pin['sha256'][:12]}" / NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned(path: Path, pin: dict) -> bool:
    return path.is_file() and path.stat().st_size == pin["bytes"] and _sha256(path) == pin["sha256"]


def find() -> Path | None:
    """The writer to run, or None (the order is in the module docstring)."""
    env = os.environ.get("WOOF_NAM_RW_RRFS_UPP")
    if env:
        return Path(env)
    bridges = os.environ.get("GPUWM_BRIDGES")
    if bridges and (Path(bridges) / NAME).is_file():
        return Path(bridges) / NAME
    p = staged_path()
    if p is not None and p.is_file() and os.access(p, os.X_OK):
        return p
    return None


def _source_url(pin: dict) -> str:
    base = os.environ.get("WOOF_NAM_BRIDGE_URL") or f"https://github.com/{REPO}/releases/download/{ASSET_RELEASE}"
    if "://" not in base:
        base = Path(base).resolve().as_uri()
    return base.rstrip("/") + "/" + pin["asset"]


def _download(url: str, dst: Path, timeout: float = 60.0, tries: int = 3) -> None:
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dst, "wb") as f:  # noqa: S310
                shutil.copyfileobj(r, f, 1 << 20)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            if getattr(e, "code", None) == 404 or url.startswith("file:"):
                break
            time.sleep(2 * (attempt + 1))
    code = getattr(last, "code", None)
    why = f"HTTP {code}" if code else str(getattr(last, "reason", None) or last)
    raise ExporterMissing(f"could not download {url}: {why}")


def fetch(source: Path | None = None, dest: Path | None = None, force: bool = False, quiet: bool = False) -> Path:
    """Stage the pinned writer (from the release asset, or from `source` on disk) and return its path."""
    key = platform_key()
    if key is None:
        raise ExporterMissing(f"no prebuilt {NAME} for {sys.platform} {platform.machine()}: build it from gpuwm "
                              f"(crate rw-grib2export, bin {NAME}) and set WOOF_NAM_RW_RRFS_UPP to it")
    pin = PINS[key]
    dst = Path(dest) / NAME if dest else staged_path(key)
    if not force and _pinned(dst, pin):
        dst.chmod(0o755)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{NAME}.", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if source is not None:
            what = str(source)
            if not Path(source).is_file():
                raise ExporterMissing(f"{source}: no such file")
            shutil.copyfile(source, tmp)
        else:
            what = _source_url(pin)
            if not quiet:
                print(f"woof-nam: fetching {NAME} ({pin['bytes'] / 1e6:.1f} MB) from {what}", file=sys.stderr,
                      flush=True)
            _download(what, tmp)
        size, sha = tmp.stat().st_size, _sha256(tmp)
        if size != pin["bytes"] or sha != pin["sha256"]:
            raise ExporterMissing(f"{what} is not the {NAME} this release pins ({size} bytes, sha256 {sha[:16]}..; "
                                  f"want {pin['bytes']} bytes, sha256 {pin['sha256'][:16]}..), not installed")
        tmp.chmod(0o755)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def missing_message(why: str = "") -> str:
    tail = f" ({why})" if why else ""
    return (f"the GRIB2 writer {NAME} is not installed{tail}; run `{FETCH_CMD}` to download it, "
            f"or set WOOF_NAM_RW_RRFS_UPP to a build of it")


def ensure(fetch_if_missing: bool = True) -> Path:
    """The writer, staging the pinned copy first if none is on this machine. Raises ExporterMissing (one line)."""
    p = find()
    if p is not None:
        if not p.is_file():
            raise ExporterMissing(f"WOOF_NAM_RW_RRFS_UPP={p} is not a file")
        return p
    if not fetch_if_missing or os.environ.get("WOOF_NAM_NO_FETCH") == "1":
        raise ExporterMissing(missing_message())
    try:
        return fetch()
    except ExporterMissing as e:
        raise ExporterMissing(missing_message(str(e))) from None


def capabilities(path: Path | None = None) -> set:
    """The profile features a writer build reads (`rw_rrfs_upp --capabilities`); empty for a build older than them.
    An older writer reads a nest profile without error and ignores what it does not know, so the nest files would
    carry U and V as separate messages and write points off the nest as data: woof-nam checks first."""
    import subprocess
    p = path or find()
    if p is None:
        return set()
    try:
        r = subprocess.run([str(p), "--capabilities"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if r.returncode != 0:
        return set()
    return set(r.stdout.split())


def require(features, what: str) -> Path:
    """The writer, refused in one line when it lacks a profile feature `what` needs."""
    p = ensure()
    have = capabilities(p)
    miss = [f for f in features if f not in have]
    if miss:
        raise ExporterMissing(
            f"{p} cannot write {what}: it lacks the profile feature(s) {', '.join(miss)}; "
            f"point WOOF_NAM_RW_RRFS_UPP at a qualified writer whose "
            f"rw_rrfs_upp --capabilities lists {', '.join(features)}")
    return p


def main(argv=None) -> int:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description=f"Download the GRIB2 writer `woof-nam products` uses ({NAME}, Linux x86-64) from the woof-nam "
                    f"{ASSET_RELEASE} release, and install it after checking its size and SHA-256 against the pins "
                    f"this release carries. The same for the helper programs of `woof-nam score` (rw_asos, "
                    f"rw_stage4) and of the UGWP drag option (rw_hex_ugwp_oro).",
        formatter_class=HelpFormatter,
        epilog=f"examples:\n  {FETCH_CMD}\n  {FETCH_CMD} --from ./{NAME}           # a copy already on disk\n"
               f"  WOOF_NAM_BRIDGE_URL=https://mirror.example/woof-nam {FETCH_CMD}\n  {FETCH_CMD} --where")
    ap.add_argument("--from", dest="source", type=Path, default=None,
                    help="stage this file instead of downloading (it must match the pins)")
    ap.add_argument("--dest", type=Path, default=None,
                    help="install into this folder (default ~/.woof-nam/bridges/<release>-<sha12>/)")
    ap.add_argument("--force", action="store_true", help="download again even if a pinned copy is installed")
    ap.add_argument("--where", action="store_true", help="print the writer `woof-nam products` would run, then exit")
    a = ap.parse_args(argv)
    from namgpu import helpers
    if a.where:
        p = find()
        print(p if p is not None else missing_message())
        for n in helpers.names():
            h = helpers.find(n)
            print(h if h is not None else helpers.missing_message(n))
        return 0 if p is not None else 1
    if a.source is not None and helpers.by_asset(a.source):      # a helper program already on disk
        n = helpers.by_asset(a.source)
        try:
            h = helpers.fetch(n, a.source, a.force)
        except ExporterMissing as e:
            print(f"woof-nam fetch-bridges: {e}", file=sys.stderr)
            return 2
        print(f"{n}: {h} (pinned)")
        return 0
    try:
        p = fetch(a.source, a.dest, a.force)
    except ExporterMissing as e:
        print(f"woof-nam fetch-bridges: {e}", file=sys.stderr)
        return 2
    print(f"{NAME}: {p} (sha256 {PINS[platform_key()]['sha256'][:16]}.., pinned)")
    if a.dest and find() != p:
        print(f"note: woof-nam products looks in {staged_path()}; set WOOF_NAM_RW_RRFS_UPP={p} to use this copy")
    rc = 0
    if a.source is None and a.dest is None:      # the helper programs beside it (namgpu/helpers.py)
        for n in helpers.names():
            try:
                print(f"{n}: {helpers.fetch(n, None, a.force)} (pinned)")
            except ExporterMissing as e:
                print(f"woof-nam fetch-bridges: {e}", file=sys.stderr)
                rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
