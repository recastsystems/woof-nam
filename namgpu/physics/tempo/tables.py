"""TEMPO v3.0.3/v3.2.5 lookup tables: both versions read hexcore's same binary64 table ABI.

Built once per machine with gpuwm-hex's ported tempo_init (no native library, no Fortran): the TEMPO
PARAMETERs captured from the pinned TEMPO source (hexcore metadata), tempo_table_init and tempo_table_compute
(Rust, all cores) and the four NOAA lookup files through hexcore's Rust lookup decoder. Cached as one .npy
(~2 GB float64) keyed by the flags; later runs load it straight to the device.
The v303 cache name is retained: hexcore v3.2.5 uses the same TABLE_LAYOUT and initialized tables.

TEMPO is NCAR/NOAA code under Apache-2.0 (NCAR/TEMPO 9adf9ef); hexcore (gpuwm-hex) is Apache-2.0.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

# The four NOAA lookup files come from gpuwm-hex's own rule (hexcore tempo_lookup_path_in: the pinned RRFS fork source it
# fetches and SHA-256 checks on first use). NAMGPU_TEMPO_LOOKUP names a folder that holds them instead. THE BREAKAGE THIS
# PREVENTS: the old default was ~/.gpuwm/tables/tempo-v3, which nothing installs, so on a plain two-wheel install
# (recast-woof-nam[modern] + gpuwm-hex) the first TEMPO run stopped at drv-physics-initialize with FileNotFoundError
# (release two-wheel check 2026-11-09; the release boxes had a prebuilt table cache, which hid it). Same bytes: the
# four files hexcore fetches have the SHA-256 the release runs' cache receipt records.
LOOKUP_DIR_DEFAULT = None
CACHE_DEFAULT = Path(os.environ.get("NAMGPU_TEMPO_CACHE", Path.home() / ".cache" / "woof-nam" / "tempo"))


class _Native:
    """The table store hexcore's FrameworkInitializationRRFS.initialize_tempo_tables writes into."""

    def __init__(self, abi):
        self.abi = abi
        self.t = {}
        self.lookup_sha256 = {}

    def get_table(self, name):
        return self.t.get(name)

    def put_table(self, name, value, lower=(), copy=True):
        self.t[name] = (np.array(value, copy=True) if copy else value, tuple(lower))


def lookup_resolver(lookup_dir: Path | None = None):
    """filename -> path of one TEMPO lookup file: lookup_dir, else NAMGPU_TEMPO_LOOKUP, else gpuwm-hex's pinned source."""
    d = lookup_dir if lookup_dir is not None else os.environ.get("NAMGPU_TEMPO_LOOKUP") or None
    if d is not None:
        d = Path(d)

        def named(filename):
            f = d / filename
            if not f.is_file():
                raise SystemExit(f"woof-nam: TEMPO lookup file {filename} is not in {d} (NAMGPU_TEMPO_LOOKUP); unset it "
                                 "to use the files gpuwm-hex fetches")
            return str(f)
        return named
    from hexcore.mpasphys.rrfs.framework_init_rrfs import tempo_lookup_path_in
    return lambda filename: str(tempo_lookup_path_in(None, filename))


def build_blob(*, hailaware: bool, melt: bool, lookup_dir: Path | None = LOOKUP_DIR_DEFAULT) -> np.ndarray:
    from hexcore.mpasphys.rrfs import framework_init_rrfs as fi
    from hexcore.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT
    meta = Path(fi.__file__).resolve().parents[2] / "data" / "mpasphys-framework-rrfs" / "metadata.json"
    abi = {"tempo_constants": json.loads(meta.read_text())["tempo_constants"]}
    native = _Native(abi)
    init = object.__new__(fi.FrameworkInitializationRRFS)
    init.native = native
    init.configs = {"config_tempo_hailaware": bool(hailaware), "config_tempo_refl10cm_from_melting": bool(melt)}
    init.tempo_lookup_path = lookup_resolver(lookup_dir)
    init.initialize_tempo_tables()
    total = sum(math.prod(s) for _, s in TABLE_LAYOUT)
    blob = np.zeros(total, dtype=np.float64)
    off = 0
    for name, shape in TABLE_LAYOUT:
        n = math.prod(shape)
        item = native.get_table("module_mp_tempo_params." + name)
        if item is None:
            if melt or name not in ("sbins_radar", "dsbins_radar", "gbins_radar", "dgbins_radar"):
                raise ValueError("TEMPO table not initialized: " + name)
        else:
            v = np.asarray(item[0])
            if v.size != n:
                raise ValueError(f"TEMPO table {name}: {v.size} words, layout wants {n}")
            blob[off:off + n] = v.reshape(-1, order="F").astype(np.float64)
        off += n
    return blob, native.lookup_sha256


def load_blob(*, hailaware: bool = True, melt: bool = False, cache: Path = CACHE_DEFAULT,
              lookup_dir: Path | None = LOOKUP_DIR_DEFAULT):
    """(host float64 blob, receipt dict). Built on first use, then read from the cache."""
    from hexcore.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT
    layout = hashlib.sha256(json.dumps(TABLE_LAYOUT, separators=(",", ":")).encode()).hexdigest()[:16]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    stem = cache / f"tempo-v303-{layout}-hail{int(hailaware)}-melt{int(melt)}"
    npy, rec = stem.with_suffix(".npy"), stem.with_suffix(".json")
    if npy.is_file() and rec.is_file():
        return np.load(npy, mmap_mode="r"), json.loads(rec.read_text())
    blob, sha = build_blob(hailaware=hailaware, melt=melt, lookup_dir=lookup_dir)
    tmp = stem.with_suffix(".partial.npy")
    np.save(tmp, blob)
    os.replace(tmp, npy)
    receipt = {"layout_sha16": layout, "hailaware": hailaware, "melt": melt, "lookup_sha256": sha,
               "words": int(blob.size), "built_by": "hexcore initialize_tempo_tables (ported tempo_init)"}
    rec.write_text(json.dumps(receipt, indent=1))
    return np.load(npy, mmap_mode="r"), receipt


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    b, r = load_blob(hailaware="--no-hail" not in sys.argv)
    print(json.dumps(r), b.size, f"{time.time() - t0:.1f}s")
